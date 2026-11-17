import copy
import math
import unittest
from unittest.mock import patch

import torch

from toolkit.optimizer import get_optimizer
from toolkit.optimizers.muon import Muon, _orthogonalize


class MuonTest(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(846)

    def matrix(self, shape=(3, 5), dtype=torch.float32):
        return torch.nn.Parameter(
            torch.randn(shape, generator=self.generator).to(dtype)
        )

    def test_factory_and_defaults(self):
        optimizer = get_optimizer([self.matrix()], "muon", 2e-4)
        self.assertIsInstance(optimizer, Muon)
        group = optimizer.param_groups[0]
        self.assertEqual(group["lr"], 2e-4)
        self.assertEqual(group["weight_decay"], 0.01)
        self.assertEqual(group["momentum"], 0.95)
        self.assertTrue(group["nesterov"])
        self.assertEqual(group["ns_steps"], 5)
        self.assertEqual(group["eps"], 1e-7)
        self.assertEqual(group["adjust_lr_fn"], "match_rms_adamw")

    def test_analytic_normalization_rms_scaling_and_decay(self):
        # With polynomial (1,0,0), NS is simply normalization. This oracle
        # explicitly calculates the documented RMS scale and decoupled decay.
        parameter = torch.nn.Parameter(
            torch.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
        )
        before = parameter.detach().clone()
        gradient = torch.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
        parameter.grad = gradient
        optimizer = Muon(
            [parameter],
            lr=0.1,
            weight_decay=0.2,
            momentum=0,
            nesterov=False,
            ns_coefficients=(1.0, 0.0, 0.0),
            ns_steps=1,
        )
        optimizer.step()
        normalized = gradient.to(torch.bfloat16) / 5
        expected = before * 0.98 - 0.1 * 0.2 * math.sqrt(3) * normalized.float()
        torch.testing.assert_close(parameter, expected, rtol=0, atol=1e-7)

    @unittest.skipUnless(
        hasattr(torch.optim, "Muon"), "native PyTorch Muon unavailable"
    )
    def test_matches_native_muon_across_shapes_precision_and_steps(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for nesterov in (False, True):
                if dtype == torch.bfloat16 and not nesterov:
                    # Native 2.11 normalizes the bf16 momentum buffer in place
                    # in this case. BatchedMuon must preserve its EMA instead.
                    continue
                with self.subTest(dtype=dtype, nesterov=nesterov):
                    parameters = [
                        self.matrix(shape, dtype)
                        for shape in ((3, 7), (7, 3), (4, 4), (3, 7))
                    ]
                    reference = [
                        torch.nn.Parameter(p.detach().clone()) for p in parameters
                    ]
                    options = dict(
                        lr=1e-3,
                        weight_decay=0.01,
                        momentum=0.95,
                        nesterov=nesterov,
                        adjust_lr_fn="match_rms_adamw",
                    )
                    optimizer = Muon(parameters, **options)
                    native = torch.optim.Muon(reference, **options)
                    for _ in range(4):
                        for p, r in zip(parameters, reference):
                            gradient = torch.randn(
                                p.shape, generator=self.generator
                            ).to(dtype)
                            p.grad, r.grad = gradient.clone(), gradient.clone()
                        optimizer.step()
                        native.step()
                        for p, r in zip(parameters, reference):
                            torch.testing.assert_close(
                                p,
                                r,
                                rtol=0,
                                atol=2e-6 if dtype == torch.float32 else 0.002,
                            )
                            torch.testing.assert_close(
                                optimizer.state[p]["momentum_buffer"],
                                native.state[r]["momentum_buffer"],
                            )

    def test_bfloat16_momentum_is_not_modified_by_orthogonalization(self):
        parameter = self.matrix(dtype=torch.bfloat16)
        optimizer = Muon([parameter], nesterov=False, momentum=0.5)
        expected = torch.zeros_like(parameter)
        for _ in range(3):
            gradient = torch.randn(parameter.shape, generator=self.generator).bfloat16()
            parameter.grad = gradient
            expected = (expected.float() * 0.5 + gradient.float() * 0.5).bfloat16()
            optimizer.step()
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"], expected, rtol=0, atol=0
            )

    def test_group_metadata_order_and_distinct_rates(self):
        first, second = self.matrix(), self.matrix()
        second.data.copy_(first)
        optimizer = Muon(
            [
                {"params": [first], "lr": 0.01, "block_name": "blocks.0"},
                {"params": [second], "lr": 0.02, "block_name": "blocks.1"},
            ],
            weight_decay=0,
        )
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)
        initial = first.detach().clone()
        optimizer.step()
        torch.testing.assert_close(
            initial - second, 2 * (initial - first), atol=2e-7, rtol=1e-5
        )
        self.assertEqual(
            [g["block_name"] for g in optimizer.param_groups], ["blocks.0", "blocks.1"]
        )
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_zero_and_missing_gradient(self):
        zero, missing = self.matrix(), self.matrix()
        initial_zero, initial_missing = zero.detach().clone(), missing.detach().clone()
        optimizer = Muon([zero, missing], lr=0.1, weight_decay=0.2)
        zero.grad = torch.zeros_like(zero)
        optimizer.step()
        torch.testing.assert_close(zero, initial_zero * 0.98)
        torch.testing.assert_close(missing, initial_missing, rtol=0, atol=0)
        self.assertNotIn(missing, optimizer.state)
        self.assertTrue(torch.isfinite(zero).all())

    def test_scalar_loss_weight_is_not_true_lr(self):
        # Scaling a fresh gradient by a power of two leaves its normalized
        # direction unchanged; .25 loss weight does not quarter Muon's step.
        first, second = self.matrix(), self.matrix()
        second.data.copy_(first)
        initial = first.detach().clone()
        gradient = torch.randn(first.shape, generator=self.generator)
        first.grad, second.grad = gradient.clone(), gradient * 0.25
        Muon([first], lr=0.01, weight_decay=0).step()
        Muon([second], lr=0.01, weight_decay=0).step()
        torch.testing.assert_close(first, second, rtol=0, atol=1e-7)
        self.assertGreater((initial - first).norm().item(), 0.001)

    def test_resume_continuity_and_invalid_state_is_atomic(self):
        parameter = self.matrix()
        optimizer = Muon([parameter])
        parameter.grad = torch.randn(parameter.shape, generator=self.generator)
        optimizer.step()
        checkpoint = copy.deepcopy(optimizer.state_dict())
        restored = torch.nn.Parameter(parameter.detach().clone())
        resumed = Muon([restored])
        resumed.load_state_dict(checkpoint)
        for _ in range(3):
            gradient = torch.randn(parameter.shape, generator=self.generator)
            parameter.grad, restored.grad = gradient.clone(), gradient.clone()
            optimizer.step()
            resumed.step()
        torch.testing.assert_close(restored, parameter, rtol=0, atol=0)
        buffer_before = resumed.state[restored]["momentum_buffer"].clone()
        invalid = copy.deepcopy(checkpoint)
        invalid["state"][0] = {"exp_avg": torch.ones_like(parameter)}
        with self.assertRaisesRegex(ValueError, "Muon"):
            resumed.load_state_dict(invalid)
        torch.testing.assert_close(
            resumed.state[restored]["momentum_buffer"], buffer_before, rtol=0, atol=0
        )
        adam = torch.optim.AdamW([parameter]).state_dict()
        with self.assertRaisesRegex(ValueError, "Muon"):
            resumed.load_state_dict(adam)
        invalid = copy.deepcopy(checkpoint)
        invalid["state"][0]["momentum_buffer"] = torch.ones(2, 2)
        with self.assertRaisesRegex(ValueError, "Muon"):
            resumed.load_state_dict(invalid)

    def test_chunk_limit_and_mixed_dtypes(self):
        parameters = [self.matrix((5, 7)) for _ in range(5)] + [
            self.matrix((5, 7), torch.bfloat16)
        ]
        optimizer = Muon(parameters, max_batch_bytes=1)
        for p in parameters:
            p.grad = torch.ones_like(p)
        sizes = []

        def record(matrices, *args):
            sizes.append(matrices.shape[0])
            return _orthogonalize(matrices, *args)

        with patch("toolkit.optimizers.muon._orthogonalize", side_effect=record):
            optimizer.step()
        self.assertEqual(sizes, [1] * len(parameters))

    def test_closure(self):
        parameter = self.matrix()
        optimizer = Muon([parameter])

        def closure():
            optimizer.zero_grad()
            loss = parameter.square().sum()
            loss.backward()
            return loss

        result = optimizer.step(closure)
        self.assertIsNotNone(result.grad_fn)

    def test_invalid_options_shapes_and_sparse_gradients(self):
        for options in (
            {"lr": float("nan")},
            {"weight_decay": -1},
            {"momentum": 1},
            {"momentum": -0.1},
            {"ns_steps": 0},
            {"ns_steps": 100},
            {"ns_steps": 2.5},
            {"eps": 0},
            {"ns_coefficients": (1, 2)},
            {"adjust_lr_fn": "unknown"},
            {"max_batch_bytes": 0},
            {"nesterov": "true"},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "Muon"),
            ):
                Muon([self.matrix()], **options)
        for shape in ((), (3,), (2, 2, 2), (0, 2)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "Muon"):
                Muon([torch.nn.Parameter(torch.empty(shape))])
        with self.assertRaisesRegex(ValueError, "Muon"):
            Muon([self.matrix(dtype=torch.float64)])
        duplicate = self.matrix()
        with self.assertRaisesRegex(ValueError, "Muon"):
            Muon([duplicate, duplicate])
        first, second = self.matrix(), self.matrix()
        initial = first.detach().clone()
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second).to_sparse()
        optimizer = Muon([first, second])
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            optimizer.step()
        torch.testing.assert_close(first, initial, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
