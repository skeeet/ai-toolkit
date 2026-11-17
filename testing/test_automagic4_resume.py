"""CPU regressions for the AutoMagic4 voting and checkpoint patch."""

import copy
import itertools
import math
import unittest

import torch

from toolkit.optimizers.automagic4 import Automagic4


class AutoMagic4Tests(unittest.TestCase):
    @staticmethod
    def optimizer(parameters, **kwargs):
        return Automagic4(
            parameters,
            lr=1e-4,
            beta2=0.0,
            eps=1e-8,
            polarity_history=4,
            fused=False,
            **kwargs,
        )

    def test_balanced_sign_windows_hold_lr(self):
        parameter = torch.nn.Parameter(torch.zeros(16))
        optimizer = self.optimizer([parameter])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()  # First step measures sensitivity without changing weights.
        torch.testing.assert_close(parameter, torch.zeros_like(parameter))
        signs = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=4))).T
        for row in signs:
            parameter.grad = row.clone()
            optimizer.step()
        # Every possible fair-sign history has equal mass: exact zero mean vote.
        self.assertAlmostEqual(optimizer.state[parameter]["lr"].item(), 1e-4, places=10)

    def test_trend_and_alternation_obey_vote_cap(self):
        for alternating in (False, True):
            with self.subTest(alternating=alternating):
                parameter = torch.nn.Parameter(torch.zeros(3, 5))
                optimizer = self.optimizer([parameter], max_vote=0.05)
                for step in range(5):
                    sign = -1.0 if alternating and step % 2 else 1.0
                    parameter.grad = torch.full_like(parameter, sign)
                    optimizer.step()
                direction = -1 if alternating else 1
                self.assertAlmostEqual(
                    optimizer.state[parameter]["lr"].item(),
                    1e-4 * math.exp(direction * 0.05),
                    places=10,
                )

    def test_dead_tensors_preserve_live_sensitivity_and_late_lr(self):
        live = torch.nn.Parameter(torch.zeros(2))
        dead = torch.nn.Parameter(torch.zeros(100))
        late = torch.nn.Parameter(torch.zeros(2))
        optimizer = self.optimizer([live, dead, late])
        for _ in range(6):
            live.grad = torch.ones_like(live)
            dead.grad = torch.zeros_like(dead)
            optimizer.step()
        torch.testing.assert_close(optimizer._group_ref[0], torch.tensor(1.0))
        adapted = optimizer.state[live]["lr"].clone()
        late.grad = torch.ones_like(late)
        optimizer.step()
        torch.testing.assert_close(optimizer.state[late]["lr"], adapted)

    def test_all_dead_start_stays_finite_until_first_live_step(self):
        parameter = torch.nn.Parameter(torch.zeros(2, 3))
        optimizer = self.optimizer([parameter])
        for _ in range(3):
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
        self.assertIsNone(optimizer._group_ref[0])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.zeros_like(parameter))
        self.assertTrue(torch.isfinite(optimizer._group_ref[0]))
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertTrue((parameter < 0).all())

    def test_resume_matches_next_update(self):
        parameter = torch.nn.Parameter(torch.zeros(3, 5))
        optimizer = self.optimizer([parameter])
        for _ in range(7):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        restored = torch.nn.Parameter(parameter.detach().clone())
        resumed = self.optimizer([restored])
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for current, active in ((parameter, optimizer), (restored, resumed)):
            current.grad = -torch.ones_like(current)
            active.step()
        # Parent resume unifies LR mirrors through exp(median(log(lr))), which
        # introduces float32 roundoff even when the mirrors already agree.
        torch.testing.assert_close(restored, parameter, rtol=1e-6, atol=0)
        torch.testing.assert_close(
            resumed.state[restored]["lr"],
            optimizer.state[parameter]["lr"],
            rtol=1e-6,
            atol=0,
        )

    def test_backward_hooks_and_low_precision_steps(self):
        for dtype, fused in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16), (False, True)
        ):
            with self.subTest(dtype=dtype, fused=fused):
                parameter = torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype))
                optimizer = Automagic4([parameter], lr=1e-4, eps=1e-6, fused=fused)
                for _ in range(3):
                    parameter.float().sum().backward()
                    optimizer.step()
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertTrue((parameter < 0).all())

    def test_group_vote_weights_each_history_by_its_balanced_update(self):
        trending = torch.nn.Parameter(torch.zeros(1))
        alternating = torch.nn.Parameter(torch.zeros(1))
        optimizer = Automagic4(
            [trending, alternating],
            lr=1e-4,
            beta2=0.0,
            polarity_history=4,
            sensitivity_quantile=0.5,
            max_vote=0.8,
            fused=False,
        )
        for sign in (1, 1, -1, 1, -1):
            trending.grad = torch.ones_like(trending)
            alternating.grad = torch.full_like(alternating, 2 * sign)
            optimizer.step()
        # Reference 1 leaves the first normalized update at magnitude 1.
        # It halves the second. A trend votes +5/4, alternation votes -3/4,
        # so the weighted group vote is (5/4 - 3/8) / (1 + 1/2) = 7/12.
        self.assertAlmostEqual(
            optimizer.get_learning_rates()[0], 1e-4 * math.exp(7 / 12), places=10
        )

    def test_fp16_checkpoint_preserves_unrepresentable_sensitivity_reference(self):
        parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float16))
        config = {"lr": 1e-3, "beta2": 0.0, "fused": False}
        optimizer = Automagic4([parameter], **config)
        parameter.grad = torch.tensor([1, 0, 0], dtype=parameter.dtype)
        optimizer.step()

        restored = torch.nn.Parameter(parameter.detach().clone())
        resumed = Automagic4([restored], **config)
        resumed.load_state_dict(optimizer.state_dict())
        # Two coordinates sit at the 1e-15 sensitivity floor. Their geometric
        # mean with unit sensitivity is 1e-10, below the smallest fp16 subnormal.
        expected = torch.tensor(1e-10)
        torch.testing.assert_close(
            optimizer.state[parameter]["sens_ref"], expected, rtol=2e-6, atol=0
        )
        torch.testing.assert_close(
            resumed.state[restored]["sens_ref"], expected, rtol=2e-6, atol=0
        )

        for current, active in ((parameter, optimizer), (restored, resumed)):
            current.grad = torch.tensor([1e-6, 0, 0], dtype=current.dtype)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                active.step()
        self.assertLess(parameter[0].item(), 0.0)
        torch.testing.assert_close(restored, parameter, rtol=0, atol=0)

    def test_fp16_checkpoint_preserves_unrepresentable_learning_rate(self):
        parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float16))
        config = {"lr": 1e-8, "fused": False}
        optimizer = Automagic4([parameter], **config)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        restored = torch.nn.Parameter(parameter.detach().clone())
        resumed = Automagic4([restored], **config)
        resumed.load_state_dict(optimizer.state_dict())
        self.assertAlmostEqual(resumed.get_learning_rates()[0], 1e-8, places=14)

    def test_empty_checkpoint_clears_the_previous_sensitivity_reference(self):
        parameter = torch.nn.Parameter(torch.zeros(3))
        optimizer = Automagic4([parameter], beta2=0.0, fused=False)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        empty = Automagic4(
            [torch.nn.Parameter(torch.zeros_like(parameter))], fused=False
        )
        optimizer.load_state_dict(empty.state_dict())
        parameter.grad = torch.full_like(parameter, 2.0)
        optimizer.step()

        # The first step after reset measures the new group without changing weights.
        torch.testing.assert_close(parameter, torch.zeros_like(parameter))
        torch.testing.assert_close(
            optimizer.state[parameter]["sens_ref"], torch.tensor(2.0)
        )
