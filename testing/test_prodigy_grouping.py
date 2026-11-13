import copy
import io
import unittest
from unittest import mock

import torch
from prodigyopt import Prodigy

from toolkit.optimizer import get_optimizer
from toolkit.optimizers.prodigy_8bit import Prodigy8bit
from toolkit.optimizers.prodigy_grouped import GroupedProdigy, GroupedProdigy8bit
from toolkit.optimizers.optimizer_utils import Auto8bitTensor


def parameters():
    return [torch.nn.Parameter(torch.tensor([1., -2., 3.])),
            torch.nn.Parameter(torch.tensor([[0.5, -0.2], [1.2, 2.3]]))]


def gradients(params, step):
    for index, parameter in enumerate(params):
        parameter.grad = torch.full_like(parameter, (index + 1) * (1 if step % 3 else -0.4))


class ProdigyGroupingTest(unittest.TestCase):
    def assert_state_equal(self, left, right):
        if isinstance(left, Auto8bitTensor):
            self.assertIsInstance(right, Auto8bitTensor)
            self.assert_state_equal(left.state_dict(), right.state_dict())
        elif isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_default_factory_keeps_stock_backend_and_updates(self):
        for kind, backend in (('prodigy', Prodigy), ('prodigy8bit', Prodigy8bit)):
            with self.subTest(kind=kind):
                params = parameters()
                actual = get_optimizer([{'params': [p], 'lr': 1.0} for p in params], kind, 1.0)
                self.assertIs(type(actual), backend)
                self.assertNotIn('prodigy_grouping', actual.state_dict())
        params = parameters()
        reference_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
        actual = get_optimizer([{'params': [p], 'lr': 1.0} for p in params], 'prodigy', 1.0)
        reference = Prodigy([{'params': [p], 'lr': 1.0} for p in reference_params], lr=1.0, eps=1e-6)
        for step in range(6):
            gradients(params, step)
            gradients(reference_params, step)
            actual.step()
            reference.step()
            for p, q in zip(params, reference_params):
                self.assert_state_equal(p, q)
            self.assert_state_equal(actual.state_dict(), reference.state_dict())

    def test_factory_uses_explicit_block_group_marker(self):
        from testing.test_optimizer_param_groups import make_network
        from toolkit.optimizer_param_groups import group_network_params_by_transformer_block
        for kind, backend in (('prodigy', GroupedProdigy), ('prodigy8bit', GroupedProdigy8bit)):
            with self.subTest(kind=kind):
                model, network = make_network(['layers'])
                original = network.prepare_optimizer_params(None, 1.0, None)
                groups = group_network_params_by_transformer_block(original, model, network.unet_loras, ['layers'])
                actual = get_optimizer(groups, kind, 1.0)
                self.assertIs(type(actual), backend)
                self.assertEqual(len(actual.param_groups), 3)

    def test_trainer_rejects_independent_state_in_shared_mode(self):
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'jobs/process/BaseSDTrainProcess.py').read_text())
        guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and "'prodigy_grouping' in optimizer_state_dict" in ast.unparse(node.test))
        code = compile(ast.Module(body=[guard], type_ignores=[]), '<resume-guard>', 'exec')
        for independent in (False, True):
            context = {'optimizer': SimpleNamespace(independent_groups=independent),
                       'optimizer_state_dict': {'prodigy_grouping': {'version': 1, 'backend': 'Prodigy'}}}
            if independent:
                exec(code, context)
            else:
                with self.assertRaisesRegex(ValueError, 'matching transformer_block'):
                    exec(code, context)
        exec(code, {'optimizer': SimpleNamespace(), 'optimizer_state_dict': {}})

    def test_one_group_matches_backend(self):
        options = (
            {},
            {'betas': (0.0, 0.95), 'beta3': 0.8, 'weight_decay': 0.03, 'decouple': False},
            {'use_bias_correction': True, 'safeguard_warmup': True,
             'growth_rate': 1.03, 'weight_decay': 0.01, 'slice_p': 2},
        )
        for option in options:
            with self.subTest(options=option):
                params = parameters()
                refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
                actual = GroupedProdigy(params, lr=0.7, **option)
                reference = Prodigy(refs, lr=0.7, **option)
                for step in range(8):
                    gradients(params, step)
                    gradients(refs, step)
                    actual.step()
                    reference.step()
                    for p, q in zip(params, refs):
                        self.assert_state_equal(p, q)
                        self.assert_state_equal(actual.state[p], reference.state[q])
                    self.assert_state_equal(actual.param_groups[0], {**reference.param_groups[0], 'params': params})

    def test_bfloat16_one_group_matches_backend(self):
        params = [torch.nn.Parameter(p.detach().to(torch.bfloat16)) for p in parameters()]
        refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
        actual = GroupedProdigy(params)
        reference = Prodigy(refs)
        for step in range(8):
            gradients(params, step)
            gradients(refs, step)
            actual.step()
            reference.step()
            for p, q in zip(params, refs):
                self.assert_state_equal(p, q)
                self.assert_state_equal(actual.state[p], reference.state[q])

    def test_scalar_oracle_produces_different_group_estimates(self):
        a, b = [torch.nn.Parameter(torch.tensor([1.])) for _ in range(2)]
        optimizer = GroupedProdigy([{'params': [a]}, {'params': [b]}],
                                  lr=1.0, d0=0.1, betas=(0., 0.), beta3=0., eps=1e-8)
        for step in range(3):
            a.grad = torch.ones_like(a)
            b.grad = torch.full_like(b, 1 if step % 2 == 0 else -1)
            optimizer.step()
        # With no momentum and unit gradients, first two displacements are
        # ~0.1. The persistent group's third distance estimate is ~0.2;
        # alternating displacement cancels, leaving the other estimate at d0.
        self.assertAlmostEqual(optimizer.param_groups[0]['d'], 0.2, places=6)
        self.assertEqual(optimizer.param_groups[1]['d'], 0.1)

    def test_groups_match_separate_optimizers_with_distinct_options(self):
        params = parameters()
        refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
        options = [
            {'lr': 1.0, 'd0': 1e-5, 'betas': (0.8, 0.95), 'weight_decay': 0.03},
            {'lr': 0.3, 'd0': 3e-4, 'd_coef': 0.7, 'use_bias_correction': True,
             'safeguard_warmup': True, 'growth_rate': 1.1},
        ]
        optimizer = GroupedProdigy([{'params': [p], **option} for p, option in zip(params, options)])
        references = [Prodigy([p], **option) for p, option in zip(refs, options)]
        self.assertEqual([g['d'] for g in optimizer.param_groups], [1e-5, 3e-4])
        for step in range(8):
            gradients(params, step)
            gradients(refs, step)
            if step < 2:
                params[1].grad = refs[1].grad = None
            optimizer.step()
            for reference in references:
                reference.step()
            for i, (p, q) in enumerate(zip(params, refs)):
                self.assert_state_equal(p, q)
                self.assert_state_equal(optimizer.state.get(p, {}), references[i].state.get(q, {}))
                self.assert_state_equal(optimizer.param_groups[i], {**references[i].param_groups[0], 'params': [p]})
        rates = optimizer.get_learning_rates()
        self.assertEqual(rates, [g['d'] * g['lr'] for g in optimizer.param_groups])
        self.assertEqual(optimizer.get_avg_learning_rate(), sum(rates) / 2)

    def test_frozen_groups_closure_and_hooks(self):
        a, b = parameters()
        optimizer = GroupedProdigy([{'params': [a], 'lr': 1.0}, {'params': [b], 'lr': 0.0}])
        before_b = b.detach().clone()
        group_list = optimizer.param_groups
        calls = []
        optimizer.register_step_pre_hook(lambda opt, args, kwargs: calls.append(('pre', len(opt.param_groups))))
        optimizer.register_step_post_hook(lambda opt, args, kwargs: calls.append(('post', len(opt.param_groups))))

        def closure():
            calls.append(('closure', torch.is_grad_enabled()))
            optimizer.zero_grad()
            loss = a.square().sum() + b.square().sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(calls, [('pre', 2), ('closure', True), ('post', 2)])
        self.assertIs(optimizer.param_groups, group_list)
        self.assert_state_equal(b, before_b)
        self.assertEqual(optimizer.param_groups[1]['k'], 0)
        self.assertNotIn(b, optimizer.state)

    def test_group_list_restored_after_backend_error(self):
        params = parameters()
        optimizer = GroupedProdigy([{'params': [p]} for p in params])
        groups = optimizer.param_groups
        gradients(params, 1)
        raw_step = Prodigy.step
        while getattr(raw_step, 'hooked', False):
            raw_step = raw_step.__wrapped__
        calls = []

        def fail_second(opt, closure=None):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('test failure')
            return raw_step(opt)

        with mock.patch.object(Prodigy, 'step', fail_second):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                optimizer.step()
        self.assertIs(optimizer.param_groups, groups)

    def test_resume_roundtrip_and_reject_shared_or_other_backend_state(self):
        params = parameters()
        optimizer = GroupedProdigy([{'params': [p]} for p in params])
        for step in range(4):
            gradients(params, step)
            optimizer.step()
        buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
        restored = GroupedProdigy([{'params': [p]} for p in refs])
        restored.load_state_dict(saved)
        for step in range(4, 8):
            gradients(params, step)
            gradients(refs, step)
            optimizer.step()
            restored.step()
            for p, q in zip(params, refs):
                self.assert_state_equal(p, q)
            self.assert_state_equal(optimizer.state_dict(), restored.state_dict())
        original = copy.deepcopy(optimizer.state_dict())
        for marker in (None, {'version': 1, 'backend': 'Prodigy8bit'}, {'version': 2, 'backend': 'Prodigy'}):
            invalid = copy.deepcopy(saved)
            invalid['prodigy_grouping'] = marker
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, 'matching independent-mode'):
                optimizer.load_state_dict(invalid)
            self.assert_state_equal(optimizer.state_dict(), original)

    def test_eight_bit_matches_separate_backends_and_safe_resume(self):
        # Existing copy_stochastic rejects CPU tensors. Replace only that device
        # copy for this CPU test; actual int8 quantization/state logic still runs.
        def cpu_copy(target, source):
            target.copy_(source)

        with mock.patch('toolkit.optimizers.prodigy_8bit.copy_stochastic', cpu_copy):
            params = parameters()
            refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
            options = [{'lr': 1., 'd0': 1e-5}, {'lr': .4, 'd0': 3e-4, 'safeguard_warmup': True}]
            optimizer = GroupedProdigy8bit([{'params': [p], **o} for p, o in zip(params, options)])
            references = [Prodigy8bit([p], **o) for p, o in zip(refs, options)]
            for step in range(6):
                gradients(params, step)
                gradients(refs, step)
                optimizer.step()
                for reference in references:
                    reference.step()
                for i, (p, q) in enumerate(zip(params, refs)):
                    self.assert_state_equal(p, q)
                    self.assert_state_equal(optimizer.state[p], references[i].state[q])
                    self.assertEqual(optimizer.param_groups[i]['d'], references[i].param_groups[0]['d'])
            buffer = io.BytesIO()
            torch.save(optimizer.state_dict(), buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
            resumed = GroupedProdigy8bit([{'params': [p], **o} for p, o in zip(resumed_params, options)])
            resumed.load_state_dict(saved)
            gradients(params, 7)
            gradients(resumed_params, 7)
            optimizer.step()
            resumed.step()
            for p, q in zip(params, resumed_params):
                self.assert_state_equal(p, q)
                self.assert_state_equal(optimizer.state[p], resumed.state[q])
            original = copy.deepcopy(resumed.state_dict())
            corrupt = copy.deepcopy(saved)
            next(iter(corrupt['state'].values()))['s']['scale'] = float('nan')
            with self.assertRaisesRegex(ValueError, 'Invalid independent'):
                resumed.load_state_dict(corrupt)
            self.assert_state_equal(resumed.state_dict(), original)
            for missing in ('prodigy8bit_tensor', 's'):
                corrupt = copy.deepcopy(saved)
                values = next(iter(corrupt['state'].values()))
                if missing == 's':
                    del values['s']
                else:
                    del values['s'][missing]
                with self.subTest(missing=missing), self.assertRaises(ValueError):
                    resumed.load_state_dict(corrupt)
                self.assert_state_equal(resumed.state_dict(), original)

    def test_eight_bit_fsdp_rejected_before_updating_any_group(self):
        params = parameters()
        optimizer = GroupedProdigy8bit([{'params': [params[0]]}, {'params': [params[1]], 'fsdp_in_use': True}])
        gradients(params, 1)
        before = [p.detach().clone() for p in params]
        with self.assertRaisesRegex(ValueError, 'FSDP'):
            optimizer.step()
        for p, q in zip(params, before):
            self.assert_state_equal(p, q)
        self.assertFalse(optimizer.state)

    def test_group_d0_validation_and_late_added_group(self):
        optimizer = GroupedProdigy(parameters())
        count = len(optimizer.param_groups)
        for d0 in (0, -1, float('nan'), float('inf')):
            with self.subTest(d0=d0), self.assertRaisesRegex(ValueError, 'd0'):
                optimizer.add_param_group({'params': parameters(), 'd0': d0})
            self.assertEqual(len(optimizer.param_groups), count)
        optimizer.add_param_group({'params': parameters(), 'd0': 0.03, 'lr': 0.4})
        self.assertEqual(optimizer.param_groups[-1]['d'], 0.03)
        self.assertEqual(optimizer.param_groups[-1]['k'], 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for native stochastic copies')
    def test_native_eight_bit_cuda_steps(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                params = [torch.nn.Parameter(torch.ones(8, device='cuda', dtype=dtype)) for _ in range(2)]
                optimizer = GroupedProdigy8bit([{'params': [p]} for p in params])
                for _ in range(3):
                    optimizer.zero_grad()
                    sum(p.float().square().sum() for p in params).backward()
                    optimizer.step()
                self.assertTrue(all(p in optimizer.state for p in params))
                self.assertTrue(all(g['k'] == 3 for g in optimizer.param_groups))


if __name__ == '__main__':
    unittest.main()
