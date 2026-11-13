"""Independent Prodigy adaptation for explicitly partitioned parameter groups."""

import math

import torch
from prodigyopt import Prodigy

from toolkit.optimizers.prodigy_8bit import Prodigy8bit
from toolkit.optimizers.optimizer_utils import Auto8bitTensor


class _IndependentProdigyGroups:
    independent_groups = True
    _backend = None

    def add_param_group(self, param_group):
        param_group = dict(param_group)
        d0 = param_group.get('d0', self.defaults['d0'])
        lr = param_group.get('lr', self.defaults['lr'])
        if not math.isfinite(d0) or d0 <= 0:
            raise ValueError('Each Prodigy group must have a finite positive d0')
        if not math.isfinite(lr) or lr < 0:
            raise ValueError('Each Prodigy group must have a finite nonnegative lr')
        param_group.update(d0=d0, d=d0, d_max=d0, d_numerator=0.0, k=0)
        super().add_param_group(param_group)

    def step(self, closure=None):
        if self._backend is Prodigy8bit and any(
            group.get('fsdp_in_use', False) or any(
                hasattr(p, '_fsdp_flattened') for p in group['params']
            ) for group in self.param_groups
        ):
            # The 8-bit backend can return before its collective on an empty
            # local group, leaving other ranks waiting. Fail before any update.
            raise ValueError('Independent Prodigy8bit groups do not support FSDP')
        # Run a closure once for the full model, before updating any group.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        backend_step = self._backend.step
        # Optimizer.step profiling wrappers also dispatch user pre/post hooks.
        # Only the outer step should dispatch those, not each group update.
        while getattr(backend_step, 'hooked', False):
            backend_step = backend_step.__wrapped__

        groups = self.param_groups
        try:
            with torch.no_grad():
                for group in groups:
                    if group['lr'] == 0:
                        continue
                    fsdp = group.get('fsdp_in_use', False) or any(
                        hasattr(p, '_fsdp_flattened') for p in group['params']
                    )
                    if not fsdp and not any(
                        p.grad is not None or hasattr(p, '_accum_grad') for p in group['params']
                    ):
                        continue
                    if fsdp:
                        group['fsdp_in_use'] = True
                    self.param_groups = [group]
                    backend_step(self)
        finally:
            self.param_groups = groups
        return loss

    def state_dict(self):
        state = super().state_dict()
        state['prodigy_grouping'] = {'version': 1, 'backend': self._backend.__name__}
        if self._backend is Prodigy8bit:
            # Keep weights_only=True loading compatible with the trainer; the
            # legacy 8-bit backend otherwise serializes custom Python objects.
            state['state'] = {
                key: {
                    name: {'prodigy8bit_tensor': 1, **value.state_dict()}
                    if isinstance(value, Auto8bitTensor) else value
                    for name, value in values.items()
                } for key, values in state['state'].items()
            }
        return state

    def load_state_dict(self, state_dict):
        expected = {'version': 1, 'backend': self._backend.__name__}
        if state_dict.get('prodigy_grouping') != expected:
            raise ValueError(
                'Independent Prodigy groups require a matching independent-mode checkpoint. '
                'Start with fresh optimizer state when changing grouping or Prodigy backend.'
            )
        for group in state_dict['param_groups']:
            if any(not math.isfinite(group[name]) or group[name] <= 0
                   for name in ('d0', 'd', 'd_max')):
                raise ValueError('Invalid independent Prodigy distance estimate')
            if (not math.isfinite(group['d_numerator'])
                    or not isinstance(group['k'], int) or group['k'] < 0):
                raise ValueError('Invalid independent Prodigy adaptation state')
        if self._backend is Prodigy8bit:
            saved_groups = state_dict['param_groups']
            if len(saved_groups) != len(self.param_groups) or any(
                len(saved['params']) != len(current['params'])
                for saved, current in zip(saved_groups, self.param_groups)
            ):
                raise ValueError('Independent Prodigy checkpoint parameter groups do not match')
            parameters = {
                key: parameter
                for saved, current in zip(saved_groups, self.param_groups)
                for key, parameter in zip(saved['params'], current['params'])
            }
            state_dict = dict(state_dict)
            restored = {}
            for key, values in state_dict['state'].items():
                if key not in parameters:
                    raise ValueError('Unexpected independent Prodigy8bit parameter state')
                if (not {'step', 's', 'p0', 'exp_avg', 'exp_avg_sq'}.issubset(values)
                        or not isinstance(values['step'], int) or values['step'] < 0):
                    raise ValueError('Incomplete independent Prodigy8bit parameter state')
                restored[key] = {}
                for name, value in values.items():
                    if name in ('s', 'p0', 'exp_avg', 'exp_avg_sq'):
                        parameter = parameters[key]
                        if (not isinstance(value, dict)
                                or set(value) != {'prodigy8bit_tensor', 'quantized', 'scale', 'orig_dtype'}
                                or value['prodigy8bit_tensor'] != 1
                                or not isinstance(value['quantized'], torch.Tensor)
                                or value['quantized'].dtype != torch.int8
                                or value['quantized'].shape != parameter.shape
                                or value['orig_dtype'] != torch.float32
                                or not math.isfinite(value['scale']) or value['scale'] <= 0):
                            raise ValueError('Invalid independent Prodigy8bit tensor state')
                        value = Auto8bitTensor(value)
                        value.quantized = value.quantized.to(parameter.device)
                    restored[key][name] = value
            state_dict['state'] = restored
        return super().load_state_dict(state_dict)

    def get_learning_rates(self):
        return [float(group['d'] * group['lr']) for group in self.param_groups]

    def get_avg_learning_rate(self):
        rates = self.get_learning_rates()
        return sum(rates) / len(rates) if rates else 0.0


class GroupedProdigy(_IndependentProdigyGroups, Prodigy):
    """Apply stock Prodigy independently to each optimizer parameter group."""

    _backend = Prodigy


class GroupedProdigy8bit(_IndependentProdigyGroups, Prodigy8bit):
    """Apply Toolkit's 8-bit Prodigy independently to each parameter group."""

    _backend = Prodigy8bit
