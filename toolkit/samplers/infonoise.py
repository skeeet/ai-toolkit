"""Optional information-guided timestep sampling.

Paper: Noise Scheduling as Information-Guided Allocation in Diffusion Training,
arXiv:2602.18647v2, section 3.2 and appendix A.6. Velocity errors are converted
to clean-data errors before estimating the information profile. The sampler
accounts for the velocity objective without changing the training loss.
"""
from collections import deque
from dataclasses import asdict, dataclass, fields
import math

import numpy as np
import torch


@dataclass(frozen=True)
class InfoNoiseConfig:
    bins: int = 32
    warmup_steps: int = 500
    refresh_every: int = 100
    buffer_size: int = 256
    min_count: int = 4
    min_bin_fraction: float = 0.5
    beta: float = 0.9  # Weight on the new bin mean.
    gate_pivot: float = 0.0  # Automatic onset in sigma-density; positive values fix the pivot.
    gate_power: float = 3.0
    onset_threshold: float = 0.002
    min_u: float = 0.001
    max_u: float = 0.999
    max_stale_steps: int = 2000
    exploration: bool = True  # Revisit bins before their error estimates expire.
    # Cover log-noise bins during warmup; sigmoid can leave the tails unobserved.
    warmup_type: str = 'log_uniform'  # log_uniform, sigmoid, uniform
    warmup_shift: float = 0.0
    warmup_scale: float = 1.0
    allow_legacy_resume: bool = False

    @classmethod
    def from_dict(cls, options=None, **defaults):
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError('train.infonoise must be a mapping')
        unknown = set(options) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError('Unknown InfoNoise settings: ' + ', '.join(sorted(unknown)))
        config = cls(**{**defaults, **options})
        for name in ('bins', 'warmup_steps', 'refresh_every', 'buffer_size', 'min_count', 'max_stale_steps'):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'warmup_steps' else 1):
                raise ValueError(f'InfoNoise {name} must be an integer >= {0 if name == "warmup_steps" else 1}')
        if config.bins < 2 or config.min_count > config.buffer_size:
            raise ValueError('InfoNoise needs bins >= 2 and min_count <= buffer_size')
        for name in ('min_bin_fraction', 'beta', 'gate_pivot', 'gate_power', 'onset_threshold',
                     'min_u', 'max_u', 'warmup_shift', 'warmup_scale'):
            if isinstance(getattr(config, name), bool) or not isinstance(getattr(config, name), (int, float)) or not math.isfinite(getattr(config, name)):
                raise ValueError(f'InfoNoise {name} must be finite')
        if not (0 < config.min_u < config.max_u < 1):
            raise ValueError('InfoNoise requires 0 < min_u < max_u < 1')
        # Float32 T/1000 must remain strictly inside (0,1).
        if np.float32(config.min_u) <= 0 or np.float32(config.max_u * 1000) / np.float32(1000) >= 1:
            raise ValueError('InfoNoise support is too close to a float32 endpoint')
        if not (0 < config.min_bin_fraction <= 1 and 0 < config.beta <= 1):
            raise ValueError('InfoNoise min_bin_fraction and beta must be in (0,1]')
        if config.gate_pivot < 0 or config.gate_power < 2 or not 0 < config.onset_threshold < 1:
            raise ValueError('Invalid InfoNoise gate settings')
        if config.warmup_type not in ('sigmoid', 'uniform', 'log_uniform') or config.warmup_scale <= 0:
            raise ValueError('Invalid InfoNoise warmup distribution')
        if not isinstance(config.exploration, bool):
            raise ValueError('InfoNoise exploration must be boolean')
        if not isinstance(config.allow_legacy_resume, bool):
            raise ValueError('InfoNoise allow_legacy_resume must be boolean')
        return config


# These modes update parameters during backward, unlike ordinary fused kernels.
_BACKWARD_UPDATE_DEFAULTS = {
    'automagic2': True,
    'automagic3': True,
    'automagic4': True,
    'automagicexperiment': True,
    'adamconvrot': False,
}


def _validate_optimizer_mode(name, fused):
    if name == 'automagic2':
        raise ValueError('InfoNoise does not support automagic2: it updates weights during backward')
    if name in _BACKWARD_UPDATE_DEFAULTS and fused:
        raise ValueError(f'InfoNoise requires optimizer_params.fused: false with {name}')


def validate_optimizer(optimizer):
    """Check the actual optimizer before any backward pass can change weights."""
    for cls in type(optimizer).__mro__:
        if cls.__module__.startswith('toolkit.optimizers.'):
            _validate_optimizer_mode(cls.__name__.lower(), getattr(optimizer, 'fused', False))


def validate_infonoise(train, model, datasets):
    """Check the noise, objective, and update schedule used by the estimator."""
    if train.timestep_type != 'infonoise':
        return
    InfoNoiseConfig.from_dict(train.infonoise)
    optimizer_name = train.optimizer.lower()
    optimizer_options = train.optimizer_params or {}
    _validate_optimizer_mode(optimizer_name, optimizer_options.get(
        'fused', _BACKWARD_UPDATE_DEFAULTS.get(optimizer_name, False)))
    if model.is_v_pred:
        raise ValueError('InfoNoise requires flow velocity MSE, not VP v-prediction')
    required = {
        'noise_scheduler': 'flowmatch', 'loss_type': 'mse', 'loss_target': 'noise',
        'content_or_style': 'balanced', 'content_or_style_reg': 'balanced',
        'num_train_timesteps': 1000, 'min_denoising_steps': 0, 'max_denoising_steps': 999,
        'gradient_accumulation_steps': 1,
        'noise_offset': 0.0, 'noise_multiplier': 1.0, 'target_noise_multiplier': 1.0,
        'random_noise_multiplier': 0.0, 'random_noise_shift': 0.0,
        'noisy_latent_multiplier': 1.0, 'pred_scaler': 1.0,
        'optimal_noise_pairing_samples': 1, 'first_timestep_chance': 0.0,
        'snr_gamma': None, 'min_snr_gamma': None, 'target_norm_std': None,
        'max_loss': None, 'diffusion_feature_extractor_path': None,
        'adapter_assist_name_or_path': None, 'match_adapter_chance': 0.0,
    }
    disabled = (
        'linear_timesteps', 'linear_timesteps2', 'train_turbo', 'learnable_snr_gos',
        'dynamic_noise_offset', 'do_signal_correction_noise', 'do_batch_noise_correction',
        'do_signal_amplification', 'force_consistent_noise', 'blended_blur_noise',
        'adaptive_scaling_factor', 'do_cfg', 'do_random_cfg', 'do_guidance_loss',
        'do_differential_guidance', 'correct_pred_norm', 'match_noise_norm',
        'inverted_mask_prior', 'diff_output_preservation', 'blank_prompt_preservation',
        'do_prior_divergence', 't0_loss_target', 'do_fft_loss', 'do_blank_stabilization',
        'single_item_batching', 'short_and_long_captions', 'merge_network_on_save',
    )
    bad = [key for key, value in required.items() if getattr(train, key) != value]
    bad += [key for key in disabled if getattr(train, key)]
    if bad:
        raise ValueError('InfoNoise unsupported train settings: ' + ', '.join(bad))
    for dataset in datasets:
        if dataset.prior_reg or dataset.unconditional_path:
            raise ValueError('InfoNoise does not support teacher/prior targets or paired guidance losses')


def validate_process(process):
    if process.accelerator.num_processes != 1:
        raise ValueError('InfoNoise currently supports single-process training only; distributed statistics are not implemented')
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    if not isinstance(process, SDTrainer):
        raise ValueError('InfoNoise requires the SDTrainer loss and optimizer-step hooks')
    if process.guidance_config is not None:
        raise ValueError('InfoNoise requires the MSE loss path, not a separate guidance objective')


def validate_model(model):
    """Check flow capabilities after model initialization, before loading weights."""
    from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
    if (not isinstance(model.noise_scheduler, CustomFlowMatchEulerDiscreteScheduler)
            or getattr(model.noise_scheduler, 'fixed_train_sigmas', None) is not None
            or not model.is_flow_matching or model.x0_pred
            or model.prediction_type == 'v_prediction'):
        raise ValueError('InfoNoise requires flow velocity training with a continuous flowmatch schedule; fixed sigmas and x0/VP targets are incompatible')


def validate_target(target, noise, clean):
    """Verify the velocity convention on the first real batch."""
    with torch.no_grad():
        velocity = noise - clean
        # Let existing nonfinite-loss handling run; check the next finite batch.
        if not torch.isfinite(target).all() or not torch.isfinite(velocity).all():
            return False
        candidate = target.to(velocity.dtype)
        if not (torch.allclose(candidate, velocity) or torch.allclose(candidate, -velocity)):
            raise ValueError('InfoNoise requires a flow velocity MSE target (noise - clean, or its reverse sign)')
        return True



class InfoNoiseSampler:
    STATE_VERSION = 4  # Clean-data statistics and sigma-density onset rule.

    def __init__(self, options=None, *, seed=None, context=None, **defaults):
        self.config = InfoNoiseConfig.from_dict(options, **defaults)
        c = self.config
        self.context = context or {}
        self.edges = np.linspace(math.log(c.min_u / (1 - c.min_u)),
                                 math.log(c.max_u / (1 - c.max_u)), c.bins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.relative_sigmas = np.exp(self.centers)
        self.fifo = [deque(maxlen=c.buffer_size) for _ in range(c.bins)]
        self.mse_ema = np.ones(c.bins, dtype=np.float64)  # Algorithm 1 initial estimate.
        self.ema_steps = np.full(c.bins, -1, dtype=np.int64)
        self.total_counts = np.zeros(c.bins, dtype=np.int64)
        self.pending = []
        self.optimizer_steps = 0
        self.last_refresh = -1
        self.profile_step = -1
        self.refresh_attempts = 0
        self.invalid_observations = 0
        self.cdf = None
        self.pivot = None
        self.reason = 'warmup'
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(torch.initial_seed() if seed is None else seed)
        # Validate truncated normal mass without drawing or touching global RNG.
        if c.warmup_type == 'sigmoid':
            lo, hi, _ = self._normal_bounds()
            if hi - lo <= np.finfo(np.float64).tiny:
                raise ValueError('InfoNoise sigmoid warmup has negligible probability on configured support')

    def _normal_bounds(self):
        c = self.config
        # u=1-sigmoid(N(shift, scale²)), so z=logit(u) has mean -shift.
        a, b = ((z + c.warmup_shift) / c.warmup_scale for z in (self.edges[0], self.edges[-1]))
        if a >= 0:
            # Use survival probabilities in the positive tail; CDF values round to 1.
            return 0.5 * math.erfc(b / math.sqrt(2)), 0.5 * math.erfc(a / math.sqrt(2)), True
        return 0.5 * math.erfc(-a / math.sqrt(2)), 0.5 * math.erfc(-b / math.sqrt(2)), False

    def _uniform(self, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError('InfoNoise sample count must be positive')
        eps = torch.finfo(torch.float64).eps
        return torch.rand(count, generator=self.generator, dtype=torch.float64).numpy() * (1 - 2 * eps) + eps

    def _coverage_bins(self, count, eligible=None):
        """Oldest overdue bins, limited to available training images."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('InfoNoise sample count must be positive')
        eligible = np.ones(count, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
        if eligible.shape != (count,):
            raise ValueError('InfoNoise eligibility must match samples')
        if not self.config.exploration:
            return [], eligible
        c = self.config
        # Aim for min_count+1 visits per freshness window when capacity allows.
        # Ordinary adaptive visits satisfy this quota too. Stagger empty bins so
        # coverage starts gradually rather than replacing a whole batch at once.
        interval = max(1, c.max_stale_steps // (c.min_count + 1))
        latest = (np.arange(c.bins) + .5) * interval / c.bins - interval
        for k, buf in enumerate(self.fifo):
            if buf:
                latest[k] = buf[-1][0]
        for k, _ in self.pending:
            latest[k] = self.optimizer_steps + 1
        deadlines = latest + interval
        overdue = np.flatnonzero(deadlines <= self.optimizer_steps + 1)
        order = overdue[np.argsort(deadlines[overdue], kind='stable')]
        return order[:int(eligible.sum())].tolist(), eligible

    def sample(self, count, device='cpu', eligible=None):
        coverage, eligible = self._coverage_bins(count, eligible)
        if self.cdf is not None and self.optimizer_steps - self.profile_step >= self.config.max_stale_steps:
            self.cdf = None
            self.reason = 'stale distribution; warmup fallback'
        r = self._uniform(count)
        if self.cdf is not None:
            # Adaptive mode: invert the CDF in z=log(u/(1-u)), then return flow time u.
            # right=True skips flat CDF intervals, including leading zero mass.
            k = np.searchsorted(self.cdf, r, side='right') - 1
            mass = self.cdf[k + 1] - self.cdf[k]
            z = self.edges[k] + (r - self.cdf[k]) / mass * (self.edges[k + 1] - self.edges[k])
            u = 1 / (1 + np.exp(-z))
        elif self.config.warmup_type == 'uniform':
            # Warmup uniform in flow time. Truncation avoids singular endpoints.
            u = self.config.min_u + r * (self.config.max_u - self.config.min_u)
        elif self.config.warmup_type == 'log_uniform':
            # Warmup uniform in log-relative-noise gives every statistics bin coverage.
            z = self.edges[0] + r * (self.edges[-1] - self.edges[0])
            u = 1 / (1 + np.exp(-z))
        else:
            # Sigmoid warmup: sample the truncated normal without clipping times.
            lo, hi, reflected = self._normal_bounds()
            p = torch.from_numpy(lo + r * (hi - lo))
            normal = torch.special.ndtri(p).numpy()
            if reflected:
                normal = -normal
            z = normal * self.config.warmup_scale - self.config.warmup_shift
            u = 1 / (1 + np.exp(-z))
        if coverage:
            # Random image positions avoid tying a noise range to one batch slot.
            positions = np.flatnonzero(eligible)
            order = torch.randperm(len(positions), generator=self.generator).numpy()
            positions = positions[order[:len(coverage)]]
            z = self.edges[coverage] + r[positions] * np.diff(self.edges)[coverage]
            u[positions] = 1 / (1 + np.exp(-z))
        return torch.tensor(u, dtype=torch.float32, device=device)

    def set_train_timesteps(self, scheduler, count, device, eligible=None):
        if getattr(scheduler, 'fixed_train_sigmas', None) is not None:
            raise ValueError('InfoNoise cannot replace fixed_train_sigmas; disable InfoNoise for this model')
        if scheduler.config.num_train_timesteps != 1000:
            raise ValueError('InfoNoise requires Toolkit T=1000*u convention')
        # Draw exactly the final batch: no sorted grid, random rank selection, or shift.
        scheduler.timestep_type = 'infonoise'
        scheduler.timesteps = self.sample(count, device, eligible=eligible) * 1000
        scheduler.sigmas = torch.cat((scheduler.timesteps / 1000, scheduler.timesteps.new_zeros(1)))
        scheduler._step_index = None
        scheduler._begin_index = None
        return scheduler.timesteps

    def record(self, u, velocity_mse, eligible=None):
        u = torch.as_tensor(u).detach().cpu().double().numpy().reshape(-1)
        values = torch.as_tensor(velocity_mse).detach().cpu().double().numpy().reshape(-1)
        if len(u) != len(values):
            raise ValueError('InfoNoise requires one error per sampled time')
        eligible = np.ones(len(u), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
        if eligible.shape != u.shape:
            raise ValueError('InfoNoise eligibility must match samples')
        valid = (u > 0) & (u < 1) & np.isfinite(u) & np.isfinite(values) & (values >= 0) & (u >= self.config.min_u * (1 - 1e-6)) & (u <= self.config.max_u * (1 + 1e-6))
        self.invalid_observations += int(np.sum(eligible & ~valid))
        for time, value in zip(u[eligible & valid], values[eligible & valid]):
            k = int(np.searchsorted(self.edges[1:-1], math.log(time / (1 - time)), side='right'))
            # x_hat = x_u - u*v_hat, so clean-data MSE is u² times velocity MSE.
            # Convert each observation before bin averaging; masks are already applied.
            self.pending.append((k, float(time * time * value)))

    def discard_pending(self):
        self.pending.clear()

    def on_optimizer_step(self, *, skipped=False):
        if skipped:
            self.discard_pending()
            return False
        self.optimizer_steps += 1
        for k, value in self.pending:
            self.fifo[k].append((self.optimizer_steps, value))
            self.total_counts[k] += 1
        self.pending.clear()
        c = self.config
        if self.optimizer_steps < c.warmup_steps:
            return False
        if (self.optimizer_steps - c.warmup_steps) % c.refresh_every:
            return False
        self.refresh()
        return True

    def _degraded(self, reason):
        self.reason = reason
        if self.profile_step < 0 or self.optimizer_steps - self.profile_step >= self.config.max_stale_steps:
            self.cdf = None
            self.reason += '; warmup fallback'
        else:
            self.reason += '; retaining previous CDF'

    def refresh(self):
        c = self.config
        self.last_refresh = self.optimizer_steps
        self.refresh_attempts += 1
        for buf in self.fifo:
            while buf and self.optimizer_steps - buf[0][0] >= c.max_stale_steps:
                buf.popleft()
        reliable = np.array([len(buf) >= c.min_count for buf in self.fifo])
        if reliable.sum() < max(2, math.ceil(c.bins * c.min_bin_fraction)):
            self._degraded(f'insufficient recent observations ({reliable.sum()}/{c.bins} bins)')
            return
        means = np.array([np.mean([value for _, value in buf]) if ok else 0 for buf, ok in zip(self.fifo, reliable)])
        fresh = (self.ema_steps >= 0) & (self.optimizer_steps - self.ema_steps < c.max_stale_steps)
        candidate = np.where(fresh, self.mse_ema, 1.0)
        candidate_steps = np.where(fresh, self.ema_steps, -1)
        candidate_steps[reliable] = [self.fifo[k][-1][0] for k in np.flatnonzero(reliable)]
        # A first usable bin mean replaces the initial placeholder. Blend later
        # estimates only while their observation history remains fresh.
        candidate[reliable & ~fresh] = means[reliable & ~fresh]
        known = reliable & fresh
        candidate[known] = (1 - c.beta) * candidate[known] + c.beta * means[known]
        # Missing bins use MSE interpolation on the log-sigma grid; endpoints use nearest recent bin.
        # Do not reuse stale EMA values as if they were fresh observations.
        profile_mse = np.interp(self.centers, self.centers[reliable], candidate[reliable])
        # Paper appendix B.6 defines the onset and gate in sigma-density:
        # r_sigma = m_x/sigma³. Using a log-sigma density here would move the
        # threshold crossing. record() already converted velocity MSE to m_x.
        raw = profile_mse / self.relative_sigmas ** 3
        if not np.isfinite(raw).all() or raw.max() <= 0:
            self._degraded('nonfinite or zero information profile')
            return
        pivot = c.gate_pivot
        if pivot == 0:
            # Scan from high noise: every bin above this crossing is below the
            # onset threshold. Use the bin center, without extrapolating past support.
            pivot = self.relative_sigmas[np.flatnonzero(raw >= raw.max() * c.onset_threshold)[-1]]
        # Paper's smooth endpoint gate: g(sigma)=sigma^n/(sigma^n+c^n).
        gate = 1 / (1 + np.exp(np.clip(c.gate_power * (math.log(pivot) - self.centers), -700, 700)))
        # Paper section 3.2: pi_sigma ∝ r_sigma*g / w_x. Unit velocity MSE
        # has clean-data weight w_x=1/u² (appendix A.6). The grid is in
        # z=log(sigma), so d_sigma = sigma*dz supplies the integration Jacobian.
        # Equivalently pi_z ∝ m_v*u⁴*g/sigma²; training loss is unchanged.
        u = 1 / (1 + np.exp(-self.centers))
        mass = raw * gate * u ** 2 * self.relative_sigmas * np.diff(self.edges)
        normalizer = mass.sum()
        if not math.isfinite(normalizer) or normalizer <= np.finfo(np.float64).tiny:
            self._degraded('invalid distribution normalizer')
            return
        mass /= normalizer
        cdf = np.concatenate(([0.0], np.cumsum(mass)))
        cdf /= cdf[-1]  # Avoid a negative final interval from accumulation roundoff.
        cdf[-1] = 1.0
        self.mse_ema = candidate
        self.ema_steps = candidate_steps
        self.cdf = cdf
        self.pivot = float(pivot)
        self.profile_step = min(self.fifo[k][-1][0] for k in np.flatnonzero(reliable))
        self.reason = 'active'

    def sampling_mass(self, count=1, eligible=None):
        """Expected batch shares below u=.25, from .25 to .75, and above .75.

        Include scheduled exploration. At batch one an exploration update can
        put all probability in one bin; this is a conditional batch distribution.
        """
        coverage, _ = self._coverage_bins(count, eligible)
        c = self.config
        cuts = [0.0]
        for u in (.25, .75):
            if u <= c.min_u:
                probability = 0.0
            elif u >= c.max_u:
                probability = 1.0
            else:
                z = math.log(u / (1 - u))
                if self.cdf is not None and self.optimizer_steps - self.profile_step < c.max_stale_steps:
                    probability = float(np.interp(z, self.edges, self.cdf))
                elif c.warmup_type == 'uniform':
                    probability = (u - c.min_u) / (c.max_u - c.min_u)
                elif c.warmup_type == 'log_uniform':
                    probability = (z - self.edges[0]) / (self.edges[-1] - self.edges[0])
                else:
                    lo, hi, reflected = self._normal_bounds()
                    normal = (z + c.warmup_shift) / c.warmup_scale
                    if reflected:
                        tail = 0.5 * math.erfc(normal / math.sqrt(2))
                        probability = (hi - tail) / (hi - lo)
                    else:
                        cumulative = 0.5 * math.erfc(-normal / math.sqrt(2))
                        probability = (cumulative - lo) / (hi - lo)
            if coverage:
                z = -math.inf if u <= 0 else math.log(u / (1 - u))
                covered = np.clip((z - self.edges[coverage]) / np.diff(self.edges)[coverage], 0, 1).sum()
                probability = ((count - len(coverage)) * probability + covered) / count
            cuts.append(float(np.clip(probability, 0.0, 1.0)))
        return np.diff(cuts + [1.0]).tolist()

    def diagnostics(self):
        result = {'active': self.cdf is not None, 'optimizer_steps': self.optimizer_steps,
                  'reason': self.reason, 'refresh_attempts': self.refresh_attempts,
                  'invalid_observations': self.invalid_observations,
                  'observed_bins': sum(len(buf) >= self.config.min_count for buf in self.fifo),
                  'gate_pivot': self.pivot}
        result['mass_u_ranges'] = self.sampling_mass()
        return result

    def state_dict(self):
        return {'version': self.STATE_VERSION, 'config': asdict(self.config), 'context': dict(self.context),
                'fifo': [list(buf) for buf in self.fifo], 'mse_ema': self.mse_ema.tolist(),
                'ema_steps': self.ema_steps.tolist(),
                'total_counts': self.total_counts.tolist(), 'pending': self.pending.copy(),
                'cdf': None if self.cdf is None else self.cdf.tolist(), 'pivot': self.pivot,
                'optimizer_steps': self.optimizer_steps, 'last_refresh': self.last_refresh,
                'profile_step': self.profile_step, 'refresh_attempts': self.refresh_attempts,
                'invalid_observations': self.invalid_observations, 'reason': self.reason,
                'rng': self.generator.get_state().tolist()}

    def load_state_dict(self, state):
        # Validate into an isolated object so failed loads cannot corrupt live state.
        if not isinstance(state, dict):
            raise ValueError('InfoNoise checkpoint state must be a mapping')
        if state.get('version') != self.STATE_VERSION:
            raise ValueError('Incompatible InfoNoise state version: this sampler uses clean-data MSE '
                             'and sigma-density onset. Start a new run without old sampler state.')
        if state.get('context') != self.context:
            raise ValueError('Incompatible InfoNoise model context')
        saved_config = dict(state.get('config', {}))
        current_config = asdict(self.config)
        saved_config.setdefault('exploration', False)
        saved_config.pop('allow_legacy_resume', None)
        current_config.pop('allow_legacy_resume', None)
        if saved_config != current_config:
            raise ValueError('InfoNoise checkpoint configuration differs from current configuration')
        restored = type(self)(asdict(self.config), seed=0, context=self.context)
        n = self.config.bins
        for name in ('optimizer_steps', 'refresh_attempts', 'invalid_observations'):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Invalid InfoNoise progress: ' + name)
            setattr(restored, name, value)
        for name in ('last_refresh', 'profile_step'):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value <= restored.optimizer_steps:
                raise ValueError('Invalid InfoNoise refresh progress')
            setattr(restored, name, value)
        fifo = state['fifo']
        if len(fifo) != n:
            raise ValueError('Invalid InfoNoise bin count')
        for k, buf in enumerate(fifo):
            if len(buf) > self.config.buffer_size:
                raise ValueError('Invalid InfoNoise buffer size')
            previous = -1
            for step, error in buf:
                if isinstance(step, bool) or not isinstance(step, int) or not previous <= step <= restored.optimizer_steps or step < 0 or not math.isfinite(error) or error < 0:
                    raise ValueError('Invalid InfoNoise observation')
                previous = step
                restored.fifo[k].append((step, float(error)))
        array = np.asarray(state['mse_ema'], dtype=np.float64)
        if array.shape != (n,) or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError('Invalid InfoNoise MSE estimates')
        restored.mse_ema = array.copy()
        ages = state['ema_steps']
        if len(ages) != n or any(isinstance(value, bool) or not isinstance(value, int)
                                 or not -1 <= value <= restored.optimizer_steps for value in ages):
            raise ValueError('Invalid InfoNoise smoothing progress')
        restored.ema_steps = np.asarray(ages, dtype=np.int64)
        if np.any((restored.ema_steps < 0) & (restored.mse_ema != 1)):
            raise ValueError('InfoNoise estimate has no observation history')
        counts = state['total_counts']
        if len(counts) != n or any(
                isinstance(value, bool) or not isinstance(value, int)
                or not len(fifo[k]) <= value <= np.iinfo(np.int64).max
                for k, value in enumerate(counts)):
            raise ValueError('Invalid InfoNoise observation counts')
        restored.total_counts = np.asarray(counts, dtype=np.int64)
        if state['cdf'] is not None:
            array = np.asarray(state['cdf'], dtype=np.float64)
            if array.shape != (n + 1,) or not np.isfinite(array).all() or array[0] != 0 or array[-1] != 1 or (np.diff(array) < 0).any():
                raise ValueError('Invalid InfoNoise checkpoint CDF')
            if restored.profile_step < 0:
                raise ValueError('InfoNoise CDF has no successful refresh')
            restored.cdf = array.copy()
        pivot = state['pivot']
        if pivot is not None and (not math.isfinite(pivot) or pivot <= 0):
            raise ValueError('Invalid InfoNoise gate pivot')
        restored.pivot = pivot
        for k, error in state['pending']:
            if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n or not math.isfinite(error) or error < 0:
                raise ValueError('Invalid InfoNoise pending observation')
            restored.pending.append((k, float(error)))
        restored.generator.set_state(torch.tensor(state['rng'], dtype=torch.uint8))
        restored.reason = str(state['reason'])
        self.__dict__.update(restored.__dict__)
