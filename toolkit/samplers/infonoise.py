"""Optional information-guided timestep sampling.

Paper: Noise Scheduling as Information-Guided Allocation in Diffusion Training,
arXiv:2602.18647v2, section 3.2 and appendix A.6. Velocity errors are converted
to clean-data errors before estimating the information profile. The sampler
accounts for the velocity objective without changing the training loss.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
    from toolkit.config_modules import DatasetConfig, ModelConfig, TrainConfig
    from toolkit.models.base_model import BaseModel
    from toolkit.samplers.custom_flowmatch_sampler import (
        CustomFlowMatchEulerDiscreteScheduler,
    )


@dataclass(frozen=True)
class InfoNoiseConfig:
    bins: int = 16
    warmup_steps: int = 500
    refresh_every: int = 100
    buffer_size: int = 256
    min_count: int = 4
    min_bin_fraction: float = 0.5
    half_life_steps: int = 250
    gate_pivot: float = (
        0.0  # Automatic onset in sigma-density; positive values fix the pivot.
    )
    gate_power: float = 3.0
    onset_threshold: float = 0.002
    min_u: float = 0.001
    max_u: float = 0.999
    max_stale_steps: int = 2000
    # Cover log-noise bins during warmup; sigmoid can leave the tails unobserved.
    warmup_type: str = "log_uniform"  # log_uniform, sigmoid, uniform
    warmup_shift: float = 0.0
    warmup_scale: float = 1.0
    allow_legacy_resume: bool = False

    @classmethod
    def from_dict(
        cls, options: dict[str, Any] | None = None, **defaults: Any
    ) -> InfoNoiseConfig:
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("train.infonoise must be a mapping")
        if "beta" in options:
            # Existing v5 recipes often set beta explicitly.
            options = options.copy()
            beta = options.pop("beta")
            if isinstance(beta, bool) or not isinstance(beta, (float, int)) or not 0 < beta <= 1:
                raise ValueError("Legacy InfoNoise beta must be in (0,1]")
            warnings.warn("InfoNoise ignores legacy beta. Use half_life_steps for age weighting.",
                          FutureWarning, stacklevel=2)
        unknown = set(options) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                "Unknown InfoNoise settings: " + ", ".join(sorted(unknown))
            )
        config = cls(**{**defaults, **options})
        for name in (
            "bins",
            "warmup_steps",
            "refresh_every",
            "half_life_steps",
            "buffer_size",
            "min_count",
            "max_stale_steps",
        ):
            value = getattr(config, name)
            minimum = 0 if name == "warmup_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"InfoNoise {name} must be an integer >= {minimum}")
        if config.bins < 2 or config.min_count > config.buffer_size:
            raise ValueError("InfoNoise needs bins >= 2 and min_count <= buffer_size")
        for name in (
            "min_bin_fraction",
            "gate_pivot",
            "gate_power",
            "onset_threshold",
            "min_u",
            "max_u",
            "warmup_shift",
            "warmup_scale",
        ):
            value = getattr(config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"InfoNoise {name} must be finite")
        if not (0 < config.min_u < config.max_u < 1):
            raise ValueError("InfoNoise requires 0 < min_u < max_u < 1")
        # Float32 T/1000 must remain strictly inside (0,1).
        if (
            np.float32(config.min_u) <= 0
            or np.float32(config.max_u * 1000) / np.float32(1000) >= 1
        ):
            raise ValueError("InfoNoise support is too close to a float32 endpoint")
        if not 0 < config.min_bin_fraction <= 1:
            raise ValueError("InfoNoise min_bin_fraction must be in (0,1]")
        if (
            config.gate_pivot < 0
            or config.gate_power < 2
            or not 0 < config.onset_threshold < 1
        ):
            raise ValueError("Invalid InfoNoise gate settings")
        if (
            config.warmup_type not in ("sigmoid", "uniform", "log_uniform")
            or config.warmup_scale <= 0
        ):
            raise ValueError("Invalid InfoNoise warmup distribution")
        if not isinstance(config.allow_legacy_resume, bool):
            raise ValueError("InfoNoise allow_legacy_resume must be boolean")
        return config


# These modes update parameters during backward, unlike ordinary fused kernels.
_BACKWARD_UPDATE_DEFAULTS = {
    "automagic2": True,
    "automagic3": True,
    "automagic4": True,
    "automagicexperiment": True,
    "adamconvrot": False,
}


def _validate_optimizer_mode(name: str, fused: bool) -> None:
    if name == "automagic2":
        raise ValueError(
            "InfoNoise does not support automagic2: it updates weights during backward"
        )
    if name in _BACKWARD_UPDATE_DEFAULTS and fused:
        raise ValueError(
            f"InfoNoise requires optimizer_params.fused: false with {name}"
        )


def validate_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Check the actual optimizer before any backward pass can change weights."""
    for cls in type(optimizer).__mro__:
        if cls.__module__.startswith("toolkit.optimizers."):
            _validate_optimizer_mode(
                cls.__name__.lower(), getattr(optimizer, "fused", False)
            )


def validate_infonoise(
    train: TrainConfig, model: ModelConfig, datasets: Sequence[DatasetConfig]
) -> None:
    """Check the noise, objective, and update schedule used by the estimator."""
    if train.timestep_type != "infonoise":
        return
    InfoNoiseConfig.from_dict(train.infonoise)
    optimizer_name = train.optimizer.lower()
    optimizer_options = train.optimizer_params or {}
    _validate_optimizer_mode(
        optimizer_name,
        optimizer_options.get(
            "fused", _BACKWARD_UPDATE_DEFAULTS.get(optimizer_name, False)
        ),
    )
    if model.is_v_pred:
        raise ValueError("InfoNoise requires flow velocity MSE, not VP v-prediction")
    required = {
        "noise_scheduler": "flowmatch",
        "loss_type": "mse",
        "loss_target": "noise",
        "content_or_style": "balanced",
        "content_or_style_reg": "balanced",
        "num_train_timesteps": 1000,
        "min_denoising_steps": 0,
        "max_denoising_steps": 999,
        "gradient_accumulation_steps": 1,
        "noise_offset": 0.0,
        "noise_multiplier": 1.0,
        "target_noise_multiplier": 1.0,
        "random_noise_multiplier": 0.0,
        "random_noise_shift": 0.0,
        "noisy_latent_multiplier": 1.0,
        "pred_scaler": 1.0,
        "optimal_noise_pairing_samples": 1,
        "first_timestep_chance": 0.0,
        "snr_gamma": None,
        "min_snr_gamma": None,
        "target_norm_std": None,
        "max_loss": None,
        "diffusion_feature_extractor_path": None,
        "adapter_assist_name_or_path": None,
        "match_adapter_chance": 0.0,
    }
    disabled = (
        "linear_timesteps",
        "linear_timesteps2",
        "train_turbo",
        "learnable_snr_gos",
        "dynamic_noise_offset",
        "do_signal_correction_noise",
        "do_batch_noise_correction",
        "do_signal_amplification",
        "force_consistent_noise",
        "blended_blur_noise",
        "adaptive_scaling_factor",
        "do_cfg",
        "do_random_cfg",
        "do_guidance_loss",
        "do_differential_guidance",
        "correct_pred_norm",
        "match_noise_norm",
        "inverted_mask_prior",
        "diff_output_preservation",
        "blank_prompt_preservation",
        "do_prior_divergence",
        "t0_loss_target",
        "do_fft_loss",
        "do_blank_stabilization",
        "single_item_batching",
        "short_and_long_captions",
        "merge_network_on_save",
    )
    bad = [key for key, value in required.items() if getattr(train, key) != value]
    bad += [key for key in disabled if getattr(train, key)]
    if bad:
        raise ValueError("InfoNoise unsupported train settings: " + ", ".join(bad))
    for dataset in datasets:
        if dataset.prior_reg or dataset.unconditional_path:
            raise ValueError(
                "InfoNoise does not support teacher/prior targets or paired "
                "guidance losses"
            )


def validate_process(process: BaseSDTrainProcess) -> None:
    if process.accelerator.num_processes != 1:
        raise ValueError(
            "InfoNoise currently supports single-process training only; "
            "distributed statistics are not implemented"
        )
    # SDTrainer subclasses BaseSDTrainProcess, which imports this module.
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    if not isinstance(process, SDTrainer):
        raise ValueError(
            "InfoNoise requires the SDTrainer loss and optimizer-step hooks"
        )
    if process.guidance_config is not None:
        raise ValueError(
            "InfoNoise requires the MSE loss path, not a separate guidance objective"
        )


def validate_model(model: BaseModel) -> None:
    """Check flow capabilities after model initialization, before loading weights."""
    if getattr(model, "is_multistage", False):
        raise ValueError("InfoNoise does not support multistage training schedules")
    # Estimator-only users do not need the Diffusers scheduler dependency.
    from toolkit.samplers.custom_flowmatch_sampler import (
        CustomFlowMatchEulerDiscreteScheduler,
    )

    if (
        not isinstance(model.noise_scheduler, CustomFlowMatchEulerDiscreteScheduler)
        or getattr(model.noise_scheduler, "fixed_train_sigmas", None) is not None
        or not model.is_flow_matching
        or model.x0_pred
        or model.prediction_type == "v_prediction"
    ):
        raise ValueError(
            "InfoNoise requires flow velocity training with a continuous "
            "flowmatch schedule; fixed sigmas and x0/VP targets are incompatible"
        )


def validate_target(
    target: torch.Tensor, noise: torch.Tensor, clean: torch.Tensor
) -> bool:
    """Verify the velocity convention on the first real batch."""
    with torch.no_grad():
        velocity = noise - clean
        # Let existing nonfinite-loss handling run; check the next finite batch.
        if not torch.isfinite(target).all() or not torch.isfinite(velocity).all():
            return False
        candidate = target.to(velocity.dtype)
        if not (
            torch.allclose(candidate, velocity) or torch.allclose(candidate, -velocity)
        ):
            raise ValueError(
                "InfoNoise requires a flow velocity MSE target (noise - clean, or "
                "its reverse sign)"
            )
        return True


class InfoNoiseSampler:
    STATE_VERSION = 6  # Age-weighted nodes and a local low-noise correction.

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        seed: int | None = None,
        context: dict[str, Any] | None = None,
        **defaults: Any,
    ) -> None:
        self.config = InfoNoiseConfig.from_dict(options, **defaults)
        c = self.config
        self.context = context or {}
        self.edges = np.linspace(
            math.log(c.min_u / (1 - c.min_u)),
            math.log(c.max_u / (1 - c.max_u)),
            c.bins + 1,
        )
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.cdf_edges = np.linspace(self.edges[0], self.edges[-1], 2 * c.bins + 1)
        # A training draw must never cross the floor boundary inside a CDF cell.
        if c.min_u < .25 < c.max_u:
            cut = math.log(1 / 3)
            self.cdf_edges = np.union1d(self.cdf_edges, [cut])
        self.integration_centers = (self.cdf_edges[:-1] + self.cdf_edges[1:]) / 2
        self.fifo = [deque(maxlen=c.buffer_size) for _ in range(c.bins)]
        self.total_counts = np.zeros(c.bins, dtype=np.int64)
        self.pending = []
        self.last_probe_step = 0
        self.probe_observations = 0
        self.pending_probe_count = 0
        self.probe_attempted = False
        self.optimizer_steps = 0
        self.last_refresh = -1
        self.profile_step = -1
        self.refresh_attempts = 0
        self.invalid_observations = 0
        self.cdf = None
        self.pivot = None
        self.proposed_pivot = None
        self.pre_floor_mass = None
        self.reason = "warmup"
        self.generator = torch.Generator(device="cpu")
        seed = torch.initial_seed() if seed is None else seed
        self.generator.manual_seed(seed)
        self.probe_generator = torch.Generator(device="cpu")
        self.probe_generator.manual_seed((seed & (2**64 - 1)) ^ 0x9E3779B97F4A7C15)
        # Validate truncated normal mass without drawing or touching global RNG.
        if c.warmup_type == "sigmoid":
            lo, hi, _ = self._normal_bounds()
            if hi - lo <= np.finfo(np.float64).tiny:
                raise ValueError(
                    "InfoNoise sigmoid warmup has negligible probability on "
                    "configured support"
                )

    def _normal_bounds(self) -> tuple[float, float, bool]:
        c = self.config
        # u=1-sigmoid(N(shift, scale²)), so z=logit(u) has mean -shift.
        a, b = (
            (z + c.warmup_shift) / c.warmup_scale
            for z in (self.edges[0], self.edges[-1])
        )
        if a >= 0:
            # Use survival probabilities in the positive tail; CDF values round to 1.
            return (
                0.5 * math.erfc(b / math.sqrt(2)),
                0.5 * math.erfc(a / math.sqrt(2)),
                True,
            )
        return (
            0.5 * math.erfc(-a / math.sqrt(2)),
            0.5 * math.erfc(-b / math.sqrt(2)),
            False,
        )

    def _uniform(self, count: int) -> NDArray[np.float64]:
        if not isinstance(count, int) or count < 1:
            raise ValueError("InfoNoise sample count must be positive")
        eps = torch.finfo(torch.float64).eps
        return (
            torch.rand(count, generator=self.generator, dtype=torch.float64).numpy()
            * (1 - 2 * eps)
            + eps
        )

    def probe_times(self, eligible: Sequence[bool]) -> torch.Tensor | None:
        """Budget an extra forward for strata short of fresh observations."""
        c = self.config
        count = len(eligible)
        if not any(eligible) or self.probe_attempted:
            return None
        step = self.optimizer_steps + 1
        # Budget forward examples, including regularization examples carried
        # in the same batch. Run at most one probe batch per optimizer update.
        rate = c.bins * (c.min_count + 1) / c.max_stale_steps
        if (step - self.last_probe_step) * rate < count:
            return None
        counts = np.bincount([k for k, _, _ in self.pending], minlength=c.bins)
        interval = min(c.max_stale_steps, math.ceil(count / rate))
        latest = np.full(c.bins, -c.max_stale_steps)
        for k, buf in enumerate(self.fifo):
            counts[k] += sum(step + interval - t < c.max_stale_steps for t, _, _ in buf)
            if buf:
                latest[k] = buf[-1][0]
        if counts.min() >= c.min_count:
            return None
        bins = []
        for use in eligible:
            k = int(np.lexsort((latest, counts))[0])
            bins.append(k)
            if use:
                counts[k] += 1
                latest[k] = step
        self.probe_attempted = True
        r = torch.rand(count, generator=self.probe_generator, dtype=torch.float64).numpy()
        z = self.edges[bins] + r * np.diff(self.edges)[bins]
        return torch.as_tensor(1000 / (1 + np.exp(-z)), dtype=torch.float32)

    @staticmethod
    def _validate_eligibility(count, eligible):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("InfoNoise sample count must be positive")
        if eligible is not None and np.shape(eligible) != (count,):
            raise ValueError("InfoNoise eligibility must match samples")

    def sample(
        self,
        count: int,
        device: str | torch.device = "cpu",
        eligible: ArrayLike | torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_eligibility(count, eligible)
        if (
            self.cdf is not None
            and self.optimizer_steps - self.profile_step >= self.config.max_stale_steps
        ):
            self.cdf = None
            self.reason = "stale distribution; warmup fallback"
        r = self._uniform(count)
        if self.cdf is not None:
            # Adaptive mode: invert the CDF in z=log(u/(1-u)), then return flow time u.
            # right=True skips flat CDF intervals, including leading zero mass.
            k = np.searchsorted(self.cdf, r, side="right") - 1
            mass = self.cdf[k + 1] - self.cdf[k]
            z = self.cdf_edges[k] + (r - self.cdf[k]) / mass * (
                self.cdf_edges[k + 1] - self.cdf_edges[k]
            )
            u = 1 / (1 + np.exp(-z))
        elif self.config.warmup_type == "uniform":
            # Warmup uniform in flow time. Truncation avoids singular endpoints.
            u = self.config.min_u + r * (self.config.max_u - self.config.min_u)
        elif self.config.warmup_type == "log_uniform":
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
        return torch.tensor(u, dtype=torch.float32, device=device)

    def set_train_timesteps(
        self,
        scheduler: CustomFlowMatchEulerDiscreteScheduler,
        count: int,
        device: str | torch.device,
        eligible: ArrayLike | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(scheduler, "fixed_train_sigmas", None) is not None:
            raise ValueError(
                "InfoNoise cannot replace fixed_train_sigmas; disable InfoNoise "
                "for this model"
            )
        if scheduler.config.num_train_timesteps != 1000:
            raise ValueError("InfoNoise requires Toolkit T=1000*u convention")
        # Draw exactly the final batch: no sorted grid, random rank selection, or shift.
        scheduler.timestep_type = "infonoise"
        scheduler.timesteps = self.sample(count, device, eligible=eligible) * 1000
        scheduler.sigmas = torch.cat(
            (scheduler.timesteps / 1000, scheduler.timesteps.new_zeros(1))
        )
        scheduler._step_index = None
        scheduler._begin_index = None
        return scheduler.timesteps

    def record(
        self,
        u: ArrayLike | torch.Tensor,
        velocity_mse: ArrayLike | torch.Tensor,
        eligible: ArrayLike | torch.Tensor | None = None,
        *,
        probe: bool = False,
    ) -> None:
        u = torch.as_tensor(u).detach().cpu().double().numpy().reshape(-1)
        values = (
            torch.as_tensor(velocity_mse).detach().cpu().double().numpy().reshape(-1)
        )
        if len(u) != len(values):
            raise ValueError("InfoNoise requires one error per sampled time")
        eligible = (
            np.ones(len(u), dtype=bool)
            if eligible is None
            else np.asarray(eligible, dtype=bool)
        )
        if eligible.shape != u.shape:
            raise ValueError("InfoNoise eligibility must match samples")
        valid = (
            (u > 0)
            & (u < 1)
            & np.isfinite(u)
            & np.isfinite(values)
            & (values >= 0)
            & (u >= self.config.min_u * (1 - 1e-6))
            & (u <= self.config.max_u * (1 + 1e-6))
        )
        self.invalid_observations += int(np.sum(eligible & ~valid))
        for time, value in zip(u[eligible & valid], values[eligible & valid]):
            k = int(
                np.searchsorted(
                    self.edges[1:-1], math.log(time / (1 - time)), side="right"
                )
            )
            # x_hat = x_u - u*v_hat, so clean-data MSE is u² times velocity MSE.
            # Convert each observation before bin averaging; masks are already applied.
            self.pending.append((k, float(time * time * value), float(time)))
        if probe:
            self.pending_probe_count += int(np.sum(eligible & valid))

    def discard_pending(self) -> None:
        self.pending.clear()
        self.pending_probe_count = 0
        self.probe_attempted = False

    def on_optimizer_step(self, *, skipped: bool = False) -> bool:
        if skipped:
            self.discard_pending()
            return False
        self.optimizer_steps += 1
        for k, value, time in self.pending:
            self.fifo[k].append((self.optimizer_steps, value, time))
            self.total_counts[k] += 1
        if self.probe_attempted:
            self.last_probe_step = self.optimizer_steps
        self.probe_observations += self.pending_probe_count
        self.discard_pending()
        c = self.config
        if self.optimizer_steps < c.warmup_steps:
            return False
        if (self.optimizer_steps - c.warmup_steps) % c.refresh_every:
            return False
        self.refresh()
        return True

    def _degraded(self, reason: str) -> None:
        self.reason = reason
        if (
            self.profile_step < 0
            or self.optimizer_steps - self.profile_step >= self.config.max_stale_steps
        ):
            self.cdf = None
            self.reason += "; warmup fallback"
        else:
            self.reason += "; retaining previous CDF"

    def refresh(self) -> None:
        c = self.config
        self.last_refresh = self.optimizer_steps
        self.refresh_attempts += 1
        for buf in self.fifo:
            while buf and self.optimizer_steps - buf[0][0] >= c.max_stale_steps:
                buf.popleft()
        reliable = np.array([len(buf) >= c.min_count for buf in self.fifo])
        if reliable.sum() < max(2, math.ceil(c.bins * c.min_bin_fraction)):
            self._degraded(
                f"insufficient recent observations ({reliable.sum()}/{c.bins} bins)"
            )
            return
        profile, profile_step = self._estimate_profile(reliable)
        sigma = np.exp(self.integration_centers)
        u = sigma / (1 + sigma)
        # Appendix B.6 sets the onset using density per unit sigma. Integrating
        # the velocity sampler in log-sigma multiplies by u² * sigma * dz.
        raw = profile / sigma**3
        if not np.isfinite(raw).all() or raw.max() <= 0:
            self._degraded("nonfinite or zero information profile")
            return
        ungated = profile * (1 - u)**2 * np.diff(self.cdf_edges)
        proposal = self._gate_pivot(raw)
        pivot = proposal
        if c.gate_pivot == 0 and self.pivot is not None:
            pivot = math.exp((math.log(proposal) + math.log(self.pivot)) / 2)
        mass = self._gate_mass(ungated, sigma, pivot)
        normalizer = mass.sum()
        if not math.isfinite(normalizer) or normalizer <= 0:
            self._degraded("invalid distribution normalizer")
            return
        mass /= normalizer
        pre_floor_mass = self._mass_ranges(mass)
        if c.min_u < .25 < c.max_u:
            low = self.cdf_edges[1:] <= math.log(1 / 3)
            target = min(0.05, ungated[low].sum() / ungated.sum())
            share = mass[low].sum()
            if share < target:
                # Normalize each side independently to retain its conditional shape,
                # even if the low tail underflows relative to the global peak.
                low_mass = self._gate_mass(ungated[low], sigma[low], pivot)
                mass[low] = target * (low_mass / low_mass.sum())
                mass[~low] *= (1 - target) / mass[~low].sum()
        cdf = np.concatenate(([0.0], np.cumsum(mass)))
        cdf /= cdf[-1]
        cdf[-1] = 1.0
        self.proposed_pivot = proposal
        self.pre_floor_mass = pre_floor_mass
        self.cdf = cdf
        self.pivot = float(pivot)
        self.profile_step = profile_step
        self.reason = "active"

    def _estimate_profile(self, reliable):
        nodes, values, latest = [], [], []
        for k in np.flatnonzero(reliable):
            steps, errors, times = np.asarray(self.fifo[k]).T
            # A common factor cancels in a weighted mean. Anchor at the newest
            # observation so an old, still-eligible bin cannot underflow.
            weights = np.exp2((steps - steps[-1]) / self.config.half_life_steps)
            nodes.append(np.average(np.log(times / (1 - times)), weights=weights))
            values.append(np.average(errors, weights=weights))
            latest.append(self.fifo[k][-1][0])
        # Extend velocity risk into sparsely observed tails. Extending clean-data
        # error as a constant can invent a low-noise peak after division by sigma³.
        z = self.integration_centers
        profile = np.interp(z, nodes, values)
        u = 1 / (1 + np.exp(-z))
        for boundary, outside, value in ((nodes[0], z < nodes[0], values[0]),
                                          (nodes[-1], z > nodes[-1], values[-1])):
            anchor_u = 1 / (1 + math.exp(-boundary))
            profile[outside] = value * (u[outside] / anchor_u)**2
        return profile, min(latest)

    def _gate_mass(self, ungated, sigma, pivot):
        # Subtract the maximum log gate on supported cells. Its multiplicative
        # constant cancels during normalization, including for very large pivots.
        positive = ungated > 0
        mass = np.zeros_like(ungated)
        if positive.any():
            exponent = self.config.gate_power * (math.log(pivot) - np.log(sigma[positive]))
            log_mass = np.log(ungated[positive]) - np.logaddexp(0, exponent)
            mass[positive] = np.exp(log_mass - log_mass.max())
        return mass

    def _gate_pivot(self, raw):
        if self.config.gate_pivot > 0:
            return self.config.gate_pivot
        threshold = raw.max() * self.config.onset_threshold
        k = int(np.flatnonzero(raw >= threshold)[-1])
        crossing = self.integration_centers[k]
        if k + 1 < len(raw):
            fraction = (raw[k] - threshold) / (raw[k] - raw[k + 1])
            crossing += fraction * (self.integration_centers[k + 1] - crossing)
        return math.exp(crossing)

    def _mass_ranges(self, mass):
        cdf = np.r_[0., np.cumsum(mass)]
        cuts = np.interp([math.log(1 / 3), math.log(3)], self.cdf_edges, cdf)
        return np.diff(np.r_[0., np.clip(cuts, 0., 1.), 1.]).tolist()

    def sampling_mass(
        self, count: int = 1, eligible: ArrayLike | torch.Tensor | None = None
    ) -> list[float]:
        """Training probabilities below u=.25, between .25 and .75, and above .75."""
        self._validate_eligibility(count, eligible)
        c = self.config
        cuts = [0.0]
        for u in (0.25, 0.75):
            if u <= c.min_u:
                probability = 0.0
            elif u >= c.max_u:
                probability = 1.0
            else:
                z = math.log(u / (1 - u))
                if (
                    self.cdf is not None
                    and self.optimizer_steps - self.profile_step < c.max_stale_steps
                ):
                    probability = float(np.interp(z, self.cdf_edges, self.cdf))
                elif c.warmup_type == "uniform":
                    probability = (u - c.min_u) / (c.max_u - c.min_u)
                elif c.warmup_type == "log_uniform":
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
            cuts.append(float(np.clip(probability, 0.0, 1.0)))
        return np.diff(cuts + [1.0]).tolist()

    def diagnostics(self) -> dict[str, Any]:
        result = {
            "active": self.cdf is not None,
            "optimizer_steps": self.optimizer_steps,
            "reason": self.reason,
            "refresh_attempts": self.refresh_attempts,
            "invalid_observations": self.invalid_observations,
            "observed_bins": sum(
                len(buf) >= self.config.min_count for buf in self.fifo
            ),
            "gate_pivot": self.pivot,
            "proposed_pivot": self.proposed_pivot,
            "pre_floor_mass": None if self.pre_floor_mass is None else self.pre_floor_mass.copy(),
            "probe_observations": self.probe_observations,
        }
        result["mass_u_ranges"] = self.sampling_mass()
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config": asdict(self.config),
            "context": dict(self.context),
            "fifo": [list(buf) for buf in self.fifo],
            "total_counts": self.total_counts.tolist(),
            "pending": self.pending.copy(),
            "cdf_edges": self.cdf_edges.tolist(),
            "cdf": None if self.cdf is None else self.cdf.tolist(),
            "pivot": self.pivot,
            "proposed_pivot": self.proposed_pivot,
            "pre_floor_mass": None if self.pre_floor_mass is None else self.pre_floor_mass.copy(),
            "optimizer_steps": self.optimizer_steps,
            "last_refresh": self.last_refresh,
            "profile_step": self.profile_step,
            "refresh_attempts": self.refresh_attempts,
            "invalid_observations": self.invalid_observations,
            "reason": self.reason,
            "rng": self.generator.get_state().tolist(),
            "probe_rng": self.probe_generator.get_state().tolist(),
            "last_probe_step": self.last_probe_step,
            "probe_observations": self.probe_observations,
            "pending_probe_count": self.pending_probe_count,
            "probe_attempted": self.probe_attempted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # Validate into an isolated object so failed loads cannot corrupt live state.
        if not isinstance(state, dict):
            raise ValueError("InfoNoise checkpoint state must be a mapping")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(
                "Incompatible InfoNoise state version: v6 requires an age-weighted profile "
                "and local low-noise floor. Start a new run without the old sampler state."
            )
        if state.get("context") != self.context:
            raise ValueError("Incompatible InfoNoise model context")
        saved_config = dict(state.get("config", {}))
        current_config = asdict(self.config)
        saved_config.pop("allow_legacy_resume", None)
        current_config.pop("allow_legacy_resume", None)
        if saved_config != current_config:
            raise ValueError(
                "InfoNoise checkpoint configuration differs from current configuration"
            )
        restored = type(self)(asdict(self.config), seed=0, context=self.context)
        n = self.config.bins
        for name in ("optimizer_steps", "refresh_attempts", "invalid_observations",
                     "last_probe_step", "probe_observations", "pending_probe_count"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Invalid InfoNoise progress: " + name)
            setattr(restored, name, value)
        for name in ("last_refresh", "profile_step"):
            value = state[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not -1 <= value <= restored.optimizer_steps
            ):
                raise ValueError("Invalid InfoNoise refresh progress")
            setattr(restored, name, value)
        def check_time(k, time):
            if not math.isfinite(time) or not 0 < time < 1:
                raise ValueError("Invalid InfoNoise observation time")
            if not self.config.min_u * (1 - 1e-6) <= time <= self.config.max_u * (1 + 1e-6):
                raise ValueError("InfoNoise observation time outside support")
            z = math.log(time / (1 - time))
            if k != np.searchsorted(self.edges[1:-1], z, side="right"):
                raise ValueError("InfoNoise observation time does not match its stratum")

        fifo = state["fifo"]
        if len(fifo) != n:
            raise ValueError("Invalid InfoNoise bin count")
        for k, buf in enumerate(fifo):
            if len(buf) > self.config.buffer_size:
                raise ValueError("Invalid InfoNoise buffer size")
            previous = -1
            for step, error, time in buf:
                if (
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or not previous <= step <= restored.optimizer_steps
                    or step < 0
                    or not math.isfinite(error)
                    or error < 0
                ):
                    raise ValueError("Invalid InfoNoise observation")
                previous = step
                check_time(k, time)
                restored.fifo[k].append((step, float(error), float(time)))
        counts = state["total_counts"]
        if len(counts) != n or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not len(fifo[k]) <= value <= np.iinfo(np.int64).max
            for k, value in enumerate(counts)
        ):
            raise ValueError("Invalid InfoNoise observation counts")
        restored.total_counts = np.asarray(counts, dtype=np.int64)
        if not np.array_equal(state["cdf_edges"], self.cdf_edges):
            raise ValueError("Invalid InfoNoise checkpoint CDF grid")
        if state["cdf"] is not None:
            array = np.asarray(state["cdf"], dtype=np.float64)
            if (
                array.shape != self.cdf_edges.shape
                or not np.isfinite(array).all()
                or array[0] != 0
                or array[-1] != 1
                or (np.diff(array) < 0).any()
            ):
                raise ValueError("Invalid InfoNoise checkpoint CDF")
            if restored.profile_step < 0:
                raise ValueError("InfoNoise CDF has no successful refresh")
            restored.cdf = array.copy()
        pivot = state["pivot"]
        if pivot is not None and (not math.isfinite(pivot) or pivot < 0):
            raise ValueError("Invalid InfoNoise gate pivot")
        proposal = state["proposed_pivot"]
        shares = state["pre_floor_mass"]
        if any(v is not None for v in (pivot, proposal, shares)) or restored.cdf is not None:
            if (pivot is None or pivot <= 0 or proposal is None
                    or not math.isfinite(proposal) or proposal <= 0):
                raise ValueError("Invalid InfoNoise gate metadata")
            shares = np.asarray(shares, dtype=np.float64)
            if (shares.shape != (3,) or not np.isfinite(shares).all()
                    or (shares < 0).any() or not math.isclose(shares.sum(), 1.)):
                raise ValueError("Invalid InfoNoise pre-floor mass")
            restored.pre_floor_mass = shares.tolist()
            restored.proposed_pivot = proposal
            restored.pivot = pivot
        for k, error, time in state["pending"]:
            if (
                isinstance(k, bool)
                or not isinstance(k, int)
                or not 0 <= k < n
                or not math.isfinite(error)
                or error < 0
            ):
                raise ValueError("Invalid InfoNoise pending observation")
            check_time(k, time)
            restored.pending.append((k, float(error), float(time)))
        for field, generator in (("rng", restored.generator), ("probe_rng", restored.probe_generator)):
            rng = state[field]
            if not isinstance(rng, (list, tuple)) or any(
                type(value) is not int or not 0 <= value <= 255 for value in rng
            ):
                raise ValueError("Invalid InfoNoise checkpoint RNG bytes")
            generator.set_state(torch.tensor(rng, dtype=torch.uint8))
        restored.probe_attempted = state["probe_attempted"]
        if (type(restored.probe_attempted) is not bool
                or restored.last_probe_step > restored.optimizer_steps
                or restored.pending_probe_count > len(restored.pending)
                or restored.probe_observations > sum(restored.total_counts.tolist())):
            raise ValueError("Invalid InfoNoise probe state")
        restored.reason = str(state["reason"])
        self.__dict__.update(restored.__dict__)
