"""CPU tests for the sampler and trainer, using a tiny model without downloaded weights."""
from collections import OrderedDict
from contextlib import nullcontext, redirect_stdout
import copy
import json
import io
import random
import sqlite3
import math
from pathlib import Path
import os
import tempfile
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import yaml

# Albumentations otherwise checks for updates during trainer import.
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.diffusion_models.qwen_image_2.qwen_image_2 import QwenImage2Model
from toolkit.models.base_model import BaseModel, BlankNetwork
from toolkit.timer import Timer
from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState

from toolkit.config import get_config
from toolkit.config_modules import (DatasetConfig, ModelConfig, NetworkConfig,
                                    SampleConfig, SaveConfig, TrainConfig,
                                    preprocess_dataset_raw_config, validate_configs)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.samplers.infonoise import (InfoNoiseConfig, InfoNoiseSampler,
                                       validate_infonoise, validate_process, validate_optimizer, validate_model)

def integrate(y, x):
    return np.sum((y[1:] + y[:-1]) * np.diff(x) / 2)


ROOT = Path(__file__).resolve().parents[1]
QWEN_YAML = '''---
job: extension
config:
  name: qwen21_infonoise_lora
  process:
    - type: sd_trainer
      training_folder: /REPLACE_WITH_OUTPUT_PATH
      device: cuda:0
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
        save_format: safetensors
      datasets:
        - folder_path: /REPLACE_WITH_DATASET_PATH
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: 1024
      train:
        batch_size: 1
        gradient_accumulation: 4
        steps: 3000
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        loss_type: mse
        optimizer: adamw
        lr: 0.0001
        dtype: bf16
        timestep_type: infonoise # Opt in; existing recipes stay unchanged.
        infonoise:
          warmup_type: log_uniform # Broad coverage in log-relative-noise.
          warmup_steps: 500 # Optimizer updates, including accumulation.
          refresh_every: 100
          bins: 32
          buffer_size: 256
          min_count: 4
          min_bin_fraction: 0.5 # Interpolate sparse bins after sufficient coverage.
          beta: 0.9 # Weight on the new bin mean.
          gate_pivot: 0 # Derive the gate pivot from the current information profile.
          gate_power: 3
          max_stale_steps: 2000
      model:
        arch: qwen_image_2
        name_or_path: Comfy-Org/Qwen-Image-2.1
        extras_name_or_path: Qwen/Qwen-Image-2.1
        quantize: true
        qtype: convrot8
        quantize_te: true
        qtype_te: convrot8
        model_kwargs:
          turbo: false
          rgba: false
      sample:
        sampler: flowmatch
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
          - A woman with short dark hair, neutral expression, wearing a cotton shirt, standing beside a plain wall, natural window light.
        seed: 42
        walk_seed: false
        guidance_scale: 4.0
        sample_steps: 30
meta:
  name: qwen21_infonoise_lora
  version: '1.0'
'''



PREPARE = BaseSDTrainProcess.process_general_training_batch
LOSS = SDTrainer.calculate_loss
BASE_SIGMAS = BaseSDTrainProcess.get_sigmas
QWEN_TARGET = QwenImage2Model.get_loss_target
ADD_NOISE = BaseModel.add_noise
LOAD_META = BaseSDTrainProcess.load_training_state_from_metadata
SAVE = BaseSDTrainProcess.save


def parse_metadata(path):
    with safe_open(path, framework='pt') as f:
        return {k: json.loads(v) if k != 'format' else v for k, v in f.metadata().items()}


def batch(size=3, shape=(2, 3, 4), regs=None, multiplier=1.0):
    regs = [False] * size if regs is None else regs
    return NS(latents=torch.arange(size * math.prod(shape)).reshape(size, *shape).float() / 20,
              tensor=None, unconditional_tensor=None, unconditional_latents=None,
              mask_tensor=None, control_tensor=None, sigmas=None,
              file_items=[NS(is_reg=reg, prior_reg=False) for reg in regs], loss_multiplier_list=[multiplier] * size,
              get_caption_list=lambda: ['plain photograph'] * size,
              get_is_reg_list=lambda: regs)



def expected_mask_weights(mask, latents):
    if mask is None:
        return 1.0
    # Independent nearest-neighbor indexing, followed by existing batch-wide normalization.
    h, w = latents.shape[-2:]
    rows = torch.arange(h) * mask.shape[-2] // h
    cols = torch.arange(w) * mask.shape[-1] // w
    weights = mask.half().index_select(2, rows).index_select(3, cols)
    weights = weights.expand(-1, latents.shape[1], -1, -1).to(latents.dtype)
    return weights / weights.mean()


def process(sampler=None, **train_options):
    scheduler = CustomFlowMatchEulerDiscreteScheduler()
    train = TrainConfig(noise_scheduler='flowmatch', **train_options)
    sd = NS(noise_scheduler=scheduler, is_flux=False, arch='qwen_image_2', is_multistage=False,
            unet=NS(config=NS(patch_size=2)), is_flow_matching=True, prediction_type='epsilon',
            scale_loss=lambda loss: loss, do_masked_loss=True)
    sd.add_noise = MethodType(ADD_NOISE, sd)
    sd.get_loss_target = lambda **kw: QWEN_TARGET(sd, **kw)
    p = NS(train_config=train, model_config=ModelConfig(arch='qwen_image_2', name_or_path='unused'),
           sd=sd, infonoise_sampler=sampler, timer=lambda _: nullcontext(), device='cpu',
           device_torch=torch.device('cpu'), embedding=None, adapter=None, trigger_word=None,
           dfe=None, snr_gos=None, additional_logs={}, step_num=0,
           logging_config=NS(log_every=1), logger=NS(log=lambda _: None),
           accelerator=NS(is_main_process=True))
    p.get_noise = lambda latents, *args, **kw: torch.randn_like(latents)
    return p


def ready_sampler(**options):
    s = InfoNoiseSampler({'bins': 16, 'min_count': 1, 'warmup_steps': 1,
                          'refresh_every': 1, 'beta': 1.0, 'exploration': False, 'gate_pivot': .15, **options}, seed=77)
    u = 1 / (1 + np.exp(-s.centers))
    s.record(u, np.ones_like(u))
    s.on_optimizer_step()
    return s


class InfoNoiseTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        # Accelerate shares state between instances; keep CPU fixtures isolated from other tests.
        self.enterContext(mock.patch.object(PartialState, '_shared_state', {}))
        self.enterContext(mock.patch.object(AcceleratorState, '_shared_state', {}))
        self.enterContext(mock.patch('toolkit.accelerator.global_accelerator', Accelerator(cpu=True)))
        original_print = print
        def checked_print(*args, **kwargs):
            # Trainer mask failures are caught and printed; they must fail every test.
            if args and str(args[0]) == 'Could not apply mask multiplier to loss':
                self.fail('Unexpected mask application failure')
            original_print(*args, **kwargs)
        self.enterContext(mock.patch('builtins.print', side_effect=checked_print))


class InfoNoiseMathTest(InfoNoiseTestCase):
    def test_velocity_conversion_and_objective_weight(self):
        gen = torch.Generator().manual_seed(19)
        x = torch.randn(17, 2, 3, 4, generator=gen, dtype=torch.float64)
        eps = torch.randn(x.shape, generator=gen, dtype=torch.float64)
        u = torch.linspace(.01, .99, 17, dtype=torch.float64).reshape(-1, 1, 1, 1)
        v = eps - x
        prediction = v + torch.randn(v.shape, generator=gen, dtype=torch.float64)
        noisy = (1 - u) * x + u * eps
        xhat = noisy - u * prediction
        torch.testing.assert_close((xhat - x).square(), u.square() * (v - prediction).square())
        torch.testing.assert_close((xhat - x).square() / u.square(), (v - prediction).square())

    def test_gaussian_prior_entropy_derivative(self):
        # Independent derivative of closed-form Gaussian conditional entropy.
        tau2 = 2.3
        u = np.linspace(.03, .97, 400)
        def variance(time):
            return 1 / (1 / tau2 + ((1 - time) / time) ** 2)
        h = 1e-6
        derivative = (np.log(variance(u + h)) - np.log(variance(u - h))) / (4 * h)
        expected = (1 - u) / u ** 3 * variance(u)
        np.testing.assert_allclose(derivative, expected, rtol=2e-8)
        # Fixed velocity weight divides the information rate by 1/u**2.
        paper_pi_u = derivative * u ** 2
        velocity_risk = variance(u) / u ** 2
        np.testing.assert_allclose(paper_pi_u, u * (1 - u) * velocity_risk, rtol=2e-8)

    def test_gaussian_oracle_matches_paper_velocity_allocation(self):
        for pivot in (0., .15):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(pivot=pivot, dtype=dtype):
                    s = InfoNoiseSampler({'bins': 256, 'warmup_steps': 1, 'refresh_every': 1,
                                          'min_count': 1, 'beta': 1., 'gate_pivot': pivot}, seed=4)
                    u = torch.sigmoid(torch.tensor(s.centers, dtype=torch.float32))
                    tau2 = 1.7
                    velocity_risk = tau2 / ((1 - u) ** 2 * tau2 + u ** 2)
                    s.record(u, velocity_risk.to(dtype))
                    s.on_optimizer_step()
                    # Integrate closed-form density densely in flow time, independently
                    # of midpoint bin integration. Input statistics include low precision.
                    grid = np.linspace(s.edges[0], s.edges[-1], 100001)
                    t = 1 / (1 + np.exp(-grid))
                    sigma = t / (1 - t)
                    if pivot == 0:
                        v = 1 / (1 + np.exp(-s.centers))
                        exact_clean_risk = tau2 * v ** 2 / ((1 - v) ** 2 * tau2 + v ** 2)
                        entropy_rate = exact_clean_risk * np.exp(-3 * s.centers)
                        crossing = np.flatnonzero(entropy_rate >= .002 * entropy_rate.max())[-1]
                        pivot_value = math.exp(s.centers[crossing])
                        self.assertEqual(s.pivot, pivot_value)
                    else:
                        pivot_value = pivot
                    gate = sigma ** 3 / (sigma ** 3 + pivot_value ** 3)
                    denom = (1 - t) ** 2 * tau2 + t ** 2
                    density_u = gate * tau2 * t * (1 - t) / denom
                    dx = np.diff(t)
                    cumulative = np.r_[0, np.cumsum((density_u[1:] + density_u[:-1]) * dx / 2)]
                    cumulative /= cumulative[-1]
                    expected = np.interp(s.edges, grid, cumulative)
                    np.testing.assert_allclose(s.cdf, expected, atol=4e-4, rtol=0)
                    # The old velocity-error rule omitted t**4 and overallocated low noise.
                    old_density = density_u / t ** 4
                    old_density /= integrate(old_density, t)
                    old_low = integrate(old_density[t < .25], t[t < .25])
                    self.assertLess(s.diagnostics()['mass_u_ranges'][0], old_low - .3)

    def test_clean_data_errors_are_converted_before_bin_averaging(self):
        # Different times in the same bin must use their own conversion factors.
        times = torch.tensor([.06, .2, .7, .9], dtype=torch.float32)
        velocity_mse = torch.tensor([2., 5., 3., 7.], dtype=torch.float32)
        for size in (1, 4):
            with self.subTest(batch=size):
                sampler = InfoNoiseSampler({'bins': 2, 'min_u': .05, 'max_u': .95,
                                            'min_count': 1, 'warmup_steps': 4 // size,
                                            'refresh_every': 1, 'beta': 1., 'exploration': False})
                for start in range(0, len(times), size):
                    sampler.record(times[start:start + size], velocity_mse[start:start + size])
                    sampler.on_optimizer_step()
                clean_mse = times.double().square().numpy() * velocity_mse.numpy()
                expected = clean_mse.reshape(2, 2).mean(1)
                np.testing.assert_allclose(sampler.mse_ema, expected, rtol=0, atol=1e-15)
                center_u = 1 / (1 + np.exp(-sampler.centers))
                self.assertFalse(np.allclose(expected, velocity_mse.numpy().reshape(2, 2).mean(1) * center_u ** 2))

    def test_masked_velocity_conversion_matches_direct_clean_reconstruction(self):
        for size in (1, 4):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(batch=size, dtype=dtype):
                    gen = torch.Generator().manual_seed(19)
                    clean = torch.randn((size, 2, 3, 4), generator=gen, dtype=torch.float32).to(dtype)
                    noise = torch.randn(clean.shape, generator=gen, dtype=torch.float32).to(dtype)
                    velocity = noise - clean
                    prediction = (velocity + .2).requires_grad_(True)
                    times = torch.linspace(.1, .9, size, dtype=torch.float32)
                    # Mirror the real latent-dtype target, then reconstruct in float32.
                    u = times.reshape(-1, 1, 1, 1)
                    mask = torch.ones_like(clean, dtype=torch.float32)
                    mask[:, :, :, :2] = .1
                    mask /= mask.mean()
                    masked_velocity_error = ((prediction.float() - velocity.float()).square() * mask).flatten(1).mean(1)
                    sampler = InfoNoiseSampler()
                    sampler.record(times, masked_velocity_error)
                    # x_hat-x = u*(v-v_hat), evaluated independently in reconstruction space.
                    reconstruction = clean.float() + u * (velocity.float() - prediction.float())
                    direct_error = ((reconstruction - clean.float()).square() * mask).flatten(1).mean(1)
                    np.testing.assert_allclose([value for _, value in sampler.pending],
                                               direct_error.detach().numpy(), rtol=2e-5, atol=1e-8)
                    self.assertFalse(any(torch.is_tensor(v) for _, v in sampler.pending))

    def test_coordinate_jacobians_give_same_probabilities(self):
        # Smooth arbitrary profile; integrate independently in four coordinates.
        t = np.linspace(.02, .98, 20001)
        sigma = t / (1 - t)
        z = np.log(sigma)
        logsnr = -2 * z
        density_z = np.exp(-.5 * (z - .4) ** 2)
        integrals = [integrate(density_z, z),
                     integrate(density_z / (t * (1 - t)), t),
                     integrate(density_z / sigma, sigma),
                     integrate((density_z / 2)[::-1], logsnr[::-1])]
        np.testing.assert_allclose(integrals, integrals[0], rtol=2e-7)

    def test_actual_gaussian_bayes_predictor_error(self):
        s = InfoNoiseSampler({'bins': 8}, seed=0)
        gen = torch.Generator().manual_seed(22)
        tau2, u = 1.7, .4
        x = torch.randn((20000, 1, 1, 1), generator=gen) * math.sqrt(tau2)
        eps = torch.randn(x.shape, generator=gen)
        noisy = (1 - u) * x + u * eps
        posterior_mean = tau2 * (1 - u) / (tau2 * (1 - u) ** 2 + u ** 2) * noisy
        prediction = (noisy - posterior_mean) / u
        b = batch(size=len(x), shape=(1, 1, 1))
        b.latents = x
        LOSS(process(s), prediction, eps, noisy, torch.full((len(x),), 1000 * u), b)
        observed = np.mean([error for _, error in s.pending])
        expected = 1 / (1 / tau2 + ((1 - u) / u) ** 2)
        self.assertAlmostEqual(observed / expected, 1, delta=.025)


class InfoNoiseSamplerTest(InfoNoiseTestCase):
    def test_sampling_mass_matches_draws_in_warmup_adaptive_and_fallback(self):
        options = [
            {'warmup_type': kind, **support}
            for kind in ('uniform', 'log_uniform', 'sigmoid')
            for support in ({}, {'min_u': .3, 'max_u': .6},
                            {'min_u': .8, 'max_u': .95})
        ] + [{'warmup_type': 'sigmoid', 'warmup_shift': shift, 'warmup_scale': scale}
             for shift, scale in ((2., .7), (-2., 1.5), (8., 1.), (-8., 1.))]
        samplers = [InfoNoiseSampler(option, seed=17) for option in options]
        samplers.append(ready_sampler())
        fallback = ready_sampler()
        fallback.optimizer_steps += fallback.config.max_stale_steps
        fallback.sample(1)
        self.assertIsNone(fallback.cdf)
        samplers.append(fallback)
        for sampler in samplers:
            with self.subTest(config=sampler.config, reason=sampler.reason):
                before = sampler.state_dict()
                mass = sampler.sampling_mass()
                self.assertEqual(sampler.state_dict(), before)
                self.assertEqual(mass, sampler.diagnostics()['mass_u_ranges'])
                self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1 for value in mass))
                self.assertAlmostEqual(sum(mass), 1.)
                draws = sampler.sample(80000).double().numpy()
                empirical = [np.mean(draws < .25), np.mean((draws >= .25) & (draws <= .75)),
                             np.mean(draws > .75)]
                np.testing.assert_allclose(mass, empirical, atol=.006, rtol=0)
                resumed = InfoNoiseSampler(before['config'])
                resumed.load_state_dict(before)
                self.assertEqual(resumed.sampling_mass(), mass)

    def test_warmup_mass_has_analytic_values_before_first_draw(self):
        uniform = InfoNoiseSampler({'warmup_type': 'uniform'})
        low = (.25 - .001) / .998
        np.testing.assert_allclose(uniform.sampling_mass(), [low, 1 - 2 * low, low], atol=1e-14)
        log_uniform = InfoNoiseSampler({'warmup_type': 'log_uniform'})
        middle = math.log(9) / math.log(999 ** 2)
        np.testing.assert_allclose(log_uniform.sampling_mass(),
                                   [(1 - middle) / 2, middle, (1 - middle) / 2], atol=1e-14)

    def test_empirical_bin_probabilities(self):
        s = ready_sampler()
        self.assertEqual(s.cdf[0], 0)
        self.assertEqual(s.cdf[-1], 1)
        self.assertTrue((np.diff(s.cdf) >= 0).all())
        draws = s.sample(120000).double().numpy()
        counts = np.histogram(np.log(draws / (1 - draws)), bins=s.edges)[0] / len(draws)
        np.testing.assert_allclose(counts, np.diff(s.cdf), atol=.003, rtol=0)

    def test_flat_sections_and_endpoints(self):
        s = ready_sampler(bins=4)
        s.cdf = np.array([0, 0, .5, .5, 1.0])
        u = s.sample(10000).double().numpy()
        bins = np.searchsorted(s.edges[1:-1], np.log(u / (1 - u)))
        self.assertTrue(set(bins) <= {1, 3})
        self.assertGreater(u.min(), s.config.min_u)
        self.assertLess(u.max(), s.config.max_u)
        for warmup_type in ('sigmoid', 'uniform', 'log_uniform'):
            s = InfoNoiseSampler({'warmup_type': warmup_type}, seed=99)
            draws = s.sample(10000)
            self.assertTrue(torch.all((draws > s.config.min_u) & (draws < s.config.max_u)))
            # No interior [150,850] truncation.
            if warmup_type != 'sigmoid':
                self.assertTrue((draws < .15).any() and (draws > .85).any())

    def test_private_rng_does_not_change_global_rng(self):
        torch.manual_seed(51)
        before = torch.get_rng_state().clone()
        s = InfoNoiseSampler()
        s.sample(100)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_warmup_refresh_and_microbatch_accumulation(self):
        for accumulation in (1, 2, 4):
            s = InfoNoiseSampler({'bins': 4, 'min_count': 1, 'warmup_steps': 3,
                                  'refresh_every': 2}, seed=5)
            u = torch.sigmoid(torch.tensor(s.centers))
            for step in range(1, 8):
                for _ in range(accumulation):
                    s.record(u, torch.ones(4))
                    self.assertEqual(s.optimizer_steps, step - 1)
                attempted = s.on_optimizer_step()
                self.assertEqual(attempted, step in (3, 5, 7))
                self.assertEqual(s.optimizer_steps, step)
            self.assertEqual(s.refresh_attempts, 3)
            self.assertEqual(sum(s.total_counts), 7 * accumulation * 4)

    def test_ema_beta_weights_new_estimate(self):
        s = ready_sampler(beta=.9, buffer_size=1)
        np.testing.assert_allclose(s.mse_ema, (1 / (1 + np.exp(-s.centers))) ** 2)
        s.record(torch.sigmoid(torch.tensor(s.centers)), torch.full((16,), 3.0))
        s.on_optimizer_step()
        np.testing.assert_allclose(s.mse_ema, (.1 + .9 * 3) * (1 / (1 + np.exp(-s.centers))) ** 2)

    def test_sparse_bins_never_become_zero_errors(self):
        s = InfoNoiseSampler({'bins': 8, 'min_count': 2, 'warmup_steps': 1,
                              'refresh_every': 1, 'min_bin_fraction': .5})
        times = torch.sigmoid(torch.tensor(s.centers[[0, 2, 4, 6]]))
        s.record(times, torch.ones(4))
        s.on_optimizer_step()
        self.assertIsNone(s.cdf)
        self.assertIn('insufficient', s.reason)
        s.record(times, torch.ones(4))
        s.on_optimizer_step()
        self.assertIsNotNone(s.cdf)
        self.assertTrue((np.diff(s.cdf) > 0).all())

    def test_nonfinite_and_zero_profiles_fall_back(self):
        s = InfoNoiseSampler({'bins': 4, 'min_count': 1, 'warmup_steps': 1})
        s.record([.2, .3, .4, .5, float('nan')], [1, float('inf'), -2, 0, 1])
        self.assertEqual(s.invalid_observations, 3)
        self.assertEqual(len(s.pending), 2)
        zero = ready_sampler()
        zero.fifo = [type(buf)([(zero.optimizer_steps, 0.0)], maxlen=256) for buf in zero.fifo]
        zero.refresh()
        self.assertIn('zero', zero.reason)
        self.assertIsNotNone(zero.cdf) # Recent previous CDF is retained.
        zero.optimizer_steps += zero.config.max_stale_steps
        zero.refresh()
        self.assertIsNone(zero.cdf)
        self.assertIn('fallback', zero.reason)

    def test_stale_distribution_expires_between_refreshes(self):
        s = ready_sampler(max_stale_steps=3, refresh_every=100)
        self.assertIsNotNone(s.cdf)
        s.optimizer_steps += 3
        draws = s.sample(7)
        self.assertIsNone(s.cdf)
        self.assertIn('stale distribution', s.reason)
        self.assertTrue(torch.isfinite(draws).all())

    def test_refresh_cannot_make_old_observations_fresh(self):
        s = ready_sampler(max_stale_steps=3)
        for _ in range(3):
            s.on_optimizer_step()
        s.sample(1)
        self.assertIsNone(s.cdf)

    def test_expired_ema_cannot_dominate_new_observations(self):
        sampler = ready_sampler(beta=.9, max_stale_steps=3)
        sampler.mse_ema[5] = 1e8
        sampler.optimizer_steps += 4
        u = 1 / (1 + np.exp(-sampler.centers))
        sampler.record(u, np.ones_like(u))
        sampler.on_optimizer_step()
        np.testing.assert_allclose(sampler.mse_ema, u ** 2)
        np.testing.assert_allclose(sampler.cdf, ready_sampler().cdf, rtol=0, atol=1e-14)

    def test_extreme_sigmoid_warmup_matches_truncated_normal(self):
        from scipy.stats import truncnorm
        for shift in (-14., 0., 14.):
            s = InfoNoiseSampler({'warmup_type': 'sigmoid', 'warmup_shift': shift}, seed=1)
            u = s.sample(100000).double().numpy()
            self.assertTrue(np.isfinite(u).all())
            self.assertTrue(((u > 0) & (u < 1)).all())
            z = np.log(u / (1 - u))
            cdf = truncnorm.cdf(z, s.edges[0] + shift, s.edges[-1] + shift, loc=-shift)
            empirical = np.arange(1, len(u) + 1) / len(u)
            self.assertLess(np.max(np.abs(np.sort(cdf) - empirical)), .006)

    def test_exact_endpoints_are_never_observations(self):
        s = InfoNoiseSampler({'max_u': .9999999})
        s.record([0., 1., float('nan'), float('inf')], [1.] * 4)
        self.assertEqual(s.invalid_observations, 4)
        self.assertFalse(s.pending)

    def test_out_of_range_counts_are_rejected_on_resume(self):
        s = ready_sampler()
        original = s.state_dict()
        for bad_value in (1e30, -1, 1.5, True):
            state = copy.deepcopy(original)
            state['total_counts'] = [bad_value] * s.config.bins
            with self.subTest(value=bad_value), self.assertRaises(ValueError):
                s.load_state_dict(state)
            self.assertEqual(s.state_dict(), original)

    def test_invalid_normalizer_does_not_replace_cdf(self):
        s = ready_sampler()
        before = s.cdf.copy()
        with mock.patch('toolkit.samplers.infonoise.np.diff', return_value=np.zeros(s.config.bins)):
            s.refresh()
        np.testing.assert_array_equal(s.cdf, before)
        self.assertIn('normalizer', s.reason)

    def test_gate_pivot_dynamic_and_fixed(self):
        s = ready_sampler(gate_pivot=0)
        expected_raw = (1 / (1 + np.exp(-s.centers))) ** 2 * np.exp(-3 * s.centers)
        threshold_crossing = np.flatnonzero(expected_raw >= expected_raw.max() * .002)[-1]
        self.assertAlmostEqual(s.pivot, math.exp(s.centers[threshold_crossing]))
        self.assertEqual(ready_sampler().pivot, .15)

    def test_automatic_pivot_uses_sigma_density_not_log_density(self):
        for size in (1, 4):
            with self.subTest(batch=size):
                sampler = InfoNoiseSampler({'bins': 32, 'min_count': 1, 'warmup_steps': 32 // size,
                                            'refresh_every': 1, 'beta': 1., 'exploration': False})
                self.assertEqual(sampler.config.gate_pivot, 0)
                times = torch.sigmoid(torch.tensor(sampler.centers, dtype=torch.float32))
                # Constant clean-data error gives r_sigma = sigma^-3.
                for start in range(0, len(times), size):
                    chunk = times[start:start + size]
                    sampler.record(chunk, chunk.double().reciprocal().square())
                    sampler.on_optimizer_step()
                normalized_sigma = np.exp(-3 * (sampler.centers - sampler.centers[0]))
                selected = np.flatnonzero(normalized_sigma >= .002)[-1]
                normalized_log = np.exp(-2 * (sampler.centers - sampler.centers[0]))
                wrong = np.flatnonzero(normalized_log >= .002)[-1]
                self.assertNotEqual(selected, wrong)
                self.assertAlmostEqual(sampler.pivot, math.exp(sampler.centers[selected]))
                self.assertTrue(np.all(normalized_sigma[selected + 1:] < .002))

    def test_onset_uses_first_crossing_from_high_noise(self):
        sampler = InfoNoiseSampler({'bins': 8, 'min_count': 1, 'warmup_steps': 1,
                                    'refresh_every': 1, 'beta': 1., 'exploration': False})
        times = 1 / (1 + np.exp(-sampler.centers))
        # Include a second interior peak; the entire high-noise tail must be quiet.
        raw_sigma = np.array([1., .1, .01, .001, .1, .003, .001, .0001])
        velocity_mse = raw_sigma * sampler.relative_sigmas ** 3 / times ** 2
        sampler.record(times, velocity_mse)
        sampler.on_optimizer_step()
        self.assertEqual(sampler.pivot, sampler.relative_sigmas[5])
        # When the profile reaches the support edge, do not invent an outer crossing.
        sampler.record(times, sampler.relative_sigmas ** 3 / times ** 2)
        sampler.fifo = [type(buf)([buf[-1]], maxlen=256) for buf in sampler.fifo]
        sampler.on_optimizer_step()
        self.assertEqual(sampler.pivot, sampler.relative_sigmas[-1])

    def test_initial_error_placeholder_is_replaced_by_first_bin_mean(self):
        sampler = InfoNoiseSampler({'bins': 2, 'min_count': 1, 'warmup_steps': 1,
                                    'refresh_every': 1, 'beta': .1})
        np.testing.assert_array_equal(sampler.mse_ema, np.ones(2))
        times = torch.tensor([.1, .9], dtype=torch.float32)
        sampler.record(times, torch.tensor([2., 3.]))
        sampler.on_optimizer_step()
        np.testing.assert_allclose(sampler.mse_ema, times.double().numpy() ** 2 * [2, 3])

    def test_skipped_step_and_oom_discard_pending(self):
        s = ready_sampler()
        before = s.state_dict()
        s.record([.4], [99])
        s.on_optimizer_step(skipped=True)
        self.assertEqual(before, s.state_dict())
        s.record([.4], [99])
        s.discard_pending()
        self.assertEqual(before, s.state_dict())

    def test_resume_is_exact_including_pending_and_rng(self):
        for active in (False, True):
            s = ready_sampler() if active else InfoNoiseSampler(seed=77)
            s.sample(37)
            s.record([.3, .4], [1.2, 2.5])
            state = json.loads(json.dumps(s.state_dict()))
            resumed = InfoNoiseSampler(state['config'], seed=999)
            resumed.load_state_dict(state)
            for _ in range(3):
                torch.testing.assert_close(s.sample(73), resumed.sample(73), rtol=0, atol=0)
                s.on_optimizer_step()
                resumed.on_optimizer_step()
                self.assertEqual(s.state_dict(), resumed.state_dict())

    def test_previous_sampler_versions_are_rejected_atomically(self):
        for active in (False, True):
            sampler = ready_sampler() if active else InfoNoiseSampler(seed=7)
            baseline = sampler.state_dict()
            for version in (1, 2, 3):
                with self.subTest(active=active, version=version):
                    old = copy.deepcopy(baseline)
                    old['version'] = version
                    with self.assertRaisesRegex(ValueError, 'clean-data MSE'):
                        sampler.load_state_dict(old)
                    self.assertEqual(sampler.state_dict(), baseline)

    def test_corrupt_or_incompatible_resume_is_atomic(self):
        s = ready_sampler()
        baseline = s.state_dict()
        for key, value in [('version', 999), ('context', {'arch': 'other'}),
                           ('mse_ema', [-1] * 16), ('cdf', [0] * 17),
                           ('fifo', []), ('rng', [1, 2])]:
            with self.subTest(key=key):
                bad = copy.deepcopy(baseline)
                bad[key] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    s.load_state_dict(bad)
                self.assertEqual(baseline, s.state_dict())
        incompatible = InfoNoiseSampler({'bins': 16, 'gate_pivot': .3})
        with self.assertRaisesRegex(ValueError, 'configuration'):
            incompatible.load_state_dict(baseline)

    def test_bad_inputs_fail_without_observations(self):
        for options in ([], {'min_u': 1e-100}, {'max_u': 1-1e-12}, {'allow_legacy_resume': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                InfoNoiseConfig.from_dict(options)
        with self.assertRaisesRegex(ValueError, 'negligible'):
            InfoNoiseSampler({'warmup_type': 'sigmoid', 'warmup_shift': 1000})
        sampler = InfoNoiseSampler()
        state = sampler.state_dict()
        for count in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                sampler.sample(count)
        for times, errors, mask in (([.3], [1, 2], None), ([.3], [1], [True, False])):
            with self.assertRaises(ValueError):
                sampler.record(times, errors, mask)
        tensor = torch.zeros(2, 1, 2, 2)
        for prediction, times, flags in ((tensor[0], [.3], [False]),
                                          (tensor, [.3], [False, False]),
                                          (tensor, [.3, .4], [False])):
            with self.assertRaises(ValueError):
                b = batch(size=2, shape=(1, 2, 2), regs=flags)
                LOSS(process(sampler), prediction, tensor, tensor, torch.tensor(times), b)
        self.assertEqual(sampler.state_dict(), state)
        scheduler = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=500)
        with self.assertRaisesRegex(ValueError, '1000'):
            sampler.set_train_timesteps(scheduler, 2, 'cpu')
        self.assertEqual(sampler.state_dict(), state)

    def test_corrupt_checkpoint_fields_rejected_atomically(self):
        sampler = ready_sampler()
        original = sampler.state_dict()
        bad_states = [None]
        for key, value in (('optimizer_steps', -1), ('optimizer_steps', True),
                           ('last_refresh', 1000), ('profile_step', -1),
                           ('pivot', 0), ('pending', [(16, 1)]), ('ema_steps', [1000]*16),
                           ('ema_steps', [-1]*16)):
            state = copy.deepcopy(original)
            state[key] = value
            bad_states.append(state)
        for buffer in ([(0, 1)] * 257, [(-1, 1)], [(1, float('nan'))]):
            state = copy.deepcopy(original)
            state['fifo'][0] = buffer
            bad_states.append(state)
        for state in bad_states:
            with self.assertRaises(ValueError):
                sampler.load_state_dict(state)
            self.assertEqual(sampler.state_dict(), original)
        state = copy.deepcopy(original)
        state['total_counts'] = [2**53 + 1] * sampler.config.bins
        sampler.load_state_dict(state)
        self.assertTrue(np.all(sampler.total_counts == 2**53 + 1))

    def test_online_gaussian_profile_tracks_changed_risk(self):
        for size in (1, 4):
            with self.subTest(batch=size):
                sampler = InfoNoiseSampler({'warmup_type': 'log_uniform', 'bins': 32,
                                            'warmup_steps': 100, 'refresh_every': 20,
                                            'max_stale_steps': 500, 'gate_pivot': .15}, seed=77)
                active = 0
                errors = []
                for step in range(2000):
                    u = sampler.sample(size).double().numpy()
                    variance = .5 if step < 1000 else 3.
                    risk = variance / ((1-u)**2 * variance + u**2)
                    sampler.record(u, risk)
                    sampler.on_optimizer_step()
                    if sampler.cdf is not None:
                        active += 1
                        if step > 1500 and step % 20 == 19:
                            v = 1 / (1 + np.exp(-sampler.centers))
                            sigma = v / (1-v)
                            target = variance / ((1-v)**2 * variance + v**2) * v**4 * sigma / (sigma**3 + .15**3)
                            target /= target.sum()
                            errors.append(np.abs(np.diff(sampler.cdf) - target).sum() / 2)
                self.assertGreater(active, 1000)
                self.assertTrue(errors)
                self.assertLess(max(errors), .05)

    def test_default_grid_and_warmup_bin_coverage(self):
        sampler = InfoNoiseSampler(seed=17)
        self.assertEqual(sampler.config.bins, 32)
        self.assertTrue(sampler.config.exploration)
        u = sampler.sample(64000).numpy().astype(np.float64)
        counts = np.histogram(np.log(u / (1 - u)), bins=sampler.edges)[0]
        np.testing.assert_allclose(counts / counts.sum(), np.full(32, 1 / 32), atol=.002)
        # Uniform flow times are not uniform in the log-noise statistics grid.
        edges_u = 1 / (1 + np.exp(-sampler.edges))
        uniform_mass = np.diff(edges_u) / (sampler.config.max_u - sampler.config.min_u)
        self.assertGreater(uniform_mass[16], 100 * uniform_mass[0])

    def test_default_coverage_stays_fresh_with_small_batches(self):
        for size in (1, 4):
            for accumulation in (1, 4):
                with self.subTest(size=size, accumulation=accumulation):
                    sampler = InfoNoiseSampler(seed=17)
                    exploration_count = 0
                    for step in range(4000):
                        for _ in range(accumulation):
                            coverage, _ = sampler._coverage_bins(size)
                            exploration_count += len(coverage)
                            u = sampler.sample(size)
                            sampler.record(u, torch.ones(size, dtype=torch.float32))
                        sampler.on_optimizer_step()
                        if step >= 2500 and step % 100 == 0:
                            recent = [sum(sampler.optimizer_steps - t < 2000 for t, _ in buf)
                                      for buf in sampler.fifo]
                            self.assertGreaterEqual(min(recent), sampler.config.min_count)
                            self.assertIsNotNone(sampler.cdf)
                    self.assertTrue(np.all(sampler.total_counts > 0))
                    # Broad warmup and ordinary adaptive visits reduce the 8%/2% budget.
                    self.assertLessEqual(exploration_count / (4000 * size * accumulation),
                                         .08 / (size * accumulation))

    def test_collapsed_adaptive_bins_are_revisited(self):
        for size in (1, 4):
            with self.subTest(size=size):
                sampler = ready_sampler(bins=32, min_count=4, exploration=True,
                                        max_stale_steps=2000, refresh_every=100000)
                sampler.cdf = np.r_[0., np.ones(32)]  # All adaptive draws in bin zero.
                for _ in range(2400):
                    sampler.profile_step = sampler.optimizer_steps
                    u = sampler.sample(size)
                    sampler.record(u, torch.ones(size))
                    sampler.on_optimizer_step()
                self.assertGreaterEqual(min(len(buf) for buf in sampler.fifo), 5)
                for buf in sampler.fifo:
                    recent = [t for t, _ in buf if sampler.optimizer_steps - t < 2000]
                    self.assertGreaterEqual(len(recent), 4)

    def test_coverage_eligibility_pending_skips_and_invalid_errors(self):
        sampler = InfoNoiseSampler({'warmup_steps': 0}, seed=3)
        sampler.optimizer_steps = 500
        all_bins, _ = sampler._coverage_bins(32)
        self.assertEqual(len(all_bins), 32)
        self.assertEqual(sampler._coverage_bins(4, [False] * 4)[0], [])
        selected, _ = sampler._coverage_bins(4, [False, True, False, False])
        self.assertEqual(len(selected), 1)
        u = sampler.sample(4, eligible=[False, True, False, False])
        actual = np.searchsorted(sampler.edges[1:-1], torch.logit(u.double()).numpy())
        self.assertEqual(actual[1], selected[0])
        sampler.record(u, torch.ones(4), eligible=[False, True, False, False])
        self.assertNotIn(selected[0], sampler._coverage_bins(32)[0])
        sampler.on_optimizer_step(skipped=True)
        self.assertIn(selected[0], sampler._coverage_bins(32)[0])
        sampler.record([float(u[1])], [float('nan')])
        sampler.on_optimizer_step()
        self.assertIn(selected[0], sampler._coverage_bins(32)[0])
        sampler.record([float(u[1])], [1.])
        sampler.on_optimizer_step()
        self.assertNotIn(selected[0], sampler._coverage_bins(32)[0])
        self.assertEqual(sampler.invalid_observations, 1)

    def test_exploration_mass_matches_conditional_draws_and_resume(self):
        for size in (1, 4):
            for support in ({}, {'min_u': .3, 'max_u': .6}):
                sampler = InfoNoiseSampler(support, seed=19)
                sampler.optimizer_steps = 500
                eligible = [True] * size
                if size == 4:
                    eligible[0] = False
                before = sampler.state_dict()
                expected = sampler.sampling_mass(size, eligible)
                self.assertEqual(before, sampler.state_dict())
                resumed = InfoNoiseSampler(before['config'])
                resumed.load_state_dict(json.loads(json.dumps(before)))
                torch.testing.assert_close(sampler.sample(size, eligible=eligible),
                                           resumed.sample(size, eligible=eligible), rtol=0, atol=0)
                draws = torch.cat([sampler.sample(size, eligible=eligible) for _ in range(3000)]).numpy()
                empirical = [np.mean(draws < .25), np.mean((draws >= .25) & (draws < .75)),
                             np.mean(draws >= .75)]
                np.testing.assert_allclose(empirical, expected, atol=.015, rtol=0)
                self.assertAlmostEqual(sum(expected), 1.)
                with self.assertRaisesRegex(ValueError, 'eligibility'):
                    sampler.sample(size, eligible=[])
                with self.assertRaisesRegex(ValueError, 'eligibility'):
                    sampler.sampling_mass(size, eligible=[])

    def test_legacy_state_requires_original_coverage_behavior(self):
        sampler = InfoNoiseSampler({'bins': 64, 'exploration': False}, seed=7)
        old = sampler.state_dict()
        del old['config']['exploration']
        sampler.load_state_dict(old)
        modern = InfoNoiseSampler({'bins': 64}, seed=7)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            modern.load_state_dict(old)
        with self.assertRaisesRegex(ValueError, 'exploration'):
            InfoNoiseSampler({'exploration': 1})

    def test_default_small_batch_warmup_activates(self):
        for seed in (0, 1, 42):
            for batch_size in (1, 4):
                with self.subTest(seed=seed, batch_size=batch_size):
                    sampler = InfoNoiseSampler(seed=seed)
                    for step in range(sampler.config.warmup_steps):
                        u = sampler.sample(batch_size)
                        sampler.record(u, torch.ones(batch_size))
                        sampler.on_optimizer_step()
                        if step + 1 < sampler.config.warmup_steps:
                            self.assertIsNone(sampler.cdf)
                    self.assertIsNotNone(sampler.cdf)
                    self.assertEqual(sampler.reason, 'active')

    def test_settings_validation(self):
        for options in ({'bins': 1}, {'bins': 4.0}, {'min_count': 257}, {'beta': 0},
                        {'gate_power': 1}, {'gate_pivot': -1}, {'warmup_type': 'other'},
                        {'warmup_scale': 0}, {'max_u': 1}, {'min_u': 0},
                        {'beta': float('nan')}, {'unknown': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                InfoNoiseConfig.from_dict(options)


class InfoNoiseIntegrationTest(InfoNoiseTestCase):
    def test_final_draws_reach_full_batch_and_matching_sigmas(self):
        s = ready_sampler()
        expected = InfoNoiseSampler(s.state_dict()['config'])
        expected.load_state_dict(s.state_dict())
        expected_t = expected.sample(3) * 1000
        p, b = process(s, timestep_type='infonoise'), batch()
        noisy, noise, times, _, _ = PREPARE(p, b)
        torch.testing.assert_close(times, expected_t, rtol=0, atol=0)
        u = times.reshape(-1, 1, 1, 1) / 1000
        torch.testing.assert_close(noisy, (1 - u) * b.latents + u * noise)
        torch.testing.assert_close(p.sd.noise_scheduler.sigmas[:-1], times / 1000, rtol=0, atol=0)
        # Duplicate draw lookup must never use an ambiguous equality .item().
        duplicate = times[[0, 0, 2]]
        for actual in (BASE_SIGMAS(p, duplicate), p.sd.noise_scheduler.get_sigmas(duplicate, 4, torch.float32, 'cpu')):
            torch.testing.assert_close(actual, duplicate.reshape(-1, 1, 1, 1) / 1000, rtol=0, atol=0)

    def test_disabled_batch_samples_and_rng_match_upstream(self):
        torch.manual_seed(777)
        grid = torch.sort((1 - torch.sigmoid(torch.randn(1000))) * 1000, descending=True).values
        expected_times = grid[torch.randint(0, 999, (3,))]
        expected_noise = torch.randn_like(batch().latents)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(777)
        p, b = process(), batch()
        actual = PREPARE(p, b)
        torch.testing.assert_close(actual[2], expected_times, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected_noise, rtol=0, atol=0)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_loss_and_gradients_unchanged_and_reg_is_excluded(self):
        b = batch(regs=[False, True, False], multiplier=7)
        eps = torch.zeros_like(b.latents)
        pred = torch.ones_like(b.latents, requires_grad=True)
        times = torch.tensor([200., 400., 800.])
        p = process(InfoNoiseSampler())
        observed_loss = LOSS(p, pred, eps, b.latents, times, b)
        expected_loss = 7 * (pred + b.latents).square().mean()
        torch.testing.assert_close(observed_loss, expected_loss)
        g1 = torch.autograd.grad(observed_loss, pred)[0]
        g2 = torch.autograd.grad(expected_loss, pred)[0]
        torch.testing.assert_close(g1, g2)
        raw = (pred.detach() + b.latents).square().flatten(1).mean(1)
        self.assertEqual(len(p.infonoise_sampler.pending), 2)
        np.testing.assert_allclose([v for _, v in p.infonoise_sampler.pending],
                                   raw[[0, 2]].double() * (times[[0, 2]] / 1000).double().square())
        p.infonoise_sampler = None
        torch.testing.assert_close(LOSS(p, pred, eps, b.latents, times, b), expected_loss)

    def test_estimator_reuses_masked_loss_without_changing_gradients(self):
        for size in (1, 4):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for kind in ('none', 'ones', 'soft', 'hard'):
                    with self.subTest(batch=size, dtype=dtype, mask=kind):
                        regs = [False] * size
                        if size == 4:
                            regs[1] = True
                        b = batch(size=size, regs=regs)
                        b.latents = b.latents.to(dtype)
                        b.loss_multiplier_list = list(range(2, size + 2))
                        if kind != 'none':
                            mask = torch.zeros(size, 1, 6, 8)
                            for index in range(size):
                                mask[index, :, :, :2 * (index + 1)] = 1
                            if kind == 'ones':
                                mask.fill_(1)
                            elif kind == 'soft':
                                mask = .1 + .9 * mask
                            b.mask_tensor = mask
                        weights = expected_mask_weights(b.mask_tensor, b.latents)
                        prediction = torch.linspace(.2, 1.7, b.latents.numel(), dtype=dtype).reshape_as(b.latents).requires_grad_(True)
                        noise = torch.zeros_like(b.latents)
                        times = torch.linspace(200, 800, size)
                        sampler = InfoNoiseSampler()
                        trainer = process(sampler)
                        actual = LOSS(trainer, prediction, noise, b.latents, times, b, mask_multiplier=weights)
                        raw = (prediction.float() + b.latents.float()).square()
                        per_image = (raw * weights).flatten(1).mean(1)
                        expected = (per_image * torch.tensor(b.loss_multiplier_list)).mean()
                        torch.testing.assert_close(actual, expected)
                        gradient = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
                        torch.testing.assert_close(gradient, torch.autograd.grad(expected, prediction, retain_graph=True)[0])
                        trainer.infonoise_sampler = None
                        baseline = LOSS(trainer, prediction, noise, b.latents, times, b, mask_multiplier=weights)
                        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
                        torch.testing.assert_close(gradient, torch.autograd.grad(baseline, prediction)[0], rtol=0, atol=0)
                        selected = [i for i, reg in enumerate(regs) if not reg]
                        np.testing.assert_allclose([value for _, value in sampler.pending],
                                                   per_image.detach().double().numpy()[selected] *
                                                   (times / 1000).double().numpy()[selected] ** 2, rtol=0, atol=0)
                        if kind in ('soft', 'hard'):
                            self.assertFalse(torch.allclose(per_image, raw.flatten(1).mean(1)))
                        sampler.on_optimizer_step()
                        self.assertEqual(sum(sampler.total_counts), len(selected))

    def test_masks_leave_batch_noise_and_timestep_preparation_unchanged(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        for size in (1, 4):
            masked = process(InfoNoiseSampler(seed=12))
            plain = process(InfoNoiseSampler(seed=12))
            b = batch(size=size)
            b.mask_tensor = torch.linspace(.1, 1, size * 6 * 8).reshape(size, 1, 6, 8)
            original = b.mask_tensor.clone()
            rng = torch.get_rng_state()
            actual = PREPARE(masked, b)
            torch.set_rng_state(rng)
            expected = PREPARE(plain, batch(size=size))
            for index in (0, 1, 2):
                torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
            torch.testing.assert_close(b.mask_tensor, original, rtol=0, atol=0)
            self.assertEqual(masked.infonoise_sampler.state_dict(), plain.infonoise_sampler.state_dict())

    def test_mask_datasets_pass_configuration_validation(self):
        train = TrainConfig(timestep_type='infonoise', noise_scheduler='flowmatch')
        for arch in ('qwen_image', 'qwen_image_2'):
            model = ModelConfig(arch=arch, name_or_path='unused')
            for options in ({'mask_path': 'masks'}, {'alpha_mask': True}):
                dataset = DatasetConfig(folder_path='images', mask_min_value=.1, **options)
                validate_infonoise(train, model, [dataset])

    def test_variable_size_images_have_per_image_component_mean(self):
        for shape in ((1, 2, 3), (4, 7, 9)):
            b = batch(size=1, shape=shape)
            s = InfoNoiseSampler()
            LOSS(process(s), 3 - b.latents, torch.zeros_like(b.latents), b.latents,
                 torch.tensor([500.]), b)
            self.assertAlmostEqual(s.pending[0][1], 9 * .5 ** 2)

    def test_low_precision_statistics_use_actual_training_target(self):
        for dtype in (torch.float16, torch.bfloat16):
            b = batch(size=1)
            b.latents = b.latents.to(dtype)
            noise = torch.full_like(b.latents, .123)
            pred = torch.full_like(b.latents, .456)
            s = InfoNoiseSampler()
            LOSS(process(s), pred, noise, b.latents, torch.tensor([500.]), b)
            target = QWEN_TARGET(None, noise=noise, batch=b)
            expected = (pred.float() - target.float()).square().mean().item()
            self.assertEqual(s.pending[0][1], expected * .5 ** 2)

    def test_prior_targets_rejected(self):
        p, b = process(InfoNoiseSampler()), batch()
        args = (p, b.latents, b.latents, b.latents, torch.ones(3) * 500, b)
        with self.assertRaisesRegex(ValueError, 'teacher'):
            LOSS(*args, prior_pred=b.latents)
        self.assertFalse(p.infonoise_sampler.pending)

    def test_paired_guidance_batches_rejected_before_sampling(self):
        s = InfoNoiseSampler(seed=7)
        p = process(s, timestep_type='infonoise')
        state = s.state_dict()
        for attribute in ('unconditional_tensor', 'unconditional_latents'):
            b = batch()
            setattr(b, attribute, torch.ones(1))
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, 'paired guidance'):
                PREPARE(p, b)
            self.assertEqual(state, s.state_dict())

    def test_fixed_schedule_retains_priority_when_disabled_and_rejects_optin(self):
        p, b = process(), batch()
        p.sd.noise_scheduler.fixed_train_sigmas = [1., .7, .2]
        output = PREPARE(p, b)
        self.assertTrue(set(output[2].tolist()) <= {1000., 700., 200.})
        old_sigmas = p.sd.noise_scheduler.sigmas.clone()
        p.infonoise_sampler = InfoNoiseSampler()
        with self.assertRaisesRegex(ValueError, 'fixed_train_sigmas'):
            PREPARE(p, b)
        torch.testing.assert_close(p.sd.noise_scheduler.sigmas, old_sigmas)

    def test_real_checkpoint_save_load_pairs_state_with_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = process(ready_sampler())
            p.accelerator = NS(is_main_process=True)
            p.save_root, p.job, p.meta = temporary, NS(name='test'), OrderedDict()
            p.ema, p.embed_config, p.decorator, p.optimizer = None, None, None, None
            p.snr_gos = None
            p.named_lora, p.is_fine_tuning = False, False
            p.save_config, p.network_config = SaveConfig(), NetworkConfig(type='lora')
            def save_weights(path, *, metadata, **kw):
                save_file({'lora': torch.tensor([7.])}, path, metadata=metadata)
            p.network = NS(multiplier=1, save_weights=save_weights)
            p.update_training_metadata = lambda: p.meta.update(training_info={'step': 23, 'epoch': 2})
            p.clean_up_saves = lambda: None
            p.post_save_hook = lambda _: None
            p.step_num = 23
            SAVE(p, step=23)
            checkpoint = str(Path(temporary) / 'test_000000023.safetensors')
            saved = parse_metadata(checkpoint)
            self.assertEqual(saved['infonoise'], json.loads(json.dumps(p.infonoise_sampler.state_dict())))
            resumed = process(InfoNoiseSampler(saved['infonoise']['config']))
            resumed.accelerator = NS(is_main_process=True)
            resumed.network_config = NetworkConfig(type='lora')
            LOAD_META(resumed, checkpoint)
            self.assertEqual(resumed.step_num, 23)
            torch.testing.assert_close(p.infonoise_sampler.sample(40), resumed.infonoise_sampler.sample(40), rtol=0, atol=0)
            legacy = str(Path(temporary) / 'legacy.safetensors')
            save_file({'lora': torch.tensor([7.])}, legacy, metadata={'training_info': json.dumps({'step': 23})})
            with self.assertRaisesRegex(ValueError, 'no InfoNoise state'):
                LOAD_META(resumed, legacy)
            resumed.infonoise_sampler = InfoNoiseSampler({'allow_legacy_resume': True})
            LOAD_META(resumed, legacy)
            self.assertEqual(resumed.infonoise_sampler.optimizer_steps, 0)

    def test_qwen_yaml_real_configuration_loading_path(self):
        example = QWEN_YAML
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / 'qwen.yaml'
            config_path.write_text(example)
            raw = get_config(str(config_path))
        conf = raw['config']['process'][0]
        train, model, save = TrainConfig(**conf['train']), ModelConfig(**conf['model']), SaveConfig(**conf['save'])
        datasets = [DatasetConfig(**d) for d in preprocess_dataset_raw_config(conf['datasets'])]
        network, sample = NetworkConfig(**conf['network']), SampleConfig(**conf['sample'])
        validate_configs(train, model, save, datasets)
        self.assertEqual(model.arch, 'qwen_image_2')
        self.assertEqual(model.qtype, 'convrot8')
        self.assertEqual(model.qtype_te, 'convrot8')
        self.assertEqual(model.model_kwargs, {'turbo': False, 'rgba': False})
        self.assertEqual(network.linear, 16)
        self.assertEqual(sample.sampler, 'flowmatch')
        self.assertEqual(train.noise_scheduler, 'flowmatch')
        self.assertEqual(InfoNoiseConfig.from_dict(train.infonoise).warmup_steps, 500)
        fake = SDTrainer.__new__(SDTrainer)
        fake.accelerator, fake.network_config = NS(num_processes=1), network
        fake.embed_config = fake.adapter_config = fake.decorator_config = None
        fake.guidance_config = fake.custom_pipeline = None
        fake.do_lorm = False
        validate_process(fake)
        fake.accelerator.num_processes = 2
        with self.assertRaisesRegex(ValueError, 'single-process'):
            validate_process(fake)

    def test_update_in_backward_optimizers_rejected_early(self):
        model = ModelConfig(arch='qwen_image_2', name_or_path='unused')
        for name in ('automagic2', 'automagic3', 'automagic4', 'automagicexperiment', 'adamconvrot'):
            for fused in (True, False):
                train = TrainConfig(timestep_type='infonoise', noise_scheduler='flowmatch',
                                    optimizer=name, optimizer_params={'fused': fused}, gradient_accumulation=4)
                with self.subTest(name=name, fused=fused):
                    if name == 'automagic2' or fused:
                        with self.assertRaisesRegex(ValueError, 'backward|fused'):
                            validate_infonoise(train, model, [])
                    else:
                        validate_infonoise(train, model, [])
            if name != 'adamconvrot':
                train = TrainConfig(timestep_type='infonoise', noise_scheduler='flowmatch', optimizer=name)
                with self.assertRaises(ValueError):
                    validate_infonoise(train, model, [])
        for name in ('adamw', 'automagic', 'adam8', 'adamw8', 'prodigy'):
            train = TrainConfig(timestep_type='infonoise', noise_scheduler='flowmatch',
                                optimizer=name, optimizer_params={'fused': True} if name == 'adamw' else {})
            validate_infonoise(train, model, [])
        validate_infonoise(TrainConfig(optimizer='automagic3'), model, [])

    def test_models_and_training_methods_are_not_whitelisted(self):
        train = TrainConfig(timestep_type='infonoise', noise_scheduler='flowmatch')
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.accelerator = NS(num_processes=1)
        trainer.guidance_config = None
        for arch in ('flux', 'sd3', 'flux2', 'zimage', 'unseen_flow_extension'):
            validate_infonoise(train, ModelConfig(arch=arch, name_or_path='unused'), [])
        for kind in (None, 'lora', 'lokr', 'locon', 'lorm', 'dora', 'fullrank'):
            with self.subTest(network=kind):
                trainer.network_config = None if kind is None else NetworkConfig(type=kind, lorm={})
                validate_process(trainer)

    def test_flow_capabilities_checked_without_model_names(self):
        class DerivedScheduler(CustomFlowMatchEulerDiscreteScheduler):
            pass
        def model():
            return NS(noise_scheduler=DerivedScheduler(), is_flow_matching=True,
                      x0_pred=False, prediction_type='epsilon')
        validate_model(model())
        for key, value in (('is_flow_matching', False), ('x0_pred', True),
                           ('prediction_type', 'v_prediction'), ('noise_scheduler', object())):
            bad = model()
            setattr(bad, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'flow velocity'):
                validate_model(bad)
        bad = model()
        bad.noise_scheduler.fixed_train_sigmas = [1., .5]
        with self.assertRaisesRegex(ValueError, 'fixed sigmas'):
            validate_model(bad)

    def test_conditioning_keeps_adaptive_draws_unchanged(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        for size in (1, 4):
            conditioned = process(InfoNoiseSampler(seed=12))
            plain = process(InfoNoiseSampler(seed=12))
            b = batch(size=size)
            b.control_tensor = torch.ones(size, 3, 6, 8)
            b.control_tensor_list = [b.control_tensor]
            rng = torch.get_rng_state()
            actual = PREPARE(conditioned, b)
            torch.set_rng_state(rng)
            expected = PREPARE(plain, batch(size=size))
            for index in (0, 1, 2):
                torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)

    def test_video_error_reduces_per_sample(self):
        for size in (1, 4):
            b = batch(size=size, shape=(2, 3, 4, 5))
            prediction = torch.ones_like(b.latents, requires_grad=True)
            noise = torch.zeros_like(b.latents)
            sampler = InfoNoiseSampler()
            loss = LOSS(process(sampler), prediction, noise, b.latents,
                        torch.full((size,), 500.), b, mask_multiplier=torch.ones(size, 1, 4, 5))
            expected = (prediction + b.latents).square()
            torch.testing.assert_close(loss, expected.mean())
            torch.testing.assert_close(torch.autograd.grad(loss, prediction)[0],
                                       2 * (prediction + b.latents) / prediction.numel())
            np.testing.assert_allclose([value for _, value in sampler.pending],
                                       expected.detach().flatten(1).mean(1).double().numpy() * .5 ** 2, rtol=0, atol=0)

    def test_flow_model_noise_and_target_contracts(self):
        from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
        from extensions_built_in.diffusion_models.flux2.flux2_model import Flux2Model
        for cls in (ZImageModel, Flux2Model):
            model = cls(device='cpu', model_config=ModelConfig(arch=cls.arch, name_or_path='unused'),
                        dtype='fp32', noise_scheduler=cls.get_train_scheduler())
            validate_model(model)
            for size in (1, 4):
                with self.subTest(model=cls.arch, batch=size):
                    sampler = InfoNoiseSampler(seed=12)
                    times = sampler.set_train_timesteps(model.noise_scheduler, size, 'cpu')
                    b = batch(size=size)
                    noise = model.get_latent_noise_from_latents(b.latents)
                    noisy = model.add_noise(b.latents, noise, times)
                    u = (times / 1000).reshape(-1, 1, 1, 1)
                    torch.testing.assert_close(noisy, (1-u) * b.latents + u * noise)
                    target = model.get_loss_target(noise=noise, batch=b, timesteps=times)
                    torch.testing.assert_close(target, noise - b.latents)
                    trainer = process(sampler)
                    trainer.sd.get_loss_target = model.get_loss_target
                    trainer.sd.scale_loss = model.scale_loss
                    prediction = .3 * noisy
                    LOSS(trainer, prediction, noise, noisy, times, b)
                    expected = (prediction - target).square().flatten(1).mean(1)
                    np.testing.assert_allclose([value for _, value in sampler.pending],
                                               expected.double().numpy() * (times / 1000).double().numpy() ** 2,
                                               rtol=0, atol=0)

    def test_actual_x0_target_rejected_without_an_architecture_check(self):
        from extensions_built_in.diffusion_models.prx_pixel_t2i.prx_pixel_t2i import PRXPixelT2IModel
        model = PRXPixelT2IModel(device='cpu', model_config=ModelConfig(arch='prx_pixel', name_or_path='unused'),
                                dtype='fp32', noise_scheduler=PRXPixelT2IModel.get_train_scheduler())
        trainer = process(InfoNoiseSampler())
        trainer.sd.get_loss_target = model.get_loss_target
        b = batch(size=4)
        noise = torch.ones_like(b.latents)
        with self.assertRaisesRegex(ValueError, 'flow velocity MSE target'):
            LOSS(trainer, noise, noise, b.latents, torch.full((4,), 500.), b)
        self.assertFalse(trainer.infonoise_sampler.pending)
        trainer.sd.get_loss_target = lambda **kwargs: kwargs['batch'].latents - kwargs['noise']
        LOSS(trainer, noise, noise, b.latents, torch.full((4,), 500.), b)
        expected = (noise - (b.latents - noise)).square().flatten(1).mean(1)
        np.testing.assert_allclose([value for _, value in trainer.infonoise_sampler.pending], expected.numpy() * .5 ** 2)

    def test_velocity_check_waits_for_a_finite_batch(self):
        trainer = process(InfoNoiseSampler())
        b = batch(size=1)
        b.latents.fill_(float('nan'))
        noise = torch.ones_like(b.latents)
        prediction = torch.zeros_like(noise)
        loss = LOSS(trainer, prediction, noise, b.latents, torch.tensor([500.]), b)
        self.assertTrue(torch.isnan(loss))
        self.assertFalse(trainer._infonoise_target_checked)
        self.assertFalse(trainer.infonoise_sampler.pending)
        b.latents.zero_()
        loss = LOSS(trainer, prediction, noise, b.latents, torch.tensor([500.]), b)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(trainer._infonoise_target_checked)
        self.assertEqual(len(trainer.infonoise_sampler.pending), 1)

    def test_component_and_full_model_checkpoints_restore_sampler(self):
        from toolkit.embedding import Embedding
        class TinyEmbedding(Embedding):
            def __init__(self):
                self.sd = NS(is_xl=False)
                self.name, self.step = 'token', 3
                self.vector = torch.tensor([[.2, .4]])
            def _get_vec(self, index=0):
                return self.vector
            def _set_vec(self, value, index=0):
                self.vector = value
        class TinyControl(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([.5]))
            @property
            def device(self):
                return self.weight.device
            @property
            def dtype(self):
                return self.weight.dtype
            def save_pretrained(self, path, **kwargs):
                Path(path).mkdir()
                save_file(self.state_dict(), str(Path(path) / 'weights.safetensors'))
        for mode in ('embedding_safetensors', 'embedding_pt', 'control_net', 'full'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                p = process(ready_sampler())
                p.accelerator = NS(is_main_process=True)
                p.save_root, p.job, p.meta = temporary, NS(name='component'), OrderedDict(name='job')
                p.ema = p.decorator = p.optimizer = p.snr_gos = None
                p.network = p.network_config = p.embedding = p.adapter = p.embed_config = None
                p.named_lora, p.is_fine_tuning = False, mode == 'full'
                p.save_config = SaveConfig(dtype='float32')
                p.update_training_metadata = lambda: p.meta.update(training_info={'step': 3, 'epoch': 1})
                p.clean_up_saves = lambda: None
                p.post_save_hook = lambda _: None
                p.step_num = 3
                if mode.startswith('embedding'):
                    p.embedding = TinyEmbedding()
                    p.embed_config = NS(trigger='token', save_format='pt' if mode.endswith('_pt') else 'safetensors')
                elif mode == 'control_net':
                    p.adapter = TinyControl()
                    p.adapter_config = NS(train=True, type='control_net')
                else:
                    p.sd.refiner_unet = None
                    def save_model(path, metadata, dtype):
                        save_file({'weight': torch.tensor([.5])}, path, metadata=metadata)
                    p.sd.save = save_model
                SAVE(p, step=3)
                if mode.startswith('embedding'):
                    extension = 'pt' if mode.endswith('_pt') else 'safetensors'
                    path = str(Path(temporary) / f'token_000000003.{extension}')
                    loaded = TinyEmbedding()
                    loaded.load_embedding_from_file(path, 'cpu')
                    torch.testing.assert_close(loaded.vec, p.embedding.vec)
                    self.assertEqual(loaded.step, 3)
                    if extension == 'safetensors':
                        self.assertEqual(parse_metadata(path)['name'], 'token')
                elif mode == 'control_net':
                    path = str(Path(temporary) / 'component_000000003')
                    self.assertEqual(yaml.safe_load((Path(path) / 'aitk_meta.yaml').read_text())['infonoise'],
                                     json.loads(json.dumps(p.infonoise_sampler.state_dict())))
                else:
                    path = str(Path(temporary) / 'component_000000003.safetensors')
                resumed = process(InfoNoiseSampler(p.infonoise_sampler.state_dict()['config']))
                resumed.accelerator, resumed.network_config = p.accelerator, None
                LOAD_META(resumed, path)
                self.assertEqual(resumed.step_num, 3)
                self.assertEqual(resumed.infonoise_sampler.state_dict(), p.infonoise_sampler.state_dict())
                torch.testing.assert_close(resumed.infonoise_sampler.sample(4), p.infonoise_sampler.sample(4), rtol=0, atol=0)

    def test_actual_optimizer_guard(self):
        from toolkit.optimizers.automagic2 import Automagic2
        from toolkit.optimizers.automagic3 import Automagic3
        from toolkit.optimizers.automagicEXPERIMENT import AutomagicEXPERIMENT
        from toolkit.optimizers.adamconvrot import AdamConvRot

        for optimizer_class in (Automagic3, AutomagicEXPERIMENT, AdamConvRot):
            for fused in (True, False):
                with self.subTest(optimizer=optimizer_class.__name__, fused=fused):
                    parameter = torch.nn.Parameter(torch.tensor(.3))
                    optimizer = optimizer_class([parameter], fused=fused)
                    if fused:
                        with self.assertRaisesRegex(ValueError, 'fused'):
                            validate_optimizer(optimizer)
                    else:
                        validate_optimizer(optimizer)
        parameter = torch.nn.Parameter(torch.tensor(.3))
        with self.assertRaisesRegex(ValueError, 'backward'):
            validate_optimizer(Automagic2([parameter]))
        validate_optimizer(torch.optim.AdamW([torch.nn.Parameter(torch.tensor(.3))], fused=True))
        # A subclass must not bypass the base optimizer's backward-update guard.
        class CustomAutomagic(Automagic3):
            pass
        with self.assertRaisesRegex(ValueError, 'fused'):
            validate_optimizer(CustomAutomagic([torch.nn.Parameter(torch.tensor(.3))]))

    def test_unsupported_settings_fail_configuration_validation(self):
        model = ModelConfig(arch='qwen_image_2', name_or_path='unused')
        for key, value in [('loss_type', 'mae'), ('snr_gamma', 5), ('noise_offset', .1),
                           ('do_guidance_loss', True), ('content_or_style', 'style'),
                           ('first_timestep_chance', .1), ('linear_timesteps', True),
                           ('single_item_batching', True), ('t0_loss_target', True),
                           ('gradient_accumulation_steps', 3), ('gradient_accumulation_steps', -1)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_infonoise(TrainConfig(timestep_type='infonoise', noise_scheduler='flowmatch', **{key: value}), model, [])
        train = TrainConfig(timestep_type='infonoise', noise_scheduler='flowmatch')
        for key, value in [('unconditional_path', 'pair'), ('prior_reg', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_infonoise(train, model, [DatasetConfig(folder_path='unused', **{key: value})])
        for kwargs in ({'turbo': True}, {'rgba': True}):
            validate_infonoise(train, ModelConfig(arch='unseen_flow_extension', name_or_path='unused', model_kwargs=kwargs), [])
        validate_infonoise(train, model, [DatasetConfig(folder_path='unused', num_frames=8,
                                                       control_path='control', network_weight=.5)])
        with self.assertRaisesRegex(ValueError, 'VP v-prediction'):
            validate_infonoise(train, ModelConfig(arch='unused', name_or_path='unused', is_v_pred=True), [])
        wrong_process = NS(accelerator=NS(num_processes=1))
        with self.assertRaisesRegex(ValueError, 'SDTrainer'):
            validate_process(wrong_process)
        valid_process = SDTrainer.__new__(SDTrainer)
        valid_process.accelerator = NS(num_processes=1)
        valid_process.guidance_config = valid_process.custom_pipeline = None
        valid_process.do_lorm = False
        valid_process.network_config = NetworkConfig(type='lora')
        valid_process.embed_config = valid_process.adapter_config = valid_process.decorator_config = None
        for field, value in (('guidance_config', object()),):
            changed = copy.copy(valid_process)
            setattr(changed, field, value)
            with self.assertRaises(ValueError):
                validate_process(changed)
        model.arch = 'unseen_flow_extension'
        validate_infonoise(train, model, [])
        # Disabled configuration is not subjected to new restrictions.
        validate_infonoise(TrainConfig(noise_offset=.1), model, [])



class InfoNoiseTrainingTest(InfoNoiseTestCase):
    def trainer(self):
        base = process(InfoNoiseSampler({'warmup_steps': 100}, seed=7),
                       dtype='fp32', max_grad_norm=1e9)
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.__dict__.update(base.__dict__)
        trainer.timer = Timer()
        trainer.network = BlankNetwork()
        trainer.assistant_adapter = trainer.adapter_config = trainer.decorator = None
        trainer._clip_image_embeds_unconditional = None
        trainer.do_long_prompts = trainer.is_caching_text_embeddings = False
        trainer.do_prior_prediction = trainer.do_guided_loss = False
        trainer.sd.vae = NS(dtype=torch.float32)
        trainer.sd.vae_torch_dtype = trainer.sd.te_torch_dtype = torch.float32
        trainer.sd.text_encoder = torch.nn.Identity()
        trainer.sd.text_encoder.dtype = torch.float32
        trainer.sd.encode_control_in_text_embeddings = False
        trainer.sd.encode_prompt = lambda prompts, *args, **kwargs: torch.zeros(len(prompts), 1)
        trainer.sd.is_xl = False
        trainer.sd.condition_noisy_latents = lambda x, b: QwenImage2Model.condition_noisy_latents(trainer.sd, x, b)
        parameter = torch.nn.Parameter(torch.tensor(.3))
        forwards = []
        current_noise = []
        def noise(latents, *args, **kwargs):
            value = torch.randn_like(latents)
            current_noise[:] = [value]
            return value
        trainer.get_noise = noise
        def predict(**kwargs):
            x = kwargs['noisy_latents']
            b = kwargs['batch']
            target = current_noise[0] - b.latents
            forwards.append((x.detach().clone(), target.detach().clone(), kwargs['timesteps'].clone()))
            return parameter * x
        trainer.predict_noise = predict
        trainer.params = [parameter]
        trainer.accelerator = Accelerator(cpu=True)
        trainer.optimizer = trainer.accelerator.prepare(torch.optim.SGD([parameter], lr=.01))
        trainer.lr_scheduler = NS(step=lambda: None)
        trainer.ema = None
        trainer.steps_this_boundary = 0
        trainer.is_grad_accumulation_step = False
        return trainer, parameter, forwards

    def test_qwen_example_constructs_real_trainers_without_loading_models(self):
        from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(random.setstate, random.getstate())
        with tempfile.TemporaryDirectory() as temporary:
            for trainer_class in (SDTrainer, DiffusionTrainer):
                for mode, network_type in ((mode, kind) for mode in ('infonoise', 'sigmoid')
                                           for kind in (None, 'lora', 'lokr', 'dora', 'locon')):
                    raw = yaml.safe_load(QWEN_YAML)
                    conf = raw['config']['process'][0]
                    conf['training_folder'] = temporary
                    conf['sqlite_db_path'] = str(Path(temporary) / 'missing.db')
                    conf['train']['timestep_type'] = mode
                    if network_type is None:
                        conf.pop('network')
                    else:
                        conf['network']['type'] = network_type
                    job = NS(name=raw['config']['name'], meta=OrderedDict(), raw_config=raw,
                             training_folder=temporary, training_seed=42, log_dir=None)
                    with redirect_stdout(io.StringIO()):
                        trainer = trainer_class(0, job, conf)
                    self.assertEqual(trainer.infonoise_sampler is not None, mode == 'infonoise')
                    self.assertEqual(None if trainer.network_config is None else trainer.network_config.type, network_type)
                    self.assertIsNone(trainer.optimizer)
                    self.assertFalse(hasattr(trainer, 'sd'))
                    if mode == 'infonoise':
                        self.assertEqual(trainer.infonoise_sampler.optimizer_steps, 0)
                        self.assertEqual(trainer.infonoise_sampler.context['name_or_path'], 'Comfy-Org/Qwen-Image-2.1')
                        enabled_rng = torch.get_rng_state().clone()
                    else:
                        self.assertTrue(torch.equal(enabled_rng, torch.get_rng_state()))

    def test_flow_configs_construct_with_full_model_or_adapters(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(random.setstate, random.getstate())
        with tempfile.TemporaryDirectory() as temporary:
            for arch in ('flux', 'sd3', 'flux2', 'zimage'):
                for kind in (None, 'lora', 'lokr', 'dora'):
                    raw = yaml.safe_load(QWEN_YAML)
                    conf = raw['config']['process'][0]
                    conf['training_folder'] = temporary
                    conf['sqlite_db_path'] = str(Path(temporary) / 'missing.db')
                    conf['model']['arch'] = arch
                    if kind is None:
                        conf.pop('network')
                    else:
                        conf['network']['type'] = kind
                    job = NS(name=raw['config']['name'], meta=OrderedDict(), raw_config=raw,
                             training_folder=temporary, training_seed=42, log_dir=None)
                    with self.subTest(arch=arch, network=kind), redirect_stdout(io.StringIO()):
                        trainer = SDTrainer(0, job, conf)
                        self.assertIsNotNone(trainer.infonoise_sampler)
                        self.assertEqual(trainer.infonoise_sampler.context['arch'], arch)
                        self.assertEqual(trainer.infonoise_sampler.context['network']['type'], kind or 'full')
                        self.assertEqual(trainer.is_fine_tuning, kind is None)

    def test_masked_prodigy_lokr_config_constructs(self):
        from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(random.setstate, random.getstate())
        with tempfile.TemporaryDirectory() as temporary:
            for size in (1, 4):
                raw = yaml.safe_load(QWEN_YAML)
                conf = raw['config']['process'][0]
                conf['type'] = 'diffusion_trainer'
                conf['training_folder'] = temporary
                conf['sqlite_db_path'] = str(Path(temporary) / 'missing.db')
                conf['network'].update(type='lokr', lokr_full_rank=True, lokr_factor=4, conv=16, conv_alpha=16)
                conf['datasets'][0].update(mask_path='masks', mask_min_value=.1)
                conf['train'].update(batch_size=size, gradient_accumulation=1, optimizer='prodigyopt',
                                     optimizer_params={'weight_decay': .0001}, lr=1)
                job = NS(name=raw['config']['name'], meta=OrderedDict(), raw_config=raw,
                         training_folder=temporary, training_seed=42, log_dir=None)
                with redirect_stdout(io.StringIO()):
                    trainer = DiffusionTrainer(0, job, conf)
                self.assertIsNotNone(trainer.infonoise_sampler)
                self.assertEqual(trainer.train_config.batch_size, size)
                self.assertEqual(trainer.train_config.optimizer, 'prodigyopt')
                self.assertEqual(trainer.network_config.type, 'lokr')
                self.assertEqual(trainer.dataset_configs[0].mask_path, 'masks')
                self.assertEqual(trainer.dataset_configs[0].mask_min_value, .1)

    def test_resume_rejects_changed_adapter_family_or_lokr_settings(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(random.setstate, random.getstate())
        samplers = []
        networks = (
            {'type': 'lora'},
            {'type': 'lokr', 'lokr_factor': 2},
            {'type': 'lokr', 'lokr_factor': 2, 'lokr_full_rank': False, 'linear': 1},
            {'type': 'lokr', 'lokr_factor': 8},
            {'type': 'lokr', 'lokr_factor': 2, 'lokr_full_rank': False, 'linear': 2},
            {'type': 'lokr', 'lokr_factor': 2, 'lokr_full_rank': False, 'linear': 1, 'linear_alpha': 2},
        )
        with tempfile.TemporaryDirectory() as temporary:
            for options in networks:
                raw = yaml.safe_load(QWEN_YAML)
                conf = raw['config']['process'][0]
                conf['training_folder'] = temporary
                conf['sqlite_db_path'] = str(Path(temporary) / 'missing.db')
                conf['network'] = options
                job = NS(name=raw['config']['name'], meta=OrderedDict(), raw_config=raw,
                         training_folder=temporary, training_seed=42, log_dir=None)
                with redirect_stdout(io.StringIO()):
                    trainer = SDTrainer(0, job, conf)
                sampler = trainer.infonoise_sampler
                sampler.record(sampler.sample(4), torch.ones(4))
                sampler.on_optimizer_step()
                samplers.append(sampler)
        for index, sampler in enumerate(samplers):
            state = sampler.state_dict()
            for other_index, other in enumerate(samplers):
                with self.subTest(source=index, target=other_index):
                    before = other.state_dict()
                    if index == other_index:
                        other.load_state_dict(state)
                    else:
                        with self.assertRaisesRegex(ValueError, 'context'):
                            other.load_state_dict(state)
                    self.assertEqual(other.state_dict(), before)

    def test_graph_logs_sampling_mass_from_step_zero(self):
        from toolkit.logging_aitk import UILogger
        for size in (1, 4):
            for kind in ('uniform', 'log_uniform', 'sigmoid'):
                with self.subTest(batch_size=size, warmup=kind), tempfile.TemporaryDirectory() as temporary:
                    trainer, _, _ = self.trainer()
                    sampler = InfoNoiseSampler({'bins': 4, 'min_count': 1, 'warmup_steps': 2,
                                                'refresh_every': 1, 'warmup_type': kind}, seed=7)
                    trainer.infonoise_sampler = sampler
                    sampler.record(torch.sigmoid(torch.tensor(sampler.centers)), torch.ones(4))
                    path = str(Path(temporary) / 'loss_log.db')
                    logger = UILogger(path)
                    self.addCleanup(logger.finish)
                    trainer.logger = logger
                    expected_keys = {'infonoise/mass_low', 'infonoise/mass_middle', 'infonoise/mass_high'}
                    expected_rows = []
                    for step in range(4):
                        trainer.step_num = step
                        expected = dict(zip(('infonoise/mass_low', 'infonoise/mass_middle',
                                             'infonoise/mass_high'), sampler.sampling_mass(size)))
                        b = batch(size=size)
                        b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
                        b.get_network_weight_list = lambda: [1.] * size
                        output = io.StringIO()
                        with redirect_stdout(output):
                            trainer.hook_train_loop([b])
                        if step == 0:
                            self.assertEqual(output.getvalue(), '')
                        else:
                            self.assertRegex(output.getvalue(),
                                             r'^InfoNoise mass: low=[01]\.\d{3}, middle=[01]\.\d{3}, high=[01]\.\d{3}\n$')
                            printed = [float(item.split('=')[1]) for item in output.getvalue().split(': ')[1].split(', ')]
                            self.assertEqual(printed, [float(f'{value:.3f}') for value in expected.values()])
                        self.assertEqual(trainer.additional_logs, {})
                        self.assertAlmostEqual(sum(expected.values()), 1.)
                        self.assertEqual(logger._log_to_commit, expected)
                        logger.commit(step=step)
                        expected_rows.extend((step, key, value, None) for key, value in sorted(expected.items()))
                        if step == 1:
                            self.assertIsNotNone(sampler.cdf)
                    logger.finish()
                    with sqlite3.connect(path) as connection:
                        rows = connection.execute(
                            'SELECT step, key, value_real, value_text FROM metrics ORDER BY step, key').fetchall()
                        self.assertEqual(rows, expected_rows)
                        keys = connection.execute('SELECT key, first_seen_step, last_seen_step FROM metric_keys').fetchall()
                        self.assertEqual({key for key, _, _ in keys}, expected_keys)
                        self.assertTrue(all(first == 0 and last == 3 for _, first, last in keys))

    def test_mass_logging_averages_exploration_across_microbatches(self):
        from toolkit.logging_aitk import UILogger
        for sizes in ((1, 1, 1, 1), (4, 4, 4, 4), (1, 4)):
            with self.subTest(sizes=sizes), tempfile.TemporaryDirectory() as temporary:
                trainer, _, _ = self.trainer()
                sampler = InfoNoiseSampler({'warmup_steps': 0, 'refresh_every': 1}, seed=7)
                sampler.optimizer_steps = 500
                sampler.profile_step = 500
                sampler.cdf = np.r_[np.zeros(32), 1.]  # Adaptive mass is entirely high noise.
                sampler.record(torch.sigmoid(torch.tensor(sampler.centers)), torch.ones(32))
                sampler.on_optimizer_step()
                sampler.cdf = np.r_[np.zeros(32), 1.]
                # Only bin zero is overdue; first microbatch covers it, later ones don't.
                sampler.fifo[0].clear()
                trainer.infonoise_sampler = sampler
                path = str(Path(temporary) / 'loss_log.db')
                logger = UILogger(path)
                self.addCleanup(logger.finish)
                trainer.logger = logger
                expected = [1 / sum(sizes), 0., 1 - 1 / sum(sizes)]
                batches = []
                for size in sizes:
                    b = batch(size=size)
                    b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
                    b.get_network_weight_list = lambda size=size: [1.] * size
                    batches.append(b)
                output = io.StringIO()
                with redirect_stdout(output):
                    trainer.hook_train_loop(batches)
                keys = ('infonoise/mass_low', 'infonoise/mass_middle', 'infonoise/mass_high')
                np.testing.assert_allclose([logger._log_to_commit[key] for key in keys], expected, atol=1e-12)
                printed = [float(item.split('=')[1]) for item in output.getvalue().strip().split(': ')[1].split(', ')]
                self.assertEqual(printed, [float(f'{value:.3f}') for value in expected])
                logger.commit(step=0)
                logger.finish()
                with sqlite3.connect(path) as connection:
                    values = dict(connection.execute('SELECT key, value_real FROM metrics'))
                np.testing.assert_allclose([values[key] for key in keys], expected, atol=1e-12)

    def test_mass_logging_respects_cadence_and_main_process(self):
        for every in (None, 0, 1, 2):
            for main in (False, True):
                with self.subTest(log_every=every, main_process=main):
                    trainer, _, _ = self.trainer()
                    trainer.infonoise_sampler = ready_sampler()
                    trainer.logging_config.log_every = every
                    trainer.logger = mock.Mock()
                    with mock.patch.object(Accelerator, 'is_main_process', new_callable=mock.PropertyMock,
                                           return_value=main):
                        for step in range(4):
                            trainer.step_num = step
                            b = batch(size=1)
                            b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
                            b.get_network_weight_list = lambda: [1.]
                            with redirect_stdout(io.StringIO()):
                                trainer.hook_train_loop([b])
                    expected = (4 if every is None else 0 if every == 0 else 4 // every) if main else 0
                    self.assertEqual(trainer.logger.log.call_count, expected)

    def test_full_trainer_update_matches_analytical_gradient(self):
        for accumulation in (1, 2, 4):
            trainer, parameter, forwards = self.trainer()
            trainer.train_config.reg_weight = .25
            batches = []
            for index in range(accumulation):
                b = batch(size=2, regs=[index % 2 == 1] * 2)
                b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
                b.get_network_weight_list = lambda: [1., 1.]
                batches.append(b)
            oracle = InfoNoiseSampler(trainer.infonoise_sampler.state_dict()['config'])
            oracle.load_state_dict(trainer.infonoise_sampler.state_dict())
            expected_times = [oracle.sample(2) * 1000 for _ in batches]
            initial = parameter.detach().clone()
            result = trainer.hook_train_loop(batches)
            gradients, losses = [], []
            for index, (x, target, times) in enumerate(forwards):
                weight = .25 if index % 2 else 1.0
                gradients.append(weight * (2 * (initial * x - target) * x).mean())
                losses.append(weight * (initial * x - target).square().mean())
                torch.testing.assert_close(times, expected_times[index], rtol=0, atol=0)
            expected = initial - .01 * torch.stack(gradients).mean()
            torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(result['loss'], torch.stack(losses).mean().item(), rtol=1e-6, atol=1e-6)
            self.assertEqual(len(forwards), accumulation)
            self.assertEqual(trainer.infonoise_sampler.optimizer_steps, 1)
            self.assertEqual(sum(trainer.infonoise_sampler.total_counts), 2 * ((accumulation + 1) // 2))
            self.assertFalse(trainer.infonoise_sampler.pending)

    def test_video_training_update_matches_velocity_gradient(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        for seed in (0, 17, 42):
            for size in (1, 4):
                with self.subTest(seed=seed, batch_size=size):
                    torch.manual_seed(seed)
                    trainer, parameter, forwards = self.trainer()
                    b = batch(size=size, shape=(2, 3, 4, 5))
                    b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
                    b.get_network_weight_list = lambda: [1.] * size
                    initial = parameter.detach().clone()
                    noise_fn, noises = trainer.get_noise, []
                    def capture_noise(*args, **kwargs):
                        noise = noise_fn(*args, **kwargs)
                        noises.append(noise)
                        return noise
                    trainer.get_noise = capture_noise
                    output = io.StringIO()
                    with redirect_stdout(output):
                        result = trainer.hook_train_loop(b)
                        x, target, times = forwards[0]
                        baseline_parameter = initial.clone().requires_grad_(True)
                        # The trainer supplies a spatial mask tensor, even when every weight is one.
                        mask = x.new_ones((size, 1, 1, 1))
                        baseline_loss = LOSS(process(), baseline_parameter * x, noises[0], x, times, b,
                                             mask_multiplier=mask)
                    self.assertEqual(output.getvalue(), '')
                    self.assertEqual(result['loss'], baseline_loss.item())
                    baseline_gradient = torch.autograd.grad(baseline_loss, baseline_parameter)[0]
                    torch.testing.assert_close(parameter, initial - .01 * baseline_gradient)
                    self.assertEqual(x.dtype, torch.float32)
                    self.assertEqual(target.dtype, torch.float32)
                    self.assertEqual(parameter.dtype, torch.float32)
                    expected_loss = (initial * x - target).square().mean()
                    gradient = (2 * (initial * x - target) * x).mean()
                    torch.testing.assert_close(parameter, initial - .01 * gradient)
                    # Float32 reduction order can change the last few bits at larger loss values.
                    torch.testing.assert_close(result['loss'], expected_loss.item(), rtol=1e-6, atol=1e-6)
                    self.assertEqual(sum(trainer.infonoise_sampler.total_counts), size)

    def test_masked_trainer_update_and_statistics_match_existing_loss(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(23)
        for size in (1, 4):
            for accumulation in (1, 4):
                for masked_loss in (True, False):
                    with self.subTest(batch=size, accumulation=accumulation, masked_loss=masked_loss):
                        trainer, parameter, forwards = self.trainer()
                        trainer.sd.do_masked_loss = masked_loss
                        sampler = InfoNoiseSampler(trainer.infonoise_sampler.state_dict()['config'])
                        sampler.load_state_dict(trainer.infonoise_sampler.state_dict())
                        expected_times = [sampler.sample(size) * 1000 for _ in range(accumulation)]
                        batches = []
                        for index in range(accumulation):
                            b = batch(size=size)
                            b.mask_tensor = torch.full((size, 1, 6, 8), .1)
                            for image in range(size):
                                b.mask_tensor[image, :, :, :2 * ((image + index) % 4 + 1)] = 1
                            b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
                            b.get_network_weight_list = lambda: [1.] * size
                            batches.append(b)
                        initial = parameter.detach().clone().requires_grad_(True)
                        result = trainer.hook_train_loop(batches)
                        losses = []
                        for index, (x, target, times) in enumerate(forwards):
                            weights = expected_mask_weights(batches[index].mask_tensor, x) if masked_loss else 1.0
                            per_image = ((initial * x - target).square() * weights).flatten(1).mean(1)
                            losses.append(per_image.mean())
                            sampler.record(times / 1000, per_image.detach())
                            torch.testing.assert_close(times, expected_times[index], rtol=0, atol=0)
                        loss = torch.stack(losses).mean()
                        gradient = torch.autograd.grad(loss, initial)[0]
                        torch.testing.assert_close(parameter, initial - .01 * gradient, rtol=1e-6, atol=1e-7)
                        torch.testing.assert_close(result['loss'], loss.item(), rtol=1e-6, atol=1e-6)
                        sampler.on_optimizer_step()
                        self.assertEqual(trainer.infonoise_sampler.state_dict(), sampler.state_dict())
                        self.assertEqual(sum(sampler.total_counts), size * accumulation)

    def test_lokr_training_matches_kronecker_gradient_and_resumes(self):
        from toolkit.lora_special import LoRASpecialNetwork
        from toolkit.models.lokr import LokrModule

        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(17)

        class TinyFlow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(32, 32)

            def forward(self, x):
                return self.proj(x.flatten(1)).reshape_as(x)

        def make_network(config, base_state=None):
            model = TinyFlow().requires_grad_(False)
            if base_state is not None:
                model.load_state_dict(base_state)
            network = LoRASpecialNetwork(
                text_encoder=None, unet=model, lora_dim=config.linear,
                alpha=config.linear_alpha, train_unet=True, train_text_encoder=False,
                network_config=config, network_type=config.type, is_transformer=True,
                target_lin_modules=['TinyFlow'], target_conv_modules=[])
            network.force_to('cpu', torch.float32)
            network.apply_to(None, model, False, True)
            network._update_torch_multiplier()
            return model, network

        for full_rank, masked in ((True, False), (False, False), (True, True), (False, True)):
            for batch_size in (1, 4):
                for accumulation in (1, 4):
                    with self.subTest(full_rank=full_rank, masked=masked, batch_size=batch_size, accumulation=accumulation):
                        trainer, _, _ = self.trainer()
                        config = NetworkConfig(type='lokr', lokr_full_rank=full_rank,
                                               linear=1, linear_alpha=1, lokr_factor=2)
                        model, network = make_network(config)
                        self.assertEqual(len(network.unet_loras), 1)
                        module = network.unet_loras[0]
                        self.assertIsInstance(module, LokrModule)
                        self.assertEqual(module.use_w2, full_rank)
                        trainer.network = trainer.sd.network = network
                        trainer.network_config = config
                        trainer.guidance_config = trainer.custom_pipeline = None
                        trainer.do_lorm = False
                        trainer.embed_config = trainer.adapter_config = trainer.decorator_config = None
                        trainer.infonoise_sampler = ready_sampler()
                        trainer.train_config.gradient_accumulation = accumulation
                        validate_process(trainer)
                        groups = network.prepare_optimizer_params(None, .01, .01)
                        trainer.params = [p for group in groups for p in group['params']]
                        trainer.optimizer = trainer.accelerator.prepare(torch.optim.SGD(groups, lr=.01))
                        base_state = copy.deepcopy(model.state_dict())
                        forwards, noise_now = [], []
                        def noise(latents, *args, **kwargs):
                            noise_now[:] = [torch.randn_like(latents)]
                            return noise_now[0]
                        def predict(**kwargs):
                            x = kwargs['noisy_latents']
                            target = noise_now[0] - kwargs['batch'].latents
                            prediction = model(x)
                            forwards.append((x.detach(), target.detach(), kwargs['timesteps'].clone(),
                                             prediction.detach()))
                            return prediction
                        trainer.get_noise, trainer.predict_noise = noise, predict
                        for update in range(2):
                            forwards.clear()
                            oracle = {name: p.detach().clone().requires_grad_(True)
                                      for name, p in module.named_parameters()}
                            sampler = InfoNoiseSampler(trainer.infonoise_sampler.state_dict()['config'])
                            sampler.load_state_dict(trainer.infonoise_sampler.state_dict())
                            expected_times = [sampler.sample(batch_size) * 1000 for _ in range(accumulation)]
                            batches = []
                            for _ in range(accumulation):
                                b = batch(size=batch_size, shape=(1, 2, 16))
                                if masked:
                                    b.mask_tensor = torch.full((batch_size, 1, 4, 32), .1)
                                    for image in range(batch_size):
                                        b.mask_tensor[image, :, :, :4 * (image + 1)] = 1
                                b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
                                b.get_network_weight_list = lambda: [1.] * batch_size
                                batches.append(b)
                            result = trainer.hook_train_loop(batches)
                            w2 = oracle['lokr_w2'] if full_rank else oracle['lokr_w2_a'] @ oracle['lokr_w2_b']
                            weight = base_state['proj.weight'] + torch.kron(oracle['lokr_w1'], w2) * module.scale
                            losses = []
                            for index, (x, target, times, prediction) in enumerate(forwards):
                                expected = torch.nn.functional.linear(x.flatten(1), weight, base_state['proj.bias']).reshape_as(x)
                                torch.testing.assert_close(prediction, expected, rtol=1e-5, atol=1e-6)
                                torch.testing.assert_close(times, expected_times[index], rtol=0, atol=0)
                                weights = expected_mask_weights(batches[index].mask_tensor, x)
                                losses.append(((expected - target).square() * weights).mean())
                                sampler.record(times / 1000, ((prediction - target).square() * weights).flatten(1).mean(1))
                            loss = torch.stack(losses).mean()
                            gradients = torch.autograd.grad(loss, tuple(oracle.values()))
                            for (name, initial), gradient in zip(oracle.items(), gradients):
                                torch.testing.assert_close(dict(module.named_parameters())[name], initial - .01 * gradient,
                                                           rtol=1e-5, atol=1e-6)
                            torch.testing.assert_close(result['loss'], loss.item(), rtol=1e-6, atol=1e-6)
                            sampler.on_optimizer_step()
                            self.assertEqual(trainer.infonoise_sampler.state_dict(), sampler.state_dict())
                            self.assertEqual(trainer.infonoise_sampler.optimizer_steps, update + 2)
                            self.assertEqual(sum(trainer.infonoise_sampler.total_counts), 16 + (update + 1) * batch_size * accumulation)
                            for name, value in model.state_dict().items():
                                torch.testing.assert_close(value, base_state[name], rtol=0, atol=0)
                        self.assertTrue(any(not torch.equal(p.detach(), oracle[name].detach())
                                            for name, p in module.named_parameters()))
                        with tempfile.TemporaryDirectory() as temporary:
                            trainer.save_root, trainer.job, trainer.meta = temporary, NS(name='lokr'), OrderedDict()
                            trainer.embed_config = trainer.decorator = None
                            trainer.named_lora = trainer.is_fine_tuning = False
                            trainer.save_config = SaveConfig(dtype='float32')
                            trainer.update_training_metadata = lambda: trainer.meta.update(training_info={'step': 2, 'epoch': 0})
                            trainer.clean_up_saves = lambda: None
                            trainer.post_save_hook = lambda _: None
                            trainer.step_num = 2
                            SAVE(trainer, step=2)
                            checkpoint = str(Path(temporary) / 'lokr_000000002.safetensors')
                            resumed_model, resumed_network = make_network(config, base_state)
                            resumed_network.load_weights(checkpoint)
                            resumed = process(InfoNoiseSampler(sampler.state_dict()['config']))
                            resumed.accelerator = trainer.accelerator
                            resumed.network_config = config
                            LOAD_META(resumed, checkpoint)
                            self.assertEqual(resumed.step_num, 2)
                            self.assertEqual(resumed.infonoise_sampler.state_dict(), sampler.state_dict())
                            torch.testing.assert_close(trainer.infonoise_sampler.sample(batch_size),
                                                       resumed.infonoise_sampler.sample(batch_size), rtol=0, atol=0)
                            x = torch.randn(batch_size, 1, 2, 16)
                            with network, resumed_network:
                                torch.testing.assert_close(model(x), resumed_model(x), rtol=0, atol=0)

    def test_nonfused_automagic_accumulates_before_one_update(self):
        from toolkit.optimizers.automagic3 import Automagic3
        trainer, parameter, forwards = self.trainer()
        optimizer = Automagic3([parameter], lr=1e-6, fused=False)
        validate_optimizer(optimizer)
        trainer.optimizer = trainer.accelerator.prepare(optimizer)
        initial = parameter.detach().clone()
        batches = []
        for _ in range(4):
            b = batch(size=1)
            b.clip_image_tensor = b.clip_image_embeds = b.control_tensor_list = None
            b.get_network_weight_list = lambda: [1.]
            batches.append(b)
        trainer.hook_train_loop(batches)
        self.assertEqual(len(forwards), 4)
        self.assertEqual(optimizer.state[parameter]['step'], 1)
        self.assertEqual(trainer.infonoise_sampler.optimizer_steps, 1)
        self.assertEqual(sum(trainer.infonoise_sampler.total_counts), 4)
        self.assertFalse(trainer.infonoise_sampler.pending)
        self.assertFalse(torch.equal(initial, parameter))

    def test_sampler_device_transfers_and_detachment(self):
        devices = ['cpu']
        if torch.backends.mps.is_available():
            devices.append('mps')
        if torch.cuda.is_available():
            devices.append('cuda')
        for device in devices:
            with self.subTest(device=device):
                sampler = InfoNoiseSampler(seed=4)
                times = sampler.sample(5, device) * 1000
                self.assertEqual(times.device.type, device)
                clean = torch.zeros(5, 1, 2, 2, device=device)
                noise = torch.ones_like(clean)
                prediction = torch.zeros_like(clean, requires_grad=True)
                b = batch(size=5, shape=(1, 2, 2))
                b.latents = clean
                trainer = process(sampler)
                trainer.device_torch = torch.device(device)
                LOSS(trainer, prediction, noise, clean, times, b)
                sampler.on_optimizer_step()
                self.assertEqual(sum(sampler.total_counts), 5)
                self.assertTrue(all(isinstance(error, float) for buf in sampler.fifo for _, error in buf))
                self.assertIsNone(prediction.grad)


if __name__ == '__main__':
    unittest.main()
