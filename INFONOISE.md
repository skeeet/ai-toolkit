# InfoNoise for flow-velocity training

InfoNoise adapts how often AI Toolkit draws each training timestep. It estimates a noise-dependent error curve and rebuilds a sampling distribution from it. Ordinary velocity MSE, mask weighting, and optimizer updates still run through the training path.

The conversion from denoising error to effective training allocation follows Sections 2 and 3 and Appendix A.4 of the paper. The gate follows Appendix B.6. Mask-weighted measurements, forward-only probes, age-weighted estimation, and the local low-noise floor are extensions in this patch. [Noise Scheduling as Information-Guided Allocation in Diffusion Training, v2](https://arxiv.org/html/2602.18647v2)

## Noise coordinates and training objective

| Symbol | Meaning |
| --- | --- |
| `u` | Flow time. Zero is clean data and one is pure noise. |
| `T = 1000u` | AI Toolkit training timestep. The scheduler's training sigmas hold `u`. |
| `sigma = u/(1-u)` | Relative noise, the noise coefficient divided by the signal coefficient. |
| `z = log(sigma)` | Coordinate used by the bins and CDF. Log-SNR is `-2z`. |

The configured `min_u` and `max_u` bound every training draw and probe. The range `0.05–0.95` corresponds to training timesteps 50–950.

The linear Gaussian flow path and its velocity target are

```text
x_u = (1-u) x_0 + u epsilon
v = epsilon - x_0
x̂_0 = x_u - u v̂
x̂_0 - x_0 = u (v - v̂)
```

So the clean-data squared error is `m_x = u² m_v`. The sampler applies this conversion to each observation at its actual `u`, before averaging by bin. Unit-weight velocity MSE equals clean-data MSE multiplied by `w_x = 1/u²`.

Dividing `x_u` by `1-u` gives an additive Gaussian channel with relative noise `sigma`. In that coordinate the paper's entropy-rate estimate is `r_sigma = m_x / sigma³`. Let `g` be the low-noise gate. To target the gated entropy rate with a fixed loss weight, divide by `w_x` before normalizing. The change of coordinates then gives

```text
pi_sigma ∝ g * r_sigma / w_x
pi_z     ∝ pi_sigma * sigma
         ∝ g * m_x * u² / sigma²
         = g * m_x * (1-u)²       up to the normalization constant
pi_u     = pi_z / [u(1-u)]       since dz/du = 1/[u(1-u)]
```

The CDF integrates `pi_z`. Substituting `m_x = u² m_v` gives a factor of `u⁴` on velocity error in the expression with `sigma²`. One `u²` converts the measured error and the other compensates for the loss weight.

## Warmup and observations

The initial distribution is `warmup_type` on the configured range. `log_uniform` gives equal probability to equal-width log-noise bins. `uniform` gives equal probability to equal-width flow-time intervals. `sigmoid` uses a normal distribution in `z`, truncated to the configured range, with mean `-warmup_shift` and standard deviation `warmup_scale`.

Each training image contributes its mean squared velocity error over latent components after the trainer applies its spatial mask. With masks disabled, the mean covers the full latent. InfoNoise reads the observation before per-image loss weighting and batch reduction. Regularization images and invalid observations are excluded from the estimate.

Training and probe observations are staged until the optimizer update succeeds. A skipped update discards them. Each coverage bin keeps a bounded FIFO of observations, including exact `u` and optimizer update number. Masked model error is a proxy for denoising difficulty. The entropy-rate identity itself uses Bayes MMSE.

## Forward-only probes

Small batches and concentrated sampling distributions can leave bins with too few recent measurements. Before a probe, the sampler counts pending observations and estimates how many saved observations will stay fresh through the next probe interval. If a bin falls below `min_count`, the probe picks the bins with the smallest projected counts, breaking ties in favor of the least recently sampled. It draws uniformly in log-noise within each selected bin.

The trainer reuses the batch, conditioning, and Gaussian noise draw from the training pass. It rebuilds the noisy latents at the probe timesteps and runs a second forward under `torch.no_grad()` after backward and before the optimizer step. It records the resulting masked error through the same code as the training pass. OstrisLinear quantized layers share their packed weight storage during the probe. Other registered buffers are isolated so a forward can update running statistics safely. The probe has its own timestep RNG and leaves the training timestep draws and gradients alone.

The budget counts extra forward examples, including any regularization rows in the batch. The allowed rate per committed optimizer update is `bins × (min_count + 1) / max_stale_steps`, with at most one probe batch per update. For 16 bins, `min_count: 4`, and `max_stale_steps: 2000`, that is one extra example per 25 updates at batch 1, or one four-example forward per 100 updates at batch 4, while coverage is short. Actual cost depends on the forward pass and how often coverage runs short.

## Rebuilding the error curve

Rebuilds are attempted at `warmup_steps + k × refresh_every` completed optimizer updates for nonnegative integers `k` and positive update counts. A rebuild expires observations at age `max_stale_steps`. At least `max(2, ceil(bins × min_bin_fraction))` coverage bins must each have `min_count` recent observations.

Each usable bin contributes one node. An observation from update `k` has weight `2^(-(step-k)/half_life_steps)`. The default half-life is 250 completed optimizer updates. The node's position is the weighted mean of the observed log-noise coordinates. Its value is the weighted mean of their clean-data errors. Both means use the same weights. The default coverage grid has 16 bins. Each refresh recomputes the nodes from the current FIFO, so a repeated refresh cannot give an observation extra weight.

Between nodes the estimate interpolates clean-data error linearly in log noise. Outside the first and last nodes it extends the nearest velocity error as a constant and converts back with `u²`. The FIFO limit and maximum observation age still apply.

## Gate and inverse CDF

The gate is `g(sigma) = sigma^n / (sigma^n + c^n)`, where `n` is `gate_power` and defaults to 3. The pivot `c` is expressed in relative noise. For automatic onset, find the highest-noise grid point where `r_sigma` reaches `onset_threshold × max(r_sigma)`. Interpolate its crossing with the next higher point in `z` if that point exists. Once a positive pivot has been accepted, the next automatic proposal averages the new and previous log pivots with equal weight. A positive configured `gate_pivot` supplies the proposal directly.

The integration grid has two cells per coverage bin, with an additional edge at `u=0.25` whenever that point falls inside a cell. Each cell receives its midpoint density times its width in log noise. After gating, the sampler preserves at least `min(0.05, ungated_low_share)` below the cutoff. If gated mass falls short, the sampler rescales the cells below the cutoff to the target share and the cells above it to the remainder. Each side retains its conditional shape. This adjustment keeps the selected pivot intact. A range entirely on one side needs no adjustment.

The sampler normalizes the cell masses, inverts their CDF in log noise, and converts each draw to `T = 1000 / (1 + exp(-z))`.

A failed rebuild keeps the previous CDF while its contributing observations are fresh. If the oldest of the contributing points' latest observations reaches `max_stale_steps`, training uses the warmup prior until a new CDF is accepted.

## Reading logs and checkpoints

The low, middle, and high graph series integrate the **training distribution** over `u < 0.25`, `0.25 ≤ u < 0.75`, and `u ≥ 0.75`. They are probabilities from step 0, including warmup. A probe does not produce a separate graph point or substitute a training timestep. Changes in the CDF can still move the curves at refresh boundaries.

Active distributions also log `infonoise/pre_floor_low`, `pre_floor_middle`, `pre_floor_high`, `proposed_pivot` (the raw proposal), and `gate_pivot` (after automatic pivot smoothing). Compare the pre-floor probabilities with the normal mass series for the same update.

Checkpoint state v6 stores the exact observations, counters, allocation and both RNG streams. Resume requires matching v6 sampler state and configuration. To reuse older adapter weights, initialize a new run with fresh sampler state. A legacy `beta` configuration value receives a deprecation warning and has no effect.

For configuration, see [optional InfoNoise timestep sampling](README.md#optional-infonoise-timestep-sampling).
