# Per-image adaptive loss weighting

The controller tracks each eligible training image across epochs. In `adaptive`
mode, each image's multiplier scales its primary loss before the batch mean and
gradient accumulation. In `observe` mode, the report exposes the same suggested
multipliers and training uses a multiplier of 1. The feature defaults to
`mode: off`. If a policy is omitted, the controller uses `legacy`.

The watcher records the error before the adaptive multiplier, so a lower weight
does not itself appear as an improvement. Successful optimizer updates commit the
observations. Regularization images retain their usual loss and are excluded from
the watcher. InfoNoise observes its own pre-weighting errors. The multiplier scales
a contribution to the gradient. Muon's orthogonalization and other optimizer
normalizations affect how that gradient translates to a parameter update.

## Coverage

Coverage compares each image against its own earlier performance at roughly the
same flow time. The default bands are four equal-width intervals in `u = T / 1000`.
Within each epoch, repeat observations for an image and band are averaged. For the
diagnostic error, the watcher divides the masked error by the mean applied mask
weight. This expresses error per unit of mask weight.

The default window starts at 16 observed epochs and grows in even lengths to 40
epochs. Set `trend_window` to change the cap. An explicit cap below 16 can begin
earlier. `warmup_epochs` can delay the start. Each comparison splits the window
into an early half and a recent half. The image needs at least two distinct
observed epochs in each half for the band. The cohort reference comes from recent
means for at least three images with sufficient recent observations in that band.

For an image and band, let `early` and `recent` be the means of the epoch losses
in each half. The uncertainty estimate is the square root of the sum of the two
estimated variances of those means.

- **Improving** requires a decrease of at least 12% of `early` and greater than the
  estimated uncertainty.
- **Stable** requires `abs(early - recent) + uncertainty <= 0.08 * early`.
- **Hard** means the recent loss is positive and at least 15% above the cohort's
  lower-third reference.
- **Easy** means it is at or below the lower-third reference and at least 15% below
  the cohort median.

A hard and improving band votes for `learning`. An easy and stable band votes for
`easy`. A hard and stable band votes for `stuck`. Other comparisons abstain.

The image needs at least two agreeing decisive bands, no conflicting decisive
band, and a fresh observation in an agreeing band. Confirmation counts consecutive
epoch decisions. A missing or changed decision resets that confirmation sequence.

| Decision | Consecutive confirmations | Loss weight |
| --- | ---: | ---: |
| Hard and improving | 2 | 1.10 |
| Easy and stable | 2 | 0.95 |
| Hard and stable | 3 | 0.75 |
| Warmup, awaiting confirmation, or insufficient evidence | | 1.00 |

Coverage is a heuristic about training loss. The cohort ranks and stability
threshold can be sensitive to sparse visits, noise within a band, and changes in
the sampling schedule. A measured plateau does not identify a bad training image.
Evaluate any quality benefit with matched training runs and held-out evaluation.

## Legacy

Legacy follows Fizgig's residual and persistence rules. The default tracker
maintains the all-run mean error for each of 40 flow-time bins. For each image and
epoch, it subtracts the current mean of the corresponding bin from each observed
loss and averages the resulting residuals. This lets it compare images that were
observed at different noise levels. Historical residuals use the current bin
means when recomputed.

The difficulty score is the mean residual over the last five observed epochs.
To estimate improvement, the policy compares the two halves of an up-to-eight
epoch trend. The drop must exceed the configured fractional, absolute, and
uncertainty thresholds. The earliest three observed epochs supply a baseline
for detecting clear prior progress. The cohort provides quantile cutoffs and
outlier fences based on the median and interquartile range.

| Decision | Default loss weight |
| --- | ---: |
| Warmup, ordinary, or hard and learning | 1.00 |
| Easy after three observed epochs | 1.10 |
| Suspect outlier | 0.70 |
| Previously improved, now plateaued above the bin reference | 0.60 |
| Confirmed stuck | starts at 0.50, floor 0.25 |

Legacy begins classifying after two observed epochs by default. Two consecutive
stuck votes confirm an image, and three clear votes release it. Confirmed stuck
weights halve every two confirmed boundaries down to the floor. Images absent
from an epoch receive no fresh vote. A persistent high residual is evidence about
the current training objective and can reflect content, captions, masks, or noise.

The report at `per_image_adaptive.json` shows the active policy, counts,
classifications, applied multipliers, and suggested multipliers.

[Fizgig loss watcher](https://github.com/shootthesound/Fizgig/blob/11c43efe2eba2d519eb709cb3327e9257b14d4e6/src/fizgig/training/loss_logger.py)
