"""Timestep-normalized per-image loss diagnostics and bounded loss weighting.

The legacy classifier follows Fizgig's residual and persistence rules.
Coverage compares each image's progress within matched noise bands.
Weights change loss contributions, not the optimizer's learning rate. In
particular, Muon's normalized matrix update need not scale with the loss.
Only observations from successful optimizer updates should be recorded.
"""

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean, median, stdev
from typing import Any

DEFAULTS = {
    "mode": "off",
    "policy": "legacy",
    "bins": 40,
    "warmup_epochs": 2,
    "window": 5,
    "trend_window": 8,
    "throttle_mult": 0.5,
    "easy_mult": 1.1,
    "hi_q": 0.66,
    "lo_q": 0.33,
    "persist_on": 2,
    "persist_off": 3,
    "improve_frac": 0.12,
    "improve_floor": 0.02,
    "suspect_mult": 0.7,
    "easy_from_epoch": 3,
    "exhausted_mult": 0.6,
    "exhaust_drop_frac": 0.3,
    "exhaust_on": 2,
    "stuck_floor": 0.25,
    "escalate_every": 2,
}
_COVERAGE_MIN_EPOCHS = 16
_COVERAGE_MAX_EPOCHS = 40
_COVERAGE_DECISIONS = {"learning": (2, 1.10), "easy": (2, 0.95), "stuck": (3, 0.75)}
_COVERAGE_STATS = {
    "verdict",
    "candidate",
    "votes",
    "matched_bands",
    "improving",
    "mean_loss",
    "epochs",
}
_COUNTERS = (
    "stuck_votes",
    "clear_votes",
    "suspect_votes",
    "exhaust_votes",
    "stuck_epochs",
)
_VERDICTS = {
    "unobserved",
    "warmup",
    "stuck",
    "suspect",
    "exhausted",
    "watch",
    "learning",
    "easy",
    "mid",
}


def _integer(value: object, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value: object, minimum: float = 0.0) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= minimum
        )
    except OverflowError:
        return False


def _config(options: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(options, Mapping):
        raise ValueError("per-image adaptive configuration must be a mapping")
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            "unknown per-image adaptive options: "
            + ", ".join(sorted(map(str, unknown)))
        )
    cfg = {**DEFAULTS, **options}
    if cfg["mode"] not in ("off", "observe", "adaptive"):
        raise ValueError("per-image adaptive mode must be off, observe or adaptive")
    if cfg["policy"] not in ("legacy", "coverage"):
        raise ValueError("per-image adaptive policy must be legacy or coverage")
    if cfg["policy"] == "coverage":
        cfg["bins"] = options.get("bins", 4)
        cfg["trend_window"] = options.get("trend_window", _COVERAGE_MAX_EPOCHS)
    ints = (
        "bins",
        "warmup_epochs",
        "window",
        "trend_window",
        "persist_on",
        "persist_off",
        "easy_from_epoch",
        "exhaust_on",
        "escalate_every",
    )
    for name in ints:
        if not _integer(cfg[name], 1):
            raise ValueError(f"{name} must be a positive integer")
    if cfg["trend_window"] < 4 or cfg["trend_window"] % 2:
        raise ValueError("trend_window must be an even integer of at least four")
    if cfg["policy"] == "coverage" and cfg["bins"] < 2:
        raise ValueError("coverage requires at least two noise bins")
    for name in set(DEFAULTS) - set(ints) - {"mode", "policy"}:
        if not _number(cfg[name]):
            raise ValueError(f"{name} must be finite and nonnegative")
        cfg[name] = float(cfg[name])
    if not 0 <= cfg["lo_q"] < cfg["hi_q"] <= 1:
        raise ValueError("quantiles must satisfy 0 <= lo_q < hi_q <= 1")
    if not 0 < cfg["stuck_floor"] <= cfg["throttle_mult"] <= 1:
        raise ValueError("require 0 < stuck_floor <= throttle_mult <= 1")
    if not 0 < cfg["suspect_mult"] <= 1 or not 0 < cfg["exhausted_mult"] <= 1:
        raise ValueError("suspect and exhausted multipliers must be in (0, 1]")
    if not 1 <= cfg["easy_mult"] <= 2:
        raise ValueError("easy_mult must be in [1, 2]")
    if cfg["improve_frac"] > 1 or cfg["exhaust_drop_frac"] > 1:
        raise ValueError("improvement fractions must be in [0, 1]")
    return cfg


def _new_image() -> dict[str, Any]:
    return {
        "history": {},
        "loss_sum": 0.0,
        "observations": 0,
        "epochs": 0,
        **{name: 0 for name in _COUNTERS},
        "confirmed": False,
        "multiplier": 1.0,
        "stats": {},
    }


def _coverage_band_vote(
    early: list[float], late: list[float], current: float, lower: float, middle: float
) -> str:
    # Scale the comparison to avoid an absolute loss threshold across models.
    previous = mean(early)
    change = previous - current
    error = math.hypot(
        stdev(early) / math.sqrt(len(early)), stdev(late) / math.sqrt(len(late))
    )
    hard = current > 0 and current / 1.15 >= lower
    easy = middle > 0 and current <= 0.85 * middle and current <= lower
    stable = abs(change) <= 0.08 * previous - error
    if easy and stable:
        return "easy"
    if hard and change > error and change >= 0.12 * previous:
        return "learning"
    if hard and stable:
        return "stuck"
    return "mid"


class PerImageAdaptive:
    """Portable controller; all state is CPU Python data and JSON-serializable.

    Bind stable dataset-relative image identities before use. Dataset repeats
    share an identity. Epochs are zero-based and observations must arrive before
    their epoch is finalized. Images absent from an epoch do not receive votes.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = _config({} if config is None else config)
        self._keys = None
        self._images = {}
        self._sums = [0.0] * self.config["bins"]
        self._counts = [0] * self.config["bins"]
        self._last_epoch = -1
        self._pending_epoch = None
        self._epochs_seen = 0

    def bind_dataset(self, keys: Iterable[str]) -> None:
        values = list(keys)
        if any(not isinstance(key, str) or not key for key in values):
            raise ValueError("dataset identities must be nonempty strings")
        normalized = tuple(sorted(set(values)))
        if self._keys is not None and normalized != self._keys:
            raise ValueError("per-image adaptive dataset identities changed")
        self._keys = normalized
        for key in normalized:
            self._images.setdefault(key, _new_image())

    def _bound(self) -> None:
        if self._keys is None:
            raise ValueError(
                "bind_dataset must be called before using per-image adaptive state"
            )

    def record(
        self, key: str, epoch: int, timestep_fraction: float, raw_masked_loss: float
    ) -> bool:
        """Return False for invalid observations, without changing any state."""
        if self.config["mode"] == "off":
            return False
        self._bound()
        if key not in self._images:
            raise KeyError(f"unknown per-image adaptive identity: {key}")
        if (
            not _integer(epoch)
            or not _number(timestep_fraction)
            or timestep_fraction > 1
            or not _number(raw_masked_loss)
        ):
            return False
        if epoch <= self._last_epoch:
            raise ValueError("cannot record a finalized epoch")
        if self._pending_epoch is not None and epoch != self._pending_epoch:
            raise ValueError("finalize the pending epoch before recording another")
        bucket = min(
            int(timestep_fraction * self.config["bins"]), self.config["bins"] - 1
        )
        item = self._images[key]
        total = self._sums[bucket] + raw_masked_loss
        image_total = item["loss_sum"] + raw_masked_loss
        previous = item["history"].get(epoch, {}).get(bucket, [0.0, 0])
        bin_total = previous[0] + raw_masked_loss
        if not all(math.isfinite(x) for x in (total, image_total, bin_total)):
            return False
        if epoch not in item["history"]:
            item["epochs"] += 1
        item["history"].setdefault(epoch, {})[bucket] = [bin_total, previous[1] + 1]
        item["loss_sum"] = image_total
        item["observations"] += 1
        self._sums[bucket] = total
        self._counts[bucket] += 1
        self._pending_epoch = epoch
        return True

    def _statistics(
        self, item: dict[str, Any], means: Sequence[float]
    ) -> dict[str, float]:
        eps = sorted(item["history"])
        series = []
        for epoch in eps:
            values = item["history"][epoch]
            count = sum(v[1] for v in values.values())
            series.append(
                sum(
                    total - count_b * means[b] for b, (total, count_b) in values.items()
                )
                / count
            )
        tw = min(self.config["trend_window"], item["epochs"] - item["epochs"] % 2)
        old = new = se = 0.0
        if tw >= 4:
            recent = series[-tw:]
            half = tw // 2
            old, new = sum(recent[:half]) / half, sum(recent[half:]) / half
            var = (
                sum((x - old) ** 2 for x in recent[:half])
                + sum((x - new) ** 2 for x in recent[half:])
            ) / (tw - 2)
            se = math.sqrt(2 * var / half)
        recent = series[-self.config["window"] :]
        recent_eps = eps[-self.config["window"] :]
        mx, my = sum(recent_eps) / len(recent_eps), sum(recent) / len(recent)
        denom = sum((x - mx) ** 2 for x in recent_eps)
        slope = (
            sum((x - mx) * (y - my) for x, y in zip(recent_eps, recent)) / denom
            if denom
            else 0.0
        )
        baseline = sum(series[: min(3, len(series))]) / min(3, len(series))
        return {
            "mean_residual": my,
            "slope": slope,
            "first": old,
            "last": new,
            "se": se,
            "trend_epochs": tw,
            "baseline": baseline,
            "total_drop": baseline - new if tw >= 4 else 0.0,
            "mean_loss": item["loss_sum"] / item["observations"],
            "epochs": item["epochs"],
        }

    def _update_coverage(self, epoch: int, active: set[str]) -> None:
        window = self.config["trend_window"]
        min_epochs = min(window, _COVERAGE_MIN_EPOCHS)
        warmup = self._epochs_seen < max(min_epochs, self.config["warmup_epochs"])
        size = min(window, self._epochs_seen)
        size -= size % 2
        split, start = epoch - size // 2, epoch - size + 1
        # Keep one mean per image, band and epoch. Repeats within one epoch
        # cannot create extra evidence or dominate another image's reference.
        windows = {}
        cohort = {}
        if not warmup:
            for key in active:
                early, late = {}, {}
                for e, bands in self._images[key]["history"].items():
                    if e < start:
                        continue
                    window = early if e <= split else late
                    for band, (total, count) in bands.items():
                        window.setdefault(band, []).append(total / count)
                recent = {
                    band: mean(values)
                    for band, values in late.items()
                    if len(values) >= 2
                }
                for band, value in recent.items():
                    cohort.setdefault(band, []).append(value)
                windows[key] = early, late, recent
        references = {}
        for band, values in cohort.items():
            if len(values) >= 3:
                values.sort()
                references[band] = values[int(0.33 * (len(values) - 1))], median(values)

        for key, item in self._images.items():
            if key not in active and not item["observations"]:
                continue
            early, late, recent = windows.get(key, ({}, {}, {}))
            votes = {
                band: _coverage_band_vote(
                    early[band], late[band], value, *references[band]
                )
                for band, value in recent.items()
                if len(early.get(band, ())) >= 2 and band in references
            }
            candidate = "mid"
            decisive = {band: vote for band, vote in votes.items() if vote != "mid"}
            fresh = any(band in decisive for band in item["history"].get(epoch, ()))
            if fresh and len(decisive) >= 2 and len(set(decisive.values())) == 1:
                candidate = next(iter(decisive.values()))
            previous = item["stats"]
            consecutive = 0
            verdict = "warmup" if warmup else "mid"
            multiplier = 1.0
            if candidate in _COVERAGE_DECISIONS:
                needed, weight = _COVERAGE_DECISIONS[candidate]
                consecutive = 1
                if (
                    epoch == self._last_epoch + 1
                    and previous.get("candidate") == candidate
                ):
                    consecutive = min(needed, previous["votes"] + 1)
                verdict = candidate if consecutive >= needed else "watch"
                if verdict == candidate:
                    multiplier = weight
            item["multiplier"] = multiplier
            item["stats"] = {
                "verdict": verdict,
                "candidate": candidate,
                "votes": consecutive,
                "matched_bands": len(votes),
                "improving": candidate == "learning",
                "mean_loss": item["loss_sum"] / item["observations"],
                "epochs": item["epochs"],
            }

    def end_epoch(self, epoch: int) -> dict[str, Any]:
        self._bound()
        if not _integer(epoch):
            raise ValueError("epoch must be a nonnegative integer")
        if self.config["mode"] == "off":
            return self.report()
        if epoch == self._last_epoch:
            return self.report()
        if epoch < self._last_epoch or (
            self._pending_epoch is not None and epoch != self._pending_epoch
        ):
            raise ValueError("epoch boundary does not match pending observations")
        active = {key for key, item in self._images.items() if epoch in item["history"]}
        if active:
            self._epochs_seen += 1
        if self.config["policy"] == "coverage":
            self._update_coverage(epoch, active)
        else:
            self._update_legacy(active)
        # Preserve the first three epochs and the latest history needed by the
        # classifiers. Keep all-run bin means for renormalizing legacy residuals.
        keep_recent = max(self.config["window"], self.config["trend_window"])
        for item in self._images.values():
            eps = sorted(item["history"])
            keep = set(eps[:3]) | set(eps[-keep_recent:])
            item["history"] = {e: item["history"][e] for e in eps if e in keep}
        self._last_epoch = epoch
        self._pending_epoch = None
        return self.report()

    def _update_legacy(self, active: set[str]) -> None:
        means = [s / n if n else 0.0 for s, n in zip(self._sums, self._counts)]
        stats = {
            key: self._statistics(item, means)
            for key, item in self._images.items()
            if item["observations"]
        }
        cfg = self.config
        if stats and self._epochs_seen >= cfg["warmup_epochs"]:
            ordered = sorted(s["mean_residual"] for s in stats.values())
            n = len(ordered)
            hi = ordered[int(cfg["hi_q"] * (n - 1))]
            lo = ordered[int(cfg["lo_q"] * (n - 1))]
            med = ordered[n // 2]
            iqr = ordered[int(0.75 * (n - 1))] - ordered[int(0.25 * (n - 1))]
            fence, extreme_fence = max(med + 1.5 * iqr, hi), max(med + 3 * iqr, hi)
            for key in active:
                item, s = self._images[key], stats[key]
                drop = s["first"] - s["last"]
                improving = s["trend_epochs"] >= 4 and drop >= max(
                    cfg["improve_frac"] * max(s["first"], 0),
                    cfg["improve_floor"],
                    s["se"],
                )
                good = (
                    s["trend_epochs"] >= 4
                    and s["baseline"] > 0
                    and s["total_drop"]
                    >= max(
                        cfg["exhaust_drop_frac"] * s["baseline"],
                        2 * cfg["improve_floor"],
                    )
                )
                stuck_vote = (
                    s["trend_epochs"] >= 4
                    and s["mean_residual"] >= fence
                    and s["mean_residual"] > s["se"]
                    and s["last"] > 0
                    and not improving
                    and not good
                )
                if item["confirmed"]:
                    item["stuck_epochs"] += 1
                    item["clear_votes"] = 0 if stuck_vote else item["clear_votes"] + 1
                    if item["clear_votes"] >= cfg["persist_off"]:
                        item["confirmed"] = False
                        item["clear_votes"] = item["stuck_votes"] = 0
                else:
                    item["stuck_votes"] = item["stuck_votes"] + 1 if stuck_vote else 0
                    if item["stuck_votes"] >= cfg["persist_on"]:
                        item["confirmed"] = True
                        item["clear_votes"] = 0
                        item["stuck_epochs"] += 1
                # A degenerate IQR can put both fences at zero. Equality alone
                # is not evidence of difficulty: otherwise an exactly matched
                # clean cohort is entirely throttled during early warmup.
                extreme = (
                    s["mean_residual"] > 0
                    and s["mean_residual"] >= fence
                    and not improving
                    and not good
                    and (s["trend_epochs"] < 4 or s["mean_residual"] > s["se"])
                )
                item["suspect_votes"] = item["suspect_votes"] + 1 if extreme else 0
                suspect = not item["confirmed"] and (
                    item["suspect_votes"] >= 2
                    or (
                        item["suspect_votes"] >= 1
                        and s["mean_residual"] >= extreme_fence
                    )
                )
                exhausted_vote = good and not improving and s["last"] > 0
                item["exhaust_votes"] = (
                    item["exhaust_votes"] + 1 if exhausted_vote else 0
                )
                if item["confirmed"]:
                    exponent = min(
                        1074, (item["stuck_epochs"] - 1) // cfg["escalate_every"]
                    )
                    verdict, mult = (
                        "stuck",
                        max(cfg["stuck_floor"], cfg["throttle_mult"] * 0.5**exponent),
                    )
                elif suspect:
                    verdict, mult = "suspect", cfg["suspect_mult"]
                elif item["exhaust_votes"] >= cfg["exhaust_on"]:
                    verdict, mult = "exhausted", cfg["exhausted_mult"]
                elif stuck_vote:
                    verdict, mult = "watch", 1.0
                elif s["mean_residual"] >= hi:
                    verdict, mult = "learning", 1.0
                elif (
                    s["mean_residual"] <= lo
                    and self._epochs_seen >= cfg["easy_from_epoch"]
                ):
                    verdict, mult = "easy", cfg["easy_mult"]
                else:
                    verdict, mult = "mid", 1.0
                item["multiplier"] = mult
                s.update(
                    verdict=verdict,
                    improving=improving,
                    good_run=good,
                    stuck_epochs=item["stuck_epochs"],
                    release_votes=item["clear_votes"],
                )
                item["stats"] = s
        else:
            for key in active:
                self._images[key]["stats"] = {
                    **stats[key],
                    "verdict": "warmup",
                    "improving": False,
                }

    def multiplier(self, key: str, is_reg: bool = False) -> float:
        if is_reg or self.config["mode"] != "adaptive":
            return 1.0
        self._bound()
        if key not in self._images:
            raise KeyError(f"unknown per-image adaptive identity: {key}")
        return self._images[key]["multiplier"]

    def report(self) -> dict[str, Any]:
        images = {}
        for key, item in self._images.items():
            images[key] = {
                **copy.deepcopy(item["stats"]),
                "verdict": item["stats"].get("verdict", "unobserved"),
                "multiplier": item["multiplier"]
                if self.config["mode"] == "adaptive"
                else 1.0,
                "suggested_multiplier": item["multiplier"],
                "observations": item["observations"],
            }
        return {
            "mode": self.config["mode"],
            "policy": self.config["policy"],
            "epoch": self._last_epoch,
            "observed_epochs": self._epochs_seen,
            "bin_counts": list(self._counts),
            "images": images,
        }

    def state_dict(self) -> dict[str, Any]:
        self._bound()
        images = copy.deepcopy(self._images)
        for item in images.values():
            item["history"] = [
                [e, [[b, v[0], v[1]] for b, v in sorted(values.items())]]
                for e, values in sorted(item["history"].items())
            ]
        return {
            "version": 1,
            "config": dict(self.config),
            "dataset_keys": list(self._keys),
            "bin_sums": list(self._sums),
            "bin_counts": list(self._counts),
            "last_epoch": self._last_epoch,
            "pending_epoch": self._pending_epoch,
            "observed_epochs": self._epochs_seen,
            "images": images,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate everything before replacing live state; mismatches are errors."""
        self._bound()
        expected = {
            "version",
            "config",
            "dataset_keys",
            "bin_sums",
            "bin_counts",
            "last_epoch",
            "pending_epoch",
            "observed_epochs",
            "images",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or not _integer(state["version"], 1)
            or state["version"] != 1
        ):
            raise ValueError("unsupported per-image adaptive checkpoint schema")
        saved_config = state["config"]
        _config(saved_config)
        # Snapshots from before policy selection contain every legacy option
        # except this key. Continue to reject incomplete or changed settings.
        if {"policy": "legacy", **saved_config} != self.config:
            raise ValueError("per-image adaptive checkpoint configuration changed")
        if state["dataset_keys"] != list(self._keys):
            raise ValueError("per-image adaptive checkpoint dataset identities changed")
        candidate = copy.deepcopy(state)
        sums, counts = candidate["bin_sums"], candidate["bin_counts"]
        if (
            not isinstance(sums, list)
            or not isinstance(counts, list)
            or len(sums) != self.config["bins"]
            or len(counts) != len(sums)
            or not all(_number(x) for x in sums)
            or not all(_integer(x) for x in counts)
            or any(n == 0 and s != 0 for s, n in zip(sums, counts))
        ):
            raise ValueError("invalid per-image adaptive bin statistics")
        last, pending = candidate["last_epoch"], candidate["pending_epoch"]
        if (
            not _integer(last, -1)
            or not _integer(candidate["observed_epochs"])
            or candidate["observed_epochs"] > last + 1
            or (pending is not None and (not _integer(pending) or pending <= last))
        ):
            raise ValueError("invalid per-image adaptive epoch state")
        images = candidate["images"]
        if not isinstance(images, dict) or set(images) != set(self._keys):
            raise ValueError("invalid per-image adaptive image state")
        for item in images.values():
            if (
                not isinstance(item, dict)
                or set(item) != set(_new_image())
                or not all(
                    _integer(item[n]) for n in (*_COUNTERS, "observations", "epochs")
                )
                or not _number(item["loss_sum"])
                or not isinstance(item["confirmed"], bool)
                or not _number(item["multiplier"])
                or not 0 < item["multiplier"] <= 2
            ):
                raise ValueError("invalid per-image adaptive image counters")
            history, previous = {}, -1
            if not isinstance(item["history"], list):
                raise ValueError("invalid per-image adaptive history")
            for row in item["history"]:
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError("invalid per-image adaptive epoch aggregate")
                ep, bins = row
                if (
                    not _integer(ep)
                    or ep <= previous
                    or (ep > last and ep != pending)
                    or not isinstance(bins, list)
                    or not bins
                ):
                    raise ValueError("invalid per-image adaptive history epoch")
                previous = ep
                values = {}
                for entry in bins:
                    if (
                        not isinstance(entry, list)
                        or len(entry) != 3
                        or not _integer(entry[0])
                        or entry[0] >= len(sums)
                        or entry[0] in values
                        or not _number(entry[1])
                        or not _integer(entry[2], 1)
                    ):
                        raise ValueError("invalid per-image adaptive history bin")
                    values[entry[0]] = [entry[1], entry[2]]
                history[ep] = values
            retained_count = sum(
                v[1] for values in history.values() for v in values.values()
            )
            retained_loss = sum(
                v[0] for values in history.values() for v in values.values()
            )
            max_history = (
                3
                + max(self.config["window"], self.config["trend_window"])
                + (pending is not None)
            )
            if (
                item["epochs"] < len(history)
                or item["observations"] < item["epochs"]
                or bool(history) != bool(item["observations"])
                or len(history) > max_history
                or retained_count > item["observations"]
                or retained_loss > item["loss_sum"] + 1e-12 * max(1, item["loss_sum"])
            ):
                raise ValueError("inconsistent per-image adaptive history")
            stats = item["stats"]
            allowed = {
                "mean_residual",
                "slope",
                "first",
                "last",
                "se",
                "trend_epochs",
                "baseline",
                "total_drop",
                "mean_loss",
                "epochs",
                "verdict",
                "improving",
                "good_run",
                "stuck_epochs",
                "release_votes",
            }
            coverage = self.config["policy"] == "coverage"
            if coverage:
                allowed = _COVERAGE_STATS
            verdict = (
                stats.get("verdict", "unobserved") if isinstance(stats, dict) else None
            )
            if (
                not isinstance(stats, dict)
                or set(stats) - allowed
                or not isinstance(verdict, str)
                or verdict not in _VERDICTS
            ):
                raise ValueError("invalid per-image adaptive verdict")
            if coverage:
                self._validate_coverage_evidence(item, history, last, pending)
            for name, value in stats.items():
                if name in ("verdict", "candidate"):
                    continue
                if not isinstance(value, (int, float, bool)) or not math.isfinite(
                    value
                ):
                    raise ValueError("invalid per-image adaptive report statistic")
            item["history"] = history
        if sum(counts) != sum(item["observations"] for item in images.values()):
            raise ValueError("inconsistent per-image adaptive observation totals")
        if (pending is not None) != any(
            pending in item["history"] for item in images.values()
        ):
            raise ValueError("inconsistent per-image adaptive pending epoch")
        # All compatibility and value checks passed. No earlier operation mutates
        # this instance, including failed loads after partial history validation.
        self._sums, self._counts, self._images = sums, counts, images
        self._last_epoch, self._pending_epoch = last, pending
        self._epochs_seen = candidate["observed_epochs"]

    def _validate_coverage_evidence(
        self,
        item: dict[str, Any],
        history: dict[int, Any],
        last: int,
        pending: int | None,
    ) -> None:
        stats = item["stats"]
        epochs = item["epochs"] - (pending in history)
        error = "invalid per-image adaptive coverage evidence"
        if not epochs and not stats and item["multiplier"] == 1.0:
            return
        if not epochs or set(stats) != _COVERAGE_STATS:
            raise ValueError(error)
        candidate = stats["candidate"]
        if candidate not in ("mid", *_COVERAGE_DECISIONS):
            raise ValueError(error)
        needed, multiplier = _COVERAGE_DECISIONS.get(candidate, (0, 1.0))
        votes, bands = stats["votes"], stats["matched_bands"]
        verdicts = ("mid", "warmup")
        if needed:
            verdicts = (candidate if votes == needed else "watch",)
        if (
            not _integer(votes, 1 if needed else 0)
            or votes > needed
            or not _integer(bands, 2 if needed else 0)
            or bands > self.config["bins"]
            or stats["epochs"] != epochs
            or stats["improving"] is not (candidate == "learning")
            or stats["verdict"] not in verdicts
            or (needed and last not in history)
            or item["multiplier"] != (multiplier if votes == needed else 1.0)
        ):
            raise ValueError(error)
