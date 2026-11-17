"""Behavioral tests for normalized per-image loss weighting; no Torch required."""

import copy
import json
import math
import unittest

from toolkit.per_image_adaptive import PerImageAdaptive

KEYS = [f"ordinary-{i}" for i in range(12)] + ["candidate", "easy"]


def controller(mode="adaptive", **options):
    result = PerImageAdaptive({"mode": mode, **options})
    result.bind_dataset(KEYS)
    return result


def epoch(watch, e, candidate=8.0, ordinary=1.0, easy=0.1, sigma=0.25):
    for key in KEYS:
        value = candidate if key == "candidate" else easy if key == "easy" else ordinary
        watch.record(key, e, sigma, value)
    return watch.end_epoch(e)


def coverage_epoch(watch, e, *, sigmas=(0.125, 0.875), skip=()):
    for key in KEYS:
        if key in skip:
            continue
        for sigma in sigmas:
            loss = {"easy": 0.2, "candidate": 5 + 12 * 0.9**e}.get(key, 1.0)
            watch.record(key, e, sigma, loss * (100 if sigma > 0.5 else 1))
    return watch.end_epoch(e)["images"]


class CoveragePolicyTests(unittest.TestCase):
    def test_default_window_grows_without_delaying_first_decisions(self):
        watch = controller(policy="coverage")
        self.assertEqual(watch.config["trend_window"], 40)
        for e in range(17):
            rows = coverage_epoch(watch, e)
        self.assertEqual(rows["easy"]["multiplier"], 0.95)

    def test_neutral_band_abstains_and_cannot_supply_fresh_confirmation(self):
        for fresh in (True, False):
            with self.subTest(fresh=fresh):
                watch = controller(policy="coverage")
                for e in range(18):
                    for key in KEYS:
                        sigmas = (
                            (0.375,)
                            if e == 17 and key == "candidate" and not fresh
                            else (0.125, 0.375, 0.875)
                        )
                        for sigma in sigmas:
                            loss = 4.0 if key == "candidate" and sigma != 0.375 else 1.0
                            watch.record(key, e, sigma, loss)
                    row = watch.end_epoch(e)["images"]["candidate"]
                self.assertEqual(row["matched_bands"], 3)
                self.assertEqual(row["multiplier"], 0.75 if fresh else 1.0)

    def test_opposing_decisive_band_vetoes_other_two(self):
        watch = controller(policy="coverage")
        for e in range(18):
            for key in KEYS:
                for sigma in (0.125, 0.375, 0.875):
                    loss = (0.2 if sigma == 0.375 else 4.0) if key == "candidate" else 1.0
                    watch.record(key, e, sigma, loss)
            row = watch.end_epoch(e)["images"]["candidate"]
        self.assertEqual(row["matched_bands"], 3)
        self.assertEqual(row["candidate"], "mid")
        self.assertEqual(row["multiplier"], 1.0)

    def test_confirms_hard_improving_and_stable_easy_after_warmup(self):
        watch = controller(policy="coverage")
        for e in range(17):
            rows = coverage_epoch(watch, e)
            if e < 16:
                self.assertTrue(all(row["multiplier"] == 1 for row in rows.values()))
        self.assertEqual(rows["candidate"]["multiplier"], 1.10)
        self.assertEqual(rows["easy"]["multiplier"], 0.95)
        self.assertEqual(rows["ordinary-0"]["multiplier"], 1.0)
        self.assertEqual(rows["candidate"]["matched_bands"], 2)

    def test_confirmed_plateau_is_bounded_and_releases_when_evidence_is_missing(self):
        keys = [f"easy-{i}" for i in range(3)] + [f"hard-{i}" for i in range(7)]
        # The seven difficult images can agree with one another and still be hard.
        watch = PerImageAdaptive({"policy": "coverage", "mode": "adaptive"})
        watch.bind_dataset(keys)
        for e in range(35):
            for key in keys:
                for sigma in (0.125, 0.875):
                    watch.record(key, e, sigma, 1.0 if key.startswith("easy") else 4.0)
            watch.end_epoch(e)
            self.assertEqual(watch.multiplier("hard-0"), 1.0 if e < 17 else 0.75)
        self.assertEqual(watch.multiplier("easy-0"), 0.95)
        for sigma in (0.125, 0.875):
            watch.record("easy-0", 35, sigma, 1.0)
        watch.end_epoch(35)
        self.assertEqual(watch.multiplier("hard-0"), 1.0)
        self.assertEqual(watch.multiplier("hard-0", is_reg=True), 1.0)

    def test_noise_mix_and_repeated_same_epoch_do_not_create_evidence(self):
        watch = controller(policy="coverage", trend_window=16)
        for e in range(32):
            for key in KEYS:
                sigmas = (0.875, 0.625) if e % 16 < 8 else (0.125, 0.375)
                for sigma in sigmas:
                    for _ in range(1 if key == "easy" else 3):
                        watch.record(key, e, sigma, sigma * 100)
            rows = watch.end_epoch(e)["images"]
            self.assertTrue(all(row["multiplier"] == 1 for row in rows.values()))
        self.assertEqual(rows["candidate"]["matched_bands"], 0)

        watch = controller(policy="coverage")
        for e in range(17):
            rows = coverage_epoch(watch, e, sigmas=(0.125,) * 12)
        self.assertEqual(rows["candidate"]["matched_bands"], 1)
        self.assertEqual(watch.multiplier("candidate"), 1.0)

    def test_alternating_single_observations_can_confirm_when_bands_match(self):
        watch = controller(policy="coverage")
        for e in range(17):
            rows = coverage_epoch(watch, e, sigmas=(0.125 if e % 2 else 0.875,))
        self.assertEqual(rows["candidate"]["observations"], 17)
        self.assertEqual(watch.multiplier("candidate"), 1.1)
        self.assertEqual(watch.multiplier("easy"), 0.95)

    def test_plateau_cannot_be_confirmed_by_replaying_the_same_bands(self):
        watch = controller(policy="coverage")
        for e, band in enumerate(
            (1, 1, 0, 0, 3, 3, 1, 1, 1, 2, 0, 0, 3, 3, 2, 2, 2, 2)
        ):
            for key in KEYS:
                watch.record(
                    key, e, (band + 0.5) / 4, 4.0 if key == "candidate" else 1.0
                )
            rows = watch.end_epoch(e)["images"]
        self.assertEqual(rows["candidate"]["matched_bands"], 2)
        self.assertEqual(watch.multiplier("candidate"), 1.0)

    def test_one_observed_epoch_per_half_cannot_be_replaced_by_repeats(self):
        watch = controller(policy="coverage")
        for e in range(17):
            rare_band = e % 8 == 3
            rows = coverage_epoch(
                watch, e, sigmas=(0.875 if rare_band else 0.125,) * 12
            )
        self.assertEqual(rows["candidate"]["matched_bands"], 1)
        self.assertEqual(watch.multiplier("candidate"), 1.0)

    def test_disagreement_and_noisy_flat_averages_are_neutral(self):
        watch = controller(policy="coverage")
        for e in range(20):
            for key in KEYS:
                for sigma in (0.125, 0.875):
                    value = 1.0
                    if key == "candidate":
                        value = 5 + 12 * 0.9**e if sigma < 0.5 else 10.0
                    if key == "easy":
                        value = 4.0 if e % 2 else 20.0
                    watch.record(key, e, sigma, value)
            watch.end_epoch(e)
        self.assertEqual(watch.multiplier("candidate"), 1.0)
        self.assertEqual(watch.multiplier("easy"), 1.0)

    def test_equal_zero_losses_and_tiny_cohorts_are_neutral(self):
        for keys in (["only"], ["low", "high"], KEYS):
            with self.subTest(size=len(keys)):
                watch = PerImageAdaptive({"mode": "adaptive", "policy": "coverage"})
                watch.bind_dataset(keys)
                for e in range(20):
                    for key in keys:
                        for sigma in (0.125, 0.875):
                            watch.record(key, e, sigma, 100.0 if key == "high" else 0.0)
                    report = watch.end_epoch(e)
                self.assertTrue(
                    all(row["multiplier"] == 1 for row in report["images"].values())
                )

    def test_json_resume_preserves_votes_and_observe_preserves_loss(self):
        watch = controller("observe", policy="coverage")
        for e in range(16):
            coverage_epoch(watch, e)
        snapshot = json.loads(json.dumps(watch.state_dict()))
        resume = controller("observe", policy="coverage")
        resume.load_state_dict(snapshot)
        expected = coverage_epoch(watch, 16)
        self.assertEqual(coverage_epoch(resume, 16), expected)
        self.assertEqual(resume.state_dict(), watch.state_dict())
        self.assertEqual(expected["candidate"]["suggested_multiplier"], 1.10)
        self.assertEqual(resume.multiplier("candidate"), 1.0)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            controller("observe").load_state_dict(snapshot)

    def test_pre_policy_checkpoint_resumes_only_with_legacy_policy(self):
        watch = controller()
        for e in range(7):
            epoch(watch, e)
        before = watch.state_dict()
        old = copy.deepcopy(before)
        old["config"].pop("policy", None)
        watch.load_state_dict(old)
        self.assertEqual(watch.state_dict(), before)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            controller(policy="coverage").load_state_dict(old)
        old["config"].pop("bins")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            watch.load_state_dict(old)

    def test_invalid_coverage_checkpoint_does_not_change_live_state(self):
        watch = controller(policy="coverage")
        for e in range(17):
            coverage_epoch(watch, e)
        before = watch.state_dict()
        for mutation in ("votes", "empty"):
            invalid = copy.deepcopy(before)
            item = invalid["images"]["candidate"]
            if mutation == "votes":
                item["stats"]["votes"] = 1
            else:
                item["stats"] = {}
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "coverage evidence"):
                    watch.load_state_dict(invalid)
                self.assertEqual(watch.state_dict(), before)

    def test_unobserved_image_cannot_restore_another_images_weight(self):
        watch = controller(policy="coverage")
        for e in range(17):
            coverage_epoch(watch, e, skip=("ordinary-0",))
        before = watch.state_dict()
        invalid = copy.deepcopy(before)
        donor = invalid["images"]["candidate"]
        invalid["images"]["ordinary-0"].update(
            stats=donor["stats"], multiplier=donor["multiplier"]
        )
        with self.assertRaisesRegex(ValueError, "coverage evidence"):
            watch.load_state_dict(invalid)
        self.assertEqual(watch.state_dict(), before)


class PerImageAdaptiveTests(unittest.TestCase):
    def test_warmup_easy_suspect_stuck_escalation(self):
        watch = controller()
        first = epoch(watch, 0)
        self.assertEqual(first["images"]["candidate"]["verdict"], "warmup")
        self.assertEqual(watch.multiplier("candidate"), 1)
        self.assertEqual(epoch(watch, 1)["images"]["candidate"]["verdict"], "suspect")
        self.assertEqual(watch.multiplier("candidate"), 0.7)
        epoch(watch, 2)
        self.assertEqual(watch.multiplier("easy"), 1.1)
        epoch(watch, 3)
        self.assertEqual(epoch(watch, 4)["images"]["candidate"]["verdict"], "stuck")
        self.assertEqual(watch.multiplier("candidate"), 0.5)
        epoch(watch, 5)
        epoch(watch, 6)
        self.assertEqual(watch.multiplier("candidate"), 0.25)
        self.assertEqual(watch.multiplier("candidate", is_reg=True), 1)

    def test_hard_improving_is_not_throttled(self):
        watch = controller()
        for e, loss in enumerate([20, 18, 16, 14, 12, 10, 8, 6]):
            report = epoch(watch, e, candidate=loss)
        row = report["images"]["candidate"]
        self.assertTrue(row["improving"])
        self.assertEqual(row["verdict"], "learning")
        self.assertEqual(watch.multiplier("candidate"), 1)

    def test_previously_improved_plateau_is_exhausted(self):
        watch = controller()
        for e, loss in enumerate([20, 18, 16, 14, 12, 10, 8, 6] + [6] * 12):
            report = epoch(watch, e, candidate=loss)
        self.assertEqual(report["images"]["candidate"]["verdict"], "exhausted")
        self.assertEqual(watch.multiplier("candidate"), 0.6)

    def test_confirmed_stuck_releases_only_after_clear_votes(self):
        watch = controller()
        for e in range(7):
            epoch(watch, e)
        for e in range(7, 15):
            before = watch.state_dict()["images"]["candidate"]
            row = epoch(watch, e, candidate=0.01)["images"]["candidate"]
            after = watch.state_dict()["images"]["candidate"]
            if before["confirmed"] and not after["confirmed"]:
                self.assertEqual(before["clear_votes"], 2)
                self.assertNotEqual(row["verdict"], "stuck")
                break
        else:
            self.fail("stuck image never released")

    def test_missing_images_do_not_receive_stale_votes(self):
        watch = controller()
        epoch(watch, 0)
        epoch(watch, 1)
        previous = watch.state_dict()["images"]["candidate"]
        for e in range(2, 9):
            watch.record("ordinary-0", e, 0.25, 1)
            watch.end_epoch(e)
        after = watch.state_dict()["images"]["candidate"]
        for name in (
            "stuck_votes",
            "suspect_votes",
            "exhaust_votes",
            "stuck_epochs",
            "multiplier",
        ):
            self.assertEqual(previous[name], after[name])

    def test_normalization_tracks_noise_not_raw_loss(self):
        watch = controller(bins=2)
        watch.bind_dataset(KEYS + KEYS)  # Repeats refer to the same source image.
        for e in range(8):
            for i, key in enumerate(KEYS):
                # Each key sees a different noise band on alternating epochs.
                sigma = 0.25 if (i + e) % 2 else 0.75
                watch.record(key, e, sigma, 1 if sigma < 0.5 else 100)
            report = watch.end_epoch(e)
        for row in report["images"].values():
            self.assertAlmostEqual(row["mean_residual"], 0)
            self.assertNotIn(row["verdict"], ("stuck", "suspect", "exhausted"))
        self.assertEqual(report["bin_counts"], [56, 56])

    def test_degenerate_residual_fences_never_make_early_clean_cohort_suspect(self):
        watch = controller(bins=2)
        for e in range(6):
            for i, key in enumerate(KEYS):
                sigma = 0.25 if (i + e) % 2 else 0.75
                watch.record(key, e, sigma, 1 if sigma < 0.5 else 100)
            report = watch.end_epoch(e)
            for row in report["images"].values():
                self.assertEqual(row["mean_residual"], 0)
                self.assertNotIn(row["verdict"], ("suspect", "stuck", "exhausted"))
                self.assertGreaterEqual(row["suggested_multiplier"], 1)

    def test_compact_aggregates_match_raw_epoch_residuals(self):
        watch = controller(bins=4)
        records = []
        for e in range(35):
            for i, key in enumerate(KEYS):
                for repeat in range(1 + (e + i) % 3):
                    sigma = ((e * 7 + i * 3 + repeat) % 4 + 0.5) / 4
                    loss = 2 + i * 0.07 + sigma * 5 + math.sin(e + i) * 0.2
                    watch.record(key, e, sigma, loss)
                    records.append((key, e, int(sigma * 4), loss))
            report = watch.end_epoch(e)
        means = [
            sum(r[3] for r in records if r[2] == b) / sum(r[2] == b for r in records)
            for b in range(4)
        ]
        for key in KEYS:
            series = []
            for e in range(35):
                raw = [r[3] - means[r[2]] for r in records if r[0] == key and r[1] == e]
                series.append(sum(raw) / len(raw))
            row = report["images"][key]
            self.assertAlmostEqual(row["mean_residual"], sum(series[-5:]) / 5)
            self.assertAlmostEqual(row["baseline"], sum(series[:3]) / 3)
            self.assertAlmostEqual(row["first"], sum(series[-8:-4]) / 4)
            self.assertAlmostEqual(row["last"], sum(series[-4:]) / 4)
            self.assertLessEqual(len(watch.state_dict()["images"][key]["history"]), 11)

    def test_resume_preserves_pending_observations_and_exact_history(self):
        watch = controller()
        for e in range(14):
            epoch(watch, e)
        watch.record("candidate", 14, 0.25, 5)
        resumed = controller()
        resumed.load_state_dict(
            json.loads(json.dumps(watch.state_dict(), allow_nan=False))
        )
        self.assertEqual(watch.state_dict(), resumed.state_dict())
        self.assertEqual(watch.report(), resumed.report())
        for e in range(14, 25):
            self.assertEqual(
                epoch(watch, e, candidate=5), epoch(resumed, e, candidate=5)
            )
            self.assertEqual(watch.state_dict(), resumed.state_dict())

    def test_invalid_observations_are_atomic(self):
        watch = controller()
        for sigma, loss in [
            (math.nan, 1),
            (0.5, math.inf),
            (0.5, -1),
            (-0.1, 1),
            (1.1, 1),
            (True, 1),
        ]:
            previous = watch.state_dict()
            self.assertFalse(watch.record("candidate", 0, sigma, loss))
            self.assertEqual(previous, watch.state_dict())

    def test_load_rejects_mismatch_and_corruption_transactionally(self):
        watch = controller()
        epoch(watch, 0)
        original = watch.state_dict()
        variants = []
        bad = copy.deepcopy(original)
        bad["config"]["mode"] = "observe"
        variants.append(bad)
        bad = copy.deepcopy(original)
        bad["dataset_keys"] = ["changed"]
        variants.append(bad)
        bad = copy.deepcopy(original)
        bad["images"]["easy"]["history"][0][1][0][1] = math.nan
        variants.append(bad)
        bad = copy.deepcopy(original)
        bad["bin_counts"][0] = -1
        variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                watch.load_state_dict(bad)
            self.assertEqual(original, watch.state_dict())

    def test_modes_and_boundaries(self):
        off, observe, adaptive = controller("off"), controller("observe"), controller()
        before = off.state_dict()
        for e in range(8):
            epoch(off, e)
            epoch(observe, e)
            epoch(adaptive, e)
        self.assertEqual(before, off.state_dict())
        self.assertEqual(observe.multiplier("candidate"), 1)
        self.assertEqual(
            observe.report()["images"]["candidate"]["suggested_multiplier"], 0.25
        )
        self.assertEqual(adaptive.multiplier("candidate"), 0.25)
        state = adaptive.state_dict()
        adaptive.end_epoch(7)
        self.assertEqual(adaptive.state_dict(), state)
        with self.assertRaises(ValueError):
            adaptive.record("candidate", 7, 0.5, 1)
        with self.assertRaises(ValueError):
            off.load_state_dict(observe.state_dict())

    def test_strict_configuration(self):
        for options in (
            {"mode": "wrong"},
            {"bins": True},
            {"surprise": 1},
            {"easy_mult": math.nan},
            {"lo_q": 0.8, "hi_q": 0.5},
            {"trend_window": 5},
            {"stuck_floor": 0},
        ):
            with self.assertRaises(ValueError):
                PerImageAdaptive(options)


if __name__ == "__main__":
    unittest.main()
