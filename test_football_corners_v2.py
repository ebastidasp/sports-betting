"""Safety and probability checks for the independent corners v2 model."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from threadpoolctl import threadpool_limits

import football_corners_v2 as m


def fixtures():
    start = pd.Timestamp("2023-01-01T12:00:00Z")
    return pd.DataFrame([
        {"country": "test", "league": "test:1", "league_id": 1,
         "primary_league": 1, "fixture_id": fixture_id, "season": 2023,
         "status": "FT", "kickoff": start + pd.Timedelta(hours=hours),
         "home_id": 10, "away_id": 20, "home_name": "Home", "away_name": "Away",
         "home_goals": 1, "away_goals": 0,
         "home_stats": {"corners": float(corners), "shots": 12.,
                        "passes": 500., "pass_accuracy": 85.},
         "away_stats": {"corners": 3., "shots": 8.,
                        "passes": 350., "pass_accuracy": 75.}}
        for fixture_id, hours, corners in (
            (1, 0, 5), (2, 0, 6), (3, 2, 4), (4, 6, 7), (5, 24, 8))
    ])


def feature_columns(frame):
    return [column for column in frame if column not in m.NON_FEATURES]


def inputs(frame, fixture_ids):
    return (frame.loc[frame.fixture_id.isin(fixture_ids), feature_columns(frame)]
            .reset_index(drop=True))


class CornerV2HistoryTests(unittest.TestCase):
    def test_current_and_future_results_cannot_change_prematch_inputs(self):
        original = fixtures()
        changed = original.copy(deep=True)
        for index in (3, 4):
            changed.at[index, "home_stats"] = {
                "corners": 40., "shots": 90., "passes": 1500., "pass_accuracy": 99.}
            changed.at[index, "away_stats"] = {
                "corners": 30., "shots": 80., "passes": 100., "pass_accuracy": 20.}
            changed.at[index, "home_goals"] = 8
            changed.at[index, "away_goals"] = 7
        before, _ = m.build_match_features(original)
        after, _ = m.build_match_features(changed)
        pd.testing.assert_frame_equal(inputs(before, [1, 2, 3, 4]),
                                      inputs(after, [1, 2, 3, 4]))
        self.assertFalse(inputs(before, [5]).equals(inputs(after, [5])))
        self.assertEqual(after.loc[after.fixture_id == 4, "target_total"].iloc[0], 70.)

    def test_simultaneous_and_still_unavailable_results_are_excluded(self):
        original = fixtures()
        changed = original.copy(deep=True)
        changed.at[0, "home_stats"] = {"corners": 100., "shots": 200.}
        changed.at[1, "away_stats"] = {"corners": 100., "shots": 200.}
        before, _ = m.build_match_features(original)
        after, _ = m.build_match_features(changed)
        pd.testing.assert_frame_equal(inputs(before, [1, 2, 3]),
                                      inputs(after, [1, 2, 3]))
        game_columns = [column for column in feature_columns(before)
                        if column.endswith("_games")]
        self.assertTrue(game_columns)
        self.assertTrue((before.loc[before.fixture_id <= 3, game_columns] == 0).all().all())
        self.assertFalse(inputs(before, [4]).equals(inputs(after, [4])))

    def test_results_become_available_at_three_hour_boundary(self):
        rows = fixtures()
        rows.at[3, "kickoff"] = rows.kickoff.min() + pd.Timedelta(hours=3)
        frame, _ = m.build_match_features(rows)
        game_columns = [column for column in feature_columns(frame)
                        if column.endswith("_games")]
        self.assertTrue(game_columns)
        np.testing.assert_allclose(frame.loc[frame.fixture_id == 4, game_columns],
                                   np.log1p(2))

    def test_asof_excludes_results_not_available_by_requested_time(self):
        rows = fixtures()
        cutoff = rows.kickoff.min() + pd.Timedelta(hours=4)
        limited, _ = m.build_match_features(rows, as_of=cutoff)
        full, _ = m.build_match_features(rows)
        self.assertEqual(set(limited.fixture_id), {1, 2})
        pd.testing.assert_frame_equal(inputs(limited, [1, 2]), inputs(full, [1, 2]))

    def test_missing_corners_remain_unknown_and_zero_is_kept(self):
        rows = fixtures()
        rows.at[0, "home_stats"] = {"shots": 12.}
        rows.at[1, "home_stats"] = {"corners": 0.}
        frame, _ = m.build_match_features(rows)
        first = frame.loc[frame.fixture_id == 1].iloc[0]
        second = frame.loc[frame.fixture_id == 2].iloc[0]
        self.assertTrue(np.isnan(first.target_home))
        self.assertTrue(np.isnan(first.target_total))
        self.assertEqual(first.target_away, 3.)
        self.assertEqual(second.target_home, 0.)
        self.assertEqual(second.target_total, 3.)

    def test_metadata_and_targets_never_enter_model_features(self):
        frame, _ = m.build_match_features(fixtures())
        columns = set(feature_columns(frame))
        forbidden = {"fixture_id", "country", "kickoff", "season", "league_id",
                     "primary_league", "home_id", "away_id", "home_name", "away_name",
                     "home_goals", "away_goals", "home_stats", "away_stats", "status",
                     "target_home", "target_away", "target_total"}
        self.assertFalse(columns & forbidden)
        self.assertFalse(any(column.startswith("target") for column in columns))
        self.assertTrue(columns)

    def test_input_order_does_not_change_chronological_features(self):
        rows = fixtures()
        ordered, _ = m.build_match_features(rows)
        shuffled, _ = m.build_match_features(rows.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(ordered.reset_index(drop=True),
                                      shuffled.reset_index(drop=True))


class CornerV2ProbabilityTests(unittest.TestCase):
    def test_identity_adjustment_preserves_base_distribution(self):
        grid = np.arange(41)
        cuts = np.arange(3, 15)
        base = poisson.cdf(grid[None, :], np.array([4., 9., 15.])[:, None])
        adjusted = m.coherent_cdf(base, base[:, cuts], cuts)
        np.testing.assert_allclose(adjusted, base, atol=1e-10)

    def test_consistent_calibrated_cut_probabilities_are_preserved(self):
        grid = np.arange(41)
        cuts = np.arange(3, 15)
        base = poisson.cdf(grid[None, :], np.array([9., 11.])[:, None])
        calibrated = np.vstack([np.linspace(.08, .94, len(cuts)),
                                np.linspace(.03, .87, len(cuts))])
        cdf = m.coherent_cdf(base, calibrated, cuts)
        np.testing.assert_allclose(cdf[:, cuts], calibrated, atol=1e-10)
        self.assertEqual(cdf.shape, base.shape)
        self.assertTrue(np.isfinite(cdf).all())
        self.assertTrue(((cdf >= 0) & (cdf <= 1)).all())
        self.assertTrue((np.diff(cdf, axis=1) >= -1e-12).all())

    def test_crossing_cut_estimates_still_yield_normalized_distribution(self):
        grid = np.arange(41)
        cuts = np.arange(3, 15)
        base = poisson.cdf(grid[None, :], np.array([5., 10., 18.])[:, None])
        crossing = np.array([
            [.02, .12, .1, .25, .21, .5, .45, .7, .68, .9, .87, .95],
            [.01, .03, .06, .2, .18, .31, .5, .48, .65, .62, .8, .92],
            [.04, .03, .07, .06, .1, .12, .11, .25, .23, .4, .39, .7],
        ])
        cdf = m.coherent_cdf(base, crossing, cuts)
        pmf = np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)
        tail = 1 - cdf[:, -1]
        self.assertTrue(np.isfinite(cdf).all())
        self.assertTrue((pmf >= -1e-12).all())
        self.assertTrue((tail >= -1e-12).all())
        np.testing.assert_allclose(pmf.sum(axis=1) + tail, 1., atol=1e-12)

    def test_event_metrics_use_correct_survival_boundaries(self):
        y = np.array([0, 6, 7, 8, 9, 13, 20])
        means = np.array([5., 6., 7., 8., 9., 12., 15.])
        cdf = poisson.cdf(np.arange(41)[None, :], means[:, None])
        metrics = m.probability_metrics(y, cdf)
        briers = []
        for threshold in (7, 8, 9):
            probability = 1 - cdf[:, threshold - 1]
            np.testing.assert_allclose(probability, poisson.sf(threshold - 1, means))
            actual = (y >= threshold).astype(float)
            brier = np.mean((probability - actual) ** 2)
            log_loss = -np.mean(actual * np.log(probability)
                                + (1 - actual) * np.log1p(-probability))
            self.assertAlmostEqual(metrics["events"][str(threshold)]["brier"], brier)
            self.assertAlmostEqual(metrics["events"][str(threshold)]["log_loss"], log_loss)
            briers.append(brier)
        self.assertAlmostEqual(metrics["mean_brier_7_8_9"], np.mean(briers))

    def test_exact_count_log_loss_uses_pmf_and_scores_are_finite(self):
        y = np.array([0, 6, 7, 8, 9, 13, 20])
        means = np.array([5., 6., 7., 8., 9., 12., 15.])
        cdf = poisson.cdf(np.arange(41)[None, :], means[:, None])
        metrics = m.probability_metrics(y, cdf)
        expected_nll = -np.mean(poisson.logpmf(y, means))
        self.assertAlmostEqual(metrics["nll"], expected_nll)
        for key in ("rps", "nll", "mean_brier_7_8_9"):
            self.assertTrue(np.isfinite(metrics[key]))
            self.assertGreaterEqual(metrics[key], 0.)


class CornerV2BundleTests(unittest.TestCase):
    @staticmethod
    def bundle(reference=False):
        bundle = {
            "count_model": object(), "columns": ["league"],
            "selection": {"count": {"family": "poisson", "alpha": 1.},
                          "C": None, "ordinal_weight": 0.},
            "calibration": {"global": 0., "leagues": {}},
        }
        if reference:
            bundle["point_reference_model"] = object()
        return bundle

    def test_reference_point_mean_keeps_direct_probability_distribution(self):
        frame = pd.DataFrame({"league": ["test:1", "test:1"]})
        reference_inputs = pd.DataFrame({"team": ["H1", "A1", "H2", "A2"]})
        direct_mean = np.array([3., 8.])
        direct_cdf = poisson.cdf(np.arange(41)[None, :], direct_mean[:, None])
        bundle = self.bundle(reference=True)
        with patch.object(m, "predict_count", return_value=direct_mean), \
                patch.object(m, "distribution_cdf", return_value=direct_cdf) as cdf_mock, \
                patch.object(m.base, "predict_bundle", return_value=np.array([5., 7., 2., 4.])) as reference_mock:
            mean, cdf = m.predict_bundle(bundle, frame, reference_inputs=reference_inputs)
        np.testing.assert_allclose(mean, [12., 6.])
        np.testing.assert_allclose(cdf, direct_cdf)
        self.assertIs(cdf_mock.call_args.args[1], direct_mean)
        self.assertIs(reference_mock.call_args.args[0], bundle["point_reference_model"])
        self.assertIs(reference_mock.call_args.args[1], reference_inputs)

    def test_selected_reference_mean_requires_historical_team_inputs(self):
        frame = pd.DataFrame({"league": ["test:1"]})
        with patch.object(m, "predict_count", return_value=np.array([9.])), \
                patch.object(m, "distribution_cdf", return_value=np.zeros((1, 41))), \
                patch.object(m.base, "predict_bundle") as reference_mock:
            with self.assertRaises(ValueError):
                m.predict_bundle(self.bundle(reference=True), frame)
        reference_mock.assert_not_called()

    def test_selected_reference_mean_rejects_wrong_number_of_team_rows(self):
        frame = pd.DataFrame({"league": ["test:1", "test:1"]})
        reference_inputs = pd.DataFrame({"team": ["H1", "A1", "H2"]})
        with patch.object(m, "predict_count", return_value=np.array([9., 10.])), \
                patch.object(m, "distribution_cdf", return_value=np.zeros((2, 41))), \
                patch.object(m.base, "predict_bundle", return_value=np.array([3., 4., 5.])):
            with self.assertRaises(ValueError):
                m.predict_bundle(self.bundle(reference=True), frame,
                                 reference_inputs=reference_inputs)

    def test_tiny_fitted_bundle_survives_serialization(self):
        frame, _ = m.build_match_features(fixtures())
        frame = pd.concat([frame] * 25, ignore_index=True)
        selection = {"count": {"family": "poisson", "alpha": 1.},
                     "C": None, "ordinal_weight": 0.}
        calibration = {"global": 0., "leagues": {}}
        with threadpool_limits(limits=1):
            bundle = m.fit_bundle(frame, feature_columns(frame), selection, calibration)
            expected_mean, expected_cdf = m.predict_bundle(bundle, frame)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "corners_v2.joblib"
                joblib.dump(bundle, path)
                loaded_mean, loaded_cdf = m.predict_bundle(joblib.load(path), frame)
        np.testing.assert_allclose(loaded_mean, expected_mean)
        np.testing.assert_allclose(loaded_cdf, expected_cdf)
        self.assertTrue((loaded_mean > 0).all())
        self.assertEqual(loaded_cdf.shape, (len(frame), 41))
        self.assertTrue(np.isfinite(loaded_cdf).all())
        self.assertTrue((np.diff(loaded_cdf, axis=1) >= -1e-12).all())


if __name__ == "__main__":
    unittest.main()
