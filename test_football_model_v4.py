"""Exclusion, chronology and independently fitted V4 probability checks."""
import copy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import football_model_v3 as v3
import football_model_v4 as v4
from test_football_model_v2 import matches


def columns(frame):
    return [column for column in frame if column not in v3.base.NON_FEATURES]


class V4FeaturesTests(unittest.TestCase):
    def test_forbidden_statistics_and_aliases_do_not_change_any_feature(self):
        original = matches()
        changed = original.copy(deep=True)
        for side in ("home", "away"):
            changed[f"{side}_stats"] = changed[f"{side}_stats"].map(lambda stats: {
                **stats, "possession": 99., "passes": 9999., "passes_completed": 9000.,
                "pass_accuracy": 97., "Ball Possession": "99%", "Total passes": 9999,
                "Passes accurate": 9000, "Passes %": "97%"})
        before, _ = v4.build_features(original)
        after, state = v4.build_features(changed)
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(all(v4.allowed_feature(column) for column in columns(after)))
        for team in state.teams.values():
            for scope in team["history"].values():
                self.assertTrue(all(all(v4.allowed_feature(key) for key in item) for item in scope))

    def test_features_match_v3_projection_without_mutating_v3(self):
        stats_before = list(v3.base.STATS)
        values_before = list(v3.base.VALUES)
        original, _ = v3.build_features(matches())
        actual, _ = v4.build_features(matches())
        projected = original[[column for column in original if v4.allowed_feature(column)]]
        pd.testing.assert_frame_equal(actual, projected)
        self.assertEqual(len(columns(original)) - len(columns(actual)), 10)
        self.assertEqual(v3.base.STATS, stats_before)
        self.assertEqual(v3.base.VALUES, values_before)

    def test_relevant_historical_statistics_still_change_features(self):
        original = matches()
        changed = original.copy(deep=True)
        changed.at[0, "home_stats"] = {**original.iloc[0].home_stats,
                                       "xg": 12., "shots_on_target": 50.}
        before, _ = v4.build_features(original)
        after, _ = v4.build_features(changed)
        self.assertFalse(before.loc[before.fixture_id == 4, columns(before)].equals(
            after.loc[after.fixture_id == 4, columns(after)]))

    def test_current_future_and_in_progress_results_do_not_enter_inputs(self):
        original = matches()
        changed = original.copy(deep=True)
        for index in (3, 4):
            changed.at[index, "home_goals"] = 20
            changed.at[index, "away_goals"] = 15
            changed.at[index, "home_stats"] = {"xg": 30., "shots_on_target": 100.}
        before, _ = v4.build_features(original)
        after, _ = v4.build_features(changed)
        pd.testing.assert_frame_equal(before.loc[before.fixture_id <= 4, columns(before)],
                                      after.loc[after.fixture_id <= 4, columns(after)])
        np.testing.assert_array_equal(before.home_log_games.iloc[:3], 0.)

    def test_asof_preserves_result_availability_delay(self):
        rows = matches()
        cutoff = rows.kickoff.min() + pd.Timedelta(hours=4)
        limited, state = v4.build_features(rows, cutoff)
        self.assertEqual(limited.fixture_id.tolist(), [1, 2])
        self.assertEqual(state.teams[("test", 10)]["games"], 2)


class V4BundleTests(unittest.TestCase):
    def test_fit_rejects_possession_or_pass_columns(self):
        for name in ("home_all_possession_5", "diff_pass_accuracy_20", "passes_completed"):
            with self.assertRaises(ValueError):
                v4.fit_bundle(pd.DataFrame(), [name], [], {}, {}, pd.Timestamp("2025-01-01T00:00:00Z"))

    def test_wrong_version_bundle_is_rejected(self):
        with self.assertRaises(ValueError):
            v4.predict_bundle({"kind": v3.KIND}, pd.DataFrame())

    def test_independently_fitted_serialized_bundle_and_goal_probabilities(self):
        frame, _ = v4.build_features(matches())
        frame = pd.concat([frame] * 25, ignore_index=True)
        feature_columns = columns(frame)
        base_columns = [column for column in feature_columns if not column.startswith("v3_")]
        selection = {"components": [
            {"family": "boosting", "feature_set": "base", "half_life_days": 730},
            {"family": "goals", "alpha": 1., "shared": False, "rho": -.04}],
            "weights": [.25, .75], "calibration_method": "none"}
        reference = {"components": [{"family": "logistic", "C": .01}], "weights": [1.]}
        cutoff = pd.Timestamp("2025-01-01T00:00:00Z")
        original_selection = copy.deepcopy(selection)
        with threadpool_limits(limits=1):
            bundle = v4.fit_bundle(frame, feature_columns, base_columns, selection, reference, cutoff)
            probabilities = v4.predict_bundle(bundle, frame)
            totals = v4.total_goal_probabilities(bundle, frame, (3, 4))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "v4.joblib"
                joblib.dump(bundle, path)
                loaded = joblib.load(path)
                np.testing.assert_allclose(v4.predict_bundle(loaded, frame), probabilities)
                np.testing.assert_allclose(v4.total_goal_probabilities(loaded, frame, (3, 4)), totals)
        self.assertEqual(bundle["kind"], v4.KIND)
        self.assertEqual(selection, original_selection)
        self.assertEqual(len(base_columns), 157)
        self.assertEqual(len(feature_columns), 195)
        np.testing.assert_allclose(probabilities.sum(1), 1.)
        self.assertTrue(np.isfinite(totals).all())
        self.assertTrue(((totals >= 0) & (totals <= 1)).all())
        self.assertTrue((totals[:, 0] >= totals[:, 1]).all())
        modified = frame.assign(home_all_possession_5=100., passes=99999.)
        np.testing.assert_allclose(v4.predict_bundle(bundle, modified), probabilities)
        np.testing.assert_allclose(v4.total_goal_probabilities(bundle, modified, (3, 4)), totals)


if __name__ == "__main__":
    unittest.main()
