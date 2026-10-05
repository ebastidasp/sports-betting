"""Independent chronology, feature and probability checks for outcome model v3."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam
from threadpoolctl import threadpool_limits

import football_model_v3 as m
from test_football_model_v2 import matches


def columns(frame):
    return [column for column in frame if column not in m.base.NON_FEATURES]


def frames():
    old, _ = m.base.build_features(matches())
    new, _ = m.build_features(matches())
    return old, new


class OutcomeV3HistoryTests(unittest.TestCase):
    def test_current_and_future_goals_and_statistics_cannot_change_inputs(self):
        original = matches()
        changed = original.copy(deep=True)
        for index in (3, 4):
            changed.at[index, "home_goals"] = 20
            changed.at[index, "away_goals"] = 15
            changed.at[index, "home_stats"] = {"xg": 30., "shots_on_target": 100.}
            changed.at[index, "away_stats"] = {"xg": 20., "shots_on_target": 90.}
        before, _ = m.build_features(original)
        after, _ = m.build_features(changed)
        features = columns(before)
        pd.testing.assert_frame_equal(before.loc[before.fixture_id <= 4, features],
                                      after.loc[after.fixture_id <= 4, features])
        self.assertFalse(before.loc[before.fixture_id == 5, features]
                         .equals(after.loc[after.fixture_id == 5, features]))

    def test_simultaneous_and_in_progress_results_do_not_enter_features(self):
        rows = matches()
        frame, _ = m.build_features(rows)
        np.testing.assert_array_equal(frame.home_log_games.iloc[:3], 0.)
        np.testing.assert_array_equal(frame.away_log_games.iloc[:3], 0.)
        self.assertAlmostEqual(frame.home_log_games.iloc[3], np.log1p(3))
        changed = rows.copy(deep=True)
        changed.at[0, "home_goals"] = 30
        after, _ = m.build_features(changed)
        pd.testing.assert_frame_equal(frame.loc[frame.fixture_id <= 3, columns(frame)],
                                      after.loc[after.fixture_id <= 3, columns(frame)])

    def test_exact_three_hour_result_availability_boundary(self):
        rows = matches()
        rows.at[3, "kickoff"] = rows.kickoff.min() + pd.Timedelta(hours=3)
        frame, _ = m.build_features(rows)
        self.assertAlmostEqual(frame.loc[frame.fixture_id == 4, "home_log_games"].iloc[0],
                               np.log1p(2))

    def test_asof_excludes_unavailable_results_and_state(self):
        rows = matches()
        cutoff = rows.kickoff.min() + pd.Timedelta(hours=4)
        limited, state = m.build_features(rows, as_of=cutoff)
        full, _ = m.build_features(rows)
        self.assertEqual(limited.fixture_id.tolist(), [1, 2])
        self.assertEqual(state.teams[("test", 10)]["games"], 2)
        self.assertEqual(len(state.extra_teams[("test", 10)]["history"]), 2)
        pd.testing.assert_frame_equal(limited, full.loc[full.fixture_id.isin([1, 2])])

    def test_all_legacy_features_and_labels_match_v2(self):
        old, new = frames()
        pd.testing.assert_frame_equal(new[old.columns], old)
        self.assertTrue(any(column.startswith("v3_") for column in new))

    def test_same_time_input_order_does_not_change_features(self):
        forward, _ = m.build_features(matches())
        reversed_frame, _ = m.build_features(matches().iloc[::-1])
        pd.testing.assert_frame_equal(forward, reversed_frame)

    def test_missing_extra_stats_do_not_become_zero_observations(self):
        rows = matches()
        rows["home_stats"] = [{"xg": None, "shots_on_target": float("nan")} for _ in rows.index]
        rows["away_stats"] = [{"xg": -1., "shots_on_target": None} for _ in rows.index]
        frame, state = m.build_features(rows)
        extra = state.extra_teams[("test", 10)]
        for half in m.base.HALFLIVES:
            self.assertEqual(extra["moments"][half]["xg_residual"][1], 0.)
            self.assertEqual(extra["moments"][half]["sot_residual"][1], 0.)
            self.assertTrue((frame[f"v3_home_xg_residual_coverage_{half}"] == 0).all())
        numeric = frame[[column for column in columns(frame) if column != "league"]]
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())
        self.assertEqual(len(frame), len(rows))

    def test_actual_zero_extra_stats_are_valid_observations(self):
        rows = matches()
        rows.at[0, "home_stats"] = {"xg": 0., "shots_on_target": 0.}
        frame, state = m.build_features(rows)
        for half in m.base.HALFLIVES:
            self.assertGreater(state.extra_teams[("test", 10)]["moments"][half]["xg_residual"][1], 0.)
            self.assertGreater(frame.loc[frame.fixture_id == 4,
                                         f"v3_home_xg_residual_coverage_{half}"].iloc[0], 0.)

    def test_extra_incremental_smoothing_matches_explicit_capped_history(self):
        rows = pd.concat([matches()] * 30, ignore_index=True)
        rows["fixture_id"] = np.arange(len(rows))
        rows["kickoff"] = pd.date_range("2023-01-01", periods=len(rows), freq="D", tz="UTC")
        for index in rows.index:
            rows.at[index, "home_stats"] = ({} if index % 3 == 0 else
                                            {"xg": 1.2 + index % 4 / 10,
                                             "shots_on_target": float(index % 6)})
        _, state = m.build_features(rows)
        extra = state.extra_teams[("test", 10)]
        self.assertEqual(len(extra["history"]), 120)
        for half in m.base.HALFLIVES:
            for key in m.EXTRA_VALUES:
                observations = [(age, item[key]) for age, item in enumerate(reversed(extra["history"]))
                                if key in item]
                mass = sum(2 ** (-age / half) for age, _ in observations)
                total = sum(2 ** (-age / half) * value for age, value in observations)
                actual_total, actual_mass = extra["moments"][half][key]
                self.assertAlmostEqual(actual_total, total, places=9)
                self.assertAlmostEqual(actual_mass, mass, places=9)

    def test_targets_identifiers_and_postmatch_fields_are_excluded(self):
        _, frame = frames()
        features = set(columns(frame))
        self.assertFalse(features & {"fixture_id", "season", "kickoff", "league_id", "primary_league",
                                     "country", "home_name", "away_name", "home_goals", "away_goals", "target"})
        self.assertFalse(any(column.startswith("target") for column in features))


class OutcomeV3PooledTests(unittest.TestCase):
    def test_pooling_keeps_home_then_away_order_and_reverses_team_roles(self):
        frame = pd.DataFrame({
            "league": ["test:1", "test:2"], "league_home_goals": [1.5, 1.7],
            "league_away_goals": [1., 1.2], "home_all_goals_5": [2., 3.],
            "away_all_goals_5": [.5, .8], "home_home_conceded_20": [.7, .9],
            "away_away_conceded_20": [1.8, 2.], "elo_diff_10": [.2, -.4],
            "diff_goals_5": [1.5, 2.2]})
        pooled = m.pooled_inputs(frame, list(frame))
        np.testing.assert_array_equal(pooled.is_home, [1., 1., 0., 0.])
        np.testing.assert_allclose(pooled.own_all_goals_5, [2., 3., .5, .8])
        np.testing.assert_allclose(pooled.opponent_all_goals_5, [.5, .8, 2., 3.])
        np.testing.assert_allclose(pooled.own_venue_conceded_20, [.7, .9, 1.8, 2.])
        np.testing.assert_allclose(pooled.opponent_venue_conceded_20, [1.8, 2., .7, .9])
        np.testing.assert_allclose(pooled.elo_diff_10, [.2, -.4, -.2, .4])
        np.testing.assert_allclose(pooled.diff_goals_5, [1.5, 2.2, -1.5, -2.2])
        np.testing.assert_allclose(pooled.league_own_goals, [1.5, 1.7, 1., 1.2])

    def test_shared_goal_rate_prediction_unpacks_both_team_blocks(self):
        old, frame = frames()
        frame = frame.iloc[:2]
        base_columns = columns(old)
        pool = m.pooled_inputs(frame, base_columns)
        with patch.object(m, "pooled_inputs", return_value=pool):
            model = unittest.mock.Mock()
            model.predict.return_value = np.array([1., 2., 3., 4.])
            home, away = m.goal_rates({"model": model, "columns": list(pool)},
                                     frame, base_columns, {"shared": True})
        np.testing.assert_array_equal(home, [1., 2.])
        np.testing.assert_array_equal(away, [3., 4.])


class OutcomeV3ProbabilityTests(unittest.TestCase):
    def test_zero_dixon_coles_coefficient_matches_independent_poisson(self):
        home, away = np.array([.05, 1.4, 8.]), np.array([8., 1.1, .05])
        expected = np.column_stack([skellam.sf(0, home, away), skellam.pmf(0, home, away),
                                    skellam.cdf(-1, home, away)])
        np.testing.assert_allclose(m.goal_probabilities(home, away, rho=0.), expected,
                                   atol=1e-11)

    def test_dixon_coles_low_score_adjustment_conserves_probability(self):
        home, away, rho = np.array([1.4, 2.]), np.array([1.1, .9]), -.1
        original = m.goal_probabilities(home, away)
        adjusted = m.goal_probabilities(home, away, rho)
        delta = np.exp(-home - away) * home * away * rho
        np.testing.assert_allclose(adjusted - original,
                                   np.column_stack([delta, -2 * delta, delta]), atol=1e-12)
        np.testing.assert_allclose(adjusted.sum(1), 1.)
        self.assertTrue((adjusted[:, 1] > original[:, 1]).all())

    def test_extreme_goal_rates_and_rho_remain_finite_and_normalized(self):
        home = np.array([1e-6, .05, .5, 1., 8., 30., 300.])
        away = np.array([.01, 8., 1., 1., .05, 40., 300.])
        for rho in (-100., -.2, 0., .2, 100.):
            p = m.goal_probabilities(home, away, rho)
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p > 0) & (p <= 1)).all())
            np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)

    def test_invalid_goal_rates_are_rejected(self):
        for home, away in (([0.], [1.]), ([-1.], [1.]), ([float("nan")], [1.]),
                           ([1.], [float("inf")]), ([1., 2.], [1.])):
            with self.assertRaises(ValueError):
                m.goal_probabilities(home, away)

    def test_rho_fit_recovers_known_low_score_dependence(self):
        home_rate, away_rate, rho = 1.2, .9, -.12
        grid = np.arange(16)
        h, a = np.meshgrid(grid, grid, indexing="ij")
        mass = poisson.pmf(h, home_rate) * poisson.pmf(a, away_rate)
        mass[0, 0] *= 1 - home_rate * away_rate * rho
        mass[0, 1] *= 1 + home_rate * rho
        mass[1, 0] *= 1 + away_rate * rho
        mass[1, 1] *= 1 - rho
        rng = np.random.default_rng(42)
        draws = rng.choice(mass.size, size=15000, p=mass.ravel() / mass.sum())
        frame = pd.DataFrame({"home_goals": h.ravel()[draws], "away_goals": a.ravel()[draws]})
        estimated = m.fit_rho(frame, np.full(len(frame), home_rate), np.full(len(frame), away_rate))
        self.assertAlmostEqual(estimated, rho, delta=.035)

    def test_calibration_normalizes_rows_and_identity_preserves_probabilities(self):
        p = np.array([[.6, .25, .15], [.05, .1, .85], [.1, .8, .1]])
        np.testing.assert_allclose(m.calibrate(p, [0., 0., 0.]), p)
        adjusted = m.calibrate(7 * p, [.4, .2, -.1])
        self.assertTrue(np.isfinite(adjusted).all())
        self.assertTrue((adjusted > 0).all())
        np.testing.assert_allclose(adjusted.sum(1), 1.)

    def test_positive_temperature_preserves_argmax(self):
        p = np.array([[.6, .25, .15], [.05, .1, .85], [.1, .8, .1]])
        for log_temperature in (-3., -.2, .4, 3.):
            calibrated = m.calibrate(p, [log_temperature, 0., 0.])
            np.testing.assert_array_equal(calibrated.argmax(1), p.argmax(1))
            np.testing.assert_allclose(calibrated.sum(1), 1.)

    def test_fitted_temperature_keeps_bias_zero_and_improves_loss(self):
        p = np.tile([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]], (20, 1))
        y = np.tile([0, 2, 2], 20)
        parameters = m.fit_calibration(p, y, "temperature")
        self.assertEqual(parameters[1:], [0., 0.])
        q = m.calibrate(p, parameters)
        self.assertLessEqual(-np.log(q[np.arange(len(y)), y]).mean(),
                             -np.log(p[np.arange(len(y)), y]).mean())
        np.testing.assert_array_equal(q.argmax(1), p.argmax(1))
        self.assertEqual(m.fit_calibration(p, y, "none"), [0., 0., 0.])

    def test_invalid_probability_rows_are_rejected(self):
        for p in (np.ones((3, 2)), np.array([[0., 0., 0.]]),
                  np.array([[.8, -.1, .3]]), np.array([[float("nan"), .4, .6]])):
            with self.assertRaises(ValueError):
                m.calibrate(p, [0., 0., 0.])


class OutcomeV3BundleTests(unittest.TestCase):
    def test_tiny_three_class_bundle_round_trip_preserves_predictions(self):
        old, frame = frames()
        frame = pd.concat([frame] * 15, ignore_index=True)
        selection = {"components": [{"family": "logistic", "C": .01}], "weights": [1.]}
        reference = {"components": [{"family": "logistic", "C": .01}], "weights": [1.]}
        cutoff = pd.Timestamp("2024-01-01T00:00:00Z")
        with threadpool_limits(limits=1):
            bundle = m.fit_bundle(frame, columns(frame), columns(old), selection, reference,
                                  cutoff, parameters=[.3, 0., 0.])
            p = m.predict_bundle(bundle, frame)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "football_v3.joblib"
                joblib.dump(bundle, path)
                loaded_p = m.predict_bundle(joblib.load(path), frame)
        np.testing.assert_allclose(loaded_p, p)
        self.assertEqual(p.shape, (len(frame), 3))
        np.testing.assert_allclose(p.sum(1), 1.)
        self.assertTrue((p > 0).all())

    def test_v2_fallback_reproduces_reference_probabilities(self):
        old, frame = frames()
        frame = pd.concat([frame] * 15, ignore_index=True)
        reference = {"components": [{"family": "logistic", "C": .01}], "weights": [1.]}
        selection = {"components": [{"family": "v2"}], "weights": [1.]}
        cutoff = pd.Timestamp("2024-01-01T00:00:00Z")
        with threadpool_limits(limits=1):
            old_bundle = m.base.fit_bundle(frame, columns(old), reference)
            new_bundle = m.fit_bundle(frame, columns(frame), columns(old), selection, reference, cutoff)
            expected = m.base.predict_bundle(old_bundle, frame)
            actual = m.predict_bundle(new_bundle, frame)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


class OutcomeV3TotalGoalsTests(unittest.TestCase):
    @staticmethod
    def score_grid_events(home, away, minimum, rho=0.):
        """Sum actual scoreline masses, independently of a total-count SF."""
        grid = np.arange(61)
        hg, ag = np.meshgrid(grid, grid, indexing="ij")
        result = []
        for home_rate, away_rate in zip(home, away):
            mass = poisson.pmf(hg, home_rate) * poisson.pmf(ag, away_rate)
            mass[0, 0] *= 1 - home_rate * away_rate * rho
            mass[0, 1] *= 1 + home_rate * rho
            mass[1, 0] *= 1 + away_rate * rho
            mass[1, 1] *= 1 - rho
            result.append([mass[hg + ag >= value].sum() for value in minimum])
        return np.asarray(result)

    @staticmethod
    def bundle(specs, weights):
        return {"models": [object() for _ in specs], "base_columns": ["league"],
                "selection": {"components": specs, "weights": weights}}

    def test_event_probabilities_match_explicit_dixon_coles_score_grid(self):
        home, away = np.array([.4, 1.4, 2.8]), np.array([.7, 1.1, 1.8])
        frame = pd.DataFrame({"league": ["test:1"] * len(home)})
        for rho in (-.08, 0., .08):
            bundle = self.bundle([{"family": "goals", "rho": rho}], [1.])
            with patch.object(m, "goal_rates", return_value=(home, away)):
                actual = m.total_goal_probabilities(bundle, frame)
            expected = self.score_grid_events(home, away, (2, 3, 4), rho)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())
            self.assertTrue((np.diff(actual, axis=1) <= 0).all())

    def test_threshold_output_follows_requested_order(self):
        home, away = np.array([1.4]), np.array([1.1])
        frame = pd.DataFrame({"league": ["test:1"]})
        bundle = self.bundle([{"family": "goals", "rho": -.08}], [1.])
        requested = (4, 2, 3, 2)
        with patch.object(m, "goal_rates", return_value=(home, away)):
            actual = m.total_goal_probabilities(bundle, frame, minimum_goals=requested)
        np.testing.assert_allclose(actual, self.score_grid_events(home, away, requested, -.08), atol=1e-12)
        self.assertEqual(actual.shape, (1, len(requested)))
        self.assertEqual(actual[0, 1], actual[0, 3])

    def test_goal_components_mix_probabilities_and_exclude_classifier_weight(self):
        frame = pd.DataFrame({"league": ["test:1", "test:1"]})
        h1, a1 = np.array([.3, 4.]), np.array([.4, 3.])
        h2, a2 = np.array([4., .3]), np.array([3., .4])
        bundle = self.bundle([{"family": "goals"}, {"family": "logistic"},
                              {"family": "goals"}], [.25, .25, .5])
        with patch.object(m, "goal_rates", side_effect=[(h1, a1), (h2, a2)]) as rates_mock:
            actual = m.total_goal_probabilities(bundle, frame)
        expected = (self.score_grid_events(h1, a1, (2, 3, 4))
                    + 2 * self.score_grid_events(h2, a2, (2, 3, 4))) / 3
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertEqual(rates_mock.call_count, 2)
        averaged_means = self.score_grid_events((h1 + 2 * h2) / 3,
                                               (a1 + 2 * a2) / 3, (2, 3, 4))
        self.assertGreater(np.max(np.abs(actual - averaged_means)), .03)

    def test_single_goal_component_with_75_percent_weight_is_unscaled(self):
        home, away = np.array([1.8]), np.array([.9])
        frame = pd.DataFrame({"league": ["test:1"]})
        bundle = self.bundle([{"family": "v2"}, {"family": "goals", "rho": -.1}], [.25, .75])
        with patch.object(m, "goal_rates", return_value=(home, away)) as rates_mock:
            actual = m.total_goal_probabilities(bundle, frame)
        np.testing.assert_allclose(actual, self.score_grid_events(home, away, (2, 3, 4), -.1), atol=1e-12)
        self.assertEqual(rates_mock.call_count, 1)

    def test_bundle_without_positive_goal_component_returns_none(self):
        frame = pd.DataFrame({"league": ["test:1"]})
        for specs, weights in (([{"family": "v2"}], [1.]),
                               ([{"family": "goals"}, {"family": "logistic"}], [0., 1.])):
            with patch.object(m, "goal_rates") as rates_mock:
                self.assertIsNone(m.total_goal_probabilities(self.bundle(specs, weights), frame))
            rates_mock.assert_not_called()

    def test_invalid_minimum_goal_counts_are_rejected(self):
        frame = pd.DataFrame({"league": ["test:1"]})
        bundle = self.bundle([{"family": "goals"}], [1.])
        for minimum in ((), (1,), (0,), (-2,), (2.5,), (float("nan"),),
                        (float("inf"),), [[2, 3]]):
            with self.assertRaises(ValueError):
                m.total_goal_probabilities(bundle, frame, minimum_goals=minimum)


if __name__ == "__main__":
    unittest.main()
