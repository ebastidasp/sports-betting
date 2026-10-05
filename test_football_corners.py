from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import football_corners as m


def fixtures():
    start = pd.Timestamp("2023-01-01T12:00:00Z")
    return pd.DataFrame([
        {"country": "test", "league": "test:1", "league_id": 1, "primary_league": 1,
         "fixture_id": i, "season": 2023, "kickoff": start + pd.Timedelta(hours=hours),
         "home_id": 10, "away_id": 20, "home_name": "H", "away_name": "A",
         "home_goals": 1, "away_goals": 0,
         "home_stats": {"corners": float(corners), "shots": 12.},
         "away_stats": {"corners": 3., "shots": 8.}}
        for i, hours, corners in ((1, 0, 5), (2, 0, 6), (3, 2, 4), (4, 6, 7), (5, 24, 8))])


class CornerTests(unittest.TestCase):
    def test_current_match_stats_cannot_change_its_inputs(self):
        original = fixtures()
        changed = original.copy(deep=True)
        changed.at[3, "home_stats"] = {"corners": 50., "shots": 80.}
        a, _ = m.build_features(original)
        b, _ = m.build_features(changed)
        columns = [c for c in a if c not in m.NON_FEATURES]
        pd.testing.assert_frame_equal(a.loc[a.fixture_id <= 4, columns], b.loc[b.fixture_id <= 4, columns])
        self.assertFalse(a.loc[a.fixture_id == 5, columns].equals(b.loc[b.fixture_id == 5, columns]))

    def test_simultaneous_and_ongoing_matches_excluded(self):
        frame, _ = m.build_features(fixtures())
        self.assertTrue((frame.loc[frame.fixture_id <= 3, "own_games"] == 0).all())
        self.assertTrue((frame.loc[frame.fixture_id == 4, "own_games"] == np.log1p(3)).all())

    def test_missing_corners_not_zero(self):
        rows = fixtures()
        rows.at[0, "home_stats"] = {"shots": 12.}
        frame, state = m.build_features(rows)
        self.assertTrue(np.isnan(frame.loc[(frame.fixture_id == 1) & (frame.venue == "home"), "target"].iloc[0]))
        self.assertEqual(state.leagues["test:1"]["home"][1], 4)
        self.assertEqual(state.leagues["test:1"]["home"][0], 25.)
        self.assertEqual(m.pair_predictions(frame, frame.league_corners.to_numpy()).target_total.notna().sum(), 4)

    def test_actual_zero_corner_count_is_kept(self):
        rows = fixtures()
        rows.at[0, "home_stats"] = {"corners": 0.}
        frame, state = m.build_features(rows)
        value = frame.loc[(frame.fixture_id == 1) & (frame.venue == "home"), "target"].iloc[0]
        self.assertEqual(value, 0)
        self.assertEqual(state.leagues["test:1"]["home"][1], 5)

    def test_asof_excludes_unavailable_results(self):
        rows = fixtures()
        _, state = m.build_features(rows, rows.kickoff.min() + pd.Timedelta(hours=4))
        self.assertEqual(state.teams[("test", 10)]["games"], 2)

    def test_aliases_and_invalid_values(self):
        s = m.clean_stats({"Corner Kicks": "5", "Ball Possession": "53%",
                           "Shots on Goal": None, "xg": "NaN", "corners_bad": -1})
        self.assertEqual(s, {"corners": 5., "possession": 53.})
        self.assertNotIn("corners", m.clean_stats({"corners": 2.5}))

    def test_no_target_in_model_inputs(self):
        frame, _ = m.build_features(fixtures())
        columns = [c for c in frame if c not in m.NON_FEATURES]
        self.assertFalse({"target", "fixture_id", "team_id", "season"} & set(columns))

    def test_pass_statistics_affect_future_inputs_only(self):
        original = fixtures()
        changed = original.copy(deep=True)
        for index in changed.index:
            changed.at[index, "home_stats"] = {**changed.at[index, "home_stats"],
                "passes": 10000., "passes_completed": 9999., "pass_accuracy": 99.99}
            changed.at[index, "away_stats"] = {**changed.at[index, "away_stats"],
                "passes": 1., "passes_completed": 0., "pass_accuracy": 0.}
        a, _ = m.build_features(original)
        b, _ = m.build_features(changed)
        columns = [c for c in a if c not in m.NON_FEATURES]
        pass_columns = [c for c in columns if "pass" in c]
        self.assertTrue(pass_columns)
        pd.testing.assert_frame_equal(a.loc[a.fixture_id <= 3, columns], b.loc[b.fixture_id <= 3, columns])
        self.assertFalse(a.loc[a.fixture_id == 4, pass_columns].equals(b.loc[b.fixture_id == 4, pass_columns]))
        self.assertEqual(m.clean_stats({"Total passes": 500, "Passes accurate": 400,
            "Passes %": "80%", "passes": 1000, "pass_accuracy": 90, "corners": 5}),
            {"passes": 1000., "passes_completed": 400., "pass_accuracy": 90., "corners": 5.})

    def test_count_distribution_and_integer_lines(self):
        for alpha in (0., .2):
            dist = m.count_distribution(10., alpha)
            self.assertAlmostEqual(dist.mean(), 10.)
            self.assertAlmostEqual(dist.cdf(9) + dist.sf(10) + dist.pmf(10), 1.)
            self.assertGreaterEqual(dist.var(), 10.)
        self.assertAlmostEqual(m.dispersion(np.array([5., 5.]), np.array([5., 5.])), 0.)

    def test_model_saved_and_loaded_includes_both_teams(self):
        frame, _ = m.build_features(fixtures())
        frame = pd.concat([frame] * 15, ignore_index=True)
        columns = [c for c in frame if c not in m.NON_FEATURES]
        selection = {"components": [{"family": "poisson", "alpha": 1.}], "weights": [1.]}
        with threadpool_limits(limits=1):
            bundle = m.fit_bundle(frame, columns, selection)
            pred = m.predict_bundle(bundle, frame)
        self.assertTrue((pred > 0).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(bundle, path)
            np.testing.assert_allclose(m.predict_bundle(joblib.load(path), frame), pred)

    def test_read_only_sqlite_and_api_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.sqlite3"
            with closing(sqlite3.connect(path)) as con, con:
                con.execute("CREATE TABLE app_config (key TEXT, value TEXT)")
                con.execute("INSERT INTO app_config VALUES (?, ?)", ("country_divisions", json.dumps({"country": "test", "league_ids": [1]})))
                con.execute("CREATE TABLE fixtures (fixture_id INT, league_id INT, season INT, kickoff TEXT, status TEXT, home_id INT, home_name TEXT, away_id INT, away_name TEXT, home_goals INT, away_goals INT)")
                con.execute("INSERT INTO fixtures VALUES (1,1,2023,'2023-01-01T12:00:00Z','FT',10,'H',20,'A',1,0)")
                con.execute("CREATE TABLE team_stats (fixture_id INT, team_id INT, stats_json TEXT)")
                con.execute("INSERT INTO team_stats VALUES (1,10,?)", (json.dumps({"Corner Kicks": 7}),))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows, _ = m.read_databases([path])
            self.assertEqual(rows.home_stats.iloc[0]["corners"], 7)
            self.assertEqual(rows.away_stats.iloc[0], {})
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
