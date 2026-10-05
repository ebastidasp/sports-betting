import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

import football_model_v2 as m


def matches():
    start = pd.Timestamp("2023-01-01T12:00:00Z")
    return pd.DataFrame([
        {"country": "test", "fixture_id": i, "season": 2023, "league_id": 1,
         "primary_league": 1, "league": "test:1", "kickoff": start + pd.Timedelta(hours=hours),
         "home_id": 10, "away_id": 20, "home_name": "Home", "away_name": "Away",
         "home_goals": hg, "away_goals": ag, "home_stats": {"shots": 10}, "away_stats": {}}
        for i, hours, hg, ag in ((1, 0, 1, 0), (2, 0, 2, 2), (3, 2, 0, 1), (4, 6, 3, 0), (5, 24, 1, 1))])


class HistoryTests(unittest.TestCase):
    def test_current_and_future_results_do_not_change_inputs(self):
        original = matches()
        changed = original.copy(deep=True)
        changed.loc[changed.fixture_id >= 4, "home_goals"] = 50
        a, _ = m.build_features(original)
        b, _ = m.build_features(changed)
        columns = [c for c in a if c not in m.NON_FEATURES]
        pd.testing.assert_frame_equal(a.loc[a.fixture_id <= 4, columns], b.loc[b.fixture_id <= 4, columns])
        self.assertFalse(a.loc[4, columns].equals(b.loc[4, columns]))

    def test_simultaneous_and_unfinished_results_are_excluded(self):
        frame, _ = m.build_features(matches())
        self.assertEqual(frame.home_log_games.tolist()[:3], [0., 0., 0.])
        self.assertAlmostEqual(frame.home_log_games.iloc[3], np.log1p(3))

    def test_same_kickoff_input_is_independent_of_row_order(self):
        a, _ = m.build_features(matches())
        b, _ = m.build_features(matches().iloc[::-1])
        pd.testing.assert_frame_equal(a, b)

    def test_as_of_does_not_use_later_matches(self):
        rows = matches()
        cutoff = rows.kickoff.min() + pd.Timedelta(hours=4)
        _, state = m.build_features(rows, as_of=cutoff)
        self.assertEqual(state.teams[("test", 10)]["games"], 2)

    def test_missing_statistics_keep_matches(self):
        rows = matches()
        rows["home_stats"] = [{} for _ in range(len(rows))]
        frame, _ = m.build_features(rows)
        self.assertEqual(len(frame), len(rows))
        numeric = frame[[c for c in frame if c not in m.NON_FEATURES and c != "league"]]
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())

    def test_no_target_in_feature_columns(self):
        frame, _ = m.build_features(matches())
        columns = [c for c in frame if c not in m.NON_FEATURES]
        self.assertFalse({"home_goals", "away_goals", "target", "fixture_id", "season"} & set(columns))

    def test_incremental_smoothing_matches_explicit_history(self):
        rows = pd.concat([matches()] * 30, ignore_index=True)
        rows["fixture_id"] = np.arange(len(rows))
        rows["kickoff"] = pd.date_range("2023-01-01", periods=len(rows), freq="D", tz="UTC")
        rows.loc[::3, "home_stats"] = pd.Series([{} for _ in range(len(rows.loc[::3]))], index=rows.index[::3])
        _, state = m.build_features(rows)
        team = state.teams[("test", 10)]
        for half in m.HALFLIVES:
            for value in m.VALUES:
                expected, mass = m.smooth(team["history"]["home"], value, half, 1.2)
                total, actual_mass = team["moments"]["home"][half][value]
                self.assertAlmostEqual(mass, actual_mass)
                self.assertAlmostEqual(expected, (total + 3 * 1.2) / (actual_mass + 3))


class CacheTests(unittest.TestCase):
    def test_read_only_and_no_stats_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.sqlite3"
            with closing(sqlite3.connect(path)) as con, con:
                con.execute("CREATE TABLE app_config (key TEXT, value TEXT)")
                con.execute("INSERT INTO app_config VALUES (?,?)", ("country_divisions", json.dumps({"country": "test", "league_ids": [1]})))
                con.execute("CREATE TABLE fixtures (fixture_id INTEGER, league_id INTEGER, season INTEGER, kickoff TEXT, status TEXT, home_id INTEGER, home_name TEXT, away_id INTEGER, away_name TEXT, home_goals INTEGER, away_goals INTEGER)")
                for status, fid in (("FT", 1), ("AET", 2)):
                    con.execute("INSERT INTO fixtures VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                (fid, 1, 2023, "2023-01-01T12:00:00Z", status, 10, "H", 20, "A", 1, 0))
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            rows, _ = m.read_databases([path])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.home_stats.iloc[0], {})
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["test.sqlite3"])

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "absent.sqlite3"
            with self.assertRaises(ValueError):
                m.read_databases([path])
            self.assertFalse(path.exists())


class EstimatorTests(unittest.TestCase):
    def test_serialization_and_probability_order(self):
        import joblib
        from threadpoolctl import threadpool_limits
        frame, _ = m.build_features(matches())
        frame = pd.concat([frame] * 10, ignore_index=True)
        columns = [c for c in frame if c not in m.NON_FEATURES]
        selection = {"components": [{"family": "logistic", "C": .01}], "weights": [1.]}
        with threadpool_limits(limits=1):
            bundle = m.fit_bundle(frame, columns, selection)
            p = m.predict_bundle(bundle, frame)
        self.assertEqual(p.shape, (50, 3))
        np.testing.assert_allclose(p.sum(1), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(bundle, path)
            np.testing.assert_allclose(m.predict_bundle(joblib.load(path), frame), p)


if __name__ == "__main__":
    unittest.main()
