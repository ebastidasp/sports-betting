"""Bounded 2024 validation probe: no 2025/2026 outcomes are read or scored."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import skellam
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import football_model_v2 as m

OUTPUT = Path(__file__).resolve().parent
END = pd.Timestamp("2025-01-01T00:00:00Z")
VALIDATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
CALIBRATION_START = pd.Timestamp("2023-01-01T00:00:00Z")


def bounded_read(paths):
    frames, sources = [], {}
    cutoff = END.isoformat()
    for path in paths:
        path = path.resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as con:
            tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"fixtures", "app_config"}.issubset(tables):
                continue
            config = con.execute("SELECT value FROM app_config WHERE key='country_divisions'").fetchone()
            if not config:
                continue
            config = json.loads(config[0])
            rows = pd.read_sql_query(
                "SELECT * FROM fixtures WHERE status='FT' AND home_goals IS NOT NULL "
                "AND away_goals IS NOT NULL AND julianday(kickoff)<julianday(?)",
                con, params=(cutoff,))
            if rows.empty:
                continue
            country = m.normalize(config["country"])
            if country in sources:
                raise ValueError(f"Duplicate bounded country cache: {country}")
            league_ids = [int(value) for value in config["league_ids"]]
            rows = rows[rows.league_id.isin(league_ids)].copy()
            rows["country"] = country
            rows["primary_league"] = league_ids[0]
            rows["league"] = country + ":" + rows.league_id.astype(str)
            rows["kickoff"] = pd.to_datetime(rows.kickoff, utc=True)
            rows = rows[rows.kickoff + m.RESULT_DELAY <= END]
            stats = {}
            if "team_stats" in tables:
                query = ("SELECT s.fixture_id,s.team_id,s.stats_json FROM team_stats s "
                         "JOIN fixtures f ON f.fixture_id=s.fixture_id WHERE f.status='FT' "
                         "AND julianday(f.kickoff)<julianday(?)")
                stats = {(fid, tid): json.loads(payload)
                         for fid, tid, payload in con.execute(query, (cutoff,))}
            for side in ("home", "away"):
                rows[f"{side}_stats"] = [stats.get((fid, tid), {})
                                          for fid, tid in zip(rows.fixture_id, rows[f"{side}_id"])]
            frames.append(rows)
            sources[country] = {"path": str(path), "league_ids": league_ids}
    if not frames:
        raise ValueError("No bounded data available")
    rows = pd.concat(frames, ignore_index=True)
    if rows.duplicated(["country", "fixture_id"]).any():
        raise ValueError("Duplicate bounded fixtures")
    return rows.sort_values(["kickoff", "country", "fixture_id"]).reset_index(drop=True), sources


def pooled(frame, columns):
    blocks = []
    for side, other in (("home", "away"), ("away", "home")):
        data = {"league": frame.league.to_numpy(), "is_home": np.full(len(frame), float(side == "home"))}
        for column in columns:
            if column == "league":
                continue
            if column.startswith(side + "_"):
                name = "own_" + column[len(side) + 1:]
                name = name.replace("own_" + side + "_", "own_venue_")
                data[name] = frame[column].to_numpy()
            elif column.startswith(other + "_"):
                name = "opponent_" + column[len(other) + 1:]
                name = name.replace("opponent_" + other + "_", "opponent_venue_")
                data[name] = frame[column].to_numpy()
            elif column.startswith("elo_diff_") or column.startswith("diff_"):
                data[column] = frame[column].to_numpy() * (1 if side == "home" else -1)
        data["league_own_goals"] = frame[f"league_{side}_goals"].to_numpy()
        data["league_opponent_goals"] = frame[f"league_{other}_goals"].to_numpy()
        blocks.append(pd.DataFrame(data))
    return pd.concat(blocks, ignore_index=True)


def fit_rates(frame, columns, alpha, shared=False):
    if shared:
        x = pooled(frame, columns)
        pooled_columns = list(x.columns)
        model = make_pipeline(m.preprocessing(pooled_columns, True),
                              PoissonRegressor(alpha=alpha, max_iter=350))
        model.fit(x, np.concatenate([frame.home_goals, frame.away_goals]))
        return {"model": model, "columns": pooled_columns, "shared": True}
    models = []
    for side in ("home", "away"):
        model = make_pipeline(m.preprocessing(columns, True),
                              PoissonRegressor(alpha=alpha, max_iter=350))
        model.fit(frame[columns], frame[f"{side}_goals"])
        models.append(model)
    return {"models": models, "shared": False}


def rates(model, frame, columns):
    if model["shared"]:
        y = np.clip(model["model"].predict(pooled(frame, columns)[model["columns"]]), .05, 8.)
        return y[:len(frame)], y[len(frame):]
    return tuple(np.clip(candidate.predict(frame[columns]), .05, 8.) for candidate in model["models"])


def fit_rho(frame, home, away):
    hg, ag = frame.home_goals.to_numpy(), frame.away_goals.to_numpy()
    coefficient = np.zeros(len(frame))
    coefficient[(hg == 0) & (ag == 0)] = -(home * away)[(hg == 0) & (ag == 0)]
    coefficient[(hg == 0) & (ag == 1)] = home[(hg == 0) & (ag == 1)]
    coefficient[(hg == 1) & (ag == 0)] = away[(hg == 1) & (ag == 0)]
    coefficient[(hg == 1) & (ag == 1)] = -1.
    lower = max(-.2, float(np.max(-1 / home)), float(np.max(-1 / away))) + 1e-8
    upper = min(.2, float(np.min(1 / (home * away))), 1.) - 1e-8
    result = minimize_scalar(lambda rho: -np.log(1 + rho * coefficient).mean(),
                             bounds=(lower, upper), method="bounded")
    return float(result.x)


def probabilities(home, away, rho=0.):
    p = np.column_stack([skellam.sf(0, home, away), skellam.pmf(0, home, away), skellam.cdf(-1, home, away)])
    adjustment = np.exp(-home - away) * home * away * rho
    p[:, 0] += adjustment
    p[:, 1] -= 2 * adjustment
    p[:, 2] += adjustment
    p = np.maximum(p, 1e-12)
    return p / p.sum(axis=1, keepdims=True)


def main():
    root = OUTPUT.parent
    rows, sources = bounded_read(sorted(root.glob("*.sqlite3")))
    frame, _ = m.build_features(rows)
    columns = [c for c in frame if c not in m.NON_FEATURES]
    training = frame[frame.kickoff + m.RESULT_DELAY <= VALIDATION_START]
    calibration_train = frame[frame.kickoff + m.RESULT_DELAY <= CALIBRATION_START]
    calibration = frame[(frame.kickoff >= CALIBRATION_START)
                        & (frame.kickoff + m.RESULT_DELAY <= VALIDATION_START)
                        & (frame.league_id == frame.primary_league)]
    validation = frame[(frame.kickoff >= VALIDATION_START)
                       & (frame.kickoff + m.RESULT_DELAY <= END)
                       & (frame.league_id == frame.primary_league)]
    report = {"data_policy": "SQL limits exclude all 2025 and 2026 fixtures/statistics; 2024 validation only",
              "training_matches": len(training), "calibration_training_matches": len(calibration_train),
              "calibration_matches_2023": len(calibration), "validation_matches_2024": len(validation),
              "features": len(columns), "sources": sources, "candidates": []}
    predictions = validation[["country", "fixture_id", "kickoff", "home_name", "away_name", "target"]].copy()
    baseline_spec = {"family": "boosting", "leaves": 7, "iterations": 120}
    baseline = m.fit_candidate(training, columns, baseline_spec)
    baseline_p = m.candidate_probabilities(baseline, validation, columns, baseline_spec)

    def record(name, p, extra=None):
        score = {"name": name, **m.metrics(validation, p), **(extra or {})}
        report["candidates"].append(score)
        predictions[[name + "_home", name + "_draw", name + "_away"]] = p
        print(json.dumps(score), flush=True)
        (OUTPUT / "goal_probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    record("baseline_hgb_7_120", baseline_p)
    for name, alpha, shared in (("separate_poisson_0p1", .1, False),
                                ("separate_poisson_1", 1., False),
                                ("pooled_poisson_0p1", .1, True)):
        calibration_model = fit_rates(calibration_train, columns, alpha, shared)
        rho = fit_rho(calibration, *rates(calibration_model, calibration, columns))
        model = fit_rates(training, columns, alpha, shared)
        home, away = rates(model, validation, columns)
        record(name, probabilities(home, away), {"alpha": alpha, "shared": shared})
        record(name + "_dc", probabilities(home, away, rho),
               {"alpha": alpha, "shared": shared, "rho_fit_on_2023": rho})
    predictions.to_csv(OUTPUT / "goal_probe_validation_2024.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
