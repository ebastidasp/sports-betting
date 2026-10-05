"""Train a separate pre-match corner-count model from local SQLite caches.

python football_corners.py train
python football_corners.py predict --country colombia --home Millonarios --away "Santa Fe"
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import unicodedata

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

KIND = "sqlite_corners_v1"
RESULT_DELAY = pd.Timedelta(hours=3)
STAT_ALIASES = {
    "Shots on Goal": "shots_on_target", "Shots off Goal": "shots_off_target",
    "Total Shots": "shots", "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_inside_box", "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls", "Corner Kicks": "corners", "Offsides": "offsides",
    "Ball Possession": "possession", "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards", "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "passes", "Passes accurate": "passes_completed",
    "Passes %": "pass_accuracy", "expected_goals": "xg",
}
# Offensive volume, territorial pressure and opponent concessions.
STATS = ["corners", "shots", "shots_on_target", "shots_off_target", "blocked_shots",
         "shots_inside_box", "shots_outside_box", "possession", "passes", "pass_accuracy", "xg"]
VALUES = STATS + [f"{s}_against" for s in STATS] + ["goals", "goals_against"]
HALFLIVES = (5, 20)
PRIORS = {"corners": 5., "shots": 12., "shots_on_target": 4., "shots_off_target": 5.,
          "blocked_shots": 3., "shots_inside_box": 7., "shots_outside_box": 5.,
          "possession": 50., "passes": 400., "pass_accuracy": 78., "xg": 1.3,
          "goals": 1.3}
NON_FEATURES = {"country", "fixture_id", "kickoff", "season", "league_id", "primary_league",
                "team_id", "team_name", "opponent_name", "target"}


def normalize(value):
    return " ".join(unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower().split())


def clean_stats(payload):
    """Accept both API labels and normalized cache keys; null is never zero."""
    result = {}
    for key, value in payload.items():
        name = STAT_ALIASES.get(key, key)
        try:
            number = float(str(value).rstrip("%"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number) or number < 0:
            continue
        if name == "corners" and not number.is_integer():
            continue
        if name in ("possession", "pass_accuracy") and number > 100:
            continue
        result[name] = number
    return result


def read_databases(paths):
    frames, sources = [], {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Database not found: {path}")
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "fixtures" not in tables:
                continue
            fixtures = pd.read_sql_query("SELECT * FROM fixtures WHERE status='FT' AND home_goals IS NOT NULL AND away_goals IS NOT NULL", con)
            if fixtures.empty:
                continue
            config = con.execute("SELECT value FROM app_config WHERE key='country_divisions'").fetchone() if "app_config" in tables else None
            if not config:
                raise ValueError(f"Missing country/division configuration: {path}")
            config = json.loads(config[0])
            country = normalize(config["country"])
            if country in sources:
                raise ValueError(f"Duplicate country cache: {country}")
            ids = [int(x) for x in config["league_ids"]]
            sources[country] = {"path": str(path), "league_ids": ids}
            fixtures = fixtures[fixtures.league_id.isin(ids)].copy()
            fixtures["kickoff"] = pd.to_datetime(fixtures.kickoff, utc=True, errors="raise")
            fixtures["country"] = country
            fixtures["primary_league"] = ids[0]
            fixtures["league"] = country + ":" + fixtures.league_id.astype(str)
            stats = {}
            if "team_stats" in tables:
                for fid, tid, payload in con.execute("SELECT fixture_id, team_id, stats_json FROM team_stats"):
                    stats[(fid, tid)] = clean_stats(json.loads(payload))
            for side in ("home", "away"):
                fixtures[f"{side}_stats"] = [stats.get((fid, tid), {}) for fid, tid in zip(fixtures.fixture_id, fixtures[f"{side}_id"])]
            frames.append(fixtures)
    if not frames:
        raise ValueError("No completed results in the supplied databases.")
    rows = pd.concat(frames, ignore_index=True)
    if rows.duplicated(["country", "fixture_id"]).any():
        raise ValueError("Duplicate fixtures in input caches.")
    if (rows[["home_goals", "away_goals"]].to_numpy() < 0).any():
        raise ValueError("Negative goal counts in cache.")
    return rows.sort_values(["kickoff", "country", "fixture_id"]).reset_index(drop=True), sources


class CornerHistory:
    def __init__(self):
        self.teams = {}
        self.leagues = defaultdict(lambda: {"home": [0., 0], "away": [0., 0]})

    def team(self, country, tid, league_id, primary):
        key = (country, int(tid))
        if key not in self.teams:
            self.teams[key] = {"games": 0, "elo": 1500. if league_id == primary else 1350.,
                "name": None, "league_id": None, "last": None,
                "moments": {scope: {half: {value: [0., 0.] for value in VALUES}
                                    for half in HALFLIVES} for scope in ("all", "home", "away")}}
        return self.teams[key]

    def league_mean(self, league, venue):
        total, n = self.leagues[league][venue]
        prior = 5.3 if venue == "home" else 4.5
        return (total + 30 * prior) / (n + 30)

    def inputs(self, country, league_id, primary, tid, oid, venue, kickoff):
        league = f"{country}:{league_id}"
        own = self.team(country, tid, league_id, primary)
        opp = self.team(country, oid, league_id, primary)
        opp_venue = "away" if venue == "home" else "home"
        own_mean = self.league_mean(league, venue)
        opp_mean = self.league_mean(league, opp_venue)
        x = {"league": league, "venue": venue, "league_corners": own_mean,
             "league_opponent_corners": opp_mean, "elo_diff": (own["elo"] - opp["elo"]) / 400}
        for prefix, team, team_venue in (("own", own, venue), ("opponent", opp, opp_venue)):
            x[f"{prefix}_games"] = math.log1p(team["games"])
            rest = 30 if team["last"] is None else (kickoff - team["last"]).total_seconds() / 86400
            x[f"{prefix}_rest"] = float(np.clip(rest, 0, 30))
            for scope in ("all", team_venue):
                label = "all" if scope == "all" else "venue"
                for half in HALFLIVES:
                    for value in VALUES:
                        base = value.removesuffix("_against")
                        prior = PRIORS[base]
                        if base == "corners":
                            if scope == "all":
                                prior = (own_mean + opp_mean) / 2
                            else:
                                mean_venue = team_venue if not value.endswith("_against") else ("away" if team_venue == "home" else "home")
                                prior = self.league_mean(league, mean_venue)
                        total, mass = team["moments"][scope][half][value]
                        x[f"{prefix}_{label}_{value}_{half}"] = (total + 3 * prior) / (mass + 3)
                        if base in ("corners", "shots", "xg"):
                            x[f"{prefix}_{label}_{value}_coverage_{half}"] = mass
        for half in HALFLIVES:
            x[f"corner_matchup_{half}"] = (x[f"own_venue_corners_{half}"] + x[f"opponent_venue_corners_against_{half}"]) / 2
            x[f"shots_matchup_{half}"] = (x[f"own_venue_shots_{half}"] + x[f"opponent_venue_shots_against_{half}"]) / 2
            x[f"possession_diff_{half}"] = x[f"own_all_possession_{half}"] - x[f"opponent_all_possession_{half}"]
        return x

    def update(self, row):
        h = self.team(row.country, row.home_id, row.league_id, row.primary_league)
        a = self.team(row.country, row.away_id, row.league_id, row.primary_league)
        diff = (h["elo"] - a["elo"] + 70) / 400
        expected = 1 / (1 + 10 ** np.clip(-diff, -10, 10))
        result = 1 if row.home_goals > row.away_goals else 0 if row.home_goals < row.away_goals else .5
        change = 20 * (result - expected)
        h["elo"] += change
        a["elo"] -= change
        for venue, team, own, opponent, goals, conceded in (
            ("home", h, row.home_stats, row.away_stats, row.home_goals, row.away_goals),
            ("away", a, row.away_stats, row.home_stats, row.away_goals, row.home_goals)):
            item = {s: own[s] for s in STATS if s in own}
            item.update({f"{s}_against": opponent[s] for s in STATS if s in opponent})
            item.update({"goals": float(goals), "goals_against": float(conceded)})
            for scope in ("all", venue):
                for half in HALFLIVES:
                    decay = 2 ** (-1 / half)
                    for value, moment in team["moments"][scope][half].items():
                        moment[0] = moment[0] * decay + item.get(value, 0.)
                        moment[1] = moment[1] * decay + int(value in item)
            team["games"] += 1
            team["last"] = row.kickoff
            team["name"] = getattr(row, f"{venue}_name")
            team["league_id"] = row.league_id
            if "corners" in own:
                state = self.leagues[row.league][venue]
                state[0] += own["corners"]
                state[1] += 1


def build_features(rows, as_of=None):
    state, pending, records = CornerHistory(), deque(), []
    rows = rows.sort_values(["kickoff", "country", "fixture_id"])
    if as_of is not None:
        rows = rows[rows.kickoff + RESULT_DELAY <= as_of]
    for kickoff, batch in rows.groupby("kickoff", sort=True):
        while pending and pending[0].kickoff + RESULT_DELAY <= kickoff:
            state.update(pending.popleft())
        for row in batch.itertuples(index=False):
            for venue, tid, oid, stats in (("home", row.home_id, row.away_id, row.home_stats),
                                          ("away", row.away_id, row.home_id, row.away_stats)):
                x = state.inputs(row.country, row.league_id, row.primary_league, tid, oid, venue, kickoff)
                records.append({**x, "country": row.country, "fixture_id": row.fixture_id,
                    "kickoff": kickoff, "season": row.season, "league_id": row.league_id,
                    "primary_league": row.primary_league, "team_id": tid,
                    "team_name": getattr(row, f"{venue}_name"),
                    "opponent_name": getattr(row, "away_name" if venue == "home" else "home_name"),
                    "target": stats.get("corners", np.nan)})
            pending.append(row)
    while pending:
        state.update(pending.popleft())
    return pd.DataFrame(records), state


def preprocess(columns, linear):
    numeric = [c for c in columns if c not in ("league", "venue")]
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if linear:
        steps.append(StandardScaler())
    return ColumnTransformer([
        ("numeric", make_pipeline(*steps), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["league", "venue"])], sparse_threshold=0)


def fit_candidate(frame, columns, spec):
    known = frame[frame.target.notna()]
    if len(known) < 100 or known.target.sum() <= 0:
        raise ValueError("Need at least 100 observed corner counts and positive total corners.")
    if spec["family"] in ("league", "matchup"):
        return None
    if spec["family"] == "poisson":
        estimator = PoissonRegressor(alpha=spec["alpha"], max_iter=500)
    else:
        estimator = HistGradientBoostingRegressor(loss="poisson", max_iter=spec["iterations"],
            max_leaf_nodes=spec["leaves"], min_samples_leaf=100, learning_rate=.05,
            l2_regularization=20, early_stopping=False, random_state=42)
    model = make_pipeline(preprocess(columns, spec["family"] == "poisson"), estimator)
    model.fit(known[columns], known.target)
    return model


def predict_candidate(model, frame, columns, spec):
    if spec["family"] == "league":
        return frame.league_corners.to_numpy()
    if spec["family"] == "matchup":
        return frame.corner_matchup_20.to_numpy()
    return np.maximum(model.predict(frame[columns]), 1e-6)


def pair_predictions(frame, predictions):
    data = frame[["country", "fixture_id", "kickoff", "league_id", "venue", "team_name", "opponent_name", "target"]].copy()
    data["expected"] = predictions
    home = data[data.venue == "home"].drop(columns="venue")
    away = data[data.venue == "away"].drop(columns="venue")
    pairs = home.merge(away, on=["country", "fixture_id", "kickoff", "league_id"], suffixes=("_home", "_away"), validate="one_to_one")
    pairs["target_total"] = pairs.target_home + pairs.target_away
    pairs["expected_total"] = pairs.expected_home + pairs.expected_away
    return pairs


def count_metrics(actual, expected):
    mask = np.isfinite(actual)
    y, mu = np.asarray(actual)[mask], np.asarray(expected)[mask]
    if not len(y):
        return {"observations": 0}
    return {"observations": len(y), "mae": float(mean_absolute_error(y, mu)),
            "rmse": float(math.sqrt(mean_squared_error(y, mu))),
            "poisson_deviance": float(mean_poisson_deviance(y, np.maximum(mu, 1e-6))),
            "bias": float(np.mean(mu - y)),
            "within_2": float(np.mean(np.abs(mu - y) <= 2))}


def evaluate(frame, predictions):
    pairs = pair_predictions(frame, predictions)
    return {"total": count_metrics(pairs.target_total.to_numpy(), pairs.expected_total.to_numpy()),
            "home": count_metrics(pairs.target_home.to_numpy(), pairs.expected_home.to_numpy()),
            "away": count_metrics(pairs.target_away.to_numpy(), pairs.expected_away.to_numpy())}


def dispersion(actual, mean):
    y, mu = np.asarray(actual), np.asarray(mean)
    mask = np.isfinite(y)
    y, mu = y[mask], mu[mask]
    if not len(y):
        return 0.
    # NB2 Var(Y|X)=mu+alpha*mu^2, estimated from out-of-training residuals.
    return float(max(0., np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2)))


def count_distribution(mean, alpha):
    if not math.isfinite(mean) or mean <= 0 or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Mean must be positive and dispersion nonnegative, both finite.")
    return poisson(mu=mean) if alpha < 1e-8 else nbinom(n=1 / alpha, p=1 / (1 + alpha * mean))


def select_model(history, validation, columns):
    specs = [{"family": "league"}, {"family": "matchup"}]
    specs += [{"family": "poisson", "alpha": a} for a in (.1, 1., 10.)]
    specs += [{"family": "boosting", "leaves": leaves, "iterations": n}
              for leaves, n in ((7, 120), (15, 160), (15, 240))]
    scores, predictions = [], []
    for spec in specs:
        model = fit_candidate(history, columns, spec)
        pred = predict_candidate(model, validation, columns, spec)
        result = evaluate(validation, pred)
        scores.append({"components": [spec], "weights": [1.], "metrics": result})
        predictions.append(pred)
        print(f"Validation {spec}: total MAE={result['total']['mae']:.3f}, RMSE={result['total']['rmse']:.3f}", flush=True)
    linear = min(range(2, 5), key=lambda i: scores[i]["metrics"]["total"]["mae"])
    tree = min(range(5, 8), key=lambda i: scores[i]["metrics"]["total"]["mae"])
    for left, right in ((linear, tree), (1, linear), (1, tree)):
        for weight in (.25, .5, .75):
            pred = weight * predictions[left] + (1 - weight) * predictions[right]
            scores.append({"components": [specs[left], specs[right]], "weights": [weight, 1 - weight],
                           "metrics": evaluate(validation, pred)})
            predictions.append(pred)
    best = min(range(len(scores)), key=lambda i: (scores[i]["metrics"]["total"]["mae"], scores[i]["metrics"]["total"]["rmse"]))
    selected = scores[best]
    pairs = pair_predictions(validation, predictions[best])
    calibration = {side: dispersion(pairs[f"target_{side}"], pairs[f"expected_{side}"]) for side in ("home", "away", "total")}
    print(f"Selected: {selected['components']}, weights={selected['weights']}; dispersion={calibration}", flush=True)
    return selected, scores, calibration


def fit_bundle(frame, columns, selection):
    return {"kind": KIND, "columns": columns, "selection": selection,
            "models": [fit_candidate(frame, columns, spec) for spec in selection["components"]]}


def predict_bundle(bundle, frame):
    return sum(weight * predict_candidate(model, frame, bundle["columns"], spec)
               for weight, model, spec in zip(bundle["selection"]["weights"], bundle["models"], bundle["selection"]["components"]))


def train(args):
    rows, sources = read_databases(args.db or sorted(Path('.').glob('*.sqlite3')))
    limit = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if limit.tzinfo is None:
        raise ValueError("--as-of must include a timezone.")
    rows = rows[rows.kickoff + RESULT_DELAY <= limit]
    print(f"Building corner histories from {len(rows)} FT fixtures in {len(sources)} caches...", flush=True)
    frame, _ = build_features(rows)
    start_val = pd.Timestamp(f"{args.validation_year}-01-01", tz="UTC")
    start_test = pd.Timestamp(f"{args.test_year}-01-01", tz="UTC")
    end_test = pd.Timestamp(f"{args.test_year + 1}-01-01", tz="UTC")
    if args.validation_year >= args.test_year:
        raise ValueError("Validation year must precede test year.")
    history = frame[frame.kickoff + RESULT_DELAY <= start_val]
    validation = frame[(frame.kickoff >= start_val) & (frame.kickoff + RESULT_DELAY <= start_test)
                       & (frame.league_id == frame.primary_league)]
    test = frame[(frame.kickoff >= start_test) & (frame.kickoff < end_test)
                 & (frame.league_id == frame.primary_league)]
    validation_pairs = pair_predictions(validation, validation.league_corners.to_numpy())
    test_pairs = pair_predictions(test, test.league_corners.to_numpy())
    if history.target.notna().sum() < 100 or min(validation_pairs.target_total.notna().sum(), test_pairs.target_total.notna().sum()) < 100:
        raise ValueError("Need >=100 observed training counts and >=100 complete validation/test fixtures.")
    columns = [c for c in frame if c not in NON_FEATURES]
    selection, candidates, calibration = select_model(history, validation, columns)
    pretest = frame[frame.kickoff + RESULT_DELAY <= start_test]
    evaluation_model = fit_bundle(pretest, columns, selection)
    pred = predict_bundle(evaluation_model, test)
    pairs = pair_predictions(test, pred)
    report = {"sources": sources, "training_cutoff": limit.isoformat(),
              "split": {"validation_start": start_val.isoformat(), "test_start": start_test.isoformat(), "test_end": end_test.isoformat()},
              "selection": selection, "validation_candidates": candidates, "dispersion": calibration,
              "test": evaluate(test, pred), "league_baseline": evaluate(test, test.league_corners.to_numpy()),
              "matchup_baseline": evaluate(test, test.corner_matchup_20.to_numpy()),
              "test_coverage": {"fixtures": len(pairs), "observed_both_corner_counts": int(pairs.target_total.notna().sum())},
              "countries": {}, "features": columns, "training_team_observations": int(frame.target.notna().sum()),
              "result_availability_hours": 3}
    for country, group in test.groupby("country"):
        positions = test.index.get_indexer(group.index)
        report["countries"][country] = {"model": evaluate(group, pred[positions]),
                                       "baseline": evaluate(group, group.league_corners.to_numpy())}
    observed = pairs[pairs.target_total.notna()].copy()
    # Vectorized distributions for each fixture's predicted mean.
    mu, alpha = observed.expected_total.to_numpy(), calibration["total"]
    dist = poisson(mu=mu) if alpha < 1e-8 else nbinom(n=1 / alpha, p=1 / (1 + alpha * mu))
    low, high = dist.ppf(.05), dist.ppf(.95)
    report["total_interval_90pct_coverage"] = float(np.mean((observed.target_total >= low) & (observed.target_total <= high)))
    pairs["p_over_9_5"] = (poisson.sf(9, pairs.expected_total) if alpha < 1e-8 else nbinom.sf(9, n=1 / alpha, p=1 / (1 + alpha * pairs.expected_total)))
    args.output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output / "test_predictions.csv", index=False)
    print(f"External test: {report['test']['total']}; league baseline: {report['league_baseline']['total']}", flush=True)
    print(f"Final training: {report['training_team_observations']} observed team counts...", flush=True)
    final = fit_bundle(frame, columns, selection)
    final.update({"sources": sources, "training_cutoff": limit.isoformat(), "dispersion": calibration, "report": report})
    joblib.dump(final, args.output / "corners_model.joblib")
    evaluation_model.update({"sources": sources, "training_cutoff": start_test.isoformat(), "dispersion": calibration})
    joblib.dump(evaluation_model, args.output / "evaluation_model.joblib")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {(args.output / 'corners_model.joblib').resolve()}", flush=True)


def resolve_team(query, state, country, league):
    options = [(tid, team) for (c, tid), team in state.teams.items() if c == country and team["league_id"] == league and team["name"]]
    found = [(tid, t) for tid, t in options if str(tid) == query or normalize(t["name"]) == normalize(query)]
    found = found or [(tid, t) for tid, t in options if normalize(query) in normalize(t["name"])]
    if len(found) != 1:
        raise ValueError(f"Unknown/ambiguous team {query!r}: " + ", ".join(f"{tid}: {t['name']}" for tid, t in found or options))
    return found[0]


def predict(args):
    bundle = joblib.load(args.model)
    if bundle.get("kind") != KIND:
        raise ValueError("Wrong model format: select a corner model trained with pass features, or run python football_corners.py train")
    country = normalize(args.country)
    if country not in bundle["sources"]:
        raise ValueError("Country was not in the training sources.")
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None or cutoff < pd.Timestamp(bundle["training_cutoff"]):
        raise ValueError("Prediction date must include a timezone and not precede model training cutoff.")
    if not math.isfinite(args.line) or args.line < 0:
        raise ValueError("--line must be finite and nonnegative.")
    source = bundle["sources"][country]
    rows, _ = read_databases([args.db or source["path"]])
    if set(rows.country) != {country}:
        raise ValueError("Database country differs from requested country.")
    _, state = build_features(rows, as_of=cutoff)
    league = args.league_id or source["league_ids"][0]
    if league not in source["league_ids"]:
        raise ValueError("League not in trained country configuration.")
    hid, h = resolve_team(args.home, state, country, league)
    aid, a = resolve_team(args.away, state, country, league)
    if hid == aid:
        raise ValueError("Choose two different teams.")
    inputs = [state.inputs(country, league, source["league_ids"][0], hid, aid, "home", cutoff),
              state.inputs(country, league, source["league_ids"][0], aid, hid, "away", cutoff)]
    counts = predict_bundle(bundle, pd.DataFrame(inputs))
    print(f"{h['name']} vs {a['name']}; prediction cutoff {cutoff.isoformat()}")
    print(f"Model training cutoff: {bundle['training_cutoff']}")
    print(f"Expected corners: home {counts[0]:.2f}, away {counts[1]:.2f}, total {counts.sum():.2f}")
    for label, mean, side in ((h['name'], counts[0], "home"), (a['name'], counts[1], "away"), ("Total", counts.sum(), "total")):
        dist = count_distribution(float(mean), bundle["dispersion"][side])
        print(f"{label}: approximate 90% prediction interval {int(dist.ppf(.05))}-{int(dist.ppf(.95))}")
    dist = count_distribution(float(counts.sum()), bundle["dispersion"]["total"])
    print("\nProbabilidad del total de esquinas:")
    for number in range(10):
        print(f"  {number:>2}: {dist.pmf(number):.2%}")
    print(f"  >9: {dist.sf(9):.2%}")
    print()
    print(f"P(total corners > {args.line:g}): {dist.sf(math.floor(args.line)):.2%}")
    print(f"P(total corners < {args.line:g}): {dist.cdf(math.ceil(args.line) - 1):.2%}")
    if args.line.is_integer():
        print(f"P(total corners = {args.line:g}): {dist.pmf(int(args.line)):.2%}")
    for label, team, venue in ((h['name'], h, "home"), (a['name'], a, "away")):
        mass = team["moments"][venue][20]["corners"][1]
        if mass < 3:
            print(f"Limited corner history for {label} at this venue (effective observations {mass:.1f}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--db", nargs="+", type=Path)
    t.add_argument("--validation-year", type=int, default=2024)
    t.add_argument("--test-year", type=int, default=2025)
    t.add_argument("--as-of")
    t.add_argument("--output", type=Path, default=Path("corners_model"))
    p = sub.add_parser("predict")
    p.add_argument("--model", type=Path, default=Path("corners_model/corners_model.joblib"))
    p.add_argument("--country", required=True)
    p.add_argument("--db", type=Path)
    p.add_argument("--league-id", type=int)
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--as-of")
    p.add_argument("--line", type=float, default=9.5)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            globals()[args.command](args)
    except (ValueError, FileNotFoundError, sqlite3.Error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
