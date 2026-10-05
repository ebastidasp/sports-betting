"""Independent SQLite-only football predictor. No network or legacy-model imports.

python football_model_v2.py train
python football_model_v2.py train --db colombia.sqlite3
python football_model_v2.py predict --country colombia --home Millonarios --away Santa Fe
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import closing
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import unicodedata

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

KIND = "sqlite_football_classifier_v1"
LABELS = ["Local", "Empate", "Visitante"]
STATS = ["shots", "shots_on_target", "shots_inside_box", "possession", "corners", "xg"]
VALUES = ["goals", "conceded", "points", "goal_residual", "conceded_residual"] + STATS + ["shots_on_target_conceded", "xg_conceded"]
HALFLIVES = (5, 20)
RESULT_DELAY = pd.Timedelta(hours=3)


def normalize(value):
    return " ".join(unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower().split())


def read_databases(paths):
    """Open existing caches read-only; never create tables, write WAL or load .env."""
    frames, metadata, seen = [], {}, set()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Database not found: {path}")
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "fixtures" not in tables:
                continue
            fixtures = pd.read_sql_query(
                "SELECT * FROM fixtures WHERE status='FT' AND home_goals IS NOT NULL AND away_goals IS NOT NULL", con)
            if fixtures.empty:
                continue
            if "app_config" not in tables:
                raise ValueError(f"Missing country/division configuration: {path}")
            config_row = con.execute("SELECT value FROM app_config WHERE key='country_divisions'").fetchone()
            if not config_row:
                raise ValueError(f"Missing country/division configuration: {path}")
            config = json.loads(config_row[0])
            country = normalize(config["country"])
            if country in metadata:
                raise ValueError(f"Duplicate country caches: {country}")
            ids = [int(x) for x in config["league_ids"]]
            fixtures = fixtures[fixtures.league_id.isin(ids)].copy()
            metadata[country] = {"path": str(path), "league_ids": ids}
            fixtures["country"] = country
            fixtures["primary_league"] = ids[0]
            fixtures["league"] = country + ":" + fixtures.league_id.astype(str)
            fixtures["kickoff"] = pd.to_datetime(fixtures.kickoff, utc=True, errors="raise")
            if (fixtures[["home_goals", "away_goals"]].to_numpy() < 0).any():
                raise ValueError(f"Negative goal counts: {path}")
            for row in fixtures.itertuples():
                key = (country, row.fixture_id)
                if key in seen:
                    raise ValueError(f"Duplicate fixture: {key}")
                seen.add(key)
            stats = {}
            if "team_stats" in tables:
                for fid, tid, payload in con.execute("SELECT fixture_id, team_id, stats_json FROM team_stats"):
                    stats[(fid, tid)] = json.loads(payload)
            for side in ("home", "away"):
                fixtures[f"{side}_stats"] = [stats.get((fid, tid), {}) for fid, tid in
                                              zip(fixtures.fixture_id, fixtures[f"{side}_id"])]
            frames.append(fixtures)
    if not frames:
        raise ValueError("No completed FT matches in the supplied databases.")
    return pd.concat(frames, ignore_index=True).sort_values(["kickoff", "country", "fixture_id"]).reset_index(drop=True), metadata


def empty_team(initial=1500):
    return {"elo": {10: float(initial), 30: float(initial)}, "games": 0,
            "last": None, "league_id": None, "name": None,
            "moments": {scope: {half: {value: [0., 0.] for value in VALUES}
                                for half in HALFLIVES} for scope in ("all", "home", "away")},
            "history": {"all": [], "home": [], "away": []}}


def smooth(history, value, half, default):
    observations = [(i, item[value]) for i, item in enumerate(reversed(history))
                    if value in item and np.isfinite(item[value])]
    if not observations:
        return default, 0.0
    weights = np.array([2 ** (-age / half) for age, _ in observations])
    values = np.array([v for _, v in observations])
    # Three pseudo-games prevent extreme rates from tiny histories.
    mass = float(weights.sum())
    return float((weights @ values + 3 * default) / (mass + 3)), mass


class History:
    """Historical state; only completed matches available before the cutoff enter it."""
    def __init__(self):
        self.teams = {}
        self.leagues = defaultdict(lambda: {"games": 0, "home": 0., "away": 0.})

    def team(self, country, tid, league_id, primary):
        key = (country, int(tid))
        if key not in self.teams:
            self.teams[key] = empty_team(1500 if league_id == primary else 1350)
        return self.teams[key]

    def league_rates(self, league):
        state = self.leagues[league]
        return ((state["home"] + 30 * 1.5) / (state["games"] + 30),
                (state["away"] + 30 * 1.15) / (state["games"] + 30))

    def inputs(self, country, league_id, primary, hid, aid, kickoff):
        league = f"{country}:{league_id}"
        h = self.team(country, hid, league_id, primary)
        a = self.team(country, aid, league_id, primary)
        home_rate, away_rate = self.league_rates(league)
        features = {"league": league, "league_home_goals": home_rate, "league_away_goals": away_rate}
        for k in (10, 30):
            features[f"elo_diff_{k}"] = (h["elo"][k] - a["elo"][k]) / 400
        defaults = {"goals": (home_rate + away_rate) / 2,
                    "conceded": (home_rate + away_rate) / 2, "points": 1.35,
                    "goal_residual": 0, "conceded_residual": 0,
                    "shots": 12, "shots_on_target": 4, "shots_inside_box": 7,
                    "possession": 50, "corners": 5, "xg": (home_rate + away_rate) / 2,
                    "shots_on_target_conceded": 4, "xg_conceded": (home_rate + away_rate) / 2}
        for side, team in (("home", h), ("away", a)):
            features[f"{side}_log_games"] = math.log1p(team["games"])
            rest = 30. if team["last"] is None else (kickoff - team["last"]).total_seconds() / 86400
            features[f"{side}_rest"] = float(np.clip(rest, 0, 30))
            for scope in ("all", side):
                for half in HALFLIVES:
                    for value in VALUES:
                        prior = defaults[value]
                        if scope != "all" and value in ("goals", "conceded"):
                            prior = home_rate if (side == "home") == (value == "goals") else away_rate
                        total, mass = team["moments"][scope][half][value]
                        mean = (total + 3 * prior) / (mass + 3)
                        features[f"{side}_{scope}_{value}_{half}"] = mean
                        if value in ("goals", "shots_on_target", "xg"):
                            features[f"{side}_{scope}_{value}_coverage_{half}"] = mass
        # Differences aid the linear classifier; trees can learn interactions.
        for half in HALFLIVES:
            for value in VALUES:
                features[f"diff_{value}_{half}"] = (features[f"home_all_{value}_{half}"]
                                                     - features[f"away_all_{value}_{half}"])
            features[f"home_attack_vs_defense_{half}"] = (
                features[f"home_home_goals_{half}"] + features[f"away_away_conceded_{half}"]) / 2
            features[f"away_attack_vs_defense_{half}"] = (
                features[f"away_away_goals_{half}"] + features[f"home_home_conceded_{half}"]) / 2
        return features

    def update(self, row):
        h = self.team(row.country, row.home_id, row.league_id, row.primary_league)
        a = self.team(row.country, row.away_id, row.league_id, row.primary_league)
        hg, ag = float(row.home_goals), float(row.away_goals)
        home_rate, away_rate = self.league_rates(row.league)
        # Opponent-adjusted scoring residuals, based solely on earlier ratings.
        diff = (h["elo"][10] - a["elo"][10]) / 400
        expected_h = home_rate * math.exp(np.clip(0.45 * diff, -1.5, 1.5))
        expected_a = away_rate * math.exp(np.clip(-0.45 * diff, -1.5, 1.5))
        result = 1. if hg > ag else 0. if hg < ag else .5
        for k in (10, 30):
            expected = 1 / (1 + 10 ** np.clip(-(h["elo"][k] - a["elo"][k] + 70) / 400, -10, 10))
            change = k * math.sqrt(min(abs(hg - ag), 4) or 1) * (result - expected)
            h["elo"][k] += change
            a["elo"][k] -= change
        for side, team, own, conceded, own_exp, opp_exp, stats, opponent in (
            ("home", h, hg, ag, expected_h, expected_a, row.home_stats, row.away_stats),
            ("away", a, ag, hg, expected_a, expected_h, row.away_stats, row.home_stats)):
            item = {"goals": own, "conceded": conceded,
                    "points": 3 if own > conceded else 1 if own == conceded else 0,
                    "goal_residual": own - own_exp, "conceded_residual": conceded - opp_exp}
            for stat in STATS:
                value = stats.get(stat)
                if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
                    item[stat] = float(value)
            for stat in ("shots_on_target", "xg"):
                value = opponent.get(stat)
                if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
                    item[f"{stat}_conceded"] = float(value)
            for scope in ("all", side):
                team["history"][scope].append(item)
                removed = team["history"][scope][0] if len(team["history"][scope]) > 120 else {}
                for half in HALFLIVES:
                    decay = 2 ** (-1 / half)
                    for value, moment in team["moments"][scope][half].items():
                        moment[0] *= decay
                        moment[1] *= decay
                        if value in item:
                            moment[0] += item[value]
                            moment[1] += 1
                        if value in removed:
                            moment[0] -= decay ** 120 * removed[value]
                            moment[1] -= decay ** 120
                team["history"][scope] = team["history"][scope][-120:]
            team["games"] += 1
            team["last"] = row.kickoff
            team["league_id"] = row.league_id
            team["name"] = getattr(row, f"{side}_name")
        league = self.leagues[row.league]
        league["games"] += 1
        league["home"] += hg
        league["away"] += ag


def build_features(rows, as_of=None):
    state, pending, records = History(), deque(), []
    ordered = rows.sort_values(["kickoff", "country", "fixture_id"])
    if as_of is not None:
        ordered = ordered[ordered.kickoff + RESULT_DELAY <= as_of]
    for kickoff, batch in ordered.groupby("kickoff", sort=True):
        while pending and pending[0].kickoff + RESULT_DELAY <= kickoff:
            state.update(pending.popleft())
        for row in batch.itertuples(index=False):
            features = state.inputs(row.country, row.league_id, row.primary_league,
                                    row.home_id, row.away_id, kickoff)
            records.append({**features, "country": row.country, "fixture_id": row.fixture_id,
                            "kickoff": kickoff, "season": row.season, "league_id": row.league_id,
                            "primary_league": row.primary_league,
                            "home_name": row.home_name, "away_name": row.away_name,
                            "home_goals": row.home_goals, "away_goals": row.away_goals,
                            "target": 0 if row.home_goals > row.away_goals else 2 if row.home_goals < row.away_goals else 1})
            pending.append(row)
    while pending:
        state.update(pending.popleft())
    return pd.DataFrame(records), state


NON_FEATURES = {"country", "fixture_id", "kickoff", "season", "league_id", "primary_league",
                "home_name", "away_name", "home_goals", "away_goals", "target"}


def preprocessing(columns, linear):
    numeric = [c for c in columns if c != "league"]
    number_steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if linear:
        number_steps.append(StandardScaler())
    return ColumnTransformer([
        ("numeric", make_pipeline(*number_steps), numeric),
        ("league", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["league"])],
        sparse_threshold=0)


def fit_candidate(frame, columns, spec):
    family = spec["family"]
    if family == "poisson":
        # Independently trained reference: league baseline + overall Elo.
        inputs = ["league", "elo_diff_10"]
        models = []
        for side in ("home", "away"):
            model = make_pipeline(preprocessing(inputs, True), PoissonRegressor(alpha=0.1, max_iter=300))
            model.fit(frame[inputs], frame[f"{side}_goals"])
            models.append(model)
        return models
    if family == "logistic":
        estimator = LogisticRegression(C=spec["C"], max_iter=500)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=spec["iterations"],
            max_leaf_nodes=spec["leaves"], learning_rate=.05, l2_regularization=20,
            min_samples_leaf=80, early_stopping=False, random_state=42)
    model = make_pipeline(preprocessing(columns, family == "logistic"), estimator)
    model.fit(frame[columns], frame.target)
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("All three outcomes are required in training.")
    return model


def candidate_probabilities(model, frame, columns, spec):
    if spec["family"] == "poisson":
        rates = [np.clip(m.predict(frame[["league", "elo_diff_10"]]), .05, 8) for m in model]
        home, away = rates
        p = np.column_stack([skellam.sf(0, home, away), skellam.pmf(0, home, away), skellam.cdf(-1, home, away)])
    else:
        p = model.predict_proba(frame[columns])
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(axis=1, keepdims=True)


def metrics(frame, probabilities):
    y = frame.target.to_numpy(dtype=int)
    return {"matches": len(frame), "accuracy": float(accuracy_score(y, probabilities.argmax(axis=1))),
            "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
            "brier": float(np.mean(np.sum((probabilities - np.eye(3)[y]) ** 2, axis=1)))}


def select_model(history, validation, columns, metric):
    specs = [{"family": "poisson"}]
    specs += [{"family": "logistic", "C": c} for c in (.01, .1, 1)]
    specs += [{"family": "boosting", "leaves": leaves, "iterations": iterations}
              for leaves, iterations in ((7, 120), (15, 160), (15, 240))]
    fitted, probabilities, records = [], [], []
    for spec in specs:
        model = fit_candidate(history, columns, spec)
        p = candidate_probabilities(model, validation, columns, spec)
        score = metrics(validation, p)
        print(f"Validation {spec}: accuracy={score['accuracy']:.2%}, log loss={score['log_loss']:.4f}", flush=True)
        fitted.append(model)
        probabilities.append(p)
        records.append({"components": [spec], "weights": [1.], **score})
    # Blend genuinely different estimators; choose weights without test results.
    best_linear = min(range(1, 4), key=lambda i: records[i]["log_loss"])
    best_tree = min(range(4, 7), key=lambda i: records[i]["log_loss"])
    for left, right in ((best_linear, best_tree), (0, best_linear), (0, best_tree)):
        for weight in (.25, .5, .75):
            p = weight * probabilities[left] + (1 - weight) * probabilities[right]
            records.append({"components": [specs[left], specs[right]], "weights": [weight, 1 - weight],
                            **metrics(validation, p)})
    if metric == "accuracy":
        best = min(records, key=lambda r: (-r["accuracy"], r["log_loss"]))
    else:
        best = min(records, key=lambda r: (r["log_loss"], -r["accuracy"]))
    print(f"Selected on {metric}: {best}", flush=True)
    return best, records


def predict_bundle(bundle, frame):
    p = sum(weight * candidate_probabilities(model, frame, bundle["columns"], spec)
            for model, spec, weight in zip(bundle["models"], bundle["selection"]["components"], bundle["selection"]["weights"]))
    return p / p.sum(axis=1, keepdims=True)


def fit_bundle(frame, columns, selection):
    return {"kind": KIND, "columns": columns, "selection": selection,
            "models": [fit_candidate(frame, columns, spec) for spec in selection["components"]]}


def train(args):
    rows, sources = read_databases(args.db or sorted(Path('.').glob('*.sqlite3')))
    limit = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if limit.tzinfo is None:
        raise ValueError("--as-of must include a timezone.")
    rows = rows[rows.kickoff + RESULT_DELAY <= limit]
    print(f"Building pre-match features from {len(rows)} matches in {len(sources)} caches...", flush=True)
    frame, _ = build_features(rows)
    validation_start = pd.Timestamp(f"{args.validation_year}-01-01", tz="UTC")
    test_start = pd.Timestamp(f"{args.test_year}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{args.test_year + 1}-01-01", tz="UTC")
    if args.validation_year >= args.test_year:
        raise ValueError("Validation year must precede test year.")
    # Results used to fit must already be available at the boundary.
    history = frame[frame.kickoff + RESULT_DELAY <= validation_start]
    validation = frame[(frame.kickoff >= validation_start) & (frame.kickoff + RESULT_DELAY <= test_start)
                       & (frame.league_id == frame.primary_league)]
    test = frame[(frame.kickoff >= test_start) & (frame.kickoff < test_end)
                 & (frame.league_id == frame.primary_league)]
    if min(len(history), len(validation), len(test)) < 100:
        raise ValueError("Need >=100 training, validation and test matches; change years or add caches.")
    columns = [c for c in frame.columns if c not in NON_FEATURES]
    selection, candidates = select_model(history, validation, columns, args.selection_metric)
    pretest = frame[frame.kickoff + RESULT_DELAY <= test_start]
    evaluation = fit_bundle(pretest, columns, selection)
    p = predict_bundle(evaluation, test)
    reference_spec = {"family": "poisson"}
    reference = fit_candidate(pretest, columns, reference_spec)
    baseline = candidate_probabilities(reference, test, columns, reference_spec)
    report = {"split": {"training_before": validation_start.isoformat(), "test_start": test_start.isoformat(),
                         "test_end": test_end.isoformat(), "validation_matches": len(validation),
                         "pretest_training_matches": len(pretest)},
              "selection_metric": args.selection_metric, "selection": selection,
              "validation_candidates": candidates, "test": metrics(test, p), "reference": metrics(test, baseline),
              "countries": {}, "sources": sources, "result_availability_hours": 3}
    # Paired uncertainty interval: test rows never alter the chosen estimator.
    difference = ((p.argmax(1) == test.target.to_numpy()).astype(float)
                  - (baseline.argmax(1) == test.target.to_numpy()).astype(float))
    rng = np.random.default_rng(42)
    # Resample calendar weeks, preserving dependence among nearby fixtures.
    week = test.kickoff.dt.strftime("%G-%V")
    blocks = pd.DataFrame({"week": week.to_numpy(), "diff": difference}).groupby("week").agg(total=("diff", "sum"), n=("diff", "size"))
    samples = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    delta = blocks.total.to_numpy()[samples].sum(1) / blocks.n.to_numpy()[samples].sum(1)
    report["accuracy_difference_95pct_week_bootstrap"] = np.quantile(delta, [.025, .975]).tolist()
    for country, group in test.groupby("country"):
        positions = test.index.get_indexer(group.index)
        report["countries"][country] = {"new": metrics(group, p[positions]), "reference": metrics(group, baseline[positions])}
    args.output.mkdir(parents=True, exist_ok=True)
    export = test[["country", "fixture_id", "kickoff", "season", "league_id", "home_name", "away_name", "home_goals", "away_goals", "target"]].copy()
    export[["p_home", "p_draw", "p_away"]] = p
    export[["reference_home", "reference_draw", "reference_away"]] = baseline
    export["prediction"] = p.argmax(axis=1)
    export.to_csv(args.output / "test_predictions.csv", index=False)
    print(f"External test: {report['test']}; reference: {report['reference']}", flush=True)
    print(f"Training final model on all {len(frame)} available matches...", flush=True)
    final = fit_bundle(frame, columns, selection)
    final.update({"sources": sources, "training_matches": len(frame), "training_cutoff": limit.isoformat(),
                  "last_training_kickoff": frame.kickoff.max().isoformat(), "report": report})
    model_path = args.output / "football_model.joblib"
    joblib.dump(final, model_path)
    # Save the pre-test bundle separately to permit historical reproduction.
    evaluation.update({"sources": sources, "training_cutoff": test_start.isoformat(),
                       "last_training_kickoff": pretest.kickoff.max().isoformat()})
    joblib.dump(evaluation, args.output / "evaluation_model.joblib")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved trained model: {model_path.resolve()}", flush=True)


def resolve_team(query, state, country, league):
    options = [(tid, team) for (c, tid), team in state.teams.items()
               if c == country and team["league_id"] == league and team["name"]]
    exact = [(tid, team) for tid, team in options if str(tid) == query or normalize(team["name"]) == normalize(query)]
    found = exact or [(tid, team) for tid, team in options if normalize(query) in normalize(team["name"])]
    if len(found) != 1:
        raise ValueError(f"Unknown/ambiguous team {query!r}. Candidates: " + ", ".join(f"{tid}: {t['name']}" for tid, t in found or options))
    return found[0]


def predict(args):
    bundle = joblib.load(args.model)
    if bundle.get("kind") != KIND:
        raise ValueError("Unsupported model format.")
    country = normalize(args.country)
    if country not in bundle["sources"]:
        raise ValueError("Country was not in training caches.")
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None:
        raise ValueError("--as-of must include a timezone.")
    if cutoff < pd.Timestamp(bundle["training_cutoff"]):
        raise ValueError("Prediction date precedes model training cutoff; use an earlier evaluation model.")
    source = bundle["sources"][country]
    rows, _ = read_databases([args.db or source["path"]])
    if set(rows.country) != {country}:
        raise ValueError("Database country differs from selected country.")
    _, state = build_features(rows, as_of=cutoff)
    league = args.league_id or source["league_ids"][0]
    if league not in source["league_ids"]:
        raise ValueError("League is not in this country's trained configuration.")
    hid, home = resolve_team(args.home, state, country, league)
    aid, away = resolve_team(args.away, state, country, league)
    if hid == aid:
        raise ValueError("Select two different teams.")
    features = state.inputs(country, league, source["league_ids"][0], hid, aid, cutoff)
    p = predict_bundle(bundle, pd.DataFrame([features]))[0]
    print(f"{home['name']} vs {away['name']}; pre-match cutoff: {cutoff.isoformat()}")
    print(f"Model training cutoff: {bundle['training_cutoff']}")
    print(f"Historical matches: {home['games']} / {away['games']}")
    for label, probability in zip(LABELS, p):
        print(f"{label}: {probability:.2%}")
    print(f"Predicted outcome: {LABELS[int(p.argmax())]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--db", nargs="+", type=Path, help="Defaults to all .sqlite3 files in current directory")
    t.add_argument("--validation-year", type=int, default=2024)
    t.add_argument("--test-year", type=int, default=2025)
    t.add_argument("--selection-metric", choices=("accuracy", "log_loss"), default="accuracy")
    t.add_argument("--as-of", help="Only use results available at this ISO timestamp")
    t.add_argument("--output", type=Path, default=Path("new_model"))
    p = sub.add_parser("predict")
    p.add_argument("--model", type=Path, default=Path("new_model/football_model.joblib"))
    p.add_argument("--country", required=True)
    p.add_argument("--db", type=Path)
    p.add_argument("--league-id", type=int)
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--as-of")
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            globals()[args.command](args)
    except (ValueError, FileNotFoundError, sqlite3.Error) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
