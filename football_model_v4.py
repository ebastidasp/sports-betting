"""SQLite-only V3 feature ablation without possession or passing statistics.

python football_model_v4.py train
python football_model_v4.py backtest
python football_model_v4.py predict --country england --league-id 40 --home Middlesbrough --away Wolves

V4 reuses V3's mathematical estimators, not its fitted coefficients. New models
are trained separately in new_model_v4/. Original models remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

import football_model_v3 as v3

base = v3.base
KIND = "sqlite_football_classifier_v4"
EXCLUDED_STATISTICS = ("possession", "passes", "passes_completed", "pass_accuracy")
ROOT = Path(__file__).resolve().parent


def allowed_feature(name):
    """Reject canonical and source-alias possession/passing names and derivatives."""
    normalized = base.normalize(str(name))
    return "possession" not in normalized and "pass" not in normalized


def allowed_stats(stats):
    return {key: value for key, value in stats.items() if allowed_feature(key)}


def read_databases(paths):
    rows, sources = base.read_databases(paths)
    rows = rows.copy()
    for side in ("home", "away"):
        rows[f"{side}_stats"] = rows[f"{side}_stats"].map(allowed_stats)
    return rows, sources


class History(v3.History):
    def inputs(self, *args, **kwargs):
        return {key: value for key, value in super().inputs(*args, **kwargs).items()
                if allowed_feature(key)}

    def update(self, row):
        # Also sanitize here so callers cannot bypass the exclusion by constructing
        # their own fixture rows or using the shared ingestion helper directly.
        row = row._replace(home_stats=allowed_stats(row.home_stats),
                           away_stats=allowed_stats(row.away_stats))
        super().update(row)


def build_features(rows, as_of=None):
    state, pending, records = History(), deque(), []
    ordered = rows.sort_values(["kickoff", "country", "fixture_id"])
    if as_of is not None:
        ordered = ordered[ordered.kickoff + base.RESULT_DELAY <= as_of]
    for kickoff, batch in ordered.groupby("kickoff", sort=True):
        while pending and pending[0].kickoff + base.RESULT_DELAY <= kickoff:
            state.update(pending.popleft())
        for row in batch.itertuples(index=False):
            inputs = state.inputs(row.country, row.league_id, row.primary_league,
                                  row.home_id, row.away_id, kickoff)
            records.append({**inputs, "country": row.country, "fixture_id": row.fixture_id,
                "kickoff": kickoff, "season": row.season, "league_id": row.league_id,
                "primary_league": row.primary_league, "home_name": row.home_name,
                "away_name": row.away_name, "home_goals": row.home_goals,
                "away_goals": row.away_goals,
                "target": 0 if row.home_goals > row.away_goals else 2 if row.home_goals < row.away_goals else 1})
            pending.append(row)
    while pending:
        state.update(pending.popleft())
    return pd.DataFrame(records), state


def check_columns(columns):
    forbidden = [column for column in columns if not allowed_feature(column)]
    if forbidden:
        raise ValueError(f"V4 cannot use possession or passing features: {forbidden}")


def check_bundle(bundle):
    if bundle.get("kind") != KIND:
        raise ValueError("Select a separately trained V4 outcome model.")
    check_columns(bundle["columns"])
    check_columns(bundle["base_columns"])


def fit_bundle(frame, columns, base_columns, selection, reference_selection, cutoff, parameters=None):
    check_columns(columns)
    check_columns(base_columns)
    bundle = v3.fit_bundle(frame, columns, base_columns, selection,
                           reference_selection, cutoff, parameters)
    bundle.update(kind=KIND, excluded_statistics=list(EXCLUDED_STATISTICS))
    return bundle


def predict_bundle(bundle, frame, raw=False):
    check_bundle(bundle)
    return v3.predict_bundle(bundle, frame, raw=raw)


def expected_goals(bundle, frame):
    check_bundle(bundle)
    return v3.expected_goals(bundle, frame)


def total_goal_probabilities(bundle, frame, minimum_goals=(2, 3, 4)):
    check_bundle(bundle)
    return v3.total_goal_probabilities(bundle, frame, minimum_goals)


def train(args):
    from v3_v4_accuracy_comparison.run_comparison import train_and_compare
    train_and_compare(args)


def backtest(args):
    from v3_v4_accuracy_comparison.run_comparison import regenerate_reports
    regenerate_reports(args.report_output)


def predict(args):
    bundle = joblib.load(args.model)
    check_bundle(bundle)
    country = base.normalize(args.country)
    if country not in bundle["sources"]:
        raise ValueError("Country not present in training databases.")
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None or cutoff < pd.Timestamp(bundle["training_cutoff"]):
        raise ValueError("Prediction date needs a timezone and must not precede model training.")
    source = bundle["sources"][country]
    rows, _ = read_databases([args.db or source["path"]])
    if set(rows.country) != {country}:
        raise ValueError("Database country differs from requested country.")
    _, state = build_features(rows, cutoff)
    league = args.league_id or source["league_ids"][0]
    if league not in source["league_ids"]:
        raise ValueError("League not in this country's configuration.")
    hid, home = base.resolve_team(args.home, state, country, league)
    aid, away = base.resolve_team(args.away, state, country, league)
    if hid == aid:
        raise ValueError("Choose two different teams.")
    frame = pd.DataFrame([state.inputs(country, league, source["league_ids"][0], hid, aid, cutoff)])
    probabilities = predict_bundle(bundle, frame)[0]
    goals = expected_goals(bundle, frame)
    totals = total_goal_probabilities(bundle, frame)
    print(f"{home['name']} vs {away['name']}; prediction date {cutoff.isoformat()}")
    print(f"Model training cutoff: {bundle['training_cutoff']}")
    print("V4 inputs exclude possession and all passing statistics.")
    if goals is not None:
        h, a = goals[0][0], goals[1][0]
        print(f"Expected goals (xG, goal-rate component): {h:.3f} - {a:.3f} (total {h + a:.3f})")
    for label, value in zip(base.LABELS, probabilities):
        print(f"{label}: {value:.2%} (fair odds {1 / value:.2f})")
    print(f"Predicted outcome: {base.LABELS[int(probabilities.argmax())]}")
    if totals is not None:
        for minimum, value in zip((2, 3, 4), totals[0]):
            under = 1 - value
            over_odds = 1 / value if value > 0 else float("inf")
            under_odds = 1 / under if under > 0 else float("inf")
            print(f"At least {minimum} goals (over {minimum - .5:.1f}): {value:.2%} (fair odds {over_odds:.2f})")
            print(f"Under {minimum - .5:.1f} goals: {under:.2%} (fair odds {under_odds:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Fit V4 and compare annual V3/V4 backtests on the same history")
    t.add_argument("--db", type=Path, nargs="+")
    t.add_argument("--baseline-model", type=Path, default=ROOT / "new_model_v3/football_model.joblib")
    t.add_argument("--as-of")
    t.add_argument("--test-year", type=int, default=2025)
    t.add_argument("--confirm-year", type=int, default=2026)
    t.add_argument("--output", type=Path, default=ROOT / "new_model_v4")
    t.add_argument("--report-output", type=Path, default=ROOT / "v3_v4_accuracy_comparison")
    b = sub.add_parser("backtest", help="Verify and summarize the saved chronological comparison")
    b.add_argument("--report-output", type=Path, default=ROOT / "v3_v4_accuracy_comparison")
    p = sub.add_parser("predict")
    p.add_argument("--model", type=Path, default=ROOT / "new_model_v4/football_model.joblib")
    p.add_argument("--country", required=True)
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--db", type=Path)
    p.add_argument("--league-id", type=int)
    p.add_argument("--as-of")
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            globals()[args.command](args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
