"""Experimental direct-total and ordinal corner model; leaves v1 untouched.

python football_corners_v2.py train
python football_corners_v2.py predict --country usa --home "Seattle Sounders" --away "Sporting Kansas City"

SQLite ingestion and original historical team statistics are shared read-only
with football_corners.py. Models and reports live in corners_model_v2/.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import football_corners as base

KIND = "direct_total_ordinal_corners_v2"
CUTS = np.arange(3, 15)
GRID = np.arange(41)
ALL_INPUT_STATS = {"corners", "corners_against", "shots", "shots_against", "blocked_shots", "blocked_shots_against",
                   "shots_on_target", "shots_on_target_against", "possession", "passes", "pass_accuracy", "xg", "goals", "goals_against"}
ALL_INPUT_KEYS = {f"own_all_{stat}_{half}" for stat in ALL_INPUT_STATS for half in (5, 20)}
VENUE_INPUT_KEYS = {f"own_venue_{stat}_20" for stat in ("corners", "corners_against", "shots", "shots_against", "blocked_shots")}
NON_FEATURES = {"country", "fixture_id", "kickoff", "league_id", "primary_league",
                "season", "home_name", "away_name", "target_home", "target_away", "target_total"}


class MatchHistory(base.CornerHistory):
    def __init__(self):
        super().__init__()
        self.total_history = defaultdict(lambda: {"all": deque(maxlen=40), "home": deque(maxlen=20), "away": deque(maxlen=20)})
        self.league_totals = defaultdict(lambda: deque(maxlen=300))

    def update(self, row):
        super().update(row)
        home, away = row.home_stats.get("corners"), row.away_stats.get("corners")
        if home is None or away is None:
            return
        total = home + away
        self.league_totals[row.league].append(total)
        for tid, venue in ((row.home_id, "home"), (row.away_id, "away")):
            history = self.total_history[(row.country, int(tid))]
            history["all"].append(total)
            history[venue].append(total)

    def match_inputs(self, country, league_id, primary, hid, aid, kickoff):
        league = f"{country}:{league_id}"
        h = self.inputs(country, league_id, primary, hid, aid, "home", kickoff)
        a = self.inputs(country, league_id, primary, aid, hid, "away", kickoff)
        league_values = np.asarray(self.league_totals[league], float)
        default = self.league_mean(league, "home") + self.league_mean(league, "away")
        recent = float((league_values[-80:].sum() + 30 * default) / (min(len(league_values), 80) + 30))
        x = {"league": league, "elo_diff": h["elo_diff"], "league_total": default,
             "league_recent_total": recent, "league_recent_sd": float(league_values.std()) if len(league_values) > 10 else math.sqrt(default),
             "matchup_total": h["corner_matchup_20"] + a["corner_matchup_20"]}
        for label, inputs, tid, venue in (("home", h, hid, "home"), ("away", a, aid, "away")):
            for name, value in inputs.items():
                if name in ("own_games", "own_rest"):
                    x[f"{label}_{name}"] = value
                elif name in ALL_INPUT_KEYS:
                    x[f"{label}_{name}"] = value
                elif name in VENUE_INPUT_KEYS:
                    x[f"{label}_{name}"] = value
                elif "coverage" in name and name.startswith("own_all_") and name.endswith("_20"):
                    x[f"{label}_{name}"] = value
            for scope in ("all", venue):
                values = np.asarray(self.total_history[(country, int(tid))][scope], float)
                scope_name = "all" if scope == "all" else "venue"
                for window in (5, 20):
                    v = values[-window:]
                    x[f"{label}_{scope_name}_total_mean_{window}"] = float((v.sum() + 5 * recent) / (len(v) + 5))
                x[f"{label}_{scope_name}_total_sd"] = float(values.std()) if len(values) > 3 else math.sqrt(default)
                x[f"{label}_{scope_name}_observations"] = float(len(values))
                for cut in (6, 7, 8, 9):
                    prior_rate = float(np.mean(league_values > cut)) if len(league_values) >= 20 else float(poisson.sf(cut, default))
                    x[f"{label}_{scope_name}_rate_over_{cut}"] = float((np.sum(values > cut) + 20 * prior_rate) / (len(values) + 20))
        x["historical_total"] = (x["home_all_total_mean_20"] + x["away_all_total_mean_20"]) / 2
        x["tempo_change"] = (x["home_all_total_mean_5"] + x["away_all_total_mean_5"]) / 2 - x["historical_total"]
        return x


def build_match_features(rows, as_of=None):
    rows = rows.sort_values(["kickoff", "country", "fixture_id"])
    if as_of is not None:
        rows = rows[rows.kickoff + base.RESULT_DELAY <= as_of]
    state, pending, records = MatchHistory(), deque(), []
    for kickoff, batch in rows.groupby("kickoff", sort=True):
        while pending and pending[0].kickoff + base.RESULT_DELAY <= kickoff:
            state.update(pending.popleft())
        for row in batch.itertuples(index=False):
            x = state.match_inputs(row.country, row.league_id, row.primary_league, row.home_id, row.away_id, kickoff)
            home, away = row.home_stats.get("corners", np.nan), row.away_stats.get("corners", np.nan)
            records.append({**x, "country": row.country, "fixture_id": row.fixture_id, "kickoff": kickoff,
                            "league_id": row.league_id, "primary_league": row.primary_league, "season": row.season,
                            "home_name": row.home_name, "away_name": row.away_name,
                            "target_home": home, "target_away": away, "target_total": home + away})
            pending.append(row)
    while pending:
        state.update(pending.popleft())
    return pd.DataFrame(records), state


def preprocessing(columns, linear):
    numeric = [c for c in columns if c != "league"]
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if linear:
        steps.append(StandardScaler())
    return ColumnTransformer([
        ("numeric", make_pipeline(*steps), numeric),
        ("league", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["league"])], sparse_threshold=0)


def fit_count(frame, columns, spec):
    if spec["family"] in ("matchup", "historical"):
        return None
    linear = spec["family"] == "poisson"
    estimator = (PoissonRegressor(alpha=spec["alpha"], max_iter=400) if linear else
                 HistGradientBoostingRegressor(loss=spec["loss"], max_leaf_nodes=spec["leaves"],
                     max_iter=spec["iterations"], learning_rate=.04, min_samples_leaf=100,
                     l2_regularization=30, early_stopping=False, random_state=42))
    model = make_pipeline(preprocessing(columns, linear), estimator)
    model.fit(frame[columns], frame.target_total)
    return model


def predict_count(model, frame, columns, spec):
    if spec["family"] == "matchup":
        return frame.matchup_total.to_numpy()
    if spec["family"] == "historical":
        return frame.historical_total.to_numpy()
    return np.maximum(model.predict(frame[columns]), 1e-6)


def estimate_dispersion(frame, mean):
    def fit(y, mu):
        def objective(alpha):
            return -float(nbinom.logpmf(y, n=1 / alpha, p=1 / (1 + alpha * mu)).mean())
        result = minimize_scalar(objective, bounds=(1e-5, .5), method="bounded")
        return float(result.x) if result.success and objective(result.x) < -float(poisson.logpmf(y, mu).mean()) else 0.
    y = frame.target_total.to_numpy()
    global_alpha = fit(y, mean)
    leagues = {}
    for league in frame.league.unique():
        mask = frame.league.eq(league).to_numpy()
        n = int(mask.sum())
        if n < 30:
            continue
        local = fit(y[mask], mean[mask])
        leagues[league] = float((n * local + 300 * global_alpha) / (n + 300))
    return {"global": global_alpha, "leagues": leagues, "calibration_matches": len(frame)}


def distribution_cdf(frame, mean, calibration):
    alpha = np.array([calibration["leagues"].get(league, calibration["global"]) for league in frame.league])
    cdf = poisson.cdf(GRID[None, :], mean[:, None])
    mask = alpha > 1e-8
    if mask.any():
        cdf[mask] = nbinom.cdf(GRID[None, :], n=1 / alpha[mask, None], p=1 / (1 + alpha[mask, None] * mean[mask, None]))
    return cdf


def coherent_cdf(base_cdf, threshold_cdf, cuts):
    """Project threshold CDFs to a monotone sequence, then retain scaled NB tails."""
    base_cdf = np.asarray(base_cdf, float)
    threshold_cdf = np.asarray(threshold_cdf, float)
    cuts = np.asarray(cuts, int)
    if base_cdf.ndim != 2 or threshold_cdf.shape != (len(base_cdf), len(cuts)):
        raise ValueError("Invalid CDF shape.")
    if not len(cuts) or np.any(np.diff(cuts) <= 0) or cuts[0] < 0 or cuts[-1] >= base_cdf.shape[1]:
        raise ValueError("CDF cutpoints must be ordered grid indices.")
    if not np.isfinite(base_cdf).all() or not np.isfinite(threshold_cdf).all():
        raise ValueError("CDF values must be finite.")
    projected = np.array([isotonic_regression(row, y_min=0., y_max=1., increasing=True) for row in threshold_cdf])
    result = base_cdf.copy()
    first, last = int(cuts[0]), int(cuts[-1])
    result[:, :first] *= (projected[:, 0] / np.maximum(base_cdf[:, first], 1e-12))[:, None]
    result[:, last + 1:] = 1 - (1 - base_cdf[:, last + 1:]) * ((1 - projected[:, -1]) / np.maximum(1 - base_cdf[:, last], 1e-12))[:, None]
    for row in range(len(result)):
        result[row, first:last + 1] = np.interp(np.arange(first, last + 1), cuts, projected[row])
    return np.maximum.accumulate(np.clip(result, 0., 1.), axis=1)


def fit_ordinal(frame, columns, C):
    processor = preprocessing(columns, True)
    x = processor.fit_transform(frame[columns])
    models = []
    for cut in CUTS:
        target = (frame.target_total.to_numpy() <= cut).astype(int)
        if np.unique(target).size < 2:
            models.append(float(target.mean()))
        else:
            model = LogisticRegression(C=C, max_iter=400)
            model.fit(x, target)
            models.append(model)
    return {"processor": processor, "models": models}


def ordinal_cdf(model, frame, columns, scaffold):
    x = model["processor"].transform(frame[columns])
    probabilities = np.column_stack([np.full(len(frame), m) if isinstance(m, float) else m.predict_proba(x)[:, 1] for m in model["models"]])
    return coherent_cdf(scaffold, probabilities, CUTS)


def probability_metrics(y, cdf):
    y = np.asarray(y, int)
    pmf = np.diff(np.column_stack([np.zeros(len(cdf)), cdf]), axis=1)
    likelihood = np.where(y <= GRID[-1], pmf[np.arange(len(y)), np.minimum(y, GRID[-1])], 1 - cdf[:, -1])
    events = {}
    for minimum in (7, 8, 9):
        actual = y >= minimum
        p = np.clip(1 - cdf[:, minimum - 1], 1e-12, 1 - 1e-12)
        events[str(minimum)] = {"brier": float(np.mean((p - actual) ** 2)),
                               "log_loss": float(-np.mean(actual * np.log(p) + (1 - actual) * np.log1p(-p)))}
    return {"rps": float(np.mean(np.sum((cdf - (y[:, None] <= GRID[None, :])) ** 2, axis=1))),
            "nll": float(-np.log(np.maximum(likelihood, 1e-12)).mean()),
            "mean_brier_7_8_9": float(np.mean([v["brier"] for v in events.values()])), "events": events}


def fit_bundle(frame, columns, selection, calibration):
    return {"kind": KIND, "columns": columns, "selection": selection, "calibration": calibration,
            "count_model": fit_count(frame, columns, selection["count"]),
            "ordinal_model": fit_ordinal(frame, columns, selection["C"]) if selection["ordinal_weight"] > 0 else None}


def predict_bundle(bundle, frame, reference_inputs=None):
    distribution_mean = predict_count(bundle["count_model"], frame, bundle["columns"], bundle["selection"]["count"])
    scaffold = distribution_cdf(frame, distribution_mean, bundle["calibration"])
    weight = bundle["selection"]["ordinal_weight"]
    if weight > 0:
        direct = ordinal_cdf(bundle["ordinal_model"], frame, bundle["columns"], scaffold)
        scaffold = (1 - weight) * scaffold + weight * direct
    mean = distribution_mean
    if "point_reference_model" in bundle:
        if reference_inputs is None:
            raise ValueError("The selected reference count estimator needs both teams' historical inputs.")
        reference_counts = base.predict_bundle(bundle["point_reference_model"], reference_inputs)
        if len(reference_counts) != 2 * len(frame):
            raise ValueError("Need exactly two reference input rows per match.")
        mean = reference_counts.reshape(-1, 2).sum(axis=1)
    return mean, scaffold


def event_filters(y, cdf):
    report = {}
    for line in (6, 7, 8):
        p = 1 - cdf[:, line]
        report[str(line)] = {}
        for threshold in (.6, .7, .8):
            mask = p >= threshold
            n = int(mask.sum())
            correct = int((y[mask] > line).sum())
            report[str(line)][str(threshold)] = {"selected": n, "correct": correct,
                "incorrect": n - correct, "accuracy": correct / n if n else None}
    return report


def evaluate(frame, mean, cdf):
    y = frame.target_total.to_numpy()
    return {"counts": base.count_metrics(y, mean), "probabilities": probability_metrics(y, cdf),
            "over_filters": event_filters(y, cdf)}


def select_model(frame, columns, validation_year):
    val_start = pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
    val_end = pd.Timestamp(f"{validation_year + 1}-01-01", tz="UTC")
    cal_start = pd.Timestamp(f"{validation_year - 1}-01-01", tz="UTC")
    history = frame[frame.kickoff + base.RESULT_DELAY <= val_start]
    early = frame[frame.kickoff + base.RESULT_DELAY <= cal_start]
    calibration_rows = frame[(frame.kickoff >= cal_start) & (frame.kickoff + base.RESULT_DELAY <= val_start)]
    validation = frame[(frame.kickoff >= val_start) & (frame.kickoff + base.RESULT_DELAY <= val_end)
                       & (frame.league_id == frame.primary_league)]
    if min(len(history), len(early), len(calibration_rows), len(validation)) < 100:
        raise ValueError("Insufficient chronological training/calibration/validation matches.")
    specs = [{"family": "matchup"}, {"family": "historical"}]
    specs += [{"family": "poisson", "alpha": a} for a in (.01, .1, 1.)]
    specs += [{"family": "boosting", "loss": loss, "leaves": 7, "iterations": 160} for loss in ("poisson", "squared_error")]
    specs += [{"family": "boosting", "loss": "poisson", "leaves": 15, "iterations": 200}]
    counts = []
    for spec in specs:
        model = fit_count(history, columns, spec)
        mean = predict_count(model, validation, columns, spec)
        score = base.count_metrics(validation.target_total.to_numpy(), mean)
        counts.append({"spec": spec, **score})
        print(f"Validation count {spec}: MAE={score['mae']:.4f}", flush=True)
    count_spec = min(counts, key=lambda r: (r["mae"], r["rmse"]))["spec"]
    early_model = fit_count(early, columns, count_spec)
    early_mean = predict_count(early_model, calibration_rows, columns, count_spec)
    calibration = estimate_dispersion(calibration_rows, early_mean)
    count_model = fit_count(history, columns, count_spec)
    mean = predict_count(count_model, validation, columns, count_spec)
    scaffold = distribution_cdf(validation, mean, calibration)
    probability_records = [{"C": None, "ordinal_weight": 0., **probability_metrics(validation.target_total.to_numpy(), scaffold)}]
    for C in (.01, .1):
        ordinal = fit_ordinal(history, columns, C)
        direct = ordinal_cdf(ordinal, validation, columns, scaffold)
        # Keep a small positive scaffold mass for exact counts after projection.
        for weight in (.25, .5, .75, .95):
            cdf = (1 - weight) * scaffold + weight * direct
            score = probability_metrics(validation.target_total.to_numpy(), cdf)
            probability_records.append({"C": C, "ordinal_weight": weight, **score})
            print(f"Validation ordinal C={C}, weight={weight}: Brier={score['mean_brier_7_8_9']:.5f}, RPS={score['rps']:.4f}", flush=True)
    best = min(probability_records, key=lambda r: (r["mean_brier_7_8_9"], r["rps"]))
    selection = {"count": count_spec, "C": best["C"], "ordinal_weight": best["ordinal_weight"]}
    print(f"Frozen selection from {validation_year}: {selection}", flush=True)
    return selection, calibration, {"count_candidates": counts, "probability_candidates": probability_records,
                                   "validation_matches": len(validation), "calibration_matches": len(calibration_rows)}


def legacy_cdf(frame, mean, alpha):
    return poisson.cdf(GRID[None, :], mean[:, None]) if alpha < 1e-8 else nbinom.cdf(GRID[None, :], n=1 / alpha, p=1 / (1 + alpha * mean[:, None]))


def paired_report(frame, mean, cdf, reference_mean, reference_cdf):
    difference = abs(mean - frame.target_total.to_numpy()) - abs(reference_mean - frame.target_total.to_numpy())
    y = frame.target_total.to_numpy()
    brier_difference = np.mean(np.column_stack([
        ((1 - cdf[:, line]) - (y > line)) ** 2
        - ((1 - reference_cdf[:, line]) - (y > line)) ** 2
        for line in (6, 7, 8)]), axis=1)
    weeks = frame.kickoff.dt.strftime("%G-%V")
    blocks = pd.DataFrame({"week": weeks.to_numpy(), "diff": difference, "brier": brier_difference}).groupby("week").agg(
        total=("diff", "sum"), brier_total=("brier", "sum"), n=("diff", "size"))
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(blocks), (5000, len(blocks)))
    boot = blocks.total.to_numpy()[samples].sum(1) / blocks.n.to_numpy()[samples].sum(1)
    brier_boot = blocks.brier_total.to_numpy()[samples].sum(1) / blocks.n.to_numpy()[samples].sum(1)
    output = {"v2": evaluate(frame, mean, cdf), "v1": evaluate(frame, reference_mean, reference_cdf),
              "mae_difference_95pct_week_bootstrap": np.quantile(boot, [.025, .975]).tolist(),
              "mean_brier_difference": float(brier_difference.mean()),
              "mean_brier_difference_95pct_week_bootstrap": np.quantile(brier_boot, [.025, .975]).tolist(),
              "leagues": {}}
    for country in frame.country.unique():
        mask = frame.country.eq(country).to_numpy()
        output["leagues"][country] = {"v2": evaluate(frame.loc[mask], mean[mask], cdf[mask]),
                                      "v1": evaluate(frame.loc[mask], reference_mean[mask], reference_cdf[mask])}
    return output


def train(args):
    if args.validation_year >= args.test_year or args.confirm_year <= args.test_year:
        raise ValueError("Require validation-year < test-year < confirm-year.")
    protected = [Path("football_corners.py"), Path("corners_model/corners_model.joblib"), Path("corners_model/evaluation_model.joblib")]
    checksums = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    old_folders = [Path(name).resolve() for name in ("corners_model", "corners_model_with_passes", "corners_model_without_passes", "corners_backtest_2025")]
    if any(args.output.resolve() == p or p in args.output.resolve().parents for p in old_folders):
        raise ValueError("Use a separate v2 output folder.")
    rows, sources = base.read_databases(args.db or sorted(Path('.').glob('*.sqlite3')))
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None:
        raise ValueError("--as-of requires a timezone.")
    rows = rows[rows.kickoff + base.RESULT_DELAY <= cutoff]
    print(f"Building direct match features for {len(rows)} fixtures...", flush=True)
    all_frame, _ = build_match_features(rows)
    frame = all_frame[all_frame.target_total.notna()].reset_index(drop=True)
    columns = [c for c in frame if c not in NON_FEATURES]
    selection, calibration, validation_report = select_model(frame, columns, args.validation_year)
    # Build v1 inputs separately; never mutate v1 or use its latest all-data fit.
    print("Building the identical-fixture v1 reference...", flush=True)
    side_frame, _, = base.build_features(rows)
    v1_report = json.loads(Path("corners_model/evaluation.json").read_text(encoding="utf-8"))
    v1_selection = v1_report["selection"]
    ref_columns = v1_report["features"]
    # Keep the established point estimator when it wins the same internal
    # validation; probability learning remains independent of this fallback.
    val_start = pd.Timestamp(f"{args.validation_year}-01-01", tz="UTC")
    val_end = pd.Timestamp(f"{args.validation_year + 1}-01-01", tz="UTC")
    v1_val_train = side_frame[side_frame.kickoff + base.RESULT_DELAY <= val_start]
    v1_val_test = side_frame[(side_frame.kickoff >= val_start) & (side_frame.kickoff + base.RESULT_DELAY <= val_end)
                             & (side_frame.league_id == side_frame.primary_league)]
    v1_val_model = base.fit_bundle(v1_val_train, ref_columns, v1_selection)
    v1_val_mean = base.pair_predictions(v1_val_test, base.predict_bundle(v1_val_model, v1_val_test))
    v1_val_mae = base.count_metrics(v1_val_mean.target_total.to_numpy(), v1_val_mean.expected_total.to_numpy())["mae"]
    direct_val_mae = min(r["mae"] for r in validation_report["count_candidates"])
    selection["point_source"] = "v1" if v1_val_mae < direct_val_mae else "direct_total"
    validation_report["point_reference_mae"] = v1_val_mae
    print(f"Point estimator selected on {args.validation_year}: {selection['point_source']}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"sources": sources, "selection": selection, "validation": validation_report,
              "feature_count": len(columns), "training_cutoff": cutoff.isoformat(),
              "notes": f"{args.test_year} is a previously inspected comparison year; {args.confirm_year} is confirmation with settings frozen from {args.validation_year}.",
              "periods": {}, "protected_sha256": checksums}
    for year in (args.test_year, args.confirm_year):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), cutoff)
        training = frame[frame.kickoff + base.RESULT_DELAY <= start]
        test = frame[(frame.kickoff >= start) & (frame.kickoff < end) & (frame.league_id == frame.primary_league)]
        if len(test) < 100:
            raise ValueError(f"Need at least 100 complete corner totals in {year}.")
        bundle = fit_bundle(training, columns, selection, calibration)
        distribution_mean, cdf = predict_bundle(bundle, test)
        ref_train = side_frame[side_frame.kickoff + base.RESULT_DELAY <= start]
        ref_test = side_frame[(side_frame.kickoff >= start) & (side_frame.kickoff < end) & (side_frame.league_id == side_frame.primary_league)]
        reference = base.fit_bundle(ref_train, ref_columns, v1_selection)
        ref_predictions = base.pair_predictions(ref_test, base.predict_bundle(reference, ref_test))
        keys = pd.MultiIndex.from_frame(test[["country", "fixture_id"]])
        reference_rows = ref_predictions.set_index(["country", "fixture_id"]).loc[keys]
        np.testing.assert_allclose(test.target_total.to_numpy(), reference_rows.target_total.to_numpy())
        reference_mean = reference_rows.expected_total.to_numpy()
        reference_distribution = legacy_cdf(test, reference_mean, v1_report["dispersion"]["total"])
        mean = reference_mean if selection["point_source"] == "v1" else distribution_mean
        if selection["point_source"] == "v1":
            bundle["point_reference_model"] = reference
        comparison = paired_report(test, mean, cdf, reference_mean, reference_distribution)
        comparison.update({"start": start.isoformat(), "end": end.isoformat(), "dispersion": calibration})
        comparison["actual_counts_above_distribution_grid"] = int((test.target_total > GRID[-1]).sum())
        report["periods"][str(year)] = comparison
        export = test[[c for c in NON_FEATURES if c in test]].copy()
        export["expected_total_v2"], export["expected_total_v1"] = mean, reference_mean
        for line in (6, 7, 8, 9):
            export[f"p_over_{line}_v2"] = 1 - cdf[:, line]
            export[f"p_over_{line}_v1"] = 1 - reference_distribution[:, line]
        export.to_csv(args.output / f"test_predictions_{year}.csv", index=False)
        bundle.update({"sources": sources, "training_cutoff": start.isoformat()})
        joblib.dump(bundle, args.output / f"evaluation_model_{year}.joblib")
        print(f"{year}: v2 MAE={comparison['v2']['counts']['mae']:.4f}, v1={comparison['v1']['counts']['mae']:.4f}; "
              f"Brier v2={comparison['v2']['probabilities']['mean_brier_7_8_9']:.5f}, v1={comparison['v1']['probabilities']['mean_brier_7_8_9']:.5f}", flush=True)
        # Refresh dispersion only from completed out-of-training past predictions.
        if year == args.test_year:
            calibration = estimate_dispersion(test, distribution_mean)
    print("Training separate final v2 model...", flush=True)
    final = fit_bundle(frame, columns, selection, calibration)
    if selection["point_source"] == "v1":
        final["point_reference_model"] = base.fit_bundle(side_frame, ref_columns, v1_selection)
    final.update({"sources": sources, "training_cutoff": cutoff.isoformat(), "training_matches": len(frame)})
    joblib.dump(final, args.output / "corners_model.joblib")
    for path, digest in checksums.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Protected original file changed: {path}")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved independent model: {(args.output / 'corners_model.joblib').resolve()}", flush=True)


def predict(args):
    bundle = joblib.load(args.model)
    if bundle.get("kind") != KIND:
        raise ValueError("Select the independent v2 model.")
    country = base.normalize(args.country)
    if country not in bundle["sources"]:
        raise ValueError("Country not present in trained databases.")
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None or cutoff < pd.Timestamp(bundle["training_cutoff"]):
        raise ValueError("Prediction cutoff must include timezone and not precede training.")
    source = bundle["sources"][country]
    rows, _ = base.read_databases([args.db or source["path"]])
    if set(rows.country) != {country}:
        raise ValueError("Database country differs from requested country.")
    _, state = build_match_features(rows, cutoff)
    league = args.league_id or source["league_ids"][0]
    if league not in source["league_ids"]:
        raise ValueError("League not in country configuration.")
    hid, h = base.resolve_team(args.home, state, country, league)
    aid, a = base.resolve_team(args.away, state, country, league)
    if hid == aid:
        raise ValueError("Choose two different teams.")
    x = state.match_inputs(country, league, source["league_ids"][0], hid, aid, cutoff)
    reference_inputs = pd.DataFrame([
        state.inputs(country, league, source["league_ids"][0], hid, aid, "home", cutoff),
        state.inputs(country, league, source["league_ids"][0], aid, hid, "away", cutoff)])
    mean, cdf = predict_bundle(bundle, pd.DataFrame([x]), reference_inputs)
    pmf = np.diff(np.r_[0., cdf[0]])
    print(f"{h['name']} vs {a['name']}; cutoff {cutoff.isoformat()}")
    print(f"Corner count estimate: {mean[0]:.2f}")
    print("Total corner probabilities:")
    for number in range(10):
        print(f"  {number}: {pmf[number]:.2%}")
    print(f"  >9: {1 - cdf[0, 9]:.2%}")
    for line in (6, 7, 8):
        print(f"P(over {line}): {1-cdf[0,line]:.2%}; P(under {line+1}): {cdf[0,line]:.2%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--db", type=Path, nargs="+")
    t.add_argument("--validation-year", type=int, default=2024)
    t.add_argument("--test-year", type=int, default=2025)
    t.add_argument("--confirm-year", type=int, default=2026)
    t.add_argument("--as-of")
    t.add_argument("--output", type=Path, default=Path("corners_model_v2"))
    p = sub.add_parser("predict")
    p.add_argument("--model", type=Path, default=Path("corners_model_v2/corners_model.joblib"))
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
