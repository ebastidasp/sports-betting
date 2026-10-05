"""Separate SQLite match-outcome challenger; never overwrites the existing models.

python football_model_v3.py train
python football_model_v3.py predict --country colombia --home Millonarios --away "Santa Fe"

The existing v2 supplies read-only ingestion and historical features. New fitted
estimators, chronological comparisons and calibration live in new_model_v3/.
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
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import poisson, skellam
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import football_model_v2 as base

KIND = "sqlite_football_classifier_v3"
EXTRA_VALUES = ("draw", "clean_sheet", "scoreless", "goal_total", "xg_residual", "sot_residual")


class History(base.History):
    def __init__(self):
        super().__init__()
        self.extra_teams = defaultdict(lambda: {
            "history": deque(maxlen=120),
            "moments": {half: {key: [0., 0.] for key in EXTRA_VALUES} for half in base.HALFLIVES}})
        self.extra_leagues = defaultdict(lambda: deque(maxlen=300))

    def inputs(self, country, league_id, primary, hid, aid, kickoff):
        x = super().inputs(country, league_id, primary, hid, aid, kickoff)
        recent = self.extra_leagues[f"{country}:{league_id}"]
        counts = np.bincount(list(recent), minlength=3)
        priors = (counts + 40 * np.array([.45, .27, .28])) / (len(recent) + 40)
        for outcome, value in zip(("home", "draw", "away"), priors):
            x[f"v3_league_{outcome}_rate"] = float(value)
        defaults = {"draw": priors[1], "clean_sheet": .3, "scoreless": .3,
                    "goal_total": x["league_home_goals"] + x["league_away_goals"],
                    "xg_residual": 0., "sot_residual": 0.}
        for side, tid in (("home", hid), ("away", aid)):
            extra = self.extra_teams[(country, int(tid))]
            for half in base.HALFLIVES:
                for key in EXTRA_VALUES:
                    total, mass = extra["moments"][half][key]
                    x[f"v3_{side}_{key}_{half}"] = float((total + 10 * defaults[key]) / (mass + 10))
                    if key in ("xg_residual", "sot_residual"):
                        x[f"v3_{side}_{key}_coverage_{half}"] = float(mass)
        x["v3_elo_balance"] = abs(x["elo_diff_10"])
        x["v3_expected_goal_total"] = x["home_attack_vs_defense_20"] + x["away_attack_vs_defense_20"]
        x["v3_recent_goal_balance"] = abs(x["home_attack_vs_defense_5"] - x["away_attack_vs_defense_5"])
        return x

    def update(self, row):
        h = self.team(row.country, row.home_id, row.league_id, row.primary_league)
        a = self.team(row.country, row.away_id, row.league_id, row.primary_league)
        diff = (h["elo"][10] - a["elo"][10]) / 400
        lh, la = self.league_rates(row.league)
        for tid, goals, conceded, stats, expected, strength in (
            (row.home_id, row.home_goals, row.away_goals, row.home_stats, lh * math.exp(np.clip(.45 * diff, -1.5, 1.5)), diff),
            (row.away_id, row.away_goals, row.home_goals, row.away_stats, la * math.exp(np.clip(-.45 * diff, -1.5, 1.5)), -diff)):
            item = {"draw": float(goals == conceded), "clean_sheet": float(conceded == 0),
                    "scoreless": float(goals == 0), "goal_total": float(goals + conceded)}
            for stat, key, prior in (("xg", "xg_residual", expected),
                                     ("shots_on_target", "sot_residual", 4 * math.exp(np.clip(.3 * strength, -1.5, 1.5)))):
                value = stats.get(stat)
                if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                    item[key] = float(value - prior)
            extra = self.extra_teams[(row.country, int(tid))]
            removed = extra["history"][0] if len(extra["history"]) == 120 else {}
            for half in base.HALFLIVES:
                decay = 2 ** (-1 / half)
                for key, moment in extra["moments"][half].items():
                    moment[0] *= decay
                    moment[1] *= decay
                    if key in item:
                        moment[0] += item[key]
                        moment[1] += 1
                    if key in removed:
                        moment[0] -= decay ** 120 * removed[key]
                        moment[1] -= decay ** 120
            extra["history"].append(item)
        self.extra_leagues[row.league].append(0 if row.home_goals > row.away_goals else 2 if row.home_goals < row.away_goals else 1)
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
            inputs = state.inputs(row.country, row.league_id, row.primary_league, row.home_id, row.away_id, kickoff)
            records.append({**inputs, "country": row.country, "fixture_id": row.fixture_id, "kickoff": kickoff,
                "season": row.season, "league_id": row.league_id, "primary_league": row.primary_league,
                "home_name": row.home_name, "away_name": row.away_name,
                "home_goals": row.home_goals, "away_goals": row.away_goals,
                "target": 0 if row.home_goals > row.away_goals else 2 if row.home_goals < row.away_goals else 1})
            pending.append(row)
    while pending:
        state.update(pending.popleft())
    return pd.DataFrame(records), state


def pooled_inputs(frame, columns):
    blocks = []
    for side, other in (("home", "away"), ("away", "home")):
        data = {"league": frame.league.to_numpy(), "is_home": np.full(len(frame), float(side == "home"))}
        for column in columns:
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


def fit_component(frame, columns, base_columns, spec, reference_selection, cutoff):
    family = spec["family"]
    used = base_columns if spec.get("feature_set") == "base" or family in ("v2", "goals") else columns
    if family == "v2":
        return base.fit_bundle(frame, base_columns, reference_selection)
    if family == "goals":
        if spec.get("shared"):
            x = pooled_inputs(frame, used)
            model = make_pipeline(base.preprocessing(list(x), True), PoissonRegressor(alpha=spec["alpha"], max_iter=350))
            model.fit(x, np.r_[frame.home_goals, frame.away_goals])
            return {"model": model, "columns": list(x)}
        models = []
        for side in ("home", "away"):
            model = make_pipeline(base.preprocessing(used, True), PoissonRegressor(alpha=spec["alpha"], max_iter=350))
            model.fit(frame[used], frame[f"{side}_goals"])
            models.append(model)
        return {"models": models}
    if family == "logistic":
        estimator = LogisticRegression(C=spec["C"], max_iter=500)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7, learning_rate=.05,
            l2_regularization=20, min_samples_leaf=80, early_stopping=False, random_state=42)
    model = make_pipeline(base.preprocessing(used, family == "logistic"), estimator)
    if spec.get("half_life_days"):
        age = (cutoff - frame.kickoff).dt.total_seconds().to_numpy() / 86400
        weights = np.exp2(-age / spec["half_life_days"])
        weights /= weights.mean()
        model.fit(frame[used], frame.target, **{model.steps[-1][0] + "__sample_weight": weights})
    else:
        model.fit(frame[used], frame.target)
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("All three outcomes are required in training.")
    return model


def goal_rates(model, frame, columns, spec):
    if spec.get("shared"):
        rates = np.clip(model["model"].predict(pooled_inputs(frame, columns)[model["columns"]]), .05, 8)
        return rates[:len(frame)], rates[len(frame):]
    return tuple(np.clip(m.predict(frame[columns]), .05, 8) for m in model["models"])


def goal_probabilities(home, away, rho=0.):
    home, away = np.asarray(home, float), np.asarray(away, float)
    if home.shape != away.shape or np.any(home <= 0) or np.any(away <= 0) or not np.isfinite(home + away).all():
        raise ValueError("Goal rates must be finite, positive and have matching shapes.")
    # Keep all four Dixon-Coles low-score adjustments nonnegative.
    lower = np.maximum(-1 / home, -1 / away) + 1e-10
    upper = np.minimum(1., 1 / (home * away)) - 1e-10
    local_rho = np.clip(rho, lower, upper)
    p = np.column_stack([skellam.sf(0, home, away), skellam.pmf(0, home, away), skellam.cdf(-1, home, away)])
    adjustment = np.exp(-home - away) * home * away * local_rho
    p[:, 0] += adjustment
    p[:, 1] -= 2 * adjustment
    p[:, 2] += adjustment
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(1, keepdims=True)


def fit_rho(frame, home, away):
    hg, ag = frame.home_goals.to_numpy(), frame.away_goals.to_numpy()
    coefficient = np.select([(hg == 0) & (ag == 0), (hg == 0) & (ag == 1), (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)],
                             [-home * away, home, away, -np.ones(len(frame))], default=0.)
    lower = max(-.2, float(np.max(-1 / home)), float(np.max(-1 / away))) + 1e-8
    upper = min(.2, float(np.min(1 / (home * away))), 1.) - 1e-8
    fit = minimize_scalar(lambda rho: -np.log1p(rho * coefficient).mean(), bounds=(lower, upper), method="bounded")
    return float(fit.x) if fit.success else 0.


def component_probabilities(model, frame, columns, base_columns, spec):
    if spec["family"] == "v2":
        return base.predict_bundle(model, frame)
    if spec["family"] == "goals":
        return goal_probabilities(*goal_rates(model, frame, base_columns, spec), spec.get("rho", 0.))
    used = base_columns if spec.get("feature_set") == "base" else columns
    return model.predict_proba(frame[used])


def calibrate(p, parameters):
    p = np.asarray(p, float)
    parameters = np.asarray(parameters, float)
    if p.ndim != 2 or p.shape[1] != 3 or parameters.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Expected finite three-outcome probabilities and three calibration parameters.")
    if not np.isfinite(parameters).all() or np.any(p.sum(1) <= 0):
        raise ValueError("Invalid calibration parameters or empty probability row.")
    p = p / p.sum(1, keepdims=True)
    logits = np.log(np.clip(p, 1e-12, 1)) / np.exp(parameters[0]) + np.r_[parameters[1:], 0.]
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_calibration(p, y, method):
    if method == "none":
        return [0., 0., 0.]
    y = np.asarray(y, int)
    def objective(parameters):
        q = calibrate(p, parameters)
        return float(-np.log(q[np.arange(len(y)), y]).mean() + .01 * np.sum(np.asarray(parameters) ** 2))
    bounds = [(-1.4, 1.4)] + ([(0., 0.), (0., 0.)] if method == "temperature" else [(-.5, .5), (-.5, .5)])
    fit = minimize(objective, np.zeros(3), method="L-BFGS-B", bounds=bounds)
    return fit.x.tolist() if fit.success else [0., 0., 0.]


def fit_bundle(frame, columns, base_columns, selection, reference_selection, cutoff, parameters=None):
    return {"kind": KIND, "columns": columns, "base_columns": base_columns, "selection": selection,
            "reference_selection": reference_selection, "calibration_parameters": parameters or [0., 0., 0.],
            "models": [fit_component(frame, columns, base_columns, spec, reference_selection, cutoff) for spec in selection["components"]]}


def predict_bundle(bundle, frame, raw=False):
    p = sum(weight * component_probabilities(model, frame, bundle["columns"], bundle["base_columns"], spec)
            for model, spec, weight in zip(bundle["models"], bundle["selection"]["components"], bundle["selection"]["weights"]))
    p = p / p.sum(1, keepdims=True)
    return p if raw else calibrate(p, bundle["calibration_parameters"])


def expected_goals(bundle, frame):
    """Goal-component forecasts; classifier probabilities have no goal-count mean."""
    components = [(model, spec, weight) for model, spec, weight in
        zip(bundle["models"], bundle["selection"]["components"], bundle["selection"]["weights"])
        if spec["family"] == "goals" and weight > 0]
    if not components:
        return None
    home, away, mass = np.zeros(len(frame)), np.zeros(len(frame)), 0.
    for model, spec, weight in components:
        h, a = goal_rates(model, frame, bundle["base_columns"], spec)
        home += weight * h
        away += weight * a
        mass += weight
    return home / mass, away / mass


def total_goal_probabilities(bundle, frame, minimum_goals=(2, 3, 4)):
    """P(total >= minimum), using the normalized mixture of goal components."""
    minimum = np.asarray(minimum_goals)
    if minimum.ndim != 1 or not len(minimum) or not np.isfinite(minimum).all() or np.any(minimum < 2) or np.any(minimum != np.floor(minimum)):
        raise ValueError("Minimum goal counts must be integers of at least two.")
    minimum = minimum.astype(int)
    probabilities, mass = np.zeros((len(frame), len(minimum))), 0.
    for model, spec, weight in zip(bundle["models"], bundle["selection"]["components"], bundle["selection"]["weights"]):
        if spec["family"] != "goals" or weight <= 0:
            continue
        home, away = goal_rates(model, frame, bundle["base_columns"], spec)
        p = poisson.sf(minimum[None, :] - 1, (home + away)[:, None])
        if np.any(minimum == 2):
            lower = np.maximum(-1 / home, -1 / away) + 1e-10
            upper = np.minimum(1., 1 / (home * away)) - 1e-10
            rho = np.clip(spec.get("rho", 0.), lower, upper)
            # Of the four corrected scorelines, only 1-1 reaches two goals.
            adjustment = np.exp(-home - away) * home * away * rho
            p[:, minimum == 2] -= adjustment[:, None]
        probabilities += weight * p
        mass += weight
    return np.clip(probabilities / mass, 0., 1.) if mass > 0 else None


def metrics(frame, p):
    result = base.metrics(frame, p)
    confidence, prediction = p.max(1), p.argmax(1)
    correct = prediction == frame.target.to_numpy()
    result["confidence_filters"] = {}
    for threshold in (.6, .7, .8):
        mask = confidence >= threshold
        n = int(mask.sum())
        result["confidence_filters"][str(threshold)] = {"selected": n, "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()) if n else None,
            "coverage": n / len(frame), "mean_confidence": float(confidence[mask].mean()) if n else None}
    result["outcomes"] = {}
    for i, label in enumerate(base.LABELS):
        actual, picked = frame.target.to_numpy() == i, prediction == i
        result["outcomes"][label] = {"actual": int(actual.sum()), "predicted": int(picked.sum()),
            "correct": int((actual & picked).sum()), "mean_probability": float(p[:, i].mean())}
    return result


def select_model(frame, columns, base_columns, reference_selection, year, metric):
    start, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    cal_start = pd.Timestamp(f"{year-1}-01-01", tz="UTC")
    train = frame[frame.kickoff + base.RESULT_DELAY <= start]
    early = frame[frame.kickoff + base.RESULT_DELAY <= cal_start]
    calibration = frame[(frame.kickoff >= cal_start) & (frame.kickoff + base.RESULT_DELAY <= start) & (frame.league_id == frame.primary_league)]
    val = frame[(frame.kickoff >= start) & (frame.kickoff + base.RESULT_DELAY <= end) & (frame.league_id == frame.primary_league)]
    if min(len(train), len(early), len(calibration), len(val)) < 100:
        raise ValueError("Need at least 100 chronological training, calibration and validation matches.")
    records, predictions = [], []
    specs = [{"family": "v2"}]
    specs += [{"family": "boosting", "feature_set": "base", "half_life_days": half} for half in (730, 1460)]
    specs += [{"family": "boosting", "feature_set": "enriched", "half_life_days": half} for half in (0, 730)]
    specs += [{"family": "logistic", "feature_set": "enriched", "C": c} for c in (.01, .1)]
    def record(specs, weights, p):
        r = {"components": specs, "weights": weights, **metrics(val, p)}
        records.append(r)
        predictions.append(p)
        print(f"Validation {specs}, weights={weights}: accuracy={r['accuracy']:.2%}, log loss={r['log_loss']:.5f}", flush=True)
    for spec in specs:
        model = fit_component(train, columns, base_columns, spec, reference_selection, start)
        record([spec], [1.], component_probabilities(model, val, columns, base_columns, spec))
    for alpha, shared in ((.1, False), (1., False), (.1, True)):
        spec = {"family": "goals", "alpha": alpha, "shared": shared}
        early_model = fit_component(early, columns, base_columns, spec, reference_selection, cal_start)
        rho = fit_rho(calibration, *goal_rates(early_model, calibration, base_columns, spec))
        model = fit_component(train, columns, base_columns, spec, reference_selection, start)
        home, away = goal_rates(model, val, base_columns, spec)
        for adjustment in (0., rho):
            candidate = {**spec, "rho": adjustment}
            record([candidate], [1.], goal_probabilities(home, away, adjustment))
    def key(r):
        return (-r["accuracy"], r["log_loss"]) if metric == "accuracy" else (r["log_loss"], -r["accuracy"])
    best_tree = min(range(5), key=lambda i: key(records[i]))
    best_linear = min(range(5, 7), key=lambda i: key(records[i]))
    best_goal = min(range(7, len(records)), key=lambda i: key(records[i]))
    for left, right in ((best_tree, best_goal), (best_tree, best_linear), (best_goal, best_linear)):
        for weight in (.25, .5, .75):
            record(records[left]["components"] + records[right]["components"], [weight, 1-weight],
                   weight * predictions[left] + (1-weight) * predictions[right])
    winner = min(range(len(records)), key=lambda i: key(records[i]))
    raw_selection = {k: records[winner][k] for k in ("components", "weights")}
    raw_p = predictions[winner]
    early_bundle = fit_bundle(early, columns, base_columns, raw_selection, reference_selection, cal_start)
    early_p = predict_bundle(early_bundle, calibration, raw=True)
    calibration_candidates = []
    for method in ("none", "temperature", "temperature_bias"):
        params = fit_calibration(early_p, calibration.target.to_numpy(), method)
        r = {"method": method, "parameters_from_previous_year": params, **metrics(val, calibrate(raw_p, params))}
        calibration_candidates.append(r)
    cal = min(calibration_candidates, key=key)
    selection = {**raw_selection, "calibration_method": cal["method"], "selection_metric": metric}
    # Refit calibration from held-out validation predictions for the next year.
    params = fit_calibration(raw_p, val.target.to_numpy(), cal["method"])
    return selection, params, {"matches": len(val), "candidates": records,
        "calibration_candidates": calibration_candidates, "selected_validation": cal,
        "calibration_parameter_fit_year": year-1, "calibration_method_selection_year": year}


def paired_report(frame, p, reference):
    y = frame.target.to_numpy()
    losses = {"accuracy": (p.argmax(1) == y).astype(float) - (reference.argmax(1) == y).astype(float),
              "log_loss": -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)) + np.log(np.clip(reference[np.arange(len(y)), y], 1e-12, 1)),
              "brier": np.sum((p - np.eye(3)[y]) ** 2, axis=1) - np.sum((reference - np.eye(3)[y]) ** 2, axis=1)}
    week = frame.kickoff.dt.strftime("%G-%V").to_numpy()
    report = {"new": metrics(frame, p), "current_v2": metrics(frame, reference), "paired_differences": {}, "leagues": {}}
    rng = np.random.default_rng(42)
    for name, difference in losses.items():
        blocks = pd.DataFrame({"week": week, "diff": difference}).groupby("week").agg(total=("diff", "sum"), n=("diff", "size"))
        samples = rng.integers(0, len(blocks), (5000, len(blocks)))
        bootstrap = blocks.total.to_numpy()[samples].sum(1) / blocks.n.to_numpy()[samples].sum(1)
        report["paired_differences"][name] = {"new_minus_current": float(difference.mean()),
            "95pct_week_bootstrap": np.quantile(bootstrap, [.025, .975]).tolist()}
    for country in frame.country.unique():
        mask = frame.country.eq(country).to_numpy()
        report["leagues"][country] = {"new": metrics(frame.loc[mask], p[mask]), "current_v2": metrics(frame.loc[mask], reference[mask])}
    return report


def train(args):
    if not args.validation_year < args.test_year < args.confirm_year:
        raise ValueError("Require validation-year < test-year < confirm-year.")
    reference_report = json.loads(Path("new_model/evaluation.json").read_text(encoding="utf-8"))
    reference_year = pd.Timestamp(reference_report["split"]["training_before"]).year
    if args.validation_year < reference_year:
        raise ValueError(f"The current reference was selected in {reference_year}; use validation-year >= {reference_year}.")
    old_folders = [Path(name).resolve() for name in ("new_model", "corners_model", "corners_model_v2", "corners_model_with_passes", "corners_model_without_passes")]
    if any(args.output.resolve() == p or p in args.output.resolve().parents for p in old_folders):
        raise ValueError("Choose a separate outcome model output directory.")
    protected = [Path(name) for name in ("football_poisson.py", "football_model_v2.py", "new_model/evaluation.json",
        "new_model/football_model.joblib", "new_model/evaluation_model.joblib")] + list(Path('.').glob('*_poisson.joblib'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    rows, sources = base.read_databases(args.db or sorted(Path('.').glob('*.sqlite3')))
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None:
        raise ValueError("--as-of requires a timezone.")
    rows = rows[rows.kickoff + base.RESULT_DELAY <= cutoff]
    print(f"Building previous-match inputs for {len(rows)} fixtures...", flush=True)
    frame, _ = build_features(rows)
    columns = [c for c in frame if c not in base.NON_FEATURES]
    base_columns = [c for c in columns if not c.startswith("v3_")]
    reference_selection = reference_report["selection"]
    selection, parameters, validation = select_model(frame, columns, base_columns, reference_selection,
        args.validation_year, args.selection_metric)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = {"selection": selection, "selected_before_later_evaluations": pd.Timestamp.now(tz="UTC").isoformat(),
        "validation_year": args.validation_year, "reference_selection": reference_selection,
        "initial_calibration_parameters": parameters, "initial_calibration_parameter_fit_year": args.validation_year}
    (args.output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Frozen selection from {args.validation_year}: {selection}", flush=True)
    report = {"sources": sources, "training_cutoff": cutoff.isoformat(), "training_matches": len(frame),
        "feature_count": len(columns), "selection": selection, "validation": validation,
        "selected_input_feature_count": len(columns) if any(s.get("feature_set") == "enriched" for s in selection["components"]) else len(base_columns),
        "protected_sha256": hashes, "reference_selection": reference_selection, "periods": {},
        "notes": f"{args.test_year} is previously inspected; architecture is selected on {args.validation_year} before {args.confirm_year} confirmation. Each evaluation year uses only past results. Calibration parameters refresh from previous out-of-training predictions; final calibration may use the completed confirmation period."}
    for year in (args.test_year, args.confirm_year):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year+1}-01-01", tz="UTC"), cutoff)
        training = frame[frame.kickoff + base.RESULT_DELAY <= start]
        test = frame[(frame.kickoff >= start) & (frame.kickoff < end) & (frame.league_id == frame.primary_league)]
        if len(test) < 100:
            raise ValueError(f"Need at least 100 observed outcomes in {year}.")
        bundle = fit_bundle(training, columns, base_columns, selection, reference_selection, start, parameters)
        raw_p = predict_bundle(bundle, test, raw=True)
        p = calibrate(raw_p, parameters)
        reference_model = base.fit_bundle(training, base_columns, reference_selection)
        reference = base.predict_bundle(reference_model, test)
        comparison = paired_report(test, p, reference)
        comparison.update({"start": start.isoformat(), "end": end.isoformat(), "training_matches": len(training),
            "calibration_parameters": parameters, "latest_test_kickoff": test.kickoff.max().isoformat()})
        report["periods"][str(year)] = comparison
        export = test[[c for c in base.NON_FEATURES if c in test]].copy()
        export[["p_home_v3", "p_draw_v3", "p_away_v3"]] = p
        export[["p_home_v2", "p_draw_v2", "p_away_v2"]] = reference
        export.to_csv(args.output / f"test_predictions_{year}.csv", index=False)
        bundle.update({"sources": sources, "training_cutoff": start.isoformat()})
        joblib.dump(bundle, args.output / f"evaluation_model_{year}.joblib")
        print(f"{year}: accuracy v3={comparison['new']['accuracy']:.2%}, current={comparison['current_v2']['accuracy']:.2%}; "
              f"log loss v3={comparison['new']['log_loss']:.5f}, current={comparison['current_v2']['log_loss']:.5f}", flush=True)
        parameters = fit_calibration(raw_p, test.target.to_numpy(), selection["calibration_method"])
    print("Training separate final outcome model...", flush=True)
    final = fit_bundle(frame, columns, base_columns, selection, reference_selection, cutoff, parameters)
    final.update({"sources": sources, "training_cutoff": cutoff.isoformat(), "training_matches": len(frame)})
    joblib.dump(final, args.output / "football_model.joblib")
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Existing model file changed: {path}")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved separate model: {(args.output / 'football_model.joblib').resolve()}", flush=True)


def predict(args):
    bundle = joblib.load(args.model)
    if bundle.get("kind") != KIND:
        raise ValueError("Select the separate v3 outcome model.")
    country = base.normalize(args.country)
    if country not in bundle["sources"]:
        raise ValueError("Country not present in training databases.")
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None or cutoff < pd.Timestamp(bundle["training_cutoff"]):
        raise ValueError("Prediction date needs a timezone and must not precede model training.")
    source = bundle["sources"][country]
    rows, _ = base.read_databases([args.db or source["path"]])
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
    x = state.inputs(country, league, source["league_ids"][0], hid, aid, cutoff)
    frame = pd.DataFrame([x])
    p = predict_bundle(bundle, frame)[0]
    goals = expected_goals(bundle, frame)
    total_probabilities = total_goal_probabilities(bundle, frame)
    print(f"{home['name']} vs {away['name']}; prediction date {cutoff.isoformat()}")
    print(f"Model training cutoff: {bundle['training_cutoff']}")
    if goals is not None:
        home_goals, away_goals = goals[0][0], goals[1][0]
        print(f"Expected goals (xG, goal-rate component): {home_goals:.3f} - {away_goals:.3f} "
              f"(total {home_goals + away_goals:.3f})")
    else:
        print("Expected goals (xG): unavailable for the selected classifier.")
    for label, value in zip(base.LABELS, p):
        fair = 1 / value if value > 0 else float("inf")
        print(f"{label}: {value:.2%} (fair odds {fair:.2f})")
    print(f"Predicted outcome: {base.LABELS[int(p.argmax())]}")
    if total_probabilities is not None:
        print("Total goals (goal-rate component):")
        for minimum, value in zip((2, 3, 4), total_probabilities[0]):
            fair = 1 / value if value > 0 else float("inf")
            print(f"At least {minimum} goals (over {minimum - .5:.1f}): {value:.2%} (fair odds {fair:.2f})")
    else:
        print("Total-goal probabilities: unavailable for the selected classifier.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--db", type=Path, nargs="+")
    t.add_argument("--validation-year", type=int, default=2024)
    t.add_argument("--test-year", type=int, default=2025)
    t.add_argument("--confirm-year", type=int, default=2026)
    t.add_argument("--selection-metric", choices=("accuracy", "log_loss"), default="accuracy")
    t.add_argument("--as-of")
    t.add_argument("--output", type=Path, default=Path("new_model_v3"))
    p = sub.add_parser("predict")
    p.add_argument("--model", type=Path, default=Path("new_model_v3/football_model.joblib"))
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
