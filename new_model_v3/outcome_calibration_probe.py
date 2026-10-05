"""Read-only v2 feature probe; development results stop at 2024."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import football_model_v2 as base

SPEC = {"family": "boosting", "leaves": 7, "iterations": 120}


def calibrate(p, parameters):
    parameters = np.asarray(parameters, float)
    logits = np.log(np.clip(p, 1e-12, 1)) / np.exp(parameters[0])
    logits += np.array([parameters[1], parameters[2], 0.])
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_calibration(p, y):
    y = np.asarray(y, int)
    def objective(parameters):
        q = calibrate(p, parameters)
        return float(-np.log(q[np.arange(len(y)), y]).mean() + .01 * np.sum(np.asarray(parameters) ** 2))
    result = minimize(objective, np.zeros(3), method="L-BFGS-B", bounds=[(-1.4, 1.4), (-.5, .5), (-.5, .5)])
    return result.x.tolist() if result.success else [0., 0., 0.]


def fit_weighted(frame, columns, cutoff, half_life):
    # Match v2's exact estimator/preprocessing, supplying only sample weights.
    model = base.fit_candidate(frame, columns, SPEC) if half_life is None else None
    if model is not None:
        return model
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    estimator = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7,
        learning_rate=.05, l2_regularization=20, min_samples_leaf=80,
        early_stopping=False, random_state=42)
    model = make_pipeline(base.preprocessing(columns, False), estimator)
    age_days = (cutoff - frame.kickoff).dt.total_seconds().to_numpy() / 86400
    weight = np.exp2(-age_days / half_life)
    weight /= weight.mean()
    model.fit(frame[columns], frame.target, histgradientboostingclassifier__sample_weight=weight)
    return model


def metrics(frame, p):
    result = base.metrics(frame, p)
    prediction = p.argmax(axis=1)
    correct = prediction == frame.target.to_numpy()
    confidence = p.max(axis=1)
    result["probability_filters"] = {}
    for threshold in (.6, .7, .8):
        mask = confidence >= threshold
        n = int(mask.sum())
        result["probability_filters"][str(threshold)] = {"selected": n, "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()) if n else None,
            "mean_probability": float(confidence[mask].mean()) if n else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("report.json"))
    args = parser.parse_args()
    val_start, val_end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")
    cal_start = pd.Timestamp("2023-01-01", tz="UTC")
    if args.features:
        frame = pd.read_pickle(args.features)
    else:
        rows, _ = base.read_databases(sorted(Path('.').glob('*.sqlite3')))
        rows = rows[rows.kickoff + base.RESULT_DELAY <= val_end]
        print(f"Building v2 features for {len(rows)} pre-2025 results", flush=True)
        frame, _ = base.build_features(rows)
    history = frame[frame.kickoff + base.RESULT_DELAY <= val_start]
    early = frame[frame.kickoff + base.RESULT_DELAY <= cal_start]
    calibration = frame[(frame.kickoff >= cal_start) & (frame.kickoff + base.RESULT_DELAY <= val_start)
                        & (frame.league_id == frame.primary_league)]
    validation = frame[(frame.kickoff >= val_start) & (frame.kickoff + base.RESULT_DELAY <= val_end)
                       & (frame.league_id == frame.primary_league)]
    columns = [c for c in frame.columns if c not in base.NON_FEATURES]
    report = {"training_cutoff": val_start.isoformat(), "calibration_start": cal_start.isoformat(),
        "validation_end": val_end.isoformat(), "feature_count": len(columns), "spec": SPEC,
        "calibration_matches": len(calibration), "validation_matches": len(validation),
        "notes": "Candidate results use only 2024; calibrators fit only 2023 predictions from pre-2023 models. No 2025/2026 evaluation.",
        "candidates": []}
    for half_life in (None, 730, 1460):
        print(f"Fitting pre-2023 calibration model, half-life={half_life}", flush=True)
        early_model = fit_weighted(early, columns, cal_start, half_life)
        calibration_p = base.candidate_probabilities(early_model, calibration, columns, SPEC)
        parameters = fit_calibration(calibration_p, calibration.target.to_numpy())
        print(f"Fitting pre-2024 validation model, half-life={half_life}; calibration={parameters}", flush=True)
        model = fit_weighted(history, columns, val_start, half_life)
        p = base.candidate_probabilities(model, validation, columns, SPEC)
        for calibrated in (False, True):
            q = calibrate(p, parameters) if calibrated else p
            result = {"half_life_days": half_life, "calibrated": calibrated,
                "calibration_parameters": parameters if calibrated else [0., 0., 0.], **metrics(validation, q)}
            report["candidates"].append(result)
            print(f"2024 half-life={half_life}, calibrated={calibrated}: accuracy={result['accuracy']:.4%}; log_loss={result['log_loss']:.6f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["selected_accuracy"] = min(report["candidates"], key=lambda r: (-r["accuracy"], r["log_loss"]))
    report["selected_log_loss"] = min(report["candidates"], key=lambda r: (r["log_loss"], -r["accuracy"]))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
