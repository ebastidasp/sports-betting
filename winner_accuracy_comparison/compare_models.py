"""Chronological 1X2 comparison across first and second divisions.

python winner_accuracy_comparison/compare_models.py
python winner_accuracy_comparison/compare_models.py --mode report
No API calls or changes to saved models. V2 is fitted in memory with fixed settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "goals_accuracy_comparison" / "backtest"
sys.path.insert(0, str(ROOT))
import football_model_v2 as base
import football_model_v3 as v3
import football_poisson as legacy

YEARS = (2025, 2026)
MODELS = ("legacy", "v2", "v3")
OUTCOMES = ("home", "draw", "away")
THRESHOLDS = (.6, .7, .8)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def probability_columns(model):
    return [f"p_{outcome}_{model}" for outcome in OUTCOMES]


def validate_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite home/draw/away probabilities.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probability outside [0, 1].")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)
    return probabilities


def generate_predictions():
    metadata = json.loads((ROOT / "new_model_v3" / "evaluation.json").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    bundles = {year: joblib.load(ROOT / "new_model_v3" / f"evaluation_model_{year}.joblib") for year in YEARS}
    rows, sources = base.read_databases([source["path"] for source in metadata["sources"].values()])
    rows = rows[rows.kickoff + base.RESULT_DELAY <= cutoff]
    needs_enriched = any(spec.get("feature_set") == "enriched" for bundle in bundles.values()
                         for spec in bundle["selection"]["components"])
    print(f"Building chronological inputs for {len(rows)} completed fixtures", flush=True)
    frame, _ = (v3.build_features if needs_enriched else base.build_features)(rows)
    forecast_info = {"data_cutoff": cutoff.isoformat(), "sources": sources, "periods": {},
                     "baseline": "Original football_poisson.py and fixed selected V2 classifier",
                     "new_model": "Full V3 1X2 blend, including saved calibration"}
    for year, bundle in bundles.items():
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), cutoff)
        if bundle.get("kind") != v3.KIND or pd.Timestamp(bundle["training_cutoff"]) > start:
            raise ValueError("V3 bundle must be trained before its evaluation year.")
        training = frame[frame.kickoff + base.RESULT_DELAY <= start]
        test = frame[(frame.kickoff >= start) & (frame.kickoff < end)]
        print(f"Scoring V3 {year}: {len(test)} fixtures across both divisions", flush=True)
        p_v3 = validate_probabilities(v3.predict_bundle(bundle, test))
        print(f"Fitting fixed V2 reference before {year}: {len(training)} training fixtures", flush=True)
        reference = base.fit_bundle(training, bundle["base_columns"], bundle["reference_selection"])
        p_v2 = validate_probabilities(base.predict_bundle(reference, test))
        export = test[["country", "fixture_id", "kickoff", "league_id", "primary_league", "season",
                       "home_name", "away_name", "home_goals", "away_goals", "target"]].copy()
        export["division"] = np.where(export.league_id == export.primary_league, "first", "second")
        export[probability_columns("v3")] = p_v3
        export[probability_columns("v2")] = p_v2
        # Reproduce the already reported first-division reference comparisons.
        previous = pd.read_csv(ROOT / "new_model_v3" / f"test_predictions_{year}.csv").sort_values(["country", "fixture_id"])
        primary = export[export.division == "first"].sort_values(["country", "fixture_id"])
        np.testing.assert_array_equal(primary[["country", "fixture_id"]].to_numpy(), previous[["country", "fixture_id"]].to_numpy())
        for model in ("v2", "v3"):
            np.testing.assert_allclose(primary[probability_columns(model)].to_numpy(),
                                       previous[probability_columns(model)].to_numpy(), atol=1e-10, rtol=0)
        # The goal backtest provides the same eligible historical legacy fits.
        cached = pd.read_csv(SOURCE / f"comparison_predictions_{year}.csv")
        legacy_columns = ["country", "fixture_id", "kickoff", "league_id", "home_goals", "away_goals",
                          "expected_home_legacy", "expected_away_legacy", "p_over25_legacy"]
        common = export.merge(cached[legacy_columns], on=["country", "fixture_id"],
                              validate="one_to_one", suffixes=("", "_legacy"))
        if len(common) != len(cached):
            raise ValueError("All legacy-supported fixtures must have V3/V2 historical forecasts.")
        for name in ("league_id", "home_goals", "away_goals"):
            np.testing.assert_array_equal(common[name], common[f"{name}_legacy"])
        np.testing.assert_array_equal(pd.to_datetime(common.kickoff, utc=True), pd.to_datetime(common.kickoff_legacy, utc=True))
        country_info = {info["country"]: info for path in SOURCE.glob("legacy_*_metadata.json")
                        for info in [json.loads(path.read_text(encoding="utf-8"))]}
        rho = common.country.map({country: info["periods"][str(year)].get("rho", np.nan)
                                  for country, info in country_info.items()}).to_numpy(dtype=float)
        if not np.isfinite(rho).all():
            raise ValueError("Missing historical legacy rho.")
        p_legacy = validate_probabilities(np.array([legacy.outcome_probabilities(float(home), float(away), rho=float(value))
                           for home, away, value in zip(common.expected_home_legacy, common.expected_away_legacy, rho)]))
        common["rho_legacy"] = rho
        common[probability_columns("legacy")] = p_legacy
        for model in MODELS:
            common[f"pick_{model}"] = p = common[probability_columns(model)].to_numpy().argmax(axis=1)
            common[f"correct_{model}"] = (p == common.target).astype(int)
        export.to_csv(OUTPUT / f"all_predictions_{year}.csv", index=False)
        common.to_csv(OUTPUT / f"common_predictions_{year}.csv", index=False)
        forecast_info["periods"][str(year)] = {"training_start": bundle["training_cutoff"],
            "test_end": end.isoformat(), "v3_selection": bundle["selection"],
            "v3_calibration_parameters": bundle["calibration_parameters"],
            "v2_selection": bundle["reference_selection"], "v2_training_matches": len(training),
            "legacy": country_info, "all_fixtures": len(export), "legacy_common_fixtures": len(common),
            "first_division_forecasts_reproduced": True}
        print(f"Saved {year}: {len(export)} V2/V3 matches, {len(common)} shared with legacy", flush=True)
    write_json(OUTPUT / "forecast_metadata.json", forecast_info)


def selected_metrics(y, p, mask):
    prediction, confidence = p.argmax(axis=1), p.max(axis=1)
    n = int(mask.sum())
    correct = int((prediction[mask] == y[mask]).sum())
    rate = correct / n if n else None
    interval = None
    if n:
        z = 1.959963984540054
        denominator = 1 + z ** 2 / n
        center = (rate + z ** 2 / (2 * n)) / denominator
        radius = z * np.sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)) / denominator
        interval = [float(center - radius), float(center + radius)]
    return {"selected": n, "correct": correct, "incorrect": n - correct,
            "accuracy": rate, "coverage": n / len(y),
            "mean_confidence": float(confidence[mask].mean()) if n else None,
            "95pct_wilson_interval": interval}


def outcome_metrics(frame, model):
    y = frame.target.to_numpy(dtype=int)
    p = validate_probabilities(frame[probability_columns(model)].to_numpy())
    prediction, confidence = p.argmax(axis=1), p.max(axis=1)
    result = {"matches": len(y), "correct": int((prediction == y).sum()),
        "accuracy": float((prediction == y).mean()),
        "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
        "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "mean_confidence": float(confidence.mean()), "confidence_filters": {},
        "winner_picks_only": selected_metrics(y, p, prediction != 1), "outcomes": {},
        "confusion_matrix_actual_rows_predicted_columns": [[int(((y == actual) & (prediction == picked)).sum())
                                                               for picked in range(3)] for actual in range(3)]}
    for index, outcome in enumerate(OUTCOMES):
        result["outcomes"][outcome] = {"actual": int((y == index).sum()), "predicted": int((prediction == index).sum()),
            "correct": int(((y == index) & (prediction == index)).sum()), "mean_probability": float(p[:, index].mean())}
    for threshold in THRESHOLDS:
        result["confidence_filters"][str(threshold)] = selected_metrics(y, p, confidence >= threshold)
    return result


def paired_metrics(frame, baseline):
    y = frame.target.to_numpy(dtype=int)
    old, new = [validate_probabilities(frame[probability_columns(model)]) for model in (baseline, "v3")]
    one_hot = np.eye(3)[y]
    delta = {"accuracy": (new.argmax(axis=1) == y).astype(float) - (old.argmax(axis=1) == y),
             "brier": np.sum((new - one_hot) ** 2, axis=1) - np.sum((old - one_hot) ** 2, axis=1),
             "log_loss": -np.log(np.clip(new[np.arange(len(y)), y], 1e-12, 1))
                         + np.log(np.clip(old[np.arange(len(y)), y], 1e-12, 1))}
    week = pd.to_datetime(frame.kickoff, utc=True).dt.strftime("%G-%V")
    rng = np.random.default_rng(42)
    result = {}
    for metric, values in delta.items():
        blocks = pd.DataFrame({"week": week, "delta": values}).groupby("week").agg(total=("delta", "sum"), n=("delta", "size"))
        indices = rng.integers(0, len(blocks), (5000, len(blocks)))
        bootstrap = blocks.total.to_numpy()[indices].sum(axis=1) / blocks.n.to_numpy()[indices].sum(axis=1)
        result[metric] = {"v3_minus_baseline": float(values.mean()),
                          "95pct_week_bootstrap": np.quantile(bootstrap, [.025, .975]).tolist()}
    return result


def summarize(frame, model_names, paired=True):
    result = {"matches": len(frame), "models": {model: outcome_metrics(frame, model) for model in model_names}}
    if paired:
        result["paired_differences"] = {model: paired_metrics(frame, model) for model in model_names if model != "v3"}
    return result


def generate_report():
    from render_readme import readme
    info = json.loads((OUTPUT / "forecast_metadata.json").read_text(encoding="utf-8"))
    report = {"data_cutoff": info["data_cutoff"], "sources": info["sources"], "forecast_metadata": info,
              "outcomes": list(OUTCOMES), "thresholds": list(THRESHOLDS),
              "periods": {}, "notes": "Argmax 1X2; actual draws remain in all samples. Inclusive cumulative probability cutoffs. V3 is the full selected blend."}
    all_frames, common_frames, summary_rows = [], [], []
    for year in YEARS:
        all_frames.append(pd.read_csv(OUTPUT / f"all_predictions_{year}.csv"))
        common_frames.append(pd.read_csv(OUTPUT / f"common_predictions_{year}.csv"))
    datasets = [(str(year), all_data, common) for year, all_data, common in zip(YEARS, all_frames, common_frames)]
    datasets.append(("combined", pd.concat(all_frames, ignore_index=True), pd.concat(common_frames, ignore_index=True)))
    for period, all_data, common in datasets:
        output = {"full_v2_v3": {}, "legacy_common": {}}
        for population, frame, models in (("full_v2_v3", all_data, ("v2", "v3")), ("legacy_common", common, MODELS)):
            if frame.duplicated(["country", "fixture_id"]).any():
                raise ValueError("Repeated country/fixture key.")
            expected_y = np.where(frame.home_goals > frame.away_goals, 0, np.where(frame.home_goals == frame.away_goals, 1, 2))
            np.testing.assert_array_equal(frame.target, expected_y)
            scopes = [("all", frame), *[(division, group) for division, group in frame.groupby("division")]]
            for scope, group in scopes:
                output[population][scope] = summarize(group, models)
            output[population]["leagues"] = {f"{country}:{int(league)}": {
                "country": country, "league_id": int(league), "division": group.division.iloc[0],
                **summarize(group, models, False)} for (country, league), group in frame.groupby(["country", "league_id"])}
            for scope, values in output[population].items():
                records = values.items() if scope == "leagues" else [(scope, values)]
                for key, item in records:
                    for model, metrics in item["models"].items():
                        for threshold in THRESHOLDS:
                            row = dict(metrics["confidence_filters"][str(threshold)])
                            interval = row.pop("95pct_wilson_interval")
                            row.update({"period": period, "population": population, "scope": key,
                                        "model": model, "minimum_probability": threshold,
                                        "wilson_lower": interval[0] if interval else None,
                                        "wilson_upper": interval[1] if interval else None})
                            summary_rows.append(row)
        report["periods"][period] = output
    write_json(OUTPUT / "evaluation.json", report)
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "threshold_summary.csv", index=False)
    (OUTPUT / "README.md").write_text(readme(report), encoding="utf-8")
    for period, data in report["periods"].items():
        for population in ("legacy_common", "full_v2_v3"):
            for division in ("all", "first", "second"):
                item = data[population][division]
                print(period, population, division, item["matches"],
                      {model: f"{metrics['correct']}/{metrics['matches']} ({metrics['accuracy']:.2%})"
                       for model, metrics in item["models"].items()}, flush=True)
    print(f"Updated {OUTPUT / 'README.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "predictions", "report"), default="all")
    args = parser.parse_args()
    protected = [ROOT / name for name in ("football_poisson.py", "football_model_v2.py", "football_model_v3.py")]
    protected += list(ROOT.glob("*_poisson.joblib"))
    for folder in (ROOT / "new_model", ROOT / "new_model_v3", SOURCE):
        protected += [path for path in folder.glob("*") if path.is_file()]
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    with threadpool_limits(limits=2):
        if args.mode in ("all", "predictions"):
            generate_predictions()
        if args.mode in ("all", "report"):
            generate_report()
    for relative, digest in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"An existing input or model changed: {relative}")
    write_json(OUTPUT / "protected_input_sha256.json", hashes)
    print("Original models and input reports unchanged.", flush=True)


if __name__ == "__main__":
    main()
