"""Fit the possession-free V4 and compare chronological V3/V4 forecasts.

The saved V3 architecture is fixed. Both annual models use identical historical
fixtures; no model selection uses the evaluation outcomes. Existing models and
SQLite databases are protected and never overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import football_model_v2 as base
import football_model_v3 as v3
import football_model_v4 as v4

OUTCOMES = ("home", "draw", "away")
EXCLUDED_STATISTICS = ("possession", "passes", "passes_completed", "pass_accuracy")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def check_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"An input changed during training: {name}")


def protected_inputs(baseline_path, sources):
    paths = [ROOT / name for name in (
        "football_poisson.py", "football_model_v2.py", "football_model_v3.py",
        "football_corners.py", "football_corners_v2.py", "football_model_v4.py",
    )]
    paths += [Path(__file__).resolve(), baseline_path]
    paths += list(ROOT.glob("*_poisson.joblib"))
    for name in ("new_model", "new_model_v3", "corners_model", "corners_model_v2",
                 "corners_model_with_passes", "corners_model_without_passes"):
        paths += [p for p in (ROOT / name).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts]
    # A SQLite WAL can contain newer committed rows than the main database.
    for source in sources.values():
        database = Path(source["path"]).resolve()
        paths.append(database)
        wal = Path(str(database) + "-wal")
        if wal.is_file():
            paths.append(wal)
    return {relative(p): digest(p) for p in dict.fromkeys(paths) if p.is_file()}


def validate_probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite home/draw/away probabilities.")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("Outcome probability outside [0, 1].")
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-10, rtol=0)
    return p


def fit_rho_without_excluded_inputs(frame, columns, base_columns, selection,
                                    reference_selection, validation_year):
    """Estimate the goal correction from the preceding held-out year only."""
    result = deepcopy(selection)
    fit_cutoff = pd.Timestamp(f"{validation_year - 1}-01-01", tz="UTC")
    end = pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
    training = frame[frame.kickoff + base.RESULT_DELAY <= fit_cutoff]
    held_out = frame[(frame.kickoff >= fit_cutoff)
                     & (frame.kickoff + base.RESULT_DELAY <= end)
                     & (frame.league_id == frame.primary_league)]
    details = {"fit_cutoff": fit_cutoff.isoformat(),
               "calibration_year": validation_year - 1,
               "training_matches": len(training), "held_out_matches": len(held_out),
               "components": []}
    for index, spec in enumerate(result["components"]):
        if spec["family"] != "goals":
            continue
        # A baseline that explicitly disabled Dixon-Coles keeps it disabled.
        if spec.get("rho", 0.) == 0.:
            details["components"].append({"index": index, "rho": 0., "refitted": False})
            continue
        if len(training) < 100 or len(held_out) < 100:
            raise ValueError("Need at least 100 early-training and held-out rho fixtures.")
        model = v3.fit_component(training, columns, base_columns, spec,
                                 reference_selection, fit_cutoff)
        home, away = v3.goal_rates(model, held_out, base_columns, spec)
        spec["rho"] = v3.fit_rho(held_out, home, away)
        details["components"].append({"index": index, "rho": spec["rho"], "refitted": True})
    return result, details


def initial_calibration(frame, columns, base_columns, selection,
                        reference_selection, validation_year, fit_function):
    method = selection.get("calibration_method", "none")
    if method == "none":
        return [0., 0., 0.]
    start = pd.Timestamp(f"{validation_year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{validation_year + 1}-01-01", tz="UTC")
    training = frame[frame.kickoff + base.RESULT_DELAY <= start]
    held_out = frame[(frame.kickoff >= start)
                     & (frame.kickoff + base.RESULT_DELAY <= end)
                     & (frame.league_id == frame.primary_league)]
    if len(training) < 100 or len(held_out) < 100:
        raise ValueError("Need chronological training and held-out calibration fixtures.")
    bundle = fit_function(training, columns, base_columns, selection,
                          reference_selection, start)
    return v3.fit_calibration(v3.predict_bundle(bundle, held_out, raw=True),
                              held_out.target.to_numpy(), method)


def prediction_columns(bundle, frame, model_name):
    p = validate_probabilities(v3.predict_bundle(bundle, frame))
    goals = v3.expected_goals(bundle, frame)
    over = v3.total_goal_probabilities(bundle, frame, (3, 4))
    if goals is None or over is None:
        raise ValueError("The fixed architecture must retain a goal-rate component.")
    if not np.isfinite(over).all() or np.any(over < 0) or np.any(over > 1):
        raise ValueError("Invalid total-goal probabilities.")
    if np.any(over[:, 0] < over[:, 1] - 1e-12):
        raise ValueError("Over 3.5 cannot be more likely than over 2.5.")
    result = {f"p_{name}_{model_name}": p[:, i] for i, name in enumerate(OUTCOMES)}
    result.update({f"expected_home_{model_name}": goals[0],
                   f"expected_away_{model_name}": goals[1],
                   f"p_over25_{model_name}": over[:, 0],
                   f"p_under25_{model_name}": 1. - over[:, 0],
                   f"p_over35_{model_name}": over[:, 1],
                   f"p_under35_{model_name}": 1. - over[:, 1]})
    return result


def saved_baseline_reproduction(baseline_path, year, bundle, test, export):
    """Confirm a same-history refit represents the saved annual V3 model."""
    path = baseline_path.parent / f"evaluation_model_{year}.joblib"
    details = {"saved_bundle": relative(path), "available": path.is_file()}
    if not path.is_file():
        details.update({"reproduced": False, "reason": "No saved annual V3 bundle."})
        return details
    saved = joblib.load(path)
    if saved.get("kind") != v3.KIND:
        raise ValueError("Unexpected saved baseline evaluation model kind.")
    if pd.Timestamp(saved["training_cutoff"]) > pd.Timestamp(f"{year}-01-01", tz="UTC"):
        raise ValueError("Saved V3 evaluation bundle overlaps its test period.")
    expected = validate_probabilities(v3.predict_bundle(saved, test))
    actual = export[[f"p_{name}_v3" for name in OUTCOMES]].to_numpy()
    expected_goals = np.column_stack(v3.expected_goals(saved, test))
    actual_goals = export[["expected_home_v3", "expected_away_v3"]].to_numpy()
    probability_difference = float(np.max(np.abs(expected - actual))) if len(test) else 0.
    goal_difference = float(np.max(np.abs(expected_goals - actual_goals))) if len(test) else 0.
    reproduced = probability_difference <= 1e-10 and goal_difference <= 1e-10
    details.update({"reproduced": reproduced, "fixtures_checked": len(test),
                    "max_probability_difference": probability_difference,
                    "max_expected_goal_difference": goal_difference,
                    "saved_calibration_parameters": saved["calibration_parameters"]})
    if not reproduced:
        details["reason"] = "Saved V3 differs from its refit on current identical-history partitions."
    csv = baseline_path.parent / f"test_predictions_{year}.csv"
    if csv.is_file():
        original = pd.read_csv(csv)
        matched = export.merge(original, on=["country", "fixture_id"],
                               validate="one_to_one", suffixes=("", "_saved"))
        columns = [f"p_{name}_v3" for name in OUTCOMES]
        difference = float(np.max(np.abs(matched[columns].to_numpy()
                             - matched[[c + "_saved" for c in columns]].to_numpy()))) if len(matched) else None
        details.update({"saved_primary_fixtures_checked": len(matched),
                        "saved_primary_max_probability_difference": difference,
                        "saved_primary_reproduced": difference is not None and difference <= 1e-10})
        for name in ("home_goals", "away_goals", "league_id", "target"):
            np.testing.assert_array_equal(matched[name], matched[name + "_saved"])
    return details


def validate_destinations(output, report_output, baseline_path):
    forbidden = [ROOT / name for name in ("new_model", "new_model_v3", "corners_model",
                 "corners_model_v2", "corners_model_with_passes", "corners_model_without_passes")]
    forbidden.append(baseline_path.parent)
    for path in (output, report_output):
        if path == ROOT or any(path == old.resolve() or old.resolve() in path.parents for old in forbidden):
            raise ValueError("Choose separate V4 model and comparison output directories.")
    if output == report_output or output in report_output.parents or report_output in output.parents:
        raise ValueError("V4 model and report directories must be separate.")


def regenerate_reports(output):
    """Validate and summarize complete saved forecasts without fitting models."""
    output = Path(output).resolve()
    metadata_path = output / "forecast_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    check_hashes(metadata["input_sha256"])
    frames = []
    for year in metadata["years"]:
        period = metadata["periods"][str(year)]
        for path_key, sha_key in (("predictions_path", "predictions_sha256"),
                                  ("baseline_model_path", "baseline_model_sha256"),
                                  ("v4_model_path", "v4_model_sha256")):
            check_hashes({period[path_key]: period[sha_key]})
        path = Path(period["predictions_path"])
        if not path.is_absolute():
            path = ROOT / path
        frame = pd.read_csv(path)
        if len(frame) != period["test_matches"]:
            raise ValueError(f"Saved {year} forecast count differs from metadata.")
        frames.append(frame)
    final = metadata["final_v4_model"]
    check_hashes({final["path"]: final["sha256"]})
    if metadata.get("implementation_sha256"):
        check_hashes(metadata["implementation_sha256"])
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(["country", "fixture_id"]).any():
        raise ValueError("Duplicated fixture in annual prediction files.")
    from v3_v4_accuracy_comparison.report import write_reports
    result = write_reports(predictions, metadata, output)
    check_hashes(metadata["input_sha256"])
    return result


def train_and_compare(args):
    baseline_path = Path(args.baseline_model).resolve()
    baseline = joblib.load(baseline_path)
    if baseline.get("kind") != v3.KIND:
        raise ValueError("The baseline must be a saved V3 outcome model.")
    output = Path(args.output).resolve()
    report_output = Path(args.report_output).resolve()
    validate_destinations(output, report_output, baseline_path)
    cutoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is None:
        raise ValueError("--as-of requires an explicit timezone.")
    years = (int(args.test_year), int(args.confirm_year))
    if years[0] >= years[1]:
        raise ValueError("Require test-year < confirm-year.")
    frozen_path = baseline_path.parent / "frozen_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.is_file() else {}
    validation_year = int(frozen.get("validation_year", years[0] - 1))
    if validation_year >= years[0]:
        raise ValueError("Evaluation years must follow the baseline architecture-selection year.")
    selection = deepcopy(baseline["selection"])
    if not any(s["family"] == "goals" and w > 0 for s, w in
               zip(selection["components"], selection["weights"])):
        raise ValueError("V3 baseline needs a selected goal component for this comparison.")
    reference_selection = deepcopy(baseline["reference_selection"])
    paths = args.db or [Path(s["path"]) for s in baseline["sources"].values()]
    # Preserve inputs before loading SQLite. Reading is always mode=ro.
    input_hashes = protected_inputs(baseline_path, baseline["sources"])
    rows, sources = base.read_databases(paths)
    rows = rows[rows.kickoff + base.RESULT_DELAY <= cutoff].copy()
    if rows.empty:
        raise ValueError("No completed fixtures available by the training cutoff.")
    for country, source in sources.items():
        if country not in baseline["sources"] or source["league_ids"] != baseline["sources"][country]["league_ids"]:
            raise ValueError("Comparison databases must use the V3 country/division configuration.")
        database = Path(source["path"])
        for path in (database, Path(str(database) + "-wal")):
            if path.is_file():
                input_hashes.setdefault(relative(path), digest(path))
    print(f"Building chronological V3/V4 inputs from {len(rows):,} fixtures...", flush=True)
    frame3, _ = v3.build_features(rows)
    feature_columns3 = [c for c in frame3 if c not in base.NON_FEATURES]
    base_columns3 = [c for c in feature_columns3 if not c.startswith("v3_")]
    forbidden_features = [c for c in feature_columns3 if not v4.allowed_feature(c)]
    frame4 = frame3.drop(columns=forbidden_features).copy()
    feature_columns4 = [c for c in feature_columns3 if v4.allowed_feature(c)]
    base_columns4 = [c for c in base_columns3 if v4.allowed_feature(c)]
    if any(not v4.allowed_feature(c) for c in feature_columns4 + base_columns4):
        raise ValueError("An excluded statistic survived the V4 feature filter.")
    print(f"V4 excludes {len(forbidden_features)} inputs: active base features "
          f"{len(base_columns3)} -> {len(base_columns4)}; candidates "
          f"{len(feature_columns3)} -> {len(feature_columns4)}.", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)
    baseline_output = report_output / "baseline_v3"
    baseline_output.mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        print(f"Refitting V4 low-score correction on held-out {validation_year - 1} fixtures...", flush=True)
        selection4, rho_details = fit_rho_without_excluded_inputs(
            frame4, feature_columns4, base_columns4, selection,
            reference_selection, validation_year)
        parameters3 = initial_calibration(frame3, feature_columns3, base_columns3,
                                           selection, reference_selection, validation_year, v3.fit_bundle)
        parameters4 = initial_calibration(frame4, feature_columns4, base_columns4,
                                           selection4, reference_selection, validation_year, v4.fit_bundle)
        metadata = {
            "data_cutoff": cutoff.isoformat(), "training_cutoff": cutoff.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
            "years": list(years), "models": ["v3", "v4"], "thresholds": [.6, .7, .8],
            "baseline_model": relative(baseline_path),
            "baseline_saved_training_cutoff": baseline["training_cutoff"],
            "baseline_saved_training_matches": baseline.get("training_matches"),
            "baseline_selection": selection, "v4_selection": selection4,
            "reference_selection": reference_selection,
            "validation_year": validation_year,
            "excluded_statistics": list(EXCLUDED_STATISTICS),
            "excluded_feature_columns": forbidden_features,
            "baseline_feature_count": len(feature_columns3),
            "baseline_selected_input_feature_count": len(base_columns3),
            "v4_feature_count": len(feature_columns4),
            "v4_selected_input_feature_count": len(base_columns4),
            "training_matches": len(frame4), "rho_refit": rho_details,
            "input_sha256": input_hashes, "periods": {},
            "notes": "Fixed V3 architecture and weights; V4 refits coefficients and the "
                     "previously enabled Dixon-Coles correction after excluding possession "
                     "and passes. Both annual fits use identical prior fixtures. Architecture "
                     "is not selected again. Each model uses its own threshold; draws are "
                     "failed winner predictions. The 2025/2026 historical results have been "
                     "previously inspected and are not a new untouched future holdout.",
        }
        for year in years:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), cutoff)
            mask_training = frame3.kickoff + base.RESULT_DELAY <= start
            mask_test = (frame3.kickoff >= start) & (frame3.kickoff < end)
            training3, training4 = frame3[mask_training], frame4[mask_training]
            test3, test4 = frame3[mask_test], frame4[mask_test]
            if len(training3) < 100 or len(test3) < 100:
                raise ValueError(f"Need at least 100 training and test fixtures for {year}.")
            if (training3.kickoff + base.RESULT_DELAY > start).any():
                raise ValueError("Training contains unavailable test-year results.")
            np.testing.assert_array_equal(training3[["country", "fixture_id"]], training4[["country", "fixture_id"]])
            print(f"{year}: fitting V3 and V4 on {len(training3):,} past fixtures; "
                  f"testing {len(test3):,} fixtures across all divisions...", flush=True)
            bundle3 = v3.fit_bundle(training3, feature_columns3, base_columns3, selection,
                                    reference_selection, start, parameters3)
            bundle4 = v4.fit_bundle(training4, feature_columns4, base_columns4, selection4,
                                    reference_selection, start, parameters4)
            names = [c for c in ("country", "fixture_id", "kickoff", "season", "league_id", "primary_league",
                     "home_name", "away_name", "home_goals", "away_goals", "target") if c in test3]
            export = test3[names].copy()
            export["division"] = np.where(export.league_id == export.primary_league, "first", "second")
            for name, bundle, test in (("v3", bundle3, test3), ("v4", bundle4, test4)):
                for column, values in prediction_columns(bundle, test, name).items():
                    export[column] = values
            reproduction = saved_baseline_reproduction(baseline_path, year, bundle3, test3, export)
            for bundle in (bundle3, bundle4):
                bundle.update({"sources": sources, "training_cutoff": start.isoformat(),
                               "training_matches": len(training3)})
            bundle4.update({"excluded_statistics": list(EXCLUDED_STATISTICS),
                            "excluded_feature_columns": forbidden_features,
                            "baseline_architecture": selection})
            model3_path = baseline_output / f"evaluation_model_{year}.joblib"
            model4_path = output / f"evaluation_model_{year}.joblib"
            joblib.dump(bundle3, model3_path)
            joblib.dump(bundle4, model4_path)
            csv_path = report_output / f"predictions_{year}.csv"
            export.to_csv(csv_path, index=False, float_format="%.17g")
            period = {"start": start.isoformat(), "end": end.isoformat(),
                      "training_cutoff": start.isoformat(), "training_matches": len(training3),
                      "test_matches": len(export), "all_fixtures": len(export),
                      "first_division_matches": int((export.division == "first").sum()),
                      "second_division_matches": int((export.division == "second").sum()),
                      "latest_test_kickoff": test3.kickoff.max().isoformat(),
                      "calibration_parameters_v3": parameters3,
                      "calibration_parameters_v4": parameters4,
                      "baseline_saved_reproduction": reproduction,
                      "predictions_path": relative(csv_path),
                      "predictions_sha256": digest(csv_path),
                      "baseline_model_path": relative(model3_path),
                      "baseline_model_sha256": digest(model3_path),
                      "v4_model_path": relative(model4_path),
                      "v4_model_sha256": digest(model4_path)}
            metadata["periods"][str(year)] = period
            # Calibrate only from genuinely out-of-training primary predictions,
            # following the same convention as the existing V3 trainer.
            primary = test3.league_id == test3.primary_league
            for name, bundle, test in (("v3", bundle3, test3), ("v4", bundle4, test4)):
                method = bundle["selection"].get("calibration_method", "none")
                parameters = v3.fit_calibration(v3.predict_bundle(bundle, test[primary], raw=True),
                                                test.loc[primary, "target"].to_numpy(), method)
                if name == "v3":
                    parameters3 = parameters
                else:
                    parameters4 = parameters
            print(f"Saved {year} predictions; saved V3 annual reproduction="
                  f"{reproduction.get('reproduced', False)}.", flush=True)
        print(f"Fitting final V4 on all {len(frame4):,} historical fixtures...", flush=True)
        final = v4.fit_bundle(frame4, feature_columns4, base_columns4, selection4,
                              reference_selection, cutoff, parameters4)
        final.update({"sources": sources, "training_cutoff": cutoff.isoformat(),
                      "training_matches": len(frame4),
                      "excluded_statistics": list(EXCLUDED_STATISTICS),
                      "excluded_feature_columns": forbidden_features,
                      "baseline_architecture": selection,
                      "baseline_model": relative(baseline_path)})
        final_path = output / "football_model.joblib"
        joblib.dump(final, final_path)
        frozen4 = {"selection": selection4, "baseline_selection": selection,
                   "architecture_fixed_before_evaluation": True,
                   "validation_year": validation_year, "rho_refit": rho_details,
                   "excluded_statistics": list(EXCLUDED_STATISTICS),
                   "excluded_feature_columns": forbidden_features,
                   "reference_selection": reference_selection}
        write_json(output / "frozen_selection.json", frozen4)
        metadata["final_v4_model"] = {"path": relative(final_path), "sha256": digest(final_path),
                                      "calibration_parameters": parameters4}
        metadata["final_v3_calibration_parameters_context"] = parameters3
        report_script = ROOT / "v3_v4_accuracy_comparison" / "report.py"
        metadata["implementation_sha256"] = {
            relative(path): digest(path) for path in (
                ROOT / "football_model_v4.py", Path(__file__).resolve(), report_script,
            ) if path.is_file()
        }
        check_hashes(input_hashes)
        write_json(report_output / "forecast_metadata.json", metadata)
        write_json(output / "evaluation.json", metadata)
    regenerate_reports(report_output)
    check_hashes(input_hashes)
    print(f"Saved separate V4 model: {final_path}", flush=True)
    print("All existing model files and SQLite databases remain unchanged.", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, nargs="+")
    parser.add_argument("--as-of")
    parser.add_argument("--baseline-model", type=Path, default=Path("new_model_v3/football_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("new_model_v4"))
    parser.add_argument("--report-output", type=Path, default=Path("v3_v4_accuracy_comparison"))
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--confirm-year", type=int, default=2026)
    args = parser.parse_args()
    train_and_compare(args)


if __name__ == "__main__":
    main()
