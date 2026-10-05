"""Write paired V3/V4 historical accuracy reports from already generated forecasts.

The public entry point is ``write_reports(predictions, metadata, output)``.
No model fitting, SQLite writes, network access or saved-model mutation occurs here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODELS = ("v3", "v4")
DEFAULT_YEARS = (2025, 2026)
THRESHOLDS = (.6, .7, .8)
EVENTS = ("winner", "over25", "under25", "over35", "under35")
LABELS = {"winner": "Team wins", "over25": "Over 2.5 goals",
          "under25": "Under 2.5 goals", "over35": "Over 3.5 goals",
          "under35": "Under 3.5 goals"}
COUNTRIES = {"argentina": "Argentina", "chile": "Chile", "colombia": "Colombia",
             "england": "England", "germany": "Germany", "iceland": "Iceland",
             "ireland": "Ireland", "italy": "Italy", "south-korea": "South Korea",
             "spain": "Spain", "usa": "United States"}


def _native(value: Any) -> Any:
    """Keep exported metadata readable and prohibit NaN in JSON."""
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Report metadata contains a nonfinite number.")
    return value


def _years(metadata: dict, frame: pd.DataFrame | None = None) -> tuple[int, ...]:
    recorded = metadata.get("periods", {})
    years = sorted(int(key) for key in recorded if str(key).isdigit())
    if not years and metadata.get("years"):
        years = sorted(set(int(year) for year in metadata["years"]))
    if not years and frame is not None and len(frame):
        years = sorted(int(year) for year in frame.kickoff.dt.year.unique())
    return tuple(years or DEFAULT_YEARS)


def _validate(predictions: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    frame = predictions.copy()
    required = {"country", "fixture_id", "kickoff", "league_id", "primary_league",
                "home_goals", "away_goals", "target"}
    required |= {f"p_{outcome}_{model}" for model in MODELS for outcome in ("home", "draw", "away")}
    required |= {f"p_{event}_{model}" for model in MODELS for event in EVENTS if event != "winner"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}")
    if frame.empty:
        raise ValueError("The comparison needs at least one held-out fixture.")
    if frame.duplicated(["country", "fixture_id"]).any():
        raise ValueError("Duplicate country/fixture keys in the comparison.")
    frame["kickoff"] = pd.to_datetime(frame.kickoff, utc=True, errors="raise")
    years = _years(metadata, frame)
    if frame.kickoff.isna().any() or not frame.kickoff.dt.year.isin(years).all():
        raise ValueError(f"Comparison fixtures must fall in the recorded evaluation years: {years}.")
    goals = frame[["home_goals", "away_goals"]].to_numpy(dtype=float)
    if not np.isfinite(goals).all() or (goals < 0).any() or (goals != np.floor(goals)).any():
        raise ValueError("Actual goals must be finite nonnegative integers.")
    target = np.where(goals[:, 0] > goals[:, 1], 0, np.where(goals[:, 0] == goals[:, 1], 1, 2))
    if not np.array_equal(frame.target.to_numpy(), target):
        raise ValueError("Outcome targets disagree with actual goals.")
    frame["target"] = target
    division = np.where(frame.league_id == frame.primary_league, "first", "second")
    if "division" in frame and not np.array_equal(frame.division.to_numpy(), division):
        raise ValueError("Division labels disagree with primary league IDs.")
    frame["division"] = division
    if metadata.get("data_cutoff"):
        cutoff = pd.Timestamp(metadata["data_cutoff"])
        if cutoff.tzinfo is None:
            raise ValueError("The data cutoff needs a timezone.")
        if (frame.kickoff + pd.Timedelta(hours=3) > cutoff).any():
            raise ValueError("A fixture result was unavailable at the recorded cutoff.")
    for model in MODELS:
        p = frame[[f"p_{outcome}_{model}" for outcome in ("home", "draw", "away")]].to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
            raise ValueError(f"Invalid {model} 1X2 probabilities.")
        for event in EVENTS[1:]:
            q = frame[f"p_{event}_{model}"].to_numpy(dtype=float)
            if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
                raise ValueError(f"Invalid {model} {event} probabilities.")
        for suffix in ("25", "35"):
            if not np.allclose(frame[f"p_over{suffix}_{model}"] + frame[f"p_under{suffix}_{model}"], 1, atol=1e-10, rtol=0):
                raise ValueError(f"{model} over/under probabilities are not complements.")
        if (frame[f"p_over35_{model}"] > frame[f"p_over25_{model}"] + 1e-12).any():
            raise ValueError(f"{model} goal probabilities are not nested.")
    return frame.sort_values(["kickoff", "country", "fixture_id"]).reset_index(drop=True)


def _wilson(correct: int, selected: int) -> list[float] | None:
    if not selected:
        return None
    z = 1.959963984540054
    rate = correct / selected
    denominator = 1 + z * z / selected
    center = (rate + z * z / (2 * selected)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / selected + z * z / (4 * selected * selected)) / denominator
    return [float(center - radius), float(center + radius)]


def _event_arrays(frame: pd.DataFrame, event: str, model: str):
    if event == "winner":
        home = frame[f"p_home_{model}"].to_numpy(dtype=float)
        away = frame[f"p_away_{model}"].to_numpy(dtype=float)
        picked = np.where(home >= away, 0, 2)
        return np.maximum(home, away), picked == frame.target.to_numpy(dtype=int), picked
    total = frame.home_goals.to_numpy(dtype=int) + frame.away_goals.to_numpy(dtype=int)
    minimum = 3 if event.endswith("25") else 4
    actual = total >= minimum if event.startswith("over") else total < minimum
    return frame[f"p_{event}_{model}"].to_numpy(dtype=float), actual, None


def _filtered(frame, event, model, threshold, additional_mask=None):
    probability, correct, picked = _event_arrays(frame, event, model)
    selected = probability >= threshold
    if additional_mask is not None:
        selected &= additional_mask
    count, hits = int(selected.sum()), int(correct[selected].sum())
    result = {"selected": count, "correct": hits, "incorrect": count - hits,
              "accuracy": hits / count if count else None,
              "coverage": count / len(frame) if len(frame) else None,
              "mean_probability": float(probability[selected].mean()) if count else None,
              "wilson_95_interval": _wilson(hits, count)}
    if event == "winner":
        actual = frame.target.to_numpy(dtype=int)
        draws = int((actual[selected] == 1).sum())
        result.update({"draw_failures": draws, "other_team_wins": count - hits - draws,
                       "home_picks": int((picked[selected] == 0).sum()),
                       "away_picks": int((picked[selected] == 2).sum())})
    return result


def _calibration(probability, actual):
    probability = np.asarray(probability, dtype=float)
    actual = np.asarray(actual, dtype=float)
    bins = []
    error = 0.
    for i in range(10):
        lower, upper = i / 10, (i + 1) / 10
        mask = (probability >= lower) & ((probability < upper) if i < 9 else (probability <= upper))
        n = int(mask.sum())
        average = float(probability[mask].mean()) if n else None
        observed = float(actual[mask].mean()) if n else None
        if n:
            error += n / len(actual) * abs(average - observed)
        bins.append({"lower": lower, "upper": upper, "matches": n,
                     "mean_probability": average, "observed_rate": observed})
    return {"ece_10_bins": float(error) if len(actual) else None, "bins": bins}


def _binary_scores(probability, actual):
    q = np.asarray(probability, dtype=float)
    y = np.asarray(actual, dtype=float)
    if not len(y):
        return {"matches": 0, "actual_event_rate": None, "mean_probability": None,
                "calibration_gap": None, "brier": None, "log_loss": None,
                "calibration": _calibration(q, y)}
    bounded = np.clip(q, 1e-12, 1 - 1e-12)
    return {"matches": len(y), "actual_event_rate": float(y.mean()),
            "mean_probability": float(q.mean()), "calibration_gap": float(q.mean() - y.mean()),
            "brier": float(np.mean((q - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(bounded) + (1 - y) * np.log1p(-bounded))),
            "calibration": _calibration(q, y)}


def _unfiltered(frame, model):
    p = frame[[f"p_{outcome}_{model}" for outcome in ("home", "draw", "away")]].to_numpy(dtype=float)
    y = frame.target.to_numpy(dtype=int)
    predicted = p.argmax(axis=1)
    correct = predicted == y
    hda = {"matches": len(y), "correct": int(correct.sum()),
           "accuracy": float(correct.mean()) if len(y) else None,
           "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))) if len(y) else None,
           "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()) if len(y) else None,
           "mean_confidence": float(p.max(1).mean()) if len(y) else None,
           "calibration": _calibration(p.max(1), correct)}
    goals = {}
    for event in EVENTS[1:]:
        probability, actual, _ = _event_arrays(frame, event, model)
        goals[event] = _binary_scores(probability, actual)
    return {"home_draw_away": hda, "goals": goals}


def _paired_differences(frame, unfiltered, bootstrap=False):
    """Paired proper-score differences use all identical held-out fixtures."""
    actual = frame.target.to_numpy(dtype=int)
    if not len(actual):
        return {}
    p3, p4 = [frame[[f"p_{outcome}_{model}" for outcome in ("home", "draw", "away")]].to_numpy(dtype=float) for model in MODELS]
    deltas = {"home_draw_away_accuracy": (p4.argmax(1) == actual).astype(float) - (p3.argmax(1) == actual),
              "home_draw_away_brier": np.sum((p4 - np.eye(3)[actual]) ** 2, axis=1) - np.sum((p3 - np.eye(3)[actual]) ** 2, axis=1),
              "home_draw_away_log_loss": -np.log(np.clip(p4[np.arange(len(actual)), actual], 1e-12, 1)) + np.log(np.clip(p3[np.arange(len(actual)), actual], 1e-12, 1))}
    for event in EVENTS[1:]:
        old, target, _ = _event_arrays(frame, event, "v3")
        new, _, _ = _event_arrays(frame, event, "v4")
        target = target.astype(float)
        deltas[f"{event}_brier"] = (new - target) ** 2 - (old - target) ** 2
        old_q, new_q = np.clip(old, 1e-12, 1 - 1e-12), np.clip(new, 1e-12, 1 - 1e-12)
        deltas[f"{event}_log_loss"] = (-target * np.log(new_q) - (1 - target) * np.log1p(-new_q)
                                      + target * np.log(old_q) + (1 - target) * np.log1p(-old_q))
    weeks = frame.kickoff.dt.strftime("%G-%V").to_numpy()
    rng = np.random.default_rng(42)
    result = {}
    for name, values in deltas.items():
        entry = {"v4_minus_v3": float(values.mean())}
        if bootstrap:
            blocks = pd.DataFrame({"week": weeks, "delta": values}).groupby("week").agg(total=("delta", "sum"), count=("delta", "size"))
            indices = rng.integers(0, len(blocks), (2000, len(blocks)))
            sample = blocks.total.to_numpy()[indices].sum(1) / blocks["count"].to_numpy()[indices].sum(1)
            entry["week_bootstrap_95_interval"] = np.quantile(sample, [.025, .975]).tolist()
        result[name] = entry
    return result


def _summarize(frame, bootstrap=False):
    result = {"available_matches": len(frame), "unfiltered": {model: _unfiltered(frame, model) for model in MODELS}, "events": {}}
    result["paired_score_differences"] = _paired_differences(frame, result["unfiltered"], bootstrap)
    for event in EVENTS:
        thresholds = {}
        for threshold in THRESHOLDS:
            own = {model: _filtered(frame, event, model, threshold) for model in MODELS}
            masks = [_event_arrays(frame, event, model)[0] >= threshold for model in MODELS]
            both = masks[0] & masks[1]
            intersection = {model: _filtered(frame, event, model, threshold, both) for model in MODELS}
            common = {"selected": int(both.sum()), **intersection}
            if event == "winner":
                picks = [_event_arrays(frame, event, model)[2] for model in MODELS]
                common["same_team_picks"] = int((both & (picks[0] == picks[1])).sum())
                common["different_team_picks"] = int((both & (picks[0] != picks[1])).sum())
            common["v4_minus_v3_accuracy"] = (intersection["v4"]["accuracy"] - intersection["v3"]["accuracy"] if both.any() else None)
            thresholds[str(threshold)] = {**own, "v4_minus_v3_accuracy": (own["v4"]["accuracy"] - own["v3"]["accuracy"] if own["v3"]["selected"] and own["v4"]["selected"] else None), "qualified_by_both_models": common}
        result["events"][event] = {"thresholds": thresholds}
    return result


def _league_keys(frame, metadata):
    sources = metadata.get("sources", {})
    configured = [(country, int(league_id), "first" if i == 0 else "second")
                  for country, source in sorted(sources.items()) for i, league_id in enumerate(source["league_ids"])]
    observed = {(str(country), int(league_id)) for country, league_id in frame[["country", "league_id"]].drop_duplicates().itertuples(index=False, name=None)}
    if configured:
        if not observed <= {(country, league_id) for country, league_id, _ in configured}:
            raise ValueError("Forecasts include a league outside the recorded source configuration.")
        return configured
    return [(country, league_id, str(frame[(frame.country == country) & (frame.league_id == league_id)].division.iloc[0])) for country, league_id in sorted(observed)]


def _table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |", *["| " + " | ".join(map(str, row)) + " |" for row in rows]]


def _cell(metrics):
    return f"{metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['selected']})" if metrics["selected"] else "— (0 picks)"


def _percent(value):
    return f"{value:.2%}" if value is not None else "—"


def _league_table(period, event):
    rows = []
    for league in period["leagues"]:
        values = league["metrics"]
        row = [COUNTRIES.get(league["country"], league["country"]), league["division"].title(), league["league_id"], values["available_matches"]]
        for threshold in THRESHOLDS:
            selected = values["events"][event]["thresholds"][str(threshold)]
            row += [_cell(selected[model]) for model in MODELS]
        rows.append(row)
    return _table(["Country", "Division", "League ID", "Games", "V3 ≥60%", "V4 ≥60%", "V3 ≥70%", "V4 ≥70%", "V3 ≥80%", "V4 ≥80%"], rows)


def _render(report):
    metadata = report["metadata"]
    annual = [str(year) for year in report["years"]]
    display_periods = ["combined", *annual]
    combined_label = " + ".join(annual)
    cutoff = pd.Timestamp(metadata["data_cutoff"]).tz_convert("America/Bogota") if metadata.get("data_cutoff") else None
    lines = ["# V3 versus V4: winner and total-goal accuracy", "",
             f"The comparison contains **{report['available_matches']:,} identical held-out fixtures** in {combined_label}, across **{len(report['periods']['combined']['leagues'])} configured leagues**. First and second divisions are listed separately.", ""]
    if cutoff is not None:
        lines += [f"Data cutoff: **{cutoff:%d %B %Y, %H:%M:%S} Colombia time**. Evaluation periods later than this cutoff and cached league coverage are incomplete.", ""]
    methodology = metadata.get("methodology", metadata.get("notes"))
    if isinstance(methodology, str) and methodology:
        lines += [methodology, ""]
    elif isinstance(methodology, dict):
        lines += ["Recorded comparison method:", ""]
        method_rows = [[str(key).replace("_", " "), str(value) if isinstance(value, str) else json.dumps(_native(value), ensure_ascii=False)] for key, value in methodology.items()]
        lines += _table(["Aspect", "Method"], method_rows) + [""]
    selection = metadata.get("baseline_selection", {})
    if selection.get("components"):
        descriptions = {"boosting": "boosting classifier", "goals": "goal-rate component",
                        "v2": "V2 classifier", "logistic": "logistic classifier"}
        architecture = ", ".join(f"{float(weight):.0%} {descriptions.get(spec['family'], spec['family'])}"
                                 for spec, weight in zip(selection["components"], selection["weights"]))
        lines += [f"The V3 architecture and weights are held fixed for V4: **{architecture}**. The recorded calibration method is **{selection.get('calibration_method', 'none')}**. V4 refits the coefficients after removing the excluded inputs; architecture and threshold choices are not selected again using these evaluation years.", ""]
    if metadata.get("excluded_statistics"):
        lines += ["The current V3 feature set already omits passing statistics. V4 also blocks passing fields and removes all possession inputs, including their team/venue histories and differences.", ""]
    rho = metadata.get("rho_refit")
    if isinstance(rho, dict) and rho.get("calibration_year") is not None:
        components = rho.get("components", [])
        rho_values = ", ".join(f"{float(item['rho']):.6f}" for item in components if item.get("refitted"))
        correction = f" The refitted correction is {rho_values}." if rho_values else ""
        lines += [f"V4's previously enabled Dixon–Coles correction is refitted using held-out **{rho['calibration_year']}** predictions from goal models trained before **{rho.get('fit_cutoff', 'that year')}**, with **{rho.get('held_out_matches', 'the recorded')}** held-out matches.{correction} This uses no evaluation-year outcomes. The correction adjusts only 0–0, 0–1, 1–0 and 1–1 scorelines, so it does not directly change over/under 2.5 or 3.5 tails.", ""]
    provenance = []
    for year in annual:
        facts = metadata.get("periods", {}).get(year, {})
        proof = facts.get("baseline_saved_reproduction", {})
        if isinstance(proof, dict) and proof:
            status = "True: reproduced" if proof.get("reproduced") else "False: refit differs" if proof.get("available") else "False: saved bundle unavailable"
            probability_difference = proof.get("max_probability_difference")
            goal_difference = proof.get("max_expected_goal_difference")
            provenance.append([year, facts.get("training_cutoff", facts.get("start", "—")),
                               facts.get("training_matches", "—"), facts.get("test_matches", facts.get("all_fixtures", "—")),
                               status, proof.get("fixtures_checked", "—"),
                               f"{probability_difference:.3g}" if probability_difference is not None else "—",
                               f"{goal_difference:.3g}" if goal_difference is not None else "—",
                               proof.get("saved_primary_fixtures_checked", "—"),
                               proof.get("saved_primary_reproduced", "—"),
                               f"{proof['saved_primary_max_probability_difference']:.3g}"
                               if proof.get("saved_primary_max_probability_difference") is not None else "—"])
    if provenance:
        lines += ["The V3 baseline is refitted on the same current historical partitions as V4. The following check distinguishes that refit from the original saved V3 annual bundles; comparison coefficients always come from the years before each test.", ""]
        lines += _table(["Test year", "Fit cutoff", "Training matches", "Test matches", "Saved V3 model reproduced?", "All fixtures checked", "Maximum probability difference", "Maximum xG difference", "Primary CSV matches checked", "Original primary CSV reproduced?", "Original CSV maximum probability difference"], provenance) + [""]
        lines += ["Model reproduction evaluates the saved annual V3 coefficients on the current historical inputs. The older prediction CSVs used the cache as it existed when V3 was trained; later backfilled fixtures can change those inputs slightly. A CSV mismatch is recorded separately above. This comparison uses the same refreshed historical inputs for both variants.", ""]
    lines += ["Winner selection means **max(P(home win), P(away win)) ≥ the threshold**, choosing the more probable team. An actual draw is a failure. Over 2.5 means at least 3 total goals; under 2.5 means at most 2. Over 3.5 means at least 4; under 3.5 means at most 3.", "",
              "All thresholds are inclusive and cumulative. Each model applies the threshold to its own probability and may select different fixtures. Cells show **observed accuracy (correct / qualifying picks)**; a dash means no qualifying picks.", "",
              "## Overall threshold results", ""]
    for period_name in display_periods:
        overall = report["periods"][period_name]["overall"]
        lines += [f"### {combined_label if period_name == 'combined' else period_name} ({overall['available_matches']:,} games)", ""]
        rows = []
        for event in EVENTS:
            for threshold in THRESHOLDS:
                values = overall["events"][event]["thresholds"][str(threshold)]
                change = values["v4_minus_v3_accuracy"]
                rows.append([LABELS[event], f"≥{threshold:.0%}", _cell(values["v3"]), _cell(values["v4"]), f"{change * 100:+.2f} pp" if change is not None else "—"])
        lines += _table(["Event", "Minimum probability", "V3 accuracy (hits/picks)", "V4 accuracy (hits/picks)", "V4 − V3"], rows) + [""]
    lines += ["## Winner accuracy separately for every league", ""]
    for period_name in display_periods:
        lines += [f"### {combined_label if period_name == 'combined' else period_name}", ""]
        lines += _league_table(report["periods"][period_name], "winner") + [""]
    lines += ["The four goal markets have separate tables for every league and year in [GOALS_BY_LEAGUE.md](GOALS_BY_LEAGUE.md).", "",
              "## Probability quality on all identical fixtures", "",
              "The following scores use every fixture in the period. Home/draw/away accuracy includes draw predictions. Brier score and log loss are better when lower; their paired differences use the same fixtures for both models.", ""]
    rows = []
    for period_name in display_periods:
        scope = report["periods"][period_name]["overall"]
        for model in MODELS:
            values = scope["unfiltered"][model]["home_draw_away"]
            rows.append([period_name, model.upper(), _percent(values["accuracy"]), f"{values['brier']:.5f}" if values["brier"] is not None else "—", f"{values['log_loss']:.5f}" if values["log_loss"] is not None else "—"])
    lines += _table(["Period", "Model", "Home/draw/away accuracy", "Brier", "Log loss"], rows) + [""]
    rows = []
    for period_name in display_periods:
        scope = report["periods"][period_name]["overall"]
        for event in ("over25", "over35"):
            for model in MODELS:
                values = scope["unfiltered"][model]["goals"][event]
                rows.append([period_name, LABELS[event], model.upper(), _percent(values["mean_probability"]), _percent(values["actual_event_rate"]), f"{values['brier']:.5f}" if values["brier"] is not None else "—", f"{values['log_loss']:.5f}" if values["log_loss"] is not None else "—"])
    lines += _table(["Period", "Event", "Model", "Mean predicted", "Observed rate", "Brier", "Log loss"], rows) + [""]
    lines += ["Complementary over/under events have identical unfiltered Brier scores and log losses. Their threshold hit rates differ because they select different games. Goal probabilities come from the normalized goal component mixture; home/draw/away probabilities use the selected full model and its recorded calibration.", "",
              "## Interpretation and reproducibility", "",
              "A higher filtered hit rate may come with fewer predictions and is not by itself a paired estimate of improvement. The JSON includes the intersection of games qualifying for both models. On a common set for one fixed goal event, both models necessarily have the same correct count; winner picks can differ and their agreement counts are recorded.", "",
              "Every model/event/league/year has sample counts, coverage, mean predicted probability and a 95% Wilson interval in JSON and CSV. These intervals treat games as independent and do not account for multiple league comparisons. Small 80% samples can produce unstable hit rates. Overall paired score differences also include 95% intervals from 2,000 calendar-week bootstrap samples; these capture uncertainty conditional on the fixed training and model-selection procedure.", "",
              "The evaluation uses historical predictions generated with model coefficients fitted before the evaluation year and historical features available before each kickoff. Final models trained on the complete dataset must not be evaluated retrospectively on their own training fixtures. Both variants need the same source snapshot, cutoff and eligible fixtures. The supplied metadata records whether V3 was reused from saved annual bundles or refitted for this comparison.", "",
              "Architecture and calibration choices should be fixed using earlier validation data. The project's 2025/2026 periods have already been inspected and are historical backtests, so future fixtures are needed for a fresh confirmation. Threshold accuracy is not betting ROI: no bookmaker prices or staking policy enter these accuracy calculations.", "",
              "- [evaluation.json](evaluation.json): metadata, all annual and per-league metrics, common-qualified intersections, calibration bins and paired scores.",
              "- [threshold_summary.csv](threshold_summary.csv): model/own-filter and common-qualified rows for all five events and all periods.",
              "- [GOALS_BY_LEAGUE.md](GOALS_BY_LEAGUE.md): over/under 2.5 and 3.5 results separately by league.", ""]
    ablation = metadata.get("feature_ablation") or {key: metadata[key] for key in (
        "excluded_statistics", "excluded_feature_columns", "baseline_feature_count",
        "baseline_selected_input_feature_count", "v4_feature_count",
        "v4_selected_input_feature_count") if key in metadata}
    if ablation:
        lines += ["Recorded feature ablation:", "", "```json", json.dumps(_native(ablation), indent=2), "```", ""]
    lines += ["## Commands", "",
              "Train the separate V4 model and regenerate the chronological comparison:", "",
              "```powershell", "python football_model_v4.py train", "```", "",
              "Verify and summarize the saved comparison after training:", "",
              "```powershell", "python football_model_v4.py backtest", "```", "",
              "Predict a Championship match with V4:", "",
              "```powershell", "python football_model_v4.py predict --country england --league-id 40 --home Middlesbrough --away Wolves", "```", ""]
    return "\n".join(lines)


def _render_goals(report):
    annual = [str(year) for year in report["years"]]
    combined_label = " + ".join(annual)
    lines = ["# V3 versus V4: total-goal accuracy separately for every league", "",
             "Each cell is observed accuracy (correct / qualifying picks). Thresholds apply to each model's own probability. Zero picks have no accuracy estimate. First and second divisions are separate rows; combined results only include years with cached matches for that league.", "",
             "See [README.md](README.md) for model provenance, the evaluation method and limitations.", ""]
    for period_name in ("combined", *annual):
        lines += [f"## {combined_label if period_name == 'combined' else period_name}", ""]
        for event in EVENTS[1:]:
            lines += [f"### {LABELS[event]}", ""]
            lines += _league_table(report["periods"][period_name], event) + [""]
    return "\n".join(lines)


def _summary_rows(report):
    rows = []
    for period_name, period in report["periods"].items():
        scopes = [("overall", None, None, None, period["overall"])]
        scopes += [("league", item["country"], item["league_id"], item["division"], item["metrics"]) for item in period["leagues"]]
        for scope, country, league_id, division, metrics in scopes:
            for event in EVENTS:
                for threshold in THRESHOLDS:
                    values = metrics["events"][event]["thresholds"][str(threshold)]
                    for population, data in (("own_probability", values), ("both_models_qualify", values["qualified_by_both_models"])):
                        for model in MODELS:
                            row = dict(data[model])
                            interval = row.pop("wilson_95_interval")
                            row.update({"period": period_name, "scope": scope, "country": country,
                                        "league_id": league_id, "division": division,
                                        "available_matches": metrics["available_matches"], "event": event,
                                        "model": model, "minimum_probability": threshold, "selection": population,
                                        "wilson_lower": interval[0] if interval else None,
                                        "wilson_upper": interval[1] if interval else None})
                            rows.append(row)
    return rows


def write_reports(predictions: pd.DataFrame, metadata: dict, output: Path) -> dict:
    """Write README, league goal tables, JSON and CSV; return the JSON report dict."""
    frame = _validate(predictions, metadata)
    years = _years(metadata, frame)
    keys = _league_keys(frame, metadata)
    report = {"models": list(MODELS), "thresholds": list(THRESHOLDS),
              "events": {event: LABELS[event] for event in EVENTS},
              "available_matches": len(frame), "years": list(years), "metadata": _native(metadata), "periods": {}}
    periods = {str(year): frame[frame.kickoff.dt.year == year] for year in years}
    periods["combined"] = frame
    for period_name, period in periods.items():
        leagues = []
        for country, league_id, division in keys:
            group = period[(period.country == country) & (period.league_id == league_id)]
            if len(group) and not group.division.eq(division).all():
                raise ValueError("Source league order and fixture division labels disagree.")
            leagues.append({"country": country, "league_id": league_id, "division": division,
                            "metrics": _summarize(group)})
        report["periods"][period_name] = {"overall": _summarize(period, bootstrap=True), "leagues": leagues}
        if sum(item["metrics"]["available_matches"] for item in leagues) != len(period):
            raise ValueError("Per-league counts do not cover the complete comparison.")
    rows = _summary_rows(report)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(_render(report), encoding="utf-8")
    (output / "GOALS_BY_LEAGUE.md").write_text(_render_goals(report), encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return report
