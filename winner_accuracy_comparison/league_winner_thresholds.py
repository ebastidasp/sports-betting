"""Summarize fresh annual V2/V3 winner predictions separately for every league.

python winner_accuracy_comparison/league_winner_thresholds.py
Reconstruct forecasts first with compare_models.py after retraining V3.
This script does not fit models or access the network.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_models import validate_probabilities
from render_readme import COUNTRIES, table

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent
YEARS = (2025, 2026)
THRESHOLDS = (.6, .7, .8)
MODELS = ("v2", "v3")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_metrics(frame, model, threshold, additional_mask=None):
    probabilities = frame[[f"p_{outcome}_{model}" for outcome in ("home", "draw", "away")]].to_numpy()
    confidence = np.maximum(probabilities[:, 0], probabilities[:, 2])
    predicted = np.where(probabilities[:, 0] >= probabilities[:, 2], 0, 2)
    selected = confidence >= threshold
    if additional_mask is not None:
        selected &= additional_mask
    # For these thresholds a selected winner must also be the 1X2 argmax.
    np.testing.assert_array_equal(predicted[selected], probabilities[selected].argmax(axis=1))
    actual = frame.target.to_numpy(dtype=int)
    n = int(selected.sum())
    correct = int((predicted[selected] == actual[selected]).sum())
    draw_losses = int((actual[selected] == 1).sum())
    accuracy = correct / n if n else None
    interval = None
    if n:
        z = 1.959963984540054
        denominator = 1 + z ** 2 / n
        center = (accuracy + z ** 2 / (2 * n)) / denominator
        radius = z * np.sqrt(accuracy * (1 - accuracy) / n + z ** 2 / (4 * n ** 2)) / denominator
        interval = [float(center - radius), float(center + radius)]
    return {
        "selected": n, "correct": correct, "incorrect": n - correct,
        "draw_losses": draw_losses, "other_team_wins": n - correct - draw_losses,
        "accuracy": accuracy, "coverage": n / len(frame) if len(frame) else None,
        "mean_win_probability": float(confidence[selected].mean()) if n else None,
        "wilson_95_interval": interval,
    }


def league_metrics(frame):
    metrics = {"available_matches": len(frame), "thresholds": {}}
    for threshold in THRESHOLDS:
        values = {model: selected_metrics(frame, model, threshold) for model in MODELS}
        both = np.ones(len(frame), dtype=bool)
        for model in MODELS:
            both &= frame[[f"p_home_{model}", f"p_away_{model}"]].max(axis=1).to_numpy() >= threshold
        paired = {model: selected_metrics(frame, model, threshold, both) for model in MODELS}
        values["qualified_by_both_models"] = paired
        values["v3_minus_v2_accuracy"] = (
            values["v3"]["accuracy"] - values["v2"]["accuracy"]
            if values["v3"]["selected"] and values["v2"]["selected"] else None
        )
        metrics["thresholds"][str(threshold)] = values
    for model in MODELS:
        counts = [metrics["thresholds"][str(t)][model]["selected"] for t in THRESHOLDS]
        if not (counts[0] >= counts[1] >= counts[2]):
            raise ValueError("Threshold counts must be cumulative.")
    return metrics


def result_cell(values):
    if not values["selected"]:
        return "— (0 picks)"
    return f"{values['accuracy']:.1%} ({values['correct']}/{values['selected']})"


def render(report):
    local_cutoff = pd.Timestamp(report["data_cutoff"]).to_pydatetime().astimezone(timezone(timedelta(hours=-5)))
    lines = [
        "# Winner accuracy at 60%, 70% and 80%, separately for every league", "",
        "This analysis compares the fixed V2 reference with the retrained V3 annual evaluation bundles. Every configured first- and second-division league has its own row; leagues are never combined into one accuracy figure.", "",
        f"The combined period covers cached matches in **2025 and 2026 through {local_cutoff:%d %B %Y, %H:%M:%S} Colombia time**. The full comparison contains **{report['available_matches']:,} distinct matches across {len(report['leagues'])} leagues**. The 2026 period and cache coverage are incomplete.", "",
        "A game qualifies when **max(P(home win), P(away win)) ≥ the threshold**. The pick is the team with the higher win probability. An actual draw counts as a failed winner prediction. Thresholds are inclusive and cumulative: an 80% pick also belongs to the 70% and 60% groups.", "",
        "Each cell shows **observed accuracy (correct / qualifying picks)**. A dash means no qualifying predictions. V2 and V3 apply the threshold to their own probabilities and can therefore select different fixtures. Differences between these filtered rates are descriptive, rather than a paired estimate of model improvement.", "",
    ]
    missing = [f"{COUNTRIES.get(league['country'], league['country'])} ({league['league_id']})"
               for league in report["leagues"] if not league["periods"]["2026"]["available_matches"]]
    if missing:
        lines += ["**No cached 2026 evaluation matches:** " + ", ".join(missing)
                  + ". Their combined results therefore use 2025 matches only.", ""]
    for period in ("combined", "2025", "2026"):
        lines += [f"## {'2025 + 2026' if period == 'combined' else period}: league results", ""]
        for threshold in THRESHOLDS:
            lines += [f"### Win probability ≥{threshold:.0%}", ""]
            rows = []
            for league in report["leagues"]:
                values = league["periods"][period]
                selected = values["thresholds"][str(threshold)]
                change = selected["v3_minus_v2_accuracy"]
                rows.append([
                    COUNTRIES.get(league["country"], league["country"]),
                    "First" if league["division"] == "first" else "Second",
                    league["league_id"], values["available_matches"],
                    result_cell(selected["v2"]), result_cell(selected["v3"]),
                    f"{change * 100:+.1f} pp" if change is not None else "—",
                ])
            lines += table(["Country", "Division", "League ID", "Available games", "V2 accuracy (hits/picks)", "V3 accuracy (hits/picks)", "V3 − V2"], rows)
            lines += [""]
    lines += [
        "## How to interpret the results", "",
        "High percentages with very few picks carry substantial uncertainty. For example, 1/1 is 100% observed accuracy, but its 95% Wilson interval is approximately 20.7%–100%. The JSON includes an interval, mean predicted probability, coverage, draw failures and other-team-win failures for every league/model/threshold/year. These intervals treat games as independent and are not adjusted for comparisons across many leagues.", "",
        "A higher filtered hit rate may come with fewer predictions. A threshold is a model estimate, not a guaranteed success rate; a model assigning ≥70% can still achieve below 70% in a particular league. Choosing leagues or thresholds after seeing these results requires a new future test to validate that choice.", "",
        "Both model coefficients are fitted strictly before the evaluation year. Features update using earlier results with a three-hour availability delay. V3 uses its full annual home/draw/away blend and saved calibration. V2 is the fixed previously selected classifier, fitted on the same prior history. Architecture selection used 2024 first-division validation, so this is an annual historical test of the retrained procedure rather than a new forward test after retraining.", "",
        "The original Poisson model is not included in these full-coverage tables: its cached historical comparison has fewer supported fixtures and an earlier data snapshot. The baseline here is V2. USA has only one configured league; all other configured countries have two. Country, division and SQLite league ID identify each league without requiring external league-name data.", "",
        "## Files and reproduction", "",
        "- [league_winner_thresholds.json](league_winner_thresholds.json): all separate league results and uncertainty, including accuracy on fixtures qualifying for both models at a threshold.",
        "- [all_predictions_2025.csv](all_predictions_2025.csv) and [all_predictions_2026.csv](all_predictions_2026.csv): underlying full-coverage annual predictions.",
        "- [forecast_metadata.json](forecast_metadata.json): annual model selections and training cutoffs.", "",
        "After retraining, reconstruct annual forecasts and regenerate this report from the project directory:", "",
        "```powershell", "python winner_accuracy_comparison/compare_models.py",
        "python winner_accuracy_comparison/league_winner_thresholds.py", "```", "",
        "To summarize already refreshed predictions, run only the second command. It refuses inputs that disagree with the latest training cutoff or protected annual-model hashes. Saved model files remain unchanged.", "",
    ]
    return "\n".join(lines)


def main():
    metadata_path = OUTPUT / "forecast_metadata.json"
    info = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "new_model_v3" / "evaluation.json").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(evaluation["training_cutoff"])
    if cutoff != pd.Timestamp(info["data_cutoff"]) or info["sources"] != evaluation["sources"]:
        raise ValueError("Forecasts are stale. Run compare_models.py after retraining.")
    protected = json.loads((OUTPUT / "protected_input_sha256.json").read_text(encoding="utf-8"))
    paths = [ROOT / name for name in ("football_model_v2.py", "football_model_v3.py")]
    paths += [ROOT / "new_model_v3" / name for name in (
        "evaluation.json", "frozen_selection.json", *[f"evaluation_model_{year}.joblib" for year in YEARS],
        *[f"test_predictions_{year}.csv" for year in YEARS],
    )]
    input_hashes = {metadata_path.relative_to(ROOT).as_posix(): digest(metadata_path)}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        input_hashes[relative] = digest(path)
        if input_hashes[relative] != protected.get(relative):
            raise ValueError(f"Model input changed after forecasting: {relative}")
    frames = {}
    for year in YEARS:
        path = OUTPUT / f"all_predictions_{year}.csv"
        input_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        frame = pd.read_csv(path)
        frame["kickoff"] = pd.to_datetime(frame.kickoff, utc=True)
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), cutoff)
        period_info = info["periods"][str(year)]
        if pd.Timestamp(period_info["training_start"]) > start:
            raise ValueError("Annual model training overlaps its evaluation year.")
        if len(frame) != period_info["all_fixtures"] or not period_info["first_division_forecasts_reproduced"]:
            raise ValueError("Annual forecast provenance failed.")
        if not ((frame.kickoff >= start) & (frame.kickoff < end)
                & (frame.kickoff + pd.Timedelta(hours=3) <= cutoff)).all():
            raise ValueError("Fixture falls outside the completed evaluation period.")
        actual = np.where(frame.home_goals > frame.away_goals, 0, np.where(frame.home_goals == frame.away_goals, 1, 2))
        np.testing.assert_array_equal(actual, frame.target)
        expected_division = np.where(frame.league_id == frame.primary_league, "first", "second")
        np.testing.assert_array_equal(expected_division, frame.division)
        previous = pd.read_csv(ROOT / "new_model_v3" / f"test_predictions_{year}.csv").sort_values(["country", "fixture_id"])
        primary = frame[frame.division == "first"].sort_values(["country", "fixture_id"])
        np.testing.assert_array_equal(primary[["country", "fixture_id"]], previous[["country", "fixture_id"]])
        for model in MODELS:
            columns = [f"p_{outcome}_{model}" for outcome in ("home", "draw", "away")]
            validate_probabilities(frame[columns])
            np.testing.assert_allclose(primary[columns], previous[columns], atol=1e-10, rtol=0)
        frames[str(year)] = frame
    frames["combined"] = pd.concat(list(frames.values()), ignore_index=True)
    if frames["combined"].duplicated(["country", "fixture_id"]).any():
        raise ValueError("Duplicated historical fixture.")
    configured = {(country, int(league)) for country, source in info["sources"].items() for league in source["league_ids"]}
    observed = set(map(tuple, frames["combined"][["country", "league_id"]].drop_duplicates().to_numpy()))
    if not observed <= configured:
        raise ValueError("Forecast contains an unconfigured league.")
    report = {"data_cutoff": info["data_cutoff"], "created_at": datetime.now(timezone.utc).isoformat(),
              "available_matches": len(frames["combined"]), "models": list(MODELS),
              "thresholds": list(THRESHOLDS), "sources": info["sources"], "input_sha256": input_hashes,
              "selection_rule": "max(p_home, p_away) >= threshold; pick the more probable winner; draws are failures",
              "leagues": []}
    for country, source in sorted(info["sources"].items()):
        for index, league_id in enumerate(source["league_ids"]):
            league = {"country": country, "league_id": int(league_id), "division": "first" if index == 0 else "second", "periods": {}}
            for period, frame in frames.items():
                group = frame[(frame.country == country) & (frame.league_id == league_id)]
                league["periods"][period] = league_metrics(group)
            report["leagues"].append(league)
    if sum(league["periods"]["combined"]["available_matches"] for league in report["leagues"]) != report["available_matches"]:
        raise ValueError("Per-league fixture counts do not cover the full sample.")
    for relative, original_digest in input_hashes.items():
        if digest(ROOT / relative) != original_digest:
            raise ValueError(f"Input changed during analysis: {relative}")
    (OUTPUT / "league_winner_thresholds.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "WINNER_THRESHOLDS_BY_LEAGUE.md").write_text(render(report), encoding="utf-8")
    for league in report["leagues"]:
        values = league["periods"]["combined"]["thresholds"]
        print(f"{league['country']} {league['league_id']} ({league['division']}): " + " | ".join(
            f"{threshold:.0%}: V2 {result_cell(values[str(threshold)]['v2'])}; V3 {result_cell(values[str(threshold)]['v3'])}"
            for threshold in THRESHOLDS), flush=True)
    print(f"Updated {OUTPUT / 'WINNER_THRESHOLDS_BY_LEAGUE.md'}", flush=True)


if __name__ == "__main__":
    main()
