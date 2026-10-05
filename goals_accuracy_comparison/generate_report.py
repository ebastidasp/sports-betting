"""Report held-out goal forecasts for all configured divisions, without fitting.

Run from the project directory: python goals_accuracy_comparison/generate_report.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
SOURCE = OUTPUT / "backtest"
PREVIOUS = ROOT / "over25_comparison_v2_v3"
YEARS = (2025, 2026)
MARKETS = {"over15": 2, "over25": 3, "over35": 4}
THRESHOLDS = (.6, .7, .8)
MODEL_LABELS = {"legacy": "Legacy Poisson", "v3": "V3 goal component"}


def total_tails(home, away, rho):
    home, away, rho = [np.asarray(values, dtype=float) for values in (home, away, rho)]
    if not (np.isfinite(home).all() and np.isfinite(away).all() and np.isfinite(rho).all()
            and (home > 0).all() and (away > 0).all()):
        raise ValueError("Goal rates must be positive and finite; rho must be finite.")
    rho = np.clip(rho, np.maximum(-1 / home, -1 / away) + 1e-10,
                  np.minimum(1., 1 / (home * away)) - 1e-10)
    tails = poisson.sf(np.array([1, 2, 3])[None, :], (home + away)[:, None])
    tails[:, 0] -= np.exp(-home - away) * home * away * rho
    if ((tails < -1e-12) | (tails > 1 + 1e-12)).any():
        raise ValueError("Invalid total-goal probability.")
    tails = np.clip(tails, 0, 1)
    if (np.diff(tails, axis=1) > 1e-12).any():
        raise ValueError("Goal tail probabilities must be nested.")
    return tails, rho


def event_metrics(actual, probabilities):
    actual = np.asarray(actual, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    bounded = np.clip(probabilities, 1e-12, 1 - 1e-12)
    result = {"matches": len(actual), "actual_event_rate": float(actual.mean()),
              "mean_probability": float(probabilities.mean()),
              "brier": float(np.mean((probabilities - actual) ** 2)),
              "log_loss": float(-np.mean(actual * np.log(bounded)
                                          + (1 - actual) * np.log1p(-bounded))),
              "filters": {}}
    for threshold in THRESHOLDS:
        selected = probabilities >= threshold
        count, correct = int(selected.sum()), int(actual[selected].sum())
        accuracy = correct / count if count else None
        interval = None
        if count:
            z = 1.959963984540054
            denominator = 1 + z ** 2 / count
            center = (accuracy + z ** 2 / (2 * count)) / denominator
            radius = z * np.sqrt(accuracy * (1 - accuracy) / count
                                 + z ** 2 / (4 * count ** 2)) / denominator
            interval = [float(center - radius), float(center + radius)]
        result["filters"][str(threshold)] = {"selected": count, "correct": correct,
            "incorrect": count - correct, "accuracy": accuracy,
            "coverage": count / len(actual),
            "mean_probability": float(probabilities[selected].mean()) if count else None,
            "95pct_wilson_interval": interval}
    return result


def paired_scores(frame, market):
    actual = frame[f"actual_{market}"].to_numpy(dtype=float)
    old = frame[f"p_{market}_legacy"].to_numpy(dtype=float)
    new = frame[f"p_{market}_v3"].to_numpy(dtype=float)
    old_q, new_q = np.clip(old, 1e-12, 1 - 1e-12), np.clip(new, 1e-12, 1 - 1e-12)
    differences = {"brier": (new - actual) ** 2 - (old - actual) ** 2,
                   "log_loss": -actual * np.log(new_q) - (1 - actual) * np.log1p(-new_q)
                               + actual * np.log(old_q) + (1 - actual) * np.log1p(-old_q)}
    weeks = pd.to_datetime(frame.kickoff, utc=True).dt.strftime("%G-%V")
    rng = np.random.default_rng(42)
    result = {}
    for name, delta in differences.items():
        blocks = pd.DataFrame({"week": weeks, "delta": delta}).groupby("week").agg(
            total=("delta", "sum"), count=("delta", "size"))
        indices = rng.integers(0, len(blocks), (5000, len(blocks)))
        bootstrap = blocks.total.to_numpy()[indices].sum(axis=1) / blocks["count"].to_numpy()[indices].sum(axis=1)
        result[name] = {"v3_minus_legacy": float(delta.mean()),
                        "95pct_week_bootstrap": np.quantile(bootstrap, [.025, .975]).tolist()}
    return result


def summarize(frame, include_leagues=True, include_divisions=True):
    result = {"matched_fixtures": len(frame),
              "first_test_kickoff": pd.to_datetime(frame.kickoff, utc=True).min().isoformat(),
              "latest_test_kickoff": pd.to_datetime(frame.kickoff, utc=True).max().isoformat(),
              "markets": {}}
    for market in MARKETS:
        actual = frame[f"actual_{market}"].to_numpy(dtype=int)
        result["markets"][market] = {
            model: event_metrics(actual, frame[f"p_{market}_{model}"])
            for model in MODEL_LABELS}
        result["markets"][market]["paired_differences"] = paired_scores(frame, market)
    if include_divisions:
        result["divisions"] = {division: summarize(group, False, False)
                               for division, group in frame.groupby("division")}
    if include_leagues:
        result["leagues"] = {}
        for country, group in frame.groupby("country"):
            result["leagues"][country] = {"matched_fixtures": len(group),
                "league_ids": sorted(int(value) for value in group.league_id.unique()),
                "markets": {market: {model: event_metrics(group[f"actual_{market}"],
                                       group[f"p_{market}_{model}"])
                                      for model in MODEL_LABELS} for market in MARKETS}}
            result["leagues"][country]["divisions"] = {
                division: {"matched_fixtures": len(part), "league_id": int(part.league_id.iloc[0]),
                    "markets": {market: {model: event_metrics(part[f"actual_{market}"],
                                            part[f"p_{market}_{model}"])
                                         for model in MODEL_LABELS} for market in MARKETS}}
                for division, part in group.groupby("division")}
    return result


def percent(value):
    return f"{value:.2%}" if value is not None else "—"


def market_label(market):
    return f"Over {MARKETS[market] - .5:.1f}"


def hit_table(period):
    lines = ["| Goal market | Minimum forecast | Legacy correct / selected | Legacy accuracy | V3 correct / selected | V3 accuracy | V3 - legacy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for market in MARKETS:
        for threshold in THRESHOLDS:
            old = period["markets"][market]["legacy"]["filters"][str(threshold)]
            new = period["markets"][market]["v3"]["filters"][str(threshold)]
            delta = f"{100 * (new['accuracy'] - old['accuracy']):+.2f} pp" if old["selected"] and new["selected"] else "—"
            lines.append(f"| {market_label(market)} | ≥{threshold:.0%} | {old['correct']:,} / {old['selected']:,} | {percent(old['accuracy'])} | {new['correct']:,} / {new['selected']:,} | {percent(new['accuracy'])} | {delta} |")
    return lines



def main():
    from render_readme import readme
    backtest = json.loads((SOURCE / "evaluation.json").read_text(encoding="utf-8"))
    previous = json.loads((PREVIOUS / "evaluation.json").read_text(encoding="utf-8"))
    if backtest.get("scope") != "all_configured_divisions":
        raise ValueError("Generate the all-division source forecasts with extend_backtest.py first.")
    if backtest["training_cutoff"] != previous["training_cutoff"]:
        raise ValueError("Keep the historical data cutoff fixed when extending the comparison.")
    metadata_paths = sorted(SOURCE.glob("legacy_*_metadata.json"))
    metadata = {item["country"]: item for path in metadata_paths
                for item in [json.loads(path.read_text(encoding="utf-8"))]}
    inputs = [SOURCE / "evaluation.json", PREVIOUS / "evaluation.json", *metadata_paths]
    inputs += [SOURCE / f"comparison_predictions_{year}.csv" for year in YEARS]
    inputs += [SOURCE / f"v3_predictions_{year}.csv" for year in YEARS]
    inputs += [ROOT / "new_model_v3" / f"evaluation_model_{year}.joblib" for year in YEARS]
    inputs += [ROOT / name for name in ("football_poisson.py", "football_model_v2.py", "football_model_v3.py")]
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in inputs}
    report = {"baseline": "football_poisson.py, fixed defaults, trained before each year",
              "new_model": "football_model_v3.py, normalized goal component only",
              "data_cutoff": previous["training_cutoff"], "countries": sorted(metadata),
              "scope": "all_configured_divisions", "sources": backtest["sources"],
              "events": {market: f"at least {minimum} total FT goals" for market, minimum in MARKETS.items()},
              "thresholds": list(THRESHOLDS), "input_sha256": hashes,
              "model_parameters": {}, "periods": {}}
    all_frames, summary_rows = [], []
    for year in YEARS:
        frame = pd.read_csv(SOURCE / f"comparison_predictions_{year}.csv")
        if frame.empty or frame.duplicated(["country", "fixture_id"]).any():
            raise ValueError("Expected unique common fixtures.")
        np.testing.assert_array_equal(frame.home_goals, frame.home_goals_legacy)
        np.testing.assert_array_equal(frame.away_goals, frame.away_goals_legacy)
        np.testing.assert_array_equal(frame.league_id, frame.league_id_legacy)
        np.testing.assert_array_equal(frame.division, np.where(frame.league_id == frame.primary_league, "first", "second"))
        np.testing.assert_array_equal(pd.to_datetime(frame.kickoff, utc=True),
                                      pd.to_datetime(frame.kickoff_legacy, utc=True))
        bundle = joblib.load(ROOT / "new_model_v3" / f"evaluation_model_{year}.joblib")
        cutoff = pd.Timestamp(bundle["training_cutoff"])
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        if cutoff > start:
            raise ValueError("Evaluation model was fitted after the evaluation started.")
        dates = pd.to_datetime(frame.kickoff, utc=True)
        if ((dates < start) | (dates >= pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
                | (dates + pd.Timedelta(hours=3) > pd.Timestamp(previous["training_cutoff"]))).any():
            raise ValueError("Fixture outside the historical evaluation window.")
        goals = [(spec, weight) for spec, weight in zip(bundle["selection"]["components"], bundle["selection"]["weights"])
                 if spec["family"] == "goals" and weight > 0]
        if len(goals) != 1:
            raise ValueError("Saved mean rates support this report only with one V3 goal component.")
        legacy_rho = frame.country.map({country: item["periods"][str(year)].get("rho", np.nan)
                                       for country, item in metadata.items()}).to_numpy(dtype=float)
        rho_values = {"legacy": legacy_rho, "v3": np.full(len(frame), goals[0][0].get("rho", 0.))}
        report["model_parameters"][str(year)] = {"v3_training_cutoff": bundle["training_cutoff"],
            "v3_selection": bundle["selection"],
            "legacy": {country: {"settings": item["settings"], "period": item["periods"][str(year)]}
                       for country, item in metadata.items()}}
        for model in MODEL_LABELS:
            home = frame[f"expected_home_{model}"].to_numpy(dtype=float)
            away = frame[f"expected_away_{model}"].to_numpy(dtype=float)
            tails, local_rho = total_tails(home, away, rho_values[model])
            np.testing.assert_allclose(tails[:, 1], frame[f"p_over25_{model}"], atol=1e-12, rtol=0)
            # Independently sum corrected score grids to verify all three tail formulas.
            scores = np.arange(81)
            for index in np.linspace(0, len(frame) - 1, 5, dtype=int):
                grid = np.outer(poisson.pmf(scores, home[index]), poisson.pmf(scores, away[index]))
                grid[0, 0] *= 1 - home[index] * away[index] * local_rho[index]
                grid[0, 1] *= 1 + home[index] * local_rho[index]
                grid[1, 0] *= 1 + away[index] * local_rho[index]
                grid[1, 1] *= 1 - local_rho[index]
                direct = [grid[np.add.outer(scores, scores) >= minimum].sum() for minimum in MARKETS.values()]
                np.testing.assert_allclose(tails[index], direct, atol=1e-12, rtol=0)
            frame[f"rho_{model}"] = local_rho
            for column, market in enumerate(MARKETS):
                frame[f"p_{market}_{model}"] = tails[:, column]
        for market, minimum in MARKETS.items():
            frame[f"actual_{market}"] = (frame.home_goals + frame.away_goals >= minimum).astype(int)
        period = summarize(frame)
        period["all_v3_fixtures"] = backtest["periods"][str(year)]["all_v3_fixtures"]
        period["available_v3_divisions"] = backtest["periods"][str(year)]["available_v3_divisions"]
        available = pd.read_csv(SOURCE / f"v3_predictions_{year}.csv")
        available["division"] = np.where(available.league_id == available.primary_league, "first", "second")
        period["available_v3_leagues"] = {
            country: {division: int(count) for division, count in group.division.value_counts().items()}
            for country, group in available.groupby("country")}
        for model, old_label in (("legacy", "legacy"), ("v3", "v3_goal_component")):
            old = previous["periods"][str(year)][old_label]
            new = period["divisions"]["first"]["markets"]["over25"][model]
            for metric in ("brier", "log_loss"):
                np.testing.assert_allclose(new[metric], old[metric], atol=1e-12, rtol=0)
            for threshold in THRESHOLDS:
                for metric in ("selected", "correct"):
                    assert new["filters"][str(threshold)][metric] == old["filters"][str(threshold)][metric]
        report["periods"][str(year)] = period
        all_frames.append(frame)
    report["periods"]["combined"] = summarize(pd.concat(all_frames, ignore_index=True))
    combined = report["periods"]["combined"]
    combined["all_v3_fixtures"] = sum(report["periods"][str(year)]["all_v3_fixtures"] for year in YEARS)
    combined["available_v3_divisions"] = {
        division: sum(report["periods"][str(year)]["available_v3_divisions"].get(division, 0) for year in YEARS)
        for division in ("first", "second")}
    combined["available_v3_leagues"] = {
        country: {division: sum(report["periods"][str(year)]["available_v3_leagues"].get(country, {}).get(division, 0)
                               for year in YEARS) for division in ("first", "second")}
        for country in report["countries"]}
    for period_name, period in report["periods"].items():
        scopes = [("aggregate", "all", None, None, period)]
        scopes += [("division", division, None, None, values) for division, values in period["divisions"].items()]
        for country, values in period["leagues"].items():
            scopes.append(("country", "all", country, None, values))
            scopes += [("league", division, country, part["league_id"], part)
                       for division, part in values["divisions"].items()]
        for scope, division, country, league_id, values in scopes:
            for market, models in values["markets"].items():
                for model in MODEL_LABELS:
                    for threshold in THRESHOLDS:
                        row = models[model]["filters"][str(threshold)].copy()
                        interval = row.pop("95pct_wilson_interval")
                        row.update({"period": period_name, "scope": scope, "division": division,
                                    "country": country, "league_id": league_id,
                                    "market": market_label(market), "model": model,
                                    "minimum_probability": threshold,
                                    "wilson_lower": interval[0] if interval else None,
                                    "wilson_upper": interval[1] if interval else None})
                        summary_rows.append(row)
    for relative, expected in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source file changed during calculation: {relative}")
    for year, frame in zip(YEARS, all_frames):
        frame.to_csv(OUTPUT / f"predictions_{year}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "threshold_summary.csv", index=False)
    (OUTPUT / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(readme(report), encoding="utf-8")
    print(f"Created {OUTPUT / 'README.md'}")
    for year in (*YEARS, "combined"):
        print(f"\n{year}: {report['periods'][str(year)]['matched_fixtures']} common fixtures")
        console_table = "\n".join(hit_table(report["periods"][str(year)]))
        print(console_table.replace("≥", ">=").replace("—", "n/a"))
    print("\nVerified original inputs unchanged; reproduced prior first-division over-2.5 results.")


if __name__ == "__main__":
    main()
