"""Fresh chronological V1/V3 winner comparison, separately for every league.

python winner_accuracy_comparison/compare_v1_v3.py
python winner_accuracy_comparison/compare_v1_v3.py --mode report
Original models and SQLite caches are never overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare_models import ROOT, YEARS, validate_probabilities
from league_winner_thresholds import digest, result_cell, selected_metrics
from render_readme import COUNTRIES, table
import football_poisson as v1

PARENT = Path(__file__).resolve().parent
OUTPUT = PARENT / "v1_v3"
THRESHOLDS = (.6, .7, .8)
MODELS = ("v1", "v3")
SETTINGS = {"regularization": .01, "recency_decay": .95, "ewma_alpha": .25,
            "min_history": 5, "dixon_coles": True}
EXPORT_COLUMNS = ["country", "fixture_id", "kickoff", "league_id", "home_goals", "away_goals",
                  "expected_home_v1", "expected_away_v1", "rho_v1", "p_home_v1", "p_draw_v1", "p_away_v1"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_country(country, source, cutoff_value):
    """Country state stays isolated in one worker process; models stay in memory."""
    cutoff = pd.Timestamp(cutoff_value)
    v1.activate_config({"country": country, "league_ids": source["league_ids"]})
    with closing(sqlite3.connect(Path(source["path"]).as_uri() + "?mode=ro", uri=True)) as con:
        try:
            rows = v1.load_matches(con)
        except SystemExit:
            rows = pd.DataFrame()
    info = {"country": country, "league_ids": source["league_ids"], "settings": SETTINGS, "periods": {}}
    frame = pd.DataFrame()
    if not rows.empty:
        rows["kickoff"] = pd.to_datetime(rows.kickoff, utc=True)
        rows = rows[(rows.status == "FT") & rows.league_id.isin(source["league_ids"])
                    & (rows.kickoff + pd.Timedelta(hours=3) <= cutoff)].copy()
    if not rows.empty:
        # V1 updates team states immediately after each match. Ensure no later
        # same-team fixture could observe a still unfinished earlier match.
        appearances = pd.concat([rows[[f"{venue}_id", "kickoff"]].rename(columns={f"{venue}_id": "team"})
                                 for venue in ("home", "away")], ignore_index=True)
        gaps = appearances.sort_values("kickoff").groupby("team").kickoff.diff()
        if (gaps.notna() & (gaps < pd.Timedelta(hours=3))).any():
            raise ValueError(f"Overlapping {country} fixtures require a chronology audit.")
        frame, _, _ = v1.build_features(rows, SETTINGS["ewma_alpha"], SETTINGS["min_history"])
        if not frame.empty:
            frame = frame.merge(rows[["fixture_id", "league_id"]], on="fixture_id", validate="one_to_one")
    forecasts = {}
    with threadpool_limits(limits=1):
        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), cutoff)
            training = frame[frame.kickoff + pd.Timedelta(hours=3) <= start] if not frame.empty else frame
            test = frame[(frame.kickoff >= start) & (frame.kickoff < end)] if not frame.empty else frame
            supported = pd.DataFrame()
            export = pd.DataFrame(columns=EXPORT_COLUMNS)
            rho = None
            if not training.empty and not test.empty:
                models, rho = v1.fit_team_bundle(training, start, SETTINGS["regularization"],
                                                SETTINGS["recency_decay"], SETTINGS["dixon_coles"])
                supported = v1.supported_fixtures(models, test)
                if not supported.empty:
                    home, away = v1.predict_team_rates(models, supported)
                    probabilities = validate_probabilities(v1.shared_probabilities(home, away, rho))
                    export = supported[["fixture_id", "kickoff", "league_id", "home_goals", "away_goals"]].copy()
                    export["country"] = country
                    export["expected_home_v1"], export["expected_away_v1"] = home, away
                    export["rho_v1"] = float(rho)
                    export[[f"p_{outcome}_v1" for outcome in ("home", "draw", "away")]] = probabilities
                    export = export[EXPORT_COLUMNS]
            leagues = {}
            for league in source["league_ids"]:
                stats_rows = rows[(rows.league_id == league) & (rows.kickoff >= start) & (rows.kickoff < end)] if not rows.empty else rows
                candidates = test[test.league_id == league] if not test.empty else test
                count = int((export.league_id == league).sum())
                reason = None
                if not count:
                    if stats_rows.empty:
                        reason = "No completed test fixtures with both teams' cached statistics."
                    elif candidates.empty:
                        reason = "Insufficient prior venue history or provisional promoted-team Elo."
                    elif training.empty:
                        reason = "No eligible training history before the evaluation year."
                    else:
                        reason = "No fixtures supported by the fitted prior-year home/away venue models."
                leagues[str(league)] = {"with_both_team_stats": len(stats_rows), "eligible_history": len(candidates),
                                       "forecastable": count, "unavailable_reason": reason}
            info["periods"][str(year)] = {"fit_cutoff": start.isoformat(), "test_end": end.isoformat(),
                                        "training_matches": len(training), "rho": float(rho) if rho is not None else None,
                                        "test_matches": len(export), "leagues": leagues}
            forecasts[str(year)] = export
    return info, forecasts


def current_inputs():
    evaluation_path = ROOT / "new_model_v3" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    v3_info_path = PARENT / "forecast_metadata.json"
    v3_info = json.loads(v3_info_path.read_text(encoding="utf-8"))
    if evaluation["training_cutoff"] != v3_info["data_cutoff"] or evaluation["sources"] != v3_info["sources"]:
        raise ValueError("V3 annual forecasts are stale. Run compare_models.py first.")
    protected = json.loads((PARENT / "protected_input_sha256.json").read_text(encoding="utf-8"))
    paths = [ROOT / name for name in ("football_poisson.py", "football_model_v2.py", "football_model_v3.py")]
    paths += list(ROOT.glob("*_poisson.joblib"))
    for directory in (ROOT / "new_model", ROOT / "new_model_v3"):
        paths += [path for path in directory.glob("*") if path.is_file()]
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    for relative in ("football_poisson.py", "football_model_v3.py", "new_model_v3/evaluation.json",
                     *[f"new_model_v3/evaluation_model_{year}.joblib" for year in YEARS]):
        if hashes[relative] != protected.get(relative):
            raise ValueError(f"An input changed since V3 forecasting: {relative}")
    for path in [v3_info_path, *[PARENT / f"all_predictions_{year}.csv" for year in YEARS]]:
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    for source in evaluation["sources"].values():
        path = Path(source["path"])
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        wal = Path(str(path) + "-wal")
        if wal.exists():
            hashes[wal.relative_to(ROOT).as_posix()] = digest(wal)
    return evaluation, hashes


def check_hashes(hashes):
    for relative, expected in hashes.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"An input changed during or since forecasting: {relative}")


def generate_predictions(evaluation, hashes, workers):
    cutoff = evaluation["training_cutoff"]
    info = {"data_cutoff": cutoff, "sources": evaluation["sources"], "input_sha256": hashes,
            "settings": SETTINGS, "countries": {}, "periods": {}}
    for country in evaluation["sources"]:
        previous = json.loads((ROOT / "over25_comparison_v2_v3" / f"legacy_{country}_metadata.json").read_text(encoding="utf-8"))
        if previous["settings"] != SETTINGS:
            raise ValueError(f"V1 baseline settings changed: {country}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run_country, country, source, cutoff): country
                for country, source in evaluation["sources"].items()}
        for future in as_completed(jobs):
            country = jobs[future]
            country_info, forecasts = future.result()
            info["countries"][country] = country_info
            write_json(OUTPUT / f"v1_{country}_metadata.json", country_info)
            for year, frame in forecasts.items():
                frame.to_csv(OUTPUT / f"v1_{country}_{year}.csv", index=False)
            print(f"V1 {country}: " + "; ".join(
                f"{year} {values['test_matches']} forecasts ({values['training_matches']} prior training matches)"
                for year, values in country_info["periods"].items()), flush=True)
    for year in YEARS:
        full = pd.read_csv(PARENT / f"all_predictions_{year}.csv")
        old = pd.concat([pd.read_csv(OUTPUT / f"v1_{country}_{year}.csv") for country in evaluation["sources"]], ignore_index=True)
        common = full.merge(old, on=["country", "fixture_id"], validate="one_to_one", suffixes=("", "_v1"))
        if len(common) != len(old):
            raise ValueError("A V1 fixture is absent from the refreshed V3 forecasts.")
        for column in ("league_id", "home_goals", "away_goals"):
            np.testing.assert_array_equal(common[column], common[f"{column}_v1"])
        np.testing.assert_array_equal(pd.to_datetime(common.kickoff, utc=True), pd.to_datetime(common.kickoff_v1, utc=True))
        common.to_csv(OUTPUT / f"common_predictions_{year}.csv", index=False)
        info["periods"][str(year)] = {"common_matches": len(common), "all_v3_matches": len(full),
                                    "latest_common_kickoff": pd.to_datetime(common.kickoff, utc=True).max().isoformat()}
        print(f"Common {year}: {len(common)} fresh V1/V3 fixtures", flush=True)
    check_hashes(hashes)
    write_json(OUTPUT / "forecast_metadata.json", info)


def summarize(frame):
    values = {"common_matches": len(frame), "thresholds": {}}
    for threshold in THRESHOLDS:
        row = {model: selected_metrics(frame, model, threshold) for model in MODELS}
        both = np.ones(len(frame), dtype=bool)
        for model in MODELS:
            both &= frame[[f"p_home_{model}", f"p_away_{model}"]].max(axis=1).to_numpy() >= threshold
        row["qualified_by_both_models"] = {model: selected_metrics(frame, model, threshold, both) for model in MODELS}
        row["v3_minus_v1_accuracy"] = (row["v3"]["accuracy"] - row["v1"]["accuracy"]
                                        if row["v1"]["selected"] and row["v3"]["selected"] else None)
        values["thresholds"][str(threshold)] = row
    return values


def render(report):
    local_cutoff = pd.Timestamp(report["data_cutoff"]).to_pydatetime().astimezone(timezone(timedelta(hours=-5)))
    covered = sum(bool(league["periods"]["combined"]["common_matches"]) for league in report["leagues"])
    lines = ["# V1 versus retrained V3: winner accuracy separately for every league", "",
        "V1 is the original **football_poisson.py** model. V3 uses its full saved annual home/draw/away blend, including calibration. This report refreshes V1's historical fits from the current SQLite data instead of reusing the older prediction snapshot.", "",
        f"Evaluation covers cached **2025 and 2026 matches through {local_cutoff:%d %B %Y, %H:%M:%S} Colombia time**. There are **{report['common_matches']:,} matched V1/V3 fixtures across {covered} comparable leagues**. All {len(report['leagues'])} configured leagues are listed separately, including those without V1 coverage.", "",
        "Both models are restricted to the **same forecastable match population within each league**. Each then selects games using its own **max(P(home win), P(away win)) ≥60%, ≥70% or ≥80%**. The pick is the more probable winner. Actual draws count as failures. The thresholds are inclusive and cumulative.", "",
        "Cells show **observed accuracy (correct / selected games)**. The models can qualify different fixtures at a given threshold; filtered differences are descriptive. **Unavailable** means no matched V1 forecasts exist for that league; **— (0 picks)** means matched games exist, but none meets that model's threshold.", "",
        "V3 figures here can differ from the full V2/V3 league comparison because V1's statistics, history and fitted-venue requirements exclude some fixtures. The JSON also contains V3's full available-sample results as separately labeled context.", "",
        "Ireland has no cached 2026 evaluation fixtures in either division. Its combined results use 2025 only.", "",
    ]
    for period in ("combined", "2025", "2026"):
        lines += [f"## {'2025 + 2026' if period == 'combined' else period}: separate league comparisons", ""]
        for threshold in THRESHOLDS:
            lines += [f"### Winner probability ≥{threshold:.0%}", ""]
            rows = []
            for league in report["leagues"]:
                metrics = league["periods"][period]
                row = metrics["thresholds"][str(threshold)]
                count = metrics["common_matches"]
                change = row["v3_minus_v1_accuracy"]
                rows.append([COUNTRIES.get(league["country"], league["country"]),
                             "First" if league["division"] == "first" else "Second", league["league_id"],
                             f"{count}/{metrics['all_v3_matches']}",
                             result_cell(row["v1"]) if count else "Unavailable",
                             result_cell(row["v3"]) if count else "Unavailable",
                             f"{change * 100:+.1f} pp" if change is not None else "—"])
            lines += table(["Country", "Division", "League ID", "Common / all V3 games",
                            "V1 accuracy (hits/picks)", "V3 accuracy (hits/picks)", "V3 − V1"], rows)
            lines += [""]
    lines += ["## Coverage and limitations", "",
        "V1 needs both teams' cached match statistics, at least five prior games in the relevant venue, calibrated promoted-team venue Elo, and fitted host-home/visitor-away models from the training period. Team/venue formulas additionally require at least two eligible training matches and positive weighted goals. Missing statistics or history are coverage limitations, not zero accuracy.", "",
        "V1 has no cached match statistics in Iceland, or in the second divisions of Argentina, Chile, Colombia, Ireland and South Korea. The per-country metadata gives the statistics/history/model exclusions separately for each year and league.", "",
        "V1 uses the same fixed baseline settings as the earlier comparison: EWMA alpha 0.25, minimum venue history 5, regularization 0.01, recency decay 0.95 and Dixon–Coles enabled. No parameters are selected using the 2025/2026 results. V1 coefficients and rho are freshly fitted strictly before each evaluation year, using eligible prior results from all configured divisions in the country. V3 uses the already refreshed annual bundles fitted before the same test year; architecture selection used 2024 first-division validation.", "",
        "Historical features use preceding matches only. A three-hour result availability delay applies at training and data cutoffs. V1 updates its feature states immediately after each historical match, so the script verifies that no team has consecutive fixture kickoffs less than three hours apart. All source countries passed this check. V1 probabilities come from its exact Skellam/Dixon–Coles outcome function, with its own clipping and normalization.", "",
        "The JSON records coverage, average predicted win probability, draw failures, other-team-win failures and 95% Wilson intervals for each league/model/year/threshold. It also reports both models on the same subset of games qualifying for both. Wilson intervals assume independent games and are not corrected for comparisons across many leagues. Results with a handful of selections are uncertain; 100% from one match is weak evidence.", "",
        "The 2026 period and SQLite coverage are incomplete. These are historical tests of the retrained procedure, not an independent future test after retraining. SQLite databases are opened read-only; input/model hashes are checked before and after the reconstruction. Saved models are not overwritten.", "",
        "## Reproduction", "", "From the project directory:", "", "```powershell",
        "python winner_accuracy_comparison/compare_v1_v3.py", "```", "",
        "This refreshes V1 fits in memory and writes forecasts and this report under `winner_accuracy_comparison/v1_v3/`. If V3 has been retrained again, first refresh its annual predictions with `python winner_accuracy_comparison/compare_models.py`.", "",
        "To summarize these already refreshed V1 forecasts:", "", "```powershell",
        "python winner_accuracy_comparison/compare_v1_v3.py --mode report", "```", "",
        "- [evaluation.json](evaluation.json): separate league metrics, annual results, intervals and input hashes.",
        "- [forecast_metadata.json](forecast_metadata.json): fresh V1 training sizes, settings, rho and coverage by country/year.",
        "- [common_predictions_2025.csv](common_predictions_2025.csv) and [common_predictions_2026.csv](common_predictions_2026.csv): matched fixture probabilities.",
        "- [Full V2/V3 league comparison](../WINNER_THRESHOLDS_BY_LEAGUE.md): V3's larger available population.", "",
    ]
    return "\n".join(lines)


def generate_report(evaluation):
    info = json.loads((OUTPUT / "forecast_metadata.json").read_text(encoding="utf-8"))
    if info["data_cutoff"] != evaluation["training_cutoff"] or info["sources"] != evaluation["sources"]:
        raise ValueError("V1 predictions are stale. Reconstruct them first.")
    check_hashes(info["input_sha256"])
    common, full = {}, {}
    report_hashes = dict(info["input_sha256"])
    for year in YEARS:
        path = OUTPUT / f"common_predictions_{year}.csv"
        report_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        common[str(year)] = pd.read_csv(path)
        full[str(year)] = pd.read_csv(PARENT / f"all_predictions_{year}.csv")
        frame = common[str(year)]
        if len(frame) != info["periods"][str(year)]["common_matches"]:
            raise ValueError("Common fixture count does not match forecast metadata.")
        actual = np.where(frame.home_goals > frame.away_goals, 0, np.where(frame.home_goals == frame.away_goals, 1, 2))
        np.testing.assert_array_equal(actual, frame.target)
        for model in MODELS:
            validate_probabilities(frame[[f"p_{outcome}_{model}" for outcome in ("home", "draw", "away")]])
    common["combined"] = pd.concat(list(common.values()), ignore_index=True)
    full["combined"] = pd.concat(list(full.values()), ignore_index=True)
    if common["combined"].duplicated(["country", "fixture_id"]).any():
        raise ValueError("Duplicated matched fixture.")
    report = {"data_cutoff": info["data_cutoff"], "created_at": datetime.now(timezone.utc).isoformat(),
              "sources": info["sources"], "models": list(MODELS), "thresholds": list(THRESHOLDS),
              "common_matches": len(common["combined"]), "input_sha256": report_hashes, "leagues": []}
    for country, source in sorted(info["sources"].items()):
        for index, league_id in enumerate(source["league_ids"]):
            league = {"country": country, "league_id": int(league_id), "division": "first" if index == 0 else "second", "periods": {}}
            for period in common:
                frame = common[period]
                group = frame[(frame.country == country) & (frame.league_id == league_id)]
                all_frame = full[period]
                all_group = all_frame[(all_frame.country == country) & (all_frame.league_id == league_id)]
                metrics = summarize(group)
                metrics["all_v3_matches"] = len(all_group)
                metrics["v3_full_sample_context"] = {str(t): selected_metrics(all_group, "v3", t) for t in THRESHOLDS}
                league["periods"][period] = metrics
            league["v1_annual_coverage"] = {str(year): info["countries"][country]["periods"][str(year)]["leagues"][str(league_id)] for year in YEARS}
            report["leagues"].append(league)
    check_hashes(report_hashes)
    write_json(OUTPUT / "evaluation.json", report)
    (OUTPUT / "README.md").write_text(render(report), encoding="utf-8")
    for league in report["leagues"]:
        metrics = league["periods"]["combined"]
        print(f"{league['country']} {league['league_id']}: {metrics['common_matches']} common games | " + " | ".join(
            f"{t:.0%}: V1 {result_cell(metrics['thresholds'][str(t)]['v1'])}; V3 {result_cell(metrics['thresholds'][str(t)]['v3'])}"
            for t in THRESHOLDS), flush=True)
    print(f"Updated {OUTPUT / 'README.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "predictions", "report"), default="all")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    evaluation, hashes = current_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "predictions"):
        generate_predictions(evaluation, hashes, args.workers)
    if args.mode in ("all", "report"):
        generate_report(evaluation)
    check_hashes(hashes)


if __name__ == "__main__":
    main()
