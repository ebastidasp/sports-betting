"""Extend saved annual goal backtests to all configured divisions, read-only.

python goals_accuracy_comparison/extend_backtest.py --mode legacy --countries england germany
python goals_accuracy_comparison/extend_backtest.py --mode v3
python goals_accuracy_comparison/extend_backtest.py --mode combine
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import football_model_v2 as base
import football_model_v3 as v3
import football_poisson as legacy

OUTPUT = Path(__file__).resolve().parent / "backtest"
PREVIOUS = ROOT / "over25_comparison_v2_v3"
YEARS = (2025, 2026)
LEGACY_COLUMNS = ["country", "fixture_id", "kickoff", "home_goals", "away_goals",
                  "expected_home_legacy", "expected_away_legacy", "p_over25_legacy", "league_id"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_legacy(countries, metadata):
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    for country in countries:
        source = metadata["sources"][country]
        primary = source["league_ids"][0]
        old_info = json.loads((PREVIOUS / f"legacy_{country}_metadata.json").read_text(encoding="utf-8"))
        info = {"country": country, "league_ids": source["league_ids"],
                "settings": old_info["settings"], "periods": {}}
        legacy.activate_config({"country": country, "league_ids": source["league_ids"]})
        rows, frame, reason = None, None, None
        with closing(sqlite3.connect(Path(source["path"]).as_uri() + "?mode=ro", uri=True)) as con:
            try:
                rows = legacy.load_matches(con)
            except SystemExit:
                reason = "Legacy requires both teams' cached match statistics."
        if rows is not None:
            rows["kickoff"] = pd.to_datetime(rows.kickoff, utc=True)
            rows = rows[(rows.status == "FT") & rows.league_id.isin(source["league_ids"])
                        & (rows.kickoff + base.RESULT_DELAY <= cutoff)]
            if rows.empty:
                reason = "No completed FT matches with legacy statistics."
            elif (rows.league_id != primary).any():
                dates = pd.concat([rows[["home_id", "kickoff"]].rename(columns={"home_id": "team"}),
                                   rows[["away_id", "kickoff"]].rename(columns={"away_id": "team"})])
                gaps = dates.sort_values("kickoff").groupby("team").kickoff.diff()
                if (gaps.notna() & (gaps < base.RESULT_DELAY)).any():
                    raise ValueError(f"{country}: overlapping fixtures require a chronology audit.")
                frame, _, _ = legacy.build_features(rows, .25, 5)
                frame = frame.merge(rows[["fixture_id", "league_id"]], on="fixture_id", validate="one_to_one")
        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = min(pd.Timestamp(f"{year+1}-01-01", tz="UTC"), cutoff)
            old = pd.read_csv(PREVIOUS / f"legacy_{country}_{year}.csv")
            old["league_id"] = primary
            export = old.copy()
            period = dict(old_info["periods"][str(year)])
            period["primary_forecasts_reproduced"] = True
            secondary_test = pd.DataFrame()
            if frame is not None:
                training = frame[frame.kickoff + base.RESULT_DELAY <= start]
                test = frame[(frame.kickoff >= start) & (frame.kickoff < end)]
                secondary_test = test[test.league_id != primary]
                if len(training) and len(secondary_test):
                    print(f"Fitting {country} before {year}: {len(training)} training matches; {len(secondary_test)} secondary candidates", flush=True)
                    models, rho = legacy.fit_team_bundle(training, start, .01, .95, True)
                    test = legacy.supported_fixtures(models, test)
                    home, away = legacy.predict_team_rates(models, test)
                    export = test[["fixture_id", "league_id", "kickoff", "home_goals", "away_goals"]].copy()
                    export["country"] = country
                    export["expected_home_legacy"], export["expected_away_legacy"] = home, away
                    export["p_over25_legacy"] = poisson.sf(2, home + away)
                    repeated = export[export.league_id == primary].sort_values("fixture_id")
                    expected = old.sort_values("fixture_id")
                    np.testing.assert_array_equal(repeated.fixture_id, expected.fixture_id)
                    np.testing.assert_allclose(repeated[["expected_home_legacy", "expected_away_legacy", "p_over25_legacy"]].to_numpy(dtype=float),
                                               expected[["expected_home_legacy", "expected_away_legacy", "p_over25_legacy"]].to_numpy(dtype=float), atol=1e-10, rtol=0)
                    np.testing.assert_allclose(rho, old_info["periods"][str(year)]["rho"], atol=1e-10, rtol=0)
                    period.update({"training_matches": len(training), "rho": rho,
                                   "source": "Annual legacy fit reproduced on prior results, then evaluated on all divisions"})
            if (export.league_id != primary).any():
                secondary_reason = None
            elif reason:
                secondary_reason = reason
            elif len(source["league_ids"]) == 1:
                secondary_reason = "No second division configured."
            elif rows is not None and not ((rows.league_id != primary) & (rows.kickoff >= start) & (rows.kickoff < end)).any():
                secondary_reason = "No completed second-division test fixtures with both teams' cached statistics."
            elif secondary_test.empty:
                secondary_reason = "No second-division test fixtures with eligible prior history."
            else:
                secondary_reason = "No second-division fixtures supported by the prior-year venue models."
            period.update({"test_matches": len(export), "first_division_matches": int((export.league_id == primary).sum()),
                           "second_division_matches": int((export.league_id != primary).sum()),
                           "second_division_exclusion_reason": secondary_reason})
            info["periods"][str(year)] = period
            export[LEGACY_COLUMNS].to_csv(OUTPUT / f"legacy_{country}_{year}.csv", index=False)
            print(f"Legacy {country} {year}: {period['first_division_matches']} first + {period['second_division_matches']} second", flush=True)
        write_json(OUTPUT / f"legacy_{country}_metadata.json", info)


def run_v3(metadata):
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    rows, _ = base.read_databases([source["path"] for source in metadata["sources"].values()])
    rows = rows[rows.kickoff + base.RESULT_DELAY <= cutoff]
    print(f"Reconstructing V3 inputs for {len(rows)} chronological fixtures", flush=True)
    frame, _ = base.build_features(rows)
    for year in YEARS:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year+1}-01-01", tz="UTC"), cutoff)
        bundle = joblib.load(ROOT / "new_model_v3" / f"evaluation_model_{year}.joblib")
        if bundle["kind"] != v3.KIND or pd.Timestamp(bundle["training_cutoff"]) > start:
            raise ValueError("Use V3 evaluation bundles trained before their test years.")
        test = frame[(frame.kickoff >= start) & (frame.kickoff < end)]
        goals = [(spec, weight) for spec, weight in zip(bundle["selection"]["components"], bundle["selection"]["weights"])
                 if spec["family"] == "goals" and weight > 0]
        if len(goals) != 1:
            raise ValueError("This stored-rate report requires one goal-rate component.")
        export = test[["country", "fixture_id", "kickoff", "league_id", "primary_league", "home_name", "away_name", "home_goals", "away_goals"]].copy()
        home, away = v3.expected_goals(bundle, test)
        export["expected_home_v3"], export["expected_away_v3"] = home, away
        export["p_over25_v3"] = v3.total_goal_probabilities(bundle, test)[:, 1]
        old = pd.read_csv(PREVIOUS / f"v3_predictions_{year}.csv").sort_values(["country", "fixture_id"])
        primary = export[export.league_id == export.primary_league].sort_values(["country", "fixture_id"])
        np.testing.assert_array_equal(primary[["country", "fixture_id"]].to_numpy(), old[["country", "fixture_id"]].to_numpy())
        np.testing.assert_allclose(primary[["expected_home_v3", "expected_away_v3", "p_over25_v3"]].to_numpy(dtype=float),
                                   old[["expected_home_v3", "expected_away_v3", "p_over25_v3"]].to_numpy(dtype=float), atol=1e-10, rtol=0)
        export.to_csv(OUTPUT / f"v3_predictions_{year}.csv", index=False)
        print(f"V3 {year}: {len(primary)} first + {len(export)-len(primary)} second; prior first-division forecasts reproduced", flush=True)


def combine(metadata):
    info = {"training_cutoff": metadata["training_cutoff"], "scope": "all_configured_divisions",
            "sources": metadata["sources"], "periods": {}}
    for year in YEARS:
        new = pd.read_csv(OUTPUT / f"v3_predictions_{year}.csv")
        old = pd.concat([pd.read_csv(OUTPUT / f"legacy_{country}_{year}.csv") for country in metadata["sources"]], ignore_index=True)
        common = new.merge(old, on=["country", "fixture_id"], validate="one_to_one", suffixes=("", "_legacy"))
        for column in ("home_goals", "away_goals", "league_id"):
            np.testing.assert_array_equal(common[column], common[f"{column}_legacy"])
        np.testing.assert_array_equal(pd.to_datetime(common.kickoff, utc=True), pd.to_datetime(common.kickoff_legacy, utc=True))
        common["division"] = np.where(common.league_id == common.primary_league, "first", "second")
        common["actual_over25"] = (common.home_goals + common.away_goals >= 3).astype(int)
        original = pd.read_csv(PREVIOUS / f"comparison_predictions_{year}.csv").sort_values(["country", "fixture_id"])
        primary = common[common.division == "first"].sort_values(["country", "fixture_id"])
        np.testing.assert_array_equal(primary[["country", "fixture_id"]].to_numpy(), original[["country", "fixture_id"]].to_numpy())
        for column in ("expected_home_legacy", "expected_away_legacy", "p_over25_legacy", "expected_home_v3", "expected_away_v3", "p_over25_v3"):
            np.testing.assert_allclose(primary[column].to_numpy(dtype=float), original[column].to_numpy(dtype=float), atol=1e-10, rtol=0)
        common.to_csv(OUTPUT / f"comparison_predictions_{year}.csv", index=False)
        info["periods"][str(year)] = {"matched_fixtures": len(common), "all_v3_fixtures": len(new),
            "common_divisions": common.division.value_counts().to_dict(),
            "latest_test_kickoff": pd.to_datetime(common.kickoff, utc=True).max().isoformat()}
        info["periods"][str(year)]["available_v3_divisions"] = {
            "first": int((new.league_id == new.primary_league).sum()),
            "second": int((new.league_id != new.primary_league).sum())}
        print(f"Common {year}: {len(primary)} first + {len(common)-len(primary)} second", flush=True)
    write_json(OUTPUT / "evaluation.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "v3", "combine", "all"), default="all")
    parser.add_argument("--countries", nargs="+")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "new_model_v3" / "evaluation.json").read_text(encoding="utf-8"))
    countries = args.countries or sorted(metadata["sources"])
    if set(countries) - set(metadata["sources"]):
        parser.error("Unknown country in the requested legacy subset.")
    protected = [ROOT / name for name in ("football_poisson.py", "football_model_v2.py", "football_model_v3.py")]
    protected += list(ROOT.glob("*_poisson.joblib")) + list((ROOT / "new_model_v3").glob("*.joblib"))
    protected += list(PREVIOUS.glob("*"))
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected if path.is_file()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        if args.mode in ("legacy", "all"):
            run_legacy(countries, metadata)
        if args.mode in ("v3", "all"):
            run_v3(metadata)
        if args.mode in ("combine", "all"):
            combine(metadata)
    for path, digest in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"An existing model or original report changed: {path}")
    print("Protected models and original report unchanged.", flush=True)


if __name__ == "__main__":
    main()
