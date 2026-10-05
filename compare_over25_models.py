"""Chronological over-2.5 comparison: legacy Poisson vs v3's goal component.

python compare_over25_models.py
Only reads existing SQLite caches/model snapshots; writes a separate report.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from threadpoolctl import threadpool_limits

import football_model_v2 as base
import football_model_v3 as v3
import football_poisson as legacy

LEGACY_COLUMNS = ["country", "fixture_id", "kickoff", "home_goals", "away_goals",
                  "expected_home_legacy", "expected_away_legacy", "p_over25_legacy"]


def v3_over25(bundle, frame):
    total, mass = np.zeros(len(frame)), 0.
    for model, spec, weight in zip(bundle["models"], bundle["selection"]["components"], bundle["selection"]["weights"]):
        if spec["family"] == "goals" and weight > 0:
            home, away = v3.goal_rates(model, frame, bundle["base_columns"], spec)
            # Dixon-Coles affects only totals <=2 and preserves their summed mass.
            total += weight * poisson.sf(2, home + away)
            mass += weight
    if mass <= 0:
        raise ValueError("Selected v3 bundle has no goal-rate component.")
    return total / mass


def run_v3(args, metadata):
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    rows, _ = base.read_databases([source["path"] for source in metadata["sources"].values()])
    rows = rows[rows.kickoff + base.RESULT_DELAY <= cutoff]
    print(f"Reconstructing v3 goal inputs for {len(rows)} past fixtures...", flush=True)
    frame, _ = base.build_features(rows)
    for year in args.years:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = min(pd.Timestamp(f"{year+1}-01-01", tz="UTC"), cutoff)
        bundle = joblib.load(args.v3_folder / f"evaluation_model_{year}.joblib")
        if bundle.get("kind") != v3.KIND or pd.Timestamp(bundle["training_cutoff"]) > start:
            raise ValueError("Use a v3 evaluation bundle fitted before its evaluation year.")
        test = frame[(frame.kickoff >= start) & (frame.kickoff < end) & (frame.league_id == frame.primary_league)]
        export = test[["country", "fixture_id", "kickoff", "league_id", "home_name", "away_name", "home_goals", "away_goals"]].copy()
        export["p_over25_v3"] = v3_over25(bundle, test)
        goals = v3.expected_goals(bundle, test)
        export["expected_home_v3"], export["expected_away_v3"] = goals
        export.to_csv(args.output / f"v3_predictions_{year}.csv", index=False)
        print(f"V3 {year}: {len(export)} primary-league fixtures", flush=True)


def run_legacy(args, metadata):
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    countries = args.countries or sorted(metadata["sources"])
    for country in countries:
        source = metadata["sources"][country]
        legacy.activate_config({"country": country, "league_ids": source["league_ids"]})
        reason = None
        with closing(sqlite3.connect(Path(source["path"]).as_uri() + "?mode=ro", uri=True)) as con:
            try:
                rows = legacy.load_matches(con)
            except SystemExit:
                reason = "Legacy model requires both teams' cached match statistics."
        info = {"country": country, "settings": {"regularization": .01, "recency_decay": .95,
            "ewma_alpha": .25, "min_history": 5, "dixon_coles": True}, "periods": {}}
        if reason is None:
            rows["kickoff"] = pd.to_datetime(rows.kickoff, utc=True)
            rows = rows[(rows.status == "FT") & rows.league_id.isin(source["league_ids"])
                        & (rows.kickoff + base.RESULT_DELAY <= cutoff)]
            if rows.empty:
                reason = "No completed FT matches with legacy statistics."
        if reason is None:
            # Check that immediate legacy team updates cannot expose in-progress results.
            team_dates = pd.concat([rows[["home_id", "kickoff"]].rename(columns={"home_id": "team"}),
                                    rows[["away_id", "kickoff"]].rename(columns={"away_id": "team"})])
            gaps = team_dates.sort_values("kickoff").groupby("team").kickoff.diff()
            if ((gaps.notna()) & (gaps < base.RESULT_DELAY)).any():
                raise ValueError(f"{country}: overlapping team fixtures require a separate chronology audit.")
            frame, _, _ = legacy.build_features(rows, .25, 5)
            frame = frame.merge(rows[["fixture_id", "league_id"]], on="fixture_id", validate="one_to_one")
        for year in args.years:
            export = pd.DataFrame(columns=LEGACY_COLUMNS)
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = min(pd.Timestamp(f"{year+1}-01-01", tz="UTC"), cutoff)
            if reason is None:
                training = frame[frame.kickoff + base.RESULT_DELAY <= start]
                test = frame[(frame.kickoff >= start) & (frame.kickoff < end) & (frame.league_id == source["league_ids"][0])]
                if len(training) and len(test):
                    models, rho = legacy.fit_team_bundle(training, start, .01, .95, True)
                    test = legacy.supported_fixtures(models, test)
                    if len(test):
                        home, away = legacy.predict_team_rates(models, test)
                        export = test[["fixture_id", "kickoff", "home_goals", "away_goals"]].copy()
                        export["country"] = country
                        export["expected_home_legacy"], export["expected_away_legacy"] = home, away
                        export["p_over25_legacy"] = poisson.sf(2, home + away)
                    info["periods"][str(year)] = {"training_matches": len(training), "test_matches": len(test), "rho": rho}
                else:
                    info["periods"][str(year)] = {"test_matches": 0, "reason": "No eligible training or test fixtures."}
            else:
                info["periods"][str(year)] = {"test_matches": 0, "reason": reason}
            export[LEGACY_COLUMNS].to_csv(args.output / f"legacy_{country}_{year}.csv", index=False)
            print(f"Legacy {country} {year}: {len(export)} supported primary-league fixtures", flush=True)
        (args.output / f"legacy_{country}_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def event_metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected one valid over-2.5 probability per observed result.")
    q = np.clip(p, 1e-12, 1-1e-12)
    result = {"matches": len(y), "actual_over_rate": float(np.mean(y)), "mean_probability": float(np.mean(p)),
              "brier": float(np.mean((p-y)**2)),
              "log_loss": float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))), "filters": {}}
    for threshold in (.6, .7, .8):
        mask = p >= threshold
        n, successes = int(mask.sum()), int(y[mask].sum())
        rate = successes / n if n else None
        interval = None
        if n:
            denominator = 1 + 1.96**2/n
            center = (rate + 1.96**2/(2*n))/denominator
            radius = 1.96*np.sqrt(rate*(1-rate)/n+1.96**2/(4*n*n))/denominator
            interval = [float(center-radius), float(center+radius)]
        result["filters"][str(threshold)] = {"selected": n, "correct": successes, "incorrect": n-successes,
            "accuracy": rate, "coverage": n/len(y), "mean_probability": float(p[mask].mean()) if n else None,
            "95pct_wilson_interval": interval}
    return result


def combine(args, metadata):
    report = {"baseline": "Legacy football_poisson.py, default fixed configuration refitted before each year",
        "new": "V3 goal-rate component only; goal-component weights normalized to one",
        "event": "Over 2.5 goals means at least 3 total goals in FT matches",
        "training_cutoff": metadata["training_cutoff"], "periods": {},
        "notes": "Inclusive probability thresholds; models select different match sets. Full probability scores use identical fixtures. No model or probability settings selected from these test results."}
    lines = ["# Over 2.5 goals: legacy Poisson versus v3", "", report["event"], "",
        "The legacy model is football_poisson.py, refitted with its defaults on only prior-year data. V3 uses only its goal-rate component for this market; its 1X2 classifier is excluded. All comparisons below use identical supported fixtures.", ""]
    for year in args.years:
        new = pd.read_csv(args.output / f"v3_predictions_{year}.csv")
        old = pd.concat([pd.read_csv(args.output / f"legacy_{country}_{year}.csv") for country in sorted(metadata["sources"])], ignore_index=True)
        common = new.merge(old, on=["country", "fixture_id"], how="inner", suffixes=("", "_legacy"), validate="one_to_one")
        if common.empty:
            raise ValueError(f"No common evaluated fixtures in {year}.")
        np.testing.assert_array_equal(common.home_goals, common.home_goals_legacy)
        np.testing.assert_array_equal(common.away_goals, common.away_goals_legacy)
        np.testing.assert_array_equal(pd.to_datetime(common.kickoff, utc=True), pd.to_datetime(common.kickoff_legacy, utc=True))
        common["actual_over25"] = (common.home_goals + common.away_goals >= 3).astype(int)
        y = common.actual_over25.to_numpy()
        a, b = common.p_over25_legacy.to_numpy(dtype=float), common.p_over25_v3.to_numpy(dtype=float)
        period = {"matched_fixtures": len(common), "all_v3_fixtures": len(new), "legacy": event_metrics(y, a),
                  "v3_goal_component": event_metrics(y, b), "paired_differences": {}, "leagues": {},
                  "latest_test_kickoff": pd.to_datetime(common.kickoff, utc=True).max().isoformat()}
        weeks = pd.to_datetime(common.kickoff, utc=True).dt.strftime("%G-%V")
        rng = np.random.default_rng(42)
        for name, delta in {"brier": (b-y)**2-(a-y)**2,
            "log_loss": -y*np.log(np.clip(b,1e-12,1))-(1-y)*np.log(np.clip(1-b,1e-12,1))
                        +y*np.log(np.clip(a,1e-12,1))+(1-y)*np.log(np.clip(1-a,1e-12,1))}.items():
            blocks = pd.DataFrame({"week": weeks, "delta": delta}).groupby("week").agg(total=("delta","sum"), n=("delta","size"))
            ix = rng.integers(0,len(blocks),(5000,len(blocks)))
            boot = blocks.total.to_numpy()[ix].sum(1)/blocks.n.to_numpy()[ix].sum(1)
            period["paired_differences"][name] = {"v3_minus_legacy": float(delta.mean()), "95pct_week_bootstrap": np.quantile(boot,[.025,.975]).tolist()}
        for country, group in common.groupby("country"):
            truth = group.actual_over25.to_numpy()
            period["leagues"][country] = {"legacy": event_metrics(truth,group.p_over25_legacy.to_numpy()),
                "v3_goal_component": event_metrics(truth,group.p_over25_v3.to_numpy())}
        common.to_csv(args.output / f"comparison_predictions_{year}.csv", index=False)
        report["periods"][str(year)] = period
        lines += [f"## {year}: {len(common):,} common fixtures", "", "| Minimum probability | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate |",
                  "|---|---:|---:|---:|---:|"]
        for threshold in (.6,.7,.8):
            left,right = period["legacy"]["filters"][str(threshold)],period["v3_goal_component"]["filters"][str(threshold)]
            def formatted(row):
                return f"{row['accuracy']:.2%}" if row["accuracy"] is not None else "—"
            lines.append(f"| {threshold:.0%} | {left['correct']}/{left['selected']} | {formatted(left)} | {right['correct']}/{right['selected']} | {formatted(right)} |")
        interval = period['paired_differences']['brier']['95pct_week_bootstrap']
        interpretation = "This interval favors v3." if interval[1] < 0 else "This interval favors legacy." if interval[0] > 0 else "This interval includes zero."
        lines += ["", f"Full-sample Brier: legacy **{period['legacy']['brier']:.5f}**, v3 **{period['v3_goal_component']['brier']:.5f}**. Lower is better.",
            f"Full-sample log loss: legacy **{period['legacy']['log_loss']:.5f}**, v3 **{period['v3_goal_component']['log_loss']:.5f}**.", "",
            f"The 95% weekly paired bootstrap interval for v3-minus-legacy Brier is {interval}. Negative differences favor v3. {interpretation}", ""]
        print(f"{year}: {len(common)} common fixtures; Brier legacy={period['legacy']['brier']:.5f}, v3={period['v3_goal_component']['brier']:.5f}", flush=True)
    lines += ["The thresholds are cumulative. Hit-rate differences reflect different selected fixtures; an overlap of selections predicts the same over-2.5 event. Read selected counts and uncertainty alongside hit rates.", "",
        f"2026 is incomplete. Only matches stored by {metadata['training_cutoff']} enter the analysis. Iceland has no legacy match-statistics coverage, and Ireland has no eligible 2026 sample.", "",
        "Full league results, average forecast probabilities and confidence intervals are in evaluation.json. Original scripts, trained models and SQLite caches were opened for reading only."]
    (args.output / "evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (args.output / "README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all","legacy","v3","combine"), default="all")
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--years", nargs="+", type=int, default=[2025,2026])
    parser.add_argument("--v3-folder", type=Path, default=Path("new_model_v3"))
    parser.add_argument("--output", type=Path, default=Path("over25_comparison_v2_v3"))
    args=parser.parse_args()
    protected_folders=[Path(p).resolve() for p in ("new_model","new_model_v3","corners_model","corners_model_v2")]
    if any(args.output.resolve()==p or p in args.output.resolve().parents for p in protected_folders):
        parser.error("Use a separate comparison output folder.")
    protected=[Path("football_poisson.py"),Path("football_model_v2.py"),Path("football_model_v3.py")]
    protected+=list(Path('.').glob('*_poisson.joblib'))+list(args.v3_folder.glob('*.joblib'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    metadata=json.loads((args.v3_folder/"evaluation.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        if args.mode in ("all","legacy"):
            run_legacy(args,metadata)
        if args.mode in ("all","v3"):
            run_v3(args,metadata)
        if args.mode in ("all","combine"):
            combine(args,metadata)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError("An existing model or script changed during comparison.")
    print("Existing model files unchanged.",flush=True)


if __name__=="__main__":
    main()
