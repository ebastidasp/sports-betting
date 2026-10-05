"""Audit only: compare the legacy model with new held-out predictions.

The new predictor does not import or depend on this script or the legacy model.
Both fits exclude results unavailable at 2025-01-01. Compare identical fixtures.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import football_poisson as old
from football_model_v2 import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("new_model"))
    args = parser.parse_args()
    predictions = pd.read_csv(args.folder / "test_predictions.csv")
    report = json.loads((args.folder / "evaluation.json").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(report["split"]["test_start"])
    end = pd.Timestamp(report["split"]["test_end"])
    output, comparisons = {}, []
    for country, source in report["sources"].items():
        old.activate_config({"country": country, "league_ids": source["league_ids"]})
        con = sqlite3.connect(Path(source["path"]).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            rows = old.load_matches(con)
        except SystemExit:
            output[country] = {"reason": "legacy model requires match statistics"}
            continue
        finally:
            con.close()
        rows["kickoff"] = pd.to_datetime(rows.kickoff, utc=True)
        rows = rows[(rows.status == "FT") & rows.league_id.isin(source["league_ids"])
                    & (rows.kickoff < end)]
        frame, _, _ = old.build_features(rows, .25, 5)
        frame = frame.merge(rows[["fixture_id", "league_id"]], on="fixture_id", validate="one_to_one")
        training = frame[frame.kickoff + pd.Timedelta(hours=3) <= cutoff]
        test = frame[(frame.kickoff >= cutoff) & (frame.kickoff < end)
                     & (frame.league_id == source["league_ids"][0])]
        if training.empty or test.empty:
            output[country] = {"reason": "insufficient legacy training/test coverage"}
            continue
        models, rho = old.fit_team_bundle(training, cutoff, .01, .95, True)
        test = old.supported_fixtures(models, test)
        pairs = predictions[predictions.country == country].merge(test[["fixture_id"]], on="fixture_id", validate="one_to_one")
        test = test.set_index("fixture_id").loc[pairs.fixture_id].reset_index()
        if pairs.empty:
            output[country] = {"reason": "no common predictions"}
            continue
        home, away = old.predict_team_rates(models, test)
        legacy = old.shared_probabilities(home, away, rho)
        new = pairs[["p_home", "p_draw", "p_away"]].to_numpy()
        output[country] = {"new": metrics(pairs, new), "previous": metrics(pairs, legacy),
                           "new_total_predictions": int((predictions.country == country).sum())}
        pairs[["previous_home", "previous_draw", "previous_away"]] = legacy
        comparisons.append(pairs)
        print(country, output[country], flush=True)
    (args.folder / "comparison_previous.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    if comparisons:
        common = pd.concat(comparisons, ignore_index=True)
        new_p = common[["p_home", "p_draw", "p_away"]].to_numpy()
        previous_p = common[["previous_home", "previous_draw", "previous_away"]].to_numpy()
        diff = (new_p.argmax(1) == common.target.to_numpy()).astype(float) - (previous_p.argmax(1) == common.target.to_numpy()).astype(float)
        week = pd.to_datetime(common.kickoff, utc=True).dt.strftime("%G-%V")
        blocks = pd.DataFrame({"week": week, "diff": diff}).groupby("week").agg(total=("diff", "sum"), n=("diff", "size"))
        rng = np.random.default_rng(42)
        samples = rng.integers(0, len(blocks), (2000, len(blocks)))
        deltas = blocks.total.to_numpy()[samples].sum(1) / blocks.n.to_numpy()[samples].sum(1)
        aggregate = {"new": metrics(common, new_p), "previous": metrics(common, previous_p),
                     "accuracy_difference_95pct_week_bootstrap": np.quantile(deltas, [.025, .975]).tolist()}
        common.to_csv(args.folder / "comparison_predictions.csv", index=False)
        (args.folder / "comparison_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print("Aggregate:", aggregate, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
