"""Reproduce full-precision V3 forecasts and compare the user's screenshot odds."""
from datetime import timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
sys.path.insert(0, str(ROOT))
import football_model_v2 as base
import football_model_v3 as v3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]]


def main():
    quotes = json.loads((OUTPUT / "quote_snapshot.json").read_text(encoding="utf-8"))
    cli = json.loads((OUTPUT / "cli_outputs.json").read_text(encoding="utf-8"))
    model_path = ROOT / "new_model_v3" / "football_model.joblib"
    bundle = joblib.load(model_path)
    country, league = quotes["country"], quotes["league_id"]
    as_of = pd.Timestamp(quotes["prediction_as_of"])
    if bundle["kind"] != v3.KIND or as_of < pd.Timestamp(bundle["training_cutoff"]):
        raise ValueError("Use a current V3 model fitted before prediction time.")
    db_path = Path(bundle["sources"][country]["path"])
    protected = [model_path, db_path, ROOT / "football_model_v3.py", ROOT / "football_model_v2.py"]
    wal = Path(str(db_path) + "-wal")
    if wal.exists():
        protected.append(wal)
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in protected}
    rows, _ = base.read_databases([db_path])
    with threadpool_limits(limits=2):
        _, state = v3.build_features(rows, as_of)
        inputs, resolved = [], []
        for fixture in quotes["fixtures"]:
            hid, home = base.resolve_team(fixture["home"], state, country, league)
            aid, away = base.resolve_team(fixture["away"], state, country, league)
            inputs.append(state.inputs(country, league, bundle["sources"][country]["league_ids"][0], hid, aid, as_of))
            resolved.append((hid, aid, home, away))
        frame = pd.DataFrame(inputs)
        probabilities = v3.predict_bundle(bundle, frame)
        rates = v3.expected_goals(bundle, frame)
        goal_totals = v3.total_goal_probabilities(bundle, frame)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Invalid V3 probabilities.")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as con:
        schedule = pd.read_sql_query("SELECT * FROM fixtures WHERE league_id=40 AND kickoff>='2026-10-09' AND kickoff<'2026-10-12'", con)
    records, markets, commands = [], [], []
    for index, (fixture, teams) in enumerate(zip(quotes["fixtures"], resolved)):
        hid, aid, home, away = teams
        game = schedule[(schedule.home_id == hid) & (schedule.away_id == aid)]
        if len(game) != 1 or game.iloc[0].status != "NS":
            raise ValueError("Screenshot fixture must match one cached upcoming match.")
        game = game.iloc[0]
        stdout = cli[index]["stdout"]
        if cli[index]["exit_code"] or (cli[index]["home"], cli[index]["away"]) != (fixture["home"], fixture["away"]):
            raise ValueError("A prediction command failed or fixture order changed.")
        if f"Model training cutoff: {bundle['training_cutoff']}" not in stdout or as_of.isoformat() not in stdout:
            raise ValueError("CLI predictions came from a different model or prediction time.")
        record = {"fixture_id": int(game.fixture_id), "home": home["name"], "away": away["name"],
                  "kickoff": game.kickoff, "home_id": hid, "away_id": aid,
                  "home_last_cached_match": home["last"].isoformat(), "away_last_cached_match": away["last"].isoformat(),
                  "outcomes": {}, "goal_totals": {}}
        for outcome, label, code, p in zip(("home", "draw", "away"), base.LABELS, ("h", "d", "a"), probabilities[index]):
            printed = re.search(rf"^{re.escape(label)}: ([0-9.]+)% \(fair odds ([0-9.]+)\)", stdout, re.MULTILINE)
            if not printed:
                raise ValueError("Missing outcome in CLI output.")
            np.testing.assert_allclose(float(printed.group(1)) / 100, p, atol=5.01e-5, rtol=0)
            odds = float(fixture[code])
            fair = float(1 / p)
            np.testing.assert_allclose(float(printed.group(2)), fair, atol=.00501, rtol=0)
            values = {"probability": float(p), "fair_odds": fair, "offered_odds": odds,
                      "break_even_probability": 1 / odds, "probability_edge": float(p - 1 / odds),
                      "expected_return_per_unit": float(p * odds - 1)}
            record["outcomes"][outcome] = values
            pick = record["home"] if outcome == "home" else record["away"] if outcome == "away" else "Draw"
            markets.append({"fixture_id": record["fixture_id"], "match": f"{record['home']} vs {record['away']}",
                            "outcome": outcome, "pick": pick, **values})
        if rates is not None:
            record["expected_home_goals"] = float(rates[0][index])
            record["expected_away_goals"] = float(rates[1][index])
        if goal_totals is not None:
            for minimum, probability in zip((2, 3, 4), goal_totals[index]):
                record["goal_totals"][str(minimum)] = {"probability": float(probability), "fair_odds": float(1 / probability)}
        record["best_value_outcome"] = max(record["outcomes"], key=lambda outcome: record["outcomes"][outcome]["expected_return_per_unit"])
        records.append(record)
        commands.append(f"python football_model_v3.py predict --country england --league-id 40 --home \"{fixture['home']}\" --away \"{fixture['away']}\" --as-of \"{as_of.isoformat()}\"")
    markets.sort(key=lambda market: market["expected_return_per_unit"], reverse=True)
    report = {"model": model_path.relative_to(ROOT).as_posix(), "model_training_cutoff": bundle["training_cutoff"],
              "prediction_as_of": as_of.isoformat(), "league_id": league, "country": country,
              "latest_cached_result": rows.kickoff.max().isoformat(), "input_sha256": hashes,
              "odds_source": quotes["odds_source"], "fixtures": records, "ranked_1x2_outcomes": markets,
              "expected_value_formula": "model_probability * offered_decimal_odds - 1"}
    local_asof = as_of.to_pydatetime().astimezone(timezone(timedelta(hours=-5)))
    lines = ["# V3 predictions and screenshot odds: Championship, 9–11 October 2026", "",
        f"All 12 original CLI prediction commands completed successfully. Forecast snapshot: **{local_asof:%d %B %Y, %H:%M:%S} Colombia time**. Current saved V3 training cutoff: **{bundle['training_cutoff']}**. Full-precision batch predictions reproduce each CLI probability within its display rounding.", "",
        "The offered decimal odds are transcribed from the user's screenshot; they are not a live bookmaker feed. All pairings and kickoff times match league 40's cached upcoming fixtures. The EFL independently confirms the televised fixtures in its [official schedule](https://www.efl.com/news/2026/july/30/see-the-latest-sky-sports-tv-selections/).", "",
        "**Fair odds = 1 / V3 probability. Estimated expected return per unit = V3 probability × offered odds − 1.** A price above the model's fair odds has positive estimated value. This is a return implied by the model's probability, not a measured betting profit or a confidence interval. Winning probability and value are different quantities.", "",
        "## Best-priced outcome in each game", "",
        "The table ranks each game's highest estimated-return outcome. A positive value does not guarantee a winning bet. The only winner with at least 60% estimated probability is Southampton, whose offered price has negative estimated value.", ""]
    best = []
    for record in records:
        outcome = record["best_value_outcome"]
        market = record["outcomes"][outcome]
        pick = record["home"] if outcome == "home" else record["away"] if outcome == "away" else "Draw"
        best.append((record, pick, market))
    best.sort(key=lambda value: value[2]["expected_return_per_unit"], reverse=True)
    lines += table(["Match", "Pick", "V3 probability", "V3 fair odds", "Screenshot odds", "Estimated return"],
                   [[f"{record['home']} vs {record['away']}", pick, f"{m['probability']:.2%}", f"{m['fair_odds']:.2f}",
                     f"{m['offered_odds']:.2f}", f"{m['expected_return_per_unit']:+.2%}"] for record, pick, m in best])
    lines += ["", "## All match-result probabilities and fair odds", "",
        "Each cell shows model probability / fair decimal odds. These home/draw/away forecasts use V3's full selected blend and calibration.", ""]
    lines += table(["Match", "Kickoff (Colombia)", "Home", "Draw", "Away"],
                   [[f"{r['home']} vs {r['away']}", pd.Timestamp(r["kickoff"]).tz_convert("America/Bogota").strftime("%d Oct %H:%M"),
                     *[f"{r['outcomes'][o]['probability']:.2%} / {r['outcomes'][o]['fair_odds']:.2f}" for o in ("home", "draw", "away")]] for r in records])
    lines += ["", "## Goal-total probabilities and fair odds", "",
        "These use V3's goal-rate component. No goal-total bookmaker odds were supplied, so value cannot be assessed for these markets.", ""]
    lines += table(["Match", "xG home–away", "Over 1.5", "Over 2.5", "Over 3.5"],
                   [[f"{r['home']} vs {r['away']}", f"{r['expected_home_goals']:.3f}–{r['expected_away_goals']:.3f}",
                     *[f"{r['goal_totals'][n]['probability']:.2%} / {r['goal_totals'][n]['fair_odds']:.2f}" for n in ("2", "3", "4")]] for r in records])
    lines += ["", "## Scope of the estimate", "",
        "Forecasts use the current model and SQLite results available at the fixed prediction time, rather than hypothetical updated data on kickoff day. The latest completed English league fixture in this cache is **19 September 2026**. Upcoming match lineups, new injuries and bookmaker changes are not inputs to this calculation. Re-run closer to kickoff after updating the cache to reassess the prices.", "",
        "Middlesbrough has the largest model-implied value at the supplied 1X2 prices. Watford, Lincoln and Millwall also have sizable estimated margins, but their modeled win probabilities are all below 50%. QPR is a longer shot. Margins of just a few percent can disappear with modest probability error.", "",
        "## Files and commands", "",
        "- [predictions.json](predictions.json): full-precision probabilities, fair prices, all 36 1X2 outcomes ranked by expected return, goal totals and input hashes.",
        "- [quote_snapshot.json](quote_snapshot.json): screenshot prices and prediction time.",
        "- [cli_outputs.json](cli_outputs.json): output and exit status of all 12 executed commands.",
        "- [commands.ps1](commands.ps1): those prediction commands, runnable from the project directory.", "",
        "```powershell", *commands, "```", "",
        "This analysis only runs local forecasts; it does not submit any bets.", ""]
    for relative, expected in hashes.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Model or data changed during analysis: {relative}")
    (OUTPUT / "predictions.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    for record, pick, m in best:
        print(f"{record['home']} vs {record['away']}: {pick}; probability {m['probability']:.4%}; fair {m['fair_odds']:.6f}; offered {m['offered_odds']:.2f}; EV {m['expected_return_per_unit']:+.4%}")
    print(f"Saved {OUTPUT / 'README.md'}")


if __name__ == "__main__":
    main()
