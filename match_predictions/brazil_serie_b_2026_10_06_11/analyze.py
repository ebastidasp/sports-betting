"""Run all quoted Brazil Serie B forecasts and calculate screenshot-price EV."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
sys.path.insert(0, str(ROOT))
import football_model_v3 as model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")


def single(probability, odds, stake):
    ev = probability * odds - 1
    return {"probability": float(probability), "fair_odds": float(1 / probability),
            "offered_odds": odds, "stake_cop": stake, "ev": float(ev),
            "expected_profit_cop": float(stake * ev),
            "net_profit_if_correct_cop": float(stake * (odds - 1)),
            "net_profit_if_incorrect_cop": -stake, "positive_ev": bool(ev > 0)}


def split(p_win, p_draw, win_odds, draw_odds, stake):
    win_stake = int(round(stake * draw_odds / (win_odds + draw_odds)))
    draw_stake = stake - win_stake
    win_payout, draw_payout = win_stake * win_odds, draw_stake * draw_odds
    expected = p_win * win_payout + p_draw * draw_payout - stake
    probability = p_win + p_draw
    inverse_sum = 1 / win_odds + 1 / draw_odds
    return {"probability": float(probability), "team_probability": float(p_win),
            "draw_probability": float(p_draw), "fair_odds": float(1 / probability),
            "effective_odds": float(1 / inverse_sum), "team_odds": win_odds,
            "draw_odds": draw_odds, "team_stake_cop": win_stake,
            "draw_stake_cop": draw_stake, "stake_cop": stake,
            "net_profit_on_win_cop": win_payout - stake,
            "net_profit_on_draw_cop": draw_payout - stake,
            "minimum_covered_profit_cop": min(win_payout, draw_payout) - stake,
            "net_profit_if_other_team_wins_cop": -stake,
            "ev": float(expected / stake), "ev_before_rounding": float(probability / inverse_sum - 1),
            "expected_profit_cop": float(expected), "positive_ev": bool(expected > 0)}


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows]]


def render(report):
    stake = report["stake_cop_per_bet"]
    lines = ["# Brazil Série B: screenshot odds and V3 expected value", "",
        f"All **{len(report['fixtures'])} screenshot fixtures** were predicted with the saved V3 model. Each listed plan uses **COP {stake:,} per bet**; a win-or-draw plan splits that total between two separate 1X2 bets.", "",
        f"Prediction snapshot: **{report['prediction_as_of_bogota']}**. Model data cutoff: **{report['training_cutoff_bogota']}**. The latest completed Série B fixture in SQLite is **{report['latest_completed_bogota']}**. These forecasts use cached historical results, with no bookmaker odds as model inputs.", "",
        f"The retrained V3 now uses **{report['model_selection']['weights']}** as classifier/goal blend weights, with **{report['model_selection']['calibration_method']}** outcome calibration. This is the current saved model, rather than the earlier pre-Brazil training snapshot. V4 is not trained for Brazil and is not used here.", "",
        "## Filters and stake calculation", "",
        "A direct win qualifies only when its model probability is at least 60% and its quoted price has positive EV. A team win-or-draw plan qualifies only when the sum of those mutually exclusive outcome probabilities is at least 60% and the entire rounded stake split has positive EV. Goal bets require positive EV; the user did not specify a 60% goal threshold, so that flag is reported separately.", "",
        "Single-bet EV is `probability × decimal odds − 1`. Expected profit is `COP25,000 × EV`. Fair odds are `1 / probability`. Expected profit averages winning and losing outcomes; it is not the payout for a successful bet.", "",
        "For win odds W and draw odds D, team stake is rounded from `25,000 × D / (W + D)`; draw stake is the remainder. This creates approximately equal gross payouts. Effective odds are `1 / (1/W + 1/D)`. Rounded EV is recomputed as `(Pwin × win_stake × W + Pdraw × draw_stake × D − 25,000) / 25,000`. If the other team wins, the complete COP25,000 is lost.", "",
        "## Direct team wins: probability at least 60% and positive EV", ""]
    wins = report["qualifying_wins"]
    if wins:
        lines += table(["Date (Colombia)", "Match", "Team", "Probability", "Fair odds", "Offered", "EV", "Expected profit COP"],
            [[x["kickoff_bogota"], x["match"], x["team"], f"{x['probability']:.2%}", f"{x['fair_odds']:.3f}", f"{x['offered_odds']:.2f}", f"{x['ev']:+.2%}", f"{x['expected_profit_cop']:,.0f}"] for x in wins])
    else:
        lines += ["**No direct win meets both conditions.**"]
    lines += ["", "## Win or draw: combined probability at least 60% and positive EV", ""]
    pairs = report["qualifying_win_or_draw"]
    if pairs:
        lines += table(["Date (Colombia)", "Match", "Team or draw", "Probability", "Team stake @ odds", "Draw stake @ odds", "Effective odds", "EV", "Minimum covered net profit COP", "Expected profit COP"],
            [[x["kickoff_bogota"], x["match"], x["team"], f"{x['probability']:.2%}", f"{x['team_stake_cop']:,} @ {x['team_odds']:.2f}", f"{x['draw_stake_cop']:,} @ {x['draw_odds']:.2f}", f"{x['effective_odds']:.3f}", f"{x['ev']:+.2%}", f"{x['minimum_covered_profit_cop']:,.0f}", f"{x['expected_profit_cop']:,.0f}"] for x in pairs])
    else:
        lines += ["**No equal-payout win-or-draw plan meets both conditions.**"]
    lines += ["", "## Positive-EV over/under 2.5 goals", "",
              "Over 2.5 means at least 3 total goals. Under 2.5 means at most 2. Each row stakes COP25,000; these are standalone bets, not combinations with the match-outcome bets.", ""]
    goals = report["qualifying_goals"]
    if goals:
        lines += table(["Date (Colombia)", "Match", "Market", "Probability", "At least 60%?", "Fair odds", "Offered", "EV", "Net profit if correct COP", "Expected profit COP"],
            [[x["kickoff_bogota"], x["match"], x["market"], f"{x['probability']:.2%}", "Yes" if x["probability"] >= .6 else "No", f"{x['fair_odds']:.3f}", f"{x['offered_odds']:.2f}", f"{x['ev']:+.2%}", f"{x['net_profit_if_correct_cop']:,.0f}", f"{x['expected_profit_cop']:,.0f}"] for x in goals])
    else:
        lines += ["No goal selection has positive estimated EV."]
    lines += ["", "## Highest model EV per fixture", "",
              "This optional shortlist chooses one qualifying plan per fixture, ranked by model EV. All qualifying plans remain listed above; selecting multiple plans on a fixture uses COP25,000 for each and their outcomes can be correlated.", ""]
    lines += table(["Match", "Plan", "Probability", "EV", "Expected profit COP"],
        [[x["match"], x["selection"], f"{x['probability']:.2%}", f"{x['ev']:+.2%}", f"{x['expected_profit_cop']:,.0f}"] for x in report["highest_ev_per_fixture"]])
    lines += ["", f"The one-plan-per-fixture shortlist has **{len(report['highest_ev_per_fixture'])} plans**, totaling **COP {stake * len(report['highest_ev_per_fixture']):,}**. The sum of model-implied expected profits is **COP {sum(x['expected_profit_cop'] for x in report['highest_ev_per_fixture']):,.0f}**. This does not estimate the probability that the portfolio finishes profitable.", "",
              "## Complete forecasts and price checks", ""]
    lines += table(["Match", "P(home)", "P(draw)", "P(away)", "xG home-away", "Home win EV", "Away win EV", "Home/draw EV", "Away/draw EV", "Over 2.5 EV", "Under 2.5 EV"],
        [[r["match"], *[f"{r['outcomes'][s]['probability']:.2%}" for s in ("home", "draw", "away")], f"{r['xg_home']:.3f}-{r['xg_away']:.3f}", *[f"{x['ev']:+.2%}" for x in (r["outcomes"]["home"], r["outcomes"]["away"], r["win_or_draw"]["home"], r["win_or_draw"]["away"], r["goals"]["over"], r["goals"]["under"])]] for r in report["fixtures"]])
    lines += ["", "## Scope and reproduction", "",
              "These are positive-EV estimates conditional on V3 probabilities. Historical accuracy is not demonstrated betting profitability; small margins can disappear with prediction error or price changes. Listed prices are the screenshot snapshot, without a live-price confirmation or additional charges.", "",
              "Sunday's teams play earlier in the week. Their 11 October probabilities here are forecasts made on 5 October, before those intervening results are known; rerun after the midweek matches before using Sunday selections.", "",
              f"The screenshot dates and matchups agree with the cached schedule and [the published rounds 31–34 schedule]({report['schedule_source']}); source times are Brasília time, two hours ahead of Colombia. All report dates are Colombia time.", "",
              "- [quote_snapshot.json](quote_snapshot.json): all 70 quoted prices and filters.",
              "- [predictions.json](predictions.json): full-precision probabilities, all win/pair/goal EV checks, stakes and provenance hashes.",
              "- [cli_outputs.json](cli_outputs.json): actual output and exit status of all fourteen CLI predictions.",
              "- [commands.ps1](commands.ps1): reproducible prediction commands.", "",
              "Run from the project directory to reproduce this fixed snapshot (refuses changed models/data):", "",
              "```powershell", "python match_predictions/brazil_serie_b_2026_10_06_11/analyze.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-quotes", action="store_true",
                        help="Recalculate corrected prices using unchanged verified forecasts")
    args = parser.parse_args()
    quote_path = OUTPUT / "quote_snapshot.json"
    snapshot = json.loads(quote_path.read_text(encoding="utf-8"))
    as_of = pd.Timestamp(snapshot["prediction_as_of"])
    model_path = ROOT / snapshot["model"]
    database_path = ROOT / "brazil.sqlite3"
    protected = [model_path, database_path, Path(str(database_path) + "-wal"),
                 ROOT / "football_model_v3.py", ROOT / "football_model_v2.py", quote_path]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected if p.is_file()}
    old_path = OUTPUT / "predictions.json"
    previous = None
    if old_path.is_file():
        previous = json.loads(old_path.read_text(encoding="utf-8"))
        if hashes != previous["input_sha256"]:
            quote_key = str(quote_path.relative_to(ROOT))
            old_forecast_hashes = {k: v for k, v in previous["input_sha256"].items() if k != quote_key}
            new_forecast_hashes = {k: v for k, v in hashes.items() if k != quote_key}
            if not args.refresh_quotes or old_forecast_hashes != new_forecast_hashes:
                raise ValueError("The fixed snapshot inputs changed. Use a separate folder for new predictions.")
            if pd.Timestamp(previous["prediction_as_of"]) != as_of:
                raise ValueError("Quote refresh cannot change the forecast date.")
            old_teams = [(x['fixture_id'], x['home_id'], x['away_id']) for x in previous['fixtures']]
            new_teams = [(x['fixture_id'], x['home_id'], x['away_id']) for x in snapshot['fixtures']]
            if old_teams != new_teams:
                raise ValueError("Quote refresh cannot change the predicted fixtures.")
    bundle = joblib.load(model_path)
    if bundle.get("kind") != model.KIND or "brazil" not in bundle["sources"]:
        raise ValueError("The selected model must be V3 trained for Brazil.")
    if pd.Timestamp(bundle["training_cutoff"]) > as_of:
        raise ValueError("Prediction date precedes the model training cutoff.")
    if 72 not in bundle["sources"]["brazil"]["league_ids"]:
        raise ValueError("The selected model does not include Série B.")
    rows, _ = model.base.read_databases([database_path])
    with threadpool_limits(limits=2):
        _, state = model.build_features(rows, as_of)
        inputs = []
        fixtures = []
        with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for item in snapshot["fixtures"]:
                fixture = connection.execute("SELECT league_id,kickoff,status,home_id,away_id FROM fixtures WHERE fixture_id=?", (item["fixture_id"],)).fetchone()
                if not fixture or fixture[0] != 72 or fixture[2] != "NS" or fixture[3:] != (item["home_id"], item["away_id"]):
                    raise ValueError(f"Screenshot fixture mapping failed: {item['fixture_id']}")
                if pd.Timestamp(fixture[1]) <= as_of:
                    raise ValueError("The fixture is not in the future.")
                for side in ("home", "away"):
                    resolved, _ = model.base.resolve_team(str(item[f"{side}_id"]), state, "brazil", 72)
                    if resolved != item[f"{side}_id"]:
                        raise ValueError("Wrong team resolution.")
                inputs.append(state.inputs("brazil", 72, 71, item["home_id"], item["away_id"], as_of))
                fixtures.append({**item, "kickoff": fixture[1],
                                 "kickoff_bogota": pd.Timestamp(fixture[1]).tz_convert("America/Bogota").strftime("%d Oct %H:%M")})
        frame = pd.DataFrame(inputs)
        probabilities = model.predict_bundle(bundle, frame)
        xg_home, xg_away = model.expected_goals(bundle, frame)
        over25 = model.total_goal_probabilities(bundle, frame, (3,))[:, 0]
    np.testing.assert_allclose(probabilities.sum(1), 1., atol=1e-12)
    commands = [[sys.executable, str(ROOT / "football_model_v3.py"), "predict", "--country", "brazil", "--league-id", "72", "--home", str(x["home_id"]), "--away", str(x["away_id"]), "--as-of", as_of.isoformat()] for x in fixtures]
    (OUTPUT / "commands.ps1").write_text("\n".join("python football_model_v3.py predict --country brazil --league-id 72 --home " + str(x["home_id"]) + " --away " + str(x["away_id"]) + " --as-of '" + as_of.isoformat() + "'" for x in fixtures) + "\n", encoding="utf-8")
    def run(command):
        env = dict(os.environ, PYTHONUTF8="1")
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        return {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if args.refresh_quotes and previous is not None:
        cli = json.loads((OUTPUT / "cli_outputs.json").read_text(encoding="utf-8"))
        if len(cli) != len(commands) or any(x["command"] != command for x, command in zip(cli, commands)):
            raise ValueError("Saved CLI commands differ from the unchanged forecast snapshot.")
        print(f"Rechecking {len(cli)} already executed CLI forecasts with corrected quotes...", flush=True)
    else:
        print(f"Executing all {len(commands)} Série B prediction commands...", flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            cli = list(pool.map(run, commands))
    write_json("cli_outputs.json", cli)
    if any(x["exit_code"] != 0 for x in cli):
        raise ValueError("A prediction command failed; inspect cli_outputs.json.")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "prediction_as_of": as_of.isoformat(),
              "prediction_as_of_bogota": as_of.tz_convert("America/Bogota").isoformat(),
              "training_cutoff": bundle["training_cutoff"],
              "training_cutoff_bogota": pd.Timestamp(bundle["training_cutoff"]).tz_convert("America/Bogota").isoformat(),
              "latest_completed_bogota": rows[rows.league_id == 72].kickoff.max().tz_convert("America/Bogota").isoformat(),
              "model_selection": bundle["selection"], "calibration_parameters": bundle["calibration_parameters"],
              "training_matches": bundle["training_matches"], "stake_cop_per_bet": snapshot["stake_cop_per_bet"],
              "filters": {"win": .6, "win_or_draw": .6, "goals": None},
              "schedule_source": snapshot["schedule_source"], "input_sha256": hashes,
              "fixtures": [], "qualifying_wins": [], "qualifying_win_or_draw": [],
              "qualifying_goals": [], "highest_ev_per_fixture": []}
    stake = snapshot["stake_cop_per_bet"]
    for i, item in enumerate(fixtures):
        p = probabilities[i]
        for label, value in zip(model.base.LABELS, p):
            if f"{label}: {value:.2%}" not in cli[i]["stdout"]:
                raise ValueError(f"CLI and full-precision forecast disagree: {item['fixture_id']}")
        if f"over 2.5): {over25[i]:.2%}" not in cli[i]["stdout"]:
            raise ValueError("CLI goal probability does not match the full-precision forecast.")
        match = f"{item['home']} vs {item['away']}"
        record = {**item, "match": match, "xg_home": float(xg_home[i]), "xg_away": float(xg_away[i]),
                  "outcomes": {}, "win_or_draw": {}, "goals": {}}
        context = {"fixture_id": item["fixture_id"], "match": match, "kickoff_bogota": item["kickoff_bogota"]}
        candidates = []
        for label, probability, key in zip(("home", "draw", "away"), p, ("h", "d", "a")):
            outcome = single(probability, item[key], stake)
            record["outcomes"][label] = outcome
            if label == "draw":
                continue
            if outcome["positive_ev"] and probability >= snapshot["win_minimum_probability"]:
                selection = {**context, **outcome, "team": item[label], "kind": "win", "selection": item[label] + " wins"}
                report["qualifying_wins"].append(selection)
                candidates.append(selection)
            pair = split(probability, p[1], item[key], item["d"], stake)
            record["win_or_draw"][label] = pair
            if pair["positive_ev"] and pair["probability"] >= snapshot["win_or_draw_minimum_probability"]:
                selection = {**context, **pair, "team": item[label], "kind": "win_or_draw", "selection": item[label] + " or draw"}
                report["qualifying_win_or_draw"].append(selection)
                candidates.append(selection)
        for side, probability in (("over", over25[i]), ("under", 1 - over25[i])):
            goal = single(probability, item[side], stake)
            record["goals"][side] = goal
            if goal["positive_ev"]:
                selection = {**context, **goal, "market": f"{side.title()} 2.5 goals", "kind": "goals", "selection": f"{side.title()} 2.5 goals"}
                report["qualifying_goals"].append(selection)
                candidates.append(selection)
        if candidates:
            report["highest_ev_per_fixture"].append(max(candidates, key=lambda x: x["ev"]))
        report["fixtures"].append(record)
    for key in ("qualifying_wins", "qualifying_win_or_draw", "qualifying_goals", "highest_ev_per_fixture"):
        report[key].sort(key=lambda x: x["ev"], reverse=True)
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Protected input changed: {name}")
    write_json("predictions.json", report)
    (OUTPUT / "README.md").write_text(render(report), encoding="utf-8")
    print(f"Qualifying bets: {len(report['qualifying_wins'])} wins, {len(report['qualifying_win_or_draw'])} win-or-draw plans, {len(report['qualifying_goals'])} goal bets.", flush=True)
    print(f"Saved {OUTPUT / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
