"""Combine two quoted-price snapshots and allocate COP25,000 per game."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
THRESHOLD = .60


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]]


def money(value):
    return f"{value:,.0f}"


def pair_metrics(record, winner, win_quote, draw_quote, stake):
    p_win = record["outcomes"][winner]["probability"]
    p_draw = record["outcomes"]["draw"]["probability"]
    p_covered = p_win + p_draw
    win_odds, draw_odds = win_quote["odds"], draw_quote["odds"]
    inverse_sum = 1 / win_odds + 1 / draw_odds
    win_exact = stake * (1 / win_odds) / inverse_sum
    draw_exact = stake - win_exact
    win_stake = int(round(win_exact))
    draw_stake = stake - win_stake
    win_payout, draw_payout = win_stake * win_odds, draw_stake * draw_odds
    expected_profit = p_win * win_payout + p_draw * draw_payout - stake
    if win_stake + draw_stake != stake:
        raise ValueError("Split stake exceeds the per-game budget.")
    # Check equivalent forms of equal-payout expected profit, before rounding.
    theoretical = p_covered / inverse_sum - 1
    actual_before_rounding = (p_win * win_exact * win_odds + p_draw * draw_exact * draw_odds - stake) / stake
    if abs(theoretical - actual_before_rounding) > 1e-12:
        raise ValueError("Dutching formulas disagree.")
    return {"fixture_id": record["fixture_id"], "match": f"{record['home']} vs {record['away']}",
            "winner": winner, "team": record[winner], "team_probability": p_win, "draw_probability": p_draw,
            "combined_probability": p_covered, "opponent_win_probability": 1 - p_covered,
            "win_price": win_quote, "draw_price": draw_quote,
            "effective_decimal_odds": 1 / inverse_sum, "combined_fair_odds": 1 / p_covered,
            "team_stake_exact": win_exact, "draw_stake_exact": draw_exact,
            "team_stake_cop": win_stake, "draw_stake_cop": draw_stake, "total_stake_cop": stake,
            "payout_on_team_win_cop": win_payout, "payout_on_draw_cop": draw_payout,
            "net_profit_on_team_win_cop": win_payout - stake, "net_profit_on_draw_cop": draw_payout - stake,
            "minimum_net_profit_if_covered_cop": min(win_payout, draw_payout) - stake,
            "net_profit_if_opponent_wins_cop": -stake,
            "estimated_ev_before_stake_rounding": theoretical,
            "estimated_ev": expected_profit / stake, "estimated_profit_cop": expected_profit,
            "positive_ev": expected_profit > 0, "coverage_at_least_60pct": p_covered >= THRESHOLD}


def goal_metrics(record, second, side, stake):
    line = second["line"]
    minimum = str(int(line + .5))
    over_probability = record["goal_totals"][minimum]["probability"]
    probability = over_probability if side == "over" else 1 - over_probability
    odds = second[side]
    ev = probability * odds - 1
    return {"fixture_id": record["fixture_id"], "match": f"{record['home']} vs {record['away']}",
            "market": f"{side} {line:.1f} goals", "side": side, "line": line,
            "probability": probability, "fair_odds": 1 / probability,
            "price": {"odds": odds, "source": "B"}, "stake_cop": stake,
            "net_profit_if_correct_cop": stake * (odds - 1), "net_profit_if_incorrect_cop": -stake,
            "estimated_ev": ev, "estimated_profit_cop": stake * ev,
            "positive_ev": ev > 0, "probability_at_least_60pct": probability >= THRESHOLD}


def render(report):
    stake = report["stake_cop_per_game"]
    shortlist = report["shortlist_at_least_60pct"]
    lines = ["# Two-page odds comparison: win-or-draw splits and total goals", "",
        f"The user confirmed a budget of **COP {money(stake)} per game**. **A = first screenshot; B = second screenshot.** This report selects the better quoted 1X2 price for each outcome. Equal prices default to A. Goal prices are available from B only.", "",
        f"The V3 predictions are the unchanged forecast snapshot from **{report['prediction_as_of']}** (4 October, 20:01:20 in Colombia), fitted through **{report['model_training_cutoff']}**. The quoted odds are screenshots rather than a live price feed. Source-model and data hashes still match the original prediction run.", "",
        "Each stake split backs the team's win and the draw as **two separate bets**, allocating more stake to the shorter price so both covered results pay approximately the same gross amount. This method is commonly called [dutching](https://apps.betfair.com/learning/dutching/). It can use different pages for the two bets.", "",
        "For win odds W and draw odds D, the exact team stake is `S × D / (W + D)` and draw stake is `S × W / (W + D)`. The effective price is `1 / (1/W + 1/D)`. The combined probability is `P(team win) + P(draw)` because the outcomes are mutually exclusive. Before rounding, estimated EV is `combined probability × effective price − 1`.", "",
        "Stakes below are rounded to the nearest COP and sum to COP25,000. EV is recomputed from those rounded stakes. The **covered net profit** subtracts both stakes and uses the smaller payout after rounding. If the other team wins, the entire COP25,000 is lost. The estimates use listed decimal prices without deductions for additional charges.", "",
        "## Positive EV win-or-draw splits with at least 60% combined probability", ""]
    qualifying = [pair for pair in report["positive_ev_pairs"] if pair["coverage_at_least_60pct"]]
    lines += table(["Match", "Team or draw", "Combined probability", "Team stake @ odds (page)", "Draw stake @ odds (page)", "Estimated EV", "Minimum net profit if covered"],
                   [[p["match"], p["team"], f"{p['combined_probability']:.2%}",
                     f"{money(p['team_stake_cop'])} @ {p['win_price']['odds']:.2f} ({p['win_price']['source']})",
                     f"{money(p['draw_stake_cop'])} @ {p['draw_price']['odds']:.2f} ({p['draw_price']['source']})",
                     f"{p['estimated_ev']:+.2%}", money(p["minimum_net_profit_if_covered_cop"])] for p in qualifying])
    lines += ["", "## Other positive EV win-or-draw splits below 60%", ""]
    low = [pair for pair in report["positive_ev_pairs"] if not pair["coverage_at_least_60pct"]]
    lines += table(["Match", "Team or draw", "Combined probability", "Team stake @ odds (page)", "Draw stake @ odds (page)", "Estimated EV", "Minimum net profit if covered"],
                   [[p["match"], p["team"], f"{p['combined_probability']:.2%}",
                     f"{money(p['team_stake_cop'])} @ {p['win_price']['odds']:.2f} ({p['win_price']['source']})",
                     f"{money(p['draw_stake_cop'])} @ {p['draw_price']['odds']:.2f} ({p['draw_price']['source']})",
                     f"{p['estimated_ev']:+.2%}", money(p["minimum_net_profit_if_covered_cop"])] for p in low])
    lines += ["", "## Middlesbrough example", ""]
    middle = next(p for p in report["positive_ev_pairs"] if p["team"] == "Middlesbrough")
    lines += [f"Back **Middlesbrough with COP {money(middle['team_stake_cop'])} at {middle['win_price']['odds']:.2f} on A**, and **the draw with COP {money(middle['draw_stake_cop'])} at {middle['draw_price']['odds']:.2f} on A**. Both covered results return **COP {middle['payout_on_team_win_cop']:,.2f}**, giving **COP {middle['net_profit_on_team_win_cop']:,.2f} net profit**. If Wolves win, the net loss is COP25,000.", "",
        f"The model gives Middlesbrough or draw **{middle['combined_probability']:.2%}**, equivalent fair odds **{middle['combined_fair_odds']:.4f}**. The two bets create effective odds **{middle['effective_decimal_odds']:.5f}**, estimated EV **{middle['estimated_ev']:+.2%}**, and expected profit **COP {middle['estimated_profit_cop']:,.2f}**. Expected profit averages all outcomes; it differs from the profit paid if the bet succeeds.", "",
        "## Positive EV goals options", "",
        "Each goals row is an **alternative use of the same COP 25,000 per-game budget**. Choose one plan per match to keep within that budget. Under 2.5 means at most 2 goals; under 3.5 means at most 3. Under probabilities are complements of V3's matching over probabilities from its goal component. They are not combined with 1X2 probabilities.", ""]
    lines += table(["Match", "Market", "V3 probability", "Fair odds", "B odds", "Estimated EV", "Stake", "Net profit if correct", "At least 60%?"],
                   [[g["match"], g["market"], f"{g['probability']:.2%}", f"{g['fair_odds']:.2f}", f"{g['price']['odds']:.2f}",
                     f"{g['estimated_ev']:+.2%}", money(g["stake_cop"]), money(g["net_profit_if_correct_cop"]),
                     "Yes" if g["probability_at_least_60pct"] else "No"] for g in report["positive_ev_goals"]])
    lines += ["", "Every quoted over-goals market has negative estimated EV. The only positive-value goals option meeting 60% is **West Ham–QPR under 3.5 at 1.54**. Middlesbrough–Wolves under 2.5 at 2.25 is slightly below break-even in the model and is excluded from positive-EV options.", "",
        "## One plan per game meeting 60% and positive EV", "",
        "This shortlist applies the 60% coverage threshold and chooses the largest positive EV option among eligible win-or-draw and goals plans for each game. These are candidate allocations derived from V3 estimates, rather than demonstrated profitable bets.", ""]
    rows = []
    for item in shortlist:
        p = item["plan"]
        if item["kind"] == "win_or_draw":
            action = f"{p['team']}: {money(p['team_stake_cop'])} @ {p['win_price']['odds']:.2f} ({p['win_price']['source']}); draw: {money(p['draw_stake_cop'])} @ {p['draw_price']['odds']:.2f} ({p['draw_price']['source']})"
            probability = p["combined_probability"]
        else:
            action = f"{p['market']}: {money(p['stake_cop'])} @ {p['price']['odds']:.2f} (B)"
            probability = p["probability"]
        rows.append([p["match"], action, f"{probability:.2%}", f"{p['estimated_ev']:+.2%}", money(p["estimated_profit_cop"])])
    lines += table(["Match", "COP25,000 plan", "Probability", "Estimated EV", "Expected profit, COP"], rows)
    lines += ["", f"Using all {len(shortlist)} shortlist plans would stake **COP {money(stake * len(shortlist))}**. Expected profits add across plans without requiring independence, but the report does not estimate the joint chance of an overall portfolio profit.", "",
        "## Complete price and EV checks", "",
        "Both home-win-or-draw and away-win-or-draw splits are checked in every match. The following table preserves all 12 fixtures, including cases where a positive individual win or draw price becomes negative EV after adding the other outcome.", ""]
    rows = []
    for record in report["fixtures"]:
        rows.append([record["match"],
                     *[f"{record['best_1x2_prices'][outcome]['odds']:.2f} ({record['best_1x2_prices'][outcome]['source']})" for outcome in ("home", "draw", "away")],
                     f"{record['win_or_draw']['home']['estimated_ev']:+.2%}", f"{record['win_or_draw']['away']['estimated_ev']:+.2%}",
                     f"{record['goals']['over']['estimated_ev']:+.2%}", f"{record['goals']['under']['estimated_ev']:+.2%}"])
    lines += table(["Match", "Best home", "Best draw", "Best away", "Home or draw EV", "Away or draw EV", "Over goals EV", "Under goals EV"], rows)
    lines += ["", "## Limits and files", "",
        "V3 probabilities have prediction error. More outcome coverage does not guarantee profit, and small estimated margins can disappear when probabilities or offered prices change. The English cache's latest completed result remains 19 September 2026; no updated injuries or starting lineups are incorporated. The split's estimated EV is the stake-weighted average of the individual win and draw bets' EVs.", "",
        "The two win/draw bets are separate selections, so they are not an accumulator and should not be multiplied together. Goal options are listed as separate alternatives; no independence between match outcome and total goals is assumed. Prices may differ by page, and the stake split must be recalculated if either price changes.", "",
        "- [combined_odds_stakes.json](combined_odds_stakes.json): all24 win-or-draw pairs, all24 goals markets, rounded stakes, payout states, positive-value filters and one-plan shortlist.",
        "- [second_quote_snapshot.json](second_quote_snapshot.json): all prices and goal lines transcribed from screenshot B.",
        "- [quote_snapshot.json](quote_snapshot.json): screenshot A prices.",
        "- [predictions.json](predictions.json): unchanged full-precision V3 predictions and source hashes.",
        "- [Original prediction report](README.md): all12 CLI commands, 1X2 probabilities and goal forecasts.", "",
        "To reproduce from the project directory:", "", "```powershell",
        "python match_predictions/championship_2026_10_09_11/analyze_combined_odds.py", "```", ""]
    return "\n".join(lines)


def main():
    first = read_json("quote_snapshot.json")
    second = read_json("second_quote_snapshot.json")
    forecasts = read_json("predictions.json")
    for relative, expected in forecasts["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"The original forecast inputs changed: {relative}")
    input_paths = [OUTPUT / name for name in ("quote_snapshot.json", "second_quote_snapshot.json", "predictions.json")]
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in input_paths}
    first_quotes = {(r["home"], r["away"]): r for r in first["fixtures"]}
    second_quotes = {(r["home"], r["away"]): r for r in second["fixtures"]}
    stake = int(second["stake_cop_per_game"])
    if stake != 25000 or set(first_quotes) != set(second_quotes):
        raise ValueError("Budget or fixture identity mismatch.")
    report = {"prediction_as_of": forecasts["prediction_as_of"], "model_training_cutoff": forecasts["model_training_cutoff"],
              "created_at": datetime.now(timezone.utc).isoformat(), "stake_cop_per_game": stake,
              "screen_labels": {"A": "First screenshot", "B": "Second screenshot"},
              "input_sha256": {**forecasts["input_sha256"], **hashes}, "fixtures": [],
              "positive_ev_pairs": [], "positive_ev_goals": [], "shortlist_at_least_60pct": []}
    for record in forecasts["fixtures"]:
        key = (record["home"], record["away"])
        a, b = first_quotes[key], second_quotes[key]
        best = {}
        for outcome, code in zip(("home", "draw", "away"), ("h", "d", "a")):
            best[outcome] = {"odds": max(a[code], b[code]), "source": "A" if a[code] >= b[code] else "B"}
        pairs = {winner: pair_metrics(record, winner, best[winner], best["draw"], stake) for winner in ("home", "away")}
        goals = {side: goal_metrics(record, b, side, stake) for side in ("over", "under")}
        if abs(goals["over"]["probability"] + goals["under"]["probability"] - 1) > 1e-12:
            raise ValueError("Goal probabilities must be complementary.")
        item = {"fixture_id": record["fixture_id"], "match": f"{record['home']} vs {record['away']}",
                "best_1x2_prices": best, "win_or_draw": pairs, "goals": goals}
        report["fixtures"].append(item)
        positive_pairs = [p for p in pairs.values() if p["positive_ev"]]
        positive_goals = [g for g in goals.values() if g["positive_ev"]]
        report["positive_ev_pairs"] += positive_pairs
        report["positive_ev_goals"] += positive_goals
        candidates = [{"kind": "win_or_draw", "plan": p} for p in positive_pairs if p["coverage_at_least_60pct"]]
        candidates += [{"kind": "goals", "plan": g} for g in positive_goals if g["probability_at_least_60pct"]]
        if candidates:
            report["shortlist_at_least_60pct"].append(max(candidates, key=lambda c: c["plan"]["estimated_ev"]))
    report["positive_ev_pairs"].sort(key=lambda p: p["estimated_ev"], reverse=True)
    report["positive_ev_goals"].sort(key=lambda g: g["estimated_ev"], reverse=True)
    report["shortlist_at_least_60pct"].sort(key=lambda c: c["plan"]["estimated_ev"], reverse=True)
    report["shortlist_total_stake_cop"] = len(report["shortlist_at_least_60pct"]) * stake
    report["shortlist_total_expected_profit_cop"] = sum(c["plan"]["estimated_profit_cop"] for c in report["shortlist_at_least_60pct"])
    for relative, expected in report["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"An input changed during analysis: {relative}")
    (OUTPUT / "combined_odds_stakes.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "COMBINED_ODDS_STAKES.md").write_text(render(report), encoding="utf-8")
    for p in report["positive_ev_pairs"]:
        print(f"{p['team']} or draw: {p['combined_probability']:.4%}; EV {p['estimated_ev']:+.4%}; stakes {p['team_stake_cop']}/{p['draw_stake_cop']}; prices {p['win_price']}/{p['draw_price']}; net if covered >= {p['minimum_net_profit_if_covered_cop']:.2f}; expected profit {p['estimated_profit_cop']:.2f}")
    for g in report["positive_ev_goals"]:
        print(f"{g['match']} {g['market']}: {g['probability']:.4%}; odds {g['price']['odds']}; EV {g['estimated_ev']:+.4%}")
    print(f"Shortlist: {len(report['shortlist_at_least_60pct'])} games, total stake COP{report['shortlist_total_stake_cop']:,}")
    print(f"Saved {OUTPUT / 'COMBINED_ODDS_STAKES.md'}")


if __name__ == "__main__":
    main()
