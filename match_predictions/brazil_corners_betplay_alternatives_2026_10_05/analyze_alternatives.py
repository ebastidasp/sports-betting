"""Compare newly captured alternate full-time corner lines with verified forecasts.

python match_predictions/brazil_corners_betplay_alternatives_2026_10_05/analyze_alternatives.py --require-all

Compact quote batches have columns book_event_id, observed_at, book_home,
book_away, lines; line_columns line, over, under; and rows containing arrays.
The script reads local evidence only. It neither retrains nor submits bets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "match_predictions" / "brazil_corners_betplay_2026_10_05"
LEAGUES = {71: "Brazil Serie A", 72: "Brazil Serie B"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def evidence(prediction_path):
    """Reject stale/changed inference inputs; reuse the original fixed as-of."""
    prediction_path = Path(prediction_path).resolve()
    baseline_sha = sha256(prediction_path)
    baseline = read_json(prediction_path)
    protected = baseline["provenance"]["sha256"]
    mismatches = [path for path, expected in protected.items()
                  if not Path(path).is_file() or sha256(path) != expected]
    if mismatches:
        raise ValueError("Original inference inputs changed; refresh the corner forecasts before repricing: " + ", ".join(mismatches))
    helper_path = prediction_path.with_name("analyze.py")
    if str(helper_path) not in protected:
        raise ValueError("The original pricing helper lacks provenance")
    spec = importlib.util.spec_from_file_location("verified_corner_analysis", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    proofs_path = prediction_path.with_name("cli_outputs.json")
    proof_sha = sha256(proofs_path)
    proofs = read_json(proofs_path)
    fixtures = baseline["fixtures"]
    fixture_map = {int(row["fixture_id"]): row for row in fixtures}
    proof_map = {int(row["fixture_id"]): row for row in proofs}
    if len(fixtures) != 40 or len(fixture_map) != 40 or len(proofs) != 40 or len(proof_map) != 40 or set(fixture_map) != set(proof_map):
        raise ValueError("Expected the same 40 uniquely verified Brazil fixtures")
    if baseline["cli_verification_count"] != 40 or baseline["excluded_fixtures"]:
        raise ValueError("Original forecast verification is incomplete")
    cutoff = helper.timestamp(baseline["prediction_as_of"])
    if cutoff < helper.timestamp(baseline["model_training_cutoff"]):
        raise ValueError("Original forecast precedes model training")
    event_map = {}
    for fixture_id, forecast in fixture_map.items():
        means = forecast["means"]
        if any(not math.isfinite(float(value)) or value <= 0 for value in means.values()):
            raise ValueError(f"Invalid original mean for fixture {fixture_id}")
        if not math.isclose(means["home"] + means["away"], means["total"], abs_tol=1e-12):
            raise ValueError("Original total mean does not equal both team means")
        alpha = float(forecast["distribution"]["total"]["alpha"])
        if alpha != float(baseline["dispersion"]["total"]):
            raise ValueError("Original total dispersion is inconsistent")
        proof = proof_map[fixture_id]
        if proof["exit_code"] != 0 or not proof["matches_within_display_rounding"]:
            raise ValueError("Original CLI proof failed")
        command = proof["command"]
        expected_args = {"--country": "brazil", "--league-id": str(forecast["league_id"]),
                         "--home": str(forecast["home_id"]), "--away": str(forecast["away_id"]),
                         "--model": baseline["model"], "--db": baseline["provenance"]["inference_database"],
                         "--as-of": cutoff.isoformat()}
        if any(flag not in command or command[command.index(flag) + 1] != value for flag, value in expected_args.items()):
            raise ValueError("Original CLI identity or as-of differs from the forecast")
        match = re.search(r"Expected corners: home ([\d.]+), away ([\d.]+), total ([\d.]+)", proof["stdout"])
        if match is None or any(abs(float(shown) - means[key]) > .00500001 for shown, key in zip(match.groups(), ("home", "away", "total"))):
            raise ValueError("Original CLI mean verification failed")
        distribution = helper.corners.count_distribution(means["total"], alpha)
        line = float(proof["line"])
        for symbol, probability in ((">", float(distribution.sf(math.floor(line)))), ("<", float(distribution.cdf(math.ceil(line) - 1)))):
            match = re.search(r"P\(total corners " + re.escape(symbol) + r" [\d.]+\)\s*:\s*([\d.]+)%", proof["stdout"])
            if match is None or abs(float(match.group(1)) / 100 - probability) > .00005001:
                raise ValueError("Original CLI probability verification failed")
        captured = forecast["captured_fixture"]
        event = captured.get("book_event_id")
        if event is None:
            match = re.search(r"/event/(\d+)", forecast["source_url"])
            event = match.group(1) if match else None
        if event is None or str(event) in event_map:
            raise ValueError("Original bookmaker event mapping is missing or ambiguous")
        event_map[str(event)] = forecast
    paths = {prediction_path, proofs_path, Path(__file__).resolve(), *map(Path, protected)}
    hashes = {str(path.resolve()): sha256(path) for path in sorted(paths, key=str)}
    if hashes[str(prediction_path)] != baseline_sha or hashes[str(proofs_path)] != proof_sha:
        raise ValueError("Original evidence changed while being read")
    return baseline, event_map, helper, hashes


def quote_files(patterns):
    paths = set()
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            paths.update(item.resolve() for item in path.glob("quotes_*.json"))
        else:
            paths.update(Path(item).resolve() for item in glob.glob(str(path)))
    if not paths:
        raise ValueError("No detailed quote batches were found")
    return sorted(paths, key=str)


def rows_from(payload, path):
    if "rows" in payload:
        columns = payload.get("columns", ["book_event_id", "observed_at", "book_home", "book_away", "lines"])
        line_columns = payload.get("line_columns", ["line", "over", "under"])
        for values in payload["rows"]:
            if isinstance(values, dict):
                row = dict(values)
            else:
                if len(values) != len(columns):
                    raise ValueError(f"A compact row does not match its columns: {path}")
                row = dict(zip(columns, values))
            row["lines"] = [dict(values) if isinstance(values, dict) else dict(zip(line_columns, values)) for values in row.get("lines", [])]
            row["input_file"] = str(path)
            yield row
    else:
        for row in payload.get("fixtures", []):
            yield {**row, "input_file": str(path)}


def number(value):
    result = float(value.replace(",", ".") if isinstance(value, str) else value)
    if not math.isfinite(result):
        raise ValueError("A quoted number is not finite")
    return result


def sorted_markets(rows):
    return sorted(rows, key=lambda row: (-row["ev"], row["fixture_id"], row["side"], row["line"]))


def best_by_fixture(rows):
    best = {}
    for row in sorted_markets(rows):
        best.setdefault(row["fixture_id"], row)
    return list(best.values())


def market_label(row):
    return f"{'Over' if row['side'] == 'over' else 'Under'} {row['line']:g}"


def table(rows, helper):
    lines = ["| Fixture | Kickoff Bogota | Total corner selection | Win probability | ≥60% | Offered / fair odds | EV | Expected profit COP | Previous main line |",
             "|---|---|---|---:|:---:|---:|---:|---:|:---:|"]
    if not rows:
        lines.append("| — | — | No qualifying captured selection | — | — | — | — | — | — |")
    for row in rows:
        fair = f"{row['fair_odds']:.3f}" if row["fair_odds"] is not None else "∞"
        lines.append(f"| {row['home']} – {row['away']} | {row['kickoff_bogota']} | {market_label(row)} | {row['positive_profit_probability']:.2%} | {'Yes' if row['positive_profit_probability_at_least_60pct'] else 'No'} | {row['odds']:.2f} / {fair} | {row['ev']:.2%} | {helper.cop(row['expected_profit_cop'])} | {'Yes' if row['is_previous_main_line'] else 'No'} |")
    return lines


def render(report, helper):
    complete = "Complete" if report["coverage"]["complete"] else "Partial capture"
    lines = ["# BetPlay alternate full-time total corner lines", "",
             f"{complete}: {report['coverage']['captured_fixtures']}/40 fixtures, {len(report['markets'])} priced selections. Created {report['created_at']}.", "",
             f"Forecasts retain the original **{report['prediction_as_of']}** historical as-of and model cutoff **{report['model_training_cutoff']}**. New price observations range from {report['quote_observed_range']['earliest']} to {report['quote_observed_range']['latest']}. The original 40 CLI proofs are reused; no retraining or forecast recalculation was needed because their protected inputs remain unchanged.", "",
             f"Each row represents **one alternative bet of COP {helper.cop(report['stake_cop'])}**. Choose a single line per fixture when comparing these tables; their stake amounts are not combined within the game. Different tables are alternative selection rules, not separate budgets.", "",
             "Only newly captured full-time total corner prices enter the ranking. The previous-main-line marker compares the line with the older report, without using its older price. There is no minimum probability filter for positive EV; the ≥60% table adds that condition.", "",
             "For half lines, Over 8.5 wins with at least 9 corners and Under 11.5 wins with at most 11. Fair decimal odds equal 1 / probability, and EV equals probability × offered odds − 1. Integer pushes and quarter split stakes are accounted for if explicitly captured. Probabilities refer to profit-producing outcomes, including half wins for quarter lines.", "",
             "For later rounds, refresh forecasts and prices after intervening matches finish. These frozen forecasts do not include those future results. The built-in 2025 Brazil test covers 380 Serie A matches with total-corner MAE 2.630 and excludes Serie B; it does not validate profit at the captured prices.", "",
             f"Positive selections: {len(report['positive_ev_markets'])}; fixtures with any positive selection: {len(report['best_positive_ev_per_fixture'])}; fixtures with a positive selection and probability ≥60%: {len(report['best_positive_ev_at_least_60pct_per_fixture'])}.", ""]
    for league, name in LEAGUES.items():
        unrestricted = [row for row in report["best_positive_ev_per_fixture"] if row["league_id"] == league]
        thresholded = [row for row in report["best_positive_ev_at_least_60pct_per_fixture"] if row["league_id"] == league]
        lines += [f"## {name} (league {league})", "", "### Highest positive EV per fixture", ""]
        lines += table(unrestricted, helper)
        lines += ["", "### Highest positive EV with win probability ≥60% per fixture", ""]
        lines += table(thresholded, helper)
        lines += ["", f"The first rule has {len(unrestricted)} reference bets, totaling COP {helper.cop(len(unrestricted) * report['stake_cop'])}; the ≥60% rule has {len(thresholded)}, totaling COP {helper.cop(len(thresholded) * report['stake_cop'])}. These are alternative plans. Expected profit is a model expectation, not a guaranteed return.", ""]
    lines += ["## Complete line comparisons", "",
              f"[All positive captured lines](</{HERE.as_posix()}/ALL_POSITIVE_LINES.md>) and [all captured lines, including negative EV](</{HERE.as_posix()}/ALL_LINES.md>) contain the full comparison. predictions.json preserves full precision and settlement payouts.", "",
              "### Highest available EV per captured fixture, including fixtures without positive EV", ""]
    lines += table(report["best_available_ev_per_fixture"], helper)
    if report["date_time_anomalies"]:
        lines += ["", "## Schedule discrepancies", "", "The forecast retains the cache kickoff; the original bookmaker display is preserved in the JSON.", ""]
        for row in report["date_time_anomalies"]:
            lines.append("- " + json.dumps(row, ensure_ascii=False))
    lines += ["", "## Capture coverage and exclusions", ""]
    if report["coverage"]["missing_fixture_ids"]:
        lines.append("Uncaptured original fixture IDs: " + ", ".join(map(str, report["coverage"]["missing_fixture_ids"])) + ".")
    for row in report["exclusions"]:
        lines.append(f"- Event {row.get('book_event_id')}: {row['reason']}.")
    if not report["exclusions"] and report["coverage"]["complete"]:
        lines.append("Every original fixture has captured line prices; no captured selections were excluded.")
    lines += ["", "## Reproduction and provenance", "", "```powershell",
              "python match_predictions/brazil_corners_betplay_alternatives_2026_10_05/analyze_alternatives.py --require-all", "```", "",
              "The 40 original CLI proofs and their full-precision forecast report are retained unchanged in the original directory. Input hashes below freeze the parent evidence, model/cache/code and detailed quote files used here.", "", "| Input | SHA-256 |", "|---|---|"]
    for path, digest in report["provenance"]["sha256"].items():
        lines.append(f"| {path} | `{digest}` |")
    return "\n".join(lines) + "\n"


def analyze(prediction_path, patterns, output, stake=25000, require_all=False):
    baseline, event_map, helper, hashes = evidence(prediction_path)
    files = quote_files(patterns)
    latest, superseded, exclusions, file_metadata = {}, [], [], []
    for path in files:
        digest = sha256(path)
        payload = read_json(path)
        hashes[str(path)] = digest
        file_metadata.append({"path": str(path), "metadata": {key: value for key, value in payload.items() if key not in ("rows", "fixtures")}})
        for row in rows_from(payload, path):
            event = str(row.get("book_event_id", ""))
            if event not in event_map:
                raise ValueError(f"Detailed quote event is outside the original 40-fixture mapping: {event}")
            observed = helper.timestamp(row["observed_at"])
            if observed >= helper.timestamp(event_map[event]["kickoff"]):
                raise ValueError("Captured prices are not prematch for the mapped fixture")
            row["observed_at"] = observed.isoformat()
            if event in latest:
                earlier, later = sorted((latest[event], row), key=lambda item: item["observed_at"])
                superseded.append({"book_event_id": event, "discarded_observed_at": earlier["observed_at"], "retained_observed_at": later["observed_at"]})
                latest[event] = later
            else:
                latest[event] = row
    captured_ids = {int(event_map[event]["fixture_id"]) for event in latest}
    missing = sorted(int(row["fixture_id"]) for row in baseline["fixtures"] if int(row["fixture_id"]) not in captured_ids)
    if require_all and missing:
        raise ValueError(f"Detailed prices missing for {len(missing)} original fixtures: {missing}")
    old_main_lines = {}
    for row in baseline["markets"]:
        if row["scope"] == "total":
            old_main_lines.setdefault(int(row["fixture_id"]), set()).add(float(row["line"]))
    markets, captured_fixtures = [], []
    for event, row in latest.items():
        forecast = event_map[event]
        fid = int(forecast["fixture_id"])
        mean = float(forecast["means"]["total"])
        alpha = float(forecast["distribution"]["total"]["alpha"])
        distribution = helper.corners.count_distribution(mean, alpha)
        captured_fixtures.append({"fixture_id": fid, "book_event_id": event,
                                  "league_id": int(forecast["league_id"]), "home": forecast["home"], "away": forecast["away"],
                                  "observed_at": row["observed_at"], "source_url": forecast["source_url"],
                                  "captured_row": row, "forecast": forecast})
        seen = set()
        for quote in row.get("lines", []):
            for side in ("over", "under"):
                try:
                    if quote.get("period", "full_time") != "full_time" or quote.get("scope", "total") != "total":
                        raise ValueError("Only full-time total corners are modeled")
                    if quote.get("disabled", False) or quote.get(side + "_disabled", False):
                        raise ValueError("Selection is disabled")
                    if quote.get(side) is None:
                        raise ValueError("Selection has no available price")
                    line, odds = number(quote["line"]), number(quote[side])
                    identity = (side, line)
                    if identity in seen:
                        raise ValueError("Duplicate line/side within the same event snapshot")
                    seen.add(identity)
                    pricing = helper.price_market(distribution, side, line, odds, stake)
                    markets.append({"fixture_id": fid, "book_event_id": event, "league_id": int(forecast["league_id"]),
                                    "home": forecast["home"], "away": forecast["away"],
                                    "book_home": row.get("book_home"), "book_away": row.get("book_away"),
                                    "kickoff": forecast["kickoff"], "kickoff_bogota": helper.timestamp(forecast["kickoff"]).tz_convert("America/Bogota").isoformat(),
                                    "market": "Full-time total corners", "scope": "total", "period": "full_time",
                                    "side": side, "line": line, "odds": odds, "mean": mean, "alpha": alpha,
                                    "observed_at": row["observed_at"], "source_url": forecast["source_url"],
                                    "input_file": row["input_file"], "is_previous_main_line": line in old_main_lines.get(fid, set()),
                                    **pricing})
                except (ValueError, KeyError, TypeError) as error:
                    exclusions.append({"book_event_id": event, "fixture_id": fid, "side": side, "quote": quote, "reason": str(error)})
    changed = [path for path, digest in hashes.items() if not Path(path).is_file() or sha256(path) != digest]
    if changed:
        raise ValueError("Protected forecast evidence or detailed quotes changed during analysis: " + ", ".join(changed))
    if not markets:
        raise ValueError("No available full-time total corner prices were captured")
    positive = sorted_markets([row for row in markets if row["positive_ev"]])
    qualified = [row for row in positive if row["positive_profit_probability_at_least_60pct"]]
    observations = [row["observed_at"] for row in latest.values()]
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "prediction_as_of": baseline["prediction_as_of"], "model_training_cutoff": baseline["model_training_cutoff"],
              "stake_cop": stake, "model": baseline["model"], "dispersion": baseline["dispersion"],
              "historical_evaluation": baseline["historical_evaluation"],
              "quote_observed_range": {"earliest": min(observations), "latest": max(observations)},
              "coverage": {"expected_fixtures": 40, "captured_fixtures": len(latest), "complete": not missing,
                           "missing_fixture_ids": missing, "priced_selections": len(markets),
                           "priced_fixture_ids": sorted({row["fixture_id"] for row in markets})},
              "fixtures": sorted(captured_fixtures, key=lambda row: (row["league_id"], row["fixture_id"])),
              "markets": sorted(markets, key=lambda row: (row["league_id"], row["fixture_id"], row["line"], row["side"])),
              "positive_ev_markets": positive, "best_available_ev_per_fixture": best_by_fixture(markets),
              "best_positive_ev_per_fixture": best_by_fixture(positive),
              "best_available_ev_at_least_60pct_per_fixture": best_by_fixture([row for row in markets if row["positive_profit_probability_at_least_60pct"]]),
              "best_positive_ev_at_least_60pct_per_fixture": best_by_fixture(qualified),
              "exclusions": exclusions, "superseded_event_snapshots": superseded,
              "date_time_anomalies": baseline["quote_metadata"].get("date_time_anomalies", []),
              "quote_file_metadata": file_metadata,
              "provenance": {"sha256": hashes, "original_predictions": str(Path(prediction_path).resolve()),
                             "original_cli_outputs": str(Path(prediction_path).resolve().with_name("cli_outputs.json")),
                             "original_cli_count": 40, "retrained": False, "forecasts_recomputed": False,
                             "original_inputs_unchanged": True, "original_forecast_timestamp_retained": True,
                             "ranking_uses_only_new_prices": True,
                             "current_price_scope": "Captured full-time total corner list; team-specific lists are excluded"}}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(render(report, helper), encoding="utf-8")
    for filename, title, rows in (("ALL_POSITIVE_LINES.md", "All positive-EV full-time total corner lines", positive),
                                  ("ALL_LINES.md", "All captured full-time total corner lines", report["markets"])):
        lines = ["# " + title, "", "COP25,000 per single alternative bet. Selections within a fixture are alternatives; their stakes are not compounded.", ""]
        for league, name in LEAGUES.items():
            lines += ["## " + name, ""] + table([row for row in rows if row["league_id"] == league], helper) + [""]
        (output / filename).write_text("\n".join(lines), encoding="utf-8")
    summary = {"captured_fixtures": len(latest), "priced_selections": len(markets), "positive_selections": len(positive),
               "fixtures_with_positive_ev": len(report["best_positive_ev_per_fixture"]),
               "fixtures_with_positive_ev_and_probability_at_least_60pct": len(report["best_positive_ev_at_least_60pct_per_fixture"]),
               "excluded_selections": len(exclusions), "complete": not missing, "output": str(output)}
    print(json.dumps(summary, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREVIOUS / "predictions.json")
    parser.add_argument("--quotes", nargs="+", default=[str(HERE / "quotes_*.json")], help="Quote files, glob patterns or a directory")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--stake-cop", type=int, default=25000)
    parser.add_argument("--require-all", action="store_true", help="Require detailed prices for every original fixture")
    args = parser.parse_args()
    if args.stake_cop <= 0:
        parser.error("--stake-cop must be positive")
    try:
        analyze(args.predictions, args.quotes, args.output, args.stake_cop, args.require_all)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
