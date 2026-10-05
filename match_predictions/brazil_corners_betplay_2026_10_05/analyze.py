"""Price captured full-time Brazil corner totals with the fresh original model.

Run from the repository root after quote_snapshot.json and the model exist:
    python match_predictions/brazil_corners_betplay_2026_10_05/analyze.py

This script reads local caches and captured prices. It never requests bookmaker
pages or submits bets. Generated files contain calculations, not bet slips.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import football_corners as corners

STATUSES = ("full_win", "half_win", "push", "half_loss", "full_loss")
LEAGUES = {71: "Brazil Serie A", 72: "Brazil Serie B"}


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"A timezone-aware timestamp is required: {value!r}")
    return result.tz_convert("UTC")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_from(value, quote_path):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root_path = ROOT / path
    return root_path.resolve() if root_path.exists() else (quote_path.parent / path).resolve()


def settlement_probabilities(distribution, side, line):
    """Five exact settlement categories, including the untruncated upper tail."""
    if side not in ("over", "under"):
        raise ValueError(f"Unsupported side: {side!r}")
    line = float(line)
    if not math.isfinite(line) or line < 0 or not math.isclose(line * 4, round(line * 4), abs_tol=1e-8):
        raise ValueError("Only nonnegative integer, half and quarter corner lines are supported")
    quarter = int(round(line * 4))
    k, fraction = divmod(quarter, 4)
    p = dict.fromkeys(STATUSES, 0.0)
    if fraction == 0:
        p["full_win"] = float(distribution.sf(k))
        p["push"] = float(distribution.pmf(k))
        p["full_loss"] = float(distribution.cdf(k - 1))
    elif fraction == 2:
        p["full_win"] = float(distribution.sf(k))
        p["full_loss"] = float(distribution.cdf(k))
    elif fraction == 1:
        p["full_win"] = float(distribution.sf(k))
        p["half_loss"] = float(distribution.pmf(k))
        p["full_loss"] = float(distribution.cdf(k - 1))
    else:
        p["full_win"] = float(distribution.sf(k + 1))
        p["half_win"] = float(distribution.pmf(k + 1))
        p["full_loss"] = float(distribution.cdf(k))
    if side == "under":
        p["full_win"], p["full_loss"] = p["full_loss"], p["full_win"]
        p["half_win"], p["half_loss"] = p["half_loss"], p["half_win"]
    if any(not math.isfinite(x) or x < -1e-12 for x in p.values()) or not math.isclose(sum(p.values()), 1, abs_tol=1e-10):
        raise ValueError(f"Invalid settlement probabilities: {p}")
    return p


def price_market(distribution, side, line, odds, stake):
    odds, stake = float(odds), int(stake)
    if not math.isfinite(odds) or odds <= 1 or stake <= 0:
        raise ValueError("Decimal odds must exceed 1 and the COP stake must be positive")
    probabilities = settlement_probabilities(distribution, side, line)
    winning = probabilities["full_win"] + probabilities["half_win"] / 2
    losing = probabilities["full_loss"] + probabilities["half_loss"] / 2
    refunded = probabilities["push"] + (probabilities["half_win"] + probabilities["half_loss"]) / 2
    ev = winning * (odds - 1) - losing
    gross = {"full_win": stake * odds, "half_win": stake * (odds + 1) / 2,
             "push": float(stake), "half_loss": stake / 2, "full_loss": 0.0}
    net = {key: value - stake for key, value in gross.items()}
    expected_gross = sum(probabilities[key] * gross[key] for key in STATUSES)
    if not math.isclose(expected_gross - stake, stake * ev, abs_tol=1e-7):
        raise AssertionError("Settlement EV disagrees with expected payout")
    positive_profit_probability = probabilities["full_win"] + probabilities["half_win"]
    quarter = int(round(float(line) * 4)) % 4 in (1, 3)
    return {"probabilities": probabilities,
            "positive_profit_probability": positive_profit_probability,
            "loss_probability": probabilities["full_loss"] + probabilities["half_loss"],
            "non_loss_probability": positive_profit_probability + probabilities["push"],
            "effective_winning_stake_probability": winning,
            "effective_losing_stake_probability": losing,
            "effective_refunded_stake_probability": refunded,
            "fair_odds": 1 + losing / winning if winning > 0 else None,
            "ev": ev, "positive_ev": ev > 0,
            "positive_profit_probability_at_least_60pct": positive_profit_probability >= .60,
            "stake_cop": stake, "expected_profit_cop": stake * ev,
            "expected_gross_payout_cop": expected_gross,
            "gross_payout_cop_by_settlement": gross, "net_profit_cop_by_settlement": net,
            "settlement_lines": [float(line) - .25, float(line) + .25] if quarter else [float(line)],
            "leg_stakes_cop": [stake / 2, stake / 2] if quarter else [stake]}


def self_test():
    """Boundary checks against independently enumerated discrete outcomes."""
    for alpha in (0, .08):
        distribution = corners.count_distribution(9.7, alpha)
        for line in (0, .25, .5, .75, 8, 8.25, 8.5, 8.75, 9, 10.5):
            for side in ("over", "under"):
                expected = dict.fromkeys(STATUSES, 0.0)
                legs = [line - .25, line + .25] if int(round(line * 4)) % 4 in (1, 3) else [line]
                # N=40 represents the complete N>=40 tail; all tested lines are below it.
                for n in range(41):
                    signs = [0 if n == leg else (1 if (n > leg) == (side == "over") else -1) for leg in legs]
                    fraction = sum(signs) / len(signs)
                    status = {1: "full_win", .5: "half_win", 0: "push", -.5: "half_loss", -1: "full_loss"}[fraction]
                    expected[status] += float(distribution.pmf(n) if n < 40 else distribution.sf(39))
                got = price_market(distribution, side, line, 1.91, 25000)
                for status in STATUSES:
                    assert math.isclose(got["probabilities"][status], expected[status], abs_tol=1e-10), (line, side, status)
                assert math.isclose(sum(got[f"effective_{word}_stake_probability"] for word in ("winning", "losing", "refunded")), 1, abs_tol=1e-10)
    print("Integer, half and quarter settlement checks passed.")


def cached_fixture(connection, fixture):
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT * FROM fixtures WHERE fixture_id=?", (int(fixture["fixture_id"]),)).fetchone()
    if row is None:
        raise ValueError("Fixture ID is absent from the Brazil cache")
    row = dict(row)
    for key in ("league_id", "home_id", "away_id"):
        if int(row[key]) != int(fixture[key]):
            raise ValueError(f"Captured fixture {key} differs from the cache")
    if timestamp(row["kickoff"]) != timestamp(fixture["kickoff"]):
        raise ValueError("Captured kickoff differs from the cache")
    return row


def predict_fixture(fixture, state, bundle, cutoff, known):
    league = int(fixture["league_id"])
    if league not in LEAGUES or league not in bundle["sources"]["brazil"]["league_ids"]:
        raise ValueError("Fixture league is unsupported by the trained Brazil model")
    if timestamp(fixture["kickoff"]) <= cutoff:
        raise ValueError("Fixture kickoff is not after prediction_as_of")
    home_id, home = corners.resolve_team(str(fixture["home_id"]), state, "brazil", league)
    away_id, away = corners.resolve_team(str(fixture["away_id"]), state, "brazil", league)
    if home_id == away_id:
        raise ValueError("Home and away team IDs must differ")
    primary = bundle["sources"]["brazil"]["league_ids"][0]
    inputs = [state.inputs("brazil", league, primary, home_id, away_id, "home", cutoff),
              state.inputs("brazil", league, primary, away_id, home_id, "away", cutoff)]
    counts = corners.predict_bundle(bundle, pd.DataFrame(inputs))
    if len(counts) != 2 or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Model returned invalid expected corner counts")
    means = {"home": float(counts[0]), "away": float(counts[1]), "total": float(counts.sum())}
    history = {}
    warnings = []
    for scope, tid, team in (("home", home_id, home), ("away", away_id, away)):
        games = known[(known.home_id == tid) | (known.away_id == tid)]
        mass = float(team["moments"][scope][20]["corners"][1])
        history[scope] = {"team_id": tid, "name": team["name"], "games": int(team["games"]),
                          "last_completed_kickoff": games.kickoff.max().isoformat() if len(games) else None,
                          "venue_corner_effective_mass_20": mass}
        if mass < 3:
            warnings.append(f"Sparse {scope} corner history: effective mass {mass:.3f}; estimates rely more on league priors")
    distributions = {scope: corners.count_distribution(mean, float(bundle["dispersion"][scope])) for scope, mean in means.items()}
    return {"fixture_id": int(fixture["fixture_id"]), "league_id": league,
            "home_id": home_id, "away_id": away_id, "home": home["name"], "away": away["name"],
            "captured_home": fixture["home"], "captured_away": fixture["away"],
            "kickoff": timestamp(fixture["kickoff"]).isoformat(), "means": means,
            "history": history, "warnings": warnings,
            "distribution": {scope: {"family": "poisson" if bundle["dispersion"][scope] < 1e-8 else "negative_binomial_nb2",
                                      "alpha": float(bundle["dispersion"][scope]),
                                      "central_90pct_interval": [int(distributions[scope].ppf(.05)), int(distributions[scope].ppf(.95))]}
                             for scope in means}}, distributions


def cli_proof(fixture, prediction, cutoff, model_path, db_path, line):
    command = [sys.executable, str(ROOT / "football_corners.py"), "predict", "--model", str(model_path),
               "--country", "brazil", "--league-id", str(fixture["league_id"]),
               "--home", str(fixture["home_id"]), "--away", str(fixture["away_id"]),
               "--db", str(db_path), "--as-of", cutoff.isoformat(), "--line", str(line)]
    encoding_environment = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="strict",
                            env={**os.environ, **encoding_environment}, timeout=300)
    record = {"fixture_id": int(fixture["fixture_id"]), "command": command, "line": line,
              "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
              "environment_overrides": encoding_environment}
    if result.returncode:
        raise ValueError(f"Original corner CLI failed: {result.stderr or result.stdout}")
    match = re.search(r"Expected corners: home ([\d.]+), away ([\d.]+), total ([\d.]+)", result.stdout)
    if match is None:
        raise ValueError("Could not find the original CLI expected-corners output")
    shown = list(map(float, match.groups()))
    calculated = [prediction["means"][scope] for scope in ("home", "away", "total")]
    if any(abs(a - b) > .00500001 for a, b in zip(shown, calculated)):
        raise ValueError("Full-precision means disagree with the original CLI")
    distribution = corners.count_distribution(prediction["means"]["total"], prediction["distribution"]["total"]["alpha"])
    probabilities = {">": float(distribution.sf(math.floor(line))), "<": float(distribution.cdf(math.ceil(line) - 1))}
    for symbol, probability in probabilities.items():
        match = re.search(r"P\(total corners " + re.escape(symbol) + r" [\d.]+\)\s*:\s*([\d.]+)%", result.stdout)
        if match is None or abs(float(match.group(1)) / 100 - probability) > .00005001:
            raise ValueError("Full-precision probabilities disagree with the original CLI")
    record["matches_within_display_rounding"] = True
    return record


def cop(value):
    return f"{Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP):,.0f}"


def market_name(row):
    scope = {"total": "Total", "home": row["home"], "away": row["away"]}[row["scope"]]
    return f"{scope}: {'Over' if row['side'] == 'over' else 'Under'} {row['line']:g}"


def render_report(report):
    provenance = report["provenance"]
    only_half_lines = all(math.isclose(row["line"] % 1, .5, abs_tol=1e-8) for row in report["markets"])
    settlement_text = (
        "All captured lines end in .5, so each bet either wins or loses with no push. Fair decimal odds are 1 / model win probability; EV is probability × offered decimal odds − 1. The ≥60% flag is informational."
        if only_half_lines else
        "Integer equality refunds the stake; quarter lines split it equally between the adjacent integer and half line. These settlement rules are assumptions unless the quote confirms them. Positive-profit probability includes full and half wins. Fair odds and EV account for refunded and partially lost stakes."
    )
    lines = ["# BetPlay Brazil corner price comparison", "",
             f"Created: {report['created_at']}. Historical results as of {report['prediction_as_of']}; model trained through {report['model_training_cutoff']}.", "",
             f"Reference stake: **COP {cop(report['stake_cop'])} per individual bet**. All positive-EV captured full-time totals are shown; there is no minimum probability filter. Prices are snapshots and require rechecking before use.", "",
             "The original corner model includes passes and pass accuracy. It predicts home and away means using historical finished matches available after a three-hour result delay. Total corners use their separately calibrated count dispersion, rather than assuming independent team counts.", "",
             settlement_text, "",
             "Only captured full-time corner totals are evaluated. First-half, handicap, race and other unsupported markets are excluded.", "",
             "The model's historical count validation is not a test of profit at these bookmaker prices. Market probabilities, especially sparse team history, remain estimates. Different corner selections from the same fixture are correlated; the best-per-fixture table is an alternative to choosing all positive selections.", "",
             "Forecasts use the same current historical state for all fixtures. Refresh forecasts and prices for later rounds after the intervening midweek matches finish; these estimates do not include those future results.", "",
             f"Coverage: {len(report['fixtures'])} predicted fixtures, {len(report['markets'])} priced selections, {len(report['positive_ev_markets'])} positive selections; {len(report['excluded_fixtures'])} excluded fixtures and {len(report['excluded_markets'])} excluded selections.", ""]
    brazil_test = report.get("historical_evaluation", {}).get("brazil", {}).get("model", {}).get("total")
    if brazil_test:
        lines += [f"The model's built-in 2025 Brazil test covers {brazil_test['observations']} Serie A matches, with total-corner MAE {brazil_test['mae']:.3f}. That test excludes Serie B. The count-distribution calibration uses the model's pooled validation dispersion; it is not a separate calibration for these upcoming Brazil markets.", ""]
    for league, name in LEAGUES.items():
        selected = [row for row in report["positive_ev_markets"] if row["league_id"] == league]
        best = [row for row in report["best_positive_ev_per_fixture"] if row["league_id"] == league]
        lines += [f"## {name} (league {league})", "",
                  "### Highest positive EV per fixture", ""]
        lines += market_table(best)
        lines += ["", f"One reference bet for each row would stake COP {cop(report['stake_cop'] * len(best))}. Expected profit is a model expectation, with no guarantee of a positive result.", "",
                  "### All positive-EV alternatives", ""]
        lines += market_table(selected)
        lines += [""]
    lines += ["## Settlement details", "",
              "The JSON contains full precision for every quoted selection, including negative EV. Gross payouts below include the returned stake. COP figures in this document are rounded only for display; bookmaker rounding is not modeled.", "",
              "| Fixture / selection | Full / half win | Push | Half / full loss | Gross payout: full win / half win / push / half loss / loss |", "|---|---:|---:|---:|---:|"]
    for row in report["positive_ev_markets"]:
        p, gross = row["probabilities"], row["gross_payout_cop_by_settlement"]
        lines.append(f"| {row['home']} – {row['away']}; {market_name(row)} | {p['full_win']:.2%} / {p['half_win']:.2%} | {p['push']:.2%} | {p['half_loss']:.2%} / {p['full_loss']:.2%} | " + " / ".join(cop(gross[key]) for key in STATUSES) + " |")
    lines += ["", "## Fixture forecasts and captured pages", "",
              "| League | Fixture | Kickoff UTC | Expected home / away / total | Quote observed | Source |", "|---|---|---|---:|---|---|"]
    for row in report["fixtures"]:
        means = row["means"]
        lines.append(f"| {row['league_id']} | {row['home']} – {row['away']} | {row['kickoff']} | {means['home']:.3f} / {means['away']:.3f} / {means['total']:.3f} | {row.get('observed_at', '—')} | [BetPlay]({row.get('source_url', '')}) |")
        for warning in row["warnings"]:
            lines += ["", f"{row['home']} – {row['away']}: {warning}."]
    anomalies = report["quote_metadata"].get("date_time_anomalies", report["quote_metadata"].get("anomalies", []))
    if anomalies:
        lines += ["", "### Captured schedule discrepancies", "",
                  "The forecast uses the cache kickoff; the quote snapshot retains the bookmaker display separately.", ""]
        for anomaly in anomalies:
            lines.append("- " + (anomaly if isinstance(anomaly, str) else json.dumps(anomaly, ensure_ascii=False)))
    lines += ["", "## Exclusions", ""]
    if not report["excluded_fixtures"] and not report["excluded_markets"]:
        lines.append("None among the captured quote records.")
    for row in report["excluded_fixtures"] + report["excluded_markets"]:
        lines.append(f"- Fixture {row.get('fixture_id')}: {row['reason']}.")
    lines += ["", "## Reproduction and provenance", "", "```powershell",
              "python match_predictions/brazil_corners_betplay_2026_10_05/analyze.py", "```", "",
              f"CLI verification: {report['cli_verification_count']} fixtures checked using the original predict command at the same fixed as-of timestamp. Full commands and stdout are in cli_outputs.json.", "",
              "Training source cache paths are recorded by the model. The hashes below freeze the files read for this analysis; they do not establish what those caches contained at training unless separate training provenance records are available.", "",
              "| Input | SHA-256 |", "|---|---|"]
    for path, sha in provenance["sha256"].items():
        lines.append(f"| {path} | `{sha}` |")
    return "\n".join(lines) + "\n"


def market_table(rows):
    lines = ["| Fixture | Selection | Mean | Profit probability | ≥60% | Offered / fair odds | EV | Expected profit COP |", "|---|---|---:|---:|:---:|---:|---:|---:|"]
    if not rows:
        lines.append("| — | No positive-EV captured selection | — | — | — | — | — | — |")
    for row in rows:
        fair = f"{row['fair_odds']:.3f}" if row["fair_odds"] is not None else "∞"
        lines.append(f"| {row['home']} – {row['away']} | {market_name(row)} | {row['mean']:.3f} | {row['positive_profit_probability']:.2%} | {'Yes' if row['positive_profit_probability_at_least_60pct'] else 'No'} | {row['odds']:.2f} / {fair} | {row['ev']:.2%} | {cop(row['expected_profit_cop'])} |")
    return lines


def analyze(quote_path, output, verify_cli=True, cli_workers=2):
    quote_path = Path(quote_path).resolve()
    initial_quote_hash = file_hash(quote_path)
    quotes = json.loads(quote_path.read_text(encoding="utf-8-sig"))
    model_path = path_from(quotes["model"], quote_path)
    cutoff = timestamp(quotes["prediction_as_of"])
    stake = int(quotes.get("stake_cop", 25000))
    if stake <= 0 or stake != quotes.get("stake_cop", 25000):
        raise ValueError("stake_cop must be a positive whole number")
    # Hash the artifact before loading, and every inference input before reading.
    initial_model_hash = file_hash(model_path)
    bundle = joblib.load(model_path)
    if bundle.get("kind") != corners.KIND or "brazil" not in bundle.get("sources", {}):
        raise ValueError("The supplied artifact is not a supported Brazil corner model")
    if not any("_passes_" in name for name in bundle["columns"]) or not any("_pass_accuracy_" in name for name in bundle["columns"]):
        raise ValueError("The supplied corner model does not include passes and pass accuracy")
    trained = timestamp(bundle["training_cutoff"])
    if cutoff < trained:
        raise ValueError("prediction_as_of must be at or after the model training cutoff")
    db_path = Path(bundle["sources"]["brazil"]["path"]).resolve()
    protected_paths = {quote_path, model_path, Path(__file__).resolve(), (ROOT / "football_corners.py").resolve(), db_path}
    raw_quote_path = quote_path.with_name("raw_quotes.json")
    if raw_quote_path.exists():
        protected_paths.add(raw_quote_path)
    browser_verification_path = quote_path.with_name("browser_quote_verification.json")
    if browser_verification_path.exists():
        protected_paths.add(browser_verification_path)
    evaluation_path = model_path.parent / "evaluation.json"
    if evaluation_path.exists():
        protected_paths.add(evaluation_path)
    for suffix in ("-wal",):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            protected_paths.add(candidate)
    hashes = {str(path): file_hash(path) for path in sorted(protected_paths, key=str)}
    if hashes[str(quote_path)] != initial_quote_hash:
        raise ValueError("The quote snapshot changed while being loaded")
    if hashes[str(model_path)] != initial_model_hash:
        raise ValueError("The model changed while being loaded")
    raw, sources = corners.read_databases([db_path])
    if set(sources) != {"brazil"}:
        raise ValueError("The Brazil source cache has a different country configuration")
    if sources["brazil"]["league_ids"] != bundle["sources"]["brazil"]["league_ids"]:
        raise ValueError("The cache and model league configuration differ")
    known = raw[raw.kickoff + corners.RESULT_DELAY <= cutoff].copy()
    _, state = corners.build_features(raw, as_of=cutoff)
    fixtures, markets, excluded_fixtures, excluded_markets, cli_outputs, cli_tasks = [], [], [], [], [], []
    seen_fixtures = set()
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection, threadpool_limits(limits=1):
        for captured in quotes["fixtures"]:
            fid = int(captured["fixture_id"])
            if fid in seen_fixtures:
                raise ValueError(f"Duplicate captured fixture {fid}")
            seen_fixtures.add(fid)
            try:
                cached_fixture(connection, captured)
                prediction, distributions = predict_fixture(captured, state, bundle, cutoff, known)
            except (ValueError, KeyError) as error:
                excluded_fixtures.append({"fixture_id": fid, "reason": str(error), "captured_fixture": captured})
                continue
            prediction.update({"source_url": captured.get("source_url"), "observed_at": captured.get("observed_at"),
                               "captured_fixture": {key: value for key, value in captured.items() if key != "markets"}})
            fixtures.append(prediction)
            valid_lines = []
            seen_markets = set()
            for index, quote in enumerate(captured.get("markets", [])):
                try:
                    if quote.get("period") != "full_time":
                        raise ValueError("Only full-time corner markets are modeled")
                    if quote.get("scope") not in distributions:
                        raise ValueError("Only total, home-team and away-team corner totals are modeled")
                    if quote.get("kind", "total") != "total":
                        raise ValueError("Corner handicap, race and other non-total markets are unsupported")
                    if quote.get("settlement_rule", "asian_push") != "asian_push":
                        raise ValueError("Unsupported bookmaker settlement rule")
                    scope, side = quote["scope"], quote["side"]
                    line, odds = float(quote["line"]), float(quote["odds"])
                    identity = (scope, side, line, odds)
                    if identity in seen_markets:
                        raise ValueError("Duplicate captured selection")
                    seen_markets.add(identity)
                    pricing = price_market(distributions[scope], side, line, odds, stake)
                    market = {"fixture_id": fid, "league_id": prediction["league_id"],
                              "home": prediction["home"], "away": prediction["away"], "kickoff": prediction["kickoff"],
                              "market": quote.get("market"), "scope": scope, "side": side, "line": line, "odds": odds,
                              "period": "full_time", "mean": prediction["means"][scope],
                              "source_url": quote.get("source_url", captured.get("source_url")),
                              "observed_at": quote.get("observed_at", captured.get("observed_at")),
                              "settlement_rule": "asian_push", "settlement_rule_confirmed": quote.get("settlement_rule_confirmed", False),
                              **pricing}
                    if market["observed_at"] is not None:
                        timestamp(market["observed_at"])
                    markets.append(market)
                    if scope == "total" and math.isclose(line % 1, .5, abs_tol=1e-8):
                        valid_lines.append(line)
                except (ValueError, KeyError, TypeError) as error:
                    excluded_markets.append({"fixture_id": fid, "market_index": index, "reason": str(error), "quote": quote})
            if verify_cli:
                cli_tasks.append((captured, prediction, cutoff, model_path, db_path, valid_lines[0] if valid_lines else 9.5))
    if cli_tasks:
        with ThreadPoolExecutor(max_workers=max(1, min(int(cli_workers), 4))) as pool:
            futures = [pool.submit(cli_proof, *task) for task in cli_tasks]
            for future in as_completed(futures):
                cli_outputs.append(future.result())
                print(f"Original CLI checked {len(cli_outputs)}/{len(cli_tasks)} fixtures", flush=True)
        cli_outputs.sort(key=lambda row: row["fixture_id"])
    changed = [path for path, sha in hashes.items() if not Path(path).is_file() or file_hash(path) != sha]
    appeared = [str(Path(str(db_path) + suffix)) for suffix in ("-wal",)
                if Path(str(db_path) + suffix).exists() and str(Path(str(db_path) + suffix)) not in hashes]
    if changed or appeared:
        raise ValueError(f"Inference inputs changed during analysis; rerun on a stable snapshot: {changed + appeared}")
    positive = sorted((row for row in markets if row["positive_ev"]), key=lambda row: (-row["ev"], row["fixture_id"], row["scope"], row["side"], row["line"]))
    best, seen = [], set()
    for row in positive:
        if row["fixture_id"] not in seen:
            best.append(row)
            seen.add(row["fixture_id"])
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "prediction_as_of": cutoff.isoformat(),
              "model_training_cutoff": trained.isoformat(), "stake_cop": stake,
              "model": str(model_path), "model_selection": bundle["selection"], "dispersion": bundle["dispersion"],
              "historical_evaluation": {"brazil": bundle.get("report", {}).get("countries", {}).get("brazil", {}),
                                        "split": bundle.get("report", {}).get("split", {}),
                                        "test_scope": "primary leagues only; Brazil Serie A, excluding Serie B"},
              "quote_metadata": {key: value for key, value in quotes.items() if key != "fixtures"},
              "probability_filter": None, "sixty_percent_flag_definition": "P(full win or half win), not P(non-loss)",
              "fixtures": fixtures, "markets": markets, "positive_ev_markets": positive,
              "best_positive_ev_per_fixture": best,
              "positive_ev_by_league": {str(league): [row for row in positive if row["league_id"] == league] for league in LEAGUES},
              "excluded_fixtures": excluded_fixtures, "excluded_markets": excluded_markets,
              "cli_verification_count": len(cli_outputs),
              "provenance": {"sha256": hashes, "sources": bundle["sources"],
                             "inference_database": str(db_path), "training_database_hashes_available": False,
                             "historical_finished_matches": int(len(known)),
                             "historical_matches_by_league": {str(int(key)): int(value) for key, value in known.groupby("league_id").size().items()},
                             "latest_completed_kickoff_by_league": {str(int(key)): value.isoformat() for key, value in known.groupby("league_id").kickoff.max().items()},
                             "result_availability_hours": 3, "input_feature_time": "prediction_as_of, matching the original CLI",
                             "inputs_unchanged_after_analysis": True}}
    if browser_verification_path.exists():
        report["quote_refresh_verification"] = json.loads(browser_verification_path.read_text(encoding="utf-8-sig"))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "cli_outputs.json").write_text(json.dumps(cli_outputs, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"fixtures": len(fixtures), "markets": len(markets), "positive_ev": len(positive),
                      "best_per_fixture": len(best), "excluded_fixtures": len(excluded_fixtures),
                      "excluded_markets": len(excluded_markets), "cli_verified": len(cli_outputs), "output": str(output)}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", type=Path, default=Path(__file__).with_name("quote_snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--skip-cli", action="store_true", help="Skip CLI reproduction; report records zero checks")
    parser.add_argument("--cli-workers", type=int, default=2, help="Parallel original CLI checks (1 to 4)")
    parser.add_argument("--self-test", action="store_true", help="Check settlement math without reading quotes or producing reports")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        analyze(args.quotes, args.output, verify_cli=not args.skip_cli, cli_workers=args.cli_workers)


if __name__ == "__main__":
    main()
