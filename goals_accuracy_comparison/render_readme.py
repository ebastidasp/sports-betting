"""Render the all-division comparison README from already-computed metrics."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

MARKETS = (("over15", "Over 1.5"), ("over25", "Over 2.5"), ("over35", "Over 3.5"))
THRESHOLDS = ("0.6", "0.7", "0.8")
MODEL_LABELS = {"legacy": "Legacy Poisson", "v3": "V3 goal component"}
COUNTRY_LABELS = {"argentina": "Argentina", "chile": "Chile", "colombia": "Colombia",
                  "england": "England", "germany": "Germany", "iceland": "Iceland",
                  "ireland": "Ireland", "italy": "Italy", "south-korea": "South Korea",
                  "spain": "Spain", "usa": "United States"}
SECOND_LEAGUES = {"england": "Championship", "germany": "2. Bundesliga",
                  "italy": "Serie B", "spain": "Segunda División"}


def _count(value):
    return f"{int(value):,}" if value is not None else "—"


def _percent(value):
    return f"{float(value):.2%}" if value is not None else "—"


def _score(value):
    return f"{float(value):.6f}" if value is not None else "—"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    lines = ["| " + " | ".join(map(_cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(_cell, row)) + " |" for row in rows)
    return lines


def _filter(scope, market, model, threshold):
    return scope.get("markets", {}).get(market, {}).get(model, {}).get("filters", {}).get(threshold, {})


def _hits(item):
    if not item.get("selected", 0):
        return "— (0 selected)"
    return f"{_count(item.get('correct', 0))} / {_count(item.get('selected', 0))}"


def _hit_table(scope):
    rows = []
    for market, label in MARKETS:
        for threshold in THRESHOLDS:
            old = _filter(scope, market, "legacy", threshold)
            new = _filter(scope, market, "v3", threshold)
            old_accuracy, new_accuracy = old.get("accuracy"), new.get("accuracy")
            difference = (f"{100 * (new_accuracy - old_accuracy):+.2f} pp"
                          if old_accuracy is not None and new_accuracy is not None else "—")
            rows.append([label, f"≥{float(threshold):.0%}", _hits(old), _percent(old_accuracy),
                         _hits(new), _percent(new_accuracy), difference])
    return _table(["Goal market", "Minimum forecast", "Legacy correct / selected", "Legacy hit rate",
                   "V3 correct / selected", "V3 hit rate", "V3 − legacy"], rows)


def _interval(interval, probability=True):
    if interval is None:
        return "—"
    if probability:
        return f"{_percent(interval[0])}–{_percent(interval[1])}"
    return f"[{float(interval[0]):+.6f}, {float(interval[1]):+.6f}]"


def _division(scope, division):
    return scope.get("divisions", {}).get(division, {})


def _country_division(period, country, division):
    return _division(period.get("leagues", {}).get(country, {}), division)


def _available(period, country, division):
    return period.get("available_v3_leagues", {}).get(country, {}).get(division, 0)


def _exclusion_reason(report, country, division):
    reasons = {}
    for year in ("2025", "2026"):
        metadata = report.get("model_parameters", {}).get(year, {}).get("legacy", {}).get(country, {})
        period = metadata.get("period", {})
        reason = (period.get("second_division_exclusion_reason") if division == "second"
                  else period.get("first_division_exclusion_reason"))
        reason = reason or period.get("reason")
        if reason:
            reasons.setdefault(str(reason), []).append(year)
    return "; ".join((f"{', '.join(years)}: " if len(years) < 2 else "") + reason
                     for reason, years in reasons.items())


def _coverage_table(report):
    rows = []
    for country in report.get("countries", sorted(report.get("sources", {}))):
        ids = report.get("sources", {}).get(country, {}).get("league_ids", [])
        for index, division in enumerate(("first", "second")):
            league = ids[index] if len(ids) > index else None
            common = [_country_division(report["periods"][year], country, division).get("matched_fixtures", 0)
                      for year in ("2025", "2026")]
            available = [_available(report["periods"][year], country, division) for year in ("2025", "2026")]
            if division == "second" and league is None:
                note = "No second division configured in this cache."
            else:
                note = _exclusion_reason(report, country, division)
                if not note:
                    note = "Eligible in both models." if sum(common) else "No eligible common sample."
            rows.append([COUNTRY_LABELS.get(country, country), division.title(), _count(league),
                         _count(available[0]), _count(common[0]), _count(available[1]), _count(common[1]), note])
    return _table(["Country", "Division", "League ID", "V3 available 2025", "Common 2025",
                   "V3 available 2026", "Common 2026", "Coverage note"], rows)


def _probability_table(report):
    rows = []
    for period_name in ("2025", "2026", "combined"):
        for market, label in MARKETS:
            metrics = report["periods"][period_name].get("markets", {}).get(market, {})
            old, new = metrics.get("legacy", {}), metrics.get("v3", {})
            paired = metrics.get("paired_differences", {}).get("brier", {})
            rows.append([period_name.title(), label, _percent(old.get("actual_event_rate")),
                         _score(old.get("brier")), _score(new.get("brier")),
                         _score(old.get("log_loss")), _score(new.get("log_loss")),
                         _interval(paired.get("95pct_week_bootstrap"), False)])
    return _table(["Period", "Goal market", "Actual event frequency", "Legacy Brier", "V3 Brier",
                   "Legacy log loss", "V3 log loss", "95% interval: Brier difference (V3 − legacy)"], rows)


def _uncertainty_table(scope):
    rows = []
    for market, label in MARKETS:
        for threshold in THRESHOLDS:
            for model, model_label in MODEL_LABELS.items():
                item = _filter(scope, market, model, threshold)
                rows.append([label, f"≥{float(threshold):.0%}", model_label, _hits(item),
                             _percent(item.get("mean_probability")), _percent(item.get("coverage")),
                             _percent(item.get("accuracy")), _interval(item.get("95pct_wilson_interval"))])
    return _table(["Goal market", "Minimum forecast", "Model", "Correct / selected", "Mean forecast",
                   "Coverage of common fixtures", "Hit rate", "95% Wilson interval"], rows)


def _secondary_probability_table(report):
    rows = []
    for period_name in ("2025", "2026", "combined"):
        scope = _division(report["periods"][period_name], "second")
        if not scope.get("matched_fixtures", 0):
            continue
        for market, label in MARKETS:
            metrics = scope.get("markets", {}).get(market, {})
            old, new = metrics.get("legacy", {}), metrics.get("v3", {})
            paired = metrics.get("paired_differences", {}).get("brier", {})
            rows.append([period_name.title(), label, _score(old.get("brier")), _score(new.get("brier")),
                         _score(old.get("log_loss")), _score(new.get("log_loss")),
                         _interval(paired.get("95pct_week_bootstrap"), False)])
    return _table(["Period", "Second-division market", "Legacy Brier", "V3 Brier", "Legacy log loss",
                   "V3 log loss", "95% interval: Brier difference (V3 − legacy)"], rows)


def _insights(report):
    combined = report["periods"]["combined"]
    lines = []
    for market, label, threshold in (("over15", "Over 1.5", "0.8"), ("over25", "Over 2.5", "0.7")):
        old, new = (_filter(combined, market, model, threshold) for model in ("legacy", "v3"))
        lines.append(f"- **{label}, forecast ≥{float(threshold):.0%}:** legacy {_percent(old.get('accuracy'))} "
                     f"({_hits(old)}) versus V3 {_percent(new.get('accuracy'))} ({_hits(new)}). "
                     "These are different selected match sets; use the annual and division tables to check consistency.")
    old, new = (_filter(combined, "over35", model, "0.6") for model in ("legacy", "v3"))
    sentence = (f"- **Over 3.5:** even the ≥60% filter selects only {_count(old.get('selected', 0))} legacy "
                f"and {_count(new.get('selected', 0))} V3 matches across both years. "
                "Higher-cutoff results need to be read with their sample counts")
    if max(old.get("selected", 0), new.get("selected", 0)) < 100:
        sentence += "; the small samples do not establish reliable high-confidence performance."
    else:
        sentence += "."
    lines.append(sentence)
    secondary = _division(combined, "second")
    secondary_samples = []
    for year in ("2025", "2026"):
        for market, _ in MARKETS:
            item = _division(report["periods"][year], "second").get("markets", {}).get(market, {})
            old_brier, new_brier = item.get("legacy", {}).get("brier"), item.get("v3", {}).get("brier")
            if old_brier is not None and new_brier is not None:
                secondary_samples.append(new_brier - old_brier)
    if secondary_samples:
        favor_new = sum(value < -1e-12 for value in secondary_samples)
        favor_old = sum(value > 1e-12 for value in secondary_samples)
        lines.append(f"- **Second-division probability scores:** V3 has a lower Brier score in {favor_new} "
                     f"of {len(secondary_samples)} annual market comparisons; legacy is lower in {favor_old}. "
                     "Read the differences and paired intervals separately from the all-division results; "
                     "the aggregate results do not establish a secondary-division advantage.")
    empty_high = all(not _filter(secondary, market, model, "0.8").get("selected", 0)
                     for market in ("over25", "over35") for model in MODEL_LABELS)
    if secondary.get("matched_fixtures", 0) and empty_high:
        lines.append("- **Second-division high cutoffs:** neither model selects a match at ≥80% for over 2.5 "
                     "or over 3.5. Those hit rates are unavailable, not zero. The ≥70% over-2.5 subset also "
                     "needs its very small counts to be kept visible.")
    return lines


def readme(report):
    """Return Markdown; no file access, fitting, or mutation of ``report``."""
    periods = report["periods"]
    if not report.get("sources") or any("divisions" not in periods[period]
                                        for period in ("2025", "2026", "combined")):
        raise ValueError("The all-division README requires sources and division-scoped annual/combined metrics.")
    combined = periods["combined"]
    lines = ["# Goal totals accuracy: legacy Poisson versus V3 across configured divisions", "",
             "This report compares **football_poisson.py** with the **goal-rate component of football_model_v3.py** "
             "on the same historical fixtures. It includes configured first and second divisions wherever both models "
             "have eligible data, for over 1.5, 2.5 and 3.5 goals at forecast probabilities of at least 60%, 70% and 80%.", "",
             "The saved V3 models already trained on both configured divisions. This extension broadens the evaluated "
             "fixtures and reuses those annual models; it does not retrain or change V3. Its architecture and "
             "calibration method were selected on first-division validation. Secondary-division results evaluate "
             "that unchanged selected model, with no new tuning. Legacy fits for the four secondary leagues with "
             "statistics were regenerated strictly before each year; unchanged first-division cached forecasts "
             "were reused elsewhere.", "", "## Reading the tables", ""]
    lines += _table(["Goal market", "A correct prediction requires"],
                    [["Over 1.5", "At least 2 total FT goals"], ["Over 2.5", "At least 3 total FT goals"],
                     ["Over 3.5", "At least 4 total FT goals"]])
    lines += ["", "**Hit rate** is correct predictions divided by selected predictions. Cutoffs are inclusive and "
              "cumulative: ≥60% includes forecasts above 70% and 80%. A forecast cutoff is a filter, not a guaranteed hit rate.", "",
              "The underlying eligible fixtures are identical for both models. Each probability filter can select "
              "different fixtures, so hit-rate differences describe different selected groups. On a fixture selected "
              "by both models for the same over market, their predictions succeed or fail together. Full-fixture "
              "probability scores provide the paired comparison.", "",
              "A dash means no qualifying predictions and no measurable hit rate; it does not mean 0% accuracy.", ""]
    for year, title in (("2025", "2025: complete calendar year"),
                        ("2026", "2026: available fixtures through the September cutoff"),
                        ("combined", "Both periods combined")):
        period = periods[year]
        first, second = (_division(period, division).get("matched_fixtures", 0) for division in ("first", "second"))
        available = period.get("all_v3_fixtures")
        lines += [f"## {title}", "",
                  f"**{_count(period['matched_fixtures'])} common fixtures:** {_count(first)} first-division "
                  f"and {_count(second)} second-division matches. "
                  + (f"V3 could evaluate {_count(available)} fixtures before legacy eligibility restrictions." if available is not None else ""), ""]
        if year == "combined":
            lines += ["This pools the same annual forecasts from two fitted models. It is a descriptive summary, "
                      "not an additional independent test; fixture mixes and fitted parameters differ between years.", ""]
        else:
            lines += [f"Model coefficients were fitted using results available before 1 January {year}. "
                      + ("The 2026 sample is incomplete." if year == "2026" else "Calendar boundaries use UTC."), ""]
        lines += ["### All eligible divisions", ""] + _hit_table(period) + [""]
        for division, label in (("first", "First divisions"), ("second", "Second divisions")):
            scope = _division(period, division)
            lines += [f"### {label}", ""]
            if scope.get("matched_fixtures", 0):
                lines += [f"{_count(scope['matched_fixtures'])} common fixtures.", ""] + _hit_table(scope) + [""]
            else:
                lines += ["No eligible paired fixtures in this division scope.", ""]
    lines += ["## Second-division results by league, both periods combined", "",
              "These league tables isolate the secondary divisions with eligible data in both models. "
              "They use the same pooled annual forecasts as the tables above.", ""]
    paired_secondary_countries = []
    for country in report.get("countries", sorted(combined.get("leagues", {}))):
        scope = _country_division(combined, country, "second")
        if not scope.get("matched_fixtures", 0):
            continue
        paired_secondary_countries.append(country)
        league_id = scope.get("league_id")
        title = SECOND_LEAGUES.get(country, "Second division")
        lines += [f"### {COUNTRY_LABELS.get(country, country)}: {title} (league {_count(league_id)})", "",
                  f"{_count(scope['matched_fixtures'])} common fixtures across 2025 and the available 2026 period.", ""]
        lines += _hit_table(scope) + [""]
    if not paired_secondary_countries:
        lines += ["No secondary league has an eligible paired sample.", ""]
    lines += ["## Coverage by country and division", "",
              "“V3 available” counts the annual predictions before the common-fixture intersection. “Common” counts "
              "fixtures supported by both models. Missing legacy statistics, incomplete venue history, provisional "
              "ratings, or unavailable venue formulas reduce coverage. An absent sample is not a measured failure rate.", ""]
    configured_secondary = sum(len(source.get("league_ids", [])) > 1
                               for source in report.get("sources", {}).values())
    lines += [f"Eligible paired secondary fixtures cover **{len(paired_secondary_countries)} of "
              f"{configured_secondary} configured second-division leagues** in these caches.", ""]
    lines += _coverage_table(report) + ["",
              "Secondary divisions without an eligible paired sample remain listed here. Where no cached "
              "team-statistics coverage exists, the legacy model cannot support the comparison even when V3 can "
              "use recorded goals. Countries with only one configured league have no configured secondary sample.", "",
              "## Probability quality on all common fixtures", "",
              "Brier score and binary log loss assess every event probability, including forecasts below 60%. "
              "Lower values are better. These compare identical fixtures and complement selected-group hit rates.", ""]
    lines += _probability_table(report) + ["",
              "The paired intervals resample calendar weeks with replacement 5,000 times using seed 42. Negative "
              "Brier differences favor V3. An interval containing zero does not clearly distinguish the models on "
              "that metric. Weekly blocks do not remove all dependence, and the intervals do not adjust for the "
              "multiple markets, divisions, leagues and thresholds examined. Pooled intervals are descriptive "
              "summaries of the same annual tests.", "", "### Second-division probability quality", "",
              "These scores use all eligible secondary fixtures, including forecasts below the cutoffs. "
              "They should be examined independently of the all-division scores; adding these fixtures does "
              "not imply that V3 improves every secondary market.", ""]
    lines += _secondary_probability_table(report) + ["", "## Coverage and uncertainty in the pooled filters", "",
              "Read hit rates together with selected counts, mean forecasts and coverage. Very high hit rates from "
              "a few matches provide little evidence about future reliability. Approximate 95% Wilson intervals "
              "treat match outcomes as independent; fixtures share teams and leagues. They describe each selected "
              "group, not uncertainty in the difference between two models' hit rates.", ""]
    lines += _uncertainty_table(combined) + ["", "## Interpretation", ""]
    lines += _insights(report) + ["",
              "First- and second-division results should be assessed separately. Use the annual tables to check "
              "whether pooled patterns repeat; the combined sample should not conceal weaker performance in a "
              "particular year or league. These cached historical results do not establish future hit rates or "
              "betting profitability. The architecture was selected on 2024, not these later goal-market results.", "",
              "## Evaluation method", "",
              "- The baseline is the older [football_poisson.py](../football_poisson.py). The selected V2 classifier "
              "predicts home/draw/away and is not the baseline for these total-goal markets.",
              "- Legacy settings stay fixed: regularization 0.01, recency decay 0.95, EWMA alpha 0.25, minimum venue "
              "history 5, and Dixon–Coles enabled. Regenerated country/year fits use only eligible prior results from their "
              "configured divisions. Legacy rho is shared within that country/year fit, not fitted on the test division.",
              "- Legacy requires both teams' cached match statistics, sufficient historical venue games, completed "
              "rating calibration, and supported host-home and visitor-away formulas. Only completed FT matches "
              "are scored; newly appearing or insufficiently supported teams are excluded.",
              "- V3 uses its saved pre-2025 and pre-2026 evaluation bundles, already trained on all configured "
              "divisions. Positive goal-component weights are normalized to one; the 1X2 classifier is excluded "
              "from total-goal probabilities. Model architecture and calibration selection used first-division "
              "validation, and secondary fixtures are additional evaluations without tuning. The saved models "
              "and original forecast files remain unchanged.",
              "- Historical inputs update chronologically from completed results. Annual training boundaries "
              "require kickoff plus a three-hour result delay to precede fitting. The legacy chronological check "
              "rejects overlapping team fixtures less than three hours apart; coefficients stay fixed within each year.",
              "- The extension checks that reconstructed primary-division forecasts and legacy rho reproduce the "
              "original comparison within 1e-10 before adding secondary fixtures. Common fixtures have matching country/fixture "
              "IDs, kickoff times, and FT goals. Tables are grouped by the configured league ID, not a team's future division.",
              "- Saved goal means are reused to calculate all three tails. Dixon–Coles changes only 0–0, 0–1, 1–0 "
              "and 1–1. Thus `P(total ≥ 2) = PoissonSF(1, home_mean + away_mean) − exp(−home_mean − away_mean) × "
              "home_mean × away_mean × rho`. Over 2.5 and 3.5 use `PoissonSF(2, total_mean)` and "
              "`PoissonSF(3, total_mean)`. Rho is clipped to its valid per-fixture bounds. No market-specific "
              "calibration or threshold tuning is performed.", ""]
    cutoff = report.get("data_cutoff")
    if cutoff:
        try:
            timestamp = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
            local = timestamp.astimezone(timezone(timedelta(hours=-5)))
            local_text = local.strftime("%d %B %Y, %H:%M:%S") + " in Colombia"
        except (TypeError, ValueError):
            local_text = "the recorded source cutoff"
        lines += [f"The source cutoff is **{cutoff}**, equivalent to **{local_text}**. The latest common fixture "
                  f"in the 2026 sample kicked off at **{periods['2026'].get('latest_test_kickoff', '—')}**. "
                  "Evaluation calendar boundaries use UTC.", ""]
    lines += ["## Files and reproduction", "",
              "- [evaluation.json](evaluation.json): annual and combined metrics, division and league results, "
              "model cutoffs, rho metadata, coverage exclusions, uncertainty and input hashes.",
              "- [threshold_summary.csv](threshold_summary.csv): all requested cutoffs with correct/incorrect "
              "counts, hit rates, coverage and intervals; scope, division, country and league ID identify each group.",
              "- [predictions_2025.csv](predictions_2025.csv) and [predictions_2026.csv](predictions_2026.csv): "
              "common fixture probabilities and observed events across all three goal markets.",
              "- [backtest/](backtest/): the extended annual forecast caches and legacy country/year metadata.",
              "- The [original first-division-only over-2.5 comparison](../over25_comparison_v2_v3/README.md) "
              "and its source forecasts are preserved.", "", "From the project directory:", "", "```powershell",
              "python goals_accuracy_comparison/extend_backtest.py",
              "python goals_accuracy_comparison/generate_report.py", "```", "",
              "The first command reconstructs additional historical forecasts from local SQLite data and annual "
              "V3 models, refitting only the legacy annual models. The second command regenerates this report from "
              "the saved backtest inputs; it does not query the API, retrain models, or refresh the historical sample."]
    return "\n".join(lines) + "\n"
