"""Render the 1X2 accuracy comparison from evaluated historical forecasts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

PERIODS = ("2025", "2026", "combined")
SCOPES = ("all", "first", "second")
THRESHOLDS = ("0.6", "0.7", "0.8")
LABELS = {"legacy": "Original Poisson", "v2": "V2 classifier", "v3": "V3"}
COUNTRIES = {"argentina": "Argentina", "chile": "Chile", "colombia": "Colombia",
             "england": "England", "germany": "Germany", "iceland": "Iceland", "ireland": "Ireland",
             "italy": "Italy", "south-korea": "South Korea", "spain": "Spain", "usa": "United States"}
DIVISIONS = {"all": "Both divisions", "first": "First division", "second": "Second division"}


def pct(value):
    return f"{value:.2%}" if value is not None else "—"


def pp(value):
    return f"{value * 100:+.2f} pp" if value is not None else "—"


def interval(value, percent=False):
    if value is None:
        return "—"
    return f"[{value[0] * 100:+.2f}, {value[1] * 100:+.2f}] pp" if percent else f"[{value[0]:+.6f}, {value[1]:+.6f}]"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]]


def results_text(row, filtered=False):
    n = row["selected"] if filtered else row["matches"]
    if not n:
        return "— (0 selected)"
    return f"{row['correct']:,} / {n:,} ({pct(row['accuracy'])})"


def overall_table(report, population, baseline):
    rows = []
    for period in PERIODS:
        for scope in SCOPES:
            item = report["periods"][period][population][scope]
            old, new = item["models"][baseline], item["models"]["v3"]
            paired = item["paired_differences"][baseline]["accuracy"]
            rows.append([period.title(), DIVISIONS[scope], results_text(old), results_text(new),
                         pp(new["accuracy"] - old["accuracy"]), interval(paired["95pct_week_bootstrap"], True)])
    return table(["Period", "Division", f"{LABELS[baseline]} correct / games (accuracy)",
                  "V3 correct / games (accuracy)", "V3 change", "95% paired interval for change"], rows)


def confidence_table(report, population, baseline):
    rows = []
    for period in PERIODS:
        for scope in SCOPES:
            item = report["periods"][period][population][scope]
            for threshold in THRESHOLDS:
                old, new = [item["models"][model]["confidence_filters"][threshold] for model in (baseline, "v3")]
                delta = new["accuracy"] - old["accuracy"] if old["selected"] and new["selected"] else None
                rows.append([period.title(), DIVISIONS[scope], f"≥{float(threshold):.0%}",
                             results_text(old, True), results_text(new, True), pp(delta)])
    return table(["Period", "Division", "Minimum confidence", f"{LABELS[baseline]} correct / selected (accuracy)",
                  "V3 correct / selected (accuracy)", "V3 change"], rows)


def quality_table(report, population, baseline):
    rows = []
    for period in PERIODS:
        for scope in SCOPES:
            item = report["periods"][period][population][scope]
            old, new = item["models"][baseline], item["models"]["v3"]
            ci = item["paired_differences"][baseline]["log_loss"]["95pct_week_bootstrap"]
            rows.append([period.title(), DIVISIONS[scope], f"{old['log_loss']:.6f}", f"{new['log_loss']:.6f}",
                         f"{old['brier']:.6f}", f"{new['brier']:.6f}", interval(ci)])
    return table(["Period", "Division", f"{LABELS[baseline]} log loss", "V3 log loss",
                  f"{LABELS[baseline]} Brier", "V3 Brier", "95% interval for log-loss change (V3 − baseline)"], rows)


def league_table(report, population, model_names):
    league_data = report["periods"]["combined"][population]["leagues"]
    rows = []
    for country, source in sorted(report["sources"].items()):
        for index, league_id in enumerate(source["league_ids"]):
            item = league_data.get(f"{country}:{league_id}")
            if item:
                values = [pct(item["models"][model]["accuracy"]) for model in model_names]
                count = f"{item['matches']:,}"
            else:
                values, count = ["—"] * len(model_names), "0"
            rows.append([COUNTRIES.get(country, country), "First" if index == 0 else "Second", league_id, count, *values])
    return table(["Country", "Division", "League ID", "Common games", *[LABELS[model] for model in model_names]], rows)


def draw_table(report, population, model_names):
    rows = []
    for scope in SCOPES:
        item = report["periods"]["combined"][population][scope]
        for model in model_names:
            metrics = item["models"][model]
            draws = metrics["outcomes"]["draw"]
            wins = metrics["winner_picks_only"]
            rows.append([DIVISIONS[scope], LABELS[model], f"{draws['actual']:,} / {metrics['matches']:,}",
                         f"{draws['predicted']:,}", f"{draws['correct']:,}", results_text(wins, True)])
    return table(["Division", "Model", "Actual draws / games", "Draw picks", "Correct draw picks",
                  "Home/away picks: correct / selected (accuracy)"], rows)


def findings(report, population, baseline):
    lines = []
    for scope in SCOPES:
        item = report["periods"]["combined"][population][scope]
        old, new = item["models"][baseline], item["models"]["v3"]
        paired = item["paired_differences"][baseline]["accuracy"]
        ci = paired["95pct_week_bootstrap"]
        evidence = "The interval includes zero." if ci[0] <= 0 <= ci[1] else "The interval favors V3." if ci[0] > 0 else f"The interval favors {LABELS[baseline]}."
        lines.append(f"- **{DIVISIONS[scope]}:** {LABELS[baseline]} {pct(old['accuracy'])}, V3 {pct(new['accuracy'])}; "
                     f"change {pp(new['accuracy'] - old['accuracy'])}. {evidence}")
    return lines


def readme(report):
    common = report["periods"]["combined"]["legacy_common"]
    full = report["periods"]["combined"]["full_v2_v3"]
    local_cutoff = datetime.fromisoformat(report["data_cutoff"]).astimezone(timezone(timedelta(hours=-5)))
    lines = ["# Match-winner accuracy across first and second divisions", "",
        "For the latest **winner-only ≥60%, ≥70% and ≥80% comparison separately for every first- and second-division league**, see [Winner thresholds by league](WINNER_THRESHOLDS_BY_LEAGUE.md). That report uses the full refreshed V2/V3 population.", "",
        "For the same thresholds against **V1 (the original Poisson model)**, see [Fresh V1 versus V3 league comparison](v1_v3/README.md). That report refits V1 on current SQLite history and compares both models on each league's matched fixtures.", "",
        "This report compares V3's full home/draw/away predictions with both earlier models: the original **football_poisson.py** and **football_model_v2.py**. It evaluates completed historical fixtures from the configured first and second divisions, with coefficients fitted strictly before each evaluation year.", "",
        "## Definitions and populations", "",
        "The predicted result is the outcome with the highest probability: **home win, draw or away win (1X2)**. Draws remain in the sample. A home/away winner prediction that ends in a draw is incorrect; a predicted draw is correct only when the match is drawn. A separate table below isolates home/away picks while retaining actual draws as failures.", "",
        "Confidence is the probability of the chosen outcome, not the sum of both teams' win probabilities. The ≥60%, ≥70% and ≥80% filters are inclusive and cumulative. Models can select different fixtures at the same threshold, so filtered hit-rate differences describe different selected groups. The overall accuracy and probability-score comparisons use identical fixtures.", "",
        f"- **Original Poisson versus V3:** {common['all']['matches']:,} common fixtures, including {common['second']['matches']:,} second-division matches. Eligible secondary comparisons cover England (40), Germany (79), Italy (136) and Spain (141), where the original model has sufficient cached statistics and history.",
        f"- **V2 versus V3:** {full['all']['matches']:,} fixtures, including {full['second']['matches']:,} second-division matches. Both classifiers can use fixture history even when the original Poisson model lacks match-statistics coverage.", "",
        "These populations differ. Do not compare an accuracy from one population directly with a number from the other. Within each comparison, both models score exactly the same fixtures. The 2025 period covers the stored full calendar year; 2026 is incomplete. The full V2/V3 population uses the current saved training cutoff. Original Poisson means and eligibility retain the earlier goal-backtest snapshot, so that comparison does not include newly available fixtures or refresh the original model's historical fits.", "",
        "## Original Poisson versus V3: overall accuracy", ""]
    lines += overall_table(report, "legacy_common", "legacy")
    lines += ["", "### Original Poisson versus V3: confidence cutoffs", ""]
    lines += confidence_table(report, "legacy_common", "legacy")
    lines += ["", "### Original Poisson versus V3: probability quality", "",
        "Lower multiclass log loss and Brier score are better. Brier is the average sum of squared errors across the three outcomes. These scores use all common fixtures, including predictions below 60% confidence.", ""]
    lines += quality_table(report, "legacy_common", "legacy")
    lines += ["", "### All three models on the original model's common fixtures, pooled", "",
        "All accuracy columns in this table use the same league-specific fixtures. The V2 column provides an additional baseline on the original model's smaller eligible population.", ""]
    lines += league_table(report, "legacy_common", ("legacy", "v2", "v3"))
    lines += ["", "### Original-model comparison: interpretation", ""]
    lines += findings(report, "legacy_common", "legacy")
    lines += ["", "## V2 classifier versus V3: full available population", "",
        "This comparison includes secondary fixtures unavailable to the original Poisson model. V2 is evaluated with its previously selected fixed settings; no new architecture or hyperparameters were selected using the 2025/2026 results.", ""]
    lines += overall_table(report, "full_v2_v3", "v2")
    lines += ["", "### V2 versus V3: confidence cutoffs", ""]
    lines += confidence_table(report, "full_v2_v3", "v2")
    lines += ["", "### V2 versus V3: probability quality", ""]
    lines += quality_table(report, "full_v2_v3", "v2")
    lines += ["", "### V2 versus V3 by league, pooled", ""]
    lines += league_table(report, "full_v2_v3", ("v2", "v3"))
    lines += ["", "### V2 comparison: interpretation", ""]
    lines += findings(report, "full_v2_v3", "v2")
    lines += ["", "## Draws and winner-only selections", "",
        "The following pooled tables distinguish 1X2 accuracy from precision among home/away picks. The winner-only selections exclude predicted draws, while games that actually end in a draw remain counted as incorrect winner picks. Each model can select different games, so these conditional rates are not a paired accuracy gain.", "",
        "### Common fixtures with the original model", ""]
    lines += draw_table(report, "legacy_common", ("legacy", "v2", "v3"))
    lines += ["", "### Full V2/V3 population", ""]
    lines += draw_table(report, "full_v2_v3", ("v2", "v3"))
    lines += ["", "## Evaluation details and uncertainty", "",
        "- V3 uses the **full saved annual prediction bundle**, including its goal and classifier components and saved calibration parameters. Its 1X2 blend is 75% goal probabilities and 25% recent-data boosting. This is different from the goal-total report, which uses only the goal component.",
        "- The original Poisson 1X2 forecasts are reconstructed through its own `outcome_probabilities` function from saved chronological home/away means and country/year Dixon–Coles rho. Its original probability clipping and normalization remain in effect. Those means were fitted using prior results from both configured divisions.",
        "- V2 is refitted in memory before each test year using its previously selected classifier and the same historical feature columns. Existing saved model files are not overwritten. Both V2 and V3 train on both configured divisions; their architecture selection used first-division validation.",
        "- SQLite caches are opened read-only. Only completed FT fixtures with known goals enter the analysis. Results are assumed available three hours after kickoff. Historical inputs update from earlier completed matches, while model coefficients remain fixed within each annual test.",
        "- Reconstructed V2 and V3 first-division probabilities must match their original annual forecast CSVs within 1e-10. Legacy eligibility and rates come from the extended goal backtest, which already reproduced its original first-division forecasts and checked chronology. Fixture IDs, kickoff dates, league IDs and actual goals must agree between compared inputs.",
        "- Paired 95% intervals resample calendar weeks 5,000 times with replacement (seed 42). A positive accuracy difference favors V3; a negative probability-score difference favors V3. Intervals containing zero do not clearly distinguish the models. Weekly blocking does not remove all dependence across fixtures, and intervals are not adjusted for the many thresholds, leagues and comparisons examined.",
        "- Selected counts and coverage matter. High-confidence results based on a few matches can be unstable. Zero selected matches have unavailable accuracy; they are not 0% accuracy. Wilson intervals for selected-group precision, mean confidence, outcome counts and confusion matrices are saved in the JSON report.",
        "- Pooled results summarize the same annual forecasts; they are not another independent test. Read yearly and division-specific results before interpreting an apparent overall gain. Missing original-model statistics reduce its comparable coverage; they do not demonstrate poor predictive accuracy in those excluded leagues.", "",
        f"The refreshed V2/V3 historical source cutoff is **{report['data_cutoff']}** ({local_cutoff:%d %B %Y, %H:%M:%S} in Colombia). The current all-data model is not used to predict historical test matches. These observed results do not guarantee future accuracy.", "",
        "## Files and reproduction", "",
        "- [evaluation.json](evaluation.json): overall, division and league metrics, confidence filters, draw statistics, confusion matrices and paired uncertainty.",
        "- [threshold_summary.csv](threshold_summary.csv): all confidence thresholds by population, year, division and league, with correct/incorrect counts, coverage and intervals.",
        "- [all_predictions_2025.csv](all_predictions_2025.csv) and [all_predictions_2026.csv](all_predictions_2026.csv): full V2/V3 fixture predictions across both divisions.",
        "- [common_predictions_2025.csv](common_predictions_2025.csv) and [common_predictions_2026.csv](common_predictions_2026.csv): matched fixtures with all three model probabilities.",
        "- [forecast_metadata.json](forecast_metadata.json) and [protected_input_sha256.json](protected_input_sha256.json): annual model settings, training cutoffs, legacy metadata and protected input hashes.",
        "- [Goal-total comparison](../goals_accuracy_comparison/README.md): the separate over-1.5/2.5/3.5 analysis and original-model coverage details.", "",
        "From the project directory, run the full historical reconstruction and report:", "",
        "```powershell", "python winner_accuracy_comparison/compare_models.py", "```", "",
        "To regenerate the report from saved predictions:", "",
        "```powershell", "python winner_accuracy_comparison/compare_models.py --mode report", "```", "",
        "The full command fits only an in-memory V2 reference; it does not retrain V3, overwrite the original Poisson model, or call an API."]
    return "\n".join(lines) + "\n"
