# Separate match-outcome model v3

Created and trained on 2026-09-30. This version shows **small observed gains over the existing SQLite model, without statistically conclusive evidence of an advantage**. All existing scripts and saved models remain unchanged.

The entry point is `football_model_v3.py`. It reads local SQLite fixtures and historical statistics through the existing v2's read-only helpers, and saves its own fitted model and reports here. No network requests or new data sources are used.

## Selected model

The prediction is a weighted average of three-outcome probabilities:

- **75% richer goal-rate model:** two regularized Poisson regressions predict home and away goal rates from the 167 existing historical features. A Dixon–Coles correction adjusts low-scoring outcomes, using rho = -0.049093 fitted on held-out 2023 matches.
- **25% recent-data classifier:** the existing small gradient-boosting architecture is trained with a 730-day half-life for training-match weights. Matches from two years earlier receive half the raw weight of current matches; weights are normalized to mean one.

The inputs include historical goals and points, opponent-adjusted scoring residuals, Elo, rest, shots, shots on target, inside-box shots, possession, corners, xG and attacking/defensive home-away form. Windows use five- and twenty-match half-lives. The goal-rate models use substantially richer inputs than the earlier simple league-and-Elo Poisson reference.

Additional league draw rates, scoring patterns and xG/shot residual features were tested, giving 205 candidate features in total. The selected components use the original 167 inputs. Temperature and class-bias calibration were also tested using 2023 calibration predictions and 2024 validation; neither improved the selected blend, so the delivered model uses its raw blended probabilities.

## Chronological comparison

The baseline is **`football_model_v2.py`**, refitted with its existing selected configuration on the same prior data. This comparison uses every eligible primary-league fixture, including fixtures without match statistics. Its population differs from the smaller common-fixture comparison with the legacy `football_poisson.py` in the previous README.

Settings were selected on 3,565 calendar-year 2024 matches after training on results available before 2024. The validation blend reached 49.59% accuracy and 1.01643 log loss, versus 49.23% and 1.02236 for the current classifier. `frozen_selection.json` records the settings before either later evaluation. No architecture or mixture weight was changed after viewing 2025 or 2026 results.

Each evaluation fit uses only results available before its year. Historical features update as earlier matches finish, with results assumed available three hours after kickoff. Current, simultaneous and unfinished match results cannot enter a prediction. The final future-prediction model is refitted on all 50,800 available FT fixtures after evaluation.

| Metric | V2 2025 | V3 2025 | V2 2026 | V3 2026 |
|---|---:|---:|---:|---:|
| Evaluated matches | 3,725 | 3,725 | 2,599 | 2,599 |
| Correct outcomes | 1,852 | 1,855 | 1,247 | 1,253 |
| Accuracy | 49.72% | 49.80% | 47.98% | 48.21% |
| Log loss | 1.01914 | 1.01724 | 1.02853 | 1.02666 |
| Multiclass Brier score | 0.61024 | 0.60922 | 0.61757 | 0.61587 |

Lower log loss and Brier are better. The accuracy gains are **0.08 and 0.23 percentage points**, equivalent to three and six extra correct outcomes. Weekly paired bootstrap 95% intervals for the gains are approximately [-0.63, +0.75] and [-0.49, +0.93] percentage points. The probability-score intervals also include zero. These samples therefore do not establish a reliable overall improvement.

2025 had already been inspected during previous model work. 2026 was evaluated after freezing this version's settings. It is a partial year: the training cutoff is `2026-09-30T23:24:01.792716+00:00`, and the latest evaluated fixture in the caches kicked off on September 27. Ireland has no eligible 2026 observations.

## Predictions with minimum confidence

Confidence means the highest of the home/draw/away probabilities. Accuracy is the fraction of selected predictions whose most likely outcome occurred. Each model selects its own matches.

| Period | Minimum confidence | V2 correct / selected | V2 accuracy | V3 correct / selected | V3 accuracy |
|---|---:|---:|---:|---:|---:|
| 2025 | 60% | 523 / 783 | 66.79% | 400 / 573 | 69.81% |
| 2025 | 70% | 192 / 254 | 75.59% | 141 / 171 | 82.46% |
| 2025 | 80% | 26 / 30 | 86.67% | 35 / 38 | 92.11% |
| 2026 | 60% | 335 / 518 | 64.67% | 271 / 409 | 66.26% |
| 2026 | 70% | 138 / 185 | 74.59% | 102 / 137 | 74.45% |
| 2026 | 80% | 36 / 42 | 85.71% | 26 / 30 | 86.67% |

At 60%, v3 covers 15.38% of matches in 2025 and 15.74% in 2026, versus v2's 21.02% and 19.93%. Its higher hit rate comes with fewer selections. The 70% filter does not improve accuracy in 2026, and the 80% samples are small. These filtered comparisons are not paired comparisons on an identical selected population.

## League accuracy

| League/country | 2025 matches | V2 | V3 | 2026 matches | V2 | V3 |
|---|---:|---:|---:|---:|---:|---:|
| England | 378 | 54.23% | 54.50% | 240 | 44.17% | 45.00% |
| Spain | 370 | 55.95% | 55.95% | 278 | 49.28% | 50.36% |
| Italy | 368 | 52.72% | 52.17% | 264 | 55.30% | 54.92% |
| Germany | 308 | 49.68% | 50.00% | 208 | 55.29% | 55.29% |
| Argentina | 503 | 44.14% | 42.74% | 400 | 45.25% | 46.50% |
| Colombia | 450 | 50.00% | 50.22% | 305 | 48.52% | 48.85% |
| Ireland | 181 | 47.51% | 48.62% | 0 | — | — |
| Chile | 240 | 49.17% | 52.50% | 183 | 46.99% | 49.18% |
| South Korea | 232 | 43.10% | 43.10% | 179 | 36.31% | 35.20% |
| USA / MLS | 533 | 48.59% | 48.78% | 402 | 49.00% | 47.26% |
| Iceland | 162 | 51.23% | 50.00% | 140 | 47.14% | 47.86% |

The gain is not uniform. Accuracy improves in six leagues in each period and declines in three. The JSON report also includes probability scores and confidence-filter results per league.

Draws remain a limitation when selecting the most likely outcome. V3 picks only 21 draws in 2025 and nine in 2026, correctly identifying fewer than 1% of actual draws. Its mean draw probability is nevertheless 26.13% and 25.63%, compared with actual draw frequencies of 26.07% and 26.86%. A draw can have a substantial probability while a home or away win still has the highest individual probability.

## Commands

Run from the project directory:

```powershell
python football_model_v3.py train
python football_model_v3.py predict --country colombia --home Millonarios --away "Santa Fe"
python football_model_v3.py predict --country england --home Arsenal --away Chelsea
```

`train` selects candidates, runs both chronological backtests and fits the final model, saving everything separately to `new_model_v3/`. It accepts `--db`, `--output`, `--validation-year`, `--test-year`, `--confirm-year`, `--selection-metric` and a timezone-aware `--as-of`. Validation years earlier than the baseline's 2024 selection are refused to avoid using a future-selected reference in an earlier evaluation. Output inside existing model folders is refused.

`predict` supports names or team IDs, `--league-id`, `--db`, `--model` and `--as-of`. A prediction date must not precede its model's training cutoff. The unchanged original command continues to use the original model; invoking the v3 command selects the new version explicitly.

The prediction output includes projected home, away and total goals (model xG) from the selected goal-rate component, plus fair decimal odds for each outcome. When several goal components are selected, their means are averaged with weights normalized across those components. A classifier-only selection displays expected goals as unavailable. Fair odds are `1 / probability`, using the final blended and calibrated outcome probabilities without a bookmaker margin. These display fields use the existing trained bundle and do not alter its predictions or require retraining.

`predict` also prints probabilities and fair odds for **at least 2, 3 and 4 total goals**, equivalent to over 1.5, 2.5 and 3.5 goals. These probabilities come from the goal-rate components, with their mixture weights normalized to one. The probability of at least two goals includes the Dixon–Coles adjustment to the 1–1 scoreline. The adjustment leaves probabilities of at least three or four goals unchanged. Fair decimal odds for each total-goal event are the reciprocal of that event's probability. No retraining is needed to display these markets.

For a selected calibrated candidate on a future retraining run, the calibration method stays fixed after validation. Its parameters refresh only after scoring a completed held-out year and apply to subsequent predictions. The delivered model selected `none`, so all calibration parameters are zero.

## Files and verification

- `football_model.joblib`: final v3 bundle for future predictions.
- `evaluation_model_2025.joblib` and `evaluation_model_2026.joblib`: fits using only data available before those years.
- `evaluation.json`: settings, candidate scores, protected original-file hashes, paired uncertainty intervals and league/confidence comparisons.
- `frozen_selection.json`: settings recorded before later-year evaluation.
- `test_predictions_2025.csv` and `test_predictions_2026.csv`: actual outcomes and both models' probabilities for identical fixtures.
- The two probe scripts and their reports document development comparisons restricted to 2024 and earlier.

All **43 outcome-model tests** passed, including 29 v3 tests for feature chronology, exact baseline-feature parity, missing values, smoothing, pooled goal inputs, probability coherence, calibration, serialization and total-goal markets. Six tests verify total-goal probabilities against explicit score grids, component weighting and threshold ordering. A prediction from the saved final model was verified. SHA-256 checks confirmed the existing scripts and saved models were unchanged.

```powershell
python -m unittest test_football_model.py test_football_model_v2.py test_football_model_v3.py
```
