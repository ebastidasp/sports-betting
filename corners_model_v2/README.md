# Experimental corners model v2

Created and evaluated on 2026-09-30. **This experiment did not demonstrate an overall accuracy improvement. Keep the existing model as the default.** `football_corners.py` and the saved models in `corners_model/` remain unchanged.

The new entry point is `football_corners_v2.py`. It reads only the local SQLite data and shares the original script's ingestion and historical-statistics helpers. Its fitted models and reports are stored separately in this directory.

## What was tested

- Predict the total directly with regularized Poisson regression and gradient boosting, instead of predicting each team's corners separately.
- Add lagged match-total history, recent attacking tempo, and league variability. Passes and pass accuracy remain inputs.
- Learn cumulative probabilities at totals 3 through 14 using regularized logistic classifiers, blended with a Poisson/negative-binomial distribution. Project the probabilities to an ordered cumulative distribution, so the resulting exact-count probabilities are valid.
- Estimate league dispersion with shrinkage toward a global estimate to limit unstable fits in smaller leagues.

The model uses 122 match-level features. Missing corner counts remain missing; real zero counts remain zero. Features use only previous matches whose results are assumed available three hours after kickoff.

## Selection and comparison

Candidate settings were selected using 2024, with earlier data used for training and 2023 for dispersion calibration. The chosen direct-total candidate was Poisson regression with alpha 1.0. The probability blend uses 25% ordinal probabilities with logistic regularization C=0.1.

The original count estimator won the internal 2024 comparison: MAE **2.60917**, versus **2.62281** for the best direct-total candidate. The new saved bundle therefore embeds a refitted copy of the original count estimator for its displayed corner-count estimate. Its event probabilities use the experimental distribution. The displayed count estimate and the expectation of that probability distribution can differ.

The comparison uses identical primary-league matches with both corner totals observed. Each evaluation model is fitted only on results available before that calendar year. 2025 had already been inspected in previous analyses; 2026 was used as confirmation after the settings were frozen. No parameters were changed in response to its scores.

| Metric | Original 2025 | Experimental 2025 | Original 2026 | Experimental 2026 |
|---|---:|---:|---:|---:|
| Evaluated matches | 3,425 | 3,425 | 2,355 | 2,355 |
| Mean absolute error, corners | 2.65719 | 2.65719 | 2.67710 | 2.67710 |
| Root mean squared error, corners | 3.31010 | 3.31010 | 3.32688 | 3.32688 |
| Mean Brier score, over 6/7/8 | 0.204269 | 0.204498 | 0.201559 | 0.201904 |
| Ranked probability score | 1.85500 | 1.85736 | 1.87575 | 1.87737 |
| Exact-total negative log likelihood | 2.59866 | 2.60099 | 2.60991 | 2.61419 |

Lower scores are better. Count errors tie because the original estimator is retained. The average Brier score is 0.112% worse in 2025 and 0.171% worse in 2026. Weekly paired bootstrap 95% intervals for the experimental-minus-original Brier difference are approximately [-0.000525, 0.001020] and [-0.000449, 0.001092], respectively. These intervals include zero; the experiment offers no reliable evidence of an improvement.

2026 is a partial calendar year, ending at the training cutoff `2026-09-30T23:00:16.428796+00:00`. Only completed matches stored in the databases enter the evaluation. Ireland has no eligible 2026 observations; Iceland has no usable corner targets.

The experimental dispersion is refreshed using out-of-training 2025 predictions before the 2026 evaluation and final fit. The original comparison retains its established 2024 dispersion calibration. This difference is part of the tested procedure. Neither method uses 2026 results to calibrate the 2026 evaluation model.

The full report includes league comparisons and hit rates for over 6, 7 and 8 corners at minimum probabilities of 60%, 70% and 80%. Those hit rates use different selected match sets for each model and must be read alongside the number of predictions. League gains were mixed and did not establish a consistent improvement.

## Commands

Run from the project directory:

```powershell
python football_corners_v2.py train
python football_corners_v2.py predict --country usa --home "Seattle Sounders" --away "Sporting Kansas City"
```

`train` also runs the chronological comparisons. It saves its outputs to `corners_model_v2/` by default and refuses output directories inside the existing model directories. Optional arguments include `--db`, `--output`, `--validation-year`, `--test-year`, `--confirm-year` and `--as-of` (a timezone-aware ISO date).

`predict` prints the count estimate, probabilities for 0 through 9 and more than 9 corners, plus over 6/7/8 and under 7/8/9. It uses the separate final model by default. An `--as-of` date must not precede that model's training cutoff; use the archived evaluation bundles for earlier dates.

## Saved files and checks

- `corners_model.joblib`: final experimental bundle fitted on 35,676 complete match totals, with the original count estimator embedded.
- `evaluation_model_2025.joblib` and `evaluation_model_2026.joblib`: fits using only results available before their evaluation year.
- `evaluation.json`: parameters, source databases, protected original-file hashes, probability scores, count errors and league/threshold comparisons.
- `test_predictions_2025.csv` and `test_predictions_2026.csv`: actual totals, both count estimates and both models' over-line probabilities.

All 16 new tests and all 11 original tests passed. Checks cover chronological inputs, missing values, probability normalization, threshold coherence, the count-estimator fallback and serialization. A prediction using the saved final bundle was verified. SHA-256 checks confirmed the original script and both original saved model files were unchanged.

```powershell
python -m unittest test_football_corners.py test_football_corners_v2.py
```
