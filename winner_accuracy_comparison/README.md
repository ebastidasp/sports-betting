# Match-winner accuracy across first and second divisions

For the latest **winner-only ≥60%, ≥70% and ≥80% comparison separately for every first- and second-division league**, see [Winner thresholds by league](WINNER_THRESHOLDS_BY_LEAGUE.md). That report uses the full refreshed V2/V3 population.

For the same thresholds against **V1 (the original Poisson model)**, see [Fresh V1 versus V3 league comparison](v1_v3/README.md). That report refits V1 on current SQLite history and compares both models on each league's matched fixtures.

This report compares V3's full home/draw/away predictions with both earlier models: the original **football_poisson.py** and **football_model_v2.py**. It evaluates completed historical fixtures from the configured first and second divisions, with coefficients fitted strictly before each evaluation year.

## Definitions and populations

The predicted result is the outcome with the highest probability: **home win, draw or away win (1X2)**. Draws remain in the sample. A home/away winner prediction that ends in a draw is incorrect; a predicted draw is correct only when the match is drawn. A separate table below isolates home/away picks while retaining actual draws as failures.

Confidence is the probability of the chosen outcome, not the sum of both teams' win probabilities. The ≥60%, ≥70% and ≥80% filters are inclusive and cumulative. Models can select different fixtures at the same threshold, so filtered hit-rate differences describe different selected groups. The overall accuracy and probability-score comparisons use identical fixtures.

- **Original Poisson versus V3:** 8,203 common fixtures, including 2,684 second-division matches. Eligible secondary comparisons cover England (40), Germany (79), Italy (136) and Spain (141), where the original model has sufficient cached statistics and history.
- **V2 versus V3:** 12,286 fixtures, including 5,955 second-division matches. Both classifiers can use fixture history even when the original Poisson model lacks match-statistics coverage.

These populations differ. Do not compare an accuracy from one population directly with a number from the other. Within each comparison, both models score exactly the same fixtures. The 2025 period covers the stored full calendar year; 2026 is incomplete. The full V2/V3 population uses the current saved training cutoff. Original Poisson means and eligibility retain the earlier goal-backtest snapshot, so that comparison does not include newly available fixtures or refresh the original model's historical fits.

## Original Poisson versus V3: overall accuracy

| Period | Division | Original Poisson correct / games (accuracy) | V3 correct / games (accuracy) | V3 change | 95% paired interval for change |
| --- | --- | --- | --- | --- | --- |
| 2025 | Both divisions | 2,289 / 4,809 (47.60%) | 2,346 / 4,809 (48.78%) | +1.19 pp | [+0.31, +2.01] pp |
| 2025 | First division | 1,585 / 3,249 (48.78%) | 1,633 / 3,249 (50.26%) | +1.48 pp | [+0.45, +2.50] pp |
| 2025 | Second division | 704 / 1,560 (45.13%) | 713 / 1,560 (45.71%) | +0.58 pp | [-1.28, +2.37] pp |
| 2026 | Both divisions | 1,588 / 3,394 (46.79%) | 1,642 / 3,394 (48.38%) | +1.59 pp | [+0.46, +2.71] pp |
| 2026 | First division | 1,076 / 2,270 (47.40%) | 1,111 / 2,270 (48.94%) | +1.54 pp | [-0.05, +3.21] pp |
| 2026 | Second division | 512 / 1,124 (45.55%) | 531 / 1,124 (47.24%) | +1.69 pp | [+0.00, +3.32] pp |
| Combined | Both divisions | 3,877 / 8,203 (47.26%) | 3,988 / 8,203 (48.62%) | +1.35 pp | [+0.65, +2.02] pp |
| Combined | First division | 2,661 / 5,519 (48.22%) | 2,744 / 5,519 (49.72%) | +1.50 pp | [+0.60, +2.41] pp |
| Combined | Second division | 1,216 / 2,684 (45.31%) | 1,244 / 2,684 (46.35%) | +1.04 pp | [-0.23, +2.31] pp |

### Original Poisson versus V3: confidence cutoffs

| Period | Division | Minimum confidence | Original Poisson correct / selected (accuracy) | V3 correct / selected (accuracy) | V3 change |
| --- | --- | --- | --- | --- | --- |
| 2025 | Both divisions | ≥60% | 395 / 577 (68.46%) | 441 / 633 (69.67%) | +1.21 pp |
| 2025 | Both divisions | ≥70% | 127 / 162 (78.40%) | 150 / 184 (81.52%) | +3.13 pp |
| 2025 | Both divisions | ≥80% | 30 / 36 (83.33%) | 35 / 38 (92.11%) | +8.77 pp |
| 2025 | First division | ≥60% | 327 / 471 (69.43%) | 371 / 534 (69.48%) | +0.05 pp |
| 2025 | First division | ≥70% | 119 / 148 (80.41%) | 137 / 165 (83.03%) | +2.62 pp |
| 2025 | First division | ≥80% | 30 / 36 (83.33%) | 35 / 38 (92.11%) | +8.77 pp |
| 2025 | Second division | ≥60% | 68 / 106 (64.15%) | 70 / 99 (70.71%) | +6.56 pp |
| 2025 | Second division | ≥70% | 8 / 14 (57.14%) | 13 / 19 (68.42%) | +11.28 pp |
| 2025 | Second division | ≥80% | — (0 selected) | — (0 selected) | — |
| 2026 | Both divisions | ≥60% | 325 / 477 (68.13%) | 321 / 476 (67.44%) | -0.70 pp |
| 2026 | Both divisions | ≥70% | 115 / 150 (76.67%) | 110 / 143 (76.92%) | +0.26 pp |
| 2026 | Both divisions | ≥80% | 28 / 32 (87.50%) | 27 / 32 (84.38%) | -3.12 pp |
| 2026 | First division | ≥60% | 257 / 385 (66.75%) | 254 / 381 (66.67%) | -0.09 pp |
| 2026 | First division | ≥70% | 104 / 138 (75.36%) | 99 / 131 (75.57%) | +0.21 pp |
| 2026 | First division | ≥80% | 28 / 32 (87.50%) | 26 / 30 (86.67%) | -0.83 pp |
| 2026 | Second division | ≥60% | 68 / 92 (73.91%) | 67 / 95 (70.53%) | -3.39 pp |
| 2026 | Second division | ≥70% | 11 / 12 (91.67%) | 11 / 12 (91.67%) | +0.00 pp |
| 2026 | Second division | ≥80% | — (0 selected) | 1 / 2 (50.00%) | — |
| Combined | Both divisions | ≥60% | 720 / 1,054 (68.31%) | 762 / 1,109 (68.71%) | +0.40 pp |
| Combined | Both divisions | ≥70% | 242 / 312 (77.56%) | 260 / 327 (79.51%) | +1.95 pp |
| Combined | Both divisions | ≥80% | 58 / 68 (85.29%) | 62 / 70 (88.57%) | +3.28 pp |
| Combined | First division | ≥60% | 584 / 856 (68.22%) | 625 / 915 (68.31%) | +0.08 pp |
| Combined | First division | ≥70% | 223 / 286 (77.97%) | 236 / 296 (79.73%) | +1.76 pp |
| Combined | First division | ≥80% | 58 / 68 (85.29%) | 61 / 68 (89.71%) | +4.41 pp |
| Combined | Second division | ≥60% | 136 / 198 (68.69%) | 137 / 194 (70.62%) | +1.93 pp |
| Combined | Second division | ≥70% | 19 / 26 (73.08%) | 24 / 31 (77.42%) | +4.34 pp |
| Combined | Second division | ≥80% | — (0 selected) | 1 / 2 (50.00%) | — |

### Original Poisson versus V3: probability quality

Lower multiclass log loss and Brier score are better. Brier is the average sum of squared errors across the three outcomes. These scores use all common fixtures, including predictions below 60% confidence.

| Period | Division | Original Poisson log loss | V3 log loss | Original Poisson Brier | V3 Brier | 95% interval for log-loss change (V3 − baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | Both divisions | 1.034045 | 1.023352 | 0.620770 | 0.613829 | [-0.016024, -0.005205] |
| 2025 | First division | 1.021393 | 1.013560 | 0.611552 | 0.606705 | [-0.013499, -0.001977] |
| 2025 | Second division | 1.060396 | 1.043744 | 0.639969 | 0.628665 | [-0.025365, -0.008367] |
| 2026 | Both divisions | 1.034925 | 1.027398 | 0.621763 | 0.616497 | [-0.012599, -0.002344] |
| 2026 | First division | 1.031059 | 1.023346 | 0.619096 | 0.613543 | [-0.014901, -0.000577] |
| 2026 | Second division | 1.042734 | 1.035582 | 0.627150 | 0.622462 | [-0.015917, +0.001888] |
| Combined | Both divisions | 1.034409 | 1.025026 | 0.621181 | 0.614932 | [-0.013291, -0.005406] |
| Combined | First division | 1.025368 | 1.017585 | 0.614655 | 0.609517 | [-0.012332, -0.003009] |
| Combined | Second division | 1.053000 | 1.040326 | 0.634600 | 0.626067 | [-0.019110, -0.006492] |

### All three models on the original model's common fixtures, pooled

All accuracy columns in this table use the same league-specific fixtures. The V2 column provides an additional baseline on the original model's smaller eligible population.

| Country | Division | League ID | Common games | Original Poisson | V2 classifier | V3 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 817 | 41.13% | 44.31% | 43.70% |
| Argentina | Second | 129 | 0 | — | — | — |
| Chile | First | 265 | 368 | 49.73% | 49.46% | 52.45% |
| Chile | Second | 266 | 0 | — | — | — |
| Colombia | First | 239 | 713 | 48.53% | 49.37% | 49.79% |
| Colombia | Second | 240 | 0 | — | — | — |
| England | First | 39 | 618 | 50.16% | 50.32% | 50.81% |
| England | Second | 40 | 859 | 44.59% | 46.45% | 44.47% |
| Germany | First | 78 | 516 | 51.36% | 51.94% | 52.13% |
| Germany | Second | 79 | 507 | 45.36% | 46.94% | 47.34% |
| Iceland | First | 164 | 0 | — | — | — |
| Iceland | Second | 165 | 0 | — | — | — |
| Ireland | First | 357 | 152 | 42.76% | 46.71% | 48.68% |
| Ireland | Second | 358 | 0 | — | — | — |
| Italy | First | 135 | 631 | 51.35% | 53.88% | 53.41% |
| Italy | Second | 136 | 589 | 45.67% | 45.33% | 46.35% |
| South Korea | First | 292 | 161 | 39.75% | 41.61% | 40.37% |
| South Korea | Second | 293 | 0 | — | — | — |
| Spain | First | 140 | 648 | 53.09% | 53.09% | 53.55% |
| Spain | Second | 141 | 729 | 45.82% | 47.33% | 47.87% |
| United States | First | 253 | 895 | 47.37% | 49.05% | 48.38% |

### Original-model comparison: interpretation

- **Both divisions:** Original Poisson 47.26%, V3 48.62%; change +1.35 pp. The interval favors V3.
- **First division:** Original Poisson 48.22%, V3 49.72%; change +1.50 pp. The interval favors V3.
- **Second division:** Original Poisson 45.31%, V3 46.35%; change +1.04 pp. The interval includes zero.

## V2 classifier versus V3: full available population

This comparison includes secondary fixtures unavailable to the original Poisson model. V2 is evaluated with its previously selected fixed settings; no new architecture or hyperparameters were selected using the 2025/2026 results.

| Period | Division | V2 classifier correct / games (accuracy) | V3 correct / games (accuracy) | V3 change | 95% paired interval for change |
| --- | --- | --- | --- | --- | --- |
| 2025 | Both divisions | 3,484 / 7,190 (48.46%) | 3,460 / 7,190 (48.12%) | -0.33 pp | [-0.87, +0.22] pp |
| 2025 | First division | 1,852 / 3,725 (49.72%) | 1,855 / 3,725 (49.80%) | +0.08 pp | [-0.63, +0.75] pp |
| 2025 | Second division | 1,632 / 3,465 (47.10%) | 1,605 / 3,465 (46.32%) | -0.78 pp | [-1.56, +0.05] pp |
| 2026 | Both divisions | 2,407 / 5,096 (47.23%) | 2,428 / 5,096 (47.65%) | +0.41 pp | [-0.14, +0.99] pp |
| 2026 | First division | 1,250 / 2,606 (47.97%) | 1,256 / 2,606 (48.20%) | +0.23 pp | [-0.46, +0.93] pp |
| 2026 | Second division | 1,157 / 2,490 (46.47%) | 1,172 / 2,490 (47.07%) | +0.60 pp | [-0.21, +1.41] pp |
| Combined | Both divisions | 5,891 / 12,286 (47.95%) | 5,888 / 12,286 (47.92%) | -0.02 pp | [-0.43, +0.37] pp |
| Combined | First division | 3,102 / 6,331 (49.00%) | 3,111 / 6,331 (49.14%) | +0.14 pp | [-0.38, +0.65] pp |
| Combined | Second division | 2,789 / 5,955 (46.83%) | 2,777 / 5,955 (46.63%) | -0.20 pp | [-0.79, +0.39] pp |

### V2 versus V3: confidence cutoffs

| Period | Division | Minimum confidence | V2 classifier correct / selected (accuracy) | V3 correct / selected (accuracy) | V3 change |
| --- | --- | --- | --- | --- | --- |
| 2025 | Both divisions | ≥60% | 747 / 1,110 (67.30%) | 509 / 732 (69.54%) | +2.24 pp |
| 2025 | Both divisions | ≥70% | 228 / 307 (74.27%) | 154 / 191 (80.63%) | +6.36 pp |
| 2025 | Both divisions | ≥80% | 29 / 33 (87.88%) | 35 / 38 (92.11%) | +4.23 pp |
| 2025 | First division | ≥60% | 523 / 783 (66.79%) | 400 / 573 (69.81%) | +3.01 pp |
| 2025 | First division | ≥70% | 192 / 254 (75.59%) | 141 / 171 (82.46%) | +6.87 pp |
| 2025 | First division | ≥80% | 26 / 30 (86.67%) | 35 / 38 (92.11%) | +5.44 pp |
| 2025 | Second division | ≥60% | 224 / 327 (68.50%) | 109 / 159 (68.55%) | +0.05 pp |
| 2025 | Second division | ≥70% | 36 / 53 (67.92%) | 13 / 20 (65.00%) | -2.92 pp |
| 2025 | Second division | ≥80% | 3 / 3 (100.00%) | — (0 selected) | — |
| 2026 | Both divisions | ≥60% | 508 / 784 (64.80%) | 362 / 539 (67.16%) | +2.37 pp |
| 2026 | Both divisions | ≥70% | 165 / 223 (73.99%) | 116 / 153 (75.82%) | +1.83 pp |
| 2026 | Both divisions | ≥80% | 39 / 46 (84.78%) | 27 / 32 (84.38%) | -0.41 pp |
| 2026 | First division | ≥60% | 335 / 518 (64.67%) | 271 / 409 (66.26%) | +1.59 pp |
| 2026 | First division | ≥70% | 138 / 185 (74.59%) | 102 / 137 (74.45%) | -0.14 pp |
| 2026 | First division | ≥80% | 36 / 42 (85.71%) | 26 / 30 (86.67%) | +0.95 pp |
| 2026 | Second division | ≥60% | 173 / 266 (65.04%) | 91 / 130 (70.00%) | +4.96 pp |
| 2026 | Second division | ≥70% | 27 / 38 (71.05%) | 14 / 16 (87.50%) | +16.45 pp |
| 2026 | Second division | ≥80% | 3 / 4 (75.00%) | 1 / 2 (50.00%) | -25.00 pp |
| Combined | Both divisions | ≥60% | 1,255 / 1,894 (66.26%) | 871 / 1,271 (68.53%) | +2.27 pp |
| Combined | Both divisions | ≥70% | 393 / 530 (74.15%) | 270 / 344 (78.49%) | +4.34 pp |
| Combined | Both divisions | ≥80% | 68 / 79 (86.08%) | 62 / 70 (88.57%) | +2.50 pp |
| Combined | First division | ≥60% | 858 / 1,301 (65.95%) | 671 / 982 (68.33%) | +2.38 pp |
| Combined | First division | ≥70% | 330 / 439 (75.17%) | 243 / 308 (78.90%) | +3.73 pp |
| Combined | First division | ≥80% | 62 / 72 (86.11%) | 61 / 68 (89.71%) | +3.59 pp |
| Combined | Second division | ≥60% | 397 / 593 (66.95%) | 200 / 289 (69.20%) | +2.26 pp |
| Combined | Second division | ≥70% | 63 / 91 (69.23%) | 27 / 36 (75.00%) | +5.77 pp |
| Combined | Second division | ≥80% | 6 / 7 (85.71%) | 1 / 2 (50.00%) | -35.71 pp |

### V2 versus V3: probability quality

| Period | Division | V2 classifier log loss | V3 log loss | V2 classifier Brier | V3 Brier | 95% interval for log-loss change (V3 − baseline) |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | Both divisions | 1.029715 | 1.029682 | 0.618260 | 0.618445 | [-0.002644, +0.002375] |
| 2025 | First division | 1.019141 | 1.017240 | 0.610242 | 0.609223 | [-0.005070, +0.001216] |
| 2025 | Second division | 1.041083 | 1.043057 | 0.626879 | 0.628359 | [-0.001432, +0.005423] |
| 2026 | Both divisions | 1.034476 | 1.033061 | 0.622089 | 0.620812 | [-0.003595, +0.000953] |
| 2026 | First division | 1.028587 | 1.026650 | 0.617605 | 0.615856 | [-0.004957, +0.001199] |
| 2026 | Second division | 1.040640 | 1.039772 | 0.626782 | 0.626000 | [-0.004895, +0.003160] |
| Combined | Both divisions | 1.031690 | 1.031084 | 0.619848 | 0.619427 | [-0.002312, +0.001126] |
| Combined | First division | 1.023029 | 1.021113 | 0.613273 | 0.611954 | [-0.004116, +0.000286] |
| Combined | Second division | 1.040897 | 1.041684 | 0.626838 | 0.627372 | [-0.001877, +0.003385] |

### V2 versus V3 by league, pooled

| Country | Division | League ID | Common games | V2 classifier | V3 |
| --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 904 | 44.58% | 44.36% |
| Argentina | Second | 129 | 1,190 | 47.31% | 47.06% |
| Chile | First | 265 | 423 | 48.23% | 51.06% |
| Chile | Second | 266 | 448 | 42.19% | 42.41% |
| Colombia | First | 239 | 761 | 49.41% | 49.67% |
| Colombia | Second | 240 | 532 | 49.06% | 46.99% |
| England | First | 39 | 618 | 50.32% | 50.81% |
| England | Second | 40 | 921 | 45.93% | 44.19% |
| Germany | First | 78 | 516 | 51.94% | 52.13% |
| Germany | Second | 79 | 516 | 46.71% | 47.29% |
| Iceland | First | 164 | 302 | 49.34% | 49.01% |
| Iceland | Second | 165 | 273 | 51.65% | 52.75% |
| Ireland | First | 357 | 181 | 47.51% | 48.62% |
| Ireland | Second | 358 | 183 | 53.01% | 51.37% |
| Italy | First | 135 | 632 | 53.80% | 53.32% |
| Italy | Second | 136 | 629 | 45.15% | 45.95% |
| South Korea | First | 292 | 411 | 40.15% | 39.66% |
| South Korea | Second | 293 | 491 | 46.44% | 47.05% |
| Spain | First | 140 | 648 | 53.09% | 53.55% |
| Spain | Second | 141 | 772 | 46.89% | 47.67% |
| United States | First | 253 | 935 | 48.77% | 48.13% |

### V2 comparison: interpretation

- **Both divisions:** V2 classifier 47.95%, V3 47.92%; change -0.02 pp. The interval includes zero.
- **First division:** V2 classifier 49.00%, V3 49.14%; change +0.14 pp. The interval includes zero.
- **Second division:** V2 classifier 46.83%, V3 46.63%; change -0.20 pp. The interval includes zero.

## Draws and winner-only selections

The following pooled tables distinguish 1X2 accuracy from precision among home/away picks. The winner-only selections exclude predicted draws, while games that actually end in a draw remain counted as incorrect winner picks. Each model can select different games, so these conditional rates are not a paired accuracy gain.

### Common fixtures with the original model

| Division | Model | Actual draws / games | Draw picks | Correct draw picks | Home/away picks: correct / selected (accuracy) |
| --- | --- | --- | --- | --- | --- |
| Both divisions | Original Poisson | 2,192 / 8,203 | 55 | 14 | 3,863 / 8,148 (47.41%) |
| Both divisions | V2 classifier | 2,192 / 8,203 | 219 | 82 | 3,903 / 7,984 (48.89%) |
| Both divisions | V3 | 2,192 / 8,203 | 28 | 13 | 3,975 / 8,175 (48.62%) |
| First division | Original Poisson | 1,472 / 5,519 | 55 | 14 | 2,647 / 5,464 (48.44%) |
| First division | V2 classifier | 1,472 / 5,519 | 166 | 68 | 2,668 / 5,353 (49.84%) |
| First division | V3 | 1,472 / 5,519 | 27 | 13 | 2,731 / 5,492 (49.73%) |
| Second division | Original Poisson | 720 / 2,684 | 0 | 0 | 1,216 / 2,684 (45.31%) |
| Second division | V2 classifier | 720 / 2,684 | 53 | 14 | 1,235 / 2,631 (46.94%) |
| Second division | V3 | 720 / 2,684 | 1 | 0 | 1,244 / 2,683 (46.37%) |

### Full V2/V3 population

| Division | Model | Actual draws / games | Draw picks | Correct draw picks | Home/away picks: correct / selected (accuracy) |
| --- | --- | --- | --- | --- | --- |
| Both divisions | V2 classifier | 3,355 / 12,286 | 309 | 118 | 5,773 / 11,977 (48.20%) |
| Both divisions | V3 | 3,355 / 12,286 | 43 | 19 | 5,869 / 12,243 (47.94%) |
| First division | V2 classifier | 1,671 / 6,331 | 174 | 71 | 3,031 / 6,157 (49.23%) |
| First division | V3 | 1,671 / 6,331 | 30 | 15 | 3,096 / 6,301 (49.14%) |
| Second division | V2 classifier | 1,684 / 5,955 | 135 | 47 | 2,742 / 5,820 (47.11%) |
| Second division | V3 | 1,684 / 5,955 | 13 | 4 | 2,773 / 5,942 (46.67%) |

## Evaluation details and uncertainty

- V3 uses the **full saved annual prediction bundle**, including its goal and classifier components and saved calibration parameters. Its 1X2 blend is 75% goal probabilities and 25% recent-data boosting. This is different from the goal-total report, which uses only the goal component.
- The original Poisson 1X2 forecasts are reconstructed through its own `outcome_probabilities` function from saved chronological home/away means and country/year Dixon–Coles rho. Its original probability clipping and normalization remain in effect. Those means were fitted using prior results from both configured divisions.
- V2 is refitted in memory before each test year using its previously selected classifier and the same historical feature columns. Existing saved model files are not overwritten. Both V2 and V3 train on both configured divisions; their architecture selection used first-division validation.
- SQLite caches are opened read-only. Only completed FT fixtures with known goals enter the analysis. Results are assumed available three hours after kickoff. Historical inputs update from earlier completed matches, while model coefficients remain fixed within each annual test.
- Reconstructed V2 and V3 first-division probabilities must match their original annual forecast CSVs within 1e-10. Legacy eligibility and rates come from the extended goal backtest, which already reproduced its original first-division forecasts and checked chronology. Fixture IDs, kickoff dates, league IDs and actual goals must agree between compared inputs.
- Paired 95% intervals resample calendar weeks 5,000 times with replacement (seed 42). A positive accuracy difference favors V3; a negative probability-score difference favors V3. Intervals containing zero do not clearly distinguish the models. Weekly blocking does not remove all dependence across fixtures, and intervals are not adjusted for the many thresholds, leagues and comparisons examined.
- Selected counts and coverage matter. High-confidence results based on a few matches can be unstable. Zero selected matches have unavailable accuracy; they are not 0% accuracy. Wilson intervals for selected-group precision, mean confidence, outcome counts and confusion matrices are saved in the JSON report.
- Pooled results summarize the same annual forecasts; they are not another independent test. Read yearly and division-specific results before interpreting an apparent overall gain. Missing original-model statistics reduce its comparable coverage; they do not demonstrate poor predictive accuracy in those excluded leagues.

The refreshed V2/V3 historical source cutoff is **2026-10-03T03:21:42.527730+00:00** (02 October 2026, 22:21:42 in Colombia). The current all-data model is not used to predict historical test matches. These observed results do not guarantee future accuracy.

## Files and reproduction

- [evaluation.json](evaluation.json): overall, division and league metrics, confidence filters, draw statistics, confusion matrices and paired uncertainty.
- [threshold_summary.csv](threshold_summary.csv): all confidence thresholds by population, year, division and league, with correct/incorrect counts, coverage and intervals.
- [all_predictions_2025.csv](all_predictions_2025.csv) and [all_predictions_2026.csv](all_predictions_2026.csv): full V2/V3 fixture predictions across both divisions.
- [common_predictions_2025.csv](common_predictions_2025.csv) and [common_predictions_2026.csv](common_predictions_2026.csv): matched fixtures with all three model probabilities.
- [forecast_metadata.json](forecast_metadata.json) and [protected_input_sha256.json](protected_input_sha256.json): annual model settings, training cutoffs, legacy metadata and protected input hashes.
- [Goal-total comparison](../goals_accuracy_comparison/README.md): the separate over-1.5/2.5/3.5 analysis and original-model coverage details.

From the project directory, run the full historical reconstruction and report:

```powershell
python winner_accuracy_comparison/compare_models.py
```

To regenerate the report from saved predictions:

```powershell
python winner_accuracy_comparison/compare_models.py --mode report
```

The full command fits only an in-memory V2 reference; it does not retrain V3, overwrite the original Poisson model, or call an API.
