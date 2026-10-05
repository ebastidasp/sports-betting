# Over 2.5 goals: legacy Poisson versus v3

Over 2.5 goals means at least 3 total goals in FT matches

The legacy model is football_poisson.py, refitted with its defaults on only prior-year data. V3 uses only its goal-rate component for this market; its 1X2 classifier is excluded. All comparisons below use identical supported fixtures.

## 2025: 3,249 common fixtures

| Minimum probability | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate |
|---|---:|---:|---:|---:|
| 60% | 366/581 | 62.99% | 341/524 | 65.08% |
| 70% | 79/102 | 77.45% | 62/79 | 78.48% |
| 80% | 16/19 | 84.21% | 13/14 | 92.86% |

Full-sample Brier: legacy **0.24089**, v3 **0.23945**. Lower is better.
Full-sample log loss: legacy **0.67473**, v3 **0.67154**.

The 95% weekly paired bootstrap interval for v3-minus-legacy Brier is [-0.0035710926752453827, 0.0006382605312582229]. Negative differences favor v3. This interval includes zero.

## 2026: 2,270 common fixtures

| Minimum probability | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate |
|---|---:|---:|---:|---:|
| 60% | 291/439 | 66.29% | 255/369 | 69.11% |
| 70% | 74/107 | 69.16% | 49/61 | 80.33% |
| 80% | 16/17 | 94.12% | 14/15 | 93.33% |

Full-sample Brier: legacy **0.24274**, v3 **0.24023**. Lower is better.
Full-sample log loss: legacy **0.67838**, v3 **0.67274**.

The 95% weekly paired bootstrap interval for v3-minus-legacy Brier is [-0.004166096403480262, -0.0008658686038041742]. Negative differences favor v3. This interval favors v3.

The thresholds are cumulative. Hit-rate differences reflect different selected fixtures; an overlap of selections predicts the same over-2.5 event. Read selected counts and uncertainty alongside hit rates.

2026 is incomplete. Only matches stored by 2026-09-30T23:38:15.160962+00:00 enter the analysis. Iceland has no legacy match-statistics coverage, and Ireland has no eligible 2026 sample.

Full league results, average forecast probabilities and confidence intervals are in evaluation.json. Original scripts, trained models and SQLite caches were opened for reading only.
