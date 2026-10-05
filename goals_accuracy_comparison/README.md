# Goal totals accuracy: legacy Poisson versus V3 across configured divisions

This report compares **football_poisson.py** with the **goal-rate component of football_model_v3.py** on the same historical fixtures. It includes configured first and second divisions wherever both models have eligible data, for over 1.5, 2.5 and 3.5 goals at forecast probabilities of at least 60%, 70% and 80%.

The saved V3 models already trained on both configured divisions. This extension broadens the evaluated fixtures and reuses those annual models; it does not retrain or change V3. Its architecture and calibration method were selected on first-division validation. Secondary-division results evaluate that unchanged selected model, with no new tuning. Legacy fits for the four secondary leagues with statistics were regenerated strictly before each year; unchanged first-division cached forecasts were reused elsewhere.

## Reading the tables

| Goal market | A correct prediction requires |
| --- | --- |
| Over 1.5 | At least 2 total FT goals |
| Over 2.5 | At least 3 total FT goals |
| Over 3.5 | At least 4 total FT goals |

**Hit rate** is correct predictions divided by selected predictions. Cutoffs are inclusive and cumulative: ≥60% includes forecasts above 70% and 80%. A forecast cutoff is a filter, not a guaranteed hit rate.

The underlying eligible fixtures are identical for both models. Each probability filter can select different fixtures, so hit-rate differences describe different selected groups. On a fixture selected by both models for the same over market, their predictions succeed or fail together. Full-fixture probability scores provide the paired comparison.

A dash means no qualifying predictions and no measurable hit rate; it does not mean 0% accuracy.

## 2025: complete calendar year

**4,809 common fixtures:** 3,249 first-division and 1,560 second-division matches. V3 could evaluate 7,190 fixtures before legacy eligibility restrictions.

Model coefficients were fitted using results available before 1 January 2025. Calendar boundaries use UTC.

### All eligible divisions

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 3,271 / 4,450 | 73.51% | 3,420 / 4,695 | 72.84% | -0.66 pp |
| Over 1.5 | ≥70% | 2,234 / 2,928 | 76.30% | 2,411 / 3,190 | 75.58% | -0.72 pp |
| Over 1.5 | ≥80% | 824 / 1,032 | 79.84% | 829 / 1,017 | 81.51% | +1.67 pp |
| Over 2.5 | ≥60% | 426 / 679 | 62.74% | 400 / 622 | 64.31% | +1.57 pp |
| Over 2.5 | ≥70% | 83 / 107 | 77.57% | 62 / 79 | 78.48% | +0.91 pp |
| Over 2.5 | ≥80% | 16 / 19 | 84.21% | 13 / 14 | 92.86% | +8.65 pp |
| Over 3.5 | ≥60% | 16 / 24 | 66.67% | 11 / 17 | 64.71% | -1.96 pp |
| Over 3.5 | ≥70% | 3 / 4 | 75.00% | 2 / 2 | 100.00% | +25.00 pp |
| Over 3.5 | ≥80% | 1 / 1 | 100.00% | — (0 selected) | — | — |

### First divisions

3,249 common fixtures.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 2,162 / 2,946 | 73.39% | 2,280 / 3,138 | 72.66% | -0.73 pp |
| Over 1.5 | ≥70% | 1,547 / 2,026 | 76.36% | 1,652 / 2,180 | 75.78% | -0.58 pp |
| Over 1.5 | ≥80% | 678 / 841 | 80.62% | 691 / 838 | 82.46% | +1.84 pp |
| Over 2.5 | ≥60% | 366 / 581 | 62.99% | 341 / 524 | 65.08% | +2.08 pp |
| Over 2.5 | ≥70% | 79 / 102 | 77.45% | 62 / 79 | 78.48% | +1.03 pp |
| Over 2.5 | ≥80% | 16 / 19 | 84.21% | 13 / 14 | 92.86% | +8.65 pp |
| Over 3.5 | ≥60% | 15 / 23 | 65.22% | 11 / 17 | 64.71% | -0.51 pp |
| Over 3.5 | ≥70% | 3 / 4 | 75.00% | 2 / 2 | 100.00% | +25.00 pp |
| Over 3.5 | ≥80% | 1 / 1 | 100.00% | — (0 selected) | — | — |

### Second divisions

1,560 common fixtures.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 1,109 / 1,504 | 73.74% | 1,140 / 1,557 | 73.22% | -0.52 pp |
| Over 1.5 | ≥70% | 687 / 902 | 76.16% | 759 / 1,010 | 75.15% | -1.02 pp |
| Over 1.5 | ≥80% | 146 / 191 | 76.44% | 138 / 179 | 77.09% | +0.66 pp |
| Over 2.5 | ≥60% | 60 / 98 | 61.22% | 59 / 98 | 60.20% | -1.02 pp |
| Over 2.5 | ≥70% | 4 / 5 | 80.00% | — (0 selected) | — | — |
| Over 2.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥60% | 1 / 1 | 100.00% | — (0 selected) | — | — |
| Over 3.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |

## 2026: available fixtures through the September cutoff

**3,394 common fixtures:** 2,270 first-division and 1,124 second-division matches. V3 could evaluate 5,070 fixtures before legacy eligibility restrictions.

Model coefficients were fitted using results available before 1 January 2026. The 2026 sample is incomplete.

### All eligible divisions

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 2,453 / 3,177 | 77.21% | 2,547 / 3,318 | 76.76% | -0.45 pp |
| Over 1.5 | ≥70% | 1,781 / 2,237 | 79.62% | 1,948 / 2,462 | 79.12% | -0.49 pp |
| Over 1.5 | ≥80% | 615 / 739 | 83.22% | 618 / 732 | 84.43% | +1.21 pp |
| Over 2.5 | ≥60% | 339 / 512 | 66.21% | 303 / 445 | 68.09% | +1.88 pp |
| Over 2.5 | ≥70% | 75 / 110 | 68.18% | 49 / 62 | 79.03% | +10.85 pp |
| Over 2.5 | ≥80% | 16 / 17 | 94.12% | 14 / 15 | 93.33% | -0.78 pp |
| Over 3.5 | ≥60% | 15 / 22 | 68.18% | 12 / 18 | 66.67% | -1.52 pp |
| Over 3.5 | ≥70% | 2 / 2 | 100.00% | 6 / 8 | 75.00% | -25.00 pp |
| Over 3.5 | ≥80% | — (0 selected) | — | 2 / 2 | 100.00% | — |

### First divisions

2,270 common fixtures.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 1,602 / 2,083 | 76.91% | 1,678 / 2,194 | 76.48% | -0.43 pp |
| Over 1.5 | ≥70% | 1,150 / 1,462 | 78.66% | 1,236 / 1,568 | 78.83% | +0.17 pp |
| Over 1.5 | ≥80% | 493 / 589 | 83.70% | 479 / 565 | 84.78% | +1.08 pp |
| Over 2.5 | ≥60% | 291 / 439 | 66.29% | 255 / 369 | 69.11% | +2.82 pp |
| Over 2.5 | ≥70% | 74 / 107 | 69.16% | 49 / 61 | 80.33% | +11.17 pp |
| Over 2.5 | ≥80% | 16 / 17 | 94.12% | 14 / 15 | 93.33% | -0.78 pp |
| Over 3.5 | ≥60% | 15 / 22 | 68.18% | 12 / 18 | 66.67% | -1.52 pp |
| Over 3.5 | ≥70% | 2 / 2 | 100.00% | 6 / 8 | 75.00% | -25.00 pp |
| Over 3.5 | ≥80% | — (0 selected) | — | 2 / 2 | 100.00% | — |

### Second divisions

1,124 common fixtures.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 851 / 1,094 | 77.79% | 869 / 1,124 | 77.31% | -0.47 pp |
| Over 1.5 | ≥70% | 631 / 775 | 81.42% | 712 / 894 | 79.64% | -1.78 pp |
| Over 1.5 | ≥80% | 122 / 150 | 81.33% | 139 / 167 | 83.23% | +1.90 pp |
| Over 2.5 | ≥60% | 48 / 73 | 65.75% | 48 / 76 | 63.16% | -2.60 pp |
| Over 2.5 | ≥70% | 1 / 3 | 33.33% | 0 / 1 | 0.00% | -33.33 pp |
| Over 2.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥60% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |

## Both periods combined

**8,203 common fixtures:** 5,519 first-division and 2,684 second-division matches. V3 could evaluate 12,260 fixtures before legacy eligibility restrictions.

This pools the same annual forecasts from two fitted models. It is a descriptive summary, not an additional independent test; fixture mixes and fitted parameters differ between years.

### All eligible divisions

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 5,724 / 7,627 | 75.05% | 5,967 / 8,013 | 74.47% | -0.58 pp |
| Over 1.5 | ≥70% | 4,015 / 5,165 | 77.73% | 4,359 / 5,652 | 77.12% | -0.61 pp |
| Over 1.5 | ≥80% | 1,439 / 1,771 | 81.25% | 1,447 / 1,749 | 82.73% | +1.48 pp |
| Over 2.5 | ≥60% | 765 / 1,191 | 64.23% | 703 / 1,067 | 65.89% | +1.65 pp |
| Over 2.5 | ≥70% | 158 / 217 | 72.81% | 111 / 141 | 78.72% | +5.91 pp |
| Over 2.5 | ≥80% | 32 / 36 | 88.89% | 27 / 29 | 93.10% | +4.21 pp |
| Over 3.5 | ≥60% | 31 / 46 | 67.39% | 23 / 35 | 65.71% | -1.68 pp |
| Over 3.5 | ≥70% | 5 / 6 | 83.33% | 8 / 10 | 80.00% | -3.33 pp |
| Over 3.5 | ≥80% | 1 / 1 | 100.00% | 2 / 2 | 100.00% | +0.00 pp |

### First divisions

5,519 common fixtures.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 3,764 / 5,029 | 74.85% | 3,958 / 5,332 | 74.23% | -0.61 pp |
| Over 1.5 | ≥70% | 2,697 / 3,488 | 77.32% | 2,888 / 3,748 | 77.05% | -0.27 pp |
| Over 1.5 | ≥80% | 1,171 / 1,430 | 81.89% | 1,170 / 1,403 | 83.39% | +1.50 pp |
| Over 2.5 | ≥60% | 657 / 1,020 | 64.41% | 596 / 893 | 66.74% | +2.33 pp |
| Over 2.5 | ≥70% | 153 / 209 | 73.21% | 111 / 140 | 79.29% | +6.08 pp |
| Over 2.5 | ≥80% | 32 / 36 | 88.89% | 27 / 29 | 93.10% | +4.21 pp |
| Over 3.5 | ≥60% | 30 / 45 | 66.67% | 23 / 35 | 65.71% | -0.95 pp |
| Over 3.5 | ≥70% | 5 / 6 | 83.33% | 8 / 10 | 80.00% | -3.33 pp |
| Over 3.5 | ≥80% | 1 / 1 | 100.00% | 2 / 2 | 100.00% | +0.00 pp |

### Second divisions

2,684 common fixtures.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 1,960 / 2,598 | 75.44% | 2,009 / 2,681 | 74.93% | -0.51 pp |
| Over 1.5 | ≥70% | 1,318 / 1,677 | 78.59% | 1,471 / 1,904 | 77.26% | -1.33 pp |
| Over 1.5 | ≥80% | 268 / 341 | 78.59% | 277 / 346 | 80.06% | +1.47 pp |
| Over 2.5 | ≥60% | 108 / 171 | 63.16% | 107 / 174 | 61.49% | -1.66 pp |
| Over 2.5 | ≥70% | 5 / 8 | 62.50% | 0 / 1 | 0.00% | -62.50 pp |
| Over 2.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥60% | 1 / 1 | 100.00% | — (0 selected) | — | — |
| Over 3.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |

## Second-division results by league, both periods combined

These league tables isolate the secondary divisions with eligible data in both models. They use the same pooled annual forecasts as the tables above.

### England: Championship (league 40)

859 common fixtures across 2025 and the available 2026 period.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 635 / 845 | 75.15% | 645 / 859 | 75.09% | -0.06 pp |
| Over 1.5 | ≥70% | 403 / 519 | 77.65% | 464 / 611 | 75.94% | -1.71 pp |
| Over 1.5 | ≥80% | 35 / 46 | 76.09% | 13 / 18 | 72.22% | -3.86 pp |
| Over 2.5 | ≥60% | 12 / 20 | 60.00% | 4 / 7 | 57.14% | -2.86 pp |
| Over 2.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 2.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥60% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |

### Germany: 2. Bundesliga (league 79)

507 common fixtures across 2025 and the available 2026 period.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 403 / 507 | 79.49% | 403 / 507 | 79.49% | +0.00 pp |
| Over 1.5 | ≥70% | 396 / 499 | 79.36% | 403 / 507 | 79.49% | +0.13 pp |
| Over 1.5 | ≥80% | 173 / 223 | 77.58% | 228 / 287 | 79.44% | +1.86 pp |
| Over 2.5 | ≥60% | 74 / 116 | 63.79% | 94 / 151 | 62.25% | -1.54 pp |
| Over 2.5 | ≥70% | 4 / 6 | 66.67% | 0 / 1 | 0.00% | -66.67 pp |
| Over 2.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥60% | 1 / 1 | 100.00% | — (0 selected) | — | — |
| Over 3.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |

### Italy: Serie B (league 136)

589 common fixtures across 2025 and the available 2026 period.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 440 / 582 | 75.60% | 445 / 589 | 75.55% | -0.05 pp |
| Over 1.5 | ≥70% | 221 / 281 | 78.65% | 301 / 391 | 76.98% | -1.67 pp |
| Over 1.5 | ≥80% | 12 / 13 | 92.31% | 17 / 20 | 85.00% | -7.31 pp |
| Over 2.5 | ≥60% | 4 / 7 | 57.14% | 4 / 8 | 50.00% | -7.14 pp |
| Over 2.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 2.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥60% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |

### Spain: Segunda División (league 141)

729 common fixtures across 2025 and the available 2026 period.

| Goal market | Minimum forecast | Legacy correct / selected | Legacy hit rate | V3 correct / selected | V3 hit rate | V3 − legacy |
| --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | 482 / 664 | 72.59% | 516 / 726 | 71.07% | -1.52 pp |
| Over 1.5 | ≥70% | 298 / 378 | 78.84% | 303 / 395 | 76.71% | -2.13 pp |
| Over 1.5 | ≥80% | 48 / 59 | 81.36% | 19 / 21 | 90.48% | +9.12 pp |
| Over 2.5 | ≥60% | 18 / 28 | 64.29% | 5 / 8 | 62.50% | -1.79 pp |
| Over 2.5 | ≥70% | 1 / 2 | 50.00% | — (0 selected) | — | — |
| Over 2.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥60% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥70% | — (0 selected) | — | — (0 selected) | — | — |
| Over 3.5 | ≥80% | — (0 selected) | — | — (0 selected) | — | — |

## Coverage by country and division

“V3 available” counts the annual predictions before the common-fixture intersection. “Common” counts fixtures supported by both models. Missing legacy statistics, incomplete venue history, provisional ratings, or unavailable venue formulas reduce coverage. An absent sample is not a measured failure rate.

Eligible paired secondary fixtures cover **4 of 10 configured second-division leagues** in these caches.

| Country | Division | League ID | V3 available 2025 | Common 2025 | V3 available 2026 | Common 2026 | Coverage note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 503 | 471 | 400 | 346 | Eligible in both models. |
| Argentina | Second | 129 | 632 | 0 | 547 | 0 | No completed second-division test fixtures with both teams' cached statistics. |
| Chile | First | 265 | 240 | 210 | 183 | 158 | Eligible in both models. |
| Chile | Second | 266 | 251 | 0 | 197 | 0 | No completed second-division test fixtures with both teams' cached statistics. |
| Colombia | First | 239 | 450 | 409 | 305 | 304 | Eligible in both models. |
| Colombia | Second | 240 | 306 | 0 | 218 | 0 | No completed second-division test fixtures with both teams' cached statistics. |
| England | First | 39 | 378 | 378 | 240 | 240 | Eligible in both models. |
| England | Second | 40 | 558 | 510 | 363 | 349 | Eligible in both models. |
| Germany | First | 78 | 308 | 308 | 208 | 208 | Eligible in both models. |
| Germany | Second | 79 | 307 | 306 | 209 | 201 | Eligible in both models. |
| Iceland | First | 164 | 162 | 0 | 140 | 0 | Legacy model requires both teams' cached match statistics. |
| Iceland | Second | 165 | 137 | 0 | 136 | 0 | Legacy requires both teams' cached match statistics. |
| Ireland | First | 357 | 181 | 152 | 0 | 0 | 2026: No eligible training or test fixtures. |
| Ireland | Second | 358 | 183 | 0 | 0 | 0 | No completed second-division test fixtures with both teams' cached statistics. |
| Italy | First | 135 | 368 | 367 | 264 | 264 | Eligible in both models. |
| Italy | Second | 136 | 369 | 334 | 260 | 255 | Eligible in both models. |
| South Korea | First | 292 | 232 | 89 | 179 | 72 | Eligible in both models. |
| South Korea | Second | 293 | 275 | 0 | 216 | 0 | No completed second-division test fixtures with both teams' cached statistics. |
| Spain | First | 140 | 370 | 370 | 278 | 278 | Eligible in both models. |
| Spain | Second | 141 | 447 | 410 | 325 | 319 | Eligible in both models. |
| United States | First | 253 | 533 | 495 | 402 | 400 | Eligible in both models. |
| United States | Second | — | 0 | 0 | 0 | 0 | No second division configured in this cache. |

Secondary divisions without an eligible paired sample remain listed here. Where no cached team-statistics coverage exists, the legacy model cannot support the comparison even when V3 can use recorded goals. Countries with only one configured league have no configured secondary sample.

## Probability quality on all common fixtures

Brier score and binary log loss assess every event probability, including forecasts below 60%. Lower values are better. These compare identical fixtures and complement selected-group hit rates.

| Period | Goal market | Actual event frequency | Legacy Brier | V3 Brier | Legacy log loss | V3 log loss | 95% interval: Brier difference (V3 − legacy) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | Over 1.5 | 72.57% | 0.195956 | 0.195305 | 0.579550 | 0.577779 | [-0.001879, +0.000539] |
| 2025 | Over 2.5 | 49.22% | 0.243978 | 0.243096 | 0.681024 | 0.679025 | [-0.002420, +0.000731] |
| 2025 | Over 3.5 | 26.12% | 0.188675 | 0.187142 | 0.563490 | 0.559434 | [-0.002959, -0.000092] |
| 2026 | Over 1.5 | 76.31% | 0.178703 | 0.177490 | 0.541358 | 0.538204 | [-0.002524, +0.000070] |
| 2026 | Over 2.5 | 52.92% | 0.244198 | 0.242815 | 0.681361 | 0.678196 | [-0.002625, -0.000227] |
| 2026 | Over 3.5 | 28.99% | 0.199101 | 0.197615 | 0.585799 | 0.582018 | [-0.002510, -0.000511] |
| Combined | Over 1.5 | 74.12% | 0.188818 | 0.187934 | 0.563748 | 0.561405 | [-0.001805, +0.000014] |
| Combined | Over 2.5 | 50.75% | 0.244069 | 0.242979 | 0.681164 | 0.678682 | [-0.002133, -0.000015] |
| Combined | Over 3.5 | 27.31% | 0.192989 | 0.191476 | 0.572720 | 0.568778 | [-0.002453, -0.000545] |

The paired intervals resample calendar weeks with replacement 5,000 times using seed 42. Negative Brier differences favor V3. An interval containing zero does not clearly distinguish the models on that metric. Weekly blocks do not remove all dependence, and the intervals do not adjust for the multiple markets, divisions, leagues and thresholds examined. Pooled intervals are descriptive summaries of the same annual tests.

### Second-division probability quality

These scores use all eligible secondary fixtures, including forecasts below the cutoffs. They should be examined independently of the all-division scores; adding these fixtures does not imply that V3 improves every secondary market.

| Period | Second-division market | Legacy Brier | V3 Brier | Legacy log loss | V3 log loss | 95% interval: Brier difference (V3 − legacy) |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | Over 1.5 | 0.194113 | 0.194376 | 0.576335 | 0.577028 | [-0.001501, +0.002015] |
| 2025 | Over 2.5 | 0.250411 | 0.250681 | 0.694133 | 0.694615 | [-0.002474, +0.002946] |
| 2025 | Over 3.5 | 0.193789 | 0.193635 | 0.577238 | 0.576328 | [-0.001863, +0.001556] |
| 2026 | Over 1.5 | 0.173760 | 0.173666 | 0.531496 | 0.530977 | [-0.002263, +0.002197] |
| 2026 | Over 2.5 | 0.247153 | 0.248026 | 0.687373 | 0.689213 | [-0.001906, +0.003664] |
| 2026 | Over 3.5 | 0.198304 | 0.198329 | 0.583368 | 0.584537 | [-0.001773, +0.001859] |
| Combined | Over 1.5 | 0.185590 | 0.185703 | 0.557558 | 0.557743 | [-0.001305, +0.001548] |
| Combined | Over 2.5 | 0.249046 | 0.249569 | 0.691302 | 0.692353 | [-0.001491, +0.002495] |
| Combined | Over 3.5 | 0.195680 | 0.195600 | 0.579805 | 0.579766 | [-0.001317, +0.001119] |

## Coverage and uncertainty in the pooled filters

Read hit rates together with selected counts, mean forecasts and coverage. Very high hit rates from a few matches provide little evidence about future reliability. Approximate 95% Wilson intervals treat match outcomes as independent; fixtures share teams and leagues. They describe each selected group, not uncertainty in the difference between two models' hit rates.

| Goal market | Minimum forecast | Model | Correct / selected | Mean forecast | Coverage of common fixtures | Hit rate | 95% Wilson interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Over 1.5 | ≥60% | Legacy Poisson | 5,724 / 7,627 | 74.03% | 92.98% | 75.05% | 74.07%–76.01% |
| Over 1.5 | ≥60% | V3 goal component | 5,967 / 8,013 | 74.15% | 97.68% | 74.47% | 73.50%–75.41% |
| Over 1.5 | ≥70% | Legacy Poisson | 4,015 / 5,165 | 78.00% | 62.96% | 77.73% | 76.58%–78.85% |
| Over 1.5 | ≥70% | V3 goal component | 4,359 / 5,652 | 77.51% | 68.90% | 77.12% | 76.01%–78.20% |
| Over 1.5 | ≥80% | Legacy Poisson | 1,439 / 1,771 | 84.02% | 21.59% | 81.25% | 79.37%–83.00% |
| Over 1.5 | ≥80% | V3 goal component | 1,447 / 1,749 | 83.51% | 21.32% | 82.73% | 80.89%–84.43% |
| Over 2.5 | ≥60% | Legacy Poisson | 765 / 1,191 | 65.80% | 14.52% | 64.23% | 61.47%–66.90% |
| Over 2.5 | ≥60% | V3 goal component | 703 / 1,067 | 65.12% | 13.01% | 65.89% | 62.99%–68.67% |
| Over 2.5 | ≥70% | Legacy Poisson | 158 / 217 | 75.05% | 2.65% | 72.81% | 66.53%–78.30% |
| Over 2.5 | ≥70% | V3 goal component | 111 / 141 | 75.98% | 1.72% | 78.72% | 71.25%–84.67% |
| Over 2.5 | ≥80% | Legacy Poisson | 32 / 36 | 83.21% | 0.44% | 88.89% | 74.69%–95.59% |
| Over 2.5 | ≥80% | V3 goal component | 27 / 29 | 84.20% | 0.35% | 93.10% | 78.04%–98.09% |
| Over 3.5 | ≥60% | Legacy Poisson | 31 / 46 | 65.48% | 0.56% | 67.39% | 52.97%–79.13% |
| Over 3.5 | ≥60% | V3 goal component | 23 / 35 | 67.07% | 0.43% | 65.71% | 49.15%–79.17% |
| Over 3.5 | ≥70% | Legacy Poisson | 5 / 6 | 73.44% | 0.07% | 83.33% | 43.65%–96.99% |
| Over 3.5 | ≥70% | V3 goal component | 8 / 10 | 74.79% | 0.12% | 80.00% | 49.02%–94.33% |
| Over 3.5 | ≥80% | Legacy Poisson | 1 / 1 | 81.16% | 0.01% | 100.00% | 20.65%–100.00% |
| Over 3.5 | ≥80% | V3 goal component | 2 / 2 | 80.86% | 0.02% | 100.00% | 34.24%–100.00% |

## Interpretation

- **Over 1.5, forecast ≥80%:** legacy 81.25% (1,439 / 1,771) versus V3 82.73% (1,447 / 1,749). These are different selected match sets; use the annual and division tables to check consistency.
- **Over 2.5, forecast ≥70%:** legacy 72.81% (158 / 217) versus V3 78.72% (111 / 141). These are different selected match sets; use the annual and division tables to check consistency.
- **Over 3.5:** even the ≥60% filter selects only 46 legacy and 35 V3 matches across both years. Higher-cutoff results need to be read with their sample counts; the small samples do not establish reliable high-confidence performance.
- **Second-division probability scores:** V3 has a lower Brier score in 2 of 6 annual market comparisons; legacy is lower in 4. Read the differences and paired intervals separately from the all-division results; the aggregate results do not establish a secondary-division advantage.
- **Second-division high cutoffs:** neither model selects a match at ≥80% for over 2.5 or over 3.5. Those hit rates are unavailable, not zero. The ≥70% over-2.5 subset also needs its very small counts to be kept visible.

First- and second-division results should be assessed separately. Use the annual tables to check whether pooled patterns repeat; the combined sample should not conceal weaker performance in a particular year or league. These cached historical results do not establish future hit rates or betting profitability. The architecture was selected on 2024, not these later goal-market results.

## Evaluation method

- The baseline is the older [football_poisson.py](../football_poisson.py). The selected V2 classifier predicts home/draw/away and is not the baseline for these total-goal markets.
- Legacy settings stay fixed: regularization 0.01, recency decay 0.95, EWMA alpha 0.25, minimum venue history 5, and Dixon–Coles enabled. Regenerated country/year fits use only eligible prior results from their configured divisions. Legacy rho is shared within that country/year fit, not fitted on the test division.
- Legacy requires both teams' cached match statistics, sufficient historical venue games, completed rating calibration, and supported host-home and visitor-away formulas. Only completed FT matches are scored; newly appearing or insufficiently supported teams are excluded.
- V3 uses its saved pre-2025 and pre-2026 evaluation bundles, already trained on all configured divisions. Positive goal-component weights are normalized to one; the 1X2 classifier is excluded from total-goal probabilities. Model architecture and calibration selection used first-division validation, and secondary fixtures are additional evaluations without tuning. The saved models and original forecast files remain unchanged.
- Historical inputs update chronologically from completed results. Annual training boundaries require kickoff plus a three-hour result delay to precede fitting. The legacy chronological check rejects overlapping team fixtures less than three hours apart; coefficients stay fixed within each year.
- The extension checks that reconstructed primary-division forecasts and legacy rho reproduce the original comparison within 1e-10 before adding secondary fixtures. Common fixtures have matching country/fixture IDs, kickoff times, and FT goals. Tables are grouped by the configured league ID, not a team's future division.
- Saved goal means are reused to calculate all three tails. Dixon–Coles changes only 0–0, 0–1, 1–0 and 1–1. Thus `P(total ≥ 2) = PoissonSF(1, home_mean + away_mean) − exp(−home_mean − away_mean) × home_mean × away_mean × rho`. Over 2.5 and 3.5 use `PoissonSF(2, total_mean)` and `PoissonSF(3, total_mean)`. Rho is clipped to its valid per-fixture bounds. No market-specific calibration or threshold tuning is performed.

The source cutoff is **2026-09-30T23:38:15.160962+00:00**, equivalent to **30 September 2026, 18:38:15 in Colombia**. The latest common fixture in the 2026 sample kicked off at **2026-09-27T23:00:00+00:00**. Evaluation calendar boundaries use UTC.

## Files and reproduction

- [evaluation.json](evaluation.json): annual and combined metrics, division and league results, model cutoffs, rho metadata, coverage exclusions, uncertainty and input hashes.
- [threshold_summary.csv](threshold_summary.csv): all requested cutoffs with correct/incorrect counts, hit rates, coverage and intervals; scope, division, country and league ID identify each group.
- [predictions_2025.csv](predictions_2025.csv) and [predictions_2026.csv](predictions_2026.csv): common fixture probabilities and observed events across all three goal markets.
- [backtest/](backtest/): the extended annual forecast caches and legacy country/year metadata.
- The [original first-division-only over-2.5 comparison](../over25_comparison_v2_v3/README.md) and its source forecasts are preserved.

From the project directory:

```powershell
python goals_accuracy_comparison/extend_backtest.py
python goals_accuracy_comparison/generate_report.py
```

The first command reconstructs additional historical forecasts from local SQLite data and annual V3 models, refitting only the legacy annual models. The second command regenerates this report from the saved backtest inputs; it does not query the API, retrain models, or refresh the historical sample.
