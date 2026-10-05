# V3 versus V4: winner and total-goal accuracy

The comparison contains **12,320 identical held-out fixtures** in 2025 + 2026, across **21 configured leagues**. First and second divisions are listed separately.

Data cutoff: **04 October 2026, 18:38:04 Colombia time**. Evaluation periods later than this cutoff and cached league coverage are incomplete.

Fixed V3 architecture and weights; V4 refits coefficients and the previously enabled Dixon-Coles correction after excluding possession and passes. Both annual fits use identical prior fixtures. Architecture is not selected again. Each model uses its own threshold; draws are failed winner predictions. The 2025/2026 historical results have been previously inspected and are not a new untouched future holdout.

The V3 architecture and weights are held fixed for V4: **25% boosting classifier, 75% goal-rate component**. The recorded calibration method is **none**. V4 refits the coefficients after removing the excluded inputs; architecture and threshold choices are not selected again using these evaluation years.

The current V3 feature set already omits passing statistics. V4 also blocks passing fields and removes all possession inputs, including their team/venue histories and differences.

V4's previously enabled Dixon–Coles correction is refitted using held-out **2023** predictions from goal models trained before **2023-01-01T00:00:00+00:00**, with **3706** held-out matches. The refitted correction is -0.049745. This uses no evaluation-year outcomes. The correction adjusts only 0–0, 0–1, 1–0 and 1–1 scorelines, so it does not directly change over/under 2.5 or 3.5 tails.

The V3 baseline is refitted on the same current historical partitions as V4. The following check distinguishes that refit from the original saved V3 annual bundles; comparison coefficients always come from the years before each test.

| Test year | Fit cutoff | Training matches | Test matches | Saved V3 model reproduced? | All fixtures checked | Maximum probability difference | Maximum xG difference | Primary CSV matches checked | Original primary CSV reproduced? | Original CSV maximum probability difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2025-01-01T00:00:00+00:00 | 38540 | 7190 | True: reproduced | 7190 | 0 | 0 | 3725 | True | 9.71e-17 |
| 2026 | 2026-01-01T00:00:00+00:00 | 45730 | 5130 | True: reproduced | 5130 | 0 | 0 | 2612 | False | 1.78e-05 |

Model reproduction evaluates the saved annual V3 coefficients on the current historical inputs. The older prediction CSVs used the cache as it existed when V3 was trained; later backfilled fixtures can change those inputs slightly. A CSV mismatch is recorded separately above. This comparison uses the same refreshed historical inputs for both variants.

Winner selection means **max(P(home win), P(away win)) ≥ the threshold**, choosing the more probable team. An actual draw is a failure. Over 2.5 means at least 3 total goals; under 2.5 means at most 2. Over 3.5 means at least 4; under 3.5 means at most 3.

All thresholds are inclusive and cumulative. Each model applies the threshold to its own probability and may select different fixtures. Cells show **observed accuracy (correct / qualifying picks)**; a dash means no qualifying picks.

## Overall threshold results

### 2025 + 2026 (12,320 games)

| Event | Minimum probability | V3 accuracy (hits/picks) | V4 accuracy (hits/picks) | V4 − V3 |
| --- | --- | --- | --- | --- |
| Team wins | ≥60% | 68.5% (872/1273) |t 69.2% (880/1272) | +0.68 pp |
| Team wins | ≥70% | 78.5% (270/344) | 78.2% (259/331) | -0.24 pp |
| Team wins | ≥80% | 88.6% (62/70) | 85.9% (61/71) | -2.66 pp |
| Over 2.5 goals | ≥60% | 66.3% (1099/1658) | 66.1% (1083/1639) | -0.21 pp |
| Over 2.5 goals | ≥70% | 78.1% (164/210) | 79.8% (162/203) | +1.71 pp |
| Over 2.5 goals | ≥80% | 93.8% (30/32) | 92.9% (26/28) | -0.89 pp |
| Under 2.5 goals | ≥60% | 64.7% (2062/3186) | 65.0% (2045/3147) | +0.26 pp |
| Under 2.5 goals | ≥70% | 76.7% (112/146) | 76.2% (93/122) | -0.48 pp |
| Under 2.5 goals | ≥80% | — (0 picks) | — (0 picks) | — |
| Over 3.5 goals | ≥60% | 63.2% (24/38) | 61.1% (22/36) | -2.05 pp |
| Over 3.5 goals | ≥70% | 80.0% (8/10) | 88.9% (8/9) | +8.89 pp |
| Over 3.5 goals | ≥80% | 100.0% (2/2) | 100.0% (2/2) | +0.00 pp |
| Under 3.5 goals | ≥60% | 76.2% (8452/11088) | 76.2% (8485/11135) | -0.03 pp |
| Under 3.5 goals | ≥70% | 79.0% (6827/8645) | 78.9% (6816/8642) | -0.10 pp |
| Under 3.5 goals | ≥80% | 85.3% (2793/3275) | 85.0% (2783/3273) | -0.25 pp |

### 2025 (7,190 games)

| Event | Minimum probability | V3 accuracy (hits/picks) | V4 accuracy (hits/picks) | V4 − V3 |
| --- | --- | --- | --- | --- |
| Team wins | ≥60% | 69.5% (509/732) | 69.9% (511/731) | +0.37 pp |
| Team wins | ≥70% | 80.6% (154/191) | 79.8% (146/183) | -0.85 pp |
| Team wins | ≥80% | 92.1% (35/38) | 87.2% (34/39) | -4.93 pp |
| Over 2.5 goals | ≥60% | 65.1% (604/928) | 64.9% (591/911) | -0.21 pp |
| Over 2.5 goals | ≥70% | 78.3% (72/92) | 82.4% (70/85) | +4.09 pp |
| Over 2.5 goals | ≥80% | 92.9% (13/14) | 91.7% (11/12) | -1.19 pp |
| Under 2.5 goals | ≥60% | 65.4% (1216/1859) | 65.7% (1198/1823) | +0.30 pp |
| Under 2.5 goals | ≥70% | 76.9% (60/78) | 78.8% (52/66) | +1.86 pp |
| Under 2.5 goals | ≥80% | — (0 picks) | — (0 picks) | — |
| Over 3.5 goals | ≥60% | 64.7% (11/17) | 62.5% (10/16) | -2.21 pp |
| Over 3.5 goals | ≥70% | 100.0% (2/2) | 100.0% (2/2) | +0.00 pp |
| Over 3.5 goals | ≥80% | — (0 picks) | — (0 picks) | — |
| Under 3.5 goals | ≥60% | 77.0% (5037/6543) | 77.0% (5056/6569) | -0.02 pp |
| Under 3.5 goals | ≥70% | 79.6% (4041/5074) | 79.5% (4031/5068) | -0.10 pp |
| Under 3.5 goals | ≥80% | 85.4% (1636/1915) | 85.1% (1618/1902) | -0.36 pp |

### 2026 (5,130 games)

| Event | Minimum probability | V3 accuracy (hits/picks) | V4 accuracy (hits/picks) | V4 − V3 |
| --- | --- | --- | --- | --- |
| Team wins | ≥60% | 67.1% (363/541) | 68.2% (369/541) | +1.11 pp |
| Team wins | ≥70% | 75.8% (116/153) | 76.4% (113/148) | +0.53 pp |
| Team wins | ≥80% | 84.4% (27/32) | 84.4% (27/32) | +0.00 pp |
| Over 2.5 goals | ≥60% | 67.8% (495/730) | 67.6% (492/728) | -0.23 pp |
| Over 2.5 goals | ≥70% | 78.0% (92/118) | 78.0% (92/118) | +0.00 pp |
| Over 2.5 goals | ≥80% | 94.4% (17/18) | 93.8% (15/16) | -0.69 pp |
| Under 2.5 goals | ≥60% | 63.8% (846/1327) | 64.0% (847/1324) | +0.22 pp |
| Under 2.5 goals | ≥70% | 76.5% (52/68) | 73.2% (41/56) | -3.26 pp |
| Under 2.5 goals | ≥80% | — (0 picks) | — (0 picks) | — |
| Over 3.5 goals | ≥60% | 61.9% (13/21) | 60.0% (12/20) | -1.90 pp |
| Over 3.5 goals | ≥70% | 75.0% (6/8) | 85.7% (6/7) | +10.71 pp |
| Over 3.5 goals | ≥80% | 100.0% (2/2) | 100.0% (2/2) | +0.00 pp |
| Under 3.5 goals | ≥60% | 75.1% (3415/4545) | 75.1% (3429/4566) | -0.04 pp |
| Under 3.5 goals | ≥70% | 78.0% (2786/3571) | 77.9% (2785/3574) | -0.09 pp |
| Under 3.5 goals | ≥80% | 85.1% (1157/1360) | 85.0% (1165/1371) | -0.10 pp |

## Winner accuracy separately for every league

### 2025 + 2026

| Country | Division | League ID | Games | V3 ≥60% | V4 ≥60% | V3 ≥70% | V4 ≥70% | V3 ≥80% | V4 ≥80% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 911 | 63.6% (28/44) | 62.9% (22/35) | 50.0% (2/4) | — (0 picks) | — (0 picks) | — (0 picks) |
| Argentina | Second | 129 | 1207 | 33.3% (1/3) | 33.3% (1/3) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Chile | First | 265 | 423 | 64.9% (37/57) | 66.1% (37/56) | 75.0% (9/12) | 83.3% (5/6) | — (0 picks) | — (0 picks) |
| Chile | Second | 266 | 448 | 33.3% (1/3) | 33.3% (1/3) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Colombia | First | 239 | 766 | 68.8% (64/93) | 68.8% (64/93) | 92.3% (12/13) | 91.7% (11/12) | — (0 picks) | — (0 picks) |
| Colombia | Second | 240 | 535 | 58.8% (20/34) | 63.2% (24/38) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| England | First | 39 | 618 | 64.7% (97/150) | 65.3% (98/150) | 72.0% (36/50) | 70.0% (35/50) | 100.0% (6/6) | 100.0% (6/6) |
| England | Second | 40 | 921 | 69.6% (55/79) | 71.1% (54/76) | 69.2% (9/13) | 69.2% (9/13) | 50.0% (1/2) | 50.0% (1/2) |
| Germany | First | 78 | 516 | 72.4% (105/145) | 72.7% (101/139) | 81.3% (61/75) | 80.9% (55/68) | 82.1% (23/28) | 79.3% (23/29) |
| Germany | Second | 79 | 516 | 56.4% (22/39) | 55.0% (22/40) | 33.3% (1/3) | 25.0% (1/4) | — (0 picks) | — (0 picks) |
| Iceland | First | 164 | 302 | 68.6% (24/35) | 69.4% (25/36) | 71.4% (5/7) | 62.5% (5/8) | — (0 picks) | — (0 picks) |
| Iceland | Second | 165 | 273 | 90.0% (18/20) | 90.9% (20/22) | 75.0% (3/4) | 80.0% (4/5) | — (0 picks) | — (0 picks) |
| Ireland | First | 357 | 181 | 57.1% (12/21) | 64.7% (11/17) | 100.0% (1/1) | 100.0% (1/1) | — (0 picks) | — (0 picks) |
| Ireland | Second | 358 | 183 | 77.8% (7/9) | 77.8% (7/9) | — (0 picks) | 0.0% (0/1) | — (0 picks) | — (0 picks) |
| Italy | First | 135 | 632 | 72.2% (104/144) | 71.8% (102/142) | 79.1% (34/43) | 80.5% (33/41) | 83.3% (5/6) | 83.3% (5/6) |
| Italy | Second | 136 | 629 | 82.5% (33/40) | 86.0% (37/43) | 90.0% (9/10) | 87.5% (7/8) | — (0 picks) | — (0 picks) |
| South Korea | First | 292 | 411 | 46.2% (6/13) | 42.9% (6/14) | 100.0% (1/1) | 100.0% (1/1) | — (0 picks) | — (0 picks) |
| South Korea | Second | 293 | 491 | 57.1% (8/14) | 60.0% (9/15) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Spain | First | 140 | 648 | 78.4% (116/148) | 79.1% (121/153) | 87.2% (68/78) | 89.0% (73/82) | 96.3% (26/27) | 92.6% (25/27) |
| Spain | Second | 141 | 772 | 72.9% (35/48) | 71.1% (32/45) | 83.3% (5/6) | 66.7% (4/6) | — (0 picks) | — (0 picks) |
| United States | First | 253 | 937 | 59.0% (79/134) | 60.1% (86/143) | 58.3% (14/24) | 60.0% (15/25) | 100.0% (1/1) | 100.0% (1/1) |

### 2025

| Country | Division | League ID | Games | V3 ≥60% | V4 ≥60% | V3 ≥70% | V4 ≥70% | V3 ≥80% | V4 ≥80% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 503 | 50.0% (12/24) | 50.0% (10/20) | 50.0% (1/2) | — (0 picks) | — (0 picks) | — (0 picks) |
| Argentina | Second | 129 | 632 | 33.3% (1/3) | 33.3% (1/3) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Chile | First | 265 | 240 | 75.0% (21/28) | 75.9% (22/29) | 71.4% (5/7) | 66.7% (2/3) | — (0 picks) | — (0 picks) |
| Chile | Second | 266 | 251 | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Colombia | First | 239 | 450 | 71.4% (35/49) | 70.0% (35/50) | 100.0% (4/4) | 100.0% (3/3) | — (0 picks) | — (0 picks) |
| Colombia | Second | 240 | 306 | 64.0% (16/25) | 67.9% (19/28) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| England | First | 39 | 378 | 69.2% (74/107) | 68.5% (74/108) | 78.4% (29/37) | 76.3% (29/38) | 100.0% (6/6) | 100.0% (6/6) |
| England | Second | 40 | 558 | 70.5% (31/44) | 72.5% (29/40) | 70.0% (7/10) | 70.0% (7/10) | — (0 picks) | — (0 picks) |
| Germany | First | 78 | 308 | 74.1% (60/81) | 72.7% (56/77) | 83.7% (36/43) | 81.1% (30/37) | 83.3% (15/18) | 78.9% (15/19) |
| Germany | Second | 79 | 307 | 59.1% (13/22) | 54.5% (12/22) | 33.3% (1/3) | 33.3% (1/3) | — (0 picks) | — (0 picks) |
| Iceland | First | 164 | 162 | 70.0% (14/20) | 71.4% (15/21) | 100.0% (2/2) | 66.7% (2/3) | — (0 picks) | — (0 picks) |
| Iceland | Second | 165 | 137 | 75.0% (3/4) | 75.0% (3/4) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Ireland | First | 357 | 181 | 57.1% (12/21) | 64.7% (11/17) | 100.0% (1/1) | 100.0% (1/1) | — (0 picks) | — (0 picks) |
| Ireland | Second | 358 | 183 | 77.8% (7/9) | 77.8% (7/9) | — (0 picks) | 0.0% (0/1) | — (0 picks) | — (0 picks) |
| Italy | First | 135 | 368 | 72.2% (57/79) | 71.8% (56/78) | 85.0% (17/20) | 84.2% (16/19) | 100.0% (1/1) | 100.0% (1/1) |
| Italy | Second | 136 | 369 | 75.0% (12/16) | 76.5% (13/17) | 50.0% (1/2) | 100.0% (1/1) | — (0 picks) | — (0 picks) |
| South Korea | First | 292 | 232 | 45.5% (5/11) | 41.7% (5/12) | 100.0% (1/1) | 100.0% (1/1) | — (0 picks) | — (0 picks) |
| South Korea | Second | 293 | 275 | 60.0% (6/10) | 63.6% (7/11) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Spain | First | 140 | 370 | 76.1% (70/92) | 77.9% (74/95) | 87.2% (41/47) | 88.5% (46/52) | 100.0% (13/13) | 92.3% (12/13) |
| Spain | Second | 141 | 447 | 76.9% (20/26) | 75.0% (18/24) | 80.0% (4/5) | 75.0% (3/4) | — (0 picks) | — (0 picks) |
| United States | First | 253 | 533 | 65.6% (40/61) | 66.7% (44/66) | 57.1% (4/7) | 57.1% (4/7) | — (0 picks) | — (0 picks) |

### 2026

| Country | Division | League ID | Games | V3 ≥60% | V4 ≥60% | V3 ≥70% | V4 ≥70% | V3 ≥80% | V4 ≥80% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 408 | 80.0% (16/20) | 80.0% (12/15) | 50.0% (1/2) | — (0 picks) | — (0 picks) | — (0 picks) |
| Argentina | Second | 129 | 575 | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Chile | First | 265 | 183 | 55.2% (16/29) | 55.6% (15/27) | 80.0% (4/5) | 100.0% (3/3) | — (0 picks) | — (0 picks) |
| Chile | Second | 266 | 197 | 33.3% (1/3) | 33.3% (1/3) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Colombia | First | 239 | 316 | 65.9% (29/44) | 67.4% (29/43) | 88.9% (8/9) | 88.9% (8/9) | — (0 picks) | — (0 picks) |
| Colombia | Second | 240 | 229 | 44.4% (4/9) | 50.0% (5/10) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| England | First | 39 | 240 | 53.5% (23/43) | 57.1% (24/42) | 53.8% (7/13) | 50.0% (6/12) | — (0 picks) | — (0 picks) |
| England | Second | 40 | 363 | 68.6% (24/35) | 69.4% (25/36) | 66.7% (2/3) | 66.7% (2/3) | 50.0% (1/2) | 50.0% (1/2) |
| Germany | First | 78 | 208 | 70.3% (45/64) | 72.6% (45/62) | 78.1% (25/32) | 80.6% (25/31) | 80.0% (8/10) | 80.0% (8/10) |
| Germany | Second | 79 | 209 | 52.9% (9/17) | 55.6% (10/18) | — (0 picks) | 0.0% (0/1) | — (0 picks) | — (0 picks) |
| Iceland | First | 164 | 140 | 66.7% (10/15) | 66.7% (10/15) | 60.0% (3/5) | 60.0% (3/5) | — (0 picks) | — (0 picks) |
| Iceland | Second | 165 | 136 | 93.8% (15/16) | 94.4% (17/18) | 75.0% (3/4) | 80.0% (4/5) | — (0 picks) | — (0 picks) |
| Ireland | First | 357 | 0 | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Ireland | Second | 358 | 0 | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Italy | First | 135 | 264 | 72.3% (47/65) | 71.9% (46/64) | 73.9% (17/23) | 77.3% (17/22) | 80.0% (4/5) | 80.0% (4/5) |
| Italy | Second | 136 | 260 | 87.5% (21/24) | 92.3% (24/26) | 100.0% (8/8) | 85.7% (6/7) | — (0 picks) | — (0 picks) |
| South Korea | First | 292 | 179 | 50.0% (1/2) | 50.0% (1/2) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| South Korea | Second | 293 | 216 | 50.0% (2/4) | 50.0% (2/4) | — (0 picks) | — (0 picks) | — (0 picks) | — (0 picks) |
| Spain | First | 140 | 278 | 82.1% (46/56) | 81.0% (47/58) | 87.1% (27/31) | 90.0% (27/30) | 92.9% (13/14) | 92.9% (13/14) |
| Spain | Second | 141 | 325 | 68.2% (15/22) | 66.7% (14/21) | 100.0% (1/1) | 50.0% (1/2) | — (0 picks) | — (0 picks) |
| United States | First | 253 | 404 | 53.4% (39/73) | 54.5% (42/77) | 58.8% (10/17) | 61.1% (11/18) | 100.0% (1/1) | 100.0% (1/1) |

The four goal markets have separate tables for every league and year in [GOALS_BY_LEAGUE.md](GOALS_BY_LEAGUE.md).

## Probability quality on all identical fixtures

The following scores use every fixture in the period. Home/draw/away accuracy includes draw predictions. Brier score and log loss are better when lower; their paired differences use the same fixtures for both models.

| Period | Model | Home/draw/away accuracy | Brier | Log loss |
| --- | --- | --- | --- | --- |
| combined | V3 | 47.92% | 0.61946 | 1.03113 |
| combined | V4 | 47.84% | 0.61957 | 1.03130 |
| 2025 | V3 | 48.12% | 0.61845 | 1.02968 |
| 2025 | V4 | 48.14% | 0.61841 | 1.02964 |
| 2026 | V3 | 47.64% | 0.62089 | 1.03316 |
| 2026 | V4 | 47.43% | 0.62120 | 1.03364 |

| Period | Event | Model | Mean predicted | Observed rate | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- |
| combined | Over 2.5 goals | V3 | 47.27% | 48.36% | 0.23892 | 0.67037 |
| combined | Over 2.5 goals | V4 | 47.26% | 48.36% | 0.23896 | 0.67044 |
| combined | Over 3.5 goals | V3 | 26.37% | 26.00% | 0.18282 | 0.54837 |
| combined | Over 3.5 goals | V4 | 26.35% | 26.00% | 0.18285 | 0.54840 |
| 2025 | Over 2.5 goals | V3 | 47.12% | 47.13% | 0.23861 | 0.66979 |
| 2025 | Over 2.5 goals | V4 | 47.10% | 47.13% | 0.23869 | 0.66995 |
| 2025 | Over 3.5 goals | V3 | 26.20% | 24.87% | 0.17885 | 0.53961 |
| 2025 | Over 3.5 goals | V4 | 26.17% | 24.87% | 0.17897 | 0.53986 |
| 2026 | Over 2.5 goals | V3 | 47.48% | 50.08% | 0.23936 | 0.67118 |
| 2026 | Over 2.5 goals | V4 | 47.49% | 50.08% | 0.23933 | 0.67112 |
| 2026 | Over 3.5 goals | V3 | 26.62% | 27.58% | 0.18840 | 0.56064 |
| 2026 | Over 3.5 goals | V4 | 26.61% | 27.58% | 0.18829 | 0.56038 |

Complementary over/under events have identical unfiltered Brier scores and log losses. Their threshold hit rates differ because they select different games. Goal probabilities come from the normalized goal component mixture; home/draw/away probabilities use the selected full model and its recorded calibration.

## Interpretation and reproducibility

A higher filtered hit rate may come with fewer predictions and is not by itself a paired estimate of improvement. The JSON includes the intersection of games qualifying for both models. On a common set for one fixed goal event, both models necessarily have the same correct count; winner picks can differ and their agreement counts are recorded.

Every model/event/league/year has sample counts, coverage, mean predicted probability and a 95% Wilson interval in JSON and CSV. These intervals treat games as independent and do not account for multiple league comparisons. Small 80% samples can produce unstable hit rates. Overall paired score differences also include 95% intervals from 2,000 calendar-week bootstrap samples; these capture uncertainty conditional on the fixed training and model-selection procedure.

The evaluation uses historical predictions generated with model coefficients fitted before the evaluation year and historical features available before each kickoff. Final models trained on the complete dataset must not be evaluated retrospectively on their own training fixtures. Both variants need the same source snapshot, cutoff and eligible fixtures. The supplied metadata records whether V3 was reused from saved annual bundles or refitted for this comparison.

Architecture and calibration choices should be fixed using earlier validation data. The project's 2025/2026 periods have already been inspected and are historical backtests, so future fixtures are needed for a fresh confirmation. Threshold accuracy is not betting ROI: no bookmaker prices or staking policy enter these accuracy calculations.

- [evaluation.json](evaluation.json): metadata, all annual and per-league metrics, common-qualified intersections, calibration bins and paired scores.
- [threshold_summary.csv](threshold_summary.csv): model/own-filter and common-qualified rows for all five events and all periods.
- [GOALS_BY_LEAGUE.md](GOALS_BY_LEAGUE.md): over/under 2.5 and 3.5 results separately by league.

Recorded feature ablation:

```json
{
  "excluded_statistics": [
    "possession",
    "passes",
    "passes_completed",
    "pass_accuracy"
  ],
  "excluded_feature_columns": [
    "home_all_possession_5",
    "home_all_possession_20",
    "home_home_possession_5",
    "home_home_possession_20",
    "away_all_possession_5",
    "away_all_possession_20",
    "away_away_possession_5",
    "away_away_possession_20",
    "diff_possession_5",
    "diff_possession_20"
  ],
  "baseline_feature_count": 205,
  "baseline_selected_input_feature_count": 167,
  "v4_feature_count": 195,
  "v4_selected_input_feature_count": 157
}
```

## Commands

Train the separate V4 model and regenerate the chronological comparison:

```powershell
python football_model_v4.py train
```

Verify and summarize the saved comparison after training:

```powershell
python football_model_v4.py backtest
```

Predict a Championship match with V4:

```powershell
python football_model_v4.py predict --country england --league-id 40 --home Middlesbrough --away Wolves
```
