# Winner accuracy at 60%, 70% and 80%, separately for every league

This analysis compares the fixed V2 reference with the retrained V3 annual evaluation bundles. Every configured first- and second-division league has its own row; leagues are never combined into one accuracy figure.

The combined period covers cached matches in **2025 and 2026 through 02 October 2026, 22:21:42 Colombia time**. The full comparison contains **12,286 distinct matches across 21 leagues**. The 2026 period and cache coverage are incomplete.

A game qualifies when **max(P(home win), P(away win)) ≥ the threshold**. The pick is the team with the higher win probability. An actual draw counts as a failed winner prediction. Thresholds are inclusive and cumulative: an 80% pick also belongs to the 70% and 60% groups.

Each cell shows **observed accuracy (correct / qualifying picks)**. A dash means no qualifying predictions. V2 and V3 apply the threshold to their own probabilities and can therefore select different fixtures. Differences between these filtered rates are descriptive, rather than a paired estimate of model improvement.

**No cached 2026 evaluation matches:** Ireland (357), Ireland (358). Their combined results therefore use 2025 matches only.

## 2025 + 2026: league results

### Win probability ≥60%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 904 | 63.0% (63/100) | 63.6% (28/44) | +0.6 pp |
| Argentina | Second | 129 | 1190 | 60.2% (50/83) | 33.3% (1/3) | -26.9 pp |
| Chile | First | 265 | 423 | 63.9% (46/72) | 64.9% (37/57) | +1.0 pp |
| Chile | Second | 266 | 448 | 45.5% (5/11) | 33.3% (1/3) | -12.1 pp |
| Colombia | First | 239 | 761 | 64.4% (103/160) | 69.6% (64/92) | +5.2 pp |
| Colombia | Second | 240 | 532 | 65.2% (60/92) | 58.8% (20/34) | -6.4 pp |
| England | First | 39 | 618 | 61.7% (111/180) | 64.7% (97/150) | +3.0 pp |
| England | Second | 40 | 921 | 69.5% (73/105) | 69.6% (55/79) | +0.1 pp |
| Germany | First | 78 | 516 | 71.1% (108/152) | 72.4% (105/145) | +1.4 pp |
| Germany | Second | 79 | 516 | 62.5% (30/48) | 56.4% (22/39) | -6.1 pp |
| Iceland | First | 164 | 302 | 64.9% (48/74) | 68.6% (24/35) | +3.7 pp |
| Iceland | Second | 165 | 273 | 71.4% (40/56) | 90.0% (18/20) | +18.6 pp |
| Ireland | First | 357 | 181 | 60.7% (17/28) | 57.1% (12/21) | -3.6 pp |
| Ireland | Second | 358 | 183 | 68.2% (15/22) | 77.8% (7/9) | +9.6 pp |
| Italy | First | 135 | 632 | 67.7% (126/186) | 72.2% (104/144) | +4.5 pp |
| Italy | Second | 136 | 629 | 73.0% (46/63) | 82.5% (33/40) | +9.5 pp |
| South Korea | First | 292 | 411 | 41.0% (16/39) | 46.2% (6/13) | +5.1 pp |
| South Korea | Second | 293 | 491 | 62.7% (32/51) | 57.1% (8/14) | -5.6 pp |
| Spain | First | 140 | 648 | 77.6% (132/170) | 78.4% (116/148) | +0.7 pp |
| Spain | Second | 141 | 772 | 74.2% (46/62) | 72.9% (35/48) | -1.3 pp |
| United States | First | 253 | 935 | 62.9% (88/140) | 58.6% (78/133) | -4.2 pp |

### Win probability ≥70%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 904 | 65.0% (13/20) | 50.0% (2/4) | -15.0 pp |
| Argentina | Second | 129 | 1190 | — (0 picks) | — (0 picks) | — |
| Chile | First | 265 | 423 | 59.1% (13/22) | 75.0% (9/12) | +15.9 pp |
| Chile | Second | 266 | 448 | — (0 picks) | — (0 picks) | — |
| Colombia | First | 239 | 761 | 73.0% (27/37) | 92.3% (12/13) | +19.3 pp |
| Colombia | Second | 240 | 532 | 61.5% (8/13) | — (0 picks) | — |
| England | First | 39 | 618 | 73.0% (54/74) | 72.0% (36/50) | -1.0 pp |
| England | Second | 40 | 921 | 64.3% (18/28) | 69.2% (9/13) | +4.9 pp |
| Germany | First | 78 | 516 | 79.0% (64/81) | 81.3% (61/75) | +2.3 pp |
| Germany | Second | 79 | 516 | 28.6% (2/7) | 33.3% (1/3) | +4.8 pp |
| Iceland | First | 164 | 302 | 76.2% (16/21) | 71.4% (5/7) | -4.8 pp |
| Iceland | Second | 165 | 273 | 72.7% (8/11) | 75.0% (3/4) | +2.3 pp |
| Ireland | First | 357 | 181 | 100.0% (3/3) | 100.0% (1/1) | +0.0 pp |
| Ireland | Second | 358 | 183 | 100.0% (2/2) | — (0 picks) | — |
| Italy | First | 135 | 632 | 72.0% (54/75) | 79.1% (34/43) | +7.1 pp |
| Italy | Second | 136 | 629 | 83.3% (15/18) | 90.0% (9/10) | +6.7 pp |
| South Korea | First | 292 | 411 | 66.7% (2/3) | 100.0% (1/1) | +33.3 pp |
| South Korea | Second | 293 | 491 | — (0 picks) | — (0 picks) | — |
| Spain | First | 140 | 648 | 90.8% (69/76) | 87.2% (68/78) | -3.6 pp |
| Spain | Second | 141 | 772 | 83.3% (10/12) | 83.3% (5/6) | +0.0 pp |
| United States | First | 253 | 935 | 55.6% (15/27) | 58.3% (14/24) | +2.8 pp |

### Win probability ≥80%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 904 | — (0 picks) | — (0 picks) | — |
| Argentina | Second | 129 | 1190 | — (0 picks) | — (0 picks) | — |
| Chile | First | 265 | 423 | 100.0% (1/1) | — (0 picks) | — |
| Chile | Second | 266 | 448 | — (0 picks) | — (0 picks) | — |
| Colombia | First | 239 | 761 | — (0 picks) | — (0 picks) | — |
| Colombia | Second | 240 | 532 | — (0 picks) | — (0 picks) | — |
| England | First | 39 | 618 | 77.8% (7/9) | 100.0% (6/6) | +22.2 pp |
| England | Second | 40 | 921 | 80.0% (4/5) | 50.0% (1/2) | -30.0 pp |
| Germany | First | 78 | 516 | 81.0% (17/21) | 82.1% (23/28) | +1.2 pp |
| Germany | Second | 79 | 516 | — (0 picks) | — (0 picks) | — |
| Iceland | First | 164 | 302 | — (0 picks) | — (0 picks) | — |
| Iceland | Second | 165 | 273 | — (0 picks) | — (0 picks) | — |
| Ireland | First | 357 | 181 | — (0 picks) | — (0 picks) | — |
| Ireland | Second | 358 | 183 | — (0 picks) | — (0 picks) | — |
| Italy | First | 135 | 632 | 83.3% (10/12) | 83.3% (5/6) | +0.0 pp |
| Italy | Second | 136 | 629 | 100.0% (2/2) | — (0 picks) | — |
| South Korea | First | 292 | 411 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 491 | — (0 picks) | — (0 picks) | — |
| Spain | First | 140 | 648 | 96.3% (26/27) | 96.3% (26/27) | +0.0 pp |
| Spain | Second | 141 | 772 | — (0 picks) | — (0 picks) | — |
| United States | First | 253 | 935 | 50.0% (1/2) | 100.0% (1/1) | +50.0 pp |

## 2025: league results

### Win probability ≥60%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 503 | 61.5% (32/52) | 50.0% (12/24) | -11.5 pp |
| Argentina | Second | 129 | 632 | 63.4% (26/41) | 33.3% (1/3) | -30.1 pp |
| Chile | First | 265 | 240 | 68.4% (26/38) | 75.0% (21/28) | +6.6 pp |
| Chile | Second | 266 | 251 | 100.0% (1/1) | — (0 picks) | — |
| Colombia | First | 239 | 450 | 63.0% (63/100) | 71.4% (35/49) | +8.4 pp |
| Colombia | Second | 240 | 306 | 74.6% (47/63) | 64.0% (16/25) | -10.6 pp |
| England | First | 39 | 378 | 65.8% (79/120) | 69.2% (74/107) | +3.3 pp |
| England | Second | 40 | 558 | 66.0% (35/53) | 70.5% (31/44) | +4.4 pp |
| Germany | First | 78 | 308 | 69.1% (65/94) | 74.1% (60/81) | +4.9 pp |
| Germany | Second | 79 | 307 | 69.2% (18/26) | 59.1% (13/22) | -10.1 pp |
| Iceland | First | 164 | 162 | 67.5% (27/40) | 70.0% (14/20) | +2.5 pp |
| Iceland | Second | 165 | 137 | 72.2% (13/18) | 75.0% (3/4) | +2.8 pp |
| Ireland | First | 357 | 181 | 60.7% (17/28) | 57.1% (12/21) | -3.6 pp |
| Ireland | Second | 358 | 183 | 68.2% (15/22) | 77.8% (7/9) | +9.6 pp |
| Italy | First | 135 | 368 | 65.7% (67/102) | 72.2% (57/79) | +6.5 pp |
| Italy | Second | 136 | 369 | 60.6% (20/33) | 75.0% (12/16) | +14.4 pp |
| South Korea | First | 292 | 232 | 44.0% (11/25) | 45.5% (5/11) | +1.5 pp |
| South Korea | Second | 293 | 275 | 64.7% (22/34) | 60.0% (6/10) | -4.7 pp |
| Spain | First | 140 | 370 | 76.1% (83/109) | 76.1% (70/92) | -0.1 pp |
| Spain | Second | 141 | 447 | 75.0% (27/36) | 76.9% (20/26) | +1.9 pp |
| United States | First | 253 | 533 | 70.7% (53/75) | 65.6% (40/61) | -5.1 pp |

### Win probability ≥70%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 503 | 55.6% (5/9) | 50.0% (1/2) | -5.6 pp |
| Argentina | Second | 129 | 632 | — (0 picks) | — (0 picks) | — |
| Chile | First | 265 | 240 | 60.0% (9/15) | 71.4% (5/7) | +11.4 pp |
| Chile | Second | 266 | 251 | — (0 picks) | — (0 picks) | — |
| Colombia | First | 239 | 450 | 78.9% (15/19) | 100.0% (4/4) | +21.1 pp |
| Colombia | Second | 240 | 306 | 66.7% (8/12) | — (0 picks) | — |
| England | First | 39 | 378 | 74.5% (41/55) | 78.4% (29/37) | +3.8 pp |
| England | Second | 40 | 558 | 66.7% (10/15) | 70.0% (7/10) | +3.3 pp |
| Germany | First | 78 | 308 | 75.0% (33/44) | 83.7% (36/43) | +8.7 pp |
| Germany | Second | 79 | 307 | 33.3% (1/3) | 33.3% (1/3) | +0.0 pp |
| Iceland | First | 164 | 162 | 72.7% (8/11) | 100.0% (2/2) | +27.3 pp |
| Iceland | Second | 165 | 137 | 71.4% (5/7) | — (0 picks) | — |
| Ireland | First | 357 | 181 | 100.0% (3/3) | 100.0% (1/1) | +0.0 pp |
| Ireland | Second | 358 | 183 | 100.0% (2/2) | — (0 picks) | — |
| Italy | First | 135 | 368 | 73.0% (27/37) | 85.0% (17/20) | +12.0 pp |
| Italy | Second | 136 | 369 | 60.0% (3/5) | 50.0% (1/2) | -10.0 pp |
| South Korea | First | 292 | 232 | 50.0% (1/2) | 100.0% (1/1) | +50.0 pp |
| South Korea | Second | 293 | 275 | — (0 picks) | — (0 picks) | — |
| Spain | First | 140 | 370 | 89.6% (43/48) | 87.2% (41/47) | -2.3 pp |
| Spain | Second | 141 | 447 | 77.8% (7/9) | 80.0% (4/5) | +2.2 pp |
| United States | First | 253 | 533 | 63.6% (7/11) | 57.1% (4/7) | -6.5 pp |

### Win probability ≥80%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 503 | — (0 picks) | — (0 picks) | — |
| Argentina | Second | 129 | 632 | — (0 picks) | — (0 picks) | — |
| Chile | First | 265 | 240 | — (0 picks) | — (0 picks) | — |
| Chile | Second | 266 | 251 | — (0 picks) | — (0 picks) | — |
| Colombia | First | 239 | 450 | — (0 picks) | — (0 picks) | — |
| Colombia | Second | 240 | 306 | — (0 picks) | — (0 picks) | — |
| England | First | 39 | 378 | 75.0% (6/8) | 100.0% (6/6) | +25.0 pp |
| England | Second | 40 | 558 | 100.0% (3/3) | — (0 picks) | — |
| Germany | First | 78 | 308 | 83.3% (10/12) | 83.3% (15/18) | +0.0 pp |
| Germany | Second | 79 | 307 | — (0 picks) | — (0 picks) | — |
| Iceland | First | 164 | 162 | — (0 picks) | — (0 picks) | — |
| Iceland | Second | 165 | 137 | — (0 picks) | — (0 picks) | — |
| Ireland | First | 357 | 181 | — (0 picks) | — (0 picks) | — |
| Ireland | Second | 358 | 183 | — (0 picks) | — (0 picks) | — |
| Italy | First | 135 | 368 | 100.0% (1/1) | 100.0% (1/1) | +0.0 pp |
| Italy | Second | 136 | 369 | — (0 picks) | — (0 picks) | — |
| South Korea | First | 292 | 232 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 275 | — (0 picks) | — (0 picks) | — |
| Spain | First | 140 | 370 | 100.0% (9/9) | 100.0% (13/13) | +0.0 pp |
| Spain | Second | 141 | 447 | — (0 picks) | — (0 picks) | — |
| United States | First | 253 | 533 | — (0 picks) | — (0 picks) | — |

## 2026: league results

### Win probability ≥60%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 401 | 64.6% (31/48) | 80.0% (16/20) | +15.4 pp |
| Argentina | Second | 129 | 558 | 57.1% (24/42) | — (0 picks) | — |
| Chile | First | 265 | 183 | 58.8% (20/34) | 55.2% (16/29) | -3.7 pp |
| Chile | Second | 266 | 197 | 40.0% (4/10) | 33.3% (1/3) | -6.7 pp |
| Colombia | First | 239 | 311 | 66.7% (40/60) | 67.4% (29/43) | +0.8 pp |
| Colombia | Second | 240 | 226 | 44.8% (13/29) | 44.4% (4/9) | -0.4 pp |
| England | First | 39 | 240 | 53.3% (32/60) | 53.5% (23/43) | +0.2 pp |
| England | Second | 40 | 363 | 73.1% (38/52) | 68.6% (24/35) | -4.5 pp |
| Germany | First | 78 | 208 | 74.1% (43/58) | 70.3% (45/64) | -3.8 pp |
| Germany | Second | 79 | 209 | 54.5% (12/22) | 52.9% (9/17) | -1.6 pp |
| Iceland | First | 164 | 140 | 61.8% (21/34) | 66.7% (10/15) | +4.9 pp |
| Iceland | Second | 165 | 136 | 71.1% (27/38) | 93.8% (15/16) | +22.7 pp |
| Ireland | First | 357 | 0 | — (0 picks) | — (0 picks) | — |
| Ireland | Second | 358 | 0 | — (0 picks) | — (0 picks) | — |
| Italy | First | 135 | 264 | 70.2% (59/84) | 72.3% (47/65) | +2.1 pp |
| Italy | Second | 136 | 260 | 86.7% (26/30) | 87.5% (21/24) | +0.8 pp |
| South Korea | First | 292 | 179 | 35.7% (5/14) | 50.0% (1/2) | +14.3 pp |
| South Korea | Second | 293 | 216 | 58.8% (10/17) | 50.0% (2/4) | -8.8 pp |
| Spain | First | 140 | 278 | 80.3% (49/61) | 82.1% (46/56) | +1.8 pp |
| Spain | Second | 141 | 325 | 73.1% (19/26) | 68.2% (15/22) | -4.9 pp |
| United States | First | 253 | 402 | 53.8% (35/65) | 52.8% (38/72) | -1.1 pp |

### Win probability ≥70%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 401 | 72.7% (8/11) | 50.0% (1/2) | -22.7 pp |
| Argentina | Second | 129 | 558 | — (0 picks) | — (0 picks) | — |
| Chile | First | 265 | 183 | 57.1% (4/7) | 80.0% (4/5) | +22.9 pp |
| Chile | Second | 266 | 197 | — (0 picks) | — (0 picks) | — |
| Colombia | First | 239 | 311 | 66.7% (12/18) | 88.9% (8/9) | +22.2 pp |
| Colombia | Second | 240 | 226 | 0.0% (0/1) | — (0 picks) | — |
| England | First | 39 | 240 | 68.4% (13/19) | 53.8% (7/13) | -14.6 pp |
| England | Second | 40 | 363 | 61.5% (8/13) | 66.7% (2/3) | +5.1 pp |
| Germany | First | 78 | 208 | 83.8% (31/37) | 78.1% (25/32) | -5.7 pp |
| Germany | Second | 79 | 209 | 25.0% (1/4) | — (0 picks) | — |
| Iceland | First | 164 | 140 | 80.0% (8/10) | 60.0% (3/5) | -20.0 pp |
| Iceland | Second | 165 | 136 | 75.0% (3/4) | 75.0% (3/4) | +0.0 pp |
| Ireland | First | 357 | 0 | — (0 picks) | — (0 picks) | — |
| Ireland | Second | 358 | 0 | — (0 picks) | — (0 picks) | — |
| Italy | First | 135 | 264 | 71.1% (27/38) | 73.9% (17/23) | +2.9 pp |
| Italy | Second | 136 | 260 | 92.3% (12/13) | 100.0% (8/8) | +7.7 pp |
| South Korea | First | 292 | 179 | 100.0% (1/1) | — (0 picks) | — |
| South Korea | Second | 293 | 216 | — (0 picks) | — (0 picks) | — |
| Spain | First | 140 | 278 | 92.9% (26/28) | 87.1% (27/31) | -5.8 pp |
| Spain | Second | 141 | 325 | 100.0% (3/3) | 100.0% (1/1) | +0.0 pp |
| United States | First | 253 | 402 | 50.0% (8/16) | 58.8% (10/17) | +8.8 pp |

### Win probability ≥80%

| Country | Division | League ID | Available games | V2 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V2 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 401 | — (0 picks) | — (0 picks) | — |
| Argentina | Second | 129 | 558 | — (0 picks) | — (0 picks) | — |
| Chile | First | 265 | 183 | 100.0% (1/1) | — (0 picks) | — |
| Chile | Second | 266 | 197 | — (0 picks) | — (0 picks) | — |
| Colombia | First | 239 | 311 | — (0 picks) | — (0 picks) | — |
| Colombia | Second | 240 | 226 | — (0 picks) | — (0 picks) | — |
| England | First | 39 | 240 | 100.0% (1/1) | — (0 picks) | — |
| England | Second | 40 | 363 | 50.0% (1/2) | 50.0% (1/2) | +0.0 pp |
| Germany | First | 78 | 208 | 77.8% (7/9) | 80.0% (8/10) | +2.2 pp |
| Germany | Second | 79 | 209 | — (0 picks) | — (0 picks) | — |
| Iceland | First | 164 | 140 | — (0 picks) | — (0 picks) | — |
| Iceland | Second | 165 | 136 | — (0 picks) | — (0 picks) | — |
| Ireland | First | 357 | 0 | — (0 picks) | — (0 picks) | — |
| Ireland | Second | 358 | 0 | — (0 picks) | — (0 picks) | — |
| Italy | First | 135 | 264 | 81.8% (9/11) | 80.0% (4/5) | -1.8 pp |
| Italy | Second | 136 | 260 | 100.0% (2/2) | — (0 picks) | — |
| South Korea | First | 292 | 179 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 216 | — (0 picks) | — (0 picks) | — |
| Spain | First | 140 | 278 | 94.4% (17/18) | 92.9% (13/14) | -1.6 pp |
| Spain | Second | 141 | 325 | — (0 picks) | — (0 picks) | — |
| United States | First | 253 | 402 | 50.0% (1/2) | 100.0% (1/1) | +50.0 pp |

## How to interpret the results

High percentages with very few picks carry substantial uncertainty. For example, 1/1 is 100% observed accuracy, but its 95% Wilson interval is approximately 20.7%–100%. The JSON includes an interval, mean predicted probability, coverage, draw failures and other-team-win failures for every league/model/threshold/year. These intervals treat games as independent and are not adjusted for comparisons across many leagues.

A higher filtered hit rate may come with fewer predictions. A threshold is a model estimate, not a guaranteed success rate; a model assigning ≥70% can still achieve below 70% in a particular league. Choosing leagues or thresholds after seeing these results requires a new future test to validate that choice.

Both model coefficients are fitted strictly before the evaluation year. Features update using earlier results with a three-hour availability delay. V3 uses its full annual home/draw/away blend and saved calibration. V2 is the fixed previously selected classifier, fitted on the same prior history. Architecture selection used 2024 first-division validation, so this is an annual historical test of the retrained procedure rather than a new forward test after retraining.

The original Poisson model is not included in these full-coverage tables: its cached historical comparison has fewer supported fixtures and an earlier data snapshot. The baseline here is V2. USA has only one configured league; all other configured countries have two. Country, division and SQLite league ID identify each league without requiring external league-name data.

## Files and reproduction

- [league_winner_thresholds.json](league_winner_thresholds.json): all separate league results and uncertainty, including accuracy on fixtures qualifying for both models at a threshold.
- [all_predictions_2025.csv](all_predictions_2025.csv) and [all_predictions_2026.csv](all_predictions_2026.csv): underlying full-coverage annual predictions.
- [forecast_metadata.json](forecast_metadata.json): annual model selections and training cutoffs.

After retraining, reconstruct annual forecasts and regenerate this report from the project directory:

```powershell
python winner_accuracy_comparison/compare_models.py
python winner_accuracy_comparison/league_winner_thresholds.py
```

To summarize already refreshed predictions, run only the second command. It refuses inputs that disagree with the latest training cutoff or protected annual-model hashes. Saved model files remain unchanged.
