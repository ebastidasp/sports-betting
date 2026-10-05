# V1 versus retrained V3: winner accuracy separately for every league

V1 is the original **football_poisson.py** model. V3 uses its full saved annual home/draw/away blend, including calibration. This report refreshes V1's historical fits from the current SQLite data instead of reusing the older prediction snapshot.

Evaluation covers cached **2025 and 2026 matches through 02 October 2026, 22:21:42 Colombia time**. There are **8,210 matched V1/V3 fixtures across 14 comparable leagues**. All 21 configured leagues are listed separately, including those without V1 coverage.

Both models are restricted to the **same forecastable match population within each league**. Each then selects games using its own **max(P(home win), P(away win)) ≥60%, ≥70% or ≥80%**. The pick is the more probable winner. Actual draws count as failures. The thresholds are inclusive and cumulative.

Cells show **observed accuracy (correct / selected games)**. The models can qualify different fixtures at a given threshold; filtered differences are descriptive. **Unavailable** means no matched V1 forecasts exist for that league; **— (0 picks)** means matched games exist, but none meets that model's threshold.

V3 figures here can differ from the full V2/V3 league comparison because V1's statistics, history and fitted-venue requirements exclude some fixtures. The JSON also contains V3's full available-sample results as separately labeled context.

Ireland has no cached 2026 evaluation fixtures in either division. Its combined results use 2025 only.

## 2025 + 2026: separate league comparisons

### Winner probability ≥60%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 818/904 | 54.8% (17/31) | 57.9% (22/38) | +3.1 pp |
| Argentina | Second | 129 | 0/1190 | Unavailable | Unavailable | — |
| Chile | First | 265 | 368/423 | 63.3% (38/60) | 68.1% (32/47) | +4.8 pp |
| Chile | Second | 266 | 0/448 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 719/761 | 63.6% (77/121) | 69.0% (60/87) | +5.3 pp |
| Colombia | Second | 240 | 0/532 | Unavailable | Unavailable | — |
| England | First | 39 | 618/618 | 66.7% (84/126) | 64.7% (97/150) | -2.0 pp |
| England | Second | 40 | 859/921 | 62.5% (40/64) | 69.9% (51/73) | +7.4 pp |
| Germany | First | 78 | 516/516 | 71.3% (92/129) | 72.4% (105/145) | +1.1 pp |
| Germany | Second | 79 | 507/516 | 57.6% (19/33) | 57.9% (22/38) | +0.3 pp |
| Iceland | First | 164 | 0/302 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/273 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 152/181 | 62.5% (5/8) | 52.6% (10/19) | -9.9 pp |
| Ireland | Second | 358 | 0/183 | Unavailable | Unavailable | — |
| Italy | First | 135 | 631/632 | 71.1% (86/121) | 72.2% (104/144) | +1.1 pp |
| Italy | Second | 136 | 589/629 | 87.8% (36/41) | 82.5% (33/40) | -5.3 pp |
| South Korea | First | 292 | 161/411 | 33.3% (1/3) | 14.3% (1/7) | -19.0 pp |
| South Korea | Second | 293 | 0/491 | Unavailable | Unavailable | — |
| Spain | First | 140 | 648/648 | 76.2% (138/181) | 78.4% (116/148) | +2.1 pp |
| Spain | Second | 141 | 729/772 | 68.3% (41/60) | 72.1% (31/43) | +3.8 pp |
| United States | First | 253 | 895/935 | 61.0% (47/77) | 60.0% (78/130) | -1.0 pp |

### Winner probability ≥70%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 818/904 | 50.0% (1/2) | 50.0% (2/4) | +0.0 pp |
| Argentina | Second | 129 | 0/1190 | Unavailable | Unavailable | — |
| Chile | First | 265 | 368/423 | 63.6% (7/11) | 88.9% (8/9) | +25.3 pp |
| Chile | Second | 266 | 0/448 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 719/761 | 74.3% (26/35) | 92.3% (12/13) | +18.0 pp |
| Colombia | Second | 240 | 0/532 | Unavailable | Unavailable | — |
| England | First | 39 | 618/618 | 79.4% (27/34) | 72.0% (36/50) | -7.4 pp |
| England | Second | 40 | 859/921 | 33.3% (3/9) | 69.2% (9/13) | +35.9 pp |
| Germany | First | 78 | 516/516 | 78.5% (51/65) | 81.3% (61/75) | +2.9 pp |
| Germany | Second | 79 | 507/516 | 50.0% (1/2) | 33.3% (1/3) | -16.7 pp |
| Iceland | First | 164 | 0/302 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/273 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 152/181 | 0.0% (0/1) | 100.0% (1/1) | +100.0 pp |
| Ireland | Second | 358 | 0/183 | Unavailable | Unavailable | — |
| Italy | First | 135 | 631/632 | 86.5% (32/37) | 79.1% (34/43) | -7.4 pp |
| Italy | Second | 136 | 589/629 | 100.0% (9/9) | 90.0% (9/10) | -10.0 pp |
| South Korea | First | 292 | 161/411 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 0/491 | Unavailable | Unavailable | — |
| Spain | First | 140 | 648/648 | 84.1% (74/88) | 87.2% (68/78) | +3.1 pp |
| Spain | Second | 141 | 729/772 | 100.0% (6/6) | 100.0% (5/5) | +0.0 pp |
| United States | First | 253 | 895/935 | 38.5% (5/13) | 60.9% (14/23) | +22.4 pp |

### Winner probability ≥80%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 818/904 | — (0 picks) | — (0 picks) | — |
| Argentina | Second | 129 | 0/1190 | Unavailable | Unavailable | — |
| Chile | First | 265 | 368/423 | 100.0% (1/1) | — (0 picks) | — |
| Chile | Second | 266 | 0/448 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 719/761 | 100.0% (4/4) | — (0 picks) | — |
| Colombia | Second | 240 | 0/532 | Unavailable | Unavailable | — |
| England | First | 39 | 618/618 | 66.7% (2/3) | 100.0% (6/6) | +33.3 pp |
| England | Second | 40 | 859/921 | — (0 picks) | 50.0% (1/2) | — |
| Germany | First | 78 | 516/516 | 83.3% (20/24) | 82.1% (23/28) | -1.2 pp |
| Germany | Second | 79 | 507/516 | — (0 picks) | — (0 picks) | — |
| Iceland | First | 164 | 0/302 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/273 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 152/181 | — (0 picks) | — (0 picks) | — |
| Ireland | Second | 358 | 0/183 | Unavailable | Unavailable | — |
| Italy | First | 135 | 631/632 | 100.0% (4/4) | 83.3% (5/6) | -16.7 pp |
| Italy | Second | 136 | 589/629 | — (0 picks) | — (0 picks) | — |
| South Korea | First | 292 | 161/411 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 0/491 | Unavailable | Unavailable | — |
| Spain | First | 140 | 648/648 | 86.7% (26/30) | 96.3% (26/27) | +9.6 pp |
| Spain | Second | 141 | 729/772 | — (0 picks) | — (0 picks) | — |
| United States | First | 253 | 895/935 | 50.0% (1/2) | 100.0% (1/1) | +50.0 pp |

## 2025: separate league comparisons

### Winner probability ≥60%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 471/503 | 52.4% (11/21) | 45.5% (10/22) | -6.9 pp |
| Argentina | Second | 129 | 0/632 | Unavailable | Unavailable | — |
| Chile | First | 265 | 210/240 | 66.7% (16/24) | 75.0% (18/24) | +8.3 pp |
| Chile | Second | 266 | 0/251 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 409/450 | 62.3% (38/61) | 68.9% (31/45) | +6.6 pp |
| Colombia | Second | 240 | 0/306 | Unavailable | Unavailable | — |
| England | First | 39 | 378/378 | 69.9% (58/83) | 69.2% (74/107) | -0.7 pp |
| England | Second | 40 | 510/558 | 64.9% (24/37) | 71.8% (28/39) | +6.9 pp |
| Germany | First | 78 | 308/308 | 72.3% (47/65) | 74.1% (60/81) | +1.8 pp |
| Germany | Second | 79 | 306/307 | 61.9% (13/21) | 59.1% (13/22) | -2.8 pp |
| Iceland | First | 164 | 0/162 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/137 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 152/181 | 62.5% (5/8) | 52.6% (10/19) | -9.9 pp |
| Ireland | Second | 358 | 0/183 | Unavailable | Unavailable | — |
| Italy | First | 135 | 367/368 | 69.6% (48/69) | 72.2% (57/79) | +2.6 pp |
| Italy | Second | 136 | 334/369 | 81.2% (13/16) | 75.0% (12/16) | -6.2 pp |
| South Korea | First | 292 | 89/232 | 33.3% (1/3) | 16.7% (1/6) | -16.7 pp |
| South Korea | Second | 293 | 0/275 | Unavailable | Unavailable | — |
| Spain | First | 140 | 370/370 | 77.9% (81/104) | 76.1% (70/92) | -1.8 pp |
| Spain | Second | 141 | 410/447 | 56.2% (18/32) | 77.3% (17/22) | +21.0 pp |
| United States | First | 253 | 495/533 | 66.7% (22/33) | 67.8% (40/59) | +1.1 pp |

### Winner probability ≥70%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 471/503 | 50.0% (1/2) | 50.0% (1/2) | +0.0 pp |
| Argentina | Second | 129 | 0/632 | Unavailable | Unavailable | — |
| Chile | First | 265 | 210/240 | 60.0% (3/5) | 80.0% (4/5) | +20.0 pp |
| Chile | Second | 266 | 0/251 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 409/450 | 81.8% (9/11) | 100.0% (4/4) | +18.2 pp |
| Colombia | Second | 240 | 0/306 | Unavailable | Unavailable | — |
| England | First | 39 | 378/378 | 76.0% (19/25) | 78.4% (29/37) | +2.4 pp |
| England | Second | 40 | 510/558 | 37.5% (3/8) | 70.0% (7/10) | +32.5 pp |
| Germany | First | 78 | 308/308 | 83.9% (26/31) | 83.7% (36/43) | -0.2 pp |
| Germany | Second | 79 | 306/307 | 50.0% (1/2) | 33.3% (1/3) | -16.7 pp |
| Iceland | First | 164 | 0/162 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/137 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 152/181 | 0.0% (0/1) | 100.0% (1/1) | +100.0 pp |
| Ireland | Second | 358 | 0/183 | Unavailable | Unavailable | — |
| Italy | First | 135 | 367/368 | 84.2% (16/19) | 85.0% (17/20) | +0.8 pp |
| Italy | Second | 136 | 334/369 | 100.0% (4/4) | 50.0% (1/2) | -50.0 pp |
| South Korea | First | 292 | 89/232 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 0/275 | Unavailable | Unavailable | — |
| Spain | First | 140 | 370/370 | 83.3% (45/54) | 87.2% (41/47) | +3.9 pp |
| Spain | Second | 141 | 410/447 | — (0 picks) | 100.0% (4/4) | — |
| United States | First | 253 | 495/533 | — (0 picks) | 66.7% (4/6) | — |

### Winner probability ≥80%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 471/503 | — (0 picks) | — (0 picks) | — |
| Argentina | Second | 129 | 0/632 | Unavailable | Unavailable | — |
| Chile | First | 265 | 210/240 | — (0 picks) | — (0 picks) | — |
| Chile | Second | 266 | 0/251 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 409/450 | — (0 picks) | — (0 picks) | — |
| Colombia | Second | 240 | 0/306 | Unavailable | Unavailable | — |
| England | First | 39 | 378/378 | 66.7% (2/3) | 100.0% (6/6) | +33.3 pp |
| England | Second | 40 | 510/558 | — (0 picks) | — (0 picks) | — |
| Germany | First | 78 | 308/308 | 81.8% (9/11) | 83.3% (15/18) | +1.5 pp |
| Germany | Second | 79 | 306/307 | — (0 picks) | — (0 picks) | — |
| Iceland | First | 164 | 0/162 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/137 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 152/181 | — (0 picks) | — (0 picks) | — |
| Ireland | Second | 358 | 0/183 | Unavailable | Unavailable | — |
| Italy | First | 135 | 367/368 | 100.0% (2/2) | 100.0% (1/1) | +0.0 pp |
| Italy | Second | 136 | 334/369 | — (0 picks) | — (0 picks) | — |
| South Korea | First | 292 | 89/232 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 0/275 | Unavailable | Unavailable | — |
| Spain | First | 140 | 370/370 | 85.0% (17/20) | 100.0% (13/13) | +15.0 pp |
| Spain | Second | 141 | 410/447 | — (0 picks) | — (0 picks) | — |
| United States | First | 253 | 495/533 | — (0 picks) | — (0 picks) | — |

## 2026: separate league comparisons

### Winner probability ≥60%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 347/401 | 60.0% (6/10) | 75.0% (12/16) | +15.0 pp |
| Argentina | Second | 129 | 0/558 | Unavailable | Unavailable | — |
| Chile | First | 265 | 158/183 | 61.1% (22/36) | 60.9% (14/23) | -0.2 pp |
| Chile | Second | 266 | 0/197 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 310/311 | 65.0% (39/60) | 69.0% (29/42) | +4.0 pp |
| Colombia | Second | 240 | 0/226 | Unavailable | Unavailable | — |
| England | First | 39 | 240/240 | 60.5% (26/43) | 53.5% (23/43) | -7.0 pp |
| England | Second | 40 | 349/363 | 59.3% (16/27) | 67.6% (23/34) | +8.4 pp |
| Germany | First | 78 | 208/208 | 70.3% (45/64) | 70.3% (45/64) | +0.0 pp |
| Germany | Second | 79 | 201/209 | 50.0% (6/12) | 56.2% (9/16) | +6.2 pp |
| Iceland | First | 164 | 0/140 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/136 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 0/0 | Unavailable | Unavailable | — |
| Ireland | Second | 358 | 0/0 | Unavailable | Unavailable | — |
| Italy | First | 135 | 264/264 | 73.1% (38/52) | 72.3% (47/65) | -0.8 pp |
| Italy | Second | 136 | 255/260 | 92.0% (23/25) | 87.5% (21/24) | -4.5 pp |
| South Korea | First | 292 | 72/179 | — (0 picks) | 0.0% (0/1) | — |
| South Korea | Second | 293 | 0/216 | Unavailable | Unavailable | — |
| Spain | First | 140 | 278/278 | 74.0% (57/77) | 82.1% (46/56) | +8.1 pp |
| Spain | Second | 141 | 319/325 | 82.1% (23/28) | 66.7% (14/21) | -15.5 pp |
| United States | First | 253 | 400/402 | 56.8% (25/44) | 53.5% (38/71) | -3.3 pp |

### Winner probability ≥70%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 347/401 | — (0 picks) | 50.0% (1/2) | — |
| Argentina | Second | 129 | 0/558 | Unavailable | Unavailable | — |
| Chile | First | 265 | 158/183 | 66.7% (4/6) | 100.0% (4/4) | +33.3 pp |
| Chile | Second | 266 | 0/197 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 310/311 | 70.8% (17/24) | 88.9% (8/9) | +18.1 pp |
| Colombia | Second | 240 | 0/226 | Unavailable | Unavailable | — |
| England | First | 39 | 240/240 | 88.9% (8/9) | 53.8% (7/13) | -35.0 pp |
| England | Second | 40 | 349/363 | 0.0% (0/1) | 66.7% (2/3) | +66.7 pp |
| Germany | First | 78 | 208/208 | 73.5% (25/34) | 78.1% (25/32) | +4.6 pp |
| Germany | Second | 79 | 201/209 | — (0 picks) | — (0 picks) | — |
| Iceland | First | 164 | 0/140 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/136 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 0/0 | Unavailable | Unavailable | — |
| Ireland | Second | 358 | 0/0 | Unavailable | Unavailable | — |
| Italy | First | 135 | 264/264 | 88.9% (16/18) | 73.9% (17/23) | -15.0 pp |
| Italy | Second | 136 | 255/260 | 100.0% (5/5) | 100.0% (8/8) | +0.0 pp |
| South Korea | First | 292 | 72/179 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 0/216 | Unavailable | Unavailable | — |
| Spain | First | 140 | 278/278 | 85.3% (29/34) | 87.1% (27/31) | +1.8 pp |
| Spain | Second | 141 | 319/325 | 100.0% (6/6) | 100.0% (1/1) | +0.0 pp |
| United States | First | 253 | 400/402 | 38.5% (5/13) | 58.8% (10/17) | +20.4 pp |

### Winner probability ≥80%

| Country | Division | League ID | Common / all V3 games | V1 accuracy (hits/picks) | V3 accuracy (hits/picks) | V3 − V1 |
| --- | --- | --- | --- | --- | --- | --- |
| Argentina | First | 128 | 347/401 | — (0 picks) | — (0 picks) | — |
| Argentina | Second | 129 | 0/558 | Unavailable | Unavailable | — |
| Chile | First | 265 | 158/183 | 100.0% (1/1) | — (0 picks) | — |
| Chile | Second | 266 | 0/197 | Unavailable | Unavailable | — |
| Colombia | First | 239 | 310/311 | 100.0% (4/4) | — (0 picks) | — |
| Colombia | Second | 240 | 0/226 | Unavailable | Unavailable | — |
| England | First | 39 | 240/240 | — (0 picks) | — (0 picks) | — |
| England | Second | 40 | 349/363 | — (0 picks) | 50.0% (1/2) | — |
| Germany | First | 78 | 208/208 | 84.6% (11/13) | 80.0% (8/10) | -4.6 pp |
| Germany | Second | 79 | 201/209 | — (0 picks) | — (0 picks) | — |
| Iceland | First | 164 | 0/140 | Unavailable | Unavailable | — |
| Iceland | Second | 165 | 0/136 | Unavailable | Unavailable | — |
| Ireland | First | 357 | 0/0 | Unavailable | Unavailable | — |
| Ireland | Second | 358 | 0/0 | Unavailable | Unavailable | — |
| Italy | First | 135 | 264/264 | 100.0% (2/2) | 80.0% (4/5) | -20.0 pp |
| Italy | Second | 136 | 255/260 | — (0 picks) | — (0 picks) | — |
| South Korea | First | 292 | 72/179 | — (0 picks) | — (0 picks) | — |
| South Korea | Second | 293 | 0/216 | Unavailable | Unavailable | — |
| Spain | First | 140 | 278/278 | 90.0% (9/10) | 92.9% (13/14) | +2.9 pp |
| Spain | Second | 141 | 319/325 | — (0 picks) | — (0 picks) | — |
| United States | First | 253 | 400/402 | 50.0% (1/2) | 100.0% (1/1) | +50.0 pp |

## Coverage and limitations

V1 needs both teams' cached match statistics, at least five prior games in the relevant venue, calibrated promoted-team venue Elo, and fitted host-home/visitor-away models from the training period. Team/venue formulas additionally require at least two eligible training matches and positive weighted goals. Missing statistics or history are coverage limitations, not zero accuracy.

V1 has no cached match statistics in Iceland, or in the second divisions of Argentina, Chile, Colombia, Ireland and South Korea. The per-country metadata gives the statistics/history/model exclusions separately for each year and league.

V1 uses the same fixed baseline settings as the earlier comparison: EWMA alpha 0.25, minimum venue history 5, regularization 0.01, recency decay 0.95 and Dixon–Coles enabled. No parameters are selected using the 2025/2026 results. V1 coefficients and rho are freshly fitted strictly before each evaluation year, using eligible prior results from all configured divisions in the country. V3 uses the already refreshed annual bundles fitted before the same test year; architecture selection used 2024 first-division validation.

Historical features use preceding matches only. A three-hour result availability delay applies at training and data cutoffs. V1 updates its feature states immediately after each historical match, so the script verifies that no team has consecutive fixture kickoffs less than three hours apart. All source countries passed this check. V1 probabilities come from its exact Skellam/Dixon–Coles outcome function, with its own clipping and normalization.

The JSON records coverage, average predicted win probability, draw failures, other-team-win failures and 95% Wilson intervals for each league/model/year/threshold. It also reports both models on the same subset of games qualifying for both. Wilson intervals assume independent games and are not corrected for comparisons across many leagues. Results with a handful of selections are uncertain; 100% from one match is weak evidence.

The 2026 period and SQLite coverage are incomplete. These are historical tests of the retrained procedure, not an independent future test after retraining. SQLite databases are opened read-only; input/model hashes are checked before and after the reconstruction. Saved models are not overwritten.

## Reproduction

From the project directory:

```powershell
python winner_accuracy_comparison/compare_v1_v3.py
```

This refreshes V1 fits in memory and writes forecasts and this report under `winner_accuracy_comparison/v1_v3/`. If V3 has been retrained again, first refresh its annual predictions with `python winner_accuracy_comparison/compare_models.py`.

To summarize these already refreshed V1 forecasts:

```powershell
python winner_accuracy_comparison/compare_v1_v3.py --mode report
```

- [evaluation.json](evaluation.json): separate league metrics, annual results, intervals and input hashes.
- [forecast_metadata.json](forecast_metadata.json): fresh V1 training sizes, settings, rho and coverage by country/year.
- [common_predictions_2025.csv](common_predictions_2025.csv) and [common_predictions_2026.csv](common_predictions_2026.csv): matched fixture probabilities.
- [Full V2/V3 league comparison](../WINNER_THRESHOLDS_BY_LEAGUE.md): V3's larger available population.
