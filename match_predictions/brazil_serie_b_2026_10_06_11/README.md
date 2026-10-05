# Brazil Série B: screenshot odds and V3 expected value

All **14 screenshot fixtures** were predicted with the saved V3 model. Each listed plan uses **COP 25,000 per bet**; a win-or-draw plan splits that total between two separate 1X2 bets.

Prediction snapshot: **2026-10-05T08:14:13-05:00**. Model data cutoff: **2026-10-05T08:04:40.170616-05:00**. The latest completed Série B fixture in SQLite is **2026-10-03T16:30:00-05:00**. These forecasts use cached historical results, with no bookmaker odds as model inputs.

The retrained V3 now uses **[0.75, 0.25]** as classifier/goal blend weights, with **temperature** outcome calibration. This is the current saved model, rather than the earlier pre-Brazil training snapshot. V4 is not trained for Brazil and is not used here.

## Filters and stake calculation

A direct win qualifies only when its model probability is at least 60% and its quoted price has positive EV. A team win-or-draw plan qualifies only when the sum of those mutually exclusive outcome probabilities is at least 60% and the entire rounded stake split has positive EV. Goal bets require positive EV; the user did not specify a 60% goal threshold, so that flag is reported separately.

Single-bet EV is `probability × decimal odds − 1`. Expected profit is `COP25,000 × EV`. Fair odds are `1 / probability`. Expected profit averages winning and losing outcomes; it is not the payout for a successful bet.

For win odds W and draw odds D, team stake is rounded from `25,000 × D / (W + D)`; draw stake is the remainder. This creates approximately equal gross payouts. Effective odds are `1 / (1/W + 1/D)`. Rounded EV is recomputed as `(Pwin × win_stake × W + Pdraw × draw_stake × D − 25,000) / 25,000`. If the other team wins, the complete COP25,000 is lost.

## Direct team wins: probability at least 60% and positive EV

**No direct win meets both conditions.**

## Win or draw: combined probability at least 60% and positive EV

| Date (Colombia) | Match | Team or draw | Probability | Team stake @ odds | Draw stake @ odds | Effective odds | EV | Minimum covered net profit COP | Expected profit COP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 Oct 15:30 | Náutico vs Vila Nova | Náutico | 75.10% | 14,467 @ 2.33 | 10,533 @ 3.20 | 1.348 | +1.26% | 8,706 | 315 |
| 11 Oct 16:30 | Cuiabá vs Avaí | Cuiabá | 86.23% | 16,113 @ 1.82 | 8,887 @ 3.30 | 1.173 | +1.15% | 4,326 | 288 |

## Positive-EV over/under 2.5 goals

Over 2.5 means at least 3 total goals. Under 2.5 means at most 2. Each row stakes COP25,000; these are standalone bets, not combinations with the match-outcome bets.

| Date (Colombia) | Match | Market | Probability | At least 60%? | Fair odds | Offered | EV | Net profit if correct COP | Expected profit COP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06 Oct 19:35 | Ponte Preta vs Juventude | Under 2.5 goals | 57.22% | No | 1.748 | 1.92 | +9.86% | 23,000 | 2,466 |
| 11 Oct 15:30 | Náutico vs Vila Nova | Under 2.5 goals | 58.09% | No | 1.721 | 1.75 | +1.66% | 18,750 | 416 |
| 08 Oct 17:30 | Náutico vs Novorizontino | Under 2.5 goals | 53.60% | No | 1.866 | 1.89 | +1.30% | 22,250 | 324 |
| 07 Oct 18:30 | América-MG vs Fortaleza | Under 2.5 goals | 57.82% | No | 1.729 | 1.74 | +0.62% | 18,500 | 154 |
| 07 Oct 17:30 | Avaí vs Londrina | Under 2.5 goals | 59.89% | No | 1.670 | 1.68 | +0.61% | 17,000 | 152 |

## Highest model EV per fixture

This optional shortlist chooses one qualifying plan per fixture, ranked by model EV. All qualifying plans remain listed above; selecting multiple plans on a fixture uses COP25,000 for each and their outcomes can be correlated.

| Match | Plan | Probability | EV | Expected profit COP |
| --- | --- | --- | --- | --- |
| Ponte Preta vs Juventude | Under 2.5 goals | 57.22% | +9.86% | 2,466 |
| Náutico vs Vila Nova | Under 2.5 goals | 58.09% | +1.66% | 416 |
| Náutico vs Novorizontino | Under 2.5 goals | 53.60% | +1.30% | 324 |
| Cuiabá vs Avaí | Cuiabá or draw | 86.23% | +1.15% | 288 |
| América-MG vs Fortaleza | Under 2.5 goals | 57.82% | +0.62% | 154 |
| Avaí vs Londrina | Under 2.5 goals | 59.89% | +0.61% | 152 |

The one-plan-per-fixture shortlist has **6 plans**, totaling **COP 150,000**. The sum of model-implied expected profits is **COP 3,801**. This does not estimate the probability that the portfolio finishes profitable.

## Complete forecasts and price checks

| Match | P(home) | P(draw) | P(away) | xG home-away | Home win EV | Away win EV | Home/draw EV | Away/draw EV | Over 2.5 EV | Under 2.5 EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sport Recife vs São Bernardo | 46.59% | 28.46% | 24.95% | 1.488-1.046 | -12.40% | -2.71% | -8.21% | -1.49% | -13.04% | -1.03% |
| Goiás vs Athletic Club | 51.09% | 28.73% | 20.18% | 1.426-0.877 | -10.60% | -3.12% | -7.33% | -1.82% | -13.00% | -2.36% |
| Ponte Preta vs Juventude | 18.40% | 27.28% | 54.32% | 0.733-1.657 | +56.41% | -26.13% | +36.35% | -14.35% | -22.57% | +9.86% |
| Avaí vs Londrina | 42.32% | 29.04% | 28.64% | 1.311-0.978 | -16.63% | +8.82% | -11.97% | +1.87% | -16.56% | +0.61% |
| Operário-PR vs Botafogo-SP | 48.52% | 28.80% | 22.68% | 1.384-0.953 | -6.36% | -11.55% | -5.32% | -7.22% | -13.11% | -2.10% |
| América-MG vs Fortaleza | 28.51% | 31.11% | 40.38% | 1.045-1.322 | +1.21% | -17.63% | +1.97% | -9.87% | -15.65% | +0.62% |
| CRB vs Atlético-GO | 46.26% | 26.78% | 26.96% | 1.568-1.061 | -1.92% | -12.38% | -4.62% | -10.71% | -10.03% | -3.93% |
| Vila Nova vs Cuiabá | 42.62% | 32.41% | 24.97% | 1.312-0.912 | -13.91% | -3.86% | -7.66% | -0.59% | -5.99% | -7.55% |
| Ceará vs Criciúma | 37.63% | 30.08% | 32.29% | 1.198-1.078 | -9.68% | -1.53% | -8.40% | -4.12% | -5.34% | -7.85% |
| Náutico vs Novorizontino | 30.73% | 28.84% | 40.43% | 1.143-1.388 | -13.96% | -2.97% | -8.53% | -2.54% | -15.08% | +1.30% |
| Juventude vs São Bernardo | 52.76% | 28.49% | 18.75% | 1.479-0.862 | -12.94% | -1.56% | -8.07% | +0.88% | -14.59% | -0.48% |
| Atlético-GO vs Sport Recife | 54.66% | 26.21% | 19.13% | 1.655-0.892 | +0.58% | -21.57% | -2.47% | -14.40% | -9.18% | -4.80% |
| Náutico vs Vila Nova | 43.60% | 31.51% | 24.90% | 1.339-1.018 | +1.58% | -25.31% | +1.26% | -12.67% | -17.03% | +1.66% |
| Cuiabá vs Avaí | 59.48% | 26.75% | 13.77% | 1.632-0.676 | +8.26% | -36.66% | +1.15% | -22.14% | -5.82% | -7.92% |

## Scope and reproduction

These are positive-EV estimates conditional on V3 probabilities. Historical accuracy is not demonstrated betting profitability; small margins can disappear with prediction error or price changes. Listed prices are the screenshot snapshot, without a live-price confirmation or additional charges.

Sunday's teams play earlier in the week. Their 11 October probabilities here are forecasts made on 5 October, before those intervening results are known; rerun after the midweek matches before using Sunday selections.

The screenshot dates and matchups agree with the cached schedule and [the published rounds 31–34 schedule](https://ge.globo.com/pe/futebol/brasileirao-serie-b/noticia/2026/09/14/serie-b-cbf-define-datas-e-horarios-das-rodadas-31-a-34.ghtml); source times are Brasília time, two hours ahead of Colombia. All report dates are Colombia time.

- [quote_snapshot.json](quote_snapshot.json): all 70 quoted prices and filters.
- [predictions.json](predictions.json): full-precision probabilities, all win/pair/goal EV checks, stakes and provenance hashes.
- [cli_outputs.json](cli_outputs.json): actual output and exit status of all fourteen CLI predictions.
- [commands.ps1](commands.ps1): reproducible prediction commands.

Run from the project directory to reproduce this fixed snapshot (refuses changed models/data):

```powershell
python match_predictions/brazil_serie_b_2026_10_06_11/analyze.py
```
