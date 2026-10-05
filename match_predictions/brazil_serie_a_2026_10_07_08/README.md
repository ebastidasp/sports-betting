# Brazil Série A: screenshot odds and V3 expected value

All **10 screenshot fixtures** were predicted with the saved V3 model. Each listed plan uses **COP 25,000 per bet**; a win-or-draw plan splits that total between two separate 1X2 bets.

Prediction snapshot: **2026-10-05T08:52:02-05:00**. Model data cutoff: **2026-10-05T08:04:40.170616-05:00**. The latest completed Série A fixture in SQLite is **2026-10-03T16:30:00-05:00**. These forecasts use cached historical results, with no bookmaker odds as model inputs.

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
| 07 Oct 18:30 | Botafogo vs Vasco da Gama | Botafogo | 73.07% | 13,821 @ 2.75 | 11,179 @ 3.40 | 1.520 | +11.09% | 13,008 | 2,773 |

## Positive-EV over/under 2.5 goals

Over 2.5 means at least 3 total goals. Under 2.5 means at most 2. Each row stakes COP25,000; these are standalone bets, not combinations with the match-outcome bets.

| Date (Colombia) | Match | Market | Probability | At least 60%? | Fair odds | Offered | EV | Net profit if correct COP | Expected profit COP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 07 Oct 18:00 | Vitória vs Chapecoense | Under 2.5 goals | 54.44% | No | 1.837 | 1.93 | +5.06% | 23,250 | 1,266 |
| 08 Oct 19:30 | Fluminense vs Coritiba | Over 2.5 goals | 53.90% | No | 1.855 | 1.94 | +4.57% | 23,500 | 1,144 |
| 07 Oct 17:30 | Remo vs Grêmio | Under 2.5 goals | 55.06% | No | 1.816 | 1.89 | +4.07% | 22,250 | 1,017 |
| 07 Oct 17:30 | RB Bragantino vs Mirassol | Under 2.5 goals | 52.80% | No | 1.894 | 1.94 | +2.43% | 23,500 | 607 |
| 07 Oct 19:30 | Cruzeiro vs São Paulo | Over 2.5 goals | 49.69% | No | 2.012 | 2.03 | +0.87% | 25,750 | 218 |

## Highest model EV per fixture

This optional shortlist chooses one qualifying plan per fixture, ranked by model EV. All qualifying plans remain listed above; selecting multiple plans on a fixture uses COP25,000 for each and their outcomes can be correlated.

| Match | Plan | Probability | EV | Expected profit COP |
| --- | --- | --- | --- | --- |
| Botafogo vs Vasco da Gama | Botafogo or draw | 73.07% | +11.09% | 2,773 |
| Vitória vs Chapecoense | Under 2.5 goals | 54.44% | +5.06% | 1,266 |
| Fluminense vs Coritiba | Over 2.5 goals | 53.90% | +4.57% | 1,144 |
| Remo vs Grêmio | Under 2.5 goals | 55.06% | +4.07% | 1,017 |
| RB Bragantino vs Mirassol | Under 2.5 goals | 52.80% | +2.43% | 607 |
| Cruzeiro vs São Paulo | Over 2.5 goals | 49.69% | +0.87% | 218 |

The one-plan-per-fixture shortlist has **6 plans**, totaling **COP 150,000**. The sum of model-implied expected profits is **COP 7,025**. This does not estimate the probability that the portfolio finishes profitable.

## Complete forecasts and price checks

| Match | P(home) | P(draw) | P(away) | xG home-away | Home win EV | Away win EV | Home/draw EV | Away/draw EV | Over 2.5 EV | Under 2.5 EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Internacional vs Corinthians | 41.44% | 30.64% | 27.91% | 1.335-1.062 | -12.14% | +3.28% | -8.08% | +0.48% | -4.23% | -7.00% |
| RB Bragantino vs Mirassol | 46.72% | 28.91% | 24.37% | 1.521-1.041 | -19.18% | +14.55% | -10.10% | +11.96% | -14.09% | +2.43% |
| Remo vs Grêmio | 40.33% | 28.94% | 30.73% | 1.352-1.122 | -11.26% | -3.21% | -6.42% | -1.08% | -15.97% | +4.07% |
| Vitória vs Chapecoense | 50.51% | 28.51% | 20.98% | 1.537-0.961 | -15.65% | +4.89% | -7.96% | +7.65% | -16.62% | +5.06% |
| Botafogo vs Vasco da Gama | 45.35% | 27.72% | 26.93% | 1.556-1.257 | +24.72% | -31.33% | +11.09% | -20.37% | -3.94% | -8.13% |
| Cruzeiro vs São Paulo | 49.35% | 26.34% | 24.30% | 1.621-1.041 | -4.75% | -4.00% | -5.83% | -6.02% | +0.87% | -10.45% |
| Santos vs Flamengo | 25.80% | 28.08% | 46.12% | 1.078-1.621 | +10.92% | -16.52% | +7.15% | -9.81% | -8.87% | -2.24% |
| Athletico-PR vs Atlético-MG | 43.01% | 28.40% | 28.59% | 1.389-1.052 | -10.97% | +10.08% | -10.25% | -0.41% | -8.28% | -3.85% |
| Fluminense vs Coritiba | 62.90% | 22.78% | 14.31% | 1.992-0.844 | -3.13% | -10.53% | -4.07% | -8.15% | +4.57% | -16.11% |
| Palmeiras vs Bahia | 57.86% | 24.45% | 17.69% | 1.979-0.983 | -10.32% | -2.71% | -5.94% | +2.36% | -6.79% | -5.03% |

## Scope and reproduction

These are positive-EV estimates conditional on V3 probabilities. Historical accuracy is not demonstrated betting profitability; small margins can disappear with prediction error or price changes. Listed prices are the screenshot snapshot, without a live-price confirmation or additional charges.

Forecasts were made on 5 October using only completed cached results available at that time. Recheck probabilities and quoted prices nearer kickoff.

The screenshot dates and matchups agree with the cached schedule and [the official CBF fixture listings](https://credencial.cbf.com.br/competicoes/listar/42/1/); source times are Brasília time, two hours ahead of Colombia. All report dates are Colombia time.

- [quote_snapshot.json](quote_snapshot.json): all 50 quoted prices and filters.
- [predictions.json](predictions.json): full-precision probabilities, all win/pair/goal EV checks, stakes and provenance hashes.
- [cli_outputs.json](cli_outputs.json): actual output and exit status of all ten CLI predictions.
- [commands.ps1](commands.ps1): reproducible prediction commands.

Run from the project directory to reproduce this fixed snapshot (refuses changed models/data):

```powershell
python match_predictions/brazil_serie_a_2026_10_07_08/analyze.py
```
