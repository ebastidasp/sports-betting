# Two-page odds comparison: win-or-draw splits and total goals

The user confirmed a budget of **COP 25,000 per game**. **A = first screenshot; B = second screenshot.** This report selects the better quoted 1X2 price for each outcome. Equal prices default to A. Goal prices are available from B only.

The V3 predictions are the unchanged forecast snapshot from **2026-10-05T01:01:20+00:00** (4 October, 20:01:20 in Colombia), fitted through **2026-10-04T23:38:04.083848+00:00**. The quoted odds are screenshots rather than a live price feed. Source-model and data hashes still match the original prediction run.

Each stake split backs the team's win and the draw as **two separate bets**, allocating more stake to the shorter price so both covered results pay approximately the same gross amount. This method is commonly called [dutching](https://apps.betfair.com/learning/dutching/). It can use different pages for the two bets.

For win odds W and draw odds D, the exact team stake is `S × D / (W + D)` and draw stake is `S × W / (W + D)`. The effective price is `1 / (1/W + 1/D)`. The combined probability is `P(team win) + P(draw)` because the outcomes are mutually exclusive. Before rounding, estimated EV is `combined probability × effective price − 1`.

Stakes below are rounded to the nearest COP and sum to COP25,000. EV is recomputed from those rounded stakes. The **covered net profit** subtracts both stakes and uses the smaller payout after rounding. If the other team wins, the entire COP25,000 is lost. The estimates use listed decimal prices without deductions for additional charges.

## Positive EV win-or-draw splits with at least 60% combined probability

| Match | Team or draw | Combined probability | Team stake @ odds (page) | Draw stake @ odds (page) | Estimated EV | Minimum net profit if covered |
| --- | --- | --- | --- | --- | --- | --- |
| Middlesbrough vs Wolves | Middlesbrough | 79.28% | 15,625 @ 2.25 (A) | 9,375 @ 3.75 (A) | +11.49% | 10,156 |
| Watford vs Burnley | Watford | 67.54% | 13,456 @ 2.96 (A) | 11,544 @ 3.45 (A) | +7.61% | 14,827 |
| Preston vs Millwall | Millwall | 71.95% | 14,286 @ 2.55 (B) | 10,714 @ 3.40 (A) | +4.84% | 11,428 |
| Derby vs Wrexham | Derby | 68.77% | 13,843 @ 2.70 (B) | 11,157 @ 3.35 (A) | +2.81% | 12,376 |
| Blackburn vs Cardiff | Blackburn | 67.97% | 14,361 @ 2.63 (B) | 10,639 @ 3.55 (A) | +2.69% | 12,768 |
| Charlton vs Bristol City | Charlton | 62.37% | 12,692 @ 3.20 (A) | 12,308 @ 3.30 (A) | +1.34% | 15,614 |

## Other positive EV win-or-draw splits below 60%

| Match | Team or draw | Combined probability | Team stake @ odds (page) | Draw stake @ odds (page) | Estimated EV | Minimum net profit if covered |
| --- | --- | --- | --- | --- | --- | --- |
| Sheffield Utd vs Lincoln | Lincoln | 59.37% | 11,986 @ 3.80 (A) | 13,014 @ 3.50 (A) | +8.18% | 20,547 |
| West Ham vs QPR | QPR | 41.43% | 10,714 @ 5.80 (A) | 14,286 @ 4.35 (A) | +2.97% | 37,141 |

## Middlesbrough example

Back **Middlesbrough with COP 15,625 at 2.25 on A**, and **the draw with COP 9,375 at 3.75 on A**. Both covered results return **COP 35,156.25**, giving **COP 10,156.25 net profit**. If Wolves win, the net loss is COP25,000.

The model gives Middlesbrough or draw **79.28%**, equivalent fair odds **1.2614**. The two bets create effective odds **1.40625**, estimated EV **+11.49%**, and expected profit **COP 2,871.32**. Expected profit averages all outcomes; it differs from the profit paid if the bet succeeds.

## Positive EV goals options

Each goals row is an **alternative use of the same COP 25,000 per-game budget**. Choose one plan per match to keep within that budget. Under 2.5 means at most 2 goals; under 3.5 means at most 3. Under probabilities are complements of V3's matching over probabilities from its goal component. They are not combined with 1X2 probabilities.

| Match | Market | V3 probability | Fair odds | B odds | Estimated EV | Stake | Net profit if correct | At least 60%? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| West Brom vs Birmingham | under 2.5 goals | 57.27% | 1.75 | 1.98 | +13.38% | 25,000 | 24,500 | No |
| Blackburn vs Cardiff | under 2.5 goals | 51.75% | 1.93 | 2.16 | +11.79% | 25,000 | 29,000 | No |
| Sheffield Utd vs Lincoln | under 2.5 goals | 56.08% | 1.78 | 1.86 | +4.30% | 25,000 | 21,500 | No |
| West Ham vs QPR | under 3.5 goals | 67.26% | 1.49 | 1.54 | +3.58% | 25,000 | 13,500 | Yes |
| Charlton vs Bristol City | under 2.5 goals | 59.81% | 1.67 | 1.72 | +2.87% | 25,000 | 18,000 | No |
| Watford vs Burnley | under 2.5 goals | 51.94% | 1.93 | 1.95 | +1.28% | 25,000 | 23,750 | No |

Every quoted over-goals market has negative estimated EV. The only positive-value goals option meeting 60% is **West Ham–QPR under 3.5 at 1.54**. Middlesbrough–Wolves under 2.5 at 2.25 is slightly below break-even in the model and is excluded from positive-EV options.

## One plan per game meeting 60% and positive EV

This shortlist applies the 60% coverage threshold and chooses the largest positive EV option among eligible win-or-draw and goals plans for each game. These are candidate allocations derived from V3 estimates, rather than demonstrated profitable bets.

| Match | COP25,000 plan | Probability | Estimated EV | Expected profit, COP |
| --- | --- | --- | --- | --- |
| Middlesbrough vs Wolves | Middlesbrough: 15,625 @ 2.25 (A); draw: 9,375 @ 3.75 (A) | 79.28% | +11.49% | 2,871 |
| Watford vs Burnley | Watford: 13,456 @ 2.96 (A); draw: 11,544 @ 3.45 (A) | 67.54% | +7.61% | 1,902 |
| Preston vs Millwall | Millwall: 14,286 @ 2.55 (B); draw: 10,714 @ 3.40 (A) | 71.95% | +4.84% | 1,209 |
| West Ham vs QPR | under 3.5 goals: 25,000 @ 1.54 (B) | 67.26% | +3.58% | 894 |
| Derby vs Wrexham | Derby: 13,843 @ 2.70 (B); draw: 11,157 @ 3.35 (A) | 68.77% | +2.81% | 702 |
| Blackburn vs Cardiff | Blackburn: 14,361 @ 2.63 (B); draw: 10,639 @ 3.55 (A) | 67.97% | +2.69% | 672 |
| Charlton vs Bristol City | Charlton: 12,692 @ 3.20 (A); draw: 12,308 @ 3.30 (A) | 62.37% | +1.34% | 334 |

Using all 7 shortlist plans would stake **COP 175,000**. Expected profits add across plans without requiring independence, but the report does not estimate the joint chance of an overall portfolio profit.

## Complete price and EV checks

Both home-win-or-draw and away-win-or-draw splits are checked in every match. The following table preserves all 12 fixtures, including cases where a positive individual win or draw price becomes negative EV after adding the other outcome.

| Match | Best home | Best draw | Best away | Home or draw EV | Away or draw EV | Over goals EV | Under goals EV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| West Ham vs QPR | 1.52 (A) | 4.35 (A) | 5.80 (A) | -8.44% | +2.97% | -23.05% | +3.58% |
| Swansea vs Norwich | 2.50 (A) | 3.50 (A) | 2.66 (A) | -1.88% | -11.29% | -9.87% | -3.21% |
| Charlton vs Bristol City | 3.20 (A) | 3.30 (A) | 2.25 (A) | +1.34% | -7.20% | -18.81% | +2.87% |
| West Brom vs Birmingham | 2.25 (B) | 3.55 (A) | 3.00 (A) | -1.83% | -6.46% | -24.79% | +13.38% |
| Blackburn vs Cardiff | 2.63 (B) | 3.55 (A) | 2.54 (A) | +2.69% | -11.44% | -20.87% | +11.79% |
| Bolton vs Stoke City | 2.68 (A) | 3.45 (A) | 2.65 (B) | -0.83% | -7.36% | -8.49% | -5.33% |
| Middlesbrough vs Wolves | 2.25 (A) | 3.75 (A) | 3.00 (B) | +11.49% | -28.09% | -11.98% | -0.35% |
| Derby vs Wrexham | 2.70 (B) | 3.35 (A) | 2.62 (A) | +2.81% | -12.58% | -8.84% | -5.31% |
| Sheffield Utd vs Lincoln | 1.96 (A) | 3.50 (A) | 3.80 (A) | -12.93% | +8.18% | -18.30% | +4.30% |
| Watford vs Burnley | 2.96 (A) | 3.45 (A) | 2.32 (B) | +7.61% | -15.98% | -14.45% | +1.28% |
| Preston vs Millwall | 2.70 (A) | 3.40 (A) | 2.55 (B) | -17.45% | +4.84% | -10.50% | -3.44% |
| Southampton vs Portsmouth | 1.60 (A) | 4.20 (A) | 5.30 (B) | -4.89% | -7.86% | -3.27% | -11.81% |

## Limits and files

V3 probabilities have prediction error. More outcome coverage does not guarantee profit, and small estimated margins can disappear when probabilities or offered prices change. The English cache's latest completed result remains 19 September 2026; no updated injuries or starting lineups are incorporated. The split's estimated EV is the stake-weighted average of the individual win and draw bets' EVs.

The two win/draw bets are separate selections, so they are not an accumulator and should not be multiplied together. Goal options are listed as separate alternatives; no independence between match outcome and total goals is assumed. Prices may differ by page, and the stake split must be recalculated if either price changes.

- [combined_odds_stakes.json](combined_odds_stakes.json): all24 win-or-draw pairs, all24 goals markets, rounded stakes, payout states, positive-value filters and one-plan shortlist.
- [second_quote_snapshot.json](second_quote_snapshot.json): all prices and goal lines transcribed from screenshot B.
- [quote_snapshot.json](quote_snapshot.json): screenshot A prices.
- [predictions.json](predictions.json): unchanged full-precision V3 predictions and source hashes.
- [Original prediction report](README.md): all12 CLI commands, 1X2 probabilities and goal forecasts.

To reproduce from the project directory:

```powershell
python match_predictions/championship_2026_10_09_11/analyze_combined_odds.py
```
