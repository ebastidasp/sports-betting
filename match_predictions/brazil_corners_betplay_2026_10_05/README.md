# BetPlay Brazil corner price comparison

Created: 2026-10-05T14:54:02.914608+00:00. Historical results as of 2026-10-05T14:47:32.175988+00:00; model trained through 2026-10-05T14:33:21.072548+00:00.

Reference stake: **COP 25,000 per individual bet**. All positive-EV captured full-time totals are shown; there is no minimum probability filter. Prices are snapshots and require rechecking before use.

The original corner model includes passes and pass accuracy. It predicts home and away means using historical finished matches available after a three-hour result delay. Total corners use their separately calibrated count dispersion, rather than assuming independent team counts.

All captured lines end in .5, so each bet either wins or loses with no push. Fair decimal odds are 1 / model win probability; EV is probability × offered decimal odds − 1. The ≥60% flag is informational.

Only captured full-time corner totals are evaluated. First-half, handicap, race and other unsupported markets are excluded.

The model's historical count validation is not a test of profit at these bookmaker prices. Market probabilities, especially sparse team history, remain estimates. Different corner selections from the same fixture are correlated; the best-per-fixture table is an alternative to choosing all positive selections.

Forecasts use the same current historical state for all fixtures. Refresh forecasts and prices for later rounds after the intervening midweek matches finish; these estimates do not include those future results.

Coverage: 40 predicted fixtures, 80 priced selections, 17 positive selections; 0 excluded fixtures and 0 excluded selections.

The model's built-in 2025 Brazil test covers 380 Serie A matches, with total-corner MAE 2.630. That test excludes Serie B. The count-distribution calibration uses the model's pooled validation dispersion; it is not a separate calibration for these upcoming Brazil markets.

## Brazil Serie A (league 71)

### Highest positive EV per fixture

| Fixture | Selection | Mean | Profit probability | ≥60% | Offered / fair odds | EV | Expected profit COP |
|---|---|---:|---:|:---:|---:|---:|---:|
| Remo – Gremio | Total: Under 11.5 | 10.509 | 63.45% | Yes | 1.79 / 1.576 | 13.58% | 3,395 |
| Palmeiras – Corinthians | Total: Under 11.5 | 10.481 | 63.78% | Yes | 1.74 / 1.568 | 10.97% | 2,742 |
| Gremio – Internacional | Total: Under 10.5 | 9.617 | 62.86% | Yes | 1.70 / 1.591 | 6.86% | 1,715 |
| Atletico Paranaense – Atletico-MG | Total: Under 10.5 | 10.259 | 55.26% | No | 1.93 / 1.810 | 6.64% | 1,661 |
| Bahia – Mirassol | Total: Under 10.5 | 10.428 | 53.27% | No | 1.98 / 1.877 | 5.47% | 1,368 |
| Vasco DA Gama – Remo | Total: Under 11.5 | 10.718 | 61.11% | Yes | 1.66 / 1.636 | 1.44% | 360 |
| Botafogo – Vasco DA Gama | Total: Under 10.5 | 10.006 | 58.25% | No | 1.74 / 1.717 | 1.35% | 337 |

One reference bet for each row would stake COP 175,000. Expected profit is a model expectation, with no guarantee of a positive result.

### All positive-EV alternatives

| Fixture | Selection | Mean | Profit probability | ≥60% | Offered / fair odds | EV | Expected profit COP |
|---|---|---:|---:|:---:|---:|---:|---:|
| Remo – Gremio | Total: Under 11.5 | 10.509 | 63.45% | Yes | 1.79 / 1.576 | 13.58% | 3,395 |
| Palmeiras – Corinthians | Total: Under 11.5 | 10.481 | 63.78% | Yes | 1.74 / 1.568 | 10.97% | 2,742 |
| Gremio – Internacional | Total: Under 10.5 | 9.617 | 62.86% | Yes | 1.70 / 1.591 | 6.86% | 1,715 |
| Atletico Paranaense – Atletico-MG | Total: Under 10.5 | 10.259 | 55.26% | No | 1.93 / 1.810 | 6.64% | 1,661 |
| Bahia – Mirassol | Total: Under 10.5 | 10.428 | 53.27% | No | 1.98 / 1.877 | 5.47% | 1,368 |
| Vasco DA Gama – Remo | Total: Under 11.5 | 10.718 | 61.11% | Yes | 1.66 / 1.636 | 1.44% | 360 |
| Botafogo – Vasco DA Gama | Total: Under 10.5 | 10.006 | 58.25% | No | 1.74 / 1.717 | 1.35% | 337 |

## Brazil Serie B (league 72)

### Highest positive EV per fixture

| Fixture | Selection | Mean | Profit probability | ≥60% | Offered / fair odds | EV | Expected profit COP |
|---|---|---:|---:|:---:|---:|---:|---:|
| Criciuma – America Mineiro | Total: Under 11.5 | 11.104 | 56.77% | No | 1.92 / 1.761 | 9.00% | 2,250 |
| Juventude – São Bernardo | Total: Under 10.5 | 10.211 | 55.82% | No | 1.94 / 1.791 | 8.29% | 2,072 |
| Fortaleza EC – CRB | Total: Under 10.5 | 10.242 | 55.45% | No | 1.90 / 1.803 | 5.36% | 1,340 |
| America Mineiro – Fortaleza EC | Total: Under 10.5 | 10.275 | 55.07% | No | 1.90 / 1.816 | 4.63% | 1,158 |
| Novorizontino – Ponte Preta | Total: Under 12.5 | 11.786 | 59.91% | No | 1.74 / 1.669 | 4.24% | 1,060 |
| Operario-PR – Botafogo SP | Total: Under 11.5 | 10.634 | 62.06% | Yes | 1.66 / 1.611 | 3.02% | 755 |
| Athletic Club – Londrina | Total: Under 10.5 | 10.429 | 53.26% | No | 1.93 / 1.878 | 2.79% | 698 |
| Botafogo SP – Ceara | Total: Under 10.5 | 10.505 | 52.37% | No | 1.95 / 1.910 | 2.12% | 530 |
| Ponte Preta – Juventude | Total: Under 10.5 | 9.749 | 61.29% | Yes | 1.66 / 1.631 | 1.75% | 437 |
| Vila Nova – Cuiaba | Total: Under 10.5 | 9.976 | 58.60% | No | 1.73 / 1.706 | 1.38% | 345 |

One reference bet for each row would stake COP 250,000. Expected profit is a model expectation, with no guarantee of a positive result.

### All positive-EV alternatives

| Fixture | Selection | Mean | Profit probability | ≥60% | Offered / fair odds | EV | Expected profit COP |
|---|---|---:|---:|:---:|---:|---:|---:|
| Criciuma – America Mineiro | Total: Under 11.5 | 11.104 | 56.77% | No | 1.92 / 1.761 | 9.00% | 2,250 |
| Juventude – São Bernardo | Total: Under 10.5 | 10.211 | 55.82% | No | 1.94 / 1.791 | 8.29% | 2,072 |
| Fortaleza EC – CRB | Total: Under 10.5 | 10.242 | 55.45% | No | 1.90 / 1.803 | 5.36% | 1,340 |
| America Mineiro – Fortaleza EC | Total: Under 10.5 | 10.275 | 55.07% | No | 1.90 / 1.816 | 4.63% | 1,158 |
| Novorizontino – Ponte Preta | Total: Under 12.5 | 11.786 | 59.91% | No | 1.74 / 1.669 | 4.24% | 1,060 |
| Operario-PR – Botafogo SP | Total: Under 11.5 | 10.634 | 62.06% | Yes | 1.66 / 1.611 | 3.02% | 755 |
| Athletic Club – Londrina | Total: Under 10.5 | 10.429 | 53.26% | No | 1.93 / 1.878 | 2.79% | 698 |
| Botafogo SP – Ceara | Total: Under 10.5 | 10.505 | 52.37% | No | 1.95 / 1.910 | 2.12% | 530 |
| Ponte Preta – Juventude | Total: Under 10.5 | 9.749 | 61.29% | Yes | 1.66 / 1.631 | 1.75% | 437 |
| Vila Nova – Cuiaba | Total: Under 10.5 | 9.976 | 58.60% | No | 1.73 / 1.706 | 1.38% | 345 |

## Settlement details

The JSON contains full precision for every quoted selection, including negative EV. Gross payouts below include the returned stake. COP figures in this document are rounded only for display; bookmaker rounding is not modeled.

| Fixture / selection | Full / half win | Push | Half / full loss | Gross payout: full win / half win / push / half loss / loss |
|---|---:|---:|---:|---:|
| Remo – Gremio; Total: Under 11.5 | 63.45% / 0.00% | 0.00% | 0.00% / 36.55% | 44,750 / 34,875 / 25,000 / 12,500 / 0 |
| Palmeiras – Corinthians; Total: Under 11.5 | 63.78% / 0.00% | 0.00% | 0.00% / 36.22% | 43,500 / 34,250 / 25,000 / 12,500 / 0 |
| Criciuma – America Mineiro; Total: Under 11.5 | 56.77% / 0.00% | 0.00% | 0.00% / 43.23% | 48,000 / 36,500 / 25,000 / 12,500 / 0 |
| Juventude – São Bernardo; Total: Under 10.5 | 55.82% / 0.00% | 0.00% | 0.00% / 44.18% | 48,500 / 36,750 / 25,000 / 12,500 / 0 |
| Gremio – Internacional; Total: Under 10.5 | 62.86% / 0.00% | 0.00% | 0.00% / 37.14% | 42,500 / 33,750 / 25,000 / 12,500 / 0 |
| Atletico Paranaense – Atletico-MG; Total: Under 10.5 | 55.26% / 0.00% | 0.00% | 0.00% / 44.74% | 48,250 / 36,625 / 25,000 / 12,500 / 0 |
| Bahia – Mirassol; Total: Under 10.5 | 53.27% / 0.00% | 0.00% | 0.00% / 46.73% | 49,500 / 37,250 / 25,000 / 12,500 / 0 |
| Fortaleza EC – CRB; Total: Under 10.5 | 55.45% / 0.00% | 0.00% | 0.00% / 44.55% | 47,500 / 36,250 / 25,000 / 12,500 / 0 |
| America Mineiro – Fortaleza EC; Total: Under 10.5 | 55.07% / 0.00% | 0.00% | 0.00% / 44.93% | 47,500 / 36,250 / 25,000 / 12,500 / 0 |
| Novorizontino – Ponte Preta; Total: Under 12.5 | 59.91% / 0.00% | 0.00% | 0.00% / 40.09% | 43,500 / 34,250 / 25,000 / 12,500 / 0 |
| Operario-PR – Botafogo SP; Total: Under 11.5 | 62.06% / 0.00% | 0.00% | 0.00% / 37.94% | 41,500 / 33,250 / 25,000 / 12,500 / 0 |
| Athletic Club – Londrina; Total: Under 10.5 | 53.26% / 0.00% | 0.00% | 0.00% / 46.74% | 48,250 / 36,625 / 25,000 / 12,500 / 0 |
| Botafogo SP – Ceara; Total: Under 10.5 | 52.37% / 0.00% | 0.00% | 0.00% / 47.63% | 48,750 / 36,875 / 25,000 / 12,500 / 0 |
| Ponte Preta – Juventude; Total: Under 10.5 | 61.29% / 0.00% | 0.00% | 0.00% / 38.71% | 41,500 / 33,250 / 25,000 / 12,500 / 0 |
| Vasco DA Gama – Remo; Total: Under 11.5 | 61.11% / 0.00% | 0.00% | 0.00% / 38.89% | 41,500 / 33,250 / 25,000 / 12,500 / 0 |
| Vila Nova – Cuiaba; Total: Under 10.5 | 58.60% / 0.00% | 0.00% | 0.00% / 41.40% | 43,250 / 34,125 / 25,000 / 12,500 / 0 |
| Botafogo – Vasco DA Gama; Total: Under 10.5 | 58.25% / 0.00% | 0.00% | 0.00% / 41.75% | 43,500 / 34,250 / 25,000 / 12,500 / 0 |

## Fixture forecasts and captured pages

| League | Fixture | Kickoff UTC | Expected home / away / total | Quote observed | Source |
|---|---|---|---:|---|---|
| 71 | Internacional – Corinthians | 2026-10-07T22:30:00+00:00 | 5.627 / 4.045 / 9.672 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985650) |
| 71 | RB Bragantino – Mirassol | 2026-10-07T22:30:00+00:00 | 6.225 / 4.527 / 10.752 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985628) |
| 71 | Remo – Gremio | 2026-10-07T22:30:00+00:00 | 5.637 / 4.873 / 10.509 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985646) |
| 71 | Vitoria – Chapecoense-sc | 2026-10-07T23:00:00+00:00 | 6.433 / 3.990 / 10.423 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985647) |
| 71 | Botafogo – Vasco DA Gama | 2026-10-07T23:30:00+00:00 | 5.471 / 4.534 / 10.006 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985633) |
| 71 | Cruzeiro – Sao Paulo | 2026-10-08T00:30:00+00:00 | 6.009 / 4.329 / 10.338 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985626) |
| 71 | Santos – Flamengo | 2026-10-08T22:30:00+00:00 | 4.986 / 5.003 / 9.989 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985632) |
| 71 | Atletico Paranaense – Atletico-MG | 2026-10-08T23:00:00+00:00 | 5.873 / 4.386 / 10.259 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985649) |
| 71 | Fluminense – Coritiba | 2026-10-09T00:30:00+00:00 | 7.284 / 3.498 / 10.782 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985674) |
| 71 | Palmeiras – Bahia | 2026-10-09T00:30:00+00:00 | 6.837 / 3.887 / 10.724 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985630) |
| 71 | Vasco DA Gama – Remo | 2026-10-10T20:00:00+00:00 | 7.021 / 3.698 / 10.718 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985728) |
| 71 | Sao Paulo – Vitoria | 2026-10-11T00:00:00+00:00 | 6.959 / 3.910 / 10.869 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985727) |
| 71 | Atletico-MG – Santos | 2026-10-11T19:00:00+00:00 | 5.905 / 4.477 / 10.382 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985584) |
| 71 | Flamengo – Fluminense | 2026-10-11T20:30:00+00:00 | 6.709 / 3.604 / 10.313 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985644) |
| 71 | Gremio – Internacional | 2026-10-11T20:30:00+00:00 | 4.950 / 4.666 / 9.617 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985606) |
| 71 | Palmeiras – Corinthians | 2026-10-11T20:30:00+00:00 | 6.822 / 3.658 / 10.481 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985726) |
| 71 | Bahia – Mirassol | 2026-10-11T22:30:00+00:00 | 6.154 / 4.274 / 10.428 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985602) |
| 71 | Coritiba – Botafogo | 2026-10-12T19:00:00+00:00 | 5.064 / 5.248 / 10.312 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985604) |
| 71 | Chapecoense-sc – Atletico Paranaense | 2026-10-12T22:30:00+00:00 | 5.517 / 4.787 / 10.304 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985599) |
| 71 | RB Bragantino – Cruzeiro | 2026-10-13T00:00:00+00:00 | 6.029 / 4.681 / 10.711 | 2026-10-05T14:35:15.056Z | [BetPlay](https://betplay.com.co/apuestas#/event/1025985725) |
| 72 | Sport Recife – São Bernardo | 2026-10-06T22:30:00+00:00 | 6.423 / 4.616 / 11.040 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591548) |
| 72 | Goias – Athletic Club | 2026-10-06T23:30:00+00:00 | 5.909 / 4.291 / 10.200 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591559) |
| 72 | Ponte Preta – Juventude | 2026-10-07T00:35:00+00:00 | 4.303 / 5.446 / 9.749 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591538) |
| 72 | Avai – Londrina | 2026-10-07T22:30:00+00:00 | 5.714 / 4.882 / 10.596 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591562) |
| 72 | Operario-PR – Botafogo SP | 2026-10-07T22:30:00+00:00 | 5.922 / 4.712 / 10.634 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591565) |
| 72 | America Mineiro – Fortaleza EC | 2026-10-07T23:30:00+00:00 | 5.094 / 5.181 / 10.275 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591555) |
| 72 | CRB – Atletico Goianiense | 2026-10-07T23:30:00+00:00 | 6.074 / 4.441 / 10.515 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591506) |
| 72 | Vila Nova – Cuiaba | 2026-10-07T23:30:00+00:00 | 5.704 / 4.272 / 9.976 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591509) |
| 72 | Ceara – Criciuma | 2026-10-08T22:30:00+00:00 | 5.301 / 4.710 / 10.011 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591569) |
| 72 | Nautico Recife – Novorizontino | 2026-10-08T22:30:00+00:00 | 5.940 / 5.145 / 11.086 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591498) |
| 72 | Juventude – São Bernardo | 2026-10-11T19:00:00+00:00 | 6.103 / 4.108 / 10.211 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591541) |
| 72 | Atletico Goianiense – Sport Recife | 2026-10-11T19:10:00+00:00 | 6.516 / 4.408 / 10.924 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591525) |
| 72 | Nautico Recife – Vila Nova | 2026-10-11T20:30:00+00:00 | 6.272 / 4.311 / 10.583 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591531) |
| 72 | Cuiaba – Avai | 2026-10-11T21:30:00+00:00 | 6.621 / 3.492 / 10.112 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591532) |
| 72 | Operario-PR – Goias | 2026-10-11T21:30:00+00:00 | 5.946 / 4.505 / 10.450 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591516) |
| 72 | Fortaleza EC – CRB | 2026-10-12T23:30:00+00:00 | 5.756 / 4.487 / 10.242 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591523) |
| 72 | Novorizontino – Ponte Preta | 2026-10-12T22:00:00+00:00 | 8.657 / 3.129 / 11.786 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591503) |
| 72 | Criciuma – America Mineiro | 2026-10-12T22:30:00+00:00 | 7.478 / 3.626 / 11.104 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591519) |
| 72 | Botafogo SP – Ceara | 2026-10-13T00:00:00+00:00 | 5.758 / 4.747 / 10.505 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591500) |
| 72 | Athletic Club – Londrina | 2026-10-13T22:30:00+00:00 | 5.809 / 4.620 / 10.429 | 2026-10-05T14:36:50.005Z | [BetPlay](https://betplay.com.co/apuestas#/event/1026591515) |

### Captured schedule discrepancies

The forecast uses the cache kickoff; the quote snapshot retains the bookmaker display separately.

- {"league_id": 72, "fixture_id": 1520925, "book_event_id": 1026591523, "home": "Fortaleza", "away": "CRB", "book_kickoff_bogota": "2026-10-12T14:00:00-05:00", "cached_kickoff_bogota": "2026-10-12T18:30:00-05:00", "cached_minus_book_minutes": 270.0}

## Exclusions

None among the captured quote records.

## Reproduction and provenance

```powershell
python match_predictions/brazil_corners_betplay_2026_10_05/analyze.py
```

CLI verification: 40 fixtures checked using the original predict command at the same fixed as-of timestamp. Full commands and stdout are in cli_outputs.json.

Training source cache paths are recorded by the model. The hashes below freeze the files read for this analysis; they do not establish what those caches contained at training unless separate training provenance records are available.

| Input | SHA-256 |
|---|---|
| C:\Users\ebpun\OneDrive\Documentos\Football Model\brazil.sqlite3 | `d673f539914e25a99c6e5986a03e5eca99cb29a8f5d394561e5fe2666bcd54eb` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\brazil.sqlite3-wal | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\corners_model_brazil_2026_10_05\corners_model.joblib | `bf349e2610cca6bbe4b92d63747783e721d22e8d1e3de8de6c7737addc086e05` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\corners_model_brazil_2026_10_05\evaluation.json | `eb6158f0a9cf007e64fdfbf9f31b7f52f487e5d9b8e05b896f32bc5025830a59` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\football_corners.py | `d3ffb44699268340048b10c4c3635aaf5143f3a0c0eb8ed33a1acba77c7903d4` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\match_predictions\brazil_corners_betplay_2026_10_05\analyze.py | `2834eec4667fc5c23786e74a64198d2bef9d573d0b4a9ee5b78ca5effe3ce3ba` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\match_predictions\brazil_corners_betplay_2026_10_05\browser_quote_verification.json | `df8427f09490e7b726111ac4117b97172a746055ced302886826a00d2552a308` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\match_predictions\brazil_corners_betplay_2026_10_05\quote_snapshot.json | `892b1094f3d172ab67cb3f72cdbd35105baf84e1bd57075e1b0444dffb071deb` |
| C:\Users\ebpun\OneDrive\Documentos\Football Model\match_predictions\brazil_corners_betplay_2026_10_05\raw_quotes.json | `8eb9efbfbfd3f10cb1db03efae991775fd557f1c6a4b9fa6017715b47b54bca9` |
