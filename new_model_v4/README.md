# Football model V4

V4 is trained separately by `football_model_v4.py`. It excludes possession and
all passing statistics at ingestion, history construction and estimator inputs.
V3 already excluded passing inputs; the measured feature change removes ten
possession columns, reducing the active base inputs from **167 to 157**.
Expected goals, shots, corners, team form, Elo and rest remain available.

The architecture remains V3's selected **25% gradient-boosting classifier + 75%
Poisson goal component**. V4 fits new coefficients; it does not reuse V3's fitted
estimators. Its Dixon–Coles correction is fitted from held-out 2023 matches using
goal models fitted before 2023. Goal forecasts and fair odds remain available.

The current final model contains **50,860** completed historical fixtures, with
a data cutoff of **4 October 2026, 18:38:04 Colombia time**. Predictions can use
newer completed historical results, while training refreshes fitted coefficients.

The comparison evaluates **12,320** matches in 2025 and available 2026 data across
**21 leagues**, including both divisions. Both variants use identical prior
training fixtures and available match histories. Each annual evaluation model
is fitted before 1 January of its test year. The final model is not used to
evaluate its own training matches.

Full home/draw/away accuracy is **47.92% for V3 versus 47.84% for V4**. Winner
accuracy at a minimum 60% estimated win probability is **68.50% versus 69.18%**,
respectively. Changes across other thresholds and goal markets are mixed; the
comparison does not establish a consistent improvement from removing possession.

See [the complete comparison](../v3_v4_accuracy_comparison/README.md) for winner
accuracy by league and annual results, and
[the goal tables](../v3_v4_accuracy_comparison/GOALS_BY_LEAGUE.md) for separate
over/under 2.5 and 3.5 results at 60%, 70% and 80%. These reports include hit
counts, qualifying sample sizes and uncertainty. They measure accuracy, not
betting returns.

Run from the project directory:

```powershell
python football_model_v4.py train
python football_model_v4.py predict --country england --league-id 40 --home Middlesbrough --away Wolves
python football_model_v4.py backtest
```

`train` fits a new V4 and refreshes the paired chronological comparison using the
latest SQLite data and the saved V3 architecture. `backtest` verifies the saved
forecasts and their input hashes, then regenerates the accuracy reports without
fitting models. After data or model changes, run `train` to refresh the comparison.

The original model files and SQLite databases are preserved. V4 artifacts live
in `new_model_v4/`; comparison forecasts, baseline clones and metrics live in
`v3_v4_accuracy_comparison/`.
