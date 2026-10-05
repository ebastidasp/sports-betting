# Rift Match Centre — Worlds & MSI

A local browser dashboard for the existing League of Legends model. It shows Worlds/MSI upcoming fixtures and running series when a PandaScore token is configured, plus a matchup predictor and recent results from the project's local Oracle's Elixir cache.

## Run locally

The repository already contains `lol.sqlite3` and `lol_worlds.joblib` in the current workspace. They are local data/model artifacts and are intentionally not committed.

```powershell
py -m pip install -r requirements-lol-web.txt
Copy-Item .env.example .env
```

Add your PandaScore account token to `.env`:

```dotenv
PANDASCORE_API_TOKEN=your_token_here
```

Then start the app:

```powershell
py lol_web.py
```

Open <http://127.0.0.1:5000>. Keep the token server-side; do not put it in templates, JavaScript, or a public repository.

Without a token, the **Predictor** and **Results** pages still use the local model/cache. Upcoming and live sections explicitly report that the provider is not configured; they do not generate sample fixtures.

## What the dashboard uses

- **Schedule and live series score:** PandaScore `GET /lol/matches/upcoming` and `GET /lol/matches/running`, filtered to Worlds and MSI.
- **Historical model inputs and results:** `lol.sqlite3`, populated by the existing `lol_worlds.py download` command using Oracle's Elixir data.
- **Pre-match estimates:** the existing `lol_worlds.py predict` calculations and `lol_worlds.joblib` model.
- **Live boundary:** “Live” means match status and series score. Forecasts are pre-match estimates; the app does not produce live in-game win probabilities.

Review [`DATA_SOURCES.md`](DATA_SOURCES.md) for why PandaScore was selected and its plan/coverage caveats. Its published docs list upcoming/running match endpoints for all plans, while detailed LoL live frames are plan-gated. Confirm that your account receives Worlds/MSI events before relying on the feed.

## Local settings

Environment variables can override the default local artifact paths:

- `LOL_DB_PATH` — defaults to `lol.sqlite3`
- `LOL_MODEL_PATH` — defaults to `lol_worlds.joblib`
- `PORT` — defaults to `5000` (the development server binds to `127.0.0.1`)

Provider snapshots are held in process memory: upcoming matches for 15 minutes and running scores for 30 seconds. If PandaScore is temporarily unavailable, the last successful snapshot is marked stale.
