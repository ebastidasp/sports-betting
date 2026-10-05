"""Local Worlds/MSI dashboard built on the existing LoL model."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:  # Environment variables work without python-dotenv.
    pass

import lol_worlds
from lol_match_feed import FeedSnapshot, MatchProvider, PandaScoreProvider


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def configured_path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default))
    return path if path.is_absolute() else ROOT / path


def database_summary(path: Path) -> dict[str, Any]:
    """Read cache metadata without creating or migrating the user's database."""
    if not path.is_file():
        return {"available": False, "error": f"Historical database not found: {path.name}"}
    try:
        with contextlib.closing(sqlite3.connect(str(path.resolve()))) as con:
            row = con.execute(
                "SELECT COUNT(DISTINCT gameid), COUNT(DISTINCT team), MIN(date), MAX(date) "
                "FROM team_games"
            ).fetchone()
        return {
            "available": bool(row and row[0]),
            "games": int(row[0] or 0),
            "teams": int(row[1] or 0),
            "first_game": row[2],
            "last_game": row[3],
        }
    except sqlite3.Error as exc:
        return {"available": False, "error": f"Could not read historical database: {exc}"}


def team_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        with contextlib.closing(sqlite3.connect(str(path.resolve()))) as con:
            rows = con.execute("SELECT DISTINCT team FROM team_games ORDER BY team").fetchall()
        return [str(row[0]) for row in rows if row[0]]
    except sqlite3.Error:
        LOGGER.exception("Failed reading team names from the local database")
        return []


def recent_results(path: Path, limit: int = 12) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with contextlib.closing(sqlite3.connect(str(path.resolve()))) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT gameid, team, opp, date, league, win FROM team_games "
                "WHERE UPPER(league) LIKE '%WORLD%' OR UPPER(league) LIKE '%MSI%' "
                "ORDER BY date DESC, gameid DESC, team LIMIT ?",
                (limit * 2,),
            ).fetchall()
        results = []
        seen_games = set()
        for row in rows:
            if row["gameid"] in seen_games:
                continue
            seen_games.add(row["gameid"])
            results.append(dict(row))
            if len(results) == limit:
                break
        return results
    except sqlite3.Error:
        LOGGER.exception("Failed reading recent results from the local database")
        return []


def make_prediction(
    *, db_path: Path, model_path: Path, team_a: str, team_b: str,
    best_of: int, patch: str, side_a: str,
) -> dict[str, Any]:
    """Call the existing CLI prediction implementation and return its result."""
    args = lol_worlds.argparse.Namespace(
        db=db_path,
        model=model_path,
        team_a=team_a,
        team_b=team_b,
        best_of=best_of,
        patch=patch or None,
        patch_tiers=None,
        rosters=None,
        patch_edge=None,
        side_a=side_a,
        as_of=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        odds=None,
        log_trade=None,
        stake=10.0,
        min_ev=0.03,
        force=False,
        no_champ=False,
        no_league=False,
        no_side=False,
    )
    # The CLI and website share one calculation path; suppress its terminal
    # report because the web template renders the returned structured values.
    with contextlib.redirect_stdout(io.StringIO()):
        result = lol_worlds.predict(args)
    if not isinstance(result, dict):
        raise RuntimeError("The model did not return a structured forecast.")
    return result


def create_app(
    *, provider: MatchProvider | None = None, config: dict[str, Any] | None = None,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        DB_PATH=configured_path("LOL_DB_PATH", "lol.sqlite3"),
        MODEL_PATH=configured_path("LOL_MODEL_PATH", "lol_worlds.joblib"),
        MATCH_PROVIDER=provider,
        PANDASCORE_API_TOKEN=os.environ.get("PANDASCORE_API_TOKEN", ""),
    )
    if config:
        app.config.update(config)
    if app.config["MATCH_PROVIDER"] is None:
        app.config["MATCH_PROVIDER"] = PandaScoreProvider(app.config["PANDASCORE_API_TOKEN"])

    @app.get("/")
    def dashboard():
        match_provider: MatchProvider = app.config["MATCH_PROVIDER"]
        upcoming = match_provider.snapshot("upcoming")
        live = match_provider.snapshot("running")
        summary = database_summary(Path(app.config["DB_PATH"]))
        return render_template(
            "dashboard.html",
            upcoming=upcoming,
            live=live,
            summary=summary,
            refreshed_at=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/api/matches")
    def matches_api():
        state = request.args.get("state", "running")
        if state not in ("upcoming", "running"):
            return jsonify({"error": "state must be upcoming or running"}), 400
        snapshot: FeedSnapshot = app.config["MATCH_PROVIDER"].snapshot(state)
        return jsonify({
            "state": state,
            "matches": snapshot.matches,
            "updated_at": snapshot.updated_at,
            "error": snapshot.error,
            "stale": snapshot.stale,
        })

    @app.route("/predictor", methods=["GET", "POST"])
    def predictor():
        path = Path(app.config["DB_PATH"])
        teams = team_names(path)
        if request.method == "POST":
            selected_a = request.form.get("team_a", "")
            selected_b = request.form.get("team_b", "")
            best_of_raw = request.form.get("best_of", "5")
            patch = request.form.get("patch", "").strip()
            side_a = request.form.get("side_a", "blue")
            prefill_warning = None
        else:
            requested_a = request.args.get("team_a", "")
            requested_b = request.args.get("team_b", "")
            missing = [name for name in (requested_a, requested_b) if name and name not in teams]
            selected_a = requested_a if requested_a in teams else ""
            selected_b = requested_b if requested_b in teams else ""
            best_of_raw = request.args.get("best_of", "5")
            patch = request.args.get("patch", "").strip()
            side_a = request.args.get("side_a", "blue")
            prefill_warning = (
                "The feed's team label does not exactly match the local model team list. "
                "Choose the corresponding historical team name manually: " + ", ".join(missing)
                if missing else None
            )
        result = None
        error = None

        if request.method == "POST":
            try:
                best_of = int(best_of_raw)
                if best_of not in (1, 3, 5):
                    raise ValueError("Choose a best-of format of 1, 3, or 5.")
                if selected_a not in teams or selected_b not in teams:
                    raise ValueError("Choose both teams from the historical team list.")
                if selected_a == selected_b:
                    raise ValueError("Choose two different teams.")
                if side_a not in ("blue", "red"):
                    raise ValueError("Choose blue or red side for Team A.")
                result = make_prediction(
                    db_path=path,
                    model_path=Path(app.config["MODEL_PATH"]),
                    team_a=selected_a,
                    team_b=selected_b,
                    best_of=best_of,
                    patch=patch,
                    side_a=side_a,
                )
            except (ValueError, SystemExit, FileNotFoundError, RuntimeError) as exc:
                error = str(exc) or "The model could not produce a forecast for this matchup."
            except Exception:
                LOGGER.exception("Unexpected prediction error")
                error = "The forecast failed. Check the model, database, and server log."

        return render_template(
            "predictor.html",
            teams=teams,
            selected_a=selected_a,
            selected_b=selected_b,
            best_of=best_of_raw,
            patch=patch,
            side_a=side_a,
            result=result,
            error=error,
            prefill_warning=prefill_warning,
            summary=database_summary(path),
            model_available=Path(app.config["MODEL_PATH"]).is_file(),
        )

    @app.get("/results")
    def results():
        path = Path(app.config["DB_PATH"])
        return render_template(
            "results.html",
            results=recent_results(path),
            summary=database_summary(path),
        )

    @app.get("/health")
    def health():
        summary = database_summary(Path(app.config["DB_PATH"]))
        return jsonify({
            "ok": summary.get("available", False),
            "historical_database": summary,
            "model_available": Path(app.config["MODEL_PATH"]).is_file(),
            "schedule_provider": "PandaScore",
            "schedule_token_configured": bool(app.config["PANDASCORE_API_TOKEN"]),
        }), (200 if summary.get("available") else 503)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
