"""Country-based football downloader and per-team, per-venue Poisson model.

Install: python -m pip install requests pandas numpy scipy scikit-learn joblib python-dotenv
Create a .env file next to this script containing:
  API_FOOTBALL_KEY=your_api_key_here
Existing environment variables take precedence over .env values.

Examples:
  python football_poisson.py download --country Colombia --from-season 2019 --to-season 2025
  python football_poisson.py download --country England --from-season 2019 --to-season 2025
  python football_poisson.py train --country England
  python football_poisson.py predict --country England --home Arsenal --away Chelsea
  python football_poisson.py backtest --country England --train-through 2024 --test-season 2025

Files default to <country>.sqlite3, <country>_poisson.joblib and country-specific CSVs.
Known men's senior division names are matched to the live API league catalog.
Unrecognized/ambiguous divisions prompt once for ordered IDs, then are persisted.
Use --division-ids FIRST [SECOND] to make that selection noninteractively.
Missing seasons are skipped; API access and match-statistics coverage still apply.

Home goals use the host's home attack and visitor's away defense; away goals
use the visitor's away attack and host's home defense. Attack and Elo effects
are nonnegative. Opponent goals/shots-on-target conceded have nonnegative
effects; opponent save/block rates have nonpositive effects.
Each team/venue formula is regularized toward a league-wide venue formula
fitted on the same training partition, using its shared preprocessing.
Training weights decay per team/venue game: recency_decay ** newer_game_count.
Default --recency-decay 0.95 gives weights 1, .95, .9025, ... from newest back.
Rolling statistics use normalized exponential weights (default alpha .25).
Prediction and backtesting share the same fitting and probability functions.
Retrain old models; existing fixture/statistics caches can be reused.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import unicodedata
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import poisson, skellam
from scipy.optimize import minimize_scalar, minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss, mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

API_URL = "https://v3.football.api-sports.io"
# Set from the country configuration before executing a command.
FIRST_DIVISION: int | None = None
SECOND_DIVISION_ID: int | None = None
LEAGUE_IDS: tuple[int, ...] = ()
DEFAULT_DB = None
DEFAULT_MODEL = None
MODEL_KIND = "shared_venue_opponent_elo_poisson_v9"
DEFAULT_RECENCY_DECAY = 0.95
COUNTRY = ""

# Resolve .env relative to this script, regardless of the working directory.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
INITIAL_ELO = 1500
SECOND_DIVISION_INITIAL_ELO = 1350
PROMOTED_TEAM_CALIBRATION_GAMES = 5

# API-Football labels mapped to stable internal names.
STAT_ALIASES = {
    "Shots on Goal": "shots_on_target",
    "Shots off Goal": "shots_off_target",
    "Total Shots": "shots",
    "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_inside_box",
    "Shots outsidebox": "shots_outside_box",
    "Fouls": "fouls",
    "Corner Kicks": "corners",
    "Offsides": "offsides",
    "Ball Possession": "possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "passes",
    "Passes accurate": "passes_completed",
    "Passes %": "pass_accuracy",
    "expected_goals": "xg",
}

# Pre-match, venue-specific attack and opponent defense are separate inputs.
ATTACK_STATS = ["shots", "shots_on_target", "shot_efficiency"]
DEFENSE_STATS = ["goals_conceded", "shots_on_target_conceded", "save_rate", "block_rate"]
BASE_STATS = ATTACK_STATS + DEFENSE_STATS
ROLLING_FEATURES = BASE_STATS.copy()
MODEL_FEATURES = ["elo_diff"]
TEAM_MODEL_FEATURES = (["elo_diff"] + [f"own_{x}" for x in ATTACK_STATS]
                       + [f"opponent_{x}" for x in DEFENSE_STATS])
NEGATIVE_FEATURES = {"opponent_save_rate", "opponent_block_rate"}
TEAM_INTERCEPT_PRIOR_GAMES = 20.0
STAT_WEIGHTING = "opponent_pre_match_venue_elo_over_1500"


def team_inputs(frame: pd.DataFrame, *, is_home: bool) -> pd.DataFrame:
    own, opponent = ("home", "away") if is_home else ("away", "home")
    X = pd.DataFrame(index=frame.index)
    X["elo_diff"] = frame.elo_diff if is_home else -frame.elo_diff
    for stat in ATTACK_STATS:
        X[f"own_{stat}"] = frame[f"{own}_{stat}"]
    for stat in DEFENSE_STATS:
        X[f"opponent_{stat}"] = frame[f"{opponent}_{stat}"]
    return X[TEAM_MODEL_FEATURES]


class DefensivePoissonRegressor(RegressorMixin, BaseEstimator):
    """Sign-constrained Poisson GLM with optional pooled-coefficient prior.

    Minimize weighted Poisson loss + L2(beta) + prior_strength/2 * ||beta-prior||^2.
    The intercept is not penalized. Bounds are enforced DURING fitting;
    clipping fitted coefficients afterwards would not optimize this model.
    """
    def __init__(self, alpha=0.01, max_iter=2000, tol=1e-8,
                 prior_coef=None, prior_strength=0.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.prior_coef = prior_coef
        self.prior_strength = prior_strength

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if self.alpha < 0 or np.any(y < 0):
            raise ValueError("Regularization and goals must be nonnegative.")
        if X.shape[1] != len(TEAM_MODEL_FEATURES):
            raise ValueError("Unexpected feature count; preserve empty imputer columns.")
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if (weights.shape != y.shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("Invalid sample weights.")
        weights = weights / weights.sum()
        mean = float(weights @ y)
        if mean <= 0:
            raise ValueError("At least one positively weighted goal is required.")
        prior = np.zeros(X.shape[1]) if self.prior_coef is None else np.asarray(self.prior_coef, float)
        if prior.shape != (X.shape[1],) or not np.isfinite(prior).all() or self.prior_strength < 0:
            raise ValueError("Invalid pooled coefficient prior.")
        initial = np.zeros(X.shape[1] + 1)
        initial[1:] = prior
        initial[0] = np.log(mean)

        def objective(parameters):
            intercept, beta = parameters[0], parameters[1:]
            eta = intercept + X @ beta
            with np.errstate(over="ignore", invalid="ignore"):
                mu = np.exp(eta)
                delta = beta - prior
                loss = (weights @ (mu - y * eta) + self.alpha * (beta @ beta) / 2
                        + self.prior_strength * (delta @ delta) / 2)
                residual = weights * (mu - y)
                gradient = np.r_[residual.sum(), X.T @ residual + self.alpha * beta + self.prior_strength * delta]
            return loss, gradient

        bounds = [(None, None)] + [
            (None, 0.0) if name in NEGATIVE_FEATURES else (0.0, None)
            for name in TEAM_MODEL_FEATURES
        ]
        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": self.max_iter, "ftol": self.tol})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Constrained Poisson fit failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:]
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = result.nit
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count changed; retrain the model.")
        return np.exp(np.clip(self.intercept_ + X @ self.coef_, -700, 700))


def fit_team_models(
    frame: pd.DataFrame,
    regularization: float,
    sample_weight: np.ndarray | None = None,
) -> dict[int, dict[str, Pipeline]]:
    """Share slopes within each venue; shrink team scoring multipliers toward 1.

    The multiplier has a Gamma prior with mean 1 and exposure equivalent
    to TEAM_INTERCEPT_PRIOR_GAMES average league matches. Use its posterior
    mean, (weighted goals + prior goals) / (weighted exposure + prior goals).
    No team-specific slope refits or unpenalized team intercepts are used.
    """
    models: dict[int, dict[str, Pipeline]] = {}
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None and weights.shape not in ((len(frame),), (len(frame), 2)):
        raise ValueError("sample_weight must have shape (matches,) or (matches, 2).")

    # Fit shared preprocessing and slopes using this training partition only.
    pooled = {}
    for column, venue in enumerate(("home", "away")):
        venue_weights = None if weights is None else (weights if weights.ndim == 1 else weights[:, column])
        if frame[f"{venue}_goals"].sum() <= 0:
            continue
        base = estimator(regularization)
        kwargs = {} if venue_weights is None else {
            "scale__sample_weight": venue_weights, "poisson__sample_weight": venue_weights,
        }
        base.fit(team_inputs(frame, is_home=(venue == "home")), frame[f"{venue}_goals"], **kwargs)
        pooled[venue] = base

    for team_id in sorted(set(frame.home_id) | set(frame.away_id)):
        venues = {}
        for venue in ("home", "away"):
            mask = frame[f"{venue}_id"].eq(team_id).to_numpy()
            rows = frame.loc[mask]
            y = rows[f"{venue}_goals"]
            venue_weights = None if weights is None else (
                weights[mask] if weights.ndim == 1
                else weights[mask, 0 if venue == "home" else 1]
            )
            # Do not silently substitute the other venue or a shared model.
            positive_goals = y.sum() if venue_weights is None else np.dot(y, venue_weights)
            if len(y) < 2 or positive_goals <= 0:
                continue
            if venue not in pooled:
                continue
            # Shared slopes prevent small venue samples from independently
            # discarding defensive inputs. The offset preserves team identity.
            model = copy.deepcopy(pooled[venue])
            inputs = team_inputs(rows, is_home=(venue == "home"))
            exposure = model.predict(inputs)
            local_weights = np.ones(len(rows)) if venue_weights is None else venue_weights
            column = 0 if venue == "home" else 1
            pooled_weights = (None if weights is None else
                              weights if weights.ndim == 1 else weights[:, column])
            league_goals = float(np.average(frame[f"{venue}_goals"], weights=pooled_weights))
            prior_goals = TEAM_INTERCEPT_PRIOR_GAMES * league_goals
            multiplier = (float(np.dot(local_weights, y)) + prior_goals) / (
                float(np.dot(local_weights, exposure)) + prior_goals
            )
            model.named_steps["poisson"].intercept_ += math.log(multiplier)
            venues[venue] = model
        models[int(team_id)] = venues
    return models


def venue_model_coverage(models, frame: pd.DataFrame) -> np.ndarray:
    return np.array([
        "home" in models.get(int(h), {}) and "away" in models.get(int(a), {})
        for h, a in zip(frame.home_id, frame.away_id)
    ], dtype=bool)

def predict_team_rates(models, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Select the host's home formula and visitor's away formula."""
    rates = []
    for venue in ("home", "away"):
        values = np.empty(len(frame), dtype=float)
        for team_id in frame[f"{venue}_id"].unique():
            model = models.get(int(team_id), {}).get(venue)
            if model is None:
                raise ValueError(f"No {venue} formula for team {team_id}; insufficient training data.")
            mask = frame[f"{venue}_id"].eq(team_id).to_numpy()
            values[mask] = model.predict(team_inputs(frame.loc[mask], is_home=(venue == "home")))
        rates.append(np.clip(values, 0.05, 8.0))
    return rates[0], rates[1]


def print_team_formula(name: str, model: Pipeline) -> None:
    # Use the actual transformed feature names: the imputer may drop
    # columns that contain no observed values for a particular team.
    feature_names = model[:-1].get_feature_names_out(
        TEAM_MODEL_FEATURES
    )
    poisson_model = model.named_steps["poisson"]

    pairs = sorted(
        zip(feature_names, poisson_model.coef_),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )

    terms = " + ".join(
        f"({coefficient:+.5f} * z[{feature}])"
        for feature, coefficient in pairs
    )

    print(
        f"{name}: lambda = exp("
        f"{poisson_model.intercept_:.5f} + {terms})"
    )

def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row

    con.executescript(
        """
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS app_config (key TEXT PRIMARY KEY, value TEXT NOT NULL);

        CREATE TABLE IF NOT EXISTS fixtures (
          fixture_id INTEGER PRIMARY KEY,
          league_id INTEGER NOT NULL,
          season INTEGER NOT NULL,
          kickoff TEXT NOT NULL,
          status TEXT NOT NULL,
          home_id INTEGER NOT NULL,
          home_name TEXT NOT NULL,
          away_id INTEGER NOT NULL,
          away_name TEXT NOT NULL,
          home_goals INTEGER,
          away_goals INTEGER
        );

        CREATE TABLE IF NOT EXISTS team_stats (
          fixture_id INTEGER NOT NULL,
          team_id INTEGER NOT NULL,
          stats_json TEXT NOT NULL,
          PRIMARY KEY (fixture_id, team_id),
          FOREIGN KEY (fixture_id) REFERENCES fixtures(fixture_id)
        );
        """
    )

    # Automatically update databases created by an older script version.
    fixture_columns = {
        row["name"]
        for row in con.execute("PRAGMA table_info(fixtures)")
    }

    if "league_id" not in fixture_columns:
        con.close()
        raise SystemExit("Legacy database lacks league IDs. Re-download into a new database.")

    return con

class ApiFootball:
    def __init__(self, key: str, pause: float = 0.25):
        self.session = requests.Session()
        self.session.headers["x-apisports-key"] = key
        self.pause = pause

    def get(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        response = self.session.get(f"{API_URL}/{endpoint}", params=params, timeout=45)
        if response.status_code == 429:
            raise RuntimeError("API daily/rate limit reached. Run the same command tomorrow; cached work is preserved.")
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"API-Football error: {payload['errors']}")
        time.sleep(self.pause)
        return payload.get("response", [])


# Names are hints, never league IDs or API response order. Validate against the
# live country/type catalog. Unknown names require an explicit ordered choice.
DIVISION_NAMES = {
    "colombia": (("Primera A",), ("Primera B",)),
    "england": (("Premier League",), ("Championship",)),
    "germany": (("Bundesliga",), ("2. Bundesliga",)),
    "spain": (("La Liga",), ("Segunda División", "La Liga 2")),
    "italy": (("Serie A",), ("Serie B",)),
    "france": (("Ligue 1",), ("Ligue 2",)),
    "portugal": (("Primeira Liga",), ("Segunda Liga",)),
    "netherlands": (("Eredivisie",), ("Eerste Divisie",)),
    "belgium": (("Jupiler Pro League",), ("Challenger Pro League", "Second Division")),
    "scotland": (("Premiership",), ("Championship",)),
    "austria": (("Bundesliga",), ("2. Liga",)),
    "switzerland": (("Super League",), ("Challenge League",)),
    "turkey": (("Süper Lig",), ("1. Lig",)),
    "greece": (("Super League 1",), ("Super League 2",)),
    "poland": (("Ekstraklasa",), ("I Liga",)),
    "denmark": (("Superliga",), ("1. Division",)),
    "sweden": (("Allsvenskan",), ("Superettan",)),
    "norway": (("Eliteserien",), ("1. Division",)),
    "finland": (("Veikkausliiga",), ("Ykkösliiga",)),
    "iceland": (("Úrvalsdeild",), ("1. Deild",)),
    "ireland": (("Premier Division",), ("First Division",)),
    "brazil": (("Serie A",), ("Serie B",)),
    "argentina": (("Liga Profesional Argentina",), ("Primera Nacional",)),
    "chile": (("Primera División",), ("Primera B",)),
    "peru": (("Primera División",), ("Segunda División",)),
    "ecuador": (("Liga Pro",), ("Liga Pro Serie B",)),
    "mexico": (("Liga MX",), ("Liga de Expansión MX",)),
    "japan": (("J1 League",), ("J2 League",)),
    "south-korea": (("K League 1",), ("K League 2",)),
    "china": (("Super League",), ("League One",)),
    "saudi-arabia": (("Pro League",), ("Division 1",)),
    "south-africa": (("Premier Soccer League",), ("1st Division",)),
}


def country_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def read_config(con):
    row = con.execute("SELECT value FROM app_config WHERE key='country_divisions'").fetchone()
    return None if row is None else json.loads(row["value"])


def activate_config(config):
    global FIRST_DIVISION, SECOND_DIVISION_ID, LEAGUE_IDS, COUNTRY
    LEAGUE_IDS = tuple(int(x) for x in config["league_ids"])
    if not 1 <= len(LEAGUE_IDS) <= 2 or len(set(LEAGUE_IDS)) != len(LEAGUE_IDS):
        raise SystemExit("Configuration must contain one or two distinct ordered division IDs.")
    FIRST_DIVISION = LEAGUE_IDS[0]
    SECOND_DIVISION_ID = LEAGUE_IDS[1] if len(LEAGUE_IDS) == 2 else None
    COUNTRY = config["country"]


def persist_config(con, config):
    existing = read_config(con)
    if existing and existing != config:
        raise SystemExit("Database belongs to another country/division selection. Use a different --db.")
    cached = {int(row[0]) for row in con.execute("SELECT DISTINCT league_id FROM fixtures")}
    if not cached.issubset(set(config["league_ids"])):
        raise SystemExit("Database contains other leagues. Use a separate --db for this country.")
    con.execute("INSERT OR REPLACE INTO app_config VALUES ('country_divisions', ?)",
                (json.dumps(config),))
    con.commit()
    activate_config(config)


def resolve_divisions(api, args, con):
    country = country_key(args.country)
    existing = read_config(con)
    # The countries endpoint supplies canonical spelling for the league filter.
    countries = api.get("countries")
    canonical = [item["name"] for item in countries if country_key(item["name"]) == country]
    if len(canonical) != 1:
        raise SystemExit(f"Country {args.country!r} not found. Use its API-Football name (e.g. England).")
    catalog = api.get("leagues", country=canonical[0], type="league")
    leagues = {int(item["league"]["id"]): item for item in catalog
               if item["league"]["type"].lower() == "league"
               and country_key(item["country"]["name"]) == country}
    if not leagues:
        raise SystemExit(f"No league competitions available for {canonical[0]}.")
    selected = args.division_ids
    if selected is None and existing:
        selected = existing["league_ids"]
    if selected is None:
        names = DIVISION_NAMES.get(country)
        if names:
            matches = [[lid for lid, entry in leagues.items()
                        if country_key(entry["league"]["name"]) in {country_key(n) for n in aliases}]
                       for aliases in names]
            if len(matches[0]) == 1 and len(matches[1]) == 1:
                selected = [matches[0][0], matches[1][0]]
            # Only accept a single division automatically if it is the sole
            # senior men's league candidate, not merely an unrecognized tier 2.
            elif len(matches[0]) == 1 and not matches[1]:
                senior = [lid for lid, entry in leagues.items()
                          if not re.search(r"women|femin|\bu\s*-?\s*\d{2}\b|youth|reserve",
                                           entry["league"]["name"], re.I)]
                if senior == matches[0]:
                    selected = matches[0]
    if selected is None and len(leagues) == 1:
        lid, entry = next(iter(leagues.items()))
        if not re.search(r"women|femin|\bu\s*-?\s*\d{2}\b|youth|reserve",
                         entry["league"]["name"], re.I):
            selected = [lid]
    if selected is None:
        print(f"Available league competitions for {canonical[0]}:")
        for lid, entry in sorted(leagues.items()):
            print(f"  {lid}: {entry['league']['name']}")
        if not sys.stdin.isatty():
            raise SystemExit("Division ranks cannot be determined safely. Rerun with --division-ids FIRST [SECOND].")
        answer = input("Enter first division ID, then second division ID (omit second if unavailable): ")
        try:
            selected = [int(value) for value in answer.replace(",", " ").split()]
        except ValueError:
            raise SystemExit("Enter one or two numeric league IDs.")
    if (not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected)
            or any(lid not in leagues for lid in selected)):
        raise SystemExit("Choose one or two distinct league IDs from this country's catalog, first division first.")
    config = {"country": country, "league_ids": selected}
    persist_config(con, config)
    result = [leagues[lid] for lid in selected]
    for rank, entry in enumerate(result, 1):
        print(f"Division {rank}: {entry['league']['name']} (ID {entry['league']['id']})")
        # Require explicit match-statistics coverage for each season.
        entry["statistics_seasons"] = [
            int(season["year"])
            for season in entry.get("seasons", [])
            if ((season.get("coverage") or {}).get("fixtures") or {}).get(
                "statistics_fixtures"
            ) is True
        ]
        for season in entry.get("seasons", []):
            year = int(season["year"])
            if (args.from_season <= year <= args.to_season
                    and year not in entry["statistics_seasons"]):
                print(f"  {year}: match statistics not covered; statistics requests skipped")
    return result


def configure_command(args):
    country = country_key(args.country)
    if not country:
        raise SystemExit("Country cannot be empty.")
    args.country = country
    args.db = args.db or Path(f"{country}.sqlite3")
    if hasattr(args, "model"):
        args.model = args.model or Path(f"{country}_poisson.joblib")
    if args.command == "backtest":
        args.output = args.output or Path(f"{country}_backtest_{args.test_season}.csv")
    if args.command == "download":
        args.db.parent.mkdir(parents=True, exist_ok=True)
        if args.db.exists():
            con = connect(args.db)
            try:
                config = read_config(con)
                if config and config["country"] != country:
                    raise SystemExit("Database belongs to another country; choose a different --db.")
            finally:
                con.close()
        return
    if not args.db.is_file():
        raise SystemExit(f"Database not found: {args.db}. Download this country first.")
    con = connect(args.db)
    try:
        config = read_config(con)
        if config is None:
            # Migration for the exact two league IDs in the supplied script.
            ids = {int(row[0]) for row in con.execute("SELECT DISTINCT league_id FROM fixtures")}
            if country == "colombia" and ids and ids.issubset({239, 240}):
                config = {"country": country, "league_ids": [239, 240]}
                persist_config(con, config)
            else:
                raise SystemExit("Database has no country configuration. Run download for this country first.")
        if config["country"] != country:
            raise SystemExit("Database country differs from --country.")
        activate_config(config)
    finally:
        con.close()
    if hasattr(args, "league_id"):
        args.league_id = args.league_id if args.league_id is not None else FIRST_DIVISION
        if args.league_id not in LEAGUE_IDS:
            raise SystemExit(f"--league-id must be one of {LEAGUE_IDS}.")


def save_fixtures(
    con: sqlite3.Connection,
    api: ApiFootball,
    seasons: Iterable[int],
    leagues: list[dict],
) -> None:
    for season in seasons:
        for entry in leagues:
            league_id = int(entry["league"]["id"])
            available = {int(item["year"]): item for item in entry.get("seasons", [])}
            if season not in available:
                print(f"{entry['league']['name']} {season}: season unavailable; skipped")
                continue
            items = api.get(
                "fixtures",
                league=league_id,
                season=season,
            )

            for item in items:
                fixture = item["fixture"]
                teams = item["teams"]
                goals = item["goals"]

                con.execute(
                    """
                    INSERT INTO fixtures (
                        fixture_id,
                        league_id,
                        season,
                        kickoff,
                        status,
                        home_id,
                        home_name,
                        away_id,
                        away_name,
                        home_goals,
                        away_goals
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(fixture_id) DO UPDATE SET
                        league_id = excluded.league_id,
                        status = excluded.status,
                        home_goals = excluded.home_goals,
                        away_goals = excluded.away_goals
                    """,
                    (
                        fixture["id"],
                        league_id,
                        season,
                        fixture["date"],
                        fixture["status"]["short"],
                        teams["home"]["id"],
                        teams["home"]["name"],
                        teams["away"]["id"],
                        teams["away"]["name"],
                        goals["home"],
                        goals["away"],
                    ),
                )

            con.commit()

            league_name = entry["league"]["name"]

            print(
                f"{league_name} "
                f"{season}-{str(season + 1)[-2:]}: "
                f"cached {len(items)} fixtures"
            )


def number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.rstrip("%").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def fetch_missing_stats(
        con: sqlite3.Connection, api: ApiFootball, maximum: int, leagues: list[dict]
) -> None:
    supported = {
        (int(entry["league"]["id"]), year)
        for entry in leagues
        for year in entry["statistics_seasons"]
    }
    rows = con.execute(
        """SELECT f.fixture_id, f.league_id, f.season FROM fixtures f
           WHERE f.status IN ('FT','AET','PEN')
             AND (SELECT COUNT(*) FROM team_stats s WHERE s.fixture_id=f.fixture_id) < 2
           ORDER BY f.kickoff"""
    ).fetchall()
    eligible = [row for row in rows if (row["league_id"], row["season"]) in supported]
    skipped = len(rows) - len(eligible)
    if skipped:
        print(f"Skipped statistics for {skipped} cached matches without confirmed season coverage.")
    rows = eligible
    completed = 0
    for row in rows[:maximum]:
        fixture_id = row["fixture_id"]
        response = api.get("fixtures/statistics", fixture=fixture_id)
        if len(response) != 2:
            print(f"Fixture {fixture_id}: statistics unavailable ({len(response)} team records)", file=sys.stderr)
            continue
        for team_block in response:
            parsed = {name: None for name in STAT_ALIASES.values()}
            for stat in team_block.get("statistics", []):
                internal = STAT_ALIASES.get(stat.get("type"))
                if internal:
                    parsed[internal] = number(stat.get("value"))
            con.execute(
                "INSERT OR REPLACE INTO team_stats VALUES (?, ?, ?)",
                (fixture_id, team_block["team"]["id"], json.dumps(parsed, sort_keys=True)),
            )
        con.commit()
        completed += 1
        print(f"Statistics {completed}/{min(len(rows), maximum)}: fixture {fixture_id}")
    print(f"Downloaded statistics for {completed} matches; {max(0, len(rows)-completed)} remain.")


def download(args: argparse.Namespace) -> None:
    if args.from_season > args.to_season or args.max_stat_requests < 0 or args.pause < 0:
        raise SystemExit("Invalid season range, request count or pause.")
    key = API_FOOTBALL_KEY
    if not key:
        raise SystemExit("Set API_FOOTBALL_KEY before downloading.")
    con = connect(args.db)
    try:
        api = ApiFootball(key, args.pause)
        leagues = resolve_divisions(api, args, con)
        save_fixtures(con, api, range(args.from_season, args.to_season + 1), leagues)
        fetch_missing_stats(con, api, args.max_stat_requests, leagues)
    finally:
        con.close()


def load_matches(con: sqlite3.Connection) -> pd.DataFrame:
    fixtures = pd.read_sql_query(
        """SELECT * FROM fixtures WHERE status IN ('FT','AET','PEN')
           AND home_goals IS NOT NULL AND away_goals IS NOT NULL ORDER BY kickoff, fixture_id""", con
    )
    stats = pd.read_sql_query("SELECT * FROM team_stats", con)
    if fixtures.empty or stats.empty:
        raise SystemExit("No completed fixtures/statistics found. Run the download command first.")
    expanded = pd.json_normalize(stats.pop("stats_json").map(json.loads))
    stats = pd.concat([stats, expanded], axis=1)
    home = stats.add_prefix("home_").rename(columns={"home_fixture_id": "fixture_id"})
    away = stats.add_prefix("away_").rename(columns={"away_fixture_id": "fixture_id"})
    data = fixtures.merge(home, left_on=["fixture_id", "home_id"], right_on=["fixture_id", "home_team_id"])
    data = data.merge(away, left_on=["fixture_id", "away_id"], right_on=["fixture_id", "away_team_id"])
    return data.sort_values(["kickoff", "fixture_id"]).reset_index(drop=True)


@dataclass
class TeamState:
    home_elo: float = INITIAL_ELO
    away_elo: float = INITIAL_ELO
    games: int = 0
    home_games: int = 0
    away_games: int = 0
    home_ewma: dict[str, float] = field(default_factory=dict)
    away_ewma: dict[str, float] = field(default_factory=dict)
    home_ewma_mass: dict[str, float] = field(default_factory=dict)
    away_ewma_mass: dict[str, float] = field(default_factory=dict)

    # Promoted teams calibrate their home and away Elo independently.
    calibrating_home_elo: bool = False
    calibrating_away_elo: bool = False

    # Each entry contains: (opponent venue Elo, result)
    home_elo_calibration_games: list[tuple[float, float]] = field(
        default_factory=list
    )
    away_elo_calibration_games: list[tuple[float, float]] = field(
        default_factory=list
    )


def infer_promoted_team_elo(
    games: list[tuple[float, float]],
) -> float:
    """
    Infer the Elo whose expected points equal the points earned in the
    calibration games.

    Each tuple contains:
        (opponent_venue_elo, actual_score)

    actual_score:
        1.0 = win
        0.5 = draw
        0.0 = loss
    """
    actual_points = sum(score for _, score in games)

    low = 800.0
    high = 2200.0

    for _ in range(80):
        candidate = (low + high) / 2.0

        expected_points = sum(
            1.0 / (
                1.0
                + 10.0 ** ((opponent_elo - candidate) / 400.0)
            )
            for opponent_elo, _ in games
        )

        if expected_points < actual_points:
            low = candidate
        else:
            high = candidate

    return (low + high) / 2.0

def ewma_update(
    old: float | None, new: float | None, alpha: float, mass: float,
) -> tuple[float | None, float]:
    """Normalized exponential average; each earlier game's weight is smaller.

    Track the sum of weights rather than seeding the first game at full mass.
    A missing statistic adds no observation, but existing mass still decays
    for that played game. This preserves game-order gaps between valid values.
    """
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("--ewma-alpha must be strictly between 0 and 1.")
    decayed_mass = (1.0 - alpha) * mass
    if new is None or pd.isna(new):
        return old, decayed_mass
    if old is None or pd.isna(old):
        return float(new), 1.0
    new_mass = decayed_mass + 1.0
    return (decayed_mass * old + float(new)) / new_mass, new_mass


def feature_difference(home_value: float | None, away_value: float | None) -> float:
    """Return a numeric difference, or NaN when either statistic is unavailable."""
    if home_value is None or away_value is None:
        return np.nan
    if pd.isna(home_value) or pd.isna(away_value):
        return np.nan
    return float(home_value) - float(away_value)

def shot_efficiency(goals: float, shots: float | None) -> float:
    """Goals per total shot; missing counts stay missing, zero shots give zero."""
    if shots is None or pd.isna(shots) or pd.isna(goals) or shots < 0:
        return np.nan
    if shots == 0:
        return 0.0 if goals == 0 else np.nan
    return float(goals) / float(shots)


def opponent_elo_weighted_stats(values: dict, opponent_elo: float) -> dict:
    """Adjust historical performance using the opponent's pre-match venue Elo.

    Strength = opponent Elo / INITIAL_ELO (1500). Multiply attacking stats
    and defensive success scores; divide goals/shots conceded. Adjusted rates
    are feature scores, not probabilities, and may exceed 1. No clipping.
    Apply after raw rates are derived, before EWMA and before Elo updates.
    """
    if not math.isfinite(opponent_elo) or opponent_elo <= 0:
        raise ValueError("Opponent Elo must be finite and positive.")
    strength = opponent_elo / INITIAL_ELO
    adjusted = dict(values)
    for feature in ROLLING_FEATURES:
        value = values.get(feature)
        if value is None or pd.isna(value):
            adjusted[feature] = np.nan
        elif feature in {"goals_conceded", "shots_on_target_conceded"}:
            adjusted[feature] = float(value) / strength
        else:
            adjusted[feature] = float(value) * strength
    return adjusted


def match_inputs(home: TeamState, away: TeamState) -> dict[str, float]:
    """Use opponent-Elo-adjusted venue EWMAs plus a separate Elo difference.

    Historical stats were already adjusted before averaging; do not apply
    the upcoming opponent's Elo a second time. Shared by all command paths.
    """
    result = {"elo_diff": (home.home_elo - away.away_elo) / 400.0}
    for stat in ROLLING_FEATURES:
        result[f"home_{stat}"] = home.home_ewma.get(stat, np.nan)
        result[f"away_{stat}"] = away.away_ewma.get(stat, np.nan)
    return result


def build_features(
    matches: pd.DataFrame,
    alpha: float,
    min_history: int,
) -> tuple[pd.DataFrame, dict[int, TeamState], dict[int, str]]:
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("--ewma-alpha must be strictly between 0 and 1.")
    matches = matches.copy()
    matches["kickoff"] = pd.to_datetime(matches.kickoff, utc=True)
    matches = matches.sort_values(["kickoff", "fixture_id"])
    states: dict[int, TeamState] = defaultdict(TeamState)
    names: dict[int, str] = {}
    examples: list[dict[str, Any]] = []

    # Earliest-season first-division teams start at 1500; second division
    # retains the supplied 1350 initialization and venue calibration rules.
    first_season = int(matches["season"].min())

    first_season_rows = matches[
        matches["season"] == first_season
        ]

    first_division_rows = first_season_rows[
        first_season_rows["league_id"] == FIRST_DIVISION
        ]

    second_division_rows = first_season_rows[
        first_season_rows["league_id"] == SECOND_DIVISION_ID
        ]

    first_season_first_division_teams = (
            set(first_division_rows["home_id"])
            | set(first_division_rows["away_id"])
    )

    first_season_second_division_teams = (
            set(second_division_rows["home_id"])
            | set(second_division_rows["away_id"])
    )

    first_season_teams = (
            first_season_first_division_teams
            | first_season_second_division_teams
    )

    seen_teams: set[int] = set()

    for row in matches.itertuples(index=False):
        # Initialize teams when they first appear.
        for team_id in (row.home_id, row.away_id):
            if team_id not in seen_teams:
                starts_in_second_division = (
                        row.league_id == SECOND_DIVISION_ID
                )

                if team_id in first_season_first_division_teams:
                    initial_elo = INITIAL_ELO
                elif team_id in first_season_second_division_teams:
                    initial_elo = SECOND_DIVISION_INITIAL_ELO
                else:
                    initial_elo = (
                        SECOND_DIVISION_INITIAL_ELO
                        if starts_in_second_division
                        else INITIAL_ELO
                    )

                needs_promotion_calibration = team_id not in first_season_teams

                states[team_id] = TeamState(
                    home_elo=initial_elo,
                    away_elo=initial_elo,
                    calibrating_home_elo=needs_promotion_calibration,
                    calibrating_away_elo=needs_promotion_calibration,
                )

                seen_teams.add(team_id)

        home = states[row.home_id]
        away = states[row.away_id]

        names[row.home_id] = row.home_name
        names[row.away_id] = row.away_name

        # A match cannot be used for training if the promoted home team's
        # home Elo or promoted away team's away Elo is still provisional.
        elo_is_calibrated = (
            not home.calibrating_home_elo
            and not away.calibrating_away_elo
        )

        record: dict[str, Any] = {
            "fixture_id": row.fixture_id,
            "kickoff": row.kickoff,
            "home_id": row.home_id,
            "away_id": row.away_id,
            "home_name": row.home_name,
            "away_name": row.away_name,
            "home_goals": row.home_goals,
            "away_goals": row.away_goals,
            "elo_diff": (
                home.home_elo - away.away_elo
            ) / 400.0,
            "eligible": (
                elo_is_calibrated
                and home.home_games >= min_history
                and away.away_games >= min_history
            ),
        }

        record.update(match_inputs(home, away))

        examples.append(record)

        # Update rolling statistics after generating the pre-match features
        # to prevent target leakage. Derive rates from raw observations first.
        hvals: dict[str, float | None] = {}
        avals: dict[str, float | None] = {}

        for stat in ("shots", "shots_on_target", "blocked_shots", "goalkeeper_saves"):
            hvals[stat] = getattr(row, f"home_{stat}", np.nan)
            avals[stat] = getattr(row, f"away_{stat}", np.nan)
        hvals["shot_efficiency"] = shot_efficiency(row.home_goals, hvals["shots"])
        avals["shot_efficiency"] = shot_efficiency(row.away_goals, avals["shots"])

        for values, opponent_values, conceded in (
            (hvals, avals, row.away_goals), (avals, hvals, row.home_goals),
        ):
            values["goals_conceded"] = float(conceded)
            values["shots_on_target_conceded"] = opponent_values["shots_on_target"]
            saves = values["goalkeeper_saves"]
            values["save_rate"] = (
                saves / (saves + conceded)
                if pd.notna(saves) and saves >= 0 and saves + conceded > 0 else np.nan
            )
            # API blocked shots belong to the attacking team, so use the
            # opponent's blocked attempts to measure this team's blocking.
            shots, blocked = opponent_values["shots"], opponent_values["blocked_shots"]
            values["block_rate"] = (
                blocked / shots if pd.notna(shots) and pd.notna(blocked)
                and shots > 0 and 0 <= blocked <= shots else np.nan
            )

        # Opponents' venue ratings are still pre-match here. Never use the
        # post-result Elo or the team's own Elo to weight its performance.
        hvals = opponent_elo_weighted_stats(hvals, away.away_elo)
        avals = opponent_elo_weighted_stats(avals, home.home_elo)

        for feature in ROLLING_FEATURES:
            (home.home_ewma[feature], home.home_ewma_mass[feature]) = ewma_update(
                home.home_ewma.get(feature),
                hvals[feature],
                alpha,
                home.home_ewma_mass.get(feature, 0.0),
            )

            (away.away_ewma[feature], away.away_ewma_mass[feature]) = ewma_update(
                away.away_ewma.get(feature),
                avals[feature],
                alpha,
                away.away_ewma_mass.get(feature, 0.0),
            )

        score_home = (
            1.0
            if row.home_goals > row.away_goals
            else 0.5
            if row.home_goals == row.away_goals
            else 0.0
        )

        score_away = 1.0 - score_home

        # A promoted team's first five home games calibrate home_elo.
        # Its first five away games independently calibrate away_elo.
        #
        # If either rating used in this fixture is provisional, neither
        # participating team's relevant Elo is updated.
        skip_regular_elo_update = (
            home.calibrating_home_elo
            or away.calibrating_away_elo
        )

        if home.calibrating_home_elo:
            home.home_elo_calibration_games.append(
                (
                    away.away_elo,
                    score_home,
                )
            )

        if away.calibrating_away_elo:
            away.away_elo_calibration_games.append(
                (
                    home.home_elo,
                    score_away,
                )
            )

        # Infer ratings only after both teams have recorded the opponent's
        # pre-match rating. This matters when two promoted teams play.
        if (
            home.calibrating_home_elo
            and len(home.home_elo_calibration_games)
            >= PROMOTED_TEAM_CALIBRATION_GAMES
        ):
            home.home_elo = infer_promoted_team_elo(
                home.home_elo_calibration_games
            )
            home.calibrating_home_elo = False

        if (
            away.calibrating_away_elo
            and len(away.away_elo_calibration_games)
            >= PROMOTED_TEAM_CALIBRATION_GAMES
        ):
            away.away_elo = infer_promoted_team_elo(
                away.away_elo_calibration_games
            )
            away.calibrating_away_elo = False

        # Normal Elo adjustment starts only when the home team's home Elo
        # and the away team's away Elo were already calibrated before kickoff.
        if not skip_regular_elo_update:
            expected_home = 1.0 / (
                1.0
                + 10.0
                ** (
                    (
                        away.away_elo
                        - home.home_elo
                    )
                    / 400.0
                )
            )

            multiplier = (
                1.0
                if row.home_goals == row.away_goals
                else math.sqrt(
                    abs(row.home_goals - row.away_goals)
                )
            )

            change = (
                20.0
                * multiplier
                * (score_home - expected_home)
            )

            home.home_elo += change
            away.away_elo -= change

        # Calibration matches still count for rolling-history requirements.
        home.games += 1
        away.games += 1
        home.home_games += 1
        away.away_games += 1

    frame = pd.DataFrame(examples)

    return (
        frame[frame["eligible"]]
        .drop(columns="eligible")
        .reset_index(drop=True),
        dict(states),
        names,
    )

def estimator(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("poisson", DefensivePoissonRegressor(alpha=alpha, max_iter=2000)),
    ])


def print_formula(label: str, model: Pipeline) -> None:
    """Print lambda=exp(intercept + sum beta*z) with standardized coefficients."""
    poisson_model = model.named_steps["poisson"]
    pairs = sorted(zip(TEAM_MODEL_FEATURES, poisson_model.coef_), key=lambda x: abs(x[1]), reverse=True)
    terms = " + ".join(f"({coefficient:+.5f} * z[{name}])" for name, coefficient in pairs)
    print(f"{label}: lambda = exp({poisson_model.intercept_:.5f} + {terms})")


def outcome_probabilities(lambda_home: float, lambda_away: float, max_goals: int = 10, *, rho: float = 0.0) -> tuple[float, float, float]:
    if not all(math.isfinite(x) and x > 0 for x in (lambda_home, lambda_away)):
        raise ValueError("Expected goals must be finite and positive.")
    if not math.isfinite(rho):
        raise ValueError("Dixon-Coles rho must be finite.")
    lower = max(-1 / lambda_home, -1 / lambda_away) + 1e-9
    upper = min(1.0, 1 / (lambda_home * lambda_away)) - 1e-9
    rho = float(np.clip(rho, lower, upper))
    # The four low-score corrections preserve total probability and margins.
    correction = rho * lambda_home * lambda_away * math.exp(-lambda_home - lambda_away)
    p = np.array([skellam.sf(0, lambda_home, lambda_away) + correction,
                  skellam.pmf(0, lambda_home, lambda_away) - 2 * correction,
                  skellam.cdf(-1, lambda_home, lambda_away) + correction])
    p = np.maximum(p, 0)
    return tuple(float(x) for x in p / p.sum())


def evaluate(home_model: Pipeline, away_model: Pipeline, test: pd.DataFrame) -> None:
    lh = np.clip(home_model.predict(team_inputs(test, is_home=True)), 0.05, 8)
    la = np.clip(away_model.predict(team_inputs(test, is_home=False)), 0.05, 8)
    probabilities = np.array([outcome_probabilities(h, a) for h, a in zip(lh, la)])
    outcomes = np.where(test.home_goals > test.away_goals, 0, np.where(test.home_goals == test.away_goals, 1, 2))
    print(f"Holdout matches: {len(test)}")
    print(f"1X2 log loss: {log_loss(outcomes, probabilities, labels=[0,1,2]):.4f}")
    print(f"Home-goal Poisson deviance: {mean_poisson_deviance(test.home_goals, lh):.4f}")
    print(f"Away-goal Poisson deviance: {mean_poisson_deviance(test.away_goals, la):.4f}")

def recency_decay_value(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number < 1:
        raise argparse.ArgumentTypeError("--recency-decay must be strictly between 0 and 1.")
    return number


def game_weights(frame: pd.DataFrame, cutoff: pd.Timestamp, decay: float) -> np.ndarray:
    """Return positional [home, away] weights based on each team's venue history.

    Latest game = 1; preceding game = decay; preceding one = decay**2.
    Calendar gaps never change the ratio. Compute before filtering coverage
    so removing an unsupported fixture cannot change the remaining weights.
    """
    if not math.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("Recency decay must be strictly between 0 and 1.")
    cutoff = pd.to_datetime(cutoff, utc=True)
    kickoff = pd.to_datetime(frame.kickoff, utc=True)
    if pd.isna(cutoff) or kickoff.isna().any():
        raise ValueError("Recency weighting requires valid kickoff dates and cutoff.")
    if (kickoff > cutoff).any():
        raise ValueError("Training matches must not be later than the fitting cutoff.")
    ordered = frame[["fixture_id", "home_id", "away_id"]].reset_index(drop=True).copy()
    ordered["kickoff"] = kickoff.to_numpy()
    ordered = ordered.sort_values(["kickoff", "fixture_id"], ascending=False)
    weights = np.empty((len(frame), 2), dtype=float)
    for column, venue in enumerate(("home", "away")):
        newer_games = ordered.groupby(f"{venue}_id").cumcount().to_numpy()
        values = np.power(decay, newer_games)
        if np.any(values == 0):
            raise ValueError("Decay is too strong for this history; increase --recency-decay.")
        weights[ordered.index.to_numpy(), column] = values
    return weights


def fit_rho(frame, home, away, weights) -> float:
    if frame.empty:
        return 0.0
    # Keep all four Dixon-Coles correction factors positive.
    lower = max(float(np.max(-1 / home)), float(np.max(-1 / away))) + 1e-9
    upper = min(1.0, float(np.min(1 / (home * away)))) - 1e-9
    hg, ag = frame.home_goals.to_numpy(), frame.away_goals.to_numpy()
    def objective(rho):
        tau = np.ones(len(frame))
        tau[(hg == 0) & (ag == 0)] = (1 - home * away * rho)[(hg == 0) & (ag == 0)]
        tau[(hg == 0) & (ag == 1)] = (1 + home * rho)[(hg == 0) & (ag == 1)]
        tau[(hg == 1) & (ag == 0)] = (1 + away * rho)[(hg == 1) & (ag == 0)]
        tau[(hg == 1) & (ag == 1)] = 1 - rho
        return -float(np.sum(weights * np.log(tau)))
    if not np.any((hg <= 1) & (ag <= 1)):
        return 0.0
    fit = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    return float(fit.x) if fit.success else 0.0


def has_venue_models(models: dict, home_id: int, away_id: int) -> bool:
    return "home" in models.get(int(home_id), {}) and "away" in models.get(int(away_id), {})


def supported_fixtures(models: dict, frame: pd.DataFrame) -> pd.DataFrame:
    mask = [has_venue_models(models, h, a) for h, a in zip(frame.home_id, frame.away_id)]
    return frame.loc[np.asarray(mask, dtype=bool)].copy()


def fit_team_bundle(data, cutoff, regularization, recency_decay, use_dc):
    weights = game_weights(data, cutoff, recency_decay)
    models = fit_team_models(data, regularization, weights)
    coverage = venue_model_coverage(models, data)
    covered = data.loc[coverage]
    rho = 0.0
    if use_dc and not covered.empty:
        home, away = predict_team_rates(models, covered)
        # Both teams' venue histories contribute to the shared correction.
        dc_weights = np.sqrt(weights[coverage, 0]) * np.sqrt(weights[coverage, 1])
        rho = fit_rho(covered, home, away, dc_weights)
    return models, rho


def select_parameters(data, regularization, recency_decay, use_dc, league_id):
    """Select on two expanding chronological folds inside training data only.

    Score identical fixtures for every candidate. The external holdout/test
    season is never passed here. Keep the requested baseline on near ties.
    """
    ordered = data.sort_values(["kickoff", "fixture_id"])
    baseline = (regularization, recency_decay, use_dc)
    if len(ordered) < 300:
        return baseline, {"reason": "fewer than 300 training matches", "folds": []}
    folds = []
    for start, stop in ((0.6, 0.8), (0.8, 1.0)):
        cutoff = ordered.iloc[int(len(ordered) * start)].kickoff
        end = (ordered.iloc[int(len(ordered) * stop)].kickoff
               if stop < 1 else ordered.kickoff.max() + pd.Timedelta(nanoseconds=1))
        history = ordered[ordered.kickoff < cutoff]
        validation = ordered[(ordered.kickoff >= cutoff) & (ordered.kickoff < end)
                             & (ordered.league_id == league_id)]
        models, _ = fit_team_bundle(history, cutoff, *baseline)
        validation = supported_fixtures(models, validation)
        if len(validation) < 30:
            return baseline, {"reason": "insufficient validation coverage", "folds": []}
        folds.append((history, validation, cutoff))
    candidates = [baseline]
    candidates += [(a, d, dc) for a in (0.01, 0.1, 1.0)
                   for d in (0.95, 0.98, 0.995) for dc in (False, True)
                   if (a, d, dc) != baseline]
    scores = []
    for candidate in candidates:
        losses = []
        for history, validation, cutoff in folds:
            models, rho = fit_team_bundle(history, cutoff, *candidate)
            home, away = predict_team_rates(models, validation)
            p = shared_probabilities(home, away, rho)
            losses.extend(-np.log(np.clip(p[np.arange(len(p)), actual_outcomes(validation)], 1e-15, 1)))
        scores.append(float(np.mean(losses)))
    best = int(np.argmin(scores))
    if scores[0] - scores[best] < 0.002:
        best = 0
    report = {"metric": "1X2 log loss", "baseline_loss": scores[0],
              "selected_loss": scores[best], "selected": list(candidates[best]),
              "folds": [{"cutoff": c.isoformat(), "matches": len(v)} for _, v, c in folds],
              "candidates": [{"parameters": list(c), "loss": loss} for c, loss in zip(candidates, scores)]}
    print(f"Internal validation: log loss {scores[0]:.4f} -> {scores[best]:.4f}; "
          f"alpha/decay/DC={candidates[best]}")
    return candidates[best], report


def shared_probabilities(home, away, rho):
    return np.asarray([
        outcome_probabilities(float(h), float(a), rho=rho)
        for h, a in zip(home, away)
    ], dtype=float).reshape(-1, 3)


def actual_outcomes(frame):
    return np.where(frame.home_goals > frame.away_goals, 0,
                    np.where(frame.home_goals == frame.away_goals, 1, 2))


def report_win_calibration(actual, probabilities):
    """Report genuinely held-out win confidence, including the away subset."""
    actual = np.asarray(actual)
    p = np.asarray(probabilities)
    winners = np.where(p[:, 0] >= p[:, 2], 0, 2)
    confidence = p[np.arange(len(p)), winners]
    for label, mask in (
        ("All win predictions >=60%", confidence >= 0.60),
        ("Away win predictions >=60%", (confidence >= 0.60) & (winners == 2)),
        ("Away win predictions 55-65%", (confidence >= 0.55) & (confidence < 0.65) & (winners == 2)),
    ):
        n = int(mask.sum())
        if not n:
            print(f"{label}: 0 matches; reliability not established")
            continue
        successes = int(np.sum(actual[mask] == winners[mask]))
        observed = successes / n
        z = 1.96
        denominator = 1 + z*z/n
        center = (observed + z*z/(2*n))/denominator
        radius = z * math.sqrt(observed*(1-observed)/n + z*z/(4*n*n))/denominator
        print(f"{label}: {successes}/{n} ({observed:.2%}); "
              f"mean forecast {confidence[mask].mean():.2%}; "
              f"95% Wilson interval {center-radius:.2%}-{center+radius:.2%}")


def train(args: argparse.Namespace) -> None:
    # Fit shared venue slopes and regularized team scoring offsets.
    print(f"Game-order decay: {args.recency_decay:g} per previous team/venue game")

    if not args.db.is_file():
        raise SystemExit(f"Database not found: {args.db}")
    if not 0 < args.test_fraction < 1:
        raise SystemExit("--test-fraction must be between 0 and 1.")
    if args.regularization < 0:
        raise SystemExit("Regularization must be nonnegative.")
    con = connect(args.db)
    try:
        matches = load_matches(con)
    finally:
        con.close()
    matches["kickoff"] = pd.to_datetime(matches.kickoff, utc=True)
    matches = matches[
        matches.league_id.isin(LEAGUE_IDS)
        & (matches.status == "FT")
        & (matches.kickoff < pd.Timestamp.now(tz="UTC"))
    ].copy().sort_values(["kickoff", "fixture_id"])
    if matches.empty:
        raise SystemExit("No completed matches with statistics.")
    for side in ("home", "away"):
        for stat in ("shots", "shots_on_target", "blocked_shots", "goalkeeper_saves"):
            if f"{side}_{stat}" not in matches:
                matches[f"{side}_{stat}"] = np.nan
    frame, _, _ = build_features(matches, args.ewma_alpha, args.min_history)
    if len(frame) < 100:
        raise SystemExit(f"Only {len(frame)} eligible matches; need at least 100.")
    frame = frame.merge(
        matches[["fixture_id", "league_id"]],
        on="fixture_id", validate="one_to_one",
    ).sort_values(["kickoff", "fixture_id"]).reset_index(drop=True)

    def fit_venue_models(data: pd.DataFrame, cutoff: pd.Timestamp):
        # Share the exact fitting path with backtest.
        return fit_team_bundle(
            data, cutoff, args.regularization, args.recency_decay, args.dixon_coles
        )

    # Keep simultaneous kickoffs together in the chronological holdout.
    split = min(len(frame) - 1, max(1, int(len(frame) * (1 - args.test_fraction))))
    cutoff = frame.iloc[split].kickoff
    training = frame[frame.kickoff < cutoff]
    test = frame[frame.kickoff >= cutoff]
    if training.empty:
        raise SystemExit("No training matches before the holdout cutoff.")
    tuning = None
    if args.auto_tune:
        selected, tuning = select_parameters(training, args.regularization,
                                              args.recency_decay, args.dixon_coles, FIRST_DIVISION)
        args.regularization, args.recency_decay, args.dixon_coles = selected
    evaluation_model, evaluation_rho = fit_venue_models(training, cutoff)
    covered_test = test.loc[venue_model_coverage(evaluation_model, test)]
    print(f"Holdout coverage: {len(covered_test)}/{len(test)} matches")
    print(f"Skipped without required venue formulas: {len(test) - len(covered_test)}")
    if not covered_test.empty:
        home, away = predict_team_rates(evaluation_model, covered_test)
        probabilities = shared_probabilities(home, away, evaluation_rho)
        actual = actual_outcomes(covered_test)
        report_win_calibration(actual, probabilities)
        print(f"1X2 accuracy: {np.mean(probabilities.argmax(axis=1) == actual):.2%}")
        print(f"1X2 log loss: {log_loss(actual, probabilities, labels=[0, 1, 2]):.4f}")
        print(f"Home-goal Poisson deviance: {mean_poisson_deviance(covered_test.home_goals, home):.4f}")
        print(f"Away-goal Poisson deviance: {mean_poisson_deviance(covered_test.away_goals, away):.4f}")
    else:
        print("No holdout matches have both required venue formulas; metrics unavailable.")

    # Fit NEW coefficients and rho on ALL eligible cached matches for future use.
    cutoff = matches.kickoff.max() + pd.Timedelta(nanoseconds=1)
    model, rho = fit_venue_models(frame, cutoff)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_kind": MODEL_KIND,
        "team_models": model,
        "rho": rho,
        "features": TEAM_MODEL_FEATURES,
        "ewma_alpha": args.ewma_alpha,
        "min_history": args.min_history,
        "league_ids": list(LEAGUE_IDS),
        "country": COUNTRY,
        "prediction_league_id": FIRST_DIVISION,
        "training_cutoff": cutoff.isoformat(),
        "regularization": args.regularization,
        "recency_decay": args.recency_decay,
        "weighting": "team_venue_game_order",
        "structure": "shared_venue_slopes_shrunk_team_intercepts",
        "team_intercept_prior_games": TEAM_INTERCEPT_PRIOR_GAMES,
        "stat_weighting": STAT_WEIGHTING,
        "use_dc": args.dixon_coles,
        "parameter_selection": tuning,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, args.model)
    names = dict(zip(matches.home_id, matches.home_name))
    names.update(zip(matches.away_id, matches.away_name))
    count = sum(len(venues) for venues in model.values())
    print(f"Fitted {count} team/venue formulas on {len(frame)} eligible matches; rho={rho:.5f}")
    for team_id, venues in model.items():
        for venue in ("home", "away"):
            label = f"{names.get(team_id, team_id)} [{venue}]"
            if venue in venues:
                print_team_formula(label, venues[venue])
            else:
                print(f"{label}: unavailable (need >= 2 eligible matches and positive goal total)")
    print(f"Saved: {args.model}")


def find_team(query: str, names: dict[int, str]) -> int:
    exact = [i for i, n in names.items() if n.casefold() == query.casefold()]
    if exact:
        return exact[0]
    partial = [i for i, n in names.items() if query.casefold() in n.casefold()]
    if len(partial) != 1:
        choices = ", ".join(sorted(n for n in names.values() if query.casefold() in n.casefold()))
        raise SystemExit(f"Team '{query}' is missing or ambiguous. Matches: {choices or 'none'}")
    return partial[0]

def display_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"

def print_team_comparison(
    home_name: str,
    away_name: str,
    home: TeamState,
    away: TeamState,
) -> None:
    print("\nCurrent model inputs")
    print(f"{'Metric':<30} {home_name:>20} {away_name:>20} {'Home - away':>15}")
    print("-" * 89)
    print(f"{'Home Elo':<30} {home.home_elo:>20.2f} {away.home_elo:>20.2f} {home.home_elo - away.home_elo:>15.2f}")
    print(f"{'Away Elo':<30} {home.away_elo:>20.2f} {away.away_elo:>20.2f} {home.away_elo - away.away_elo:>15.2f}")
    print(
        f"{'Match Elo used':<30} {home.home_elo:>20.2f} {away.away_elo:>20.2f} {home.home_elo - away.away_elo:>15.2f}")
    print(f"{'Matches in team state':<30} {home.games:>20d} {away.games:>20d} {home.games-away.games:>15d}")
    print(
        f"{'Home matches':<30} {home.home_games:>20d} {away.home_games:>20d} {home.home_games - away.home_games:>15d}")
    print(
        f"{'Away matches':<30} {home.away_games:>20d} {away.away_games:>20d} {home.away_games - away.away_games:>15d}")
    print("\nOpponent-Elo-weighted venue rolling statistics used for this fixture")
    for feature in ROLLING_FEATURES:
        home_value = home.home_ewma.get(feature)
        away_value = away.away_ewma.get(feature)
        difference = feature_difference(home_value, away_value)
        print(
            f"{feature:<30} {display_value(home_value):>20} "
            f"{display_value(away_value):>20} {display_value(difference):>15}"
        )

    print("\nOther-venue Elo-weighted rolling statistics (not used for this fixture)")
    print(f"{'Metric':<30} {home_name + ' away':>20} {away_name + ' home':>20}")
    print("-" * 72)
    for feature in ROLLING_FEATURES:
        print(
            f"{feature:<30} {display_value(home.away_ewma.get(feature)):>20} "
            f"{display_value(away.home_ewma.get(feature)):>20}"
        )

def _validate_three_bet_args(args):
    import math
    values = [getattr(args, k, None) for k in ("home2", "away2", "odds", "odds2")]
    enabled = any(v is not None for v in values)
    if not enabled:
        return False
    if any(v is None for v in values):
        raise SystemExit("Supply --home2, --away2, --odds H D A and --odds2 H D A together.")
    for key in ("odds", "odds2"):
        odds = getattr(args, key)
        if len(odds) != 3 or any(not math.isfinite(o) or o <= 1 for o in odds):
            raise SystemExit(f"--{key} requires three finite decimal odds greater than 1.")
    for key in ("stake", "stake_unit"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be finite and positive.")
    if not math.isfinite(args.min_roi) or args.min_roi < 0:
        raise SystemExit("--min-roi must be finite and nonnegative (percent).")
    if args.top_bets < 1:
        raise SystemExit("--top-bets must be at least 1.")
    _stake_units(args.stake, args.stake_unit)
    return True

def _stake_units(total, unit):
    from decimal import Decimal
    count = Decimal(str(total)) / Decimal(str(unit))
    if count != count.to_integral_value() or count < 3:
        raise SystemExit("--stake must be an exact multiple of --stake-unit, with at least 3 units.")
    return int(count)

def _normalized_bet_probabilities(values):
    import math
    p = tuple(float(v) for v in values)
    if (len(p) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in p)
            or not math.isclose(sum(p), 1.0, abs_tol=0.001, rel_tol=0.0)):
        raise ValueError("Expected finite H/D/A probabilities in [0, 1] summing to 1.")
    return tuple(v / sum(p) for v in p)

def find_three_bet_setups(probabilities1, probabilities2, odds1, odds2,
                          total_stake=100000.0, stake_unit=1.0, min_roi=0.0):
    """Search all 36 portfolios; enumerate the nine joint match outcomes.

    Equalize each ticket's gross payout before rounding; allocate whole stake
    units by largest remainder. Rank by total expected ROI after rounding.
    Odds products and joint probabilities assume independent matches.
    """
    import math
    from itertools import product
    if (not math.isfinite(total_stake) or total_stake <= 0
            or not math.isfinite(stake_unit) or stake_unit <= 0
            or not math.isfinite(min_roi) or min_roi < 0):
        raise ValueError("Stake/unit must be positive and ROI threshold nonnegative; all finite.")
    p1 = _normalized_bet_probabilities(probabilities1)
    p2 = _normalized_bet_probabilities(probabilities2)
    o1, o2 = tuple(map(float, odds1)), tuple(map(float, odds2))
    if any(len(o) != 3 or any(not math.isfinite(v) or v <= 1 for v in o)
           for o in (o1, o2)):
        raise ValueError("Bookmaker odds must be three finite decimal values greater than 1.")
    units = _stake_units(total_stake, stake_unit)
    tolerance = max(1e-9, total_stake * 1e-12)
    setups = []
    for leg1, leg2, single1, single2 in product(range(3), repeat=4):
        if leg1 == single1 or leg2 == single2:
            continue
        odds = (o1[leg1] * o2[leg2], o1[single1], o2[single2])
        if not all(math.isfinite(o) for o in odds):
            continue
        reciprocal_sum = sum(1 / o for o in odds)
        if reciprocal_sum >= 1:
            continue  # No equal payout can exceed the total stake.
        raw_units = [units / o / reciprocal_sum for o in odds]
        allocations = [math.floor(v) for v in raw_units]
        remaining = units - sum(allocations)
        order = sorted(range(3), key=lambda i: raw_units[i] - allocations[i], reverse=True)
        for i in order[:remaining]:
            allocations[i] += 1
        stakes = tuple(n * stake_unit for n in allocations)
        returns = tuple(stake * odd for stake, odd in zip(stakes, odds))
        if any(s <= 0 for s in stakes) or min(returns) <= total_stake + tolerance:
            continue
        ticket_probabilities = (p1[leg1] * p2[leg2], p1[single1], p2[single2])
        scenarios = []
        for h, a in product(range(3), repeat=2):
            wins = (h == leg1 and a == leg2, h == single1, a == single2)
            net = sum(r for r, won in zip(returns, wins) if won) - total_stake
            scenarios.append({"outcomes": (h, a), "probability": p1[h] * p2[a],
                              "winning_tickets": sum(wins), "net": net})
        expected_net = sum(s["probability"] * s["net"] for s in scenarios)
        roi = expected_net / total_stake
        if expected_net <= tolerance or roi + 1e-12 < min_roi / 100:
            continue
        setups.append({
            "legs": (leg1, leg2), "singles": (single1, single2),
            "odds": odds, "stakes": stakes, "returns": returns,
            "ticket_probabilities": ticket_probabilities, "scenarios": scenarios,
            "expected_net": expected_net, "roi": roi,
            "profit_probability": sum(s["probability"] for s in scenarios if s["net"] > tolerance),
        })
    return sorted(setups, key=lambda s: (s["roi"], s["profit_probability"]), reverse=True)


def print_three_bet_setups(first, second, args):
    def labels(match):
        return (f"{match['home_name']} wins", "Draw", f"{match['away_name']} wins")

    setups = find_three_bet_setups(
        first["probabilities"], second["probabilities"], args.odds, args.odds2,
        args.stake, args.stake_unit, args.min_roi,
    )
    print("\nThree-ticket search: one combined bet + two alternative singles")
    print("Match 1: " + first["home_name"] + " vs " + first["away_name"])
    print("Match 2: " + second["home_name"] + " vs " + second["away_name"])
    print("All 36 H/D/A configurations considered; ranked by overall model-estimated ROI.")
    print("Independent matches assumed. Combined odds are products: verify the offered price.")
    print("Positive overall EV does not mean each individual ticket has positive EV.")
    print(f"Total stake per setup: {args.stake:,.2f}; stake increment: {args.stake_unit:g}")
    if not setups:
        print("No qualifying positive-EV setup found at these odds/probabilities and stake rounding.")
        return
    print(f"Qualifying setups: {len(setups)}. Each displayed setup is an ALTERNATIVE, not an extra bet.")
    for n, setup in enumerate(setups[:args.top_bets], 1):
        leg1, leg2 = setup["legs"]
        single1, single2 = setup["singles"]
        l1, l2 = labels(first), labels(second)
        descriptions = (
            f"Combined: match 1 {l1[leg1]} + match 2 {l2[leg2]}",
            f"Single: match 1 {l1[single1]}",
            f"Single: match 2 {l2[single2]}",
        )
        print(f"\nSetup {n}")
        for desc, odd, stake, ret, probability in zip(
            descriptions, setup["odds"], setup["stakes"], setup["returns"],
            setup["ticket_probabilities"],
        ):
            print(desc)
            print(f"  Odds {odd:.6g} | Probability {probability:.2%} | Stake {stake:,.2f} "
                  f"| Gross payout {ret:,.2f} | Individual ROI {probability * odd - 1:+.2%}")
        print(f"Overall expected ROI: {setup['roi']:+.2%}")
        print(f"Expected net profit: {setup['expected_net']:+,.2f}")
        print(f"Chance of net profit: {setup['profit_probability']:.2%}")
        for count, title in ((0, "All tickets lose"), (1, "Exactly one ticket wins"),
                             (2, "Both singles win")):
            scenarios = [s for s in setup["scenarios"] if s["winning_tickets"] == count]
            probability = sum(s["probability"] for s in scenarios)
            low, high = min(s["net"] for s in scenarios), max(s["net"] for s in scenarios)
            print(f"  {title}: {probability:.2%}; net {low:+,.2f} to {high:+,.2f}")
        print("Uncovered result pairs (match 1 / match 2):")
        for scenario in setup["scenarios"]:
            if scenario["winning_tickets"] == 0:
                a, b = scenario["outcomes"]
                print(f"  {l1[a]} / {l2[b]}: {scenario['probability']:.2%}")
    print("Model-estimated EV only; calibration error can remove the apparent edge.")

def add_three_bet_arguments(q):
    q.add_argument("--home2", help="Second match home team; enables three-ticket search")
    q.add_argument("--away2", help="Second match away team")
    q.add_argument("--odds", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"),
                   help="First match bookmaker decimal odds, in H D A order")
    q.add_argument("--odds2", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"),
                   help="Second match bookmaker decimal odds, in H D A order")
    q.add_argument("--stake", type=float, default=100000.0,
                   help="Total stake across three tickets (default: 100000)")
    q.add_argument("--stake-unit", type=float, default=1.0,
                   help="Stake increment, e.g. 1 COP or 0.01 dollars (default: 1)")
    q.add_argument("--min-roi", type=float, default=0.0,
                   help="Minimum overall expected ROI in percent; EV must always be positive")
    q.add_argument("--top-bets", type=int, default=1,
                   help="Number of alternative three-ticket setups to print (default: 1)")

def predict(args: argparse.Namespace) -> None:
    # Use the same shared slopes and team/venue routing as the holdout.

    betting = _validate_three_bet_args(args)
    bundle = joblib.load(args.model)
    if bundle.get("model_kind") != MODEL_KIND:
        raise SystemExit(
            "Model structure changed. Retrain with football_poisson.py train for opponent-Elo-weighted statistics and shared venue slopes."
        )
    if bundle.get("features") != TEAM_MODEL_FEATURES:
        raise SystemExit("Team formula features changed. Retrain the model.")

    if bundle.get("country") != COUNTRY or bundle.get("league_ids") != list(LEAGUE_IDS):
        raise SystemExit("Saved model belongs to a different country/division configuration. Retrain.")

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    if as_of < pd.Timestamp(bundle["training_cutoff"]):
        raise SystemExit("Prediction date precedes the model training cutoff.")
    if not args.db.is_file():
        raise SystemExit(f"Database not found: {args.db}")
    con = connect(args.db)
    try:
        matches = load_matches(con)
    finally:
        con.close()
    matches["kickoff"] = pd.to_datetime(matches.kickoff, utc=True)
    matches = matches[
        matches.league_id.isin(bundle["league_ids"])
        & (matches.status == "FT")
        & (matches.kickoff < as_of)
    ].copy().sort_values(["kickoff", "fixture_id"])
    if matches.empty:
        raise SystemExit("No completed matches with statistics before the prediction date.")
    for side in ("home", "away"):
        for stat in ("shots", "shots_on_target", "blocked_shots", "goalkeeper_saves"):
            if f"{side}_{stat}" not in matches:
                matches[f"{side}_{stat}"] = np.nan
    _, states, names = build_features(matches, bundle["ewma_alpha"], bundle["min_history"])
    league_id = args.league_id if args.league_id is not None else bundle["prediction_league_id"]
    if league_id not in bundle["league_ids"]:
        raise SystemExit("League is not supported by this saved model.")

    def predict_match(home_query, away_query):
        hid, aid = find_team(home_query, names), find_team(away_query, names)
        if hid == aid:
            raise SystemExit("Choose two different teams.")
        home, away = states[hid], states[aid]
        if (home.calibrating_home_elo or away.calibrating_away_elo
                or home.home_games < bundle["min_history"]
                or away.away_games < bundle["min_history"]):
            raise SystemExit(f"{names[hid]} vs {names[aid]}: insufficient venue history or unfinished Elo calibration.")
        row = {"home_id": hid, "away_id": aid, "league_id": league_id,
               "elo_diff": (home.home_elo - away.away_elo) / 400.0}
        row.update(match_inputs(home, away))
        for team_id, venue in ((hid, "home"), (aid, "away")):
            if venue not in bundle["team_models"].get(team_id, {}):
                raise SystemExit(f"No {venue} formula for {names[team_id]}; more eligible training data is needed.")
        lh, la = predict_team_rates(bundle["team_models"], pd.DataFrame([row]))
        probabilities = shared_probabilities(lh, la, bundle["rho"])[0]
        return {"home_id": hid, "away_id": aid, "home_name": names[hid], "away_name": names[aid],
                "home": home, "away": away, "lh": float(lh[0]), "la": float(la[0]),
                "probabilities": probabilities}

    first = predict_match(args.home, args.away)
    predictions = [first]
    if betting:
        second = predict_match(args.home2, args.away2)
        ids = [first["home_id"], first["away_id"], second["home_id"], second["away_id"]]
        if len(set(ids)) != 4:
            raise SystemExit("Three-ticket search requires two different matches with four distinct teams.")
        predictions.append(second)
    for result in predictions:
        ph, pd_, pa = result["probabilities"]
        predicted = int(np.argmax(result["probabilities"]))
        p_over = float(poisson.sf(2, result["lh"] + result["la"]))
        print(f"{result['home_name']} vs {result['away_name']}")
        print_team_comparison(result["home_name"], result["away_name"], result["home"], result["away"])
        print("\nPrediction — shared venue slopes with shrunk team scoring adjustments")
        print(f"Training cutoff: {bundle['training_cutoff']}")
        print(f"Form as of: {as_of.isoformat()}")
        print(f"Latest cached match used: {matches.kickoff.max().isoformat()}")
        print(f"Dixon–Coles rho: {bundle['rho']:.5f}")
        print(f"Expected goals: {result['lh']:.3f} - {result['la']:.3f}")
        for label, probability in (("Home", ph), ("Draw", pd_), ("Away", pa)):
            fair = 1 / probability if probability > 0 else float("inf")
            print(f"{label}: {probability:.2%} (fair odds {fair:.2f})")
        print(f"Predicted result: {['H', 'D', 'A'][predicted]}")
        print(f"Over 2.5 goals: {p_over:.2%} (>= 70%: {p_over >= 0.70})")
    if betting:
        print_three_bet_setups(first, second, args)

def backtest(args: argparse.Namespace) -> None:
    print(f"Game-order decay: {args.recency_decay:g} per previous team/venue game")
    if args.train_through >= args.test_season:
        raise SystemExit("--train-through must be earlier than --test-season.")

    if not args.db.is_file():
        raise SystemExit(f"Database not found: {args.db}")

    con = connect(args.db)
    try:
        # Keep all completed test fixtures so coverage includes matches
        # omitted by load_matches() because statistics are missing.
        test_fixtures = pd.read_sql_query(
            """
            SELECT fixture_id, kickoff, home_name, away_name,
                   home_goals, away_goals
            FROM fixtures
            WHERE season = ?
              AND league_id = ?
              AND status = 'FT'
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
            ORDER BY kickoff, fixture_id
            """,
            con,
            params=(args.test_season, args.league_id),
        )

        matches = load_matches(con)
    finally:
        con.close()

    if test_fixtures.empty:
        raise SystemExit("No completed fixtures found for the test season/league.")

    # Rebuild history from the database; never load the latest trained bundle.
    matches = matches[
        (matches.season <= args.test_season)
        & matches.league_id.isin(LEAGUE_IDS)
        & (matches.status == "FT")
    ].copy()

    matches["kickoff"] = pd.to_datetime(matches["kickoff"], utc=True)
    matches = matches.sort_values(
        ["kickoff", "fixture_id"]
    ).reset_index(drop=True)

    if matches.empty:
        raise SystemExit("No matches with statistics are available.")

    for side in ("home", "away"):
        for stat in ("shots", "shots_on_target", "blocked_shots", "goalkeeper_saves"):
            if f"{side}_{stat}" not in matches:
                matches[f"{side}_{stat}"] = np.nan

    # Existing build_features() records each match's inputs BEFORE
    # updating Elo/rolling statistics with that match's result.
    frame, _, _ = build_features(
        matches,
        args.ewma_alpha,
        args.min_history,
    )

    # build_features() does not currently retain season or league_id.
    frame = frame.merge(
        matches[["fixture_id", "season", "league_id"]],
        on="fixture_id",
        how="left",
        validate="one_to_one",
    )

    test_start = pd.to_datetime(
        test_fixtures.kickoff, utc=True
    ).min()

    training = frame[
        (frame.season <= args.train_through)
        & (frame.kickoff < test_start)
    ].copy()

    test = frame[
        (frame.season == args.test_season)
        & (frame.league_id == args.league_id)
    ].sort_values(["kickoff", "fixture_id"])

    if training.empty:
        raise SystemExit("No eligible training matches before the cutoff.")

    # Fit imputers, scalers and coefficients using training data ONLY.
    # These models are never refitted on the test season.
    if args.auto_tune:
        selected, _ = select_parameters(training, args.regularization,
                                        args.recency_decay, args.dixon_coles, args.league_id)
        args.regularization, args.recency_decay, args.dixon_coles = selected
    team_models, rho = fit_team_bundle(
        training, test_start, args.regularization, args.recency_decay, args.dixon_coles
    )

    if not team_models:
        raise SystemExit("No team models could be trained.")

    predictions = []

    for index, row in test.iterrows():
        home_id = int(row.home_id)
        away_id = int(row.away_id)

        if not has_venue_models(team_models, home_id, away_id):
            continue

        fixture = test.loc[[index]]

        home_rates, away_rates = predict_team_rates(team_models, fixture)
        lambda_home, lambda_away = float(home_rates[0]), float(away_rates[0])
        p_home, p_draw, p_away = outcome_probabilities(lambda_home, lambda_away, rho=rho)

        predictions.append({
            "fixture_id": row.fixture_id,
            "expected_home_goals": lambda_home,
            "expected_away_goals": lambda_away,
            "p_over_2_5": float(poisson.sf(2, lambda_home + lambda_away)),
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "predicted_result": ["H", "D", "A"][
                int(np.argmax([p_home, p_draw, p_away]))
            ],
        })

    prediction_columns = [
        "fixture_id",
        "expected_home_goals",
        "expected_away_goals",
        "p_over_2_5",
        "p_home",
        "p_draw",
        "p_away",
        "predicted_result",
    ]

    results = test_fixtures.merge(
        pd.DataFrame(predictions, columns=prediction_columns),
        on="fixture_id",
        how="left",
        validate="one_to_one",
    )

    results["actual_result"] = np.where(
        results.home_goals > results.away_goals,
        "H",
        np.where(
            results.home_goals == results.away_goals,
            "D",
            "A",
        ),
    )

    loaded_ids = set(matches.fixture_id)
    eligible_ids = set(test.fixture_id)

    def skip_reason(fixture_id: int) -> str:
        if fixture_id not in loaded_ids:
            return "Missing match statistics"
        if fixture_id not in eligible_ids:
            return "Insufficient venue history or Elo calibration"
        return "No pre-test home formula for host or away formula for visitor"

    predicted_mask = results.predicted_result.notna()

    confidence_mask = (
            (results["p_home"] >= 0.60)
            | (results["p_away"] >= 0.60)
    )

    scored_mask = predicted_mask & confidence_mask
    results["included_in_evaluation"] = scored_mask

    results["skip_reason"] = [
        (
            ""
            if included
            else "Neither team has at least 60% win probability"
            if predicted
            else skip_reason(fixture_id)
        )
        for fixture_id, predicted, included in zip(
            results.fixture_id,
            predicted_mask,
            scored_mask,
        )
    ]

    results["correct"] = pd.Series(
        pd.NA, index=results.index, dtype="boolean"
    )
    results.loc[scored_mask, "correct"] = (
        results.loc[scored_mask, "predicted_result"]
        == results.loc[scored_mask, "actual_result"]
    )

    results["actual_over_2_5"] = (results.home_goals + results.away_goals) > 2.5
    over_mask = predicted_mask & (results["p_over_2_5"] >= 0.60)
    results["included_in_over_2_5_evaluation"] = over_mask
    results["over_2_5_correct"] = pd.Series(pd.NA, index=results.index, dtype="boolean")
    results.loc[over_mask, "over_2_5_correct"] = results.loc[over_mask, "actual_over_2_5"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    # Report the full predicted population as well as the requested confidence subset.
    all_predicted = results.loc[predicted_mask]
    if not all_predicted.empty:
        all_y = actual_outcomes(all_predicted)
        all_p = all_predicted[["p_home", "p_draw", "p_away"]].to_numpy()
        report_win_calibration(all_y, all_p)
        print(f"\nAll predicted matches: {len(all_predicted)}")
        print(f"All-match 1X2 accuracy: {np.mean(all_p.argmax(axis=1) == all_y):.2%}")
        print(f"All-match log loss: {log_loss(all_y, all_p, labels=[0, 1, 2]):.4f}")
    scored = results.loc[scored_mask].copy()

    print(f"\nTraining through: {args.train_through}")
    print(f"Training matches: {len(training)}")
    print(f"Team venue formulas: {sum(len(v) for v in team_models.values())}")
    print(f"Dixon–Coles rho: {rho:.5f}")
    print(f"Test season: {args.test_season}")
    print(f"Test league: {args.league_id}")
    print(f"Predicted matches: {int(predicted_mask.sum())}/{len(results)}")
    print(f"Evaluated matches with win probability >= 60%: {len(scored)}")
    print(f"Saved match-by-match results: {args.output}")

    skipped = results.loc[~scored_mask, "skip_reason"]
    for reason, count in skipped.value_counts().items():
        print(f"Skipped — {reason}: {count}")

    over_scored = results.loc[over_mask]
    print("\nOver 2.5 goals backtest — probability >= 60%")
    print(f"Qualified matches: {len(over_scored)}/{int(predicted_mask.sum())} predicted matches")
    if over_scored.empty:
        print("No matches meet the over-2.5 probability threshold.")
    else:
        actual_over = over_scored["actual_over_2_5"].to_numpy(dtype=int)
        probs_over = over_scored["p_over_2_5"].to_numpy(dtype=float)
        successes = int(actual_over.sum())
        print(f"Correct predictions (3+ goals): {successes}")
        print(f"Incorrect predictions (0–2 goals): {len(actual_over) - successes}")
        print(f"Over 2.5 hit rate: {actual_over.mean():.2%}")
        print(f"Average predicted probability: {probs_over.mean():.2%}")
        print(f"Expected successful predictions: {probs_over.sum():.1f}")
        print(f"Binary Brier score: {np.mean((probs_over - actual_over) ** 2):.4f}")
        loss = log_loss(actual_over, np.column_stack([1 - probs_over, probs_over]), labels=[0, 1])
        print(f"Binary log loss: {loss:.4f}")

        match_table = pd.DataFrame({
            "Date": pd.to_datetime(over_scored["kickoff"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC"),
            "Home": over_scored["home_name"],
            "Away": over_scored["away_name"],
            "P(Over 2.5)": over_scored["p_over_2_5"].map(lambda p: f"{p:.2%}"),
            "Actual score": (
                over_scored["home_goals"].astype(int).astype(str)
                + " - " + over_scored["away_goals"].astype(int).astype(str)
            ),
            "Total goals": (over_scored["home_goals"] + over_scored["away_goals"]).astype(int),
            "Correct": np.where(over_scored["actual_over_2_5"], "Yes", "No"),
        })
        print("\nAll predicted over-2.5 matches (probability >= 60%)")
        print(match_table.to_string(index=False, max_rows=None, max_cols=None))

    if scored.empty:
        print("No matches could be evaluated.")
        return

    probabilities = scored[["p_home", "p_draw", "p_away"]].to_numpy(
        dtype=float
    )
    actual = scored.actual_result.map(
        {"H": 0, "D": 1, "A": 2}
    ).to_numpy(dtype=int)

    accuracy = float(scored.correct.mean())
    brier = np.mean(np.sum(
        (probabilities - np.eye(3)[actual]) ** 2,
        axis=1,
    ))

    print(f"\n1X2 accuracy: {accuracy:.2%}")
    print(
        f"1X2 log loss: "
        f"{log_loss(actual, probabilities, labels=[0, 1, 2]):.4f}"
    )
    print(f"Multiclass Brier score (0–2): {brier:.4f}")
    print(
        f"Home-goal Poisson deviance: "
        f"{mean_poisson_deviance(scored.home_goals, scored.expected_home_goals):.4f}"
    )
    print(
        f"Away-goal Poisson deviance: "
        f"{mean_poisson_deviance(scored.away_goals, scored.expected_away_goals):.4f}"
    )

    print("\nConfusion matrix: rows=actual, columns=predicted")
    print(
        pd.crosstab(
            scored.actual_result,
            scored.predicted_result,
        ).reindex(
            index=["H", "D", "A"],
            columns=["H", "D", "A"],
            fill_value=0,
        )
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--country", default="Colombia", help="API-Football country, e.g. England or Colombia")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="Cache fixtures, then missing match statistics")
    d.add_argument("--from-season", type=int, default=2019)
    d.add_argument("--to-season", type=int, default=2025, help="Calendar season year, e.g. 2025")
    d.add_argument("--max-stat-requests", type=int, default=10000)
    d.add_argument("--pause", type=float, default=.25)
    d.add_argument("--division-ids", type=int, nargs="+",
                   help="Override: first division ID, then optional second division ID")
    d.set_defaults(func=download)
    t = sub.add_parser("train", help="Create leakage-free rolling features and fit Poisson models")
    t.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    t.add_argument("--ewma-alpha", type=float, default=.25)
    t.add_argument("--min-history", type=int, default=5)
    t.add_argument("--regularization", type=float, default=0.01)
    t.add_argument("--recency-decay", type=recency_decay_value, default=DEFAULT_RECENCY_DECAY,
                   help="Weight multiplier for each previous team/venue game (default: 0.95)")
    t.add_argument("--dixon-coles", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--test-fraction", type=float, default=.20)
    t.add_argument("--auto-tune", action=argparse.BooleanOptionalAction, default=False,
                   help="Select regularization, decay and draw correction on internal chronological folds")
    t.set_defaults(func=train)
    q = sub.add_parser(
        "predict",
        help="Predict matches and optionally find positive-EV three-ticket setups",
    )
    q.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    q.add_argument("--league-id", type=int)
    q.add_argument(
        "--as-of",
        help="Prediction cutoff in ISO format; default: now. Naive dates use UTC.",
    )
    q.add_argument("--home", required=True)
    q.add_argument("--away", required=True)

    # Registers second-match, bookmaker odds, stake and EV options.
    add_three_bet_arguments(q)

    q.set_defaults(func=predict)
    b = sub.add_parser("backtest", help="Train through one year and evaluate a later year")
    b.add_argument("--train-through", type=int, default=2024)
    b.add_argument("--test-season", type=int, default=2025)
    b.add_argument("--league-id", type=int, default=None)
    b.add_argument("--ewma-alpha", type=float, default=0.25)
    b.add_argument("--min-history", type=int, default=5)
    b.add_argument("--regularization", type=float, default=0.01)
    b.add_argument("--recency-decay", type=recency_decay_value, default=DEFAULT_RECENCY_DECAY,
                   help="Weight multiplier for each previous team/venue game (default: 0.95)")
    b.add_argument("--dixon-coles", action=argparse.BooleanOptionalAction, default=True)
    b.add_argument("--output", type=Path, default=None)
    b.add_argument("--auto-tune", action=argparse.BooleanOptionalAction, default=False)
    b.set_defaults(func=backtest)
    # Accept country/db both before and after the subcommand.
    for command in (d, t, q, b):
        command.add_argument("--country", default=argparse.SUPPRESS)
        command.add_argument("--db", type=Path, default=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    configure_command(args)
    args.func(args)
