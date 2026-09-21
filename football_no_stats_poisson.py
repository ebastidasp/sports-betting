"""Results-only Elo + shared Poisson goal model. No match-statistics requests.

Install: python -m pip install requests numpy scipy python-dotenv
Set API_FOOTBALL_KEY in .env beside this script (needed only for download).

python football_elo_poisson.py download --country Iceland --from-season 2019 --to-season 2026
python football_elo_poisson.py train --country Iceland
python football_elo_poisson.py ratings --country Iceland
python football_elo_poisson.py predict --country Iceland --home "KR Reykjavik" --away "Vikingur Reykjavik"
python football_elo_poisson.py backtest --country Iceland --train-through 2024 --test-season 2025

Defaults reuse <country>.sqlite3 fixture caches from football_poisson.py.
Only FT scores are used: extra-time/penalty results are excluded.
One overall Elo per team, initialized at 1500; K=20 and sqrt(goal margin).
Elo expected score is not the win probability (draws count as half).
A home Elo advantage of 100 is used for Elo updates, configurable with
--elo-home-advantage. Poisson learns its own home coefficient.
Shared league formula: log(lambda) = intercept + elo_coef * diff/400 + home_coef * is_home.
The Elo coefficient is constrained nonnegative. These are score-based expected
goals, not shot-based xG. No per-team regression or match statistics are needed.
Backtest freezes coefficients fitted through --train-through and updates Elo
chronologically after subsequent results. Matches with identical kickoffs use
ratings from before that kickoff. Parameters must be chosen without test results.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from dotenv import load_dotenv
from scipy.optimize import minimize
from scipy.stats import poisson, skellam

API_URL = "https://v3.football.api-sports.io"
FIRST_DIVISION = SECOND_DIVISION_ID = None
LEAGUE_IDS = ()
COUNTRY = ""
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

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
    return result

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
                f"season {season}: "
                f"cached {len(items)} fixtures"
            )

def load_results(con):
    rows = [dict(row) for row in con.execute(
        "SELECT * FROM fixtures WHERE status='FT' AND home_goals IS NOT NULL "
        "AND away_goals IS NOT NULL ORDER BY kickoff, fixture_id"
    )]
    for row in rows:
        row['_time'] = datetime.fromisoformat(row['kickoff'].replace('Z', '+00:00'))
        if row['_time'].tzinfo is None:
            raise SystemExit('Fixture kickoff must include a timezone.')
        if min(row['home_goals'], row['away_goals']) < 0:
            raise SystemExit('Invalid negative goal count in fixture cache.')
    return sorted(rows, key=lambda row: (row['_time'], row['fixture_id']))


def replay(rows, k=20.0, home_advantage=100.0):
    """Return pre-match examples and ratings; simultaneous results update in a batch."""
    ratings, games, names, latest_league = {}, defaultdict(int), {}, {}
    examples = []
    for _, batch in groupby(rows, key=lambda row: row['_time']):
        changes = defaultdict(float)
        for row in batch:
            h, a = row['home_id'], row['away_id']
            ratings.setdefault(h, 1500.0)
            ratings.setdefault(a, 1500.0)
            names.update({h: row['home_name'], a: row['away_name']})
            latest_league.update({h: row['league_id'], a: row['league_id']})
            diff = (ratings[h] - ratings[a]) / 400.0
            examples.append({**row, 'elo_diff': diff,
                             'home_elo': ratings[h], 'away_elo': ratings[a]})
            expected = 1.0 / (1.0 + 10.0 ** np.clip(-(diff + home_advantage / 400), -100, 100))
            margin = row['home_goals'] - row['away_goals']
            result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            change = k * math.sqrt(abs(margin) or 1) * (result - expected)
            changes[h] += change
            changes[a] -= change
            games[h] += 1
            games[a] += 1
        for team, change in changes.items():
            ratings[team] += change
    teams = {str(team): {'name': names[team], 'elo': elo,
                        'games': games[team], 'league_id': latest_league[team]}
             for team, elo in ratings.items()}
    return examples, teams


def fit_formula(examples, league_id, alpha):
    selected = [row for row in examples if row['league_id'] == league_id]
    if len(selected) < 20:
        raise SystemExit('At least 20 completed training matches in the selected league are required.')
    x, y = [], []
    for row in selected:
        x.extend([[1, row['elo_diff'], 1], [1, -row['elo_diff'], 0]])
        y.extend([row['home_goals'], row['away_goals']])
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.sum() == 0:
        raise SystemExit('Cannot fit a goal model with no scored goals.')

    def objective(beta):
        eta = x @ beta
        mu = np.exp(eta)
        penalty = alpha * np.dot(beta[1:], beta[1:]) / 2
        loss = np.mean(mu - y * eta) + penalty
        grad = x.T @ (mu - y) / len(y)
        grad[1:] += alpha * beta[1:]
        return loss, grad

    fit = minimize(objective, [math.log(y.mean()), 0.3, 0.2], jac=True,
                   method='L-BFGS-B', bounds=[(-10, 5), (0, 5), (-5, 5)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise SystemExit(f'Poisson optimization failed: {fit.message}')
    return fit.x.tolist(), len(selected)


def expected_goals(beta, diff):
    b0, b1, b2 = beta
    return math.exp(b0 + b1 * diff + b2), math.exp(b0 - b1 * diff)


def probabilities(home, away):
    # Skellam avoids truncating a score matrix at an arbitrary goal limit.
    draw = float(skellam.pmf(0, home, away))
    home_win = float(skellam.sf(0, home, away))
    away_win = float(skellam.cdf(-1, home, away))
    p = np.asarray([home_win, draw, away_win])
    if not np.isfinite(p).all() or p.sum() <= 0:
        raise SystemExit('Non-finite outcome probabilities; inspect model inputs.')
    return p / p.sum()


def show_formula(beta):
    b0, b1, b2 = beta
    print(f'Home xG = exp({b0:.6f} + {b1:.6f} * (home Elo - away Elo)/400 + {b2:.6f})')
    print(f'Away xG = exp({b0:.6f} - {b1:.6f} * (home Elo - away Elo)/400)')


def match_team(query, teams):
    if query in teams:
        return query
    key = country_key(query)
    exact = [tid for tid, team in teams.items() if country_key(team['name']) == key]
    matches = exact or [tid for tid, team in teams.items() if key in country_key(team['name'])]
    if len(matches) != 1:
        options = ', '.join(f"{tid}: {teams[tid]['name']}" for tid in matches)
        raise SystemExit(f'Team {query!r} not found or ambiguous. Use ratings to find its ID. {options}')
    return matches[0]


def download(args):
    key = os.environ.get('API_FOOTBALL_KEY')
    if not key:
        raise SystemExit('Set API_FOOTBALL_KEY in .env beside this script.')
    if args.from_season > args.to_season:
        raise SystemExit('--from-season must not exceed --to-season.')
    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = connect(args.db)
    try:
        api = ApiFootball(key, args.pause)
        leagues = resolve_divisions(api, args, con)
        save_fixtures(con, api, range(args.from_season, args.to_season + 1), leagues)
        print('Results cached. No match-statistics requests were made.')
    finally:
        con.close()


def read_data(args):
    if not args.db.is_file():
        raise SystemExit(f'Database not found: {args.db}. Run download first.')
    con = connect(args.db)
    try:
        config = read_config(con)
        if not config or config['country'] != country_key(args.country):
            raise SystemExit('Missing or mismatched country configuration; run download or choose another --db.')
        league = args.league_id if args.league_id is not None else config['league_ids'][0]
        if league not in config['league_ids']:
            raise SystemExit('--league-id must belong to the configured divisions.')
        rows = load_results(con)
        rows = [r for r in rows if r['league_id'] in config['league_ids']]
        if not rows:
            raise SystemExit('No completed FT results available.')
        return rows, league
    finally:
        con.close()


def train(args):
    rows, league = read_data(args)
    if args.through_season is not None:
        rows = [row for row in rows if row['season'] <= args.through_season]
    examples, teams = replay(rows, args.k, args.elo_home_advantage)
    beta, count = fit_formula(examples, league, args.alpha)
    bundle = {'kind': 'results_elo_poisson_v1', 'country': args.country,
              'league_id': league, 'coefficients': beta, 'teams': teams,
              'k': args.k, 'elo_home_advantage': args.elo_home_advantage,
              'alpha': args.alpha, 'training_matches': count,
              'as_of': rows[-1]['kickoff']}
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Trained on {count} league matches; ratings through {bundle["as_of"]}.')
    show_formula(beta)
    print(f'Saved {args.model}')


def predict(args):
    if not args.model.is_file():
        raise SystemExit(f'Model not found: {args.model}. Run train first.')
    model = json.loads(args.model.read_text(encoding='utf-8'))
    if model.get('kind') != 'results_elo_poisson_v1' or model['country'] != args.country:
        raise SystemExit('Wrong model format or country.')
    if args.league_id is not None and args.league_id != model['league_id']:
        raise SystemExit('Model was trained for a different league.')
    teams = model['teams']
    hid, aid = match_team(args.home, teams), match_team(args.away, teams)
    if hid == aid:
        raise SystemExit('Choose two different teams.')
    h, a = teams[hid], teams[aid]
    if h['league_id'] != model['league_id'] or a['league_id'] != model['league_id']:
        raise SystemExit('Both teams must belong to the model league in their latest cached results.')
    hg, ag = expected_goals(model['coefficients'], (h['elo'] - a['elo']) / 400)
    p = probabilities(hg, ag)
    print(f"{h['name']} ({h['elo']:.1f}) vs {a['name']} ({a['elo']:.1f})")
    print(f"Ratings/model as of {model['as_of']}; retrain after downloading new results.")
    print(f'Expected goals: {hg:.3f} - {ag:.3f}')
    for label, probability in zip(('Home', 'Draw', 'Away'), p):
        odds = 1 / probability if probability > 0 else math.inf
        print(f'{label}: {probability:.2%} (fair odds {odds:.2f})')
    print(f'Over 2.5 goals: {poisson.sf(2, hg + ag):.2%}')


def ratings(args):
    rows, league = read_data(args)
    _, teams = replay(rows, args.k, args.elo_home_advantage)
    print(f'Ratings through {rows[-1]["kickoff"]}')
    print(f'{"ID":>7}  {"Team":35} {"Elo":>8} {"Games":>7}')
    for tid, team in sorted(teams.items(), key=lambda pair: pair[1]['elo'], reverse=True):
        if args.league_id is None or team['league_id'] == league:
            print(f"{tid:>7}  {team['name']:35} {team['elo']:8.1f} {team['games']:7}")


def backtest(args):
    if args.train_through >= args.test_season:
        raise SystemExit('--train-through must be earlier than --test-season.')
    rows, league = read_data(args)
    tests = [r for r in rows if r['season'] == args.test_season and r['league_id'] == league]
    if not tests:
        raise SystemExit('No completed test-season fixtures for this league.')
    first_test = tests[0]['_time']
    # The timestamp guard prevents season-label overlaps from leaking results.
    train_rows = [r for r in rows if r['season'] <= args.train_through and r['_time'] < first_test]
    training, _ = replay(train_rows, args.k, args.elo_home_advantage)
    beta, count = fit_formula(training, league, args.alpha)
    # Resume from exactly the training history, incorporating subsequent results.
    cutoff = train_rows[-1]['_time']
    timeline = train_rows + [r for r in rows if cutoff < r['_time'] <= tests[-1]['_time']]
    examples, _ = replay(timeline, args.k, args.elo_home_advantage)
    output, losses, briers, correct = [], [], [], 0
    for r in examples:
        if r['league_id'] != league or r['season'] != args.test_season:
            continue
        hg, ag = expected_goals(beta, r['elo_diff'])
        p = probabilities(hg, ag)
        actual = 0 if r['home_goals'] > r['away_goals'] else 2 if r['home_goals'] < r['away_goals'] else 1
        predicted = int(np.argmax(p))
        correct += predicted == actual
        losses.append(-math.log(max(p[actual], 1e-15)))
        briers.append(float(np.sum((p - np.eye(3)[actual]) ** 2)))
        output.append({'fixture_id': r['fixture_id'], 'kickoff': r['kickoff'],
                       'home': r['home_name'], 'away': r['away_name'],
                       'home_goals': r['home_goals'], 'away_goals': r['away_goals'],
                       'home_elo': r['home_elo'], 'away_elo': r['away_elo'],
                       'home_xg': hg, 'away_xg': ag, 'p_home': p[0], 'p_draw': p[1],
                       'p_away': p[2], 'prediction': 'HDA'[predicted], 'actual': 'HDA'[actual],
                       'p_over_2_5': float(poisson.sf(2, hg + ag))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f'Training matches: {count}; evaluated: {len(output)}/{len(tests)}')
    show_formula(beta)
    print(f'1X2 accuracy: {correct/len(output):.2%}')
    print(f'Log loss: {np.mean(losses):.4f}; Brier score: {np.mean(briers):.4f}')
    print(f'Saved predictions: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('download', 'train', 'ratings', 'predict', 'backtest'):
        p = sub.add_parser(command)
        p.add_argument('--country', required=True)
        p.add_argument('--db', type=Path)
        if command == 'download':
            p.add_argument('--from-season', type=int, required=True)
            p.add_argument('--to-season', type=int, required=True)
            p.add_argument('--division-ids', nargs='+', type=int)
            p.add_argument('--pause', type=float, default=0.25)
        else:
            p.add_argument('--league-id', type=int, help='Defaults to first division')
        if command in ('train', 'predict'):
            p.add_argument('--model', type=Path)
        if command in ('train', 'ratings', 'backtest'):
            p.add_argument('--k', type=float, default=20)
            p.add_argument('--elo-home-advantage', type=float, default=100)
        if command in ('train', 'backtest'):
            p.add_argument('--alpha', type=float, default=0.01, help='L2 regularization')
        if command == 'train':
            p.add_argument('--through-season', type=int)
        if command == 'predict':
            p.add_argument('--home', required=True)
            p.add_argument('--away', required=True)
        if command == 'backtest':
            p.add_argument('--train-through', type=int, required=True)
            p.add_argument('--test-season', type=int, required=True)
            p.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.country = country_key(args.country)
    if not args.country:
        parser.error('Country cannot be empty.')
    for name in ('k', 'alpha', 'pause'):
        value = getattr(args, name, 0)
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{name} must be finite and nonnegative.')
    if not math.isfinite(getattr(args, 'elo_home_advantage', 0)):
        parser.error('--elo-home-advantage must be finite.')
    args.db = args.db or Path(f'{args.country}.sqlite3')
    if hasattr(args, 'model'):
        suffix = f'_{args.league_id}' if args.league_id is not None else ''
        args.model = args.model or Path(f'{args.country}{suffix}_elo_poisson.json')
    if args.command == 'backtest':
        args.output = args.output or Path(f'{args.country}_elo_backtest_{args.test_season}.csv')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
