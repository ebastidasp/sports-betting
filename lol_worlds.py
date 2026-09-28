"""Two-layer League of Legends Worlds prediction model (free data, no API key).

Layer 1 (baseline, last 12-24 months): domestic win rate, GD15/XPD15,
per-role lane gold@15 diffs, first-blood/herald/dragon/tower/baron rates,
dragon/baron share, KDA by role, last-10 form weighted 2x.
Roster/coach changes in the last 6 months are a strength DISCOUNT.

Champion layer: comfort (shrunk winrate on recent pool), data-driven
patch-form, pool depth (Fearless-relevant), plus optional patch_tiers.csv fit.

League prior: domestic-league strength offsets in Elo points, learned ONLY
from cross-league (Worlds/MSI) residuals — academy farmers stop outranking
major-league sides. Patch-aware EWMA memory (adapts faster on patch change)
and single-temperature recalibration are built in.

Side package: learned blue edge (global + per-patch fallback chain) replaces
the old fixed guess, plus team side-affinity (within-team blue-minus-red)
as a learned feature.

Layer 2 (tournament delta): Tournament Performance Index (TPI) =
  shrunk(intl WR at Worlds/MSI, last 2-3y) - domestic WR (same window).
Positive = clutch/tournament-tested, negative = chronic underperformer.
Thin samples (intl_n < 5) are shrunk to zero and flagged LOW confidence.

Context adjustments (additive log-odds, capped):
  patch-fit (champion pool vs patch tiers), region-vs-region H2H,
  Bo1 vs Bo3/Bo5 series math, prep-time uncertainty.

Output: win probability + fair decimal odds, top 2-3 drivers,
biggest flip risk, confidence flag, and optional EV/Kelly vs bookmaker odds.

Data: Oracle's Elixir CSVs (free, covers LCK/LPL/LEC/LCS + Worlds/MSI).
No key needed. If the direct download moves, manually download from
https://oracleselixir.com/tools/downloads and pass --csv <file>.

Install: py -m pip install pandas numpy scipy scikit-learn joblib requests python-dotenv

Examples:
  py lol_worlds.py download --from-season 2021 --to-season 2025
  py lol_worlds.py download --csv 2024_LoL_esports_match_data_from_OraclesElixir.csv --db lol.sqlite3
  py lol_worlds.py train --db lol.sqlite3 --model lol_worlds.joblib
  py lol_worlds.py ratings --db lol.sqlite3 --model lol_worlds.joblib
  py lol_worlds.py predict --db lol.sqlite3 --model lol_worlds.joblib --team-a T1 --team-b BLG --best-of 5 --odds 1.8 2.1
  py lol_worlds.py predict --db lol.sqlite3 --model lol_worlds.joblib --team-a T1 --team-b BLG --best-of 5 --odds 1.8 2.1 --log-trade trades.csv --stake 10
  py lol_worlds.py trades report --log trades.csv
  py lol_worlds.py trades settle --log trades.csv --id 1 --result A
  py lol_worlds.py backtest --db lol.sqlite3 --train-through 2024-06-01 --test-from 2024-09-01 --test-to 2025-05-01

DB: lol.sqlite3 with team-game rows derived from Oracle's player rows.
Model: lol_worlds.joblib (Elo + logistic + TPI + region H2H + patch weights).
Optional manual files (same folder, all optional):
  rosters.csv      team,date,player,role   (detects <6mo turnover -> discount)
  patch_tiers.csv  patch,champion,tier     (S/A/B/C; computes patch-fit)
  pedigree.csv     player,worlds_runs,mvps  (else derived from DB appearances)
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_KIND = "lol_worlds_two_layer_v4"
DEFAULT_DB = Path("lol.sqlite3")
DEFAULT_MODEL = Path("lol_worlds.joblib")

ORACLE_URLS = [
    "https://oracleselixir.com/tools/downloads/{y}_LoL_esports_match_data_from_OraclesElixir.csv",
    "https://oracleselixir.com/match-data/{y}_LoL_esports_match_data_from_OraclesElixir.csv",
]

# ---------------------------------------------------------------------------
# Helpers: leagues / regions
# ---------------------------------------------------------------------------

def is_intl_league(league: str) -> bool:
    u = str(league or "").upper()
    return ("WORLD" in u) or (u.strip() in {"WLDS", "WLD", "MSI"}) or ("MSI" in u)


def region_of_league(league: str) -> str:
    u = str(league or "").upper()
    mapping = {
        "LCK": "KR", "LPL": "CN", "LEC": "EU", "LCS": "NA", "LTA": "AMERICAS",
        "CBLOL": "BR", "PCS": "PCS", "VCS": "VN", "LJL": "JP", "LLA": "LAT",
        "TCL": "TR", "LCO": "OCE",
    }
    for k, v in mapping.items():
        if k in u:
            return v
    if is_intl_league(league):
        return "INTL"
    return "OTHER"


# Model features (all are A-minus-B diffs, pre-match, leakage-free).
# Base set = original v1. Firsts/lanes are ablatable via --no-firsts/--no-lanes.
BASE_FEATURES = [
    "elo_diff", "gd15_diff", "xpd15_diff", "fb_diff",
    "drag_share_diff", "baron_share_diff", "kda_diff", "form_diff",
]
FIRST_FEATURES = ["fdragon_diff", "fherald_diff", "ftower_diff", "fbaron_diff"]
LANE_FEATURES = ["rg_top_diff", "rg_jng_diff", "rg_mid_diff", "rg_bot_diff", "rg_sup_diff"]
SIDE_FEATURES = ["side_aff_diff"]
MODEL_FEATURES = BASE_FEATURES + FIRST_FEATURES + LANE_FEATURES  # default full set

EWMA_STATS = ["gd15", "xpd15", "fb", "drag_share", "baron_share", "kda",
              "fdragon", "fherald", "ftower", "fbaron",
              "rg_top", "rg_jng", "rg_mid", "rg_bot", "rg_sup",
              "wblue", "wred"]

LANE_ROLES = ["top", "jng", "mid", "bot", "sup"]


def feature_list(args) -> list[str]:
    feats = list(BASE_FEATURES)
    if not getattr(args, "no_firsts", False):
        feats += FIRST_FEATURES
    if not getattr(args, "no_lanes", False):
        feats += LANE_FEATURES
    if not getattr(args, "no_side", False):
        feats += SIDE_FEATURES
    return feats


def affinity_diff(sa, sb, side_a: str) -> float:
    """Signed side edge for A from within-team blue-minus-red gaps (strength-free).

    Matchup math: A on blue vs B on red gains (aff_A + aff_B) / 2, where
    aff_X = blueWR_X - redWR_X. Signed by A's side; NaN when no team has
    both-side history yet.
    """
    terms: list[float] = []
    for st in (sa, sb):
        wb, wr = st.ewma.get("wblue"), st.ewma.get("wred")
        if wb is not None and wr is not None:
            terms.append(float(wb) - float(wr))
    if not terms:
        return float("nan")
    s = 1.0 if str(side_a).lower() == "blue" else -1.0
    return float(sum(terms) / len(terms) / 2.0 * s)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS team_games (
          gameid TEXT NOT NULL,
          team TEXT NOT NULL,
          opp TEXT NOT NULL,
          date TEXT NOT NULL,
          league TEXT NOT NULL,
          split TEXT NOT NULL,
          playoffs INTEGER NOT NULL,
          patch TEXT NOT NULL,
          side TEXT NOT NULL,
          win INTEGER NOT NULL,
          gamelength REAL,
          kills INTEGER, deaths INTEGER, assists INTEGER,
          gd15 REAL, xpd15 REAL,
          fb INTEGER,
          dragons INTEGER, opp_dragons INTEGER,
          barons INTEGER, opp_barons INTEGER,
          kda_top REAL, kda_jng REAL, kda_mid REAL, kda_bot REAL, kda_sup REAL,
          champs_json TEXT, bans_json TEXT,
          fdragon INTEGER, fherald INTEGER, ftower INTEGER, fbaron INTEGER,
          rg_top REAL, rg_jng REAL, rg_mid REAL, rg_bot REAL, rg_sup REAL,
          PRIMARY KEY (gameid, team)
        );
        CREATE INDEX IF NOT EXISTS idx_team_games_team_date ON team_games(team, date);
        """
    )
    # Migrate databases created before the champion/objective/lane columns.
    existing = {row["name"] for row in con.execute("PRAGMA table_info(team_games)")}
    for col, typ in (("champs_json", "TEXT"), ("bans_json", "TEXT"),
                     ("fdragon", "INTEGER"), ("fherald", "INTEGER"),
                     ("ftower", "INTEGER"), ("fbaron", "INTEGER"),
                     ("rg_top", "REAL"), ("rg_jng", "REAL"), ("rg_mid", "REAL"),
                     ("rg_bot", "REAL"), ("rg_sup", "REAL")):
        if col not in existing:
            con.execute(f"ALTER TABLE team_games ADD COLUMN {col} {typ}")
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Oracle's Elixir parsing (robust to yearly schema drift)
# ---------------------------------------------------------------------------

def _cols(df: pd.DataFrame) -> dict[str, str]:
    """Map normalized name -> actual column."""
    return {str(c).strip().lower(): c for c in df.columns}


def _first(cols: dict, *names: str) -> str | None:
    for n in names:
        if n in cols:
            return cols[n]
    return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def kda(k: float, d: float, a: float) -> float:
    k = 0.0 if pd.isna(k) else float(k)
    d = 0.0 if pd.isna(d) else float(d)
    a = 0.0 if pd.isna(a) else float(a)
    return (k + a) / max(1.0, d)


def aggregate_oracle(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse Oracle player-rows (10/game) into 2 team-game rows."""
    cols = _cols(raw)
    c_game = _first(cols, "gameid", "game_id", "matchid", "match_id")
    c_team = _first(cols, "teamname", "team")
    c_date = _first(cols, "date")
    c_league = _first(cols, "league")
    c_split = _first(cols, "split")
    c_playoffs = _first(cols, "playoffs", "playoff")
    c_patch = _first(cols, "patch")
    c_side = _first(cols, "side")
    c_pos = _first(cols, "position", "role")
    c_player = _first(cols, "playername", "player")
    c_champ = _first(cols, "champion")
    c_result = _first(cols, "result")
    c_k = _first(cols, "kills", "k")
    c_d = _first(cols, "deaths", "d")
    c_a = _first(cols, "assists", "a")
    c_glen = _first(cols, "gamelength", "game_length", "gamelength_seconds")
    if c_game is None or c_team is None or c_result is None:
        raise SystemExit("CSV does not look like Oracle's Elixir (need gameid/teamname/result).")
    c_gold15 = _first(cols, "goldat15", "gold_at_15")
    c_xp15 = _first(cols, "xpat15", "xp_at_15")
    c_golddiff15 = _first(cols, "golddiffat15", "gold_diff_at_15", "golddiffat_15")
    c_xpdiff15 = _first(cols, "xpdiffat15", "xp_diff_at_15")
    c_fb = _first(cols, "firstblood", "first_blood", "fb")
    c_drag = _first(cols, "dragons", "dragon", "drakes")
    c_oppdrag = _first(cols, "opp_dragons", "oppdragons", "opponent_dragons")
    c_baron = _first(cols, "barons", "baron", "barons_killed")
    c_oppbaron = _first(cols, "opp_barons", "oppbarons", "opponent_barons")
    c_fdragon = _first(cols, "firstdragon", "first_dragon")
    c_fherald = _first(cols, "firstherald", "first_herald")
    c_ftower = _first(cols, "firsttower", "first_tower")
    c_fbaron = _first(cols, "firstbaron", "first_baron")
    c_bans = [_first(cols, f"ban{i}") for i in range(1, 6)]

    df = raw.copy()
    df["_game"] = df[c_game].astype(str)
    df["_team"] = df[c_team].astype(str)
    df["_win"] = _num(df[c_result]).fillna((df[c_result].astype(str).str.strip() == "1").astype(float)).astype(int).clip(0, 1)
    df["_date"] = pd.to_datetime(df[c_date], utc=True, errors="coerce") if c_date else pd.NaT
    df["_league"] = df[c_league].astype(str) if c_league else "UNK"
    df["_split"] = df[c_split].astype(str) if c_split else ""
    df["_playoffs"] = (_num(df[c_playoffs]).fillna(0).astype(int) if c_playoffs else 0)
    df["_patch"] = df[c_patch].astype(str) if c_patch else ""
    df["_side"] = df[c_side].astype(str).str.lower() if c_side else "unknown"
    df["_pos"] = df[c_pos].astype(str).str.lower() if c_pos else ""
    df["_player"] = df[c_player].astype(str) if c_player else ""
    df["_champ"] = df[c_champ].astype(str) if c_champ else ""
    df["_k"] = _num(df[c_k]).fillna(0) if c_k else 0.0
    df["_d"] = _num(df[c_d]).fillna(0) if c_d else 0.0
    df["_a"] = _num(df[c_a]).fillna(0) if c_a else 0.0
    df["_glen"] = (_num(df[c_glen]) / 60.0 if c_glen else np.nan)  # seconds->min if needed
    # Oracle gamelength is already like 32.5 min in recent files; seconds would be >200.
    # Heuristic: values >200 are seconds.
    df["_glen"] = df["_glen"].map(lambda v: (v / 60.0 if pd.notna(v) and v > 200 else v))
    df["_gold15"] = _num(df[c_gold15]) if c_gold15 else np.nan
    df["_xp15"] = _num(df[c_xp15]) if c_xp15 else np.nan
    df["_gdiff15"] = _num(df[c_golddiff15]) if c_golddiff15 else np.nan
    df["_xdiff15"] = _num(df[c_xpdiff15]) if c_xpdiff15 else np.nan
    df["_fb"] = _num(df[c_fb]).fillna(0).astype(int) if c_fb else 0
    df["_drag"] = _num(df[c_drag]).fillna(0) if c_drag else 0.0
    df["_oppdrag"] = _num(df[c_oppdrag]) if c_oppdrag else np.nan
    df["_baron"] = _num(df[c_baron]).fillna(0) if c_baron else 0.0
    df["_oppbaron"] = _num(df[c_oppbaron]) if c_oppbaron else np.nan
    df["_fdragon"] = _num(df[c_fdragon]) if c_fdragon else np.nan
    df["_fherald"] = _num(df[c_fherald]) if c_fherald else np.nan
    df["_ftower"] = _num(df[c_ftower]) if c_ftower else np.nan
    df["_fbaron"] = _num(df[c_fbaron]) if c_fbaron else np.nan
    df["_is_player"] = ~df["_pos"].str.contains("team", na=False)

    rows: list[dict] = []
    for gameid, g in df.groupby("_game"):
        teams = list(g["_team"].unique())
        if len(teams) != 2:
            continue
        # game date/league from first row
        date = g["_date"].iloc[0]
        if pd.isna(date):
            continue
        for team in teams:
            t = g[g["_team"] == team]
            o = g[g["_team"] != team]
            opp = str(o["_team"].iloc[0])
            win = int(t["_win"].max())
            side = str(t["_side"].iloc[0])
            # Players-only frame: Oracle files carry 2 team-summary rows per game
            # whose kills/gold equal the player totals (summing everything doubles).
            tp = t[t["_is_player"]]
            op = o[o["_is_player"]]
            if tp.empty:
                tp = t  # very old files without team rows
            if op.empty:
                op = o
            # gold/xp diff at 15: prefer direct diff col, else summed player gold diff
            if t["_gdiff15"].notna().any():
                gd15 = float(t["_gdiff15"].dropna().iloc[0])
                # diff col is from this team's perspective already in most files? No:
                # Oracle's golddiffat15 is blue-minus-red. Fix by side below.
            elif tp["_gold15"].notna().any() and op["_gold15"].notna().any():
                gd15 = float(tp["_gold15"].sum() - op["_gold15"].sum())
            else:
                gd15 = np.nan
            if t["_xdiff15"].notna().any():
                xpd15 = float(t["_xdiff15"].dropna().iloc[0])
            elif tp["_xp15"].notna().any() and op["_xp15"].notna().any():
                xpd15 = float(tp["_xp15"].sum() - op["_xp15"].sum())
            else:
                xpd15 = np.nan
            # golddiffat15 in Oracle files is BLUE minus RED regardless of team row.
            # Normalize to team perspective using side.
            if c_golddiff15 and pd.notna(gd15):
                blue_gd = float(g["_gdiff15"].dropna().iloc[0]) if g["_gdiff15"].notna().any() else gd15
                gd15 = blue_gd if side == "blue" else -blue_gd
            if c_xpdiff15 and pd.notna(xpd15):
                blue_xd = float(g["_xdiff15"].dropna().iloc[0]) if g["_xdiff15"].notna().any() else xpd15
                xpd15 = blue_xd if side == "blue" else -blue_xd
            fb = int(t["_fb"].max())
            drag = float(t["_drag"].max())
            baron = float(t["_baron"].max())
            opp_drag = float(o["_drag"].max())
            opp_baron = float(o["_baron"].max())
            if pd.notna(t["_oppdrag"]).any():
                opp_drag = float(t["_oppdrag"].dropna().iloc[0])
            if pd.notna(t["_oppbaron"]).any():
                opp_baron = float(t["_oppbaron"].dropna().iloc[0])

            def _first_flag(col: str) -> float | None:
                vals = t[col].dropna()
                if vals.empty:
                    return None
                try:
                    return int(float(vals.max()) > 0)
                except (TypeError, ValueError):
                    return None

            fdragon = _first_flag("_fdragon")
            fherald = _first_flag("_fherald")
            ftower = _first_flag("_ftower")
            fbaron = _first_flag("_fbaron")
            # KDA by role + role gold@15 diffs (lane matchup signal)
            role_kda: dict[str, float] = {}
            role_gold: dict[str, float] = {}
            champs: dict[str, str] = {}
            for role_key, pat in (("top", "top"), ("jng", "jng"), ("mid", "mid"),
                                  ("bot", "bot"), ("sup", "sup")):
                r = tp[tp["_pos"].str.contains(pat, na=False)]
                ro = op[op["_pos"].str.contains(pat, na=False)]
                if len(r):
                    role_kda[role_key] = kda(r["_k"].sum(), r["_d"].sum(), r["_a"].sum())
                    champ = str(r["_champ"].dropna().iloc[0]) if r["_champ"].notna().any() else ""
                    if champ and champ.lower() not in ("nan", "none", ""):
                        champs[role_key] = champ
                else:
                    role_kda[role_key] = np.nan
                if len(r) and len(ro) and r["_gold15"].notna().any() and ro["_gold15"].notna().any():
                    role_gold[role_key] = float(r["_gold15"].sum() - ro["_gold15"].sum())
                else:
                    role_gold[role_key] = np.nan
            bans: list[str] = []
            for bcol in c_bans:
                if bcol is None:
                    continue
                vals = t[bcol].dropna().astype(str)
                vals = vals[~vals.str.lower().isin(["nan", "none", ""])]
                if len(vals):
                    bans.append(str(vals.iloc[0]))
            import json as _json
            rows.append({
                "gameid": str(gameid), "team": str(team), "opp": opp,
                "date": pd.Timestamp(date).isoformat(),
                "league": str(t["_league"].iloc[0]), "split": str(t["_split"].iloc[0]),
                "playoffs": int(t["_playoffs"].iloc[0]), "patch": str(t["_patch"].iloc[0]),
                "side": side, "win": win,
                "gamelength": float(t["_glen"].dropna().iloc[0]) if t["_glen"].notna().any() else np.nan,
                "kills": int(tp["_k"].sum()), "deaths": int(tp["_d"].sum()),
                "assists": int(tp["_a"].sum()),
                "gd15": gd15, "xpd15": xpd15, "fb": fb,
                "dragons": drag, "opp_dragons": opp_drag,
                "barons": baron, "opp_barons": opp_baron,
                "kda_top": role_kda["top"], "kda_jng": role_kda["jng"],
                "kda_mid": role_kda["mid"], "kda_bot": role_kda["bot"],
                "kda_sup": role_kda["sup"],
                "champs_json": _json.dumps(champs, ensure_ascii=False),
                "bans_json": _json.dumps(bans, ensure_ascii=False),
                "fdragon": fdragon, "fherald": fherald,
                "ftower": ftower, "fbaron": fbaron,
                "rg_top": role_gold["top"], "rg_jng": role_gold["jng"],
                "rg_mid": role_gold["mid"], "rg_bot": role_gold["bot"],
                "rg_sup": role_gold["sup"],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], utc=True)
        out = out.sort_values(["date", "gameid", "team"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(args: argparse.Namespace) -> None:
    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = connect(args.db)
    try:
        frames: list[pd.DataFrame] = []
        if args.csv:
            for p in args.csv:
                p = Path(p)
                if not p.is_file():
                    raise SystemExit(f"CSV not found: {p}")
                print(f"Reading {p} ...")
                raw = pd.read_csv(p, low_memory=False)
                frames.append(aggregate_oracle(raw))
        else:
            sess = requests.Session()
            sess.headers["User-Agent"] = "lol-worlds-model/1.0 (research)"
            for y in range(args.from_season, args.to_season + 1):
                got = None
                for tmpl in ORACLE_URLS:
                    url = tmpl.format(y=y)
                    try:
                        r = sess.get(url, timeout=60)
                        if r.status_code == 200 and len(r.content) > 1000:
                            from io import StringIO
                            raw = pd.read_csv(StringIO(r.text), low_memory=False)
                            got = aggregate_oracle(raw)
                            print(f"{y}: {len(got)//2} games from {url}")
                            break
                    except Exception as e:  # noqa: BLE001 - try next mirror
                        print(f"{y}: {url} failed ({e})")
                    time.sleep(args.pause)
                if got is None:
                    print(f"{y}: download failed. Manually download from "
                          f"https://oracleselixir.com/tools/downloads and rerun with --csv.")
                else:
                    frames.append(got)
        if not frames:
            raise SystemExit("No games imported. Use --csv with a manually downloaded Oracle file.")
        games = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["gameid", "team"])
        games = games.sort_values(["date", "gameid"]).reset_index(drop=True)
        games.to_sql("team_games", con, if_exists="replace", index=False)
        print(f"Cached {len(games)} team-game rows ({len(games)//2} games) in {args.db}")
        print(f"Leagues: {sorted(games.league.unique())[:12]}")
        print(f"Range: {games.date.min()} .. {games.date.max()}")
    finally:
        con.close()


def load_games(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM team_games ORDER BY date, gameid, team", con)
    if df.empty:
        raise SystemExit("No games cached. Run download first.")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values(["date", "gameid", "team"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Elo + leakage-free EWMA features
# ---------------------------------------------------------------------------

def ewma_update(old: float | None, new: float | None, alpha: float) -> float | None:
    if new is None or (isinstance(new, float) and pd.isna(new)):
        return old
    return float(new) if old is None else alpha * float(new) + (1 - alpha) * float(old)


from dataclasses import dataclass


try:

    @dataclass
    class TeamState:
        elo: float = 1500.0
        games: int = 0
        ewma: dict = None
        last10: list = None
        last_patch: str = ""
        last_intl_date = None
        intl_games: int = 0
        region: str = "OTHER"
        dom_league: str = ""

        def __post_init__(self):
            if self.ewma is None:
                self.ewma = {}
            if self.last10 is None:
                self.last10 = []
except Exception:  # pragma: no cover
    TeamState = dict  # type: ignore


def build_examples(games: pd.DataFrame, alpha: float, min_history: int,
                   k_elo: float = 20.0, side_adv: float = 35.0,
                   patch_decay: bool = True):
    """Chronological pre-match features. One example per game (team A perspective).

    games: team-game rows (2 per game). Returns (examples df, states, team_region).
    patch_decay: adapt EWMAs ~4x faster on the first game of a new patch.
    """
    states: dict[str, TeamState] = defaultdict(TeamState)
    team_region: dict[str, str] = {}
    examples: list[dict] = []
    # region + domestic league = most common domestic league (fallback: overall mode)
    for team, g in games.groupby("team"):
        dom = g[~g["league"].map(is_intl_league)]
        lg = dom["league"].mode().iloc[0] if len(dom) else g["league"].mode().iloc[0]
        team_region[team] = region_of_league(lg)
        states[team].region = team_region[team]
        states[team].dom_league = str(lg)

    for gameid, g in games.groupby("gameid", sort=False):
        if len(g) != 2:
            continue
        g = g.sort_values("team")
        a_row, b_row = g.iloc[0], g.iloc[1]
        A, B = str(a_row["team"]), str(b_row["team"])
        sa, sb = states[A], states[B]
        eligible = sa.games >= min_history and sb.games >= min_history
        # pre-match diffs from EWMAs
        rec: dict = {
            "gameid": str(gameid), "date": a_row["date"],
            "team_a": A, "team_b": B,
            "side_a": str(a_row["side"]), "side_b": str(b_row["side"]),
            "league": str(a_row["league"]), "patch": str(a_row["patch"]),
            "win_a": int(a_row["win"]),
            "elo_diff": (sa.elo - sb.elo) / 400.0,
            "eligible": bool(eligible),
            "games_a": sa.games, "games_b": sb.games,
            "league_a": sa.dom_league, "league_b": sb.dom_league,
        }
        for stat in EWMA_STATS:
            va, vb = sa.ewma.get(stat), sb.ewma.get(stat)
            rec[f"{stat}_diff"] = (float(va) - float(vb)
                                  if va is not None and vb is not None else np.nan)
        rec["side_aff_diff"] = affinity_diff(sa, sb, str(a_row["side"]))
        # form: last-10 winrate diff (spec: last10 weighted 2x vs season avg -> applied at predict time;
        # here expose raw last10 so the model learns the weight; also store blended below)
        fa = float(np.mean(sa.last10)) if sa.last10 else np.nan
        fb_ = float(np.mean(sb.last10)) if sb.last10 else np.nan
        rec["form_diff"] = (fa - fb_ if pd.notna(fa) and pd.notna(fb_) else np.nan)
        rec["last10_a"] = fa
        rec["last10_b"] = fb_
        examples.append(rec)

        # ---- updates (after recording pre-match features) ----
        for row, st in ((a_row, sa), (b_row, sb)):
            tot_d = float(row["dragons"]) + float(row["opp_dragons"])
            tot_b = float(row["barons"]) + float(row["opp_barons"])
            drag_share = float(row["dragons"]) / tot_d if tot_d > 0 else np.nan
            baron_share = float(row["barons"]) / tot_b if tot_b > 0 else np.nan
            k, d, a_ = float(row["kills"]), float(row["deaths"]), float(row["assists"])
            kda_all = (k + a_) / max(1.0, d)
            side_l = str(row["side"]).lower()
            vals = {"gd15": row["gd15"], "xpd15": row["xpd15"], "fb": row["fb"],
                    "drag_share": drag_share, "baron_share": baron_share, "kda": kda_all,
                    "fdragon": row.get("fdragon"), "fherald": row.get("fherald"),
                    "ftower": row.get("ftower"), "fbaron": row.get("fbaron"),
                    "rg_top": row.get("rg_top"), "rg_jng": row.get("rg_jng"),
                    "rg_mid": row.get("rg_mid"), "rg_bot": row.get("rg_bot"),
                    "rg_sup": row.get("rg_sup"),
                    "wblue": row["win"] if side_l == "blue" else np.nan,
                    "wred": row["win"] if side_l == "red" else np.nan}
            for stat, v in vals.items():
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = np.nan
                # Patch-aware memory: a patch change means the old meta's stats
                # carry less weight — adapt ~4x faster on the first game of a patch.
                a_eff = alpha
                if patch_decay and st.last_patch and str(row["patch"]) != st.last_patch:
                    a_eff = 1.0 - (1.0 - alpha) ** 4
                st.ewma[stat] = ewma_update(st.ewma.get(stat), None if pd.isna(v) else v, a_eff)
            st.last10.append(int(row["win"]))
            st.last10 = st.last10[-10:]
            st.games += 1
            st.last_patch = str(row["patch"])
            if is_intl_league(row["league"]):
                st.last_intl_date = pd.Timestamp(row["date"])
                st.intl_games += 1
        # Elo update with side adjustment (blue edge ~ side_adv Elo)
        def eff(st: TeamState, side: str) -> float:
            return st.elo + (side_adv if str(side).lower() == "blue" else 0.0)

        ea = 1.0 / (1.0 + 10.0 ** ((eff(sb, str(b_row["side"])) - eff(sa, str(a_row["side"]))) / 400.0))
        # margin multiplier: blowouts move more, short stomps slightly more
        margin = abs(float(a_row["kills"]) - float(b_row["kills"]))
        mult = 1.0 + min(0.5, margin / 20.0)
        score_a = float(a_row["win"])
        change = k_elo * mult * (score_a - ea)
        sa.elo += change
        sb.elo -= change

    frame = pd.DataFrame(examples)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.sort_values(["date", "gameid"]).reset_index(drop=True)
    return frame, dict(states), team_region


def time_weights(dates: pd.Series, cutoff: pd.Timestamp, half_life: float) -> np.ndarray:
    if half_life <= 0:
        return np.ones(len(dates))
    age = (pd.Timestamp(cutoff) - pd.to_datetime(dates, utc=True)).dt.total_seconds() / 86400.0
    return np.exp2(-np.maximum(age.to_numpy(), 0.0) / half_life)


# ---------------------------------------------------------------------------
# TPI + region H2H + patch + pedigree (all optional-graceful)
# ---------------------------------------------------------------------------

def compute_tpi(games: pd.DataFrame, cutoff: pd.Timestamp, window_days: int = 1095) -> dict:
    """TPI = shrunk intl WR - domestic WR over [cutoff-window, cutoff)."""
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=window_days)
    w = games[(games["date"] >= start) & (games["date"] < pd.Timestamp(cutoff))]
    out: dict[str, dict] = {}
    for team, g in w.groupby("team"):
        intl = g[g["league"].map(is_intl_league)]
        dom = g[~g["league"].map(is_intl_league)]
        n_i, n_d = len(intl), len(dom)
        wr_i = float(intl["win"].mean()) if n_i else np.nan
        wr_d = float(dom["win"].mean()) if n_d else np.nan
        if pd.isna(wr_i) or pd.isna(wr_d):
            tpi_raw = 0.0
        else:
            tpi_raw = wr_i - wr_d
        shrink = n_i / (n_i + 5.0)  # thin intl samples -> 0
        tpi = float(tpi_raw * shrink) if (pd.notna(wr_i) and pd.notna(wr_d)) else 0.0
        flag = "CLUTCH" if tpi > 0.10 else ("UNDERPERFORMER" if tpi < -0.10 else "NEUTRAL")
        if n_i < 5:
            flag = "THIN_SAMPLE"
        out[team] = {"tpi": tpi, "tpi_raw": float(tpi_raw) if pd.notna(wr_i) and pd.notna(wr_d) else 0.0,
                     "intl_wr": wr_i, "dom_wr": wr_d, "intl_n": int(n_i), "dom_n": int(n_d),
                     "flag": flag}
    return out


def region_h2h(games: pd.DataFrame, cutoff: pd.Timestamp, window_days: int = 1095) -> dict:
    """Shrunk region-pair edge in log-odds from intl games. Symmetric dict."""
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=window_days)
    w = games[(games["date"] >= start) & (games["date"] < pd.Timestamp(cutoff))]
    w = w[w["league"].map(is_intl_league)]
    # need per-game region pairs: rebuild from team_games
    pair_wins: dict[tuple[str, str], list[int]] = defaultdict(list)
    treemap: dict[str, str] = {}
    for team, g in games.groupby("team"):
        dom = g[~g["league"].map(is_intl_league)]
        lg = dom["league"].mode().iloc[0] if len(dom) else g["league"].mode().iloc[0]
        treemap[team] = region_of_league(lg)
    for gameid, g in w.groupby("gameid"):
        if len(g) != 2:
            continue
        g = g.sort_values("team")
        ra, rb = treemap.get(g.iloc[0]["team"], "OTHER"), treemap.get(g.iloc[1]["team"], "OTHER")
        if ra == rb:
            continue
        pair_wins[(ra, rb)].append(int(g.iloc[0]["win"]))
    edge: dict[tuple[str, str], float] = {}
    for (ra, rb), wins in pair_wins.items():
        n = len(wins)
        p = float(np.mean(wins))
        # shrink to 0.5 with 8 pseudo-games; convert to log-odds, cap
        ps = (p * n + 0.5 * 8) / (n + 8)
        lo = math.log(ps / (1 - ps))
        edge[(ra, rb)] = float(np.clip(lo, -0.4, 0.4))
        edge[(rb, ra)] = float(-np.clip(lo, -0.4, 0.4))
    return edge


def team_domestic_leagues(games: pd.DataFrame) -> dict[str, str]:
    """team -> most common domestic league (fallback: overall mode)."""
    out: dict[str, str] = {}
    for team, g in games.groupby("team"):
        dom = g[~g["league"].map(is_intl_league)]
        lg = dom["league"].mode().iloc[0] if len(dom) else g["league"].mode().iloc[0]
        out[team] = str(lg)
    return out


def compute_league_offsets(games: pd.DataFrame, frame: pd.DataFrame, cutoff,
                           side_adv: float = 35.0, window_days: int = 1095,
                           shrink_n: float = 15.0, cap: float = 150.0) -> dict:
    """League strength in Elo points, learned ONLY from cross-league (intl) games.

    residual = actual - Elo-expected; league offset converts the shrunk mean
    residual to Elo points (~693x at p=0.5). Applied with weight 1.0, i.e. as
    a literal Elo correction: (elo_A + off_A) - (elo_B + off_B).
    """
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=window_days)
    w = frame[(frame["date"] >= start) & (frame["date"] < pd.Timestamp(cutoff))
              & (frame["league"].map(is_intl_league))].copy()
    dom = team_domestic_leagues(games[games["date"] < pd.Timestamp(cutoff)])
    acc: dict[str, list[float]] = defaultdict(list)
    for r in w.itertuples(index=False):
        try:
            la, lb = dom.get(r.team_a, "UNK"), dom.get(r.team_b, "UNK")
            s_a = side_adv / 400.0 if str(r.side_a).lower() == "blue" else 0.0
            s_b = side_adv / 400.0 if str(r.side_b).lower() == "blue" else 0.0
            exp = 1.0 / (1.0 + 10.0 ** (-(float(r.elo_diff) + s_a - s_b)))
            resid = float(r.win_a) - exp
        except (ValueError, TypeError, AttributeError):
            continue
        acc[la].append(resid)
        acc[lb].append(-resid)
    out: dict[str, dict] = {}
    for lg, resids in acc.items():
        n = len(resids)
        mean = float(np.mean(resids)) if n else 0.0
        off = float(np.clip(mean * (n / (n + shrink_n)) * 693.0, -cap, cap))
        out[lg] = {"off": off, "n": int(n), "raw": mean}
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter recalibration: p = sigmoid(logit / T). T>1 tames extremes."""
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)

    def loss(t):
        if t <= 0:
            return float("inf")
        p = 1.0 / (1.0 + np.exp(-np.clip(logits / t, -500, 500)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    fit = minimize_scalar(loss, bounds=(0.7, 1.6), method="bounded",
                          options={"xatol": 1e-3})
    return float(fit.x) if fit.success else 1.0


def compute_side_edges(games: pd.DataFrame, cutoff,
                       window_days: int = 1095, shrink_patch: float = 50.0,
                       cap: float = 0.3) -> dict:
    """Blue-side edge in log-odds, global + per-patch (shrunk to global).

    Measured from blue-side team-game rows only (one row per game).
    Fresh patches with < shrink_patch games collapse toward global;
    a patch with zero games returns the global edge (never NaN).
    """
    cut = pd.Timestamp(cutoff)
    start = cut - pd.Timedelta(days=window_days)
    w = games[(games["date"] >= start) & (games["date"] < cut)
              & (games["side"].astype(str).str.lower() == "blue")].copy()

    def _edge(wins: int, n: int, prior_p: float, prior_k: float) -> tuple[float, float]:
        p = (wins + prior_p * prior_k) / (n + prior_k) if n else prior_p
        p = float(np.clip(p, 0.05, 0.95))
        return float(np.clip(math.log(p / (1 - p)), -cap, cap)), p

    n_all = len(w)
    w_all = int(w["win"].sum()) if n_all else 0
    edge_glob, p_glob = _edge(w_all, n_all, 0.5, 200.0)
    patches: dict[str, dict] = {}
    if "patch" in w.columns:
        for patch, g in w.groupby(w["patch"].astype(str)):
            edge, p = _edge(int(g["win"].sum()), len(g), p_glob, shrink_patch)
            patches[str(patch)] = {"edge": edge, "p": p, "n": int(len(g))}
    return {"global": {"edge": edge_glob, "p": p_glob, "n": int(n_all)},
            "patches": patches}


def side_edge_for(bundle_edges: dict, patch: str | None) -> tuple[float, str]:
    """Return (edge, source) for a patch, falling back to global. Never NaN."""
    glob = bundle_edges.get("global", {"edge": 0.0, "n": 0})
    if patch:
        hit = bundle_edges.get("patches", {}).get(str(patch))
        if hit is not None:
            return float(hit["edge"]), f"patch {patch} (n={hit['n']})"
    return float(glob.get("edge", 0.0)), f"global (n={glob.get('n', 0)})"


def load_patch_tiers(path: Path | None) -> dict:
    """patch -> {champion_lower: score}. S=1.0 A=0.6 B=0.25 C=0.0. Missing file -> {}."""
    if path is None or not Path(path).is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    cols = {str(c).lower(): c for c in df.columns}
    if not {"patch", "champion", "tier"} <= set(cols):
        return {}
    out: dict[str, dict[str, float]] = defaultdict(dict)
    score = {"S": 1.0, "A": 0.6, "B": 0.25, "C": 0.0}
    for _, r in df.iterrows():
        out[str(r[cols["patch"]]).strip()][str(r[cols["champion"]]).strip().lower()] = score.get(
            str(r[cols["tier"]]).strip().upper(), 0.0)
    return dict(out)


def team_champ_pool(games: pd.DataFrame, team: str, cutoff, n: int = 20) -> list[str]:
    """Most recent champions per role from stored champs_json (players rows)."""
    import json as _json
    w = games[(games["team"] == team) & (games["date"] < pd.Timestamp(cutoff))]
    w = w.sort_values("date").tail(n)
    pool: list[str] = []
    for raw in w["champs_json"].dropna() if "champs_json" in w else []:
        try:
            pool.extend([str(v) for v in _json.loads(raw).values() if v])
        except Exception:
            continue
    return pool


def patch_fit_score(pool: list[str], tiers: dict[str, float]) -> float:
    if not pool or not tiers:
        return 0.0
    return float(np.mean([tiers.get(c.lower(), 0.0) for c in pool]))


def compute_champ_stats(games: pd.DataFrame, cutoff, patch: str | None = None,
                        pool_n: int = 20, base_n: int = 80) -> dict:
    """Pre-match champion signal, leakage-free (base window strictly before pool).

    comfort: mean shrunk role-champ winrate (from base) over the recent pool.
    patchform: shrunk winrate on `patch` (last 60) minus 0.5, data-driven fit.
    depth: distinct recent champs / 12 (Fearless-relevant pool breadth).
    Manual patch_tiers.csv adds an extra tier-fit term at predict time.
    """
    import json as _json
    cut = pd.Timestamp(cutoff)
    w = games[games["date"] < cut].sort_values("date")
    has_champs = "champs_json" in w.columns
    out: dict[str, dict] = {}
    for team, g in w.groupby("team"):
        g = g.sort_values("date")
        pool = g.tail(pool_n)
        base = g.iloc[max(0, len(g) - base_n - pool_n):max(0, len(g) - pool_n)]
        # (role, champ) -> [wins, games] from base window
        wr: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for _, r in base.iterrows():
            try:
                champs = _json.loads(r["champs_json"]) if has_champs and pd.notna(r.get("champs_json")) else {}
            except Exception:
                champs = {}
            for role, champ in champs.items():
                key = (str(role).lower(), str(champ).lower())
                wr[key][0] += int(r["win"])
                wr[key][1] += 1
        comfs: list[float] = []
        distinct: set[str] = set()
        for _, r in pool.iterrows():
            try:
                champs = _json.loads(r["champs_json"]) if has_champs and pd.notna(r.get("champs_json")) else {}
            except Exception:
                champs = {}
            for role, champ in champs.items():
                key = (str(role).lower(), str(champ).lower())
                distinct.add(key[1])
                wins, n = wr.get(key, (0, 0))
                comfs.append((wins + 2.5) / (n + 5.0) - 0.5)  # centered, shrunk
        comfort = float(np.mean(comfs)) if comfs else 0.0
        depth = float(min(1.0, len(distinct) / 12.0)) if distinct else 0.0
        patchform = 0.0
        if patch:
            pg = g[g["patch"].astype(str) == str(patch)].tail(60)
            n = len(pg)
            if n >= 3:
                patchform = float((pg["win"].sum() + 2.0) / (n + 4.0) - 0.5)
        out[team] = {"comfort": comfort, "patchform": patchform, "depth": depth,
                     "pool_n": int(len(pool)), "patch": str(patch or "")}
    return out


def champ_edge(ca: dict, cb: dict, bundle: dict) -> float:
    """Capped log-odds edge from champion signal. Zero when data is thin."""
    if ca.get("pool_n", 0) < 5 or cb.get("pool_n", 0) < 5:
        return 0.0
    edge = (bundle.get("champ_w_comfort", 0.8) * (ca.get("comfort", 0.0) - cb.get("comfort", 0.0))
            + bundle.get("champ_w_patch", 0.8) * (ca.get("patchform", 0.0) - cb.get("patchform", 0.0))
            + bundle.get("champ_w_depth", 0.4) * (ca.get("depth", 0.0) - cb.get("depth", 0.0)))
    return float(np.clip(edge, -0.35, 0.35))


def load_roster_discounts(path: Path | None, games: pd.DataFrame,
                          cutoff: pd.Timestamp, months: int = 6) -> dict[str, str]:
    """Teams with >2 starter changes in last `months` -> discount. Manual CSV overrides.

    rosters.csv columns: team,date,player,role (role in top/jng/mid/bot/sup).
    Returns {team: reason}. Missing file -> {} (no discount) unless auto-detected.
    Auto-detection from team_games is impossible (no player rows), so without the
    file we return {} and warn at predict time.
    """
    if path is not None and Path(path).is_file():
        try:
            df = pd.read_csv(path)
            cols = {str(c).lower(): c for c in df.columns}
            if {"team", "date", "player"} <= set(cols):
                df["_date"] = pd.to_datetime(df[cols["date"]], utc=True, errors="coerce")
                cut = pd.Timestamp(cutoff)
                recent = df[df["_date"] >= cut - pd.DateOffset(months=months)]
                old = df[(df["_date"] < cut - pd.DateOffset(months=months))]
                out: dict[str, str] = {}
                for team in df[cols["team"]].astype(str).unique():
                    r = set(recent[recent[cols["team"]].astype(str) == team][cols["player"]].astype(str).str.lower())
                    o = set(old[old[cols["team"]].astype(str) == team][cols["player"]].astype(str).str.lower())
                    if o and r and len(r.symmetric_difference(o)) >= 3:
                        out[team] = f"roster turnover {len(r.symmetric_difference(o))} players in {months}mo"
                return out
        except Exception:
            return {}
    return {}


def pedigree_bonus(games: pd.DataFrame, team: str, cutoff: pd.Timestamp,
                   ped_path: Path | None = None) -> float:
    """Small log-odds bonus for internationally experienced cores. Capped at +0.15."""
    if ped_path is not None and Path(ped_path).is_file():
        return 0.0  # reserved: manual pedigree.csv hook (kept neutral until validated)
    past = games[(games["team"] == team) & (games["date"] < pd.Timestamp(cutoff))
                 & (games["league"].map(is_intl_league))]
    ko = len(past[past["playoffs"] == 1])
    return float(min(0.15, 0.03 * ko))


# ---------------------------------------------------------------------------
# Probabilities: game -> series, context blending
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def series_prob(pg: float, best_of: int, bo5_boost: float = 0.12) -> float:
    """Game prob -> series prob. Favorites stabilize in Bo5 (pg pushed from 0.5)."""
    pg = float(np.clip(pg, 0.01, 0.99))
    if best_of <= 1:
        return pg
    if best_of in (3, 5):
        # favorites stabilize: stretch distance from 0.5 for longer series
        stretch = 1.0 + (bo5_boost if best_of == 5 else bo5_boost / 2)
        pg_adj = float(np.clip(0.5 + (pg - 0.5) * stretch, 0.01, 0.99))
        need = best_of // 2 + 1
        from math import comb
        return float(sum(comb(best_of, k) * pg_adj ** k * (1 - pg_adj) ** (best_of - k)
                         for k in range(need, best_of + 1)))
    raise SystemExit("--best-of must be 1, 3 or 5.")


# ---------------------------------------------------------------------------
# Train / ratings / predict / backtest
# ---------------------------------------------------------------------------

def _as_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def fit_logistic(X: pd.DataFrame, y: pd.Series, w: np.ndarray, alpha: float) -> Pipeline:
    from sklearn.impute import SimpleImputer
    C = 1.0 / max(1e-6, alpha) if alpha > 0 else 1e6
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000)),
    ])
    pipe.fit(X, y, lr__sample_weight=w)
    return pipe


def train(args: argparse.Namespace) -> None:
    con = connect(args.db)
    try:
        games = load_games(con)
    finally:
        con.close()
    games = games[games["date"] < _as_utc(args.train_through)].copy() if args.train_through else games
    frame, states, team_region = build_examples(games, args.ewma_alpha, args.min_history,
                                                args.k_elo, args.side_adv,
                                                not getattr(args, "no_patchdecay", False))
    elig = frame[frame["eligible"]].copy()
    if len(elig) < 100:
        raise SystemExit(f"Only {len(elig)} eligible games (need >=100). Lower --min-history.")
    feats = feature_list(args)
    for f in feats:
        if f not in elig.columns:
            elig[f] = 0.0
    X = elig[feats].copy()
    y = elig["win_a"].astype(int)
    cutoff = elig["date"].max() + pd.Timedelta(nanoseconds=1)
    w = time_weights(elig["date"], cutoff, args.half_life)
    model = fit_logistic(X, y, w, args.regularization)
    # chronological holdout for honest metrics
    split = min(len(elig) - 1, max(1, int(len(elig) * (1 - args.test_fraction))))
    cut_date = elig.iloc[split]["date"]
    tr, te = elig[elig["date"] < cut_date], elig[elig["date"] >= cut_date]
    temperature = 1.0
    if len(tr) >= 50 and len(te) >= 10:
        m0 = fit_logistic(tr[feats], tr["win_a"].astype(int),
                          time_weights(tr["date"], cut_date, args.half_life), args.regularization)
        p = m0.predict_proba(te[feats])[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        y_te = te["win_a"].to_numpy().astype(int)
        print(f"Holdout: {len(te)} games | acc {( (p>=0.5).astype(int)==y_te).mean():.2%} "
              f"| logloss {log_loss(y_te, p):.4f} | brier {np.mean((p-y_te)**2):.4f}")
        # full-layer holdout: base + TPI + champ + league + side, all frozen at cut_date
        tpi_h = compute_tpi(games, cut_date, args.tpi_window_days)
        off_h = compute_league_offsets(games, frame, cut_date, args.side_adv)
        edges_h = compute_side_edges(games, cut_date)
        geh = edges_h["global"]["edge"]
        peh = edges_h["patches"]
        pre_h = games[games["date"] < cut_date].copy()
        champ_h: dict[str, dict] = {}
        for patch in te["patch"].astype(str).unique():
            champ_h[patch] = compute_champ_stats(pre_h, cut_date, patch if patch else None)
        cw = {"champ_w_comfort": 0.8, "champ_w_patch": 0.8, "champ_w_depth": 0.4}
        logit_h = np.log(p / (1 - p))
        for i, (a, b, patch) in enumerate(zip(te["team_a"], te["team_b"], te["patch"].astype(str))):
            la = te.iloc[i].get("league_a", "") if hasattr(te.iloc[i], "get") else ""
            lb = te.iloc[i].get("league_b", "") if hasattr(te.iloc[i], "get") else ""
            logit_h[i] += args.tpi_weight * (tpi_h.get(a, {}).get("tpi", 0.0)
                                             - tpi_h.get(b, {}).get("tpi", 0.0))
            cs = champ_h.get(patch, {})
            try:
                logit_h[i] += champ_edge(cs.get(a, {}), cs.get(b, {}), cw)
            except Exception:
                pass
            if not getattr(args, "no_league", False):
                logit_h[i] += (off_h.get(la, {}).get("off", 0.0)
                               - off_h.get(lb, {}).get("off", 0.0)) / 400.0
            if not getattr(args, "no_side", False):
                s = 1.0 if str(te.iloc[i].get("side_a", "blue")).lower() == "blue" else -1.0
                logit_h[i] += peh.get(str(patch), {}).get("edge", geh) / 2.0 * s
        temperature = fit_temperature(logit_h, y_te)
        pc = 1 / (1 + np.exp(-np.clip(logit_h / temperature, -500, 500)))
        pc = np.clip(pc, 1e-6, 1 - 1e-6)
        print(f"Holdout full layers: logloss {log_loss(y_te, pc):.4f} | brier {np.mean((pc-y_te)**2):.4f} "
              f"| temperature {temperature:.3f}")
        print(f"Side edge at holdout: global {geh:+.3f} lo "
              f"(blue {edges_h['global']['p']:.1%}, n={edges_h['global']['n']})")
    tpi = compute_tpi(games, cutoff, args.tpi_window_days)
    h2h = region_h2h(games, cutoff, args.tpi_window_days)
    league_off = compute_league_offsets(games, frame, cutoff, args.side_adv)
    side_edges = compute_side_edges(games, cutoff)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_kind": MODEL_KIND, "logistic": model, "features": feats,
                 "no_firsts": bool(getattr(args, "no_firsts", False)),
                 "no_lanes": bool(getattr(args, "no_lanes", False)),
                 "no_champ": bool(getattr(args, "no_champ", False)),
                 "no_league": bool(getattr(args, "no_league", False)),
                 "no_side": bool(getattr(args, "no_side", False)),
                 "side_edges": side_edges,
                 "patch_decay": not bool(getattr(args, "no_patchdecay", False)),
                 "temperature": float(temperature),
                 "league_offsets": {k: v for k, v in league_off.items()},
                 "champ_w_comfort": 0.8, "champ_w_patch": 0.8, "champ_w_depth": 0.4,
                 "ewma_alpha": args.ewma_alpha, "min_history": args.min_history,
                 "k_elo": args.k_elo, "side_adv": args.side_adv,
                 "tpi": tpi, "region_h2h": {f"{a}|{b}": v for (a, b), v in h2h.items()},
                 "team_region": team_region,
                 "tpi_weight": args.tpi_weight, "patch_weight": args.patch_weight,
                 "region_weight": args.region_weight, "bo5_boost": args.bo5_boost,
                 "cutoff": pd.Timestamp(cutoff).isoformat(),
                 "trained_at": datetime.now(timezone.utc).isoformat()}, args.model)
    print(f"Trained on {len(elig)} eligible games; cutoff {pd.Timestamp(cutoff).date()}")
    print(f"Features ({len(feats)}): {', '.join(feats)}")
    print("League offsets (Elo pts, intl games only): " +
          ", ".join(f"{lg} {v['off']:+.0f}(n={v['n']})"
                    for lg, v in sorted(league_off.items(), key=lambda t: -t[1]["off"])[:8]))
    coefs = model.named_steps["lr"].coef_[0]
    for f, c in sorted(zip(feats, coefs), key=lambda t: -abs(t[1])):
        print(f"  {f:16s} coef {c:+.4f}")
    print(f"Saved: {args.model}")


def find_team(query: str, teams: list[str]) -> str:
    exact = [t for t in teams if t.casefold() == query.casefold()]
    if exact:
        return exact[0]
    part = [t for t in teams if query.casefold() in t.casefold()]
    if len(part) == 1:
        return part[0]
    raise SystemExit(f"Team '{query}' missing/ambiguous. Matches: {', '.join(sorted(part)[:10]) or 'none'}")


def _prematch_row(games: pd.DataFrame, states, team_a: str, team_b: str,
                  as_of: pd.Timestamp, alpha: float, k_elo: float, side_adv: float,
                  side_a: str = "blue", patch_decay: bool = True) -> dict:
    """Replay history to as_of, return pre-match diffs + elos without mutating bundle."""
    frame, states2, _ = build_examples(games[games["date"] < pd.Timestamp(as_of)],
                                       alpha, 0, k_elo, side_adv, patch_decay)
    if team_a not in states2 or team_b not in states2:
        raise SystemExit("One of the teams has no cached games before --as-of.")
    sa, sb = states2[team_a], states2[team_b]
    rec = {"elo_diff": (sa.elo - sb.elo) / 400.0}
    for stat in EWMA_STATS:
        va, vb = sa.ewma.get(stat), sb.ewma.get(stat)
        rec[f"{stat}_diff"] = (float(va) - float(vb)
                               if va is not None and vb is not None else 0.0)
    rec["side_aff_diff"] = affinity_diff(sa, sb, side_a)
    if pd.isna(rec["side_aff_diff"]):
        rec["side_aff_diff"] = 0.0
    fa = float(np.mean(sa.last10)) if sa.last10 else 0.5
    fb_ = float(np.mean(sb.last10)) if sb.last10 else 0.5
    rec["form_diff"] = fa - fb_
    # side edge: team A on blue by default (neutral venue -> user can flip with --side-a)
    rec["_sa"] = sa
    rec["_sb"] = sb
    rec["_side_a"] = side_a
    rec["_frame"] = frame
    return rec


def predict_proba_game(bundle: dict, rec: dict, tpi_a: dict, tpi_b: dict,
                       patch_edge: float, region_edge: float,
                       pedigree_a: float, pedigree_b: float,
                       roster_hit_a: str | None, roster_hit_b: str | None,
                       champ_edge_val: float = 0.0,
                       tier_fit_val: float = 0.0,
                       league_val: float = 0.0,
                       temperature: float = 1.0,
                       side_val: float = 0.0) -> tuple[float, dict]:
    pipe: Pipeline = bundle["logistic"]
    feats = bundle.get("features", MODEL_FEATURES)
    X = pd.DataFrame([{f: rec.get(f, 0.0) for f in feats}])
    logit_base = float(pipe.decision_function(X)[0])
    # TPI layer (spec: discount new rosters -> zero out positive TPI until proven)
    tw, pw, rw = bundle["tpi_weight"], bundle["patch_weight"], bundle["region_weight"]
    tpi_diff = float(tpi_a.get("tpi", 0.0) - tpi_b.get("tpi", 0.0))
    if roster_hit_a and tpi_diff > 0:
        tpi_diff = min(0.0, tpi_diff)  # discount, never boost
    if roster_hit_b and tpi_diff < 0:
        tpi_diff = max(0.0, tpi_diff)
    logit = logit_base + tw * tpi_diff + pw * patch_edge + rw * region_edge
    logit += pedigree_a - pedigree_b
    logit += champ_edge_val + tier_fit_val + league_val
    # roster discount: shrink toward 0 (uncertainty) + small penalty to changed team
    if roster_hit_a:
        logit = logit * 0.9 - 0.08
    if roster_hit_b:
        logit = logit * 0.9 + 0.08
    # learned side edge for game 1 (replaces the old fixed ±0.10 guess)
    logit += side_val
    side_coef = side_val
    temp = temperature if temperature and 0.7 <= temperature <= 1.6 else 1.0
    p = sigmoid(logit / temp)
    parts = {"logit_base": logit_base, "tpi_diff": tpi_diff,
             "patch_edge": patch_edge, "region_edge": region_edge,
             "pedigree": pedigree_a - pedigree_b, "side": side_coef, "logit": logit,
             "champ": champ_edge_val, "tier_fit": tier_fit_val,
             "league": league_val, "temp": temp}
    return float(np.clip(p, 0.001, 0.999)), parts


def contributions(bundle: dict, rec: dict, parts: dict, p: float) -> list[tuple[str, float]]:
    """Per-feature pp contributions on the scaled space, plus named layers."""
    pipe: Pipeline = bundle["logistic"]
    lr = pipe.named_steps["lr"]
    feats = bundle.get("features", MODEL_FEATURES)
    X = pd.DataFrame([{f: rec.get(f, 0.0) for f in feats}])
    try:
        Xs = pipe[:-1].transform(X)[0]
    except Exception:
        Xs = np.zeros(len(feats))
    coefs = lr.coef_[0]
    items = [(f, float(c * v * p * (1 - p) * 100)) for f, c, v in
             zip(feats, coefs, Xs)]
    items.append(("TPI_layer", float(bundle["tpi_weight"] * parts["tpi_diff"] * p * (1 - p) * 100)))
    items.append(("patch/region/pedigree",
                  float((bundle["patch_weight"] * parts["patch_edge"]
                         + bundle["region_weight"] * parts["region_edge"]
                         + parts["pedigree"] + parts["side"]) * p * (1 - p) * 100)))
    items.append(("champ_layer",
                  float((parts.get("champ", 0.0) + parts.get("tier_fit", 0.0)) * p * (1 - p) * 100)))
    items.append(("league_prior",
                  float(parts.get("league", 0.0) * p * (1 - p) * 100)))
    return sorted(items, key=lambda t: -abs(t[1]))


def predict(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model)
    if bundle.get("model_kind") != MODEL_KIND:
        raise SystemExit("Retrain with lol_worlds.py train.")
    con = connect(args.db)
    try:
        games = load_games(con)
    finally:
        con.close()
    as_of = pd.Timestamp(args.as_of, tz="UTC") if args.as_of else games["date"].max() + pd.Timedelta(nanoseconds=1)
    teams = sorted(games["team"].unique())
    A = find_team(args.team_a, teams)
    B = find_team(args.team_b, teams)
    if A == B:
        raise SystemExit("Choose two different teams.")
    hist = games[games["date"] < as_of].copy()
    if hist.empty:
        raise SystemExit("No cached games before --as-of.")
    rec = _prematch_row(hist, None, A, B, as_of, bundle["ewma_alpha"],
                        bundle["k_elo"], bundle["side_adv"], args.side_a,
                        bundle.get("patch_decay", True))
    sa, sb = rec["_sa"], rec["_sb"]
    if sa.games < bundle["min_history"] or sb.games < bundle["min_history"]:
        print(f"WARNING: thin baseline history ({A}:{sa.games}, {B}:{sb.games} < {bundle['min_history']}).")
    tpi = bundle.get("tpi", {})
    tpi_a = tpi.get(A, {"tpi": 0.0, "intl_n": 0, "flag": "THIN_SAMPLE"})
    tpi_b = tpi.get(B, {"tpi": 0.0, "intl_n": 0, "flag": "THIN_SAMPLE"})
    # refresh TPI at prediction time from cached games (frozen weights, fresh data)
    fresh_tpi = compute_tpi(hist, as_of, 1095)
    tpi_a = fresh_tpi.get(A, tpi_a)
    tpi_b = fresh_tpi.get(B, tpi_b)
    h2h = {(tuple(k.split("|"))): v for k, v in bundle.get("region_h2h", {}).items()}
    ra = bundle.get("team_region", {}).get(A, "OTHER")
    rb = bundle.get("team_region", {}).get(B, "OTHER")
    region_edge = float(h2h.get((ra, rb), 0.0))
    tiers = load_patch_tiers(Path(args.patch_tiers) if args.patch_tiers else Path("patch_tiers.csv"))
    patch_map = tiers.get(args.patch, {}) if args.patch else {}
    patch_edge = 0.0  # manual --patch-edge overrides the data-driven patchform below
    if args.patch_edge is not None:
        patch_edge = float(args.patch_edge)
    # Champion layer: comfort + patch-form + pool depth (fresh at prediction time).
    # Manual tier-fit adds on top when patch_tiers.csv covers the patch.
    champ_stats = compute_champ_stats(hist, as_of, args.patch)
    ca = champ_stats.get(A, {})
    cb = champ_stats.get(B, {})
    champ_on = not bundle.get("no_champ", False) and not getattr(args, "no_champ", False)
    champ_val = champ_edge(ca, cb, bundle) if champ_on else 0.0
    tier_fit_val = 0.0
    if patch_map and champ_on:
        pool_a = team_champ_pool(hist, A, as_of)
        pool_b = team_champ_pool(hist, B, as_of)
        tier_fit_val = float(np.clip(
            0.5 * (patch_fit_score(pool_a, patch_map) - patch_fit_score(pool_b, patch_map)),
            -0.2, 0.2))
    pedig_a = pedigree_bonus(hist, A, as_of)
    pedig_b = pedigree_bonus(hist, B, as_of)
    roster = load_roster_discounts(Path(args.rosters) if args.rosters else Path("rosters.csv"),
                                   hist, as_of)
    hit_a = roster.get(A)
    hit_b = roster.get(B)
    # League prior: domestic-league strength offsets, fresh data, weight 1.0 (Elo units).
    league_on = not bundle.get("no_league", False) and not getattr(args, "no_league", False)
    la_name, lb_name = sa.dom_league, sb.dom_league
    off_a = off_b = 0.0
    na = nb = 0
    if league_on:
        # reuse the replay frame built inside _prematch_row (no second replay)
        fresh_off = compute_league_offsets(hist, rec.get("_frame"), as_of, bundle["side_adv"])
        off_a = float(fresh_off.get(la_name, {}).get("off", 0.0))
        off_b = float(fresh_off.get(lb_name, {}).get("off", 0.0))
        na = int(fresh_off.get(la_name, {}).get("n", 0))
        nb = int(fresh_off.get(lb_name, {}).get("n", 0))
    league_val = (off_a - off_b) / 400.0
    # Learned side edge for game 1 (patch-aware, falls back to global).
    side_on = not bundle.get("no_side", False) and not getattr(args, "no_side", False)
    side_val, side_src = 0.0, "off"
    if side_on:
        fresh_edges = compute_side_edges(hist, as_of)
        edge, side_src = side_edge_for(fresh_edges, args.patch)
        s = 1.0 if str(args.side_a).lower() == "blue" else -1.0
        side_val = edge / 2.0 * s
    temp = float(bundle.get("temperature", 1.0))
    pg, parts = predict_proba_game(bundle, rec, tpi_a, tpi_b, patch_edge, region_edge,
                                   pedig_a, pedig_b, hit_a, hit_b,
                                   champ_val, tier_fit_val, league_val, temp, side_val)
    # prep-time uncertainty: long gap since intl -> shrink toward 0.5
    now_gap_a = (as_of - sa.last_intl_date).days if sa.last_intl_date is not None else 999
    now_gap_b = (as_of - sb.last_intl_date).days if sb.last_intl_date is not None else 999
    gap = min(now_gap_a, now_gap_b)
    shrink = min(0.12, max(0.0, (gap - 90) / 365) * 0.10) if gap != 999 else 0.08
    pg_adj = float(0.5 + (pg - 0.5) * (1 - shrink))
    p_series = series_prob(pg_adj, args.best_of, bundle.get("bo5_boost", 0.12))
    if args.best_of == 1:
        p_final, label = pg_adj, "Bo1 game"
    else:
        p_final, label = p_series, f"Bo{args.best_of} series"
    # drivers + risk
    contribs = contributions(bundle, rec, parts, pg_adj)
    drivers = contribs[:3]
    risks = [c for c in contribs if (c[1] < 0 if p_final >= 0.5 else c[1] > 0)]
    risk = risks[0] if risks else contribs[-1]
    # confidence
    thin = (tpi_a.get("intl_n", 0) < 5) or (tpi_b.get("intl_n", 0) < 5)
    low_hist = sa.games < 20 or sb.games < 20
    conf = "LOW" if (thin or low_hist or hit_a or hit_b or gap > 180) else ("MEDIUM" if gap > 90 else "HIGH")
    fair_a = 1 / p_final
    fair_b = 1 / (1 - p_final)
    print(f"{A} vs {B} — {label} (patch {args.patch or 'n/a'}, as of {pd.Timestamp(as_of).date()})")
    print(f"Baseline: {A} Elo {sa.elo:.0f} ({sa.games}g) vs {B} Elo {sb.elo:.0f} ({sb.games}g) | "
          f"elo_diff {rec['elo_diff']:+.2f}")
    print(f"TPI: {A} {tpi_a.get('tpi',0):+.3f} [{tpi_a.get('flag')}, intl_n={tpi_a.get('intl_n',0)}] vs "
          f"{B} {tpi_b.get('tpi',0):+.3f} [{tpi_b.get('flag')}, intl_n={tpi_b.get('intl_n',0)}]")
    if hit_a:
        print(f"ROSTER DISCOUNT {A}: {hit_a} (treated as discount, TPI upside zeroed)")
    if hit_b:
        print(f"ROSTER DISCOUNT {B}: {hit_b} (treated as discount, TPI upside zeroed)")
    print(f"Context: region {ra}vs{rb} edge {region_edge:+.2f} lo | patch edge {patch_edge:+.2f} | "
          f"pedigree {parts['pedigree']:+.3f} | prep gap {gap}d (shrink {shrink:.0%})")
    if side_on:
        print(f"Side: A on {args.side_a}, edge {side_val:+.3f} lo ({side_src})")
    if champ_on:
        print(f"Champs: comfort {ca.get('comfort', 0):+.3f} vs {cb.get('comfort', 0):+.3f} | "
              f"patch-form {ca.get('patchform', 0):+.3f} vs {cb.get('patchform', 0):+.3f} | "
              f"depth {ca.get('depth', 0):.2f} vs {cb.get('depth', 0):.2f} | "
              f"edge {champ_val + tier_fit_val:+.3f} lo")
    if league_on:
        print(f"League prior: {la_name} {off_a:+.0f} (n={na}) vs {lb_name} {off_b:+.0f} (n={nb}) | "
              f"edge {league_val:+.3f} lo | temp {parts.get('temp', 1.0):.3f}")
    print(f"\nWin probability {A}: {p_final:.1%}  (game prob {pg_adj:.1%}) | fair odds {fair_a:.2f} vs {fair_b:.2f}")
    print("Top drivers:")
    for name, pp in drivers:
        print(f"  {name:22s} {pp:+.1f}pp")
    print(f"Biggest flip risk: {risk[0]} ({risk[1]:+.1f}pp against). "
          f"E.g. patch/read or early-game variance flipping GD15/FB.")
    print(f"Confidence: {conf}" + (" — SAMPLE TOO THIN, treat as lean not bet." if thin else "."))
    ev_a = ev_b = None
    if args.odds:
        if len(args.odds) != 2 or any(o <= 1 for o in args.odds):
            raise SystemExit("--odds needs two decimal odds > 1: ODD_A ODD_B")
        oa, ob = args.odds
        ev_a, ev_b = p_final * oa - 1, (1 - p_final) * ob - 1
        kelly_a = max(0.0, (p_final * (oa - 1) - (1 - p_final)) / (oa - 1)) if oa > 1 else 0.0
        kelly_b = max(0.0, ((1 - p_final) * (ob - 1) - p_final) / (ob - 1)) if ob > 1 else 0.0
        print(f"\nBook: {oa:.2f} vs {ob:.2f} | EV {A} {ev_a:+.1%} (1/4 Kelly {kelly_a/4:.1%}) | "
              f"EV {B} {ev_b:+.1%} (1/4 Kelly {kelly_b/4:.1%})")
        if ev_a <= 0 and ev_b <= 0:
            print("No edge at these odds.")
    print(f"\nModel cutoff: {bundle.get('cutoff')} | retrain after downloading new games.")
    if args.log_trade is not None:
        _log_trade(args, A, B, as_of, p_final, fair_a, fair_b, ev_a, ev_b,
                   drivers, risk, conf, tpi_a, tpi_b, sa, sb)


TRADE_COLUMNS = ["id", "timestamp", "team_a", "team_b", "best_of", "patch", "as_of",
                 "p_a", "fair_a", "fair_b", "book_a", "book_b", "ev_a", "ev_b",
                 "edge_pp", "stake", "pick", "drivers", "risk", "confidence",
                 "tpi_a", "tpi_b", "tpi_flag_a", "tpi_flag_b",
                 "elo_a", "elo_b", "result", "profit"]


def _log_trade(args, A, B, as_of, p_final, fair_a, fair_b, ev_a, ev_b,
               drivers, risk, conf, tpi_a, tpi_b, sa, sb) -> None:
    """Append one row to the trade log. Stakes are flat; only +EV picks get staked."""
    if args.odds is None or ev_a is None or ev_b is None:
        raise SystemExit("--log-trade requires --odds ODD_A ODD_B (no EV without book odds).")
    if not math.isfinite(args.stake) or args.stake <= 0:
        raise SystemExit("--stake must be a finite positive number.")
    stake = float(args.stake)
    best_ev = max(ev_a, ev_b)
    if best_ev > args.min_ev:
        pick = "A" if ev_a >= ev_b else "B"
    else:
        pick = "NO_BET"
    if pick != "NO_BET" and conf == "LOW" and not args.force:
        print("LOW confidence — not logging a staked pick (use --force to log anyway). "
              f"Best EV {best_ev:+.1%} kept as a lean, not a bet.")
        pick = "NO_BET"
        blocked_low = True
    else:
        blocked_low = False
    log_path = Path(args.log_trade)
    if log_path.is_file():
        try:
            existing = pd.read_csv(log_path)
            next_id = int(existing["id"].max()) + 1 if len(existing) else 1
        except Exception:
            raise SystemExit(f"Could not read existing log: {log_path}")
    else:
        existing = None
        next_id = 1
    row = {
        "id": next_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "team_a": A, "team_b": B, "best_of": args.best_of,
        "patch": args.patch or "", "as_of": pd.Timestamp(as_of).date().isoformat(),
        "p_a": round(p_final, 4), "fair_a": round(fair_a, 2), "fair_b": round(fair_b, 2),
        "book_a": args.odds[0], "book_b": args.odds[1],
        "ev_a": round(ev_a, 4), "ev_b": round(ev_b, 4),
        "edge_pp": round(best_ev * 100, 2),
        "stake": stake if pick != "NO_BET" else 0.0,
        "pick": pick,
        "drivers": "; ".join(f"{n}:{v:+.1f}pp" for n, v in drivers),
        "risk": f"{risk[0]}:{risk[1]:+.1f}pp",
        "confidence": conf,
        "tpi_a": round(float(tpi_a.get("tpi", 0.0)), 3),
        "tpi_b": round(float(tpi_b.get("tpi", 0.0)), 3),
        "tpi_flag_a": tpi_a.get("flag", "?"), "tpi_flag_b": tpi_b.get("flag", "?"),
        "elo_a": round(float(sa.elo), 1), "elo_b": round(float(sb.elo), 1),
        "result": "VOID" if pick == "NO_BET" else "UNKNOWN",
        "profit": 0.0 if pick == "NO_BET" else "",
    }
    frame = pd.DataFrame([[row[c] for c in TRADE_COLUMNS]], columns=TRADE_COLUMNS)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(log_path, mode="a", header=(existing is None), index=False)
    if pick == "NO_BET":
        reason = ("blocked by LOW confidence" if blocked_low
                  else f"best EV {best_ev:+.1%} < {args.min_ev:.0%} threshold")
        print(f"Logged id={next_id} as NO_BET ({reason}).")
    else:
        side = A if pick == "A" else B
        print(f"Logged id={next_id}: {side} @ {args.odds[0 if pick == 'A' else 1]:.2f}, "
              f"stake {stake:g}, EV {best_ev:+.1%} -> {log_path}")


def trades_report(args: argparse.Namespace) -> None:
    log_path = Path(args.log)
    if not log_path.is_file():
        raise SystemExit(f"No trade log yet: {log_path}")
    df = pd.read_csv(log_path)
    if df.empty:
        raise SystemExit("Trade log is empty.")
    staked = df[df["pick"].isin(["A", "B"])].copy()
    settled = staked[staked["result"].isin(["A", "B"])].copy()
    pending = staked[~staked["result"].isin(["A", "B"])].copy()
    print(f"Trade log: {log_path} | rows {len(df)} | staked {len(staked)} "
          f"| settled {len(settled)} | pending {len(pending)}")
    if settled.empty:
        print("No settled picks yet. Settle with: trades settle --id <id> --result A|B|VOID")
        return
    settled["hit"] = (settled["pick"] == settled["result"]).astype(int)
    profit = pd.to_numeric(settled["profit"], errors="coerce").fillna(0.0)
    staked_sum = pd.to_numeric(settled["stake"], errors="coerce").fillna(0.0).sum()
    print(f"Hit rate: {settled['hit'].mean():.1%} ({int(settled['hit'].sum())}/{len(settled)})")
    print(f"Total profit: {profit.sum():+.2f} | Staked: {staked_sum:.2f} | "
          f"ROI: {(profit.sum() / staked_sum if staked_sum else 0):+.1%}")
    if "edge_pp" in settled:
        print(f"Avg logged edge: {pd.to_numeric(settled['edge_pp'], errors='coerce').mean():+.1f}pp")
    def pick_prob(r):
        return float(r["p_a"]) if r["pick"] == "A" else 1.0 - float(r["p_a"])
    settled["pick_prob"] = settled.apply(pick_prob, axis=1)
    print(f"Avg model prob on picked side: {settled['pick_prob'].mean():.1%} "
          f"(vs {settled['hit'].mean():.1%} actual — above = overconfident)")
    if "confidence" in settled:
        print("\nBy confidence:")
        for c, g in settled.groupby("confidence"):
            pr = pd.to_numeric(g["profit"], errors="coerce").fillna(0.0).sum()
            print(f"  {c:8s} n={len(g):3d} hit={g['hit'].mean():.0%} profit={pr:+.2f}")
    if len(settled) < 20:
        print("\nNOTE: fewer than 20 settled picks — too early to judge the model.")


def trades_settle(args: argparse.Namespace) -> None:
    log_path = Path(args.log)
    if not log_path.is_file():
        raise SystemExit(f"No trade log yet: {log_path}")
    df = pd.read_csv(log_path)
    if args.id not in df["id"].to_numpy():
        raise SystemExit(f"id={args.id} not found in {log_path}")
    i = df.index[df["id"] == args.id][0]
    pick = str(df.at[i, "pick"])
    if pick == "NO_BET":
        raise SystemExit(f"id={args.id} is a NO_BET row — nothing to settle.")
    result = args.result.upper()
    if result not in ("A", "B", "VOID"):
        raise SystemExit("--result must be A, B or VOID (A = team_a won).")
    stake = float(df.at[i, "stake"])
    if result == "VOID":
        df.at[i, "result"] = "VOID"
        df.at[i, "profit"] = 0.0
    else:
        odd = float(df.at[i, "book_a" if result == "A" else "book_b"])
        df.at[i, "result"] = result
        df.at[i, "profit"] = round(stake * (odd - 1), 2) if pick == result else round(-stake, 2)
    df.to_csv(log_path, index=False)
    print(f"Settled id={args.id}: pick {pick} vs result {result} -> profit {df.at[i, 'profit']:+.2f}")


def ratings(args: argparse.Namespace) -> None:
    bundle = joblib.load(args.model) if Path(args.model).is_file() else None
    con = connect(args.db)
    try:
        games = load_games(con)
    finally:
        con.close()
    cutoff = games["date"].max() + pd.Timedelta(nanoseconds=1)
    frame, states, team_region = build_examples(
        games, bundle["ewma_alpha"] if bundle else 0.25, 0,
        bundle["k_elo"] if bundle else 20.0, bundle["side_adv"] if bundle else 35.0)
    tpi = compute_tpi(games, cutoff)
    filt = args.region.upper() if args.region else None
    rows = []
    for team, st in states.items():
        if filt and filt not in (team_region.get(team, ""), team):
            if args.region.casefold() not in team.casefold():
                continue
        t = tpi.get(team, {})
        rows.append((team, st.elo, st.games, st.ewma.get("gd15"), st.ewma.get("xpd15"),
                     t.get("tpi", 0.0), t.get("flag", "NO_DATA"), t.get("intl_n", 0),
                     team_region.get(team, "?")))
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"Ratings through {games['date'].max().date()} ({len(games)//2} games)")
    print(f"{'Team':22s} {'Elo':>6s} {'G':>4s} {'GD15':>8s} {'XPD15':>8s} {'TPI':>6s} {'Flag':>14s} {'INTL':>4s} {'RG':>4s}")
    for r in rows[: args.top]:
        print(f"{r[0]:22s} {r[1]:6.0f} {r[2]:4d} "
              f"{(f'{r[3]:+.0f}' if r[3] is not None else 'n/a'):>8s} "
              f"{(f'{r[4]:+.0f}' if r[4] is not None else 'n/a'):>8s} "
              f"{r[5]:+6.2f} {r[6]:>14s} {r[7]:4d} {r[8]:>4s}")


def backtest(args: argparse.Namespace) -> None:
    con = connect(args.db)
    try:
        games = load_games(con)
    finally:
        con.close()
    train_end = _as_utc(args.train_through)
    test_from = _as_utc(args.test_from) if args.test_from else train_end
    test_to = _as_utc(args.test_to) if args.test_to else games["date"].max()
    frame, _, _ = build_examples(games, args.ewma_alpha, args.min_history, args.k_elo, args.side_adv,
                                   not getattr(args, "no_patchdecay", False))
    feats = feature_list(args)
    for f in feats:
        if f not in frame.columns:
            frame[f] = 0.0
    tr = frame[(frame["eligible"]) & (frame["date"] < train_end)].copy()
    te = frame[(frame["date"] >= test_from) & (frame["date"] <= test_to)].copy()
    # test rows need min history at their date (recompute eligibility is already chronological, keep all with features)
    te = te.dropna(subset=feats, how="all")
    if len(tr) < 50 or len(te) < 10:
        raise SystemExit(f"Not enough data: train {len(tr)}, test {len(te)}.")
    for f in feats:
        tr[f] = tr[f].fillna(0.0)
        te[f] = te[f].fillna(0.0)
    model = fit_logistic(tr[feats], tr["win_a"].astype(int),
                         time_weights(tr["date"], train_end, args.half_life), args.regularization)
    p = np.clip(model.predict_proba(te[feats])[:, 1], 1e-6, 1 - 1e-6)
    # add frozen TPI + champion layers (fitted through train_end only — no test leakage)
    tpi_tr = compute_tpi(games, train_end)
    tw = 2.0
    logit = np.log(p / (1 - p))
    adj = np.array([tpi_tr.get(a, {}).get("tpi", 0.0) - tpi_tr.get(b, {}).get("tpi", 0.0)
                    for a, b in zip(te["team_a"], te["team_b"])])
    champ_w = {"champ_w_comfort": 0.8, "champ_w_patch": 0.8, "champ_w_depth": 0.4}
    champ_adj = np.zeros(len(te))
    if not getattr(args, "no_champ", False):
        pre = games[games["date"] < train_end].copy()
        per_patch: dict[str, dict] = {}
        for patch in te["patch"].astype(str).unique():
            per_patch[patch] = compute_champ_stats(pre, train_end, patch if patch else None)
        for i, (a, b, patch) in enumerate(zip(te["team_a"], te["team_b"], te["patch"].astype(str))):
            cs = per_patch.get(patch, {})
            ca, cb = cs.get(a, {}), cs.get(b, {})
            try:
                champ_adj[i] = champ_edge(ca, cb, champ_w)
            except Exception:
                champ_adj[i] = 0.0
        covered = float(np.mean(champ_adj != 0.0)) if len(champ_adj) else 0.0
        print(f"Champ layer: nonzero edge on {covered:.0%} of test games")
    p2 = 1 / (1 + np.exp(-(logit + tw * adj + champ_adj)))
    # league prior layer (frozen offsets at train_end)
    league_adj = np.zeros(len(te))
    if not getattr(args, "no_league", False):
        off_tr = compute_league_offsets(games, frame, train_end, args.side_adv)
        dom_all = team_domestic_leagues(games[games["date"] < train_end])
        la = te["league_a"].fillna(te["team_a"].map(dom_all)).fillna("UNK") \
            if "league_a" in te.columns else te["team_a"].map(dom_all).fillna("UNK")
        lb = te["league_b"].fillna(te["team_b"].map(dom_all)).fillna("UNK") \
            if "league_b" in te.columns else te["team_b"].map(dom_all).fillna("UNK")
        league_adj = np.array([off_tr.get(str(x), {}).get("off", 0.0)
                               - off_tr.get(str(y_), {}).get("off", 0.0)
                               for x, y_ in zip(la, lb)]) / 400.0
        top_off = sorted(off_tr.items(), key=lambda t: -t[1]["off"])[:6]
        print("League offsets: " + ", ".join(f"{lg} {v['off']:+.0f}(n={v['n']})" for lg, v in top_off))
    # learned side edge per test game (frozen edges at train_end, actual sides)
    side_adj = np.zeros(len(te))
    if not getattr(args, "no_side", False):
        edges_tr = compute_side_edges(games, train_end)
        ge = edges_tr["global"]["edge"]
        pe = edges_tr["patches"]
        s_side = np.where(te["side_a"].astype(str).str.lower() == "blue", 1.0, -1.0)
        e_side = np.array([pe.get(str(patch), {}).get("edge", ge)
                           for patch in te["patch"].astype(str)])
        side_adj = e_side / 2.0 * s_side
        print(f"Side edge: global {ge:+.3f} lo (blue {edges_tr['global']['p']:.1%}, "
              f"n={edges_tr['global']['n']})")
    logit_full = logit + tw * adj + champ_adj + league_adj + side_adj
    p_raw = 1 / (1 + np.exp(-logit_full))
    # temperature from inner holdout of tr (layers frozen at inner cutoff)
    n_tr = len(tr)
    cut_i = min(n_tr - 1, max(1, int(n_tr * 0.8)))
    cut_d = tr.iloc[cut_i]["date"]
    tr2, ho = tr[tr["date"] < cut_d], tr[tr["date"] >= cut_d]
    temperature = 1.0
    if len(tr2) >= 50 and len(ho) >= 10:
        m_in = fit_logistic(tr2[feats], tr2["win_a"].astype(int),
                            time_weights(tr2["date"], cut_d, args.half_life), args.regularization)
        ph = np.clip(m_in.predict_proba(ho[feats])[:, 1], 1e-6, 1 - 1e-6)
        lh = np.log(ph / (1 - ph))
        tpi_h = compute_tpi(games, cut_d)
        ah = np.array([tpi_h.get(a, {}).get("tpi", 0.0) - tpi_h.get(b, {}).get("tpi", 0.0)
                       for a, b in zip(ho["team_a"], ho["team_b"])])
        ch_h = np.zeros(len(ho))
        if not getattr(args, "no_champ", False):
            pre_h = games[games["date"] < cut_d].copy()
            pph: dict[str, dict] = {}
            for patch in ho["patch"].astype(str).unique():
                pph[patch] = compute_champ_stats(pre_h, cut_d, patch if patch else None)
            for i, (a, b, patch) in enumerate(zip(ho["team_a"], ho["team_b"], ho["patch"].astype(str))):
                cs = pph.get(patch, {})
                try:
                    ch_h[i] = champ_edge(cs.get(a, {}), cs.get(b, {}), champ_w)
                except Exception:
                    ch_h[i] = 0.0
        lg_h = np.zeros(len(ho))
        if not getattr(args, "no_league", False):
            off_h = compute_league_offsets(games, frame, cut_d, args.side_adv)
            dom_h = team_domestic_leagues(games[games["date"] < cut_d])
            lah = ho["league_a"].fillna(ho["team_a"].map(dom_h)).fillna("UNK") \
                if "league_a" in ho.columns else ho["team_a"].map(dom_h).fillna("UNK")
            lbh = ho["league_b"].fillna(ho["team_b"].map(dom_h)).fillna("UNK") \
                if "league_b" in ho.columns else ho["team_b"].map(dom_h).fillna("UNK")
            lg_h = np.array([off_h.get(str(x), {}).get("off", 0.0)
                             - off_h.get(str(y_), {}).get("off", 0.0)
                             for x, y_ in zip(lah, lbh)]) / 400.0
        sd_h = np.zeros(len(ho))
        if not getattr(args, "no_side", False):
            edges_h = compute_side_edges(games, cut_d)
            geh = edges_h["global"]["edge"]
            peh = edges_h["patches"]
            s_ho = np.where(ho["side_a"].astype(str).str.lower() == "blue", 1.0, -1.0)
            e_ho = np.array([peh.get(str(patch), {}).get("edge", geh)
                             for patch in ho["patch"].astype(str)])
            sd_h = e_ho / 2.0 * s_ho
        temperature = fit_temperature(lh + tw * ah + ch_h + lg_h + sd_h,
                                      ho["win_a"].to_numpy().astype(int))
    p2 = 1 / (1 + np.exp(-np.clip(logit_full / temperature, -500, 500)))
    print(f"Features ({len(feats)}): {', '.join(feats)}"
          + ("" if not getattr(args, "no_champ", False) else " | champ layer OFF")
          + ("" if not getattr(args, "no_league", False) else " | league layer OFF")
          + ("" if not getattr(args, "no_side", False) else " | side layer OFF")
          + f" | temperature {temperature:.3f}")
    y = te["win_a"].to_numpy().astype(int)
    praw = 1 / (1 + np.exp(-logit_full))
    praw = np.clip(praw, 1e-6, 1 - 1e-6)
    print(f"Untempered: acc {(((praw >= 0.5).astype(int)) == y).mean():.2%} | "
          f"logloss {log_loss(y, praw):.4f} | brier {np.mean((praw-y)**2):.4f}")
    acc = float(((p2 >= 0.5).astype(int) == y).mean())
    ll = float(log_loss(y, np.clip(p2, 1e-6, 1 - 1e-6)))
    br = float(np.mean((p2 - y) ** 2))
    print(f"Train: {len(tr)} games (< {train_end.date()}); Test: {len(te)} games "
          f"[{test_from.date()}..{test_to.date()}] (ALL leagues, not Worlds-only)")
    print(f"Accuracy {(acc):.2%} | log loss {ll:.4f} | Brier {br:.4f}")
    print(f"Baseline check — favorite (Elo) acc: "
          f"{((te['elo_diff'] >= 0).astype(int).to_numpy() == y).mean():.2%}")
    # Slice scoreboard: overall + intl + each major domestic league.
    # A config must prove itself where it will be used, not just on average.
    print(f"\n{'Slice':8s} {'n':>5s} {'acc':>7s} {'logloss':>8s} {'brier':>7s}")
    print(f"{'ALL':8s} {len(te):5d} {acc:7.2%} {ll:8.4f} {br:7.4f}")
    slices: list[tuple[str, pd.DataFrame]] = [
        ("INTL", te[te["league"].map(is_intl_league)]),
        ("LCK", te[te["league"] == "LCK"]),
        ("LPL", te[te["league"] == "LPL"]),
        ("LEC", te[te["league"] == "LEC"]),
        ("LCS", te[te["league"] == "LCS"]),
        ("LTA", te[te["league"] == "LTA"]),
        ("LCP", te[te["league"] == "LCP"]),
    ]
    for name, part in slices:
        if len(part) < 10:
            print(f"{name:8s} {len(part):5d}   -- too few --")
            continue
        yi = part["win_a"].to_numpy().astype(int)
        pi = p2[te["gameid"].isin(part["gameid"]).to_numpy()]
        print(f"{name:8s} {len(part):5d} {(((pi >= 0.5).astype(int)) == yi).mean():7.2%} "
              f"{log_loss(yi, np.clip(pi, 1e-6, 1-1e-6)):8.4f} {np.mean((pi-yi)**2):7.4f}")
    cal = pd.DataFrame({"p": p2, "y": y})
    cal["bin"] = pd.cut(cal["p"], bins=[0, .4, .5, .6, 1.0])
    print("\nCalibration (prob bin -> hit rate):")
    print(cal.groupby("bin", observed=True).agg(n=("y", "size"), hit=("y", "mean"), avg_p=("p", "mean")).to_string())
    if args.output:
        out = te[["date", "gameid", "team_a", "team_b", "league", "win_a"]].copy()
        out["p_a"] = p2
        out["pred_a"] = (p2 >= 0.5).astype(int)
        out["correct"] = (out["pred_a"] == out["win_a"]).astype(int)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Fetch Oracle's Elixir CSVs (or import local --csv)")
    d.add_argument("--from-season", type=int, default=2021)
    d.add_argument("--to-season", type=int, default=2025)
    d.add_argument("--csv", nargs="*", default=None, help="Local Oracle CSV file(s) to import")
    d.add_argument("--pause", type=float, default=0.5)
    d.set_defaults(func=download)

    t = sub.add_parser("train", help="Fit Elo + logistic + TPI bundle")
    t.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    t.add_argument("--train-through", default=None, help="ISO date upper bound, e.g. 2024-06-01")
    t.add_argument("--ewma-alpha", type=float, default=0.25)
    t.add_argument("--min-history", type=int, default=5)
    t.add_argument("--k-elo", type=float, default=20.0)
    t.add_argument("--side-adv", type=float, default=35.0)
    t.add_argument("--regularization", type=float, default=0.5, help="1/C for logistic")
    t.add_argument("--half-life", type=float, default=365.0, help="days; 0 = equal weights")
    t.add_argument("--test-fraction", type=float, default=0.2)
    t.add_argument("--tpi-window-days", type=int, default=1095)
    t.add_argument("--tpi-weight", type=float, default=2.0)
    t.add_argument("--patch-weight", type=float, default=0.8)
    t.add_argument("--region-weight", type=float, default=1.0)
    t.add_argument("--bo5-boost", type=float, default=0.12)
    t.add_argument("--no-firsts", action="store_true", help="Exclude first-objective features")
    t.add_argument("--no-lanes", action="store_true", help="Exclude per-role lane features")
    t.add_argument("--no-champ", action="store_true", help="Exclude champion layer")
    t.add_argument("--no-league", action="store_true", help="Exclude league-strength prior")
    t.add_argument("--no-side", action="store_true", help="Exclude side edge + affinity")
    t.add_argument("--no-patchdecay", action="store_true", help="Disable patch-aware EWMA memory")
    t.set_defaults(func=train)

    r = sub.add_parser("ratings", help="Current Elo + TPI table")
    r.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    r.add_argument("--region", default=None)
    r.add_argument("--top", type=int, default=30)
    r.set_defaults(func=ratings)

    q = sub.add_parser("predict", help="P(A beats B) + fair odds + drivers + EV")
    q.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    q.add_argument("--team-a", required=True)
    q.add_argument("--team-b", required=True)
    q.add_argument("--best-of", type=int, default=5, choices=[1, 3, 5])
    q.add_argument("--patch", default=None)
    q.add_argument("--patch-tiers", default=None)
    q.add_argument("--rosters", default=None)
    q.add_argument("--patch-edge", type=float, default=None, help="Manual patch edge in log-odds [-0.5,0.5], A-minus-B")
    q.add_argument("--side-a", default="blue", choices=["blue", "red"])
    q.add_argument("--as-of", default=None)
    q.add_argument("--odds", nargs=2, type=float, default=None, metavar=("ODD_A", "ODD_B"))
    q.add_argument("--log-trade", type=Path, default=None,
                   help="Append this pick to a trade log CSV (requires --odds)")
    q.add_argument("--stake", type=float, default=10.0, help="Flat stake per bet (default: 10)")
    q.add_argument("--min-ev", type=float, default=0.03,
                   help="Minimum best-side EV to log a staked pick (default: 0.03)")
    q.add_argument("--force", action="store_true",
                   help="Allow logging a staked pick even at LOW confidence")
    q.add_argument("--no-champ", action="store_true", help="Exclude champion layer")
    q.add_argument("--no-league", action="store_true", help="Exclude league-strength prior")
    q.add_argument("--no-side", action="store_true", help="Exclude side edge + affinity")
    q.set_defaults(func=predict)

    b = sub.add_parser("backtest", help="Freeze at --train-through, evaluate later window (all matches)")
    b.add_argument("--train-through", required=True)
    b.add_argument("--test-from", default=None)
    b.add_argument("--test-to", default=None)
    b.add_argument("--ewma-alpha", type=float, default=0.25)
    b.add_argument("--min-history", type=int, default=5)
    b.add_argument("--k-elo", type=float, default=20.0)
    b.add_argument("--side-adv", type=float, default=35.0)
    b.add_argument("--regularization", type=float, default=0.5)
    b.add_argument("--half-life", type=float, default=365.0)
    b.add_argument("--output", type=Path, default=None)
    b.add_argument("--no-firsts", action="store_true", help="Exclude first-objective features")
    b.add_argument("--no-lanes", action="store_true", help="Exclude per-role lane features")
    b.add_argument("--no-champ", action="store_true", help="Exclude champion layer")
    b.add_argument("--no-league", action="store_true", help="Exclude league-strength prior")
    b.add_argument("--no-patchdecay", action="store_true", help="Disable patch-aware EWMA memory")
    b.add_argument("--no-side", action="store_true", help="Exclude side edge + affinity")
    b.set_defaults(func=backtest)
    tr = sub.add_parser("trades", help="Paper-trade log: report / settle")
    tr.add_argument("--log", type=Path, default=Path("trades.csv"),
                    help="Trade log CSV (default: trades.csv)")
    tsub = tr.add_subparsers(dest="trades_cmd", required=True)
    rep = tsub.add_parser("report", help="Hit rate, ROI, calibration, by-confidence breakdown")
    rep.add_argument("--log", type=Path, default=argparse.SUPPRESS)
    rep.set_defaults(func=trades_report)
    stl = tsub.add_parser("settle", help="Record a result: --id <n> --result A|B|VOID")
    stl.add_argument("--log", type=Path, default=argparse.SUPPRESS)
    stl.add_argument("--id", type=int, required=True)
    stl.add_argument("--result", required=True, help="A = team_a won, B = team_b won, VOID = cancelled")
    stl.set_defaults(func=trades_settle)
    # Accept --db/--model both before and after the subcommand (like football_poisson.py).
    for cmd in (d, t, r, q, b, rep, stl):
        cmd.add_argument("--db", type=Path, default=argparse.SUPPRESS)
    return p


def resolve_paths(a: argparse.Namespace) -> argparse.Namespace:
    # argparse with SUPPRESS keeps the global default unless overridden per-command.
    if not isinstance(getattr(a, "db", None), Path):
        a.db = DEFAULT_DB
    if hasattr(a, "model") and not isinstance(a.model, Path):
        a.model = DEFAULT_MODEL
    return a


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    a = resolve_paths(parser().parse_args())
    a.func(a)
