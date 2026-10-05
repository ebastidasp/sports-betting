"""Map the captured BetPlay corner prices to the read-only Brazil cache."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from zoneinfo import ZoneInfo

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
BOGOTA = ZoneInfo("America/Bogota")
MODEL = "corners_model_brazil_2026_10_05/corners_model.joblib"
TRAINING_CUTOFF = datetime.fromisoformat("2026-10-05T14:33:21.072548+00:00")


def normalized_name(value):
    value = re.sub(r"-[A-Z]{2}$", "", value.strip(), flags=re.IGNORECASE)
    value = re.sub(r"^FC\s+", "", value, flags=re.IGNORECASE)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def printed_date(value, observed):
    normalized = normalized_name(value)
    weekdays = {"lun": 0, "mar": 1, "mie": 2, "jue": 3,
                "vie": 4, "sab": 5, "dom": 6}
    anchor = observed.astimezone(BOGOTA).date()
    if normalized in weekdays:
        return anchor + timedelta(days=(weekdays[normalized] - anchor.weekday()) % 7)
    match = re.fullmatch(r"(\d{1,2}) de oct", normalized)
    if match:
        return date(anchor.year, 10, int(match[1]))
    raise ValueError(f"Unknown printed date: {value!r}")


def printed_time(value):
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unknown printed time: {value!r}")
    hour, minute = int(match[1]), int(match[2])
    if not 1 <= hour <= 12 or not 0 <= minute < 60:
        raise ValueError(f"Invalid printed time: {value!r}")
    return hour % 12 + (12 if match[3].lower() == "p" else 0), minute


def prior_aliases():
    names, display = {}, {}
    for league, directory in ((71, "brazil_serie_a_2026_10_07_08"),
                              (72, "brazil_serie_b_2026_10_06_11")):
        previous = json.loads((ROOT / "match_predictions" / directory / "quote_snapshot.json").read_text(encoding="utf-8"))
        for fixture in previous["fixtures"]:
            for side in ("home", "away"):
                key = (league, normalized_name(fixture[side]))
                tid = int(fixture[side + "_id"])
                if key in names and names[key] != tid:
                    raise ValueError(f"Ambiguous previous team alias: {key}")
                names[key] = tid
                display[(league, tid)] = fixture[side]
    # These book descriptions differ from the previously verified team labels.
    for league, book, verified in (
        (71, "Internacional P. A.", "Internacional"),
        (71, "Athletico Paranaense-PR", "Athletico-PR"),
        (71, "Atletico Mineiro-MG", "Atletico-MG"),
    ):
        names[(league, normalized_name(book))] = names[(league, normalized_name(verified))]
    return names, display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")) if args.as_of else datetime.now(timezone.utc)
    if as_of.tzinfo is None or as_of <= TRAINING_CUTOFF:
        raise ValueError("Prediction date must be timezone-aware and later than model training.")
    raw_path = OUTPUT / "raw_quotes.json"
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    aliases, display_names = prior_aliases()
    fixtures, anomalies, seen_fixtures, seen_events = [], [], set(), set()
    database = ROOT / "brazil.sqlite3"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for league in raw["leagues"]:
            league_id = int(league["league_id"])
            observed = datetime.fromisoformat(league["observed_at"].replace("Z", "+00:00"))
            for values in league["rows"]:
                quoted = dict(zip(league["columns"], values, strict=True))
                home_id = aliases[(league_id, normalized_name(quoted["book_home"]))]
                away_id = aliases[(league_id, normalized_name(quoted["book_away"]))]
                candidates = connection.execute(
                    "SELECT fixture_id,kickoff,status,home_name,away_name FROM fixtures "
                    "WHERE league_id=? AND home_id=? AND away_id=? AND status='NS' ORDER BY kickoff",
                    (league_id, home_id, away_id),
                ).fetchall()
                candidates = [row for row in candidates if datetime.fromisoformat(row[1]) > as_of]
                local_date = printed_date(quoted["displayed_date"], observed)
                same_day = [row for row in candidates if datetime.fromisoformat(row[1]).astimezone(BOGOTA).date() == local_date]
                candidates = same_day if same_day else candidates
                if len(candidates) != 1:
                    raise ValueError(f"Unknown or ambiguous cached fixture: {quoted}, candidates={candidates}")
                fixture_id, kickoff, status, database_home, database_away = candidates[0]
                book_hour, book_minute = printed_time(quoted["displayed_time"])
                book_datetime = datetime.combine(local_date, datetime.min.time(), BOGOTA).replace(hour=book_hour, minute=book_minute)
                cached_datetime = datetime.fromisoformat(kickoff).astimezone(BOGOTA)
                mismatch = cached_datetime != book_datetime
                event = int(quoted["book_event_id"])
                if fixture_id in seen_fixtures or event in seen_events:
                    raise ValueError("Duplicate fixture or bookmaker event in the captured quotes.")
                seen_fixtures.add(fixture_id)
                seen_events.add(event)
                line = float(quoted["line"])
                if line < 0 or line % 1 != .5:
                    raise ValueError("Every corner line must be a nonnegative half-integer.")
                record = {
                    "league_id": league_id, "fixture_id": int(fixture_id),
                    "home_id": home_id, "away_id": away_id,
                    "home": display_names[(league_id, home_id)],
                    "away": display_names[(league_id, away_id)],
                    "kickoff": kickoff, "kickoff_bogota": cached_datetime.isoformat(),
                    "source_url": f"https://betplay.com.co/apuestas#/event/{event}",
                    "observed_at": league["observed_at"], "book_event_id": event,
                    "book_home": quoted["book_home"], "book_away": quoted["book_away"],
                    "database_home_name": database_home, "database_away_name": database_away,
                    "book_displayed_date": quoted["displayed_date"],
                    "book_displayed_time": quoted["displayed_time"],
                    "book_kickoff_bogota": book_datetime.isoformat(),
                    "schedule_matches_cache": not mismatch, "markets": [],
                }
                for side in ("over", "under"):
                    odds = quoted[side]
                    if not isinstance(odds, (int, float)) or odds <= 1:
                        raise ValueError("Decimal odds must exceed one.")
                    record["markets"].append({
                        "market": "Total de Tiros de Esquina", "scope": "total",
                        "side": side, "line": quoted["line"], "odds": odds,
                        "period": "full_time",
                    })
                fixtures.append(record)
                if mismatch:
                    anomalies.append({
                        "league_id": league_id, "fixture_id": int(fixture_id),
                        "book_event_id": event, "home": record["home"], "away": record["away"],
                        "book_kickoff_bogota": book_datetime.isoformat(),
                        "cached_kickoff_bogota": cached_datetime.isoformat(),
                        "cached_minus_book_minutes": (cached_datetime - book_datetime).total_seconds() / 60,
                    })
    if len(fixtures) != 40 or sum(f["league_id"] == 71 for f in fixtures) != 20 or sum(f["league_id"] == 72 for f in fixtures) != 20:
        raise ValueError("Expected exactly 20 Serie A and 20 Serie B fixtures.")
    if raw_path.read_bytes() != raw_bytes:
        raise ValueError("The source quotes changed during mapping.")
    snapshot = {
        "source": raw["source"], "model": MODEL,
        "prediction_as_of": as_of.astimezone(timezone.utc).isoformat(),
        "model_training_cutoff": TRAINING_CUTOFF.isoformat(),
        "stake_cop": 25000, "country": "brazil",
        "raw_quotes_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "mapped_fixture_count": len(fixtures), "quoted_market_count": 2 * len(fixtures),
        "schedule_time_zone": "America/Bogota", "date_time_anomalies": anomalies,
        "mapping_note": "Exact cached team IDs and NS fixtures; quoted book odds are unchanged. "
                        "kickoff uses SQLite and book_kickoff_bogota preserves the bookmaker display. "
                        "Any difference is explicitly listed in date_time_anomalies.",
        "fixtures": fixtures,
    }
    path = OUTPUT / "quote_snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Mapped {len(fixtures)} fixtures and {2 * len(fixtures)} quoted markets; "
          f"{len(anomalies)} schedule discrepancy. Saved {path}")
    for anomaly in anomalies:
        print(json.dumps(anomaly, ensure_ascii=True))


if __name__ == "__main__":
    main()
