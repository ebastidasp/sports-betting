"""Worlds/MSI schedule and live-score adapter for PandaScore."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import requests


class MatchFeedError(RuntimeError):
    """A safe-to-display error from the external match feed."""


class MatchProvider(Protocol):
    def snapshot(self, state: str) -> "FeedSnapshot": ...


@dataclass(frozen=True)
class FeedSnapshot:
    matches: list[dict[str, Any]]
    updated_at: str | None
    error: str | None = None
    stale: bool = False


WORLD_PATTERN = re.compile(r"\bworlds?\b|world championship", re.IGNORECASE)
MSI_PATTERN = re.compile(r"mid[\s-]*season invitational|\bmsi\b", re.IGNORECASE)
API_BASE = "https://api.pandascore.co/lol/matches"
MAX_PAGES = 5
PAGE_SIZE = 100


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _event_info(raw: dict[str, Any]) -> str | None:
    league = _mapping(raw.get("league"))
    serie = _mapping(raw.get("serie"))
    tournament = _mapping(raw.get("tournament"))
    searchable = " ".join(
        str(value or "")
        for value in (
            league.get("name"), league.get("slug"),
            serie.get("name"), serie.get("full_name"), serie.get("slug"),
            tournament.get("name"), tournament.get("slug"),
        )
    )
    if WORLD_PATTERN.search(searchable):
        return "Worlds"
    if MSI_PATTERN.search(searchable):
        return "MSI"
    return None


def normalize_match(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Translate a PandaScore match into the app's provider-neutral shape."""
    event = _event_info(raw)
    if event is None:
        return None

    opponents = []
    for item in raw.get("opponents") or []:
        opponent = _mapping(_mapping(item).get("opponent"))
        if opponent.get("id") is not None and opponent.get("name"):
            opponents.append(opponent)
    if len(opponents) != 2:
        return None

    results = {
        str(item.get("team_id")): item.get("score")
        for item in raw.get("results") or []
        if isinstance(item, dict) and item.get("team_id") is not None
    }
    team_a, team_b = opponents
    games = raw.get("games") or []
    running_games = [
        game for game in games
        if isinstance(game, dict) and str(game.get("status", "")).lower() == "running"
    ]
    current_game = running_games[0].get("position") if running_games else None

    number_of_games = raw.get("number_of_games")
    try:
        best_of = int(number_of_games) if int(number_of_games) in (1, 3, 5) else None
    except (TypeError, ValueError):
        best_of = None

    league = _mapping(raw.get("league"))
    serie = _mapping(raw.get("serie"))
    tournament = _mapping(raw.get("tournament"))
    streams = raw.get("streams_list") or []
    official_streams = [s for s in streams if isinstance(s, dict) and s.get("official")]
    stream = next(iter(official_streams or streams), {})
    status = str(raw.get("status") or "unknown").lower()

    return {
        "id": str(raw.get("id", "")),
        "event": event,
        "event_name": league.get("name") or serie.get("full_name") or event,
        "stage": tournament.get("name") or "",
        "series_name": serie.get("full_name") or serie.get("name") or "",
        "team_a": {
            "id": str(team_a.get("id")),
            "name": str(team_a["name"]),
            "acronym": team_a.get("acronym") or "",
            "image_url": team_a.get("image_url") or team_a.get("dark_mode_image_url"),
        },
        "team_b": {
            "id": str(team_b.get("id")),
            "name": str(team_b["name"]),
            "acronym": team_b.get("acronym") or "",
            "image_url": team_b.get("image_url") or team_b.get("dark_mode_image_url"),
        },
        "score_a": results.get(str(team_a.get("id"))),
        "score_b": results.get(str(team_b.get("id"))),
        "status": status,
        "scheduled_at": raw.get("scheduled_at") or raw.get("begin_at"),
        "best_of": best_of,
        "current_game": current_game,
        "rescheduled": bool(raw.get("rescheduled")),
        "stream_url": stream.get("raw_url") or stream.get("embed_url"),
        "stream_official": bool(stream.get("official")),
        "feed_modified_at": raw.get("modified_at"),
        "tournament_id": str(tournament.get("id") or ""),
        "league_id": str(league.get("id") or ""),
        "source": "PandaScore",
    }


class PandaScoreProvider:
    """PandaScore REST feed with separate cache windows for fixtures and live scores."""

    def __init__(
        self,
        token: str | None,
        *,
        session: Any = None,
        upcoming_ttl: int = 900,
        running_ttl: int = 30,
    ) -> None:
        self.token = (token or "").strip()
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "lol-worlds-dashboard/1.0"})
        self.ttls = {"upcoming": upcoming_ttl, "running": running_ttl}
        self._cache: dict[str, tuple[float, FeedSnapshot]] = {}
        self._lock = threading.Lock()

    def snapshot(self, state: str) -> FeedSnapshot:
        if state not in self.ttls:
            raise ValueError("state must be 'upcoming' or 'running'")
        if not self.token:
            return FeedSnapshot(
                matches=[],
                updated_at=None,
                error="Set PANDASCORE_API_TOKEN to enable the Worlds/MSI match feed.",
            )

        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(state)
            if cached and now - cached[0] < self.ttls[state]:
                return cached[1]

            try:
                matches = self._fetch(state)
                snapshot = FeedSnapshot(
                    matches=matches,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                self._cache[state] = (now, snapshot)
                return snapshot
            except MatchFeedError as exc:
                if cached:
                    previous = cached[1]
                    return FeedSnapshot(
                        matches=previous.matches,
                        updated_at=previous.updated_at,
                        error=str(exc),
                        stale=True,
                    )
                return FeedSnapshot(matches=[], updated_at=None, error=str(exc))

    def _fetch(self, state: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            try:
                response = self.session.get(
                    f"{API_BASE}/{state}",
                    params={"token": self.token, "page": page, "per_page": PAGE_SIZE},
                    timeout=15,
                )
            except requests.RequestException as exc:
                raise MatchFeedError("PandaScore could not be reached. Showing cached data if available.") from exc

            if response.status_code == 401:
                raise MatchFeedError("PandaScore rejected the API token. Check PANDASCORE_API_TOKEN.")
            if response.status_code == 403:
                raise MatchFeedError("PandaScore denied this endpoint for the current account or plan.")
            if response.status_code == 429:
                raise MatchFeedError("PandaScore rate limit reached. Showing cached data if available.")
            if response.status_code >= 400:
                raise MatchFeedError(f"PandaScore returned HTTP {response.status_code}.")

            try:
                payload = response.json()
            except ValueError as exc:
                raise MatchFeedError("PandaScore returned an unreadable response.") from exc
            if not isinstance(payload, list):
                raise MatchFeedError("PandaScore returned an unexpected match-list format.")

            for raw in payload:
                if isinstance(raw, dict):
                    match = normalize_match(raw)
                    if match:
                        matches.append(match)
            if len(payload) < PAGE_SIZE:
                break

        matches.sort(key=lambda match: match.get("scheduled_at") or "")
        return matches
