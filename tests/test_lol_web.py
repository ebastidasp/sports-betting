import contextlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from lol_match_feed import FeedSnapshot, PandaScoreProvider, normalize_match
from lol_web import create_app


def pandascore_match(status="running"):
    return {
        "id": 12345,
        "status": status,
        "scheduled_at": "2026-10-04T18:00:00Z",
        "modified_at": "2026-10-04T18:05:00Z",
        "match_type": "best_of",
        "number_of_games": 5,
        "league": {"id": 1, "name": "League of Legends World Championship", "slug": "worlds"},
        "serie": {"full_name": "2026 World Championship"},
        "tournament": {"id": 2, "name": "Group Stage"},
        "opponents": [
            {"type": "Team", "opponent": {"id": 10, "name": "Team Alpha", "acronym": "ALP"}},
            {"type": "Team", "opponent": {"id": 20, "name": "Team Beta", "acronym": "BET"}},
        ],
        "results": [{"team_id": 10, "score": 2}, {"team_id": 20, "score": 1}],
        "games": [{"position": 4, "status": "running"}],
        "rescheduled": False,
        "streams_list": [],
    }


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class StubProvider:
    def snapshot(self, state):
        return FeedSnapshot(matches=[], updated_at="2026-10-04T18:00:00+00:00")


class MatchCardProvider:
    def snapshot(self, state):
        status = "running" if state == "running" else "not_started"
        match = normalize_match(pandascore_match(status))
        return FeedSnapshot(matches=[match], updated_at="2026-10-04T18:00:00+00:00")


class MatchFeedTests(unittest.TestCase):
    def test_normalizes_worlds_scores_and_running_game(self):
        match = normalize_match(pandascore_match())
        self.assertIsNotNone(match)
        self.assertEqual(match["event"], "Worlds")
        self.assertEqual(match["best_of"], 5)
        self.assertEqual((match["score_a"], match["score_b"]), (2, 1))
        self.assertEqual(match["current_game"], 4)
        self.assertEqual(match["team_a"]["acronym"], "ALP")

    def test_filters_non_international_events(self):
        raw = pandascore_match()
        raw["league"] = {"id": 3, "name": "LCK", "slug": "lck"}
        raw["serie"] = {"full_name": "Summer 2026"}
        self.assertIsNone(normalize_match(raw))

    def test_classifies_msi(self):
        raw = pandascore_match("not_started")
        raw["league"] = {"id": 4, "name": "Mid-Season Invitational", "slug": "msi"}
        raw["serie"] = {"full_name": "2026 MSI"}
        self.assertEqual(normalize_match(raw)["event"], "MSI")

    def test_requires_a_token_without_making_requests(self):
        session = FakeSession([])
        snapshot = PandaScoreProvider("", session=session).snapshot("running")
        self.assertIn("PANDASCORE_API_TOKEN", snapshot.error)
        self.assertEqual(session.calls, [])

    def test_fetch_is_cached_and_sanitizes_live_scores(self):
        session = FakeSession([FakeResponse(200, [pandascore_match()])])
        provider = PandaScoreProvider("private-token", session=session, upcoming_ttl=60)
        first = provider.snapshot("upcoming")
        second = provider.snapshot("upcoming")
        self.assertEqual(first.matches, second.matches)
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn("private-token", repr(first))

    def test_uses_last_snapshot_after_provider_error(self):
        session = FakeSession([
            FakeResponse(200, [pandascore_match()]),
            FakeResponse(429, {"error": "rate limited"}),
        ])
        provider = PandaScoreProvider("token", session=session, upcoming_ttl=0)
        first = provider.snapshot("upcoming")
        second = provider.snapshot("upcoming")
        self.assertEqual(len(first.matches), 1)
        self.assertTrue(second.stale)
        self.assertEqual(second.matches, first.matches)
        self.assertIn("rate limit", second.error.lower())


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "lol.sqlite3"
        with contextlib.closing(sqlite3.connect(self.db_path)) as con:
            con.execute(
                "CREATE TABLE team_games (gameid TEXT, team TEXT, opp TEXT, date TEXT, league TEXT, win INTEGER)"
            )
            con.executemany(
                "INSERT INTO team_games VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("g1", "Team Alpha", "Team Beta", "2026-10-01T12:00:00+00:00", "LCK", 1),
                    ("g1", "Team Beta", "Team Alpha", "2026-10-01T12:00:00+00:00", "LCK", 0),
                ],
            )
            con.commit()
        self.app = create_app(
            provider=StubProvider(),
            config={"TESTING": True, "DB_PATH": self.db_path, "MODEL_PATH": self.db_path.with_name("missing.joblib")},
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_dashboard_and_match_api_render(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upcoming series", response.data)
        data = self.client.get("/api/matches?state=running").get_json()
        self.assertEqual(data["state"], "running")
        self.assertEqual(data["matches"], [])

    def test_dashboard_renders_live_and_upcoming_cards(self):
        app = create_app(
            provider=MatchCardProvider(),
            config={"TESTING": True, "DB_PATH": self.db_path, "MODEL_PATH": self.db_path.with_name("missing.joblib")},
        )
        response = app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Team Alpha", response.data)
        self.assertIn(b"Group Stage", response.data)
        self.assertIn(b"2026-10-04T18:00:00Z", response.data)

    def test_predictor_uses_local_team_choices(self):
        response = self.client.get("/predictor")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Team Alpha", response.data)
        self.assertIn(b"Team Beta", response.data)

    def test_rejects_same_team_before_loading_model(self):
        response = self.client.post(
            "/predictor",
            data={"team_a": "Team Alpha", "team_b": "Team Alpha", "best_of": "3", "side_a": "blue"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Choose two different teams", response.data)

    def test_invalid_match_state_is_rejected(self):
        response = self.client.get("/api/matches?state=finished")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
