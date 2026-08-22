"""BDL client tests: pagination proof, rate limits, HTTP/schema errors.

Entirely offline -- a fake ``requests``-shaped session is injected, so no
BALLDONTLIE_API_KEY is required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_hybrid.providers.balldontlie.client import (
    BDLClient,
    BDLConfig,
    BDLConfigError,
    BDLHTTPError,
    BDLPaginationError,
    BDLRateLimitError,
    BDLSchemaError,
)

FIXTURES = Path(__file__).parent / "fixtures" / "balldontlie"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class _FakeResponse:
    def __init__(self, status_code: int, body: object, headers: dict | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return self._body


class _FakeSession:
    """Replays a fixed sequence of responses, one per call to .get()."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        assert headers.get("Authorization") == "test-key"
        return self._responses.pop(0)


def _client(responses: list[_FakeResponse]) -> tuple[BDLClient, _FakeSession]:
    session = _FakeSession(responses)
    client = BDLClient(BDLConfig(api_key="test-key"), session=session)
    return client, session


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("BALLDONTLIE_API_KEY", raising=False)
    client = BDLClient(BDLConfig(api_key=None), session=_FakeSession([]))
    with pytest.raises(BDLConfigError):
        client.get_games(seasons=[2025])


def test_pagination_exhausts_and_page2_rows_present():
    page1 = _load("games_page1.json")
    page2 = _load("games_page2.json")
    client, session = _client([_FakeResponse(200, page1), _FakeResponse(200, page2)])

    resp = client.get_games(seasons=[2025, 2026])

    assert resp.pages_fetched == 2
    ids = [row["id"] for row in resp.data]
    assert ids == [900001, 900002, 900003, 900004]
    # Page-2-only rows are actually present, not silently dropped after page 1.
    assert 900003 in ids and 900004 in ids
    # cursor was threaded through on the second call
    assert session.calls[1]["params"] is not None


def test_pagination_stops_when_next_cursor_is_null():
    page2 = _load("games_page2.json")  # next_cursor: null
    client, _ = _client([_FakeResponse(200, page2)])
    resp = client.get_games(seasons=[2026])
    assert resp.pages_fetched == 1


def test_pagination_repeated_cursor_raises():
    looping_page = {"data": [{"id": 1}], "meta": {"next_cursor": 42}}
    client, _ = _client(
        [
            _FakeResponse(200, looping_page),
            _FakeResponse(200, looping_page),  # same next_cursor=42 again -> loop
        ]
    )
    with pytest.raises(BDLPaginationError):
        client.get_games(seasons=[2025])


def test_pagination_exceeds_max_pages_raises():
    def page(cursor):
        return {"data": [{"id": cursor}], "meta": {"next_cursor": cursor + 1}}

    responses = [_FakeResponse(200, page(i)) for i in range(5)]
    client = BDLClient(BDLConfig(api_key="test-key", max_pages=3), session=_FakeSession(responses))
    with pytest.raises(BDLPaginationError):
        client.get_games(seasons=[2025])


def test_rate_limit_raises_no_hidden_retry():
    client, session = _client([_FakeResponse(429, "slow down", headers={"Retry-After": "5"})])
    with pytest.raises(BDLRateLimitError) as exc_info:
        client.get_games(seasons=[2025])
    assert exc_info.value.retry_after_seconds == 5.0
    assert len(session.calls) == 1  # no automatic retry


def test_http_error_raises_clear_exception():
    client, _ = _client([_FakeResponse(500, "server error")])
    with pytest.raises(BDLHTTPError) as exc_info:
        client.get_games(seasons=[2025])
    assert exc_info.value.status_code == 500


def test_unexpected_shape_raises_schema_error():
    client, _ = _client([_FakeResponse(200, {"not_data": []})])
    with pytest.raises(BDLSchemaError):
        client.get_games(seasons=[2025])


def test_get_game_single_resource():
    single = {"data": _load("games_page1.json")["data"][0]}
    client, session = _client([_FakeResponse(200, single)])
    game = client.get_game(900001)
    assert game["status_state"] == "final"
    assert "/games/900001" in session.calls[0]["url"]


def test_team_stats_has_no_weeks_filter_param():
    """Regression guard for the verified live discrepancy: /team_stats does
    not support a `weeks` filter. There must be no `weeks` kwarg on
    get_team_stats at all (see client.py's inline note)."""
    import inspect

    from nfl_hybrid.providers.balldontlie.client import BDLClient as _C

    params = inspect.signature(_C.get_team_stats).parameters
    assert "weeks" not in params
