"""Optional live BDL smoke test (Section 28).

Runs a single, small, read-only query against the real API ONLY if
BALLDONTLIE_API_KEY is set in the environment. If it is absent, this
cleanly SKIPS -- it is never a test failure, and the suite as a whole must
never require a real credential. No raw payload is written to disk or
printed; only a handful of already-public field values are asserted.
"""
from __future__ import annotations

import os

import pytest

from nfl_hybrid.providers.balldontlie.canonical import normalize_games
from nfl_hybrid.providers.balldontlie.client import BDLClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("BALLDONTLIE_API_KEY"),
    reason="BALLDONTLIE_API_KEY not set -- live smoke test skipped (offline suite is authoritative)",
)


def test_live_get_games_small_query():
    client = BDLClient()
    resp = client.get_games(seasons=[2025], weeks=[1], season_type=2)
    assert resp.pages_fetched >= 1
    assert len(resp.data) > 0
    frame = normalize_games(resp.data, season_type_hint="REG")
    assert set(frame["status_state"]).issubset(
        {"scheduled", "in_progress", "final", "postponed", "canceled", "delayed", "suspended", "abandoned", "unknown"}
    )
    assert (frame["season"] == 2025).all()
