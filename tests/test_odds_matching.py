"""Odds-API schema + matching tests.

Guards the paid-quote swap-in path: team-name/abbreviation variants must
canonicalize, the flattened schema must be stable (pinned fixture), UTC-midnight
kickoff edges must match within tolerance, and an unmatchable event must raise
rather than silently drop.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data.odds import (
    devig_two_way_groups,
    match_odds_to_games,
    require_matched_events,
)
from nfl_hybrid.data.providers.the_odds_api import TheOddsAPIAdapter, OddsAPIConfig
from nfl_hybrid.data.team_ids import try_canonical_team_id

FIXTURE = Path(__file__).parent / "fixtures" / "odds_api_current_sample.json"

EXPECTED_COLUMNS = {
    "provider_event_id", "home_team_name", "away_team_name", "home_team_id",
    "away_team_id", "commence_time_utc", "snapshot_utc", "market_type",
    "outcome_side", "line_value", "price_american", "price_decimal",
    "raw_implied_probability",
}


def test_team_name_variants_canonicalize():
    # abbreviations
    assert try_canonical_team_id("LA") == "LAR"      # historical Rams abbrev
    assert try_canonical_team_id("LAR") == "LAR"
    assert try_canonical_team_id("LAC") == "LAC"
    assert try_canonical_team_id("WSH") == "WAS"
    assert try_canonical_team_id("WAS") == "WAS"
    # full names as returned by The Odds API
    assert try_canonical_team_id("Los Angeles Rams") == "LAR"
    assert try_canonical_team_id("Los Angeles Chargers") == "LAC"
    assert try_canonical_team_id("Washington Commanders") == "WAS"
    # a genuinely unknown team is not silently coerced
    assert try_canonical_team_id("Nonexistent City Aliens") is None


def test_fixture_schema_is_stable():
    payload = json.loads(FIXTURE.read_text())
    frame = TheOddsAPIAdapter(OddsAPIConfig(api_key="test")).flatten_response(payload)
    assert len(frame) > 0
    assert EXPECTED_COLUMNS.issubset(set(frame.columns))
    assert set(frame["market_type"].unique()).issubset({"moneyline", "spread", "total"})
    # every team maps to a canonical id
    assert frame["home_team_id"].notna().all()
    assert frame["away_team_id"].notna().all()


def _odds_row(event_id, home_id, away_id, commence, side="home", price=-110, line=-3.0, market="spread"):
    return {
        "provider_event_id": event_id,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "commence_time_utc": pd.Timestamp(commence, tz="UTC"),
        "snapshot_utc": pd.Timestamp(commence, tz="UTC") - pd.Timedelta(minutes=15),
        "market_type": market,
        "outcome_side": side,
        "line_value": line,
        "bookmaker_id": "book1",
        "raw_implied_probability": 0.5238,
        "price_american": price,
    }


def test_utc_midnight_edge_matches_within_tolerance():
    # commence at 00:20 UTC (Monday-night style late kickoff), game kickoff 00:20
    odds = pd.DataFrame(
        [_odds_row("evt1", "LAR", "SEA", "2024-09-09T00:20:00Z")]
    )
    games = pd.DataFrame(
        {
            "game_id": ["g_lar_sea"],
            "home_team_id": ["LAR"],
            "away_team_id": ["SEA"],
            "scheduled_kickoff_utc": [pd.Timestamp("2024-09-09T00:20:00Z")],
        }
    )
    matched = match_odds_to_games(odds, games)
    assert (matched["game_match_status"] == "matched").all()
    assert (matched["game_id"] == "g_lar_sea").all()
    require_matched_events(matched)  # does not raise


def test_no_match_must_raise():
    odds = pd.DataFrame(
        [_odds_row("evt_bad", "GB", "CHI", "2024-09-15T17:00:00Z")]
    )
    games = pd.DataFrame(
        {
            "game_id": ["g_other"],
            "home_team_id": ["LAR"],
            "away_team_id": ["SEA"],
            "scheduled_kickoff_utc": [pd.Timestamp("2024-09-09T00:20:00Z")],
        }
    )
    matched = match_odds_to_games(odds, games)
    assert (matched["game_match_status"] == "team_pair_not_found").all()
    with pytest.raises(ValueError, match="did not match"):
        require_matched_events(matched)


def test_devig_two_way_sums_to_one():
    # home/away moneyline with an overround -> de-vigged pair sums to 1
    odds = pd.DataFrame(
        [
            {**_odds_row("e", "LAR", "SEA", "2024-09-09T00:20:00Z", side="home", market="moneyline", line=float("nan")),
             "raw_implied_probability": 0.664430},
            {**_odds_row("e", "LAR", "SEA", "2024-09-09T00:20:00Z", side="away", market="moneyline", line=float("nan")),
             "raw_implied_probability": 0.378788},
        ]
    )
    out = devig_two_way_groups(odds)
    assert out["devig_probability"].sum() == pytest.approx(1.0, abs=1e-9)
