"""Team crosswalk + canonical games/team_stats/player_stats/injuries/roster
normalization. Entirely offline, fixture-driven.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.providers.balldontlie import canonical
from nfl_hybrid.providers.balldontlie.team_crosswalk import (
    TeamCrosswalkError,
    canonical_team_for_bdl_id,
    canonical_team_from_team_object,
    completeness_report,
)

FIXTURES = Path(__file__).parent / "fixtures" / "balldontlie"


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())["data"]


# -- team crosswalk ------------------------------------------------------------

def test_crosswalk_completeness_all_32_franchises():
    report = completeness_report()
    assert report["complete"] is True
    assert report["bdl_franchise_count"] == 32
    assert report["distinct_canonical_teams"] == 32


def test_crosswalk_resolves_known_id():
    assert canonical_team_for_bdl_id(18) == "PHI"
    assert canonical_team_for_bdl_id(21) == "WAS"  # BDL abbreviates Washington "WSH"


def test_crosswalk_fails_closed_on_unknown_id():
    with pytest.raises(TeamCrosswalkError):
        canonical_team_for_bdl_id(999)


def test_crosswalk_fails_closed_on_none():
    with pytest.raises(TeamCrosswalkError):
        canonical_team_for_bdl_id(None)


def test_crosswalk_from_team_object_prefers_id():
    assert canonical_team_from_team_object({"id": 19, "abbreviation": "DAL"}) == "DAL"


# -- games ----------------------------------------------------------------------

def test_normalize_games_final_and_scheduled():
    raw = _load("games_page1.json") + _load("games_page2.json")
    frame = canonical.normalize_games(raw, season_type_hint="REG")

    final_row = frame.loc[frame["provider_game_id"] == 900001].iloc[0]
    assert final_row["status_state"] == "final"
    assert final_row["home_team"] == "PHI"
    assert final_row["away_team"] == "DAL"
    assert final_row["home_score"] == 24
    assert final_row["away_score"] == 20
    assert final_row["game_id"] == "2025_01_DAL_PHI"
    assert final_row["season_type"] == "REG"
    assert final_row["season_type_basis"] == "query_hint"

    scheduled_row = frame.loc[frame["provider_game_id"] == 900003].iloc[0]
    assert scheduled_row["status_state"] == "scheduled"
    assert pd.isna(scheduled_row["home_score"])

    in_progress_row = frame.loc[frame["provider_game_id"] == 900004].iloc[0]
    assert in_progress_row["status_state"] == "in_progress"
    # A realistic non-null score on an in-progress game must still be
    # present in the canonical frame (finality gates are a separate layer,
    # not something normalize_games decides) -- Section 8/24 Scenario A.
    assert in_progress_row["home_score"] == 17


def test_normalize_games_missing_hint_fails_closed():
    """Section 6 certification: a non-postseason row with no season_type_hint
    must fail closed, never silently default to REG."""
    raw = _load("games_page1.json")
    with pytest.raises(canonical.CanonicalizationError, match="ambiguous season type"):
        canonical.normalize_games(raw)  # no hint


def test_normalize_games_unrecognized_hint_fails_closed():
    raw = _load("games_page1.json")
    with pytest.raises(canonical.CanonicalizationError, match="ambiguous season type"):
        canonical.normalize_games(raw, season_type_hint="bogus")


def test_normalize_games_preseason_week1_and_regular_week1_not_conflated():
    """Section 6 certification: BDL returns week numbering that restarts at
    1 for both preseason and regular season with no other distinguishing
    field on non-postseason rows -- even the SAME matchup (DAL@PHI) in week
    1 of each season type must classify differently based solely on which
    season_type_hint the caller supplies (the query filter actually used),
    never on week number or any inference from the row itself."""
    preseason_raw = _load("games_preseason_week1.json")
    regular_raw = _load("games_regular_week1.json")

    preseason_frame = canonical.normalize_games(preseason_raw, season_type_hint="PRE")
    regular_frame = canonical.normalize_games(regular_raw, season_type_hint="REG")

    assert preseason_frame.iloc[0]["week"] == regular_frame.iloc[0]["week"] == 1
    assert preseason_frame.iloc[0]["home_team"] == regular_frame.iloc[0]["home_team"] == "PHI"
    assert preseason_frame.iloc[0]["away_team"] == regular_frame.iloc[0]["away_team"] == "DAL"

    assert preseason_frame.iloc[0]["season_type"] == "PRE"
    assert regular_frame.iloc[0]["season_type"] == "REG"
    assert preseason_frame.iloc[0]["season_type_basis"] == "query_hint"
    assert regular_frame.iloc[0]["season_type_basis"] == "query_hint"

    # Same raw row shape, both valid hints: proves the classification tracks
    # the hint, not any property of the row (week number is identical).
    same_row = _load("games_preseason_week1.json")
    as_pre = canonical.normalize_games(same_row, season_type_hint="PRE").iloc[0]["season_type"]
    as_reg = canonical.normalize_games(same_row, season_type_hint="REG").iloc[0]["season_type"]
    assert as_pre == "PRE"
    assert as_reg == "REG"
    assert as_pre != as_reg


def test_normalize_games_postseason_always_unambiguous():
    raw = [dict(_load("games_page1.json")[0])]
    raw[0]["postseason"] = True
    frame = canonical.normalize_games(raw, season_type_hint="PRE")  # hint ignored for POST
    assert frame.iloc[0]["season_type"] == "POST"
    assert frame.iloc[0]["season_type_basis"] == "postseason_flag"


def test_normalize_games_unknown_team_fails_closed():
    raw = _load("games_postponed_and_unknown_team.json")
    unknown_team_game = [g for g in raw if g["id"] == 900006]
    with pytest.raises(canonical.CanonicalizationError):
        canonical.normalize_games(unknown_team_game)


def test_normalize_games_postponed_status():
    raw = _load("games_postponed_and_unknown_team.json")
    postponed = [g for g in raw if g["id"] == 900005]
    frame = canonical.normalize_games(postponed, season_type_hint="REG")
    assert frame.iloc[0]["status_state"] == "postponed"


def test_source_payload_hash_is_deterministic():
    raw = _load("games_page1.json")
    frame1 = canonical.normalize_games(raw, season_type_hint="REG")
    frame2 = canonical.normalize_games(raw, season_type_hint="REG")
    assert list(frame1["source_payload_hash"]) == list(frame2["source_payload_hash"])


# -- team_stats -------------------------------------------------------------------

def test_normalize_team_stats_and_completeness_complete_case():
    raw = _load("team_stats_complete.json")
    frame = canonical.normalize_team_stats(raw)
    assert len(frame) == 2
    assert set(frame["team"]) == {"DAL", "PHI"}

    report = canonical.team_stats_completeness(
        frame, {"provider_game_id": 900001, "home_team": "PHI", "away_team": "DAL"}
    )
    assert report["box_complete"] is True
    assert report["teams_missing"] == []


def test_team_stats_completeness_missing_one_team():
    raw = _load("team_stats_missing_one_team.json")
    frame = canonical.normalize_team_stats(raw)
    report = canonical.team_stats_completeness(
        frame, {"provider_game_id": 900002, "home_team": "LAC", "away_team": "KC"}
    )
    assert report["box_complete"] is False
    assert report["teams_missing"] == ["LAC"]


# -- player_stats ---------------------------------------------------------------

def test_normalize_player_stats():
    raw = _load("player_stats.json")
    frame = canonical.normalize_player_stats(raw)
    assert len(frame) == 2
    hurts = frame.loc[frame["player_name"] == "Jalen Hurts"].iloc[0]
    assert hurts["team"] == "PHI"
    assert hurts["passing_yards"] == 201
    assert hurts["passing_touchdowns"] == 2

    report = canonical.player_stats_completeness(frame, 900001)
    assert report["player_stats_complete"] is True


# -- injuries ---------------------------------------------------------------------

def test_normalize_injuries_preserves_raw_status_no_inference():
    raw = _load("player_injuries_page1.json") + _load("player_injuries_page2.json")
    frame = canonical.normalize_injuries(raw)
    assert len(frame) == 2
    assert set(frame["status_raw"]) == {"Questionable", "Out"}
    # No practice-participation / severity columns are fabricated.
    assert "practice_status" not in frame.columns
    assert "expected_snaps" not in frame.columns
    row = frame.loc[frame["provider_player_id"] == 6001].iloc[0]
    assert row["team"] == "DET"


# -- roster -----------------------------------------------------------------------

def test_normalize_roster():
    raw = _load("roster.json")
    frame = canonical.normalize_roster(raw, team="PHI", season=2026)
    assert len(frame) == 3
    hurts = frame.loc[frame["player_name"] == "Jalen Hurts"].iloc[0]
    assert hurts["position"] == "QB"
    assert hurts["depth"] == 1
    assert hurts["injury_status"] == "Questionable"
    assert (frame["team"] == "PHI").all()
