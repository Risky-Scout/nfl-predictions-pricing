from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.data.odds import devig_two_way_groups, match_odds_to_games
from nfl_hybrid.data.providers.nfl_injuries import NFLOfficialInjuryAdapter
from nfl_hybrid.data.providers.nflverse import NflverseAdapter
from nfl_hybrid.data.providers.sportradar import SportradarInjuryAdapter
from nfl_hybrid.data.providers.spreadspoke import SpreadspokeAdapter
from nfl_hybrid.data.providers.the_odds_api import OddsAPIConfig, TheOddsAPIAdapter
from nfl_hybrid.data.team_ids import canonical_team_id


def test_team_aliases_are_stable():
    assert canonical_team_id("LVR") == "LV"
    assert canonical_team_id("Oakland Raiders") == "LV"
    assert canonical_team_id("Houston Oilers") == "TEN"
    assert canonical_team_id("Houston Texans") == "HOU"


def test_spreadspoke_normalization(tmp_path: Path):
    raw = pd.DataFrame(
        [
            {
                "schedule_date": "2025-09-07",
                "schedule_season": 2025,
                "schedule_week_str": "1",
                "schedule_playoff": False,
                "team_home": "Buffalo Bills",
                "home_team_id": "BUF",
                "score_home": 27,
                "team_away": "Baltimore Ravens",
                "away_team_id": "BAL",
                "score_away": 24,
                "team_favorite_id": "BUF",
                "spread_favorite": -2.5,
                "over_under_line": 49.5,
                "stadium": "Example Stadium",
                "stadium_neutral": False,
                "weather_temperature": 70,
                "weather_wind_mph": 8,
                "weather_humidity": 55,
                "weather_detail": "clear",
            }
        ]
    )
    path = tmp_path / "spreadspoke.csv"
    raw.to_csv(path, index=False)
    frame = SpreadspokeAdapter(path).load_games([2025]).data
    assert frame.loc[0, "home_spread_reference"] == -2.5
    assert frame.loc[0, "total_line_reference"] == 49.5
    assert not frame.loc[0, "line_timestamp_known"]


class _FakeNflverse:
    def load_schedules(self, seasons):
        return pd.DataFrame(
            [
                {
                    "game_id": "2020_01_HOU_KC",
                    "season": 2020,
                    "game_type": "REG",
                    "week": 1,
                    "gameday": "2020-09-10",
                    "gametime": "20:20",
                    "away_team": "HOU",
                    "away_score": 20,
                    "home_team": "KC",
                    "home_score": 34,
                    "location": "Home",
                    "overtime": 0,
                    "away_rest": 7,
                    "home_rest": 7,
                    "away_moneyline": 350,
                    "home_moneyline": -450,
                    "spread_line": 9.5,
                    "away_spread_odds": -110,
                    "home_spread_odds": -110,
                    "total_line": 54.0,
                    "under_odds": -110,
                    "over_odds": -110,
                    "div_game": 0,
                    "roof": "outdoors",
                    "surface": "grass",
                    "temp": 56,
                    "wind": 7,
                    "away_qb_id": "00-0023682",
                    "home_qb_id": "00-0033873",
                    "away_qb_name": "Deshaun Watson",
                    "home_qb_name": "Patrick Mahomes",
                    "away_coach": "Bill O'Brien",
                    "home_coach": "Andy Reid",
                    "referee": "Ref",
                    "stadium_id": "KAN00",
                    "stadium": "Arrowhead Stadium",
                }
            ]
        )


def test_nflverse_schedule_sign_and_timezone():
    frame = NflverseAdapter(module=_FakeNflverse()).load_games([2020]).data
    # Canonical sportsbook sign: home favorite is negative.
    assert frame.loc[0, "home_spread_reference"] == -9.5
    # 20:20 Eastern on 2020-09-10 is 00:20 UTC on 2020-09-11.
    assert str(frame.loc[0, "scheduled_kickoff_utc"]) == "2020-09-11 00:20:00+00:00"


def _odds_payload():
    return {
        "timestamp": "2021-11-25T12:00:00Z",
        "data": [
            {
                "id": "event1",
                "sport_key": "americanfootball_nfl",
                "sport_title": "NFL",
                "commence_time": "2021-11-25T17:30:00Z",
                "home_team": "Detroit Lions",
                "away_team": "Chicago Bears",
                "bookmakers": [
                    {
                        "key": "book",
                        "title": "Book",
                        "last_update": "2021-11-25T11:59:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Detroit Lions", "price": 145},
                                    {"name": "Chicago Bears", "price": -165},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Detroit Lions", "price": -110, "point": 3.0},
                                    {"name": "Chicago Bears", "price": -110, "point": -3.0},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": -105, "point": 41.5},
                                    {"name": "Under", "price": -115, "point": 41.5},
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_odds_api_flatten_and_devig():
    adapter = TheOddsAPIAdapter(OddsAPIConfig(api_key="test"))
    frame = adapter.flatten_response(_odds_payload())
    assert len(frame) == 6
    assert set(frame["market_type"]) == {"moneyline", "spread", "total"}
    assert set(frame["outcome_side"]) == {"home", "away", "over", "under"}
    assert frame["minutes_to_kickoff"].eq(330).all()

    devigged = devig_two_way_groups(frame)
    for market in ("moneyline", "spread", "total"):
        group = devigged[devigged["market_type"] == market]
        assert np.isclose(group["devig_probability"].sum(), 1.0)


def test_odds_event_match():
    odds = TheOddsAPIAdapter(OddsAPIConfig(api_key="test")).flatten_response(
        _odds_payload()
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "2021_12_CHI_DET",
                "home_team_id": "DET",
                "away_team_id": "CHI",
                "scheduled_kickoff_utc": "2021-11-25T17:30:00Z",
            }
        ]
    )
    matched = match_odds_to_games(odds, games)
    assert matched["game_id"].eq("2021_12_CHI_DET").all()
    assert matched["game_match_status"].eq("matched").all()


def test_sportradar_injury_flatten():
    payload = {
        "season": {"year": 2026, "type": "REG"},
        "week": {"sequence": 1},
        "teams": [
            {
                "id": "team-guid",
                "alias": "KC",
                "market": "Kansas City",
                "name": "Chiefs",
                "players": [
                    {
                        "id": "player-guid",
                        "sr_id": "sr:player:1",
                        "name": "Example Player",
                        "jersey": 12,
                        "position": "WR",
                        "injury": {
                            "primary": "Hamstring",
                            "status": "Questionable",
                            "status_date": "2026-09-10T00:00:00Z",
                            "practice": {
                                "status": "Limited Participation In Practice"
                            },
                        },
                    }
                ],
            }
        ],
    }
    frame = SportradarInjuryAdapter().flatten_response(payload)
    assert frame.loc[0, "team_id"] == "KC"
    assert frame.loc[0, "practice_status"] == "LIMITED"
    assert frame.loc[0, "game_status"] == "QUESTIONABLE"


def test_official_injury_export_requires_team(tmp_path: Path):
    valid = pd.DataFrame(
        [
            {
                "team": "KC",
                "player": "Example Player",
                "position": "WR",
                "injuries": "Hamstring",
                "practice_status": "Limited Participation In Practice",
                "game_status": "Questionable",
                "report_date": "2025-09-04T20:00:00Z",
            }
        ]
    )
    path = tmp_path / "injuries.csv"
    valid.to_csv(path, index=False)
    frame = NFLOfficialInjuryAdapter().load_export(
        path, season=2025, week=1
    ).data
    assert frame.loc[0, "team_id"] == "KC"
    assert frame.loc[0, "practice_status"] == "LIMITED"

    invalid = valid.drop(columns=["team"])
    invalid_path = tmp_path / "injuries_no_team.csv"
    invalid.to_csv(invalid_path, index=False)
    with pytest.raises(ValueError):
        NFLOfficialInjuryAdapter().load_export(
            invalid_path, season=2025, week=1
        )


def test_spreadspoke_null_boolean_is_false(tmp_path):
    source = pd.DataFrame(
        {
            "schedule_date": ["2025-09-07"],
            "schedule_season": [2025],
            "schedule_week_str": ["1"],
            "schedule_playoff": [np.nan],
            "team_home": ["Las Vegas Raiders"],
            "score_home": [20],
            "team_away": ["Los Angeles Chargers"],
            "score_away": [24],
            "team_favorite_id": ["LAC"],
            "spread_favorite": [-3.0],
            "over_under_line": [44.5],
            "stadium_neutral": [np.nan],
        }
    )
    path = tmp_path / "spreadspoke.csv"
    source.to_csv(path, index=False)
    result = SpreadspokeAdapter(path).load_games([2025]).data.iloc[0]
    assert not bool(result["neutral_site"])
    assert result["season_type"] == "REG"
