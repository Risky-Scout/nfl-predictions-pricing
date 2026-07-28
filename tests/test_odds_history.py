import numpy as np
import pandas as pd
from nfl_hybrid.odds_history import (
    BackfillConfig,
    build_consensus,
    build_movement_features,
    build_snapshot_plan,
    decimal_to_american,
    normalize_games,
)

def test_decimal_to_american():
    assert decimal_to_american(2.0) == 100
    assert decimal_to_american(1.5) == -200

def test_shared_kickoffs_share_requests():
    games = pd.DataFrame({
        "game_id":["A","B"], "season":[2021,2021], "week":[1,1],
        "kickoff_utc":pd.to_datetime(
            ["2021-09-12T17:00:00Z","2021-09-12T17:00:00Z"], utc=True
        ),
        "home_team_id":["KC","BUF"], "away_team_id":["CLE","PIT"],
    })
    plan = build_snapshot_plan(games, BackfillConfig(seasons=(2021,)))
    assert len(plan) == 12
    assert plan["requested_snapshot_utc"].nunique() == 6

def test_consensus_and_movement():
    rows = []
    for horizon, line in [("opening_7d",-2.5),("closing_t10",-3.0)]:
        for book in ["a","b","c"]:
            common = {
                "game_id":"G1","season":2021,"week":1,"horizon":horizon,
                "requested_snapshot_utc":"2021-09-01T00:00:00Z",
                "returned_snapshot_utc":"2021-09-01T00:00:00Z",
                "bookmaker_key":book,"market":"spreads",
            }
            rows += [
                {**common,"outcome_key":"home","price_decimal":1.91,"point":line},
                {**common,"outcome_key":"away","price_decimal":1.91,"point":-line},
            ]
    consensus = build_consensus(pd.DataFrame(rows))
    movement = build_movement_features(consensus, minimum_books=3)
    assert np.isclose(
        consensus[consensus["horizon"]=="closing_t10"].iloc[0]["consensus_line"],
        -3.0,
    )
    assert np.isclose(movement.iloc[0]["line_movement"], -0.5)


def test_normalize_games_accepts_scheduled_kickoff_utc():
    frame = pd.DataFrame(
        {
            "game_id": ["2020_01_HOU_KC"],
            "season": [2020],
            "week": [1],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2020-09-11T00:20:00Z"],
                utc=True,
            ),
            "home_team_id": ["KC"],
            "away_team_id": ["HOU"],
        }
    )

    result = normalize_games(frame)

    assert result.iloc[0]["game_id"] == "2020_01_HOU_KC"
    assert str(result.iloc[0]["kickoff_utc"]) == "2020-09-11 00:20:00+00:00"


from nfl_hybrid.odds_history import _team_id


def test_los_angeles_rams_alias_is_canonical():
    assert _team_id("Los Angeles Rams") == "LAR"
