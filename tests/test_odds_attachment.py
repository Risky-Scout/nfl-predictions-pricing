import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.odds_attachment import (
    OddsAttachmentConfig,
    attach_to_compact,
    build_market_odds_features,
    model_features,
)


def sample():
    consensus = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "season": 2021,
                "week": 1,
                "horizon": "closing_t10",
                "requested_snapshot_utc": "2021-09-12T19:15:00Z",
                "returned_snapshot_utc": "2021-09-12T19:10:00Z",
                "market": "spreads",
                "eligible_books": 12,
                "consensus_line": -3.0,
                "line_sd": 0.25,
                "consensus_home_or_over_novig_probability": 0.54,
                "probability_sd": 0.007,
                "median_hold": 0.045,
            }
        ]
    )
    movement = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "season": 2021,
                "week": 1,
                "market": "spreads",
                "closing_snapshot_utc": "2021-09-12T19:10:00Z",
                "closing_eligible_books": 12,
                "closing_line": -3.0,
                "closing_home_or_over_novig_probability": 0.54,
                "closing_line_sd": 0.25,
                "closing_probability_sd": 0.007,
                "opening_available": True,
                "opening_horizon": "opening_7d",
                "opening_line": -2.5,
                "opening_home_or_over_novig_probability": 0.51,
                "line_movement": -0.5,
                "probability_movement": 0.03,
            }
        ]
    )
    return consensus, movement


def test_ats_attachment_values():
    consensus, movement = sample()
    result = build_market_odds_features(
        consensus, movement, "pregame_ats"
    )
    row = result.iloc[0]
    assert np.isclose(row["market_line_movement"], -0.5)
    assert row["market_opening_horizon_minutes"] == 10080.0
    assert row["market_t10_eligible_books"] == 12


def test_moneyline_does_not_use_line_features():
    assert "market_t10_consensus_line" not in model_features(
        "pregame_moneyline"
    )


def test_future_snapshot_rejected():
    consensus, movement = sample()
    consensus["returned_snapshot_utc"] = "2021-09-12T19:16:00Z"
    with pytest.raises(ValueError, match="Future"):
        build_market_odds_features(
            consensus, movement, "pregame_ats"
        )


def test_2024_and_2025_are_excluded():
    consensus, movement = sample()
    odds = build_market_odds_features(
        consensus, movement, "pregame_ats"
    )
    compact = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2021, 2024, 2025],
            "week": [1, 1, 1],
        }
    )
    attached = attach_to_compact(
        compact,
        odds,
        "pregame_ats",
        OddsAttachmentConfig(development_seasons=(2021,)),
    )
    assert attached["game_id"].tolist() == ["G1"]


def test_attachment_restores_week_when_compact_lacks_week():
    consensus, movement = sample()

    odds = build_market_odds_features(
        consensus,
        movement,
        "pregame_ats",
    )

    compact = pd.DataFrame(
        {
            "game_id": ["G1"],
            "season": [2021],
            "base_feature": [1.0],
        }
    )

    attached = attach_to_compact(
        compact,
        odds,
        "pregame_ats",
        OddsAttachmentConfig(development_seasons=(2021,)),
    )

    assert attached.iloc[0]["week"] == 1
    assert attached.iloc[0]["season"] == 2021
