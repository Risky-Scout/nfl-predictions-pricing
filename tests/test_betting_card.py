import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.pricing.betting_card import (
    build_betting_card,
    readiness_from_production_spec,
)


def _games():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2025, 2025],
            "week": ["1", "1"],
            "home_team": ["NO", "BUF"],
            "away_team": ["ARI", "BAL"],
            "home_spread": [6.0, -1.5],
            "total_line": [44.5, 50.5],
        }
    )


def test_readiness_from_production_spec_maps_baseline_to_retain():
    status = readiness_from_production_spec("config/production_model_spec.json")
    assert status["moneyline"] == "RETAIN_BASELINE"
    assert status["ats"] == "RETAIN_BASELINE"
    assert status["total"] == "RETAIN_BASELINE"


def test_card_market_baseline_recommends_no_bets():
    readiness = {"moneyline": "RETAIN_BASELINE", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}
    card = build_betting_card(_games(), readiness_by_market=readiness)
    assert len(card) == 6  # 2 games x 3 markets
    assert int(card["should_bet"].sum()) == 0
    assert (card["penalized_edge"].abs() < 1e-9).all()
    assert card["no_bet_reason"].str.contains("edge not established").all()
    # required card columns
    for col in [
        "matchup", "model_spread", "closing_or_reference_line", "calibrated_probability",
        "p_cover", "p_over", "tie_probability", "penalized_edge", "recommended_stake",
        "should_bet", "readiness_status", "market_source",
    ]:
        assert col in card.columns


def test_card_probabilities_are_valid():
    readiness = {"moneyline": "RETAIN_BASELINE", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}
    card = build_betting_card(_games(), readiness_by_market=readiness)
    for col in ["calibrated_probability", "market_fair_probability", "p_cover", "p_over"]:
        assert ((card[col] >= 0.0) & (card[col] <= 1.0)).all()


def test_card_stakes_when_model_beats_market_and_supported():
    games = _games()
    # inject a strong model disagreement on the moneyline of g1
    games["model_home_win_probability"] = [0.90, 0.50]
    readiness = {"moneyline": "STATISTICALLY_SUPPORTED", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}
    card = build_betting_card(games, readiness_by_market=readiness)
    ml = card[(card["game_id"] == "g1") & (card["market"] == "moneyline")].iloc[0]
    assert ml["should_bet"]
    assert ml["recommended_stake"] > 0.0
    # unproven markets still never stake
    assert int(card[card["market"] != "moneyline"]["should_bet"].sum()) == 0
