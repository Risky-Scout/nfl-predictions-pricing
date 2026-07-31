import numpy as np
import pandas as pd

from nfl_hybrid.pricing.fundamental_shadow import attach_shadow_columns


def _card():
    return pd.DataFrame({
        "game_id": ["g1", "g1", "g1"],
        "market": ["moneyline", "ats", "total"],
        "market_fair_probability": [0.55, 0.50, 0.50],
    })


def test_missing_fundamental_stays_null_not_market():
    out = attach_shadow_columns(_card(), None, status="NO_INPUTS", artifact_version=None)
    assert out["fundamental_probability"].isna().all()  # NULL, never copied from market
    assert (out["production_probability"] == out["market_fair_probability"]).all()
    assert (out["production_source"] == "MARKET_BASELINE").all()
    assert (out["fundamental_artifact_version"] == "NO_FROZEN_FUNDAMENTAL_ARTIFACT").all()


def test_market_and_fundamental_are_distinct_columns():
    fundamental = pd.DataFrame({
        "game_id": ["g1"], "fundamental_home_win": [0.62],
        "fundamental_home_cover": [0.48], "fundamental_over": [0.53],
    })
    out = attach_shadow_columns(_card(), fundamental, status="AVAILABLE", artifact_version="shadow.v1")
    ml = out[out["market"] == "moneyline"].iloc[0]
    assert ml["fundamental_probability"] == 0.62  # from shadow, not the 0.55 market
    assert ml["market_fair_probability"] == 0.55
    assert ml["fundamental_probability"] != ml["market_fair_probability"]
    # production still market baseline despite divergence
    assert ml["production_source"] == "MARKET_BASELINE"
    assert ml["production_probability"] == 0.55
    assert np.isclose(ml["market_fundamental_difference"], 0.62 - 0.55)
