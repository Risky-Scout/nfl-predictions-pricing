"""Live fundamental scoring: complete features -> non-null, distinct from market."""

import os

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import FROZEN_FEATURES, build_live_augmented_features
from nfl_hybrid.pricing.fundamental_shadow import attach_shadow_columns, load_shadow, score_shadow

GAMES = "data/backfill_2020_2025/canonical/games.parquet"
PBP = "data/backfill_2020_2025/raw/pbp.parquet"
ARTIFACT = "config/shadow_fundamental_2026/jointscore_epa.joblib"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(GAMES) and os.path.exists(PBP) and os.path.exists(ARTIFACT)),
    reason="backfill/artifact not present",
)


def test_complete_features_produce_nonnull_finite_fundamental():
    g = pd.read_parquet(GAMES)
    pbp = pd.read_parquet(PBP)
    wk = g[(g["season"] == 2024) & (g["week"] == 10)].copy()
    wk["home_spread"] = pd.to_numeric(wk["home_spread_reference"], errors="coerce")
    wk["total_line"] = pd.to_numeric(wk["total_line_reference"], errors="coerce")
    wk = wk.dropna(subset=["home_spread", "total_line"])
    asof = "2024-11-07T12:00:00Z"
    feats, _ = build_live_augmented_features(g, wk, pbp, mode="historical_replay", as_of_utc=asof)
    fund = score_shadow(feats)
    assert fund is not None
    for col in ("fundamental_home_win", "fundamental_home_cover", "fundamental_over"):
        v = fund[col].to_numpy(float)
        assert np.isfinite(v).all()
        assert ((v >= 0) & (v <= 1)).all()


def test_fundamental_is_not_a_market_copy():
    # attach with a fundamental that differs from market; ensure columns stay distinct
    card = pd.DataFrame({
        "game_id": ["g1"], "market": ["moneyline"], "market_fair_probability": [0.55],
    })
    fundamental = pd.DataFrame({
        "game_id": ["g1"], "fundamental_home_win": [0.62],
        "fundamental_home_cover": [0.5], "fundamental_over": [0.5],
    })
    out = attach_shadow_columns(card, fundamental, status="AVAILABLE", artifact_version="v1")
    row = out.iloc[0]
    assert row["fundamental_probability"] == 0.62
    assert row["market_fair_probability"] == 0.55
    assert row["production_probability"] == 0.55
    assert row["production_source"] == "MARKET_BASELINE"


def test_missing_fundamental_lists_features_and_stays_null():
    card = pd.DataFrame({"game_id": ["g1"], "market": ["ats"], "market_fair_probability": [0.5]})
    out = attach_shadow_columns(card, None, status="UNAVAILABLE", artifact_version=None)
    assert pd.isna(out.iloc[0]["fundamental_probability"])
    assert out.iloc[0]["fundamental_missing_feature_count"] == len(FROZEN_FEATURES)


def test_clean_checkout_artifact_loads_with_19_features():
    model, features, meta = load_shadow()
    assert model is not None
    assert features == FROZEN_FEATURES
    assert len(features) == 19
    assert meta["artifact"]
