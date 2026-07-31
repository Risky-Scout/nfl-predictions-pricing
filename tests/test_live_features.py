"""Live/training feature parity + leakage tests for build_live_augmented_features.

These exercise the real feature pipeline, so they require the 2020-2025 backfill
parquets; the module skips cleanly when those are absent (e.g. minimal CI).
"""

import os

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import (
    FROZEN_FEATURES,
    build_augmented_feature_matrix,
    build_live_augmented_features,
)

GAMES = "data/backfill_2020_2025/canonical/games.parquet"
PBP = "data/backfill_2020_2025/raw/pbp.parquet"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(GAMES) and os.path.exists(PBP)),
    reason="backfill parquets not present",
)


@pytest.fixture(scope="module")
def data():
    g = pd.read_parquet(GAMES)
    pbp = pd.read_parquet(PBP)
    matrix, _ = build_augmented_feature_matrix(g, pbp)
    return g, pbp, matrix


def _target(g, gid):
    t = g[g["game_id"].astype(str) == gid].copy()
    t["home_spread"] = pd.to_numeric(t["home_spread_reference"], errors="coerce")
    t["total_line"] = pd.to_numeric(t["total_line_reference"], errors="coerce")
    return t


def _asof_before(t, minutes=10):
    return (pd.to_datetime(t["scheduled_kickoff_utc"].iloc[0], utc=True) - pd.Timedelta(minutes=minutes)).isoformat()


@pytest.mark.parametrize("season,week", [(2023, 6), (2024, 10), (2022, 14)])
def test_live_training_parity(data, season, week):
    g, pbp, matrix = data
    gid = str(matrix[(matrix["season"] == season) & (matrix["week"] == week)].head(1)["game_id"].iloc[0])
    t = _target(g, gid)
    feats, status = build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))
    hist = matrix[matrix["game_id"].astype(str) == gid].iloc[0]
    assert status["feature_row_count"] == 1
    assert list(feats.columns) == ["game_id"] + FROZEN_FEATURES  # exact order
    for c in FROZEN_FEATURES:
        lv, hv = feats[c].iloc[0], hist[c]
        assert pd.isna(lv) == pd.isna(hv), f"NaN mismatch {c}"
        if pd.notna(lv):
            assert abs(float(lv) - float(hv)) < 1e-10, f"value mismatch {c}"


def test_target_score_invariance(data):
    g, pbp, matrix = data
    gid = str(matrix[(matrix["season"] == 2024) & (matrix["week"] == 10)].head(1)["game_id"].iloc[0])
    t = _target(g, gid)
    a, _ = build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))
    g2 = g.copy()
    m = g2["game_id"].astype(str) == gid
    g2.loc[m, "home_score"] = 999  # corrupt the target's own result
    g2.loc[m, "away_score"] = 0
    b, _ = build_live_augmented_features(g2, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))
    for c in FROZEN_FEATURES:
        assert pd.isna(a[c].iloc[0]) == pd.isna(b[c].iloc[0])
        if pd.notna(a[c].iloc[0]):
            assert abs(float(a[c].iloc[0]) - float(b[c].iloc[0])) < 1e-12


def test_target_pbp_invariance(data):
    g, pbp, matrix = data
    gid = str(matrix[(matrix["season"] == 2024) & (matrix["week"] == 10)].head(1)["game_id"].iloc[0])
    t = _target(g, gid)
    a, _ = build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))
    pbp2 = pbp.copy()
    pm = pbp2["game_id"].astype(str) == gid
    pbp2.loc[pm, "epa"] = 50.0  # corrupt the target's own plays
    b, _ = build_live_augmented_features(g, t, pbp2, mode="historical_replay", as_of_utc=_asof_before(t))
    for c in FROZEN_FEATURES:
        if pd.notna(a[c].iloc[0]):
            assert abs(float(a[c].iloc[0]) - float(b[c].iloc[0])) < 1e-12


def test_target_kickoff_at_or_before_asof_fails(data):
    g, pbp, matrix = data
    gid = str(matrix[(matrix["season"] == 2024) & (matrix["week"] == 10)].head(1)["game_id"].iloc[0])
    t = _target(g, gid)
    asof = pd.to_datetime(t["scheduled_kickoff_utc"].iloc[0], utc=True).isoformat()  # exactly kickoff
    with pytest.raises(ValueError, match="at or before"):
        build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=asof)


def test_duplicate_target_fails(data):
    g, pbp, matrix = data
    gid = str(matrix[(matrix["season"] == 2024) & (matrix["week"] == 10)].head(1)["game_id"].iloc[0])
    t = pd.concat([_target(g, gid), _target(g, gid)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate target"):
        build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))


def test_post_asof_game_excluded_changes_nothing_earlier(data):
    # a later completed game must not change an earlier target's features
    g, pbp, matrix = data
    early = str(matrix[(matrix["season"] == 2024) & (matrix["week"] == 6)].head(1)["game_id"].iloc[0])
    t = _target(g, early)
    a, sa = build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))
    # as-of at week 6 excludes weeks 7+; the max eligible kickoff must be < target kickoff
    assert pd.Timestamp(sa["max_eligible_completed_kickoff"]) < pd.to_datetime(t["scheduled_kickoff_utc"].iloc[0], utc=True)


def test_end_to_end_no_partial_aggregation(data):
    # A prior contributing game that is not yet eligible must not reach aggregation, and
    # its EPA must not affect the target's features. Once eligible, features change.
    g, pbp, matrix = data
    tgt = g[(g["season"] == 2024) & (g["week"] == 10)].sort_values("scheduled_kickoff_utc").iloc[0]
    home = tgt["home_team_id"]
    prior = g[(g["season"] == 2024) & (g["week"] == 9)
              & ((g["home_team_id"] == home) | (g["away_team_id"] == home))].iloc[0]
    prior_ko = pd.to_datetime(prior["scheduled_kickoff_utc"], utc=True)
    prior_gid = str(prior["game_id"])
    target = _target(g, str(tgt["game_id"]))

    # as_of within the prior game's availability window (2h after kickoff) -> excluded
    asof_A = (prior_ko + pd.Timedelta(hours=2)).isoformat()
    a, sa = build_live_augmented_features(g, target, pbp, mode="historical_replay", as_of_utc=asof_A)
    # as_of past the availability window (6h) -> the prior game is eligible
    asof_B = (prior_ko + pd.Timedelta(hours=6)).isoformat()
    b, sb = build_live_augmented_features(g, target, pbp, mode="historical_replay", as_of_utc=asof_B)

    # the newly-eligible game increases the eligible set and changes the target's features
    assert sb["eligible_final_game_count"] > sa["eligible_final_game_count"]
    assert sa["feature_snapshot_hash"] != sb["feature_snapshot_hash"]
    changed = [c for c in FROZEN_FEATURES
               if (pd.isna(a[c].iloc[0]) != pd.isna(b[c].iloc[0]))
               or (pd.notna(a[c].iloc[0]) and pd.notna(b[c].iloc[0]) and abs(float(a[c].iloc[0]) - float(b[c].iloc[0])) > 1e-9)]
    assert changed  # at least one EPA/rest feature changed, explained by the newly-eligible game


def test_live_mode_without_evidence_yields_no_features(data):
    # real data lacks a finality feed -> live mode is fail-closed -> all EPA features NaN
    g, pbp, matrix = data
    gid = str(matrix[(matrix["season"] == 2024) & (matrix["week"] == 10)].head(1)["game_id"].iloc[0])
    t = _target(g, gid)
    feats, st = build_live_augmented_features(g, t, pbp, mode="live", as_of_utc=_asof_before(t))
    assert st["verified_final_game_count"] == 0
    assert st.get("no_eligible_completed_games") is True
    assert pd.isna(feats["epa_net_edge_season_mean"].iloc[0])
    # market lines still attach
    assert pd.notna(feats["home_spread"].iloc[0])


def test_deterministic_hash(data):
    g, pbp, matrix = data
    gid = str(matrix[(matrix["season"] == 2024) & (matrix["week"] == 10)].head(1)["game_id"].iloc[0])
    t = _target(g, gid)
    _, s1 = build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))
    _, s2 = build_live_augmented_features(g, t, pbp, mode="historical_replay", as_of_utc=_asof_before(t))
    assert s1["feature_snapshot_hash"] == s2["feature_snapshot_hash"]
