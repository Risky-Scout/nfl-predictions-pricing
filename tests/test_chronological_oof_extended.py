"""EXTENDED real-data proof for the Fix 3 chronological OOF engine
(:mod:`nfl_hybrid.evaluation.chronological_oof`).

Exercises the engine against the private 2020-2025 backfill parquets, so this
module is marked ``extended_data`` and skips cleanly when
``NFL_MODEL_DATA_ROOT`` (or the backfill assets under it) is unavailable --
e.g. GitHub CI, or any machine without the private data estate. It
complements, and does not replace, the synthetic adversarial suite in
:mod:`tests.test_chronological_oof`, which is the actual leakage-mechanism
proof and runs unconditionally.

This module proves the real-data membership invariant the task specification
asks for explicitly: for a representative real target game,
``max_training_result_available_at_utc < target_cutoff_utc`` and
``target_game_id not in training_game_ids`` -- on the real 2023-season
schema/volume, not a hand-built toy sequence. Eligibility is decided by
``result_available_at_utc`` (:func:`nfl_hybrid.evaluation.chronological_oof.compute_result_available_at_utc`),
not merely by kickoff order -- see the module docstring in
``chronological_oof.py`` for why kickoff order is not resolution order. A
handful of representative target games are evaluated (via
``target_row_positions``) rather than the full season, so this stays fast; the
full season's real games/PBP are still used, unmodified, as the eligible
training pool for every one of them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_hybrid.data.external_data import ExternalDataUnavailableError, resolve
from nfl_hybrid.evaluation.chronological_oof import (
    RESULT_AVAILABILITY_DURATION_HOURS,
    ChronologicalOOFConfig,
    attach_expanding_oof_uncertainty,
    attach_outcomes_and_residuals,
    build_oof_feature_matrix,
    generate_oof_predictions,
)
from nfl_hybrid.features.pbp_advanced import aggregate_qb_game_efficiency


def _try_resolve(key):
    try:
        return resolve(key)
    except ExternalDataUnavailableError:
        return None


GAMES_PATH = _try_resolve("backfill.games")
PBP_PATH = _try_resolve("backfill.pbp")
pytestmark = [
    pytest.mark.extended_data,
    pytest.mark.skipif(
        GAMES_PATH is None or PBP_PATH is None,
        reason="NFL_MODEL_DATA_ROOT not configured or backfill assets missing (extended_data)",
    ),
]

# Same choice, and the same rationale, as tests/test_pregame_state_extended.py:
# one full real season is real data end-to-end and keeps this fast, without
# needing the full 2020-2025 estate loaded at once.
REPRESENTATIVE_SEASON = 2023
MIN_TRAINING_GAMES = 60


@pytest.fixture(scope="module")
def real_feature_matrix():
    games = pd.read_parquet(GAMES_PATH)
    pbp = pd.read_parquet(PBP_PATH)

    season_games = games[games["season"] == REPRESENTATIVE_SEASON].copy()
    assert len(season_games) > 0, f"no games found for season {REPRESENTATIVE_SEASON}"
    season_game_ids = set(season_games["game_id"].astype(str))
    season_pbp = pbp[pbp["game_id"].astype(str).isin(season_game_ids)].copy()
    qb_game = aggregate_qb_game_efficiency(season_pbp)

    matrix, feature_columns, bundle = build_oof_feature_matrix(season_games, season_pbp, qb_game=qb_game)
    return season_games, matrix, feature_columns


def test_real_feature_matrix_covers_every_season_game(real_feature_matrix):
    season_games, matrix, _ = real_feature_matrix
    assert set(matrix["game_id"]) == set(season_games["game_id"].astype(str))


def test_result_available_at_utc_meaningfully_delays_availability_on_real_data(real_feature_matrix):
    """Confirms the availability policy is doing real work on real data, not
    degenerating back to kickoff order: every game's result becomes
    available strictly after its kickoff, by at least the documented floor,
    and the Tue/Fri batch snap pushes at least some real games further out
    than that floor alone would."""
    _, matrix, _feature_columns = real_feature_matrix
    kickoff = pd.to_datetime(matrix["scheduled_kickoff_utc"], utc=True)
    available = pd.to_datetime(matrix["result_available_at_utc"], utc=True)
    assert (available >= kickoff).all()
    delay_hours = (available - kickoff).dt.total_seconds() / 3600.0
    assert delay_hours.min() >= RESULT_AVAILABILITY_DURATION_HOURS - 1e-9
    assert delay_hours.max() > RESULT_AVAILABILITY_DURATION_HOURS  # batch snap delays some games further


def test_representative_real_target_game_membership_proof(real_feature_matrix):
    """The exact real-data proof the specification asks for:
    max_training_result_available_at_utc < target_cutoff_utc and
    target_game_id not in training_game_ids -- for a representative
    mid-season real target game, with eligibility decided by
    result_available_at_utc, not merely kickoff order."""
    _, matrix, feature_columns = real_feature_matrix
    target_position = len(matrix) // 2  # mid-season: real, substantial prior history

    preds = generate_oof_predictions(
        matrix,
        feature_columns=feature_columns,
        config=ChronologicalOOFConfig(min_training_games=MIN_TRAINING_GAMES),
        target_row_positions=[target_position],
    )
    assert len(preds) == 1
    row = preds.iloc[0]
    assert row["status"] == "OOF", "representative target lacked sufficient real warmup history"

    target_game_id = row["game_id"]
    target_cutoff = row["target_cutoff_utc"]
    max_training_result_available = row["max_training_result_available_at_utc"]
    training_ids = row["training_game_ids"]

    print(
        f"\n[Fix 3 real-data membership proof] season={REPRESENTATIVE_SEASON} "
        f"target_game_id={target_game_id} target_cutoff_utc={target_cutoff} "
        f"max_training_result_available_at_utc={max_training_result_available} "
        f"training_game_count={row['training_game_count']}"
    )

    assert max_training_result_available < target_cutoff, (
        "max_training_result_available_at_utc < target_cutoff_utc"
    )
    assert target_game_id not in training_ids, "target_game_id not in training_game_ids"
    # target_cutoff_utc, training_as_of_utc, and max_training_result_available_at_utc
    # are kept as separate, explicitly-named fields -- never substituted for
    # one another -- and the provenance of the availability timestamps is
    # recorded explicitly as a conservative derived policy, not an observed
    # final-whistle time.
    assert row["training_as_of_utc"] == target_cutoff
    assert row["result_availability_basis"] == "HISTORICAL_CONSERVATIVE_BATCH"

    # Directly re-derive the max training result-available time from the real
    # feature matrix (not just trusting the ledger's own bookkeeping).
    available_by_id = dict(
        zip(matrix["game_id"], pd.to_datetime(matrix["result_available_at_utc"], utc=True))
    )
    real_max_training_available = max(available_by_id[gid] for gid in training_ids)
    assert real_max_training_available == max_training_result_available
    assert real_max_training_available < target_cutoff


def test_several_representative_real_targets_all_satisfy_membership_invariant(real_feature_matrix):
    _, matrix, feature_columns = real_feature_matrix
    n = len(matrix)
    positions = sorted({n // 4, n // 2, (3 * n) // 4, n - 1})

    preds = generate_oof_predictions(
        matrix,
        feature_columns=feature_columns,
        config=ChronologicalOOFConfig(min_training_games=MIN_TRAINING_GAMES),
        target_row_positions=positions,
    )
    assert len(preds) == len(positions)

    available_by_id = dict(
        zip(matrix["game_id"], pd.to_datetime(matrix["result_available_at_utc"], utc=True))
    )
    for row in preds.itertuples(index=False):
        assert row.game_id not in row.training_game_ids
        if row.status == "OOF":
            assert row.max_training_result_available_at_utc < row.target_cutoff_utc
            for tid in row.training_game_ids:
                assert available_by_id[tid] < row.target_cutoff_utc


def test_real_data_oof_residual_ledger_and_uncertainty_are_internally_consistent(real_feature_matrix):
    _, matrix, feature_columns = real_feature_matrix
    n = len(matrix)
    positions = list(range(n // 2, n // 2 + 6))  # a short real, contiguous OOF run

    preds = generate_oof_predictions(
        matrix,
        feature_columns=feature_columns,
        config=ChronologicalOOFConfig(min_training_games=MIN_TRAINING_GAMES),
        target_row_positions=positions,
    )
    ledger = attach_outcomes_and_residuals(preds, matrix)
    oof_rows = ledger[ledger["status"] == "OOF"]
    assert len(oof_rows) > 0
    assert oof_rows[["predicted_margin", "predicted_total", "actual_margin", "actual_total"]].notna().all().all()

    unc = attach_expanding_oof_uncertainty(ledger, min_uncertainty_warmup=2)
    # every row's uncertainty (if eligible) must come strictly from residuals
    # of games this module already proved are strictly-prior real games.
    assert set(unc["game_id"]) == set(ledger["game_id"])
