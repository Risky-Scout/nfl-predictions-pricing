"""Fix 3 mandatory adversarial proof suite for the chronological OOF prediction
and uncertainty engine (:mod:`nfl_hybrid.evaluation.chronological_oof`).

Kept deliberately cheap: every synthetic fixture here is small (<=30 rows) and
narrow (two numeric features), `tests/conftest.py` forces every BLAS/OpenMP
backend to a single thread for the whole test session, and the engine itself
(``generate_oof_predictions``) fits one model per unique forecast-cutoff
*batch*, not per game -- a same-slate batch of games is fit once and predicted
together. None of that changes any assertion's strength; it only removes
redundant model fits and thread-pool contention. The two integration tests
that exercise the real Fix 2 interface share one small, module-scoped
pregame-state build instead of rebuilding it twice.

Correction (result availability, not just kickoff order): training/uncertainty
eligibility is ``result_available_at_utc < cutoff``, never merely
``scheduled_kickoff_utc < cutoff``. Every synthetic fixture below carries an
explicit ``result_available_at_utc`` column (default: kickoff + 1h, always
comfortably before the next row's cutoff) so the "ordinary" tests exercise the
real contract, and several tests deliberately diverge kickoff order from
result-availability order to prove the distinction is actually enforced.

Four groups of tests:

  Engine-level adversarial proofs (synthetic, hand-built feature matrices --
  full control over exactly which row's value changed, so every mutation's
  effect can be checked precisely):
    - target game absent from its own training IDs
    - future games absent from training
    - mutating a target's own outcome never changes its own OOF prediction
    - mutating a future outcome never changes an earlier OOF prediction
    - mutating a legitimate prior game's outcome CAN change a later prediction
    - a target's own residual never changes its own uncertainty
    - a future residual never changes an earlier row's uncertainty
    - fitted-training-residual attributes are never read for uncertainty
    - chronological order is deterministic; identical inputs => identical output
    - shuffled input rows do not change the (game_id-keyed) result
    - insufficient warmup is explicitly marked, never silently trained anyway

  Result-availability proofs (the correction itself):
    - an earlier-kickoff game whose result isn't available until after the
      target's cutoff is excluded (same-day/out-of-order leakage test)
    - positive control: an earlier-kickoff game whose result IS available
      before the target's cutoff may enter training
    - a residual is excluded from uncertainty until its own game's result is
      available, and may enter a later target's uncertainty once it is
    - mutating an unresolved game's residual cannot affect current uncertainty
    - batching remains exact (one fit per batch) even when a prior game's
      result is delayed past other games' cutoffs

  Batching-correctness proofs (the performance optimization itself):
    - games sharing one forecast cutoff (a slate) are fit exactly once, not
      once per game, and get the exact predictions a single shared fit would
      produce
    - batching never changes which games are eligible to train on which
      target (same membership/chronology invariants, on a slate fixture)

  Integration proofs (small synthetic games/play-by-play, run through the
  real end-to-end pipeline, pregame-state build shared across both tests):
    - the Fix 2 authoritative pregame-state interface is genuinely used
    - the full pipeline produces a well-formed, internally consistent ledger
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation import chronological_oof as oof
from nfl_hybrid.evaluation.chronological_oof import (
    ChronologicalOOFConfig,
    attach_expanding_oof_uncertainty,
    attach_outcomes_and_residuals,
    build_oof_feature_matrix,
    generate_oof_predictions,
    production_uncertainty,
    run_chronological_oof,
    training_membership_hash,
)
from nfl_hybrid.modern.joint_score import JointScoreModel


# ---------------------------------------------------------------------------
# Synthetic feature-matrix fixture (engine-level tests) -- small and narrow
# ---------------------------------------------------------------------------

N_SYNTHETIC = 26  # >= min_training_games + a handful of OOF rows either side


def _synthetic_matrix(
    n: int = N_SYNTHETIC,
    seed: int = 0,
    margin_overrides: dict[int, float] | None = None,
    total_overrides: dict[int, float] | None = None,
    result_available_overrides: dict[int, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    home_margin = 3.0 + 6.0 * x1 + rng.normal(scale=8.0, size=n)
    total_points = 44.0 + 3.0 * x2 + rng.normal(scale=7.0, size=n)
    for idx, value in (margin_overrides or {}).items():
        home_margin[idx] = value
    for idx, value in (total_overrides or {}).items():
        total_points[idx] = value
    kickoff = pd.date_range("2024-09-01", periods=n, freq="D", tz="UTC")
    result_available = kickoff + pd.Timedelta(hours=1)  # default: comfortably resolved well before the next row's cutoff
    frame = pd.DataFrame(
        {
            "game_id": [f"G{i:03d}" for i in range(n)],
            "season": 2024,
            "week": (np.arange(n) // 4) + 1,
            "scheduled_kickoff_utc": kickoff,
            "result_available_at_utc": result_available,
            "feat_x1": x1,
            "feat_x2": x2,
            "home_margin": home_margin,
            "total_points": total_points,
        }
    )
    for idx, value in (result_available_overrides or {}).items():
        frame.loc[idx, "result_available_at_utc"] = value
    return frame


FEATURE_COLUMNS = ["feat_x1", "feat_x2"]
SMALL_CONFIG = ChronologicalOOFConfig(min_training_games=6)


def _index_of(game_id: str) -> int:
    return int(game_id[1:])


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_target_game_absent_from_own_training_ids():
    frame = _synthetic_matrix()
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    checked = 0
    for row in preds.itertuples(index=False):
        if row.training_game_ids_truncated:
            continue
        assert row.game_id not in row.training_game_ids
        checked += 1
    assert checked > 0


def test_future_games_absent_from_training():
    frame = _synthetic_matrix()
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    checked = 0
    for row in preds.itertuples(index=False):
        if row.training_game_ids_truncated:
            continue
        target_idx = _index_of(row.game_id)
        for tid in row.training_game_ids:
            assert _index_of(tid) < target_idx, (row.game_id, tid)
            checked += 1
    assert checked > 0


def test_real_membership_proof_max_training_result_available_before_target_cutoff():
    """The exact real-data-proof invariant, checked here on every OOF row of
    the synthetic suite: max_training_result_available_at_utc < target_cutoff
    and the target's own id is not among the training ids."""
    frame = _synthetic_matrix()
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    oof_rows = preds[preds["status"] == "OOF"]
    assert len(oof_rows) > 0
    for row in oof_rows.itertuples(index=False):
        assert row.max_training_result_available_at_utc < row.target_cutoff_utc
        assert row.game_id not in row.training_game_ids or row.training_game_ids_truncated


# ---------------------------------------------------------------------------
# Target-game / future-game invariance
# ---------------------------------------------------------------------------

def test_mutating_targets_own_outcome_does_not_change_its_own_oof_prediction():
    base = generate_oof_predictions(_synthetic_matrix(), feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    target_idx = N_SYNTHETIC - 6
    mutated_frame = _synthetic_matrix(
        margin_overrides={target_idx: 999.0}, total_overrides={target_idx: -999.0}
    )
    mutated = generate_oof_predictions(mutated_frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)

    b = base.iloc[target_idx]
    m = mutated.iloc[target_idx]
    assert b["game_id"] == m["game_id"] == f"G{target_idx:03d}"
    assert b["status"] == m["status"] == "OOF"
    assert b["predicted_margin"] == m["predicted_margin"]
    assert b["predicted_total"] == m["predicted_total"]


def test_mutating_a_future_outcome_does_not_change_earlier_oof_predictions():
    base = generate_oof_predictions(_synthetic_matrix(), feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    late_idx = N_SYNTHETIC - 2
    mutated_frame = _synthetic_matrix(
        margin_overrides={late_idx: 999.0}, total_overrides={late_idx: -999.0}
    )
    mutated = generate_oof_predictions(mutated_frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)

    earlier = base.iloc[:late_idx]
    earlier_mutated = mutated.iloc[:late_idx]
    pd.testing.assert_frame_equal(earlier, earlier_mutated)


def test_mutating_a_legitimate_prior_game_can_change_a_later_prediction():
    base = generate_oof_predictions(_synthetic_matrix(), feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    prior_idx = 7
    mutated_frame = _synthetic_matrix(
        margin_overrides={prior_idx: 500.0}, total_overrides={prior_idx: 500.0}
    )
    mutated = generate_oof_predictions(mutated_frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)

    later = base[base["game_id"].map(_index_of) > prior_idx]
    later_mutated = mutated[mutated["game_id"].map(_index_of) > prior_idx]
    changed = (
        (later["predicted_margin"].to_numpy() != later_mutated["predicted_margin"].to_numpy())
        | (later["predicted_total"].to_numpy() != later_mutated["predicted_total"].to_numpy())
    )
    assert changed.any(), "mutating a legitimate prior training game changed no downstream prediction"


# ---------------------------------------------------------------------------
# Result-availability proofs (the correction): kickoff order != resolution order
# ---------------------------------------------------------------------------

def test_same_day_leakage_earlier_kickoff_but_late_result_excluded():
    """Game A kicks off before Game B, but A's result isn't available until
    AFTER B's forecast cutoff -- A must be excluded from B's training set
    despite kicking off first. This is exactly the gap the kickoff-only rule
    (the module's first, incorrect version) missed."""
    a_idx, b_idx = 10, 11
    base_frame = _synthetic_matrix()
    b_cutoff = base_frame.loc[b_idx, "scheduled_kickoff_utc"] - pd.Timedelta(minutes=10)
    late_result = b_cutoff + pd.Timedelta(hours=1)  # confirmed only AFTER B's own cutoff
    frame = _synthetic_matrix(result_available_overrides={a_idx: late_result})
    assert frame.loc[a_idx, "scheduled_kickoff_utc"] < frame.loc[b_idx, "scheduled_kickoff_utc"]
    assert frame.loc[a_idx, "result_available_at_utc"] > b_cutoff

    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    a_game_id, b_game_id = frame.loc[a_idx, "game_id"], frame.loc[b_idx, "game_id"]
    b_row = preds[preds["game_id"] == b_game_id].iloc[0]

    assert b_row["status"] == "OOF"
    assert a_game_id not in b_row["training_game_ids"]
    assert b_row["training_game_count"] == b_idx - 1  # every prior game except the unresolved A


def test_positive_control_early_result_available_may_enter_training():
    """Mirror of the leakage test: if A's result IS available before B's
    cutoff, A may enter B's training set."""
    a_idx, b_idx = 10, 11
    base_frame = _synthetic_matrix()
    b_cutoff = base_frame.loc[b_idx, "scheduled_kickoff_utc"] - pd.Timedelta(minutes=10)
    early_result = b_cutoff - pd.Timedelta(hours=1)  # comfortably resolved before B's cutoff
    frame = _synthetic_matrix(result_available_overrides={a_idx: early_result})

    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    a_game_id, b_game_id = frame.loc[a_idx, "game_id"], frame.loc[b_idx, "game_id"]
    b_row = preds[preds["game_id"] == b_game_id].iloc[0]

    assert b_row["status"] == "OOF"
    assert a_game_id in b_row["training_game_ids"]
    assert b_row["training_game_count"] == b_idx  # every prior game, including A


def test_residual_excluded_until_available_then_may_enter_later_uncertainty():
    g_idx = 12
    base_frame = _synthetic_matrix()
    late_result = base_frame.loc[18, "scheduled_kickoff_utc"]  # far later than g_idx's own kickoff+1h
    frame = _synthetic_matrix(result_available_overrides={g_idx: late_result})

    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    ledger = attach_outcomes_and_residuals(preds, frame)
    unc = attach_expanding_oof_uncertainty(ledger, min_uncertainty_warmup=3)
    g_game_id = frame.loc[g_idx, "game_id"]

    early_target_idx = 15
    early_gid = frame.loc[early_target_idx, "game_id"]
    early_row = unc[unc["game_id"] == early_gid].iloc[0]
    assert early_row["target_cutoff_utc"] < late_result  # g_idx not yet resolved by this cutoff
    eligible_before = ledger[
        (ledger["status"] == "OOF")
        & (ledger["result_available_at_utc"] < early_row["target_cutoff_utc"])
    ]
    assert g_game_id not in eligible_before["game_id"].to_numpy()
    expected_sd = float(np.std(eligible_before["margin_residual"].to_numpy(float), ddof=1))
    assert early_row["margin_residual_sd_oof"] == pytest.approx(expected_sd)

    late_target_idx = N_SYNTHETIC - 1
    late_gid = frame.loc[late_target_idx, "game_id"]
    late_row = unc[unc["game_id"] == late_gid].iloc[0]
    assert late_row["target_cutoff_utc"] > late_result  # g_idx resolved well before this cutoff
    eligible_late = ledger[
        (ledger["status"] == "OOF")
        & (ledger["result_available_at_utc"] < late_row["target_cutoff_utc"])
    ]
    assert g_game_id in eligible_late["game_id"].to_numpy()
    expected_sd_late = float(np.std(eligible_late["margin_residual"].to_numpy(float), ddof=1))
    assert late_row["margin_residual_sd_oof"] == pytest.approx(expected_sd_late)


def test_changing_unresolved_outcome_cannot_affect_current_uncertainty():
    g_idx = 12
    base_frame = _synthetic_matrix()
    late_result = base_frame.loc[18, "scheduled_kickoff_utc"]
    frame = _synthetic_matrix(result_available_overrides={g_idx: late_result})

    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    ledger = attach_outcomes_and_residuals(preds, frame)
    base_unc = attach_expanding_oof_uncertainty(ledger, min_uncertainty_warmup=3)

    g_game_id = frame.loc[g_idx, "game_id"]
    mutated = ledger.copy()
    g_mask = mutated["game_id"] == g_game_id
    mutated.loc[g_mask, "margin_residual"] = 888.0
    mutated.loc[g_mask, "total_residual"] = -888.0
    mutated_unc = attach_expanding_oof_uncertainty(mutated, min_uncertainty_warmup=3)

    early_gid = frame.loc[15, "game_id"]
    cols = ["margin_residual_sd_oof", "total_residual_sd_oof", "residual_correlation_oof"]
    b = base_unc.loc[base_unc["game_id"] == early_gid, cols].reset_index(drop=True)
    m = mutated_unc.loc[mutated_unc["game_id"] == early_gid, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, m)


# ---------------------------------------------------------------------------
# Uncertainty invariants (unaffected by which column decides eligibility)
# ---------------------------------------------------------------------------

def _ledger_for_uncertainty(n=N_SYNTHETIC, seed=0) -> pd.DataFrame:
    frame = _synthetic_matrix(n=n, seed=seed)
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    return attach_outcomes_and_residuals(preds, frame)


def test_target_residual_cannot_change_its_own_uncertainty():
    ledger = _ledger_for_uncertainty()
    base_unc = attach_expanding_oof_uncertainty(ledger, min_uncertainty_warmup=3)

    target_idx = N_SYNTHETIC - 5
    mutated = ledger.copy()
    assert mutated.at[target_idx, "status"] == "OOF"
    mutated.at[target_idx, "actual_margin"] = mutated.at[target_idx, "actual_margin"] + 500.0
    mutated.at[target_idx, "margin_residual"] = (
        mutated.at[target_idx, "actual_margin"] - mutated.at[target_idx, "predicted_margin"]
    )
    mutated_unc = attach_expanding_oof_uncertainty(mutated, min_uncertainty_warmup=3)

    row_cols = ["margin_residual_sd_oof", "total_residual_sd_oof", "residual_correlation_oof"]
    pd.testing.assert_series_equal(
        base_unc.iloc[target_idx][row_cols], mutated_unc.iloc[target_idx][row_cols]
    )
    # rows strictly before the mutated one are also unaffected
    pd.testing.assert_frame_equal(
        base_unc.iloc[:target_idx][row_cols], mutated_unc.iloc[:target_idx][row_cols]
    )


def test_future_residual_cannot_change_earlier_uncertainty():
    ledger = _ledger_for_uncertainty()
    base_unc = attach_expanding_oof_uncertainty(ledger, min_uncertainty_warmup=3)

    late_idx = N_SYNTHETIC - 3
    mutated = ledger.copy()
    assert mutated.at[late_idx, "status"] == "OOF"
    mutated.at[late_idx, "margin_residual"] = 750.0
    mutated.at[late_idx, "total_residual"] = -750.0
    mutated_unc = attach_expanding_oof_uncertainty(mutated, min_uncertainty_warmup=3)

    row_cols = ["margin_residual_sd_oof", "total_residual_sd_oof", "residual_correlation_oof"]
    pd.testing.assert_frame_equal(
        base_unc.iloc[:late_idx][row_cols], mutated_unc.iloc[:late_idx][row_cols]
    )


def test_fitted_training_residual_attributes_never_used_for_uncertainty():
    """Decisive rejection proof: the uncertainty/prediction functions' source
    never references JointScoreModel's fitted-training-residual attributes."""
    # Leading "." matches the actual failure mode -- reading the attribute off
    # a fitted model (``model.margin_sd_`` etc). This deliberately does not
    # match this module's own, differently-named ledger columns
    # (``margin_residual_sd_oof``, ``residual_correlation_oof``, ...).
    banned = (".margin_sd_", ".total_sd_", ".residual_correlation_")
    for fn in (
        oof.attach_expanding_oof_uncertainty,
        oof.production_uncertainty,
        oof.generate_oof_predictions,
    ):
        source = inspect.getsource(fn)
        for token in banned:
            assert token not in source, (fn.__name__, token)


def test_production_uncertainty_only_uses_residuals_before_as_of():
    ledger = _ledger_for_uncertainty()
    as_of = ledger.loc[N_SYNTHETIC - 8, "result_available_at_utc"] + pd.Timedelta(minutes=1)
    result = production_uncertainty(ledger, as_of_utc=as_of, min_uncertainty_warmup=3)
    assert result["status"] == "OK"

    eligible = ledger[(ledger["status"] == "OOF") & (ledger["result_available_at_utc"] < as_of)]
    expected_sd = float(np.std(eligible["margin_residual"].to_numpy(float), ddof=1))
    assert result["margin_residual_sd"] == pytest.approx(expected_sd)
    assert result["n_residuals"] == len(eligible)

    # mutating a row at/after as_of must never move the estimate.
    mutated = ledger.copy()
    future_rows = mutated.index[mutated["result_available_at_utc"] >= as_of]
    mutated.loc[future_rows, "margin_residual"] = 999.0
    mutated.loc[future_rows, "total_residual"] = -999.0
    result_after = production_uncertainty(mutated, as_of_utc=as_of, min_uncertainty_warmup=3)
    assert result_after == result


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_identical_oof_predictions():
    frame = _synthetic_matrix()
    first = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    second = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    pd.testing.assert_frame_equal(first, second)


def test_shuffled_input_rows_do_not_change_chronological_results():
    frame = _synthetic_matrix()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    first = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    second = generate_oof_predictions(shuffled, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)

    a = first.sort_values("game_id").reset_index(drop=True)
    b = second.sort_values("game_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_insufficient_warmup_is_marked_and_never_trains_on_future_data():
    frame = _synthetic_matrix(n=14)
    preds = generate_oof_predictions(
        frame, feature_columns=FEATURE_COLUMNS, config=ChronologicalOOFConfig(min_training_games=1000)
    )
    assert (preds["status"] == "INSUFFICIENT_WARMUP").all()
    assert preds["predicted_margin"].isna().all()
    assert preds["predicted_total"].isna().all()

    early = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    warmup_rows = early[early["status"] == "INSUFFICIENT_WARMUP"]
    assert len(warmup_rows) > 0
    for row in warmup_rows.itertuples(index=False):
        assert row.training_game_count < SMALL_CONFIG.min_training_games
        target_idx = _index_of(row.game_id)
        for tid in row.training_game_ids:
            assert _index_of(tid) < target_idx


def test_uncertainty_warmup_below_two_is_rejected():
    ledger = _ledger_for_uncertainty(n=14)
    with pytest.raises(ValueError):
        attach_expanding_oof_uncertainty(ledger, min_uncertainty_warmup=1)
    with pytest.raises(ValueError):
        production_uncertainty(ledger, as_of_utc=ledger.loc[10, "target_cutoff_utc"], min_uncertainty_warmup=1)


def test_training_membership_hash_is_order_independent():
    ids = ["G003", "G001", "G002"]
    assert training_membership_hash(ids) == training_membership_hash(list(reversed(ids)))
    assert training_membership_hash(ids) != training_membership_hash(["G001", "G002"])


def test_result_available_at_utc_column_is_required():
    frame = _synthetic_matrix().drop(columns=["result_available_at_utc"])
    with pytest.raises(ValueError):
        generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)


def test_ledger_keeps_target_cutoff_training_as_of_and_max_training_available_distinct():
    """The three cutoff-shaped fields are separate, explicitly-named columns
    -- never one substituted for another -- even though training_as_of_utc
    currently always equals target_cutoff_utc (a batch's training mask is
    evaluated against its own target cutoff)."""
    frame = _synthetic_matrix()
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    for col in ("target_cutoff_utc", "training_as_of_utc", "max_training_result_available_at_utc"):
        assert col in preds.columns
    assert (preds["training_as_of_utc"] == preds["target_cutoff_utc"]).all()
    oof_rows = preds[preds["status"] == "OOF"]
    assert (oof_rows["max_training_result_available_at_utc"] < oof_rows["target_cutoff_utc"]).all()


def test_result_availability_basis_is_recorded_as_conservative_provenance():
    frame = _synthetic_matrix()
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    assert (preds["result_availability_basis"] == oof.RESULT_AVAILABILITY_BASIS).all()
    assert oof.RESULT_AVAILABILITY_BASIS == "HISTORICAL_CONSERVATIVE_BATCH"


def test_persist_oof_ledger_resolves_under_a_separate_artifact_root(tmp_path):
    """Generated OOF ledgers/manifests must resolve under a distinct
    artifact root, never under the raw historical-data root."""
    from nfl_hybrid.evaluation.chronological_oof import (
        ChronologicalOOFResult,
        persist_oof_ledger,
    )

    frame = _synthetic_matrix()
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=SMALL_CONFIG)
    ledger = attach_outcomes_and_residuals(preds, frame)
    result = ChronologicalOOFResult(
        predictions=preds, residual_ledger=ledger,
        feature_columns=tuple(FEATURE_COLUMNS), feature_state_hash="h", model_config_hash="c",
    )
    written = persist_oof_ledger(result, root_override=tmp_path)
    for path in written.values():
        assert path.is_relative_to(tmp_path)
        assert path.exists()
    manifest = json.loads(written["oof_chronological.manifest"].read_text())
    assert manifest["result_availability_basis"] == "HISTORICAL_CONSERVATIVE_BATCH"


# ---------------------------------------------------------------------------
# Batching correctness: the performance optimization, proven directly
# ---------------------------------------------------------------------------

def _slate_matrix(n_slates: int = 6, per_slate: int = 3, seed: int = 0) -> pd.DataFrame:
    """Several games per kickoff (a "slate"), the common real-NFL case this
    batching optimization targets -- unlike ``_synthetic_matrix``, where every
    row has its own unique kickoff."""
    rng = np.random.default_rng(seed)
    n = n_slates * per_slate
    kickoffs = np.repeat(
        pd.date_range("2024-09-01", periods=n_slates, freq="7D", tz="UTC"), per_slate
    )
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    home_margin = 3.0 + 6.0 * x1 + rng.normal(scale=8.0, size=n)
    total_points = 44.0 + 3.0 * x2 + rng.normal(scale=7.0, size=n)
    return pd.DataFrame(
        {
            "game_id": [f"S{i:03d}" for i in range(n)],
            "season": 2024,
            "week": np.repeat(np.arange(n_slates) + 1, per_slate),
            "scheduled_kickoff_utc": kickoffs,
            "result_available_at_utc": kickoffs + pd.Timedelta(hours=1),
            "feat_x1": x1,
            "feat_x2": x2,
            "home_margin": home_margin,
            "total_points": total_points,
        }
    )


def test_games_sharing_a_cutoff_are_fit_exactly_once_not_once_per_game():
    n_slates, per_slate = 6, 3
    frame = _slate_matrix(n_slates=n_slates, per_slate=per_slate)
    cfg = ChronologicalOOFConfig(min_training_games=per_slate * 2)  # first 2 slates warm up

    fit_calls = []

    def counting_factory(c):
        fit_calls.append(1)
        return JointScoreModel(FEATURE_COLUMNS, (), c.model_config)

    preds = generate_oof_predictions(
        frame, feature_columns=FEATURE_COLUMNS, config=cfg, model_factory=counting_factory
    )
    oof_rows = preds[preds["status"] == "OOF"]
    n_eligible_batches = oof_rows["target_cutoff_utc"].nunique()
    assert len(oof_rows) == n_eligible_batches * per_slate  # every eligible slate is full
    assert sum(fit_calls) == n_eligible_batches  # one fit per batch, not one per game
    assert n_eligible_batches < len(oof_rows)  # batching actually reduced fit count


def test_batched_predictions_match_a_manual_single_fit_reference():
    frame = _slate_matrix()
    cfg = ChronologicalOOFConfig(min_training_games=6)
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=cfg)

    oof_rows = preds[preds["status"] == "OOF"]
    last_cutoff = oof_rows["target_cutoff_utc"].max()
    batch = oof_rows[oof_rows["target_cutoff_utc"] == last_cutoff]
    assert len(batch) > 1  # a genuine multi-game batch

    train_ids = batch.iloc[0]["training_game_ids"]
    for _, r in batch.iterrows():
        assert set(r["training_game_ids"]) == set(train_ids)  # every batch-mate shares training

    train_mask = frame["game_id"].isin(train_ids)
    reference = JointScoreModel(FEATURE_COLUMNS, (), cfg.model_config)
    reference.fit(frame.loc[train_mask, FEATURE_COLUMNS + ["home_margin", "total_points"]])
    target_gids = batch["game_id"].tolist()
    target_rows = frame.set_index("game_id").loc[target_gids][FEATURE_COLUMNS]
    ref_margin, ref_total = reference.predict_means(target_rows)

    for gid, expected_margin, expected_total in zip(target_gids, ref_margin, ref_total):
        row = batch[batch["game_id"] == gid].iloc[0]
        assert row["predicted_margin"] == pytest.approx(expected_margin)
        assert row["predicted_total"] == pytest.approx(expected_total)


def test_batching_preserves_membership_and_chronology_invariants():
    frame = _slate_matrix()
    cfg = ChronologicalOOFConfig(min_training_games=6)
    preds = generate_oof_predictions(frame, feature_columns=FEATURE_COLUMNS, config=cfg)
    slate_index = {gid: i for i, gid in enumerate(frame["game_id"])}

    for row in preds.itertuples(index=False):
        if row.training_game_ids_truncated:
            continue
        assert row.game_id not in row.training_game_ids  # no batch-mate self-training
        for tid in row.training_game_ids:
            assert slate_index[tid] < slate_index[row.game_id]
        if row.status == "OOF":
            assert row.max_training_result_available_at_utc < row.target_cutoff_utc


def test_batching_remains_exact_when_a_prior_games_result_is_delayed():
    """One earlier slate's game resolves unusually late (after a much later
    slate's cutoff) -- batching must still fit exactly once per eligible
    batch, and must still correctly exclude the delayed game from every
    batch whose cutoff precedes its (delayed) availability."""
    n_slates, per_slate = 8, 3
    frame = _slate_matrix(n_slates=n_slates, per_slate=per_slate)
    slates = frame.groupby("scheduled_kickoff_utc", sort=True)["game_id"].apply(list).tolist()
    assert len(slates) == n_slates

    delayed_game_id = slates[1][0]  # first game of the second slate
    late_slate_cutoff = frame.loc[
        frame["game_id"] == slates[4][0], "scheduled_kickoff_utc"
    ].iloc[0] - pd.Timedelta(minutes=10)
    frame.loc[frame["game_id"] == delayed_game_id, "result_available_at_utc"] = (
        late_slate_cutoff + pd.Timedelta(hours=1)
    )

    cfg = ChronologicalOOFConfig(min_training_games=per_slate * 2)
    fit_calls = []

    def counting_factory(c):
        fit_calls.append(1)
        return JointScoreModel(FEATURE_COLUMNS, (), c.model_config)

    preds = generate_oof_predictions(
        frame, feature_columns=FEATURE_COLUMNS, config=cfg, model_factory=counting_factory
    )
    oof_rows = preds[preds["status"] == "OOF"]
    n_eligible_batches = oof_rows["target_cutoff_utc"].nunique()
    assert sum(fit_calls) == n_eligible_batches  # still exactly one fit per batch

    # every batch whose own cutoff precedes the delayed availability excludes it;
    # every batch after that point may include it.
    delayed_available_at = frame.loc[
        frame["game_id"] == delayed_game_id, "result_available_at_utc"
    ].iloc[0]
    checked_excluded = checked_included = False
    for row in oof_rows.itertuples(index=False):
        if row.training_game_ids_truncated:
            continue
        if row.target_cutoff_utc <= delayed_available_at:
            assert delayed_game_id not in row.training_game_ids
            checked_excluded = True
        else:
            assert delayed_game_id in row.training_game_ids
            checked_included = True
    assert checked_excluded and checked_included


# ---------------------------------------------------------------------------
# Integration proofs: the real Fix 2 interface, genuinely used
# ---------------------------------------------------------------------------

def _tiny_games_and_pbp(n_games: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = ["KC", "BUF", "SF", "DAL"]
    rng = np.random.default_rng(seed)
    game_ids = [f"T{i:03d}" for i in range(n_games)]
    kickoff = pd.date_range("2024-09-01", periods=n_games, freq="7D", tz="UTC")
    home = [teams[i % 4] for i in range(n_games)]
    away = [teams[(i + 1) % 4] for i in range(n_games)]
    games = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": 2024,
            "week": np.arange(n_games) + 1,
            "home_team_id": home,
            "away_team_id": away,
            "scheduled_kickoff_utc": kickoff,
            "home_score": rng.integers(10, 35, n_games),
            "away_score": rng.integers(10, 35, n_games),
            "home_spread_reference": -3.0,
            "total_line_reference": 44.0,
            "neutral_site": False,
            "playoff": False,
        }
    )
    # 6 plays per team per game (not 15) -- enough for aggregate_advanced_team_game
    # to produce a row; the exact per-play count doesn't matter to this suite.
    rows = []
    for gid, h, a in zip(game_ids, home, away):
        for team, opp in [(h, a), (a, h)]:
            for _ in range(6):
                rows.append(
                    dict(
                        game_id=gid, posteam=team, defteam=opp, season=2024, week=1,
                        qb_dropback=1, pass_attempt=1, rush_attempt=0,
                        epa=float(rng.normal() * 0.1), success=1, down=1,
                        game_seconds_remaining=1800, score_differential=0,
                        passing_yards=8, complete_pass=1,
                    )
                )
    return games, pd.DataFrame(rows)


def test_run_chronological_oof_uses_the_authoritative_pregame_state_interface(monkeypatch):
    calls = []
    real = oof.build_authoritative_pregame_state

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(oof, "build_authoritative_pregame_state", spy)

    games, pbp = _tiny_games_and_pbp(n_games=8)
    run_chronological_oof(
        games, pbp,
        config=ChronologicalOOFConfig(min_training_games=3),
        target_row_positions=[6, 7],  # only the two rows with enough warmup
    )
    assert len(calls) == 1


@pytest.fixture(scope="module")
def _shared_feature_matrix():
    """Build the Fix 2 pregame-state bundle and pivot it exactly once, shared
    by every test in this module that needs a real (not hand-built) feature
    matrix -- the expensive step is the state build, not any individual
    model fit."""
    games, pbp = _tiny_games_and_pbp(n_games=8)
    return build_oof_feature_matrix(games, pbp)


def test_shared_feature_matrix_carries_result_available_at_utc(_shared_feature_matrix):
    feature_matrix, _feature_columns, _bundle = _shared_feature_matrix
    assert "result_available_at_utc" in feature_matrix.columns
    kickoff = pd.to_datetime(feature_matrix["scheduled_kickoff_utc"], utc=True)
    available = pd.to_datetime(feature_matrix["result_available_at_utc"], utc=True)
    assert (available >= kickoff).all()


def test_run_chronological_oof_end_to_end_produces_a_well_formed_ledger(_shared_feature_matrix):
    feature_matrix, feature_columns, _bundle = _shared_feature_matrix
    cfg = ChronologicalOOFConfig(min_training_games=3)
    feature_state_hash = "test-hash"
    predictions = generate_oof_predictions(
        feature_matrix,
        feature_columns=feature_columns,
        config=cfg,
        feature_state_hash=feature_state_hash,
        target_row_positions=[5, 6, 7],  # a few rows with enough warmup, not all 8
    )
    ledger = attach_outcomes_and_residuals(predictions, feature_matrix)

    oof_rows = ledger[ledger["status"] == "OOF"]
    assert len(oof_rows) > 0
    assert oof_rows[["predicted_margin", "predicted_total", "actual_margin", "actual_total"]].notna().all().all()
    np.testing.assert_allclose(
        oof_rows["margin_residual"].to_numpy(float),
        (oof_rows["actual_margin"] - oof_rows["predicted_margin"]).to_numpy(float),
    )
    for row in oof_rows.itertuples(index=False):
        assert row.training_membership_hash == training_membership_hash(row.training_game_ids) or row.training_game_ids_truncated
        assert row.feature_state_hash == feature_state_hash
        assert row.max_training_result_available_at_utc < row.target_cutoff_utc
