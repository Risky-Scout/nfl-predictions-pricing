"""Fix 4 mandatory adversarial proof suite for the chronological calibration
framework (:mod:`nfl_hybrid.evaluation.chronological_calibration`).

Mirrors ``tests/test_chronological_oof.py``'s structure: engine-level
adversarial proofs run against small, hand-built "raw" probability frames
(full control over exactly which row's value changed), plus a smaller set of
integration tests that exercise the real Fix 3 -> Fix 4 interface
(``build_raw_probabilities`` consuming a genuine Fix 3 residual/uncertainty
ledger). Every fixture is small and single-threaded (``tests/conftest.py``
forces every BLAS/OpenMP backend to one thread for the whole session).

Mandatory adversarial proofs covered here:
  - target game absent from its own calibration sample
  - unresolved prior prediction excluded
  - future prediction excluded
  - earlier resolved OOF prediction may enter
  - changing target outcome cannot change its own calibrated probability
  - changing unresolved/future outcome cannot change earlier calibration
  - once a prior outcome becomes available, it may enter later calibration
  - train=calibration fallback is impossible
  - insufficient-history behavior is explicit
  - shuffled input order produces identical results
  - identical inputs/config are deterministic
  - max_calibration_result_available_at_utc < target_cutoff_utc
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation import chronological_calibration as calib
from nfl_hybrid.evaluation.chronological_calibration import (
    ChronologicalCalibrationConfig,
    ChronologicalCalibrationResult,
    build_raw_probabilities,
    generate_chronological_calibration,
    persist_calibration_ledger,
    run_chronological_calibration,
)
from nfl_hybrid.evaluation.chronological_oof import UNCERTAINTY_LEDGER_COLUMNS

N_SYNTHETIC = 140  # comfortably above the reused calibrator's own 100-sample floor
HORIZON = "kickoff_minus_10_minutes"


def _raw_frame(
    n: int = N_SYNTHETIC,
    *,
    home_overrides: dict[int, float] | None = None,
    binary_overrides: dict[int, object] | None = None,
    result_available_overrides: dict[int, pd.Timestamp] | None = None,
    status_overrides: dict[int, str] | None = None,
    model_config_hash_overrides: dict[int, str] | None = None,
    forecast_horizon_overrides: dict[int, str] | None = None,
    single_cutoff: bool = False,
) -> pd.DataFrame:
    if single_cutoff:
        kickoff = pd.Series([pd.Timestamp("2024-09-08", tz="UTC")] * n)
    else:
        kickoff = pd.Series(pd.date_range("2024-09-01", periods=n, freq="D", tz="UTC"))
    target_cutoff = kickoff - pd.Timedelta(minutes=10)
    result_available = kickoff + pd.Timedelta(hours=1)

    raw_home = np.clip(0.5 + 0.1 * np.sin(np.arange(n)), 0.1, 0.9)
    raw_push = np.full(n, 0.02)
    raw_away = 1.0 - raw_home - raw_push
    cond = raw_home / (raw_home + raw_away)
    # Deterministic, always-balanced labels -- both classes present in any
    # window of >= 2 rows, so the only thing that can make the reused
    # calibrator decline to fit is sample size, never class imbalance.
    binary_target = pd.array([i % 2 for i in range(n)], dtype="Int8")

    for idx, v in (home_overrides or {}).items():
        raw_home[idx] = v
        raw_away[idx] = 1.0 - v - raw_push[idx]
        cond[idx] = raw_home[idx] / (raw_home[idx] + raw_away[idx])

    frame = pd.DataFrame(
        {
            "game_id": [f"R{i:03d}" for i in range(n)],
            "season": 2024,
            "week": (np.arange(n) // 4) + 1,
            "forecast_horizon": "kickoff_minus_10_minutes",
            "target_cutoff_utc": target_cutoff,
            "result_available_at_utc": result_available,
            "model_config_hash": "mc1",
            "feature_state_hash": "fs1",
            "raw_home_probability": raw_home,
            "raw_push_probability": raw_push,
            "raw_away_probability": raw_away,
            "raw_conditional_upper_probability": cond,
            "binary_target": binary_target,
            "raw_status": "RAW_READY",
        }
    )
    for idx, v in (result_available_overrides or {}).items():
        frame.loc[idx, "result_available_at_utc"] = v
    for idx, v in (binary_overrides or {}).items():
        frame.loc[idx, "binary_target"] = v
    for idx, v in (status_overrides or {}).items():
        frame.loc[idx, "raw_status"] = v
    for idx, v in (model_config_hash_overrides or {}).items():
        frame.loc[idx, "model_config_hash"] = v
    for idx, v in (forecast_horizon_overrides or {}).items():
        frame.loc[idx, "forecast_horizon"] = v
    return frame


CFG = ChronologicalCalibrationConfig()


def _index_of(game_id: str) -> int:
    return int(game_id[1:])


def _calibrated(ledger: pd.DataFrame) -> pd.DataFrame:
    return ledger[ledger["calibration_status"] == "CALIBRATED"]


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_target_game_absent_from_own_calibration_sample():
    frame = _raw_frame()
    ledger = generate_chronological_calibration(frame, config=CFG)
    calibrated = _calibrated(ledger)
    assert len(calibrated) > 0
    for row in calibrated.itertuples(index=False):
        if row.calibration_prediction_ids_truncated:
            continue
        assert row.game_id not in row.calibration_prediction_ids


def test_future_prediction_excluded():
    frame = _raw_frame()
    ledger = generate_chronological_calibration(frame, config=CFG)
    calibrated = _calibrated(ledger)
    checked = 0
    for row in calibrated.itertuples(index=False):
        if row.calibration_prediction_ids_truncated:
            continue
        target_idx = _index_of(row.game_id)
        for cid in row.calibration_prediction_ids:
            assert _index_of(cid) < target_idx, (row.game_id, cid)
            checked += 1
    assert checked > 0


def test_max_calibration_result_available_before_target_cutoff():
    """The exact real-data-proof invariant, independently re-checked here on
    every CALIBRATED row of the synthetic suite (not just trusted from the
    ledger's own internal self-check)."""
    frame = _raw_frame()
    ledger = generate_chronological_calibration(frame, config=CFG)
    calibrated = _calibrated(ledger)
    assert len(calibrated) > 0
    for row in calibrated.itertuples(index=False):
        assert row.max_calibration_result_available_at_utc < row.target_cutoff_utc
        assert row.game_id not in row.calibration_prediction_ids or row.calibration_prediction_ids_truncated


# ---------------------------------------------------------------------------
# Result-availability proofs
# ---------------------------------------------------------------------------

def test_unresolved_prior_prediction_excluded_same_day_leakage():
    """A kicks off before B, but A's result isn't available until AFTER B's
    forecast cutoff -- A must be excluded from B's calibration set despite
    being an earlier-indexed game."""
    a_idx, b_idx = 60, 61
    base = _raw_frame()
    b_cutoff = base.loc[b_idx, "target_cutoff_utc"]
    late_result = b_cutoff + pd.Timedelta(hours=1)  # confirmed only AFTER B's own cutoff
    frame = _raw_frame(result_available_overrides={a_idx: late_result})
    assert frame.loc[a_idx, "result_available_at_utc"] > b_cutoff

    ledger = generate_chronological_calibration(frame, config=CFG)
    a_gid, b_gid = frame.loc[a_idx, "game_id"], frame.loc[b_idx, "game_id"]
    b_row = ledger[ledger["game_id"] == b_gid].iloc[0]

    assert b_row["calibration_status"] in ("CALIBRATED", "UNCALIBRATED_INSUFFICIENT_HISTORY")
    assert a_gid not in b_row["calibration_prediction_ids"]
    assert b_row["calibration_sample_count"] == b_idx - 1  # every prior labeled row except delayed A


def test_earlier_resolved_prediction_may_enter_positive_control():
    a_idx, b_idx = 60, 61
    base = _raw_frame()
    b_cutoff = base.loc[b_idx, "target_cutoff_utc"]
    early_result = b_cutoff - pd.Timedelta(hours=1)
    frame = _raw_frame(result_available_overrides={a_idx: early_result})

    ledger = generate_chronological_calibration(frame, config=CFG)
    a_gid, b_gid = frame.loc[a_idx, "game_id"], frame.loc[b_idx, "game_id"]
    b_row = ledger[ledger["game_id"] == b_gid].iloc[0]

    assert a_gid in b_row["calibration_prediction_ids"]
    assert b_row["calibration_sample_count"] == b_idx


def test_prior_outcome_may_enter_once_available_but_not_before():
    """Mirrors Fix 3's residual-availability test one layer up: a resolved
    game G is delayed past an EARLY target's cutoff (excluded there) but
    resolves before a LATER target's cutoff (may enter there)."""
    g_idx = 40
    base = _raw_frame()
    late_result = base.loc[100, "target_cutoff_utc"]  # far later than g_idx's own kickoff+1h
    frame = _raw_frame(result_available_overrides={g_idx: late_result})
    g_gid = frame.loc[g_idx, "game_id"]

    ledger = generate_chronological_calibration(frame, config=CFG)

    early_gid = frame.loc[50, "game_id"]
    early_row = ledger[ledger["game_id"] == early_gid].iloc[0]
    assert early_row["target_cutoff_utc"] < late_result
    assert g_gid not in early_row["calibration_prediction_ids"]

    later_gid = frame.loc[N_SYNTHETIC - 1, "game_id"]
    later_row = ledger[ledger["game_id"] == later_gid].iloc[0]
    assert later_row["target_cutoff_utc"] > late_result
    assert g_gid in later_row["calibration_prediction_ids"]


# ---------------------------------------------------------------------------
# Outcome-mutation invariance
# ---------------------------------------------------------------------------

def test_changing_targets_own_outcome_does_not_change_its_own_calibrated_probability():
    target_idx = 130
    base = generate_chronological_calibration(_raw_frame(), config=CFG)
    mutated_frame = _raw_frame(binary_overrides={target_idx: 1 - (target_idx % 2)})
    mutated = generate_chronological_calibration(mutated_frame, config=CFG)

    b = base.iloc[target_idx]
    m = mutated.iloc[target_idx]
    assert b["game_id"] == m["game_id"]
    assert b["calibration_status"] == m["calibration_status"] == "CALIBRATED"
    assert b["calibrated_home_probability"] == m["calibrated_home_probability"]
    assert b["calibrated_push_probability"] == m["calibrated_push_probability"]
    assert b["calibrated_away_probability"] == m["calibrated_away_probability"]


def test_changing_a_future_outcome_does_not_change_earlier_calibration():
    late_idx = N_SYNTHETIC - 2
    base = generate_chronological_calibration(_raw_frame(), config=CFG)
    mutated_frame = _raw_frame(binary_overrides={late_idx: 1 - (late_idx % 2)})
    mutated = generate_chronological_calibration(mutated_frame, config=CFG)

    earlier = base.iloc[:late_idx]
    earlier_mutated = mutated.iloc[:late_idx]
    pd.testing.assert_frame_equal(earlier, earlier_mutated)


def test_mutating_a_legitimate_prior_labeled_row_can_change_a_later_calibration():
    base = generate_chronological_calibration(_raw_frame(), config=CFG)
    prior_idx = 20
    mutated_frame = _raw_frame(home_overrides={prior_idx: 0.99})
    mutated = generate_chronological_calibration(mutated_frame, config=CFG)

    later = base[base["game_id"].map(_index_of) > prior_idx]
    later_mutated = mutated[mutated["game_id"].map(_index_of) > prior_idx]
    changed = (
        later["calibrated_home_probability"].to_numpy() != later_mutated["calibrated_home_probability"].to_numpy()
    )
    assert changed.any(), "mutating a legitimate prior calibration-eligible row changed no downstream calibration"


def test_changing_unresolved_row_outcome_cannot_affect_current_calibration():
    """The row that is unresolved-by-cutoff for an early target has its
    binary_target mutated; the early target's calibration must be unaffected
    because that row was never in its calibration set to begin with."""
    g_idx = 40
    base_frame = _raw_frame()
    late_result = base_frame.loc[100, "target_cutoff_utc"]
    frame = _raw_frame(result_available_overrides={g_idx: late_result})
    base_ledger = generate_chronological_calibration(frame, config=CFG)

    mutated_frame = frame.copy()
    mutated_frame.loc[g_idx, "binary_target"] = 1 - (g_idx % 2)
    mutated_ledger = generate_chronological_calibration(mutated_frame, config=CFG)

    early_gid = frame.loc[50, "game_id"]
    b = base_ledger[base_ledger["game_id"] == early_gid].reset_index(drop=True)
    m = mutated_ledger[mutated_ledger["game_id"] == early_gid].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, m)


# ---------------------------------------------------------------------------
# No train=calibration fallback / explicit insufficient-history status
# ---------------------------------------------------------------------------

def test_train_equals_calibration_fallback_is_impossible():
    """Every row shares the exact same target_cutoff_utc (one giant batch,
    plenty of labeled same-batch data) -- with zero rows strictly PRIOR to
    that shared cutoff, every row must be UNCALIBRATED_INSUFFICIENT_HISTORY,
    never silently fit on its own batch."""
    frame = _raw_frame(n=150, single_cutoff=True)
    ledger = generate_chronological_calibration(frame, config=CFG)
    assert (ledger["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY").all()
    assert (ledger["calibration_sample_count"] == 0).all()
    assert ledger["calibrated_home_probability"].isna().all()


def test_insufficient_history_is_explicit_not_silent():
    frame = _raw_frame(n=30)  # well under the reused calibrator's 100-sample floor
    ledger = generate_chronological_calibration(frame, config=CFG)
    assert (ledger["calibration_status"] != "CALIBRATED").all()
    assert (ledger["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY").any()
    insufficient = ledger[ledger["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY"]
    assert insufficient["calibrated_home_probability"].isna().all()
    assert insufficient["calibrated_push_probability"].isna().all()
    assert insufficient["calibrated_away_probability"].isna().all()
    # raw probabilities are still preserved even when uncalibrated
    assert insufficient["raw_home_probability"].notna().all()


def test_raw_and_calibrated_probabilities_are_preserved_separately():
    ledger = generate_chronological_calibration(_raw_frame(), config=CFG)
    calibrated = _calibrated(ledger)
    assert len(calibrated) > 0
    # a genuine calibration shift actually happened for at least one row
    assert (calibrated["raw_home_probability"] != calibrated["calibrated_home_probability"]).any()
    for _, row in calibrated.iterrows():
        assert np.isfinite(row["raw_home_probability"])
        assert np.isfinite(row["calibrated_home_probability"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_identical_calibration():
    frame = _raw_frame()
    first = generate_chronological_calibration(frame, config=CFG)
    second = generate_chronological_calibration(frame, config=CFG)
    pd.testing.assert_frame_equal(first, second)


def test_shuffled_input_rows_do_not_change_results():
    frame = _raw_frame()
    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)

    first = generate_chronological_calibration(frame, config=CFG)
    second = generate_chronological_calibration(shuffled, config=CFG)

    a = first.sort_values("game_id").reset_index(drop=True)
    b = second.sort_values("game_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Push/tie, missing columns, RAW_UNAVAILABLE passthrough, horizon guard
# ---------------------------------------------------------------------------

def test_raw_unavailable_rows_pass_through_and_are_never_calibrated():
    frame = _raw_frame(status_overrides={5: "RAW_UNAVAILABLE", 6: "RAW_UNAVAILABLE"})
    ledger = generate_chronological_calibration(frame, config=CFG)
    for idx in (5, 6):
        gid = frame.loc[idx, "game_id"]
        row = ledger[ledger["game_id"] == gid].iloc[0]
        assert row["calibration_status"] == "RAW_UNAVAILABLE"
        assert pd.isna(row["raw_home_probability"])
        assert pd.isna(row["calibrated_home_probability"])
        assert row["calibration_sample_count"] == 0


def test_tied_prior_row_is_ineligible_as_a_calibration_label_but_still_raw_ready():
    """A prior row with a null (tied) binary_target must never enter a later
    target's calibration set as a labeled point -- the same tie/push policy
    (nfl_hybrid.labels.edge_to_nullable_binary) enforced one layer down in
    Fix 3, applied here to calibration-set eligibility."""
    tie_idx = 60
    frame = _raw_frame(binary_overrides={tie_idx: pd.NA})
    tie_gid = frame.loc[tie_idx, "game_id"]
    assert frame.loc[tie_idx, "raw_status"] == "RAW_READY"  # still priced

    ledger = generate_chronological_calibration(frame, config=CFG)
    later_gid = frame.loc[N_SYNTHETIC - 1, "game_id"]
    later_row = ledger[ledger["game_id"] == later_gid].iloc[0]
    assert tie_gid not in later_row["calibration_prediction_ids"]


def test_required_columns_are_enforced():
    frame = _raw_frame().drop(columns=["result_available_at_utc"])
    with pytest.raises(ValueError):
        generate_chronological_calibration(frame, config=CFG)


def test_duplicate_game_ids_are_rejected():
    frame = _raw_frame()
    frame.loc[1, "game_id"] = frame.loc[0, "game_id"]
    with pytest.raises(ValueError):
        generate_chronological_calibration(frame, config=CFG)


def test_result_available_before_own_target_cutoff_is_rejected():
    frame = _raw_frame()
    frame.loc[0, "result_available_at_utc"] = frame.loc[0, "target_cutoff_utc"] - pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        generate_chronological_calibration(frame, config=CFG)


def test_mixed_forecast_horizons_are_rejected_unless_explicitly_pooled():
    frame = _raw_frame(model_config_hash_overrides={0: "mc2"})
    with pytest.raises(ValueError):
        generate_chronological_calibration(frame, config=CFG)
    # explicit opt-in pools them without raising
    pooled_cfg = ChronologicalCalibrationConfig(pool_across_horizons=True)
    generate_chronological_calibration(frame, config=pooled_cfg)  # must not raise


def test_calibration_membership_hash_is_order_independent():
    from nfl_hybrid.evaluation.chronological_oof import training_membership_hash

    ids = ["R003", "R001", "R002"]
    assert training_membership_hash(ids) == training_membership_hash(list(reversed(ids)))


def test_calibrator_family_and_config_hash_are_recorded():
    ledger = generate_chronological_calibration(_raw_frame(), config=CFG)
    assert (ledger["calibrator_family"] == calib.CALIBRATOR_FAMILY).all()
    assert (ledger["calibrator_config_hash"] == calib.compute_calibrator_config_hash(CFG)).all()


def test_ledger_keeps_target_cutoff_and_calibration_as_of_distinct_fields():
    ledger = generate_chronological_calibration(_raw_frame(), config=CFG)
    for col in ("target_cutoff_utc", "calibration_as_of_utc", "max_calibration_result_available_at_utc"):
        assert col in ledger.columns
    assert (ledger["calibration_as_of_utc"] == ledger["target_cutoff_utc"]).all()


# ---------------------------------------------------------------------------
# Integration: build_raw_probabilities from a genuine Fix 3 ledger shape
# ---------------------------------------------------------------------------

def _fix3_shaped_residual_ledger(n: int = 150, seed: int = 0, *, vary_totals: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kickoff = pd.date_range("2024-09-01", periods=n, freq="D", tz="UTC")
    predicted_margin = rng.normal(scale=6, size=n)
    actual_margin = predicted_margin + rng.normal(scale=8, size=n)
    if vary_totals:
        predicted_total = 44.0 + rng.normal(scale=4, size=n)
        actual_total = predicted_total + rng.normal(scale=7, size=n)
    else:
        predicted_total = 44.0
        actual_total = 44.0
    frame = pd.DataFrame(
        {
            "game_id": [f"F{i:03d}" for i in range(n)],
            "season": 2024,
            "week": (np.arange(n) // 4) + 1,
            "target_cutoff_utc": kickoff - pd.Timedelta(minutes=10),
            "training_as_of_utc": kickoff - pd.Timedelta(minutes=10),
            "result_available_at_utc": kickoff + pd.Timedelta(hours=1),
            "max_training_result_available_at_utc": kickoff - pd.Timedelta(minutes=30),
            "result_availability_basis": "HISTORICAL_CONSERVATIVE_BATCH",
            "training_game_count": np.arange(n),
            "training_game_ids": [() for _ in range(n)],
            "training_game_ids_truncated": False,
            "training_membership_hash": "h",
            "model_config_hash": "mc1",
            "feature_state_hash": "fs1",
            "status": "OOF",
            "predicted_margin": predicted_margin,
            "predicted_total": predicted_total,
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "margin_residual": actual_margin - predicted_margin,
            "total_residual": (
                (actual_total - predicted_total) if vary_totals else 0.0
            ),
            "margin_residual_sd_oof": 8.0,
            "total_residual_sd_oof": 7.0,
            "residual_correlation_oof": 0.0,
            "uncertainty_eligible": np.arange(n) >= 32,
        }
    )
    assert set(UNCERTAINTY_LEDGER_COLUMNS) <= set(frame.columns)
    return frame


def test_build_raw_probabilities_marks_uncertainty_ineligible_rows_unavailable():
    ledger = _fix3_shaped_residual_ledger()
    raw = build_raw_probabilities(ledger, market=calib.MARKET_MONEYLINE, forecast_horizon=HORIZON)
    early = raw[raw["game_id"].map(lambda g: int(g[1:])) < 32]
    assert (early["raw_status"] == "RAW_UNAVAILABLE").all()
    assert early["raw_home_probability"].isna().all()
    late = raw[raw["game_id"].map(lambda g: int(g[1:])) >= 32]
    assert (late["raw_status"] == "RAW_READY").all()
    assert late["raw_home_probability"].notna().all()


def test_build_raw_probabilities_never_reads_a_fitted_models_own_sigma():
    import inspect

    source = inspect.getsource(build_raw_probabilities)
    for token in (".margin_sd_", ".total_sd_", ".residual_correlation_"):
        assert token not in source


def test_run_chronological_calibration_end_to_end_produces_a_well_formed_ledger():
    residual_ledger = _fix3_shaped_residual_ledger()
    result = run_chronological_calibration(
        residual_ledger, market=calib.MARKET_MONEYLINE, forecast_horizon=HORIZON, config=CFG
    )
    assert isinstance(result, ChronologicalCalibrationResult)
    assert len(result.ledger) == len(residual_ledger)
    calibrated = _calibrated(result.ledger)
    assert len(calibrated) > 0
    for row in calibrated.itertuples(index=False):
        assert row.max_calibration_result_available_at_utc < row.target_cutoff_utc
        assert row.game_id not in row.calibration_prediction_ids or row.calibration_prediction_ids_truncated


def test_persist_calibration_ledger_resolves_under_a_separate_artifact_root(tmp_path):
    residual_ledger = _fix3_shaped_residual_ledger()
    result = run_chronological_calibration(
        residual_ledger, market=calib.MARKET_MONEYLINE, forecast_horizon=HORIZON, config=CFG
    )
    written = persist_calibration_ledger(result, root_override=tmp_path)
    for path in written.values():
        assert path.is_relative_to(tmp_path)
        assert path.exists()
    manifest = json.loads(written["chronological_calibration.manifest"].read_text())
    assert manifest["calibrator_family"] == calib.CALIBRATOR_FAMILY
    assert manifest["n_calibrated"] > 0


# ---------------------------------------------------------------------------
# Market genericity (Fix 4, item 5): MONEYLINE / ATS / TOTAL semantics
# ---------------------------------------------------------------------------

def test_market_is_a_required_keyword_argument_with_no_default():
    """Fix 4 repair (item 1): the original real-data-proof defect was
    build_raw_probabilities() silently pricing MONEYLINE whenever a caller
    omitted market= -- there used to be a ``market: str = MARKET_MONEYLINE``
    default. There is no longer any default: omitting market is a TypeError
    at the Python call site, before any ledger/pricing logic runs, so this
    failure mode is now impossible to reintroduce silently."""
    ledger = _fix3_shaped_residual_ledger()
    with pytest.raises(TypeError):
        build_raw_probabilities(ledger, forecast_horizon=HORIZON)  # missing market=


def test_forecast_horizon_is_also_a_required_keyword_argument_with_no_default():
    ledger = _fix3_shaped_residual_ledger()
    with pytest.raises(TypeError):
        build_raw_probabilities(ledger, market=calib.MARKET_MONEYLINE)  # missing forecast_horizon=


def test_moneyline_still_reachable_only_via_an_explicit_market_argument():
    """The market-genericity refactor changed no MONEYLINE math -- pricing an
    explicit MARKET_MONEYLINE call still matches the pre-refactor behavior
    (an explicit ATS call at threshold 0.0 prices identically, proven
    separately by test_ats_threshold_of_zero_matches_moneyline below)."""
    ledger = _fix3_shaped_residual_ledger()
    raw = build_raw_probabilities(ledger, market=calib.MARKET_MONEYLINE, forecast_horizon=HORIZON)
    assert (raw["forecast_horizon"] == HORIZON).all()


def test_moneyline_rejects_an_explicit_threshold():
    ledger = _fix3_shaped_residual_ledger()
    with pytest.raises(ValueError):
        build_raw_probabilities(ledger, market=calib.MARKET_MONEYLINE, threshold=-3.0, forecast_horizon=HORIZON)


def test_unknown_market_is_rejected():
    ledger = _fix3_shaped_residual_ledger()
    with pytest.raises(ValueError):
        build_raw_probabilities(ledger, market="PLAYER_PROPS", forecast_horizon=HORIZON)


def test_ats_requires_an_explicit_threshold():
    ledger = _fix3_shaped_residual_ledger()
    with pytest.raises(ValueError):
        build_raw_probabilities(ledger, market=calib.MARKET_ATS, forecast_horizon=HORIZON)


def test_total_requires_an_explicit_threshold():
    ledger = _fix3_shaped_residual_ledger(vary_totals=True)
    with pytest.raises(ValueError):
        build_raw_probabilities(ledger, market=calib.MARKET_TOTAL, forecast_horizon=HORIZON)


def test_ats_threshold_of_zero_matches_moneyline():
    """ATS priced at a 0 spread is exactly the moneyline market -- the same
    probability surface, thresholded at the same line."""
    ledger = _fix3_shaped_residual_ledger()
    moneyline = build_raw_probabilities(ledger, market=calib.MARKET_MONEYLINE, forecast_horizon=HORIZON)
    ats_zero = build_raw_probabilities(ledger, market=calib.MARKET_ATS, threshold=0.0, forecast_horizon=HORIZON)
    pd.testing.assert_series_equal(moneyline["raw_home_probability"], ats_zero["raw_home_probability"])
    pd.testing.assert_series_equal(moneyline["binary_target"], ats_zero["binary_target"])


def test_ats_negative_spread_makes_home_cover_strictly_harder():
    """home covers line L iff home_margin + L > 0 (module convention). A
    negative line (home favored) is strictly harder to cover than pick'em, so
    home-cover probability must be strictly lower for every ready row."""
    ledger = _fix3_shaped_residual_ledger()
    pick_em = build_raw_probabilities(ledger, market=calib.MARKET_ATS, threshold=0.0, forecast_horizon=HORIZON)
    favored = build_raw_probabilities(ledger, market=calib.MARKET_ATS, threshold=-7.0, forecast_horizon=HORIZON)
    ready = pick_em["raw_status"] == "RAW_READY"
    assert ready.any()
    assert (favored.loc[ready, "raw_home_probability"] < pick_em.loc[ready, "raw_home_probability"]).all()


def test_ats_positive_spread_makes_home_cover_strictly_easier():
    ledger = _fix3_shaped_residual_ledger()
    pick_em = build_raw_probabilities(ledger, market=calib.MARKET_ATS, threshold=0.0, forecast_horizon=HORIZON)
    getting_points = build_raw_probabilities(ledger, market=calib.MARKET_ATS, threshold=7.0, forecast_horizon=HORIZON)
    ready = pick_em["raw_status"] == "RAW_READY"
    assert ready.any()
    assert (getting_points.loc[ready, "raw_home_probability"] > pick_em.loc[ready, "raw_home_probability"]).all()


def test_ats_binary_target_reflects_spread_adjusted_edge_not_raw_margin():
    n = 150
    ledger = _fix3_shaped_residual_ledger(n=n)
    ledger = ledger.copy()
    # Row 0: home wins by exactly 3 -- a push against a -3 spread, but a
    # (home) cover of moneyline (threshold 0).
    ledger.loc[0, "actual_margin"] = 3.0
    raw_moneyline = build_raw_probabilities(ledger, market=calib.MARKET_MONEYLINE, forecast_horizon=HORIZON)
    raw_ats = build_raw_probabilities(ledger, market=calib.MARKET_ATS, threshold=-3.0, forecast_horizon=HORIZON)
    assert raw_moneyline.loc[0, "binary_target"] == 1
    assert pd.isna(raw_ats.loc[0, "binary_target"])


def test_total_market_reads_total_quantities_not_margin():
    ledger = _fix3_shaped_residual_ledger(vary_totals=True)
    raw = build_raw_probabilities(ledger, market=calib.MARKET_TOTAL, threshold=44.0, forecast_horizon=HORIZON)
    ready = raw["raw_status"] == "RAW_READY"
    assert ready.any()
    # Over probability must track predicted_total vs. the line, not predicted_margin.
    predicted_total = ledger.loc[ready, "predicted_total"].to_numpy()
    over_prob = raw.loc[ready, "raw_home_probability"].to_numpy()
    higher_total = predicted_total > 44.0
    lower_total = predicted_total < 44.0
    assert over_prob[higher_total].mean() > over_prob[lower_total].mean()


def test_total_binary_target_is_over_under_push_against_the_declared_line():
    ledger = _fix3_shaped_residual_ledger(vary_totals=True)
    ledger = ledger.copy()
    ledger.loc[0, "predicted_total"] = 44.0
    ledger.loc[0, "actual_total"] = 44.0  # exact push
    ledger.loc[1, "predicted_total"] = 44.0
    ledger.loc[1, "actual_total"] = 51.0  # over
    ledger.loc[2, "predicted_total"] = 44.0
    ledger.loc[2, "actual_total"] = 37.0  # under
    raw = build_raw_probabilities(ledger, market=calib.MARKET_TOTAL, threshold=44.0, forecast_horizon=HORIZON)
    assert pd.isna(raw.loc[0, "binary_target"])
    assert raw.loc[1, "binary_target"] == 1
    assert raw.loc[2, "binary_target"] == 0


def test_total_higher_line_makes_over_strictly_harder():
    ledger = _fix3_shaped_residual_ledger(vary_totals=True)
    low_line = build_raw_probabilities(ledger, market=calib.MARKET_TOTAL, threshold=40.0, forecast_horizon=HORIZON)
    high_line = build_raw_probabilities(ledger, market=calib.MARKET_TOTAL, threshold=50.0, forecast_horizon=HORIZON)
    ready = low_line["raw_status"] == "RAW_READY"
    assert ready.any()
    assert (high_line.loc[ready, "raw_home_probability"] < low_line.loc[ready, "raw_home_probability"]).all()


def test_threshold_array_must_match_ledger_length():
    ledger = _fix3_shaped_residual_ledger()
    with pytest.raises(ValueError):
        build_raw_probabilities(
            ledger, market=calib.MARKET_ATS, threshold=np.array([1.0, 2.0, 3.0]), forecast_horizon=HORIZON
        )


@pytest.mark.parametrize(
    "market,threshold",
    [
        (calib.MARKET_MONEYLINE, None),
        (calib.MARKET_ATS, -3.0),
        (calib.MARKET_TOTAL, 44.0),
    ],
)
def test_calibration_engine_is_not_hardcoded_to_moneyline_end_to_end(market, threshold):
    """The downstream calibration machinery (generate_chronological_calibration
    and the reused three_way calibrator) never inspects ``market`` -- it only
    ever sees the abstract (upper/push/lower) probabilities build_raw_probabilities
    produced. This proves that end to end for all three supported markets: a
    genuine Fix 3-shaped residual ledger, priced under each market, still
    produces CALIBRATED rows whose membership invariant holds."""
    residual_ledger = _fix3_shaped_residual_ledger(vary_totals=(market == calib.MARKET_TOTAL))
    kwargs = {} if threshold is None else {"threshold": threshold}
    raw = build_raw_probabilities(residual_ledger, market=market, **kwargs, forecast_horizon=HORIZON)
    ledger = generate_chronological_calibration(raw, config=CFG)
    calibrated = _calibrated(ledger)
    assert len(calibrated) > 0
    for row in calibrated.itertuples(index=False):
        assert row.max_calibration_result_available_at_utc < row.target_cutoff_utc
        assert row.game_id not in row.calibration_prediction_ids or row.calibration_prediction_ids_truncated
        for prob_col in ("calibrated_home_probability", "calibrated_push_probability", "calibrated_away_probability"):
            value = getattr(row, prob_col)
            assert np.isfinite(value)
            assert 0.0 <= value <= 1.0
        total = row.calibrated_home_probability + row.calibrated_push_probability + row.calibrated_away_probability
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Forecast-horizon isolation (Fix 4, item 10): forecast_horizon is first-class
# row data, not solely inferred from model_config_hash.
# ---------------------------------------------------------------------------


def test_forecast_horizon_is_stamped_on_every_raw_row():
    ledger = _fix3_shaped_residual_ledger()
    raw = build_raw_probabilities(ledger, market=calib.MARKET_MONEYLINE, forecast_horizon=HORIZON)
    assert (raw["forecast_horizon"] == HORIZON).all()


def test_same_horizon_targets_may_share_calibration_history():
    """Two rows both stamped with HORIZON, differing only in game identity,
    calibrate normally against each other -- the default (non-pooled) path
    permits same-horizon sharing."""
    frame = _raw_frame()
    ledger = generate_chronological_calibration(frame, config=CFG)
    calibrated = _calibrated(ledger)
    assert len(calibrated) > 0
    assert (ledger["forecast_horizon"] == HORIZON).all()


def test_different_horizon_rows_cannot_share_history_by_default():
    """A single row stamped with a different forecast_horizon than the rest
    must be rejected by the default (non-pooled) path -- the identical
    invariant already enforced for model_config_hash, now also enforced via
    the explicit, first-class forecast_horizon column."""
    frame = _raw_frame(forecast_horizon_overrides={0: "kickoff_minus_60_minutes"})
    with pytest.raises(ValueError):
        generate_chronological_calibration(frame, config=CFG)


def test_different_horizon_rows_may_share_history_only_with_explicit_pooling():
    frame = _raw_frame(forecast_horizon_overrides={0: "kickoff_minus_60_minutes"})
    pooled_cfg = ChronologicalCalibrationConfig(pool_across_horizons=True)
    ledger = generate_chronological_calibration(frame, config=pooled_cfg)  # must not raise
    assert len(ledger) == len(frame)


def test_mismatched_model_config_hash_alone_is_also_rejected_even_with_matching_horizon():
    """forecast_horizon is the primary gate, but a model_config_hash mismatch
    is still independently caught -- Fix 4 item 10 says "not solely"
    model_config_hash, not "never check it": two rows can share a horizon
    label yet disagree on the underlying Fix 3 config, which is still a real
    mismatch worth failing on."""
    frame = _raw_frame(model_config_hash_overrides={0: "mc2"})
    assert (frame["forecast_horizon"] == HORIZON).all()  # horizon itself is uniform
    with pytest.raises(ValueError):
        generate_chronological_calibration(frame, config=CFG)


# ---------------------------------------------------------------------------
# Conditional-probability persistence (Fix 4, item 12): the calibrator's own
# pre-recombination conditional output must be persisted, not only the
# recombined three-way probabilities -- so diagnostics can score the
# conditional target the calibrator actually fit, never the unconditional
# probability against the conditional (non-push) binary target.
# ---------------------------------------------------------------------------


def test_calibrated_conditional_upper_probability_is_persisted_for_calibrated_rows():
    ledger = generate_chronological_calibration(_raw_frame(), config=CFG)
    calibrated = _calibrated(ledger)
    assert len(calibrated) > 0
    assert calibrated["calibrated_conditional_upper_probability"].notna().all()
    assert calibrated["calibrated_conditional_upper_probability"].between(0.0, 1.0).all()
    assert calibrated["raw_conditional_upper_probability"].notna().all()


def test_calibrated_conditional_upper_probability_is_nan_when_uncalibrated():
    frame = _raw_frame(n=30)  # under the 100-sample floor
    ledger = generate_chronological_calibration(frame, config=CFG)
    insufficient = ledger[ledger["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY"]
    assert len(insufficient) > 0
    assert insufficient["calibrated_conditional_upper_probability"].isna().all()


def test_conditional_recombines_to_the_persisted_unconditional_probabilities():
    """calibrated_home_probability must equal exactly what _recombine()
    produces from calibrated_conditional_upper_probability and the raw push
    mass -- proving the persisted conditional value is the actual value fed
    to recombination, not a separately-derived duplicate."""
    from nfl_hybrid.calibration.three_way import _recombine

    ledger = generate_chronological_calibration(_raw_frame(), config=CFG)
    calibrated = _calibrated(ledger)
    assert len(calibrated) > 0
    lower, push, upper = _recombine(
        calibrated["calibrated_conditional_upper_probability"].to_numpy(),
        calibrated["raw_push_probability"].to_numpy(),
    )
    assert np.allclose(upper, calibrated["calibrated_home_probability"].to_numpy(), atol=1e-12)
    assert np.allclose(push, calibrated["calibrated_push_probability"].to_numpy(), atol=1e-12)
    assert np.allclose(lower, calibrated["calibrated_away_probability"].to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# Chronological push-mass calibration (certification-blocker repair item 3).
# ---------------------------------------------------------------------------


def test_push_calibration_produces_normalized_three_way_probabilities():
    from nfl_hybrid.evaluation.chronological_calibration import calibrate_chronological_push_probability

    raw = _raw_frame()
    ledger = generate_chronological_calibration(raw, config=CFG)
    market_line = pd.Series([-3.0] * len(raw))
    pushed = calibrate_chronological_push_probability(raw, ledger, market=calib.MARKET_ATS, market_line=market_line)

    calibrated = pushed[pushed["push_calibration_policy"] == calib.PUSH_CALIBRATION_POLICY_CHRONOLOGICAL]
    assert len(calibrated) > 0
    total = (
        calibrated["calibrated_home_probability"]
        + calibrated["calibrated_push_probability"]
        + calibrated["calibrated_away_probability"]
    )
    assert np.allclose(total.to_numpy(), 1.0, atol=1e-9)


def test_push_calibration_membership_matches_conditional_when_no_actual_pushes():
    """The default synthetic fixture has zero actual pushes (binary_target is
    always 0/1, never null), so the push-fit training set (every RAW_READY
    row before cutoff) is identical in size to the conditional calibrator's
    own labeled training set (which only additionally excludes pushes -- of
    which there are none here)."""
    from nfl_hybrid.evaluation.chronological_calibration import calibrate_chronological_push_probability

    raw = _raw_frame()
    ledger = generate_chronological_calibration(raw, config=CFG)
    market_line = pd.Series([-3.0] * len(raw))
    pushed = calibrate_chronological_push_probability(raw, ledger, market=calib.MARKET_ATS, market_line=market_line)

    calibrated = pushed[pushed["push_calibration_policy"] == calib.PUSH_CALIBRATION_POLICY_CHRONOLOGICAL]
    assert (calibrated["push_calibration_membership_count"] == calibrated["calibration_sample_count"]).all()


def test_push_calibration_never_touches_uncalibrated_rows():
    from nfl_hybrid.evaluation.chronological_calibration import calibrate_chronological_push_probability

    raw = _raw_frame(n=30)  # under the 100-sample floor -- nothing gets CALIBRATED
    ledger = generate_chronological_calibration(raw, config=CFG)
    market_line = pd.Series([-3.0] * len(raw))
    pushed = calibrate_chronological_push_probability(raw, ledger, market=calib.MARKET_ATS, market_line=market_line)
    assert (pushed["push_calibration_policy"] == calib.PUSH_CALIBRATION_POLICY_RAW).all()
    assert (pushed["push_calibration_membership_count"] == 0).all()
    pd.testing.assert_series_equal(
        pushed["calibrated_home_probability"], ledger["calibrated_home_probability"], check_names=False
    )


def test_push_calibration_target_not_in_its_own_training_membership():
    """Structural, mirroring the conditional engine's own proof: push-fit
    training is strictly rows with result_available_at_utc < the target's own
    target_cutoff_utc, so the target itself (whose own result_available_at_utc
    is always after its own cutoff) can never be in its own push-fit set --
    verified indirectly via the membership COUNT never exceeding the number
    of strictly-earlier rows."""
    from nfl_hybrid.evaluation.chronological_calibration import calibrate_chronological_push_probability

    raw = _raw_frame()
    ledger = generate_chronological_calibration(raw, config=CFG)
    market_line = pd.Series([-3.0] * len(raw))
    pushed = calibrate_chronological_push_probability(raw, ledger, market=calib.MARKET_ATS, market_line=market_line)

    calibrated = pushed[pushed["push_calibration_policy"] == calib.PUSH_CALIBRATION_POLICY_CHRONOLOGICAL]
    for row in calibrated.itertuples(index=False):
        target_idx = _index_of(row.game_id)
        assert row.push_calibration_membership_count <= target_idx


def test_push_calibration_rejects_mismatched_market_line_length():
    from nfl_hybrid.evaluation.chronological_calibration import calibrate_chronological_push_probability

    raw = _raw_frame()
    ledger = generate_chronological_calibration(raw, config=CFG)
    with pytest.raises(ValueError):
        calibrate_chronological_push_probability(
            raw, ledger, market=calib.MARKET_ATS, market_line=pd.Series([-3.0, -3.0])
        )


def test_push_calibration_rejects_unknown_market():
    from nfl_hybrid.evaluation.chronological_calibration import calibrate_chronological_push_probability

    raw = _raw_frame()
    ledger = generate_chronological_calibration(raw, config=CFG)
    with pytest.raises(ValueError):
        calibrate_chronological_push_probability(
            raw, ledger, market="PLAYER_PROPS", market_line=pd.Series([-3.0] * len(raw))
        )


def test_push_calibration_config_hash_is_recorded():
    from nfl_hybrid.evaluation.chronological_calibration import calibrate_chronological_push_probability

    raw = _raw_frame()
    ledger = generate_chronological_calibration(raw, config=CFG)
    market_line = pd.Series([-3.0] * len(raw))
    pushed = calibrate_chronological_push_probability(raw, ledger, market=calib.MARKET_ATS, market_line=market_line)
    expected_hash = calib.compute_push_calibration_config_hash(calib.CalibrationConfig(), calib.MARKET_ATS)
    assert (pushed["push_calibration_config_hash"] == expected_hash).all()
