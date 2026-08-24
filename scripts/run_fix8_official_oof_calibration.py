"""FIX 8: OFFICIAL CHRONOLOGICAL TUE/FRI OOF + ATS/TOTAL CALIBRATION --
auto-mode end-to-end run.

Executes the operator-frozen Fix 8 contract exactly: official card-scoped
TUE/FRI chronological Ridge OOF (frozen ``RIDGE_ALPHA_100`` on the frozen
six Elo features, Fix 7/Fix 7.1 V2), same-horizon uncertainty (Fix 3,
unchanged), raw timestamped Odds API market reconstruction (never the
pre-generated T10/opening exports), ATS/TOTAL pricing, four independent
calibration streams (ATS_TUE, ATS_FRI, TOTAL_TUE, TOTAL_FRI), a 2025
post-freeze chronological replay/audit, and a production calibration seed
for the first 2026 forecast.

Does not redesign Fix 7.1 semantics, feature selection, model family
selection, Ridge hyperparameters, horizon definitions, market
reconstruction rules, or calibration family/rules -- those are frozen
inputs, verified by hash at Phase 1 and re-verified at every later phase
entry via :func:`_verify_preregistration_unchanged`.

Phase order:
  1. GATE: repo/base/origin-main/clean-tree + Fix 7.1 V2 hash verification
     (including a live recomputation of the CERTIFIED 2020-2024-only
     horizon membership ledger hash -- an algorithm-fidelity check,
     distinct from Phase 2's own full 2020-2025 official ledger).
  2. Official 2020-2025 card-scoped horizon membership ledger + hard
     membership-count gates (TUE=1693, FRI=1581; 2025 subset TUE=285,
     FRI=264).
  3. Official eligible-target-only TUE/FRI feature matrices (frozen six
     Elo features).
  4. PREREGISTRATION FREEZE (both horizons, both markets, all rules) +
     focused prefit tests -- before any real fit or any 2025 metric.
  5. Official chronological OOF Ridge (TUE, then FRI) + same-horizon
     uncertainty; fit-ceiling gate.
  6. Raw timestamped bookmaker-history market reconstruction (ATS, TOTAL)
     + market reconciliation hard gates (historical and 2025 subset).
  7. ATS/TOTAL pricing (line-independent joint state first, market
     introduced only after).
  8. Four independent calibration streams (conditional + push).
  9. Calibration-readiness-by-end-2025 gate + descriptive 2025 post-freeze
     audit (never used to select or tune anything).
 10. Production calibration seed for the first 2026 forecast.
 11. Persist row-level evidence (large -> NFL_MODEL_ARTIFACT_ROOT; small
     deterministic summaries -> outputs/).
 12. Final freeze + full pytest + ``git diff --check``.
 13. ``docs/MIKE_HANDOFF.md`` (success only) + status-doc pointer.

Any :class:`nfl_hybrid.selection.feature_deduction_2026.HardGateFailure`
stops the run immediately, no retry. Never commits, never opens a PR.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.calibration.three_way import (
    CalibrationConfig,
    _fit_conditional_calibrator,
    _fit_push_scales,
)
from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve
from nfl_hybrid.evaluation import official_horizon_oof as ohf
from nfl_hybrid.evaluation import raw_market_reconstruction as rmr
from nfl_hybrid.evaluation.chronological_calibration import (
    CALIBRATOR_FAMILY,
    MARKET_ATS,
    MARKET_TOTAL,
    ChronologicalCalibrationConfig,
    build_raw_probabilities,
    calibrate_chronological_push_probability,
    compute_calibrator_config_hash,
    compute_push_calibration_config_hash,
    generate_chronological_calibration,
)
from nfl_hybrid.evaluation.week1_reliability import NOT_ESTIMABLE, binary_log_loss, brier, equal_mass_ece
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.selection import feature_deduction_2026 as fd
from nfl_hybrid.selection import model_family_selection_2026 as mfs

SCHEMA_VERSION = "fix8-official-oof-tue-fri-calibration-v1"

EXPECTED_BRANCH = "fix/official-oof-tue-fri-calibration-2026"
EXPECTED_BASE_SHA = "404f126cf887aca6d7a991d5d8338db2486a4b12"

EXPECTED_HORIZON_FEATURE_SEMANTICS_VERSION = "HORIZON_CUTOFF_ASOF_ELO_V2_CARD_SCOPED"
EXPECTED_HORIZON_FEATURE_SEMANTICS_HASH = "bf0b136dd9b7c7f3741617b6c088926e539406d60b82563839eb1825de9fc72d"
EXPECTED_HORIZON_MEMBERSHIP_LEDGER_HASH = "b095a2bf9177029eb67e1516fbc79b080ef800cfe18b86989254dd4d0a8dae49"
EXPECTED_OPERATIONAL_MODEL_SPEC_HASH = "3b230bfeee3279c0e3ed9b6a7118931c1a5cf08203155be011f362c66ee8d722"
EXPECTED_RIDGE_ALPHA_100_SPEC_HASH = "27b3eec6554e1bdefdcdda3f6a6bb9311efa675c850f059aa71881d6c77978d7"
EXPECTED_FIX7_FINAL_MODEL_SPEC_HASH = "418724cb32fcb9a6a58f44fc77a7d579dbbc64d8078991c13d363751892ddede"

ALLOWED_DIRTY_PATHS = {
    "REMEDIATION_STATUS_2026.md",
    "docs/MIKE_HANDOFF.md",
    "src/nfl_hybrid/evaluation/official_horizon_oof.py",
    "src/nfl_hybrid/evaluation/raw_market_reconstruction.py",
    "scripts/run_fix8_official_oof_calibration.py",
    "tests/test_official_horizon_oof.py",
    "tests/test_raw_market_reconstruction.py",
}
ALLOWED_DIRTY_PREFIXES = ("outputs/fix8_",)

FOCUSED_TEST_FILES = [
    "tests/test_official_horizon_oof.py",
    "tests/test_raw_market_reconstruction.py",
    "tests/test_horizon_elo.py",
    "tests/test_chronological_oof.py",
    "tests/test_chronological_calibration.py",
    "tests/test_three_way_calibration.py",
    "tests/test_labels.py",
]

FAILURE_ARTIFACT_PREFIX = "outputs/fix8_failure_"

# Section 2 hard membership gates (full 2020-2025 official population).
EXPECTED_TUE_ELIGIBLE = 1693
EXPECTED_FRI_ELIGIBLE = 1581
EXPECTED_TUE_ELIGIBLE_2025 = 285
EXPECTED_FRI_ELIGIBLE_2025 = 264

# Section 4 fit ceiling.
MAX_PAIRED_FITS = 289
MAX_INDIVIDUAL_FITS = 578
EXPECTED_MAX_TUE_BATCHES = 131
EXPECTED_MAX_FRI_BATCHES = 131

# Section 6/7 reconciliation.
FRESHNESS_MAX_AGE_HOURS = rmr.FRESHNESS_MAX_AGE_HOURS
MINIMUM_FRESH_COHERENT_BOOKS = rmr.MINIMUM_FRESH_COHERENT_BOOKS
EXPECTED_MARKET_RECONCILIATION = {
    ("TUE", MARKET_ATS): {"eligible": 1693, "coherent_leq_cutoff": 917, "ge3_before_age_filter": 894, "market_ready": 844},
    ("TUE", MARKET_TOTAL): {"eligible": 1693, "coherent_leq_cutoff": 903, "ge3_before_age_filter": 876, "market_ready": 826},
    ("FRI", MARKET_ATS): {"eligible": 1581, "coherent_leq_cutoff": 1568, "ge3_before_age_filter": 1568, "market_ready": 1448},
    ("FRI", MARKET_TOTAL): {"eligible": 1581, "coherent_leq_cutoff": 1568, "ge3_before_age_filter": 1565, "market_ready": 1446},
}
EXPECTED_MARKET_RECONCILIATION_2025 = {
    ("TUE", MARKET_ATS): {"eligible": 285, "coherent_leq_cutoff": 190, "ge3_before_age_filter": 189, "market_ready": 177},
    ("TUE", MARKET_TOTAL): {"eligible": 285, "coherent_leq_cutoff": 190, "ge3_before_age_filter": 188, "market_ready": 176},
    ("FRI", MARKET_ATS): {"eligible": 264, "coherent_leq_cutoff": 264, "ge3_before_age_filter": 264, "market_ready": 242},
    ("FRI", MARKET_TOTAL): {"eligible": 264, "coherent_leq_cutoff": 264, "ge3_before_age_filter": 264, "market_ready": 242},
}

MIN_CALIBRATION_SAMPLE_COUNT = 100
STREAM_NAMES = ("ATS_TUE", "ATS_FRI", "TOTAL_TUE", "TOTAL_FRI")
_MARKET_BY_STREAM = {"ATS_TUE": MARKET_ATS, "ATS_FRI": MARKET_ATS, "TOTAL_TUE": MARKET_TOTAL, "TOTAL_FRI": MARKET_TOTAL}
_HORIZON_BY_STREAM = {"ATS_TUE": "TUE", "ATS_FRI": "FRI", "TOTAL_TUE": "TUE", "TOTAL_FRI": "FRI"}
_FORECAST_HORIZON_LABEL = {"TUE": "official_card_scoped_tue", "FRI": "official_card_scoped_fri"}
_LEGACY_PUSH_MARKET_NAME = {MARKET_ATS: "pregame_ats", MARKET_TOTAL: "pregame_total"}


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise fd.HardGateFailure(f"command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout.rstrip("\n")


def _single_threaded_env() -> dict:
    import os

    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _run_focused_tests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"] + FOCUSED_TEST_FILES,
        cwd=REPO_ROOT, capture_output=True, text=True, env=_single_threaded_env(),
    )
    print(result.stdout[-4000:])
    if result.returncode != 0:
        raise fd.HardGateFailure(f"REQUIRED_TEST_FAILED:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")
    return result.stdout[-2000:]


def _run_full_pytest() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=_single_threaded_env(), timeout=2400,
    )
    print(result.stdout[-6000:])
    if result.returncode != 0:
        raise fd.HardGateFailure(f"FULL_PYTEST_FAILED:\n{result.stdout[-6000:]}\n{result.stderr[-2000:]}")
    return result.stdout[-3000:]


def _canonical_json(payload: dict) -> bytes:
    # Fix 8 contract: canonical JSON primitives only -- no default=str, no
    # timestamps/paths/mtimes in scientific hash payloads. A non-primitive
    # object in a hash payload must raise, never be silently stringified.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(payload: dict) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


# ===========================================================================
# PHASE 1 -- repo/base/origin-main/clean-tree gate + Fix 7.1 V2 hash
# verification, including a live recomputation of the CERTIFIED
# 2020-2024-only horizon membership ledger hash (algorithm-fidelity check).
# ===========================================================================
def phase1_gate() -> dict:
    branch = _run(["git", "branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise fd.HardGateFailure(f"WRONG_BRANCH: on {branch!r}, expected {EXPECTED_BRANCH!r}")

    head_sha = _run(["git", "rev-parse", "HEAD"])
    if head_sha != EXPECTED_BASE_SHA:
        _run(["git", "merge-base", "--is-ancestor", EXPECTED_BASE_SHA, "HEAD"])

    origin_main_sha = _run(["git", "rev-parse", "origin/main"])
    if origin_main_sha != EXPECTED_BASE_SHA:
        raise fd.HardGateFailure(f"ORIGIN_MAIN_MOVED: {origin_main_sha!r} != {EXPECTED_BASE_SHA!r}")

    status_lines = [line for line in _run(["git", "status", "--short"]).splitlines() if line.strip()]
    unexpected = []
    for line in status_lines:
        path = line[3:].strip()
        if path in ALLOWED_DIRTY_PATHS or any(path.startswith(p) for p in ALLOWED_DIRTY_PREFIXES):
            continue
        unexpected.append(path)
    if unexpected:
        raise fd.HardGateFailure(f"UNEXPECTED_DIRTY_FILES: {unexpected}")

    fix71_summary = json.loads(
        (REPO_ROOT / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text()
    )
    if fix71_summary["horizon_feature_semantics_version"] != EXPECTED_HORIZON_FEATURE_SEMANTICS_VERSION:
        raise fd.HardGateFailure("HORIZON_FEATURE_SEMANTICS_VERSION_MISMATCH")
    if fix71_summary["horizon_feature_semantics_hash"] != EXPECTED_HORIZON_FEATURE_SEMANTICS_HASH:
        raise fd.HardGateFailure("HORIZON_FEATURE_SEMANTICS_HASH_MISMATCH")
    if fix71_summary["horizon_membership_ledger_hash"] != EXPECTED_HORIZON_MEMBERSHIP_LEDGER_HASH:
        raise fd.HardGateFailure("HORIZON_MEMBERSHIP_LEDGER_HASH_MISMATCH")
    if fix71_summary["operational_model_spec_hash"] != EXPECTED_OPERATIONAL_MODEL_SPEC_HASH:
        raise fd.HardGateFailure("OPERATIONAL_MODEL_SPEC_HASH_MISMATCH")
    if fix71_summary["fix7_final_model_spec_hash"] != EXPECTED_FIX7_FINAL_MODEL_SPEC_HASH:
        raise fd.HardGateFailure("FIX7_FINAL_MODEL_SPEC_HASH_MISMATCH")

    live_registry_hash = mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY)
    ridge_100 = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "RIDGE_ALPHA_100")
    live_ridge_spec_hash = mfs.compute_candidate_spec_hash(ridge_100)
    if live_ridge_spec_hash != EXPECTED_RIDGE_ALPHA_100_SPEC_HASH:
        raise fd.HardGateFailure(f"RIDGE_ALPHA_100_SPEC_HASH_MISMATCH: live={live_ridge_spec_hash}")

    frozen_feature_columns = list(fix71_summary["frozen_feature_columns"])
    if tuple(frozen_feature_columns) != ohf.ELO_FEATURE_COLUMNS:
        raise fd.HardGateFailure("FROZEN_FEATURE_COLUMNS_MISMATCH_LIVE_REGISTRY")

    # Algorithm-fidelity check: rebuild the EXACT population Fix 7.1 itself
    # certified (season<=2024 via fd.enforce_2025_firewall) and confirm the
    # membership-ledger algorithm still reproduces the certified hash. This
    # is NOT Fix 8's own official 2020-2025 ledger (Phase 2 builds that).
    raw_games = pd.read_parquet(resolve("backfill.games"))
    fix71_scope_games, _ = fd.enforce_2025_firewall(raw_games)
    fix71_scope_ledger = he.build_horizon_membership_ledger(fix71_scope_games)
    live_fix71_ledger_hash = he.compute_horizon_membership_ledger_hash(fix71_scope_ledger)
    if live_fix71_ledger_hash != EXPECTED_HORIZON_MEMBERSHIP_LEDGER_HASH:
        raise fd.HardGateFailure(
            f"LIVE_FIX71_LEDGER_ALGORITHM_DRIFTED: live={live_fix71_ledger_hash} expected={EXPECTED_HORIZON_MEMBERSHIP_LEDGER_HASH}"
        )

    return {
        "branch": branch, "head_sha": head_sha, "origin_main_sha": origin_main_sha,
        "dirty_files": [line[3:].strip() for line in status_lines],
        "horizon_feature_semantics_version": EXPECTED_HORIZON_FEATURE_SEMANTICS_VERSION,
        "horizon_feature_semantics_hash": EXPECTED_HORIZON_FEATURE_SEMANTICS_HASH,
        "horizon_membership_ledger_hash_fix71_certified_2020_2024_scope": EXPECTED_HORIZON_MEMBERSHIP_LEDGER_HASH,
        "operational_model_spec_hash": EXPECTED_OPERATIONAL_MODEL_SPEC_HASH,
        "ridge_alpha_100_candidate_spec_hash": live_ridge_spec_hash,
        "candidate_registry_hash": live_registry_hash,
        "fix7_final_model_spec_hash": EXPECTED_FIX7_FINAL_MODEL_SPEC_HASH,
        "frozen_feature_columns": frozen_feature_columns,
    }


# ===========================================================================
# PHASE 2 -- official 2020-2025 card-scoped horizon membership ledger +
# hard membership-count gates.
# ===========================================================================
def phase2_membership(gate: dict) -> dict:
    raw_games = pd.read_parquet(resolve("backfill.games"))
    raw_games = raw_games.copy()
    raw_games["game_id"] = raw_games["game_id"].astype(str)
    seasons_all = pd.to_numeric(raw_games["season"], errors="raise")
    raw_rows_season_2026_seen = int((seasons_all >= 2026).sum())

    games = raw_games[raw_games["season_type"].isin(["REG", "POST"])].copy()
    games = games[pd.to_numeric(games["season"], errors="raise") <= 2025].reset_index(drop=True)
    if int(pd.to_numeric(games["season"], errors="raise").max()) > 2025:
        raise fd.HardGateFailure("2026_OUTCOME_LEAKED_INTO_OFFICIAL_POPULATION")

    ledger = he.build_horizon_membership_ledger(games)

    tue_eligible = int(ledger["tue_eligible"].sum())
    fri_eligible = int(ledger["fri_eligible"].sum())
    if tue_eligible != EXPECTED_TUE_ELIGIBLE:
        raise fd.HardGateFailure(f"TUE_ELIGIBLE_MISMATCH: {tue_eligible} != {EXPECTED_TUE_ELIGIBLE}")
    if fri_eligible != EXPECTED_FRI_ELIGIBLE:
        raise fd.HardGateFailure(f"FRI_ELIGIBLE_MISMATCH: {fri_eligible} != {EXPECTED_FRI_ELIGIBLE}")

    season_num = pd.to_numeric(ledger["season"], errors="raise")
    tue_eligible_2025 = int((ledger["tue_eligible"] & (season_num == 2025)).sum())
    fri_eligible_2025 = int((ledger["fri_eligible"] & (season_num == 2025)).sum())
    if tue_eligible_2025 != EXPECTED_TUE_ELIGIBLE_2025:
        raise fd.HardGateFailure(f"TUE_ELIGIBLE_2025_MISMATCH: {tue_eligible_2025} != {EXPECTED_TUE_ELIGIBLE_2025}")
    if fri_eligible_2025 != EXPECTED_FRI_ELIGIBLE_2025:
        raise fd.HardGateFailure(f"FRI_ELIGIBLE_2025_MISMATCH: {fri_eligible_2025} != {EXPECTED_FRI_ELIGIBLE_2025}")

    membership_counts = {
        "tue_eligible": tue_eligible, "fri_eligible": fri_eligible,
        "tue_eligible_2025": tue_eligible_2025, "fri_eligible_2025": fri_eligible_2025,
        "raw_rows_season_2026_seen_by_firewall": raw_rows_season_2026_seen,
        "rows_season_ge_2026_used_for_anything": 0,
    }
    print(f"Phase 2 OK: {membership_counts}")
    return {"games": games, "ledger": ledger, "membership_counts": membership_counts}


# ===========================================================================
# PHASE 3 -- official eligible-target-only TUE/FRI matrices.
# ===========================================================================
def phase3_build_matrices(membership: dict) -> dict:
    games, ledger = membership["games"], membership["ledger"]
    matrices = {}
    for horizon in he.HORIZONS:
        matrix = ohf.build_official_horizon_matrix(games, horizon, ledger)
        matrices[horizon] = matrix
        n_batches = int(matrix["target_cutoff_utc"].nunique())
        print(f"Phase 3 [{horizon}] OK: rows={len(matrix)} n_batches={n_batches}")
    return {"matrices": matrices}


# ===========================================================================
# PHASE 4 -- preregistration freeze (both horizons, both markets, every
# rule) + focused prefit tests, before any real fit or any 2025 metric.
# ===========================================================================
def phase4_preregistration(gate: dict, membership: dict, matrix_info: dict) -> dict:
    cfg = ohf.OfficialHorizonOOFConfig()
    reconciliation = {f"{h}_{m}": v for (h, m), v in EXPECTED_MARKET_RECONCILIATION.items()}
    reconciliation_2025 = {f"{h}_{m}": v for (h, m), v in EXPECTED_MARKET_RECONCILIATION_2025.items()}

    body = {
        "schema_version": SCHEMA_VERSION,
        "repo": {
            "branch": gate["branch"], "head_sha": gate["head_sha"],
            "origin_main_sha": gate["origin_main_sha"], "required_base_sha": EXPECTED_BASE_SHA,
        },
        "fix71_hashes": {
            "horizon_feature_semantics_version": gate["horizon_feature_semantics_version"],
            "horizon_feature_semantics_hash": gate["horizon_feature_semantics_hash"],
            "horizon_membership_ledger_hash_fix71_certified_2020_2024_scope": gate[
                "horizon_membership_ledger_hash_fix71_certified_2020_2024_scope"
            ],
            "operational_model_spec_hash": gate["operational_model_spec_hash"],
        },
        "fix7_final_model_spec_hash": gate["fix7_final_model_spec_hash"],
        "ridge_alpha_100_candidate_spec_hash": gate["ridge_alpha_100_candidate_spec_hash"],
        "candidate_registry_hash": gate["candidate_registry_hash"],
        "ordered_six_elo_features": list(gate["frozen_feature_columns"]),
        "estimator": {
            "name": ohf.MODEL_NAME, "preprocessing": ohf.RIDGE_PREPROCESSING, "hyperparameters": ohf.RIDGE_HYPERPARAMETERS,
            "target_sharing_rule": "SAME_FAMILY_AND_HYPERPARAMETERS_FOR_MARGIN_AND_TOTAL_FIT_SEPARATELY",
        },
        "card_cutoff_contract": {
            "card_membership_key": list(he.CARD_KEY),
            "card_monday_algorithm": "earliest kickoff in card -> America/New_York local date -> minus weekday() days",
            "tue_cutoff_rule": "card_monday + 1 day, 12:00 America/New_York, DST-aware, converted to UTC",
            "fri_cutoff_rule": "card_monday + 4 days, 12:00 America/New_York, DST-aware, converted to UTC",
        },
        "target_eligibility_rule": (
            "target_cutoff_utc(card, horizon) < scheduled_kickoff_utc(game) (STRICT); an ineligible target "
            "is excluded entirely, never floated to a different card's cutoff"
        ),
        "chronological_training_eligibility_rule": (
            "a training row must belong to the SAME horizon's own eligible-target population AND satisfy "
            "result_available_at_utc < target_cutoff_utc (STRICT), using its own historically-generated "
            "horizon-as-of feature vector"
        ),
        "result_availability_basis": he.RESULT_AVAILABILITY_BASIS,
        "result_availability_duration_hours": he.RESULT_AVAILABILITY_DURATION_HOURS,
        "model_readiness_rule": {
            "min_training_games": cfg.min_training_games, "not_ready_status": "MODEL_NOT_READY",
            "source": "nfl_hybrid.evaluation.chronological_oof.ChronologicalOOFConfig.min_training_games (reused unchanged)",
        },
        "official_oof_construction": (
            "fit once per horizon cutoff batch (one card), never once per game; margin and total fit "
            "separately; reuses Fix 3's own batching-by-shared-cutoff optimization and its own "
            "attach_outcomes_and_residuals"
        ),
        "same_horizon_uncertainty_rule": {
            "method": "nfl_hybrid.evaluation.chronological_oof.attach_expanding_oof_uncertainty (unchanged)",
            "min_uncertainty_warmup": cfg.min_uncertainty_warmup,
            "rho_clip": [-0.95, 0.95],
            "pooled_across_horizons": False,
        },
        "raw_market_reconstruction": {
            "source": (
                "raw Odds API timestamped bookmaker snapshot history (odds_history.2020_2023 / "
                "2024_confirmation / 2025_final_test bookmaker_quotes.parquet) -- never pre-generated "
                "opening/T10 horizon exports"
            ),
            "coherence_rule": (
                "both sides of a market quote from the same (game_id, bookmaker_key, returned_snapshot_utc, "
                "market_last_update); spread points exact opposites; total points identical; both decimal "
                "prices > 1.0"
            ),
            "as_of_rule": "market_last_update <= returned_snapshot_utc <= target_cutoff_utc",
            "freshness_max_age_hours": FRESHNESS_MAX_AGE_HOURS,
            "freshness_applied_before_book_counting": True,
            "minimum_fresh_coherent_books": MINIMUM_FRESH_COHERENT_BOOKS,
            "consensus_method": rmr.CONSENSUS_METHOD,
            "no_t10_five_minute_lag_rule": True,
            "no_synthetic_minus_110": True,
        },
        "ats_pricing": (
            "line-independent joint football state computed first (official Ridge OOF + same-horizon "
            "uncertainty); nfl_hybrid.evaluation.chronological_calibration.build_raw_probabilities"
            "(market=ATS, threshold=official raw consensus spread) introduced only after"
        ),
        "total_pricing": (
            "nfl_hybrid.evaluation.chronological_calibration.build_raw_probabilities"
            "(market=TOTAL, threshold=official raw consensus total)"
        ),
        "push_handling": (
            "nfl_hybrid.labels.edge_to_nullable_binary tie/push policy; push probability calibrated via "
            "nfl_hybrid.calibration.three_way._fit_push_scales/_predict_push (reused, chronological membership)"
        ),
        "four_calibration_streams": list(STREAM_NAMES),
        "calibration_family": CALIBRATOR_FAMILY,
        "calibrator_config_hash": compute_calibrator_config_hash(ChronologicalCalibrationConfig()),
        "calibration_warmup_minimum": MIN_CALIBRATION_SAMPLE_COUNT,
        "calibration_chronology_rule": (
            "an observation enters a stream's calibration history only if it shares that stream's own "
            "market/horizon, its raw prediction was OOF at its own historical cutoff, and "
            "result_available_at_utc < C (STRICT)"
        ),
        "no_horizon_pooling": True,
        "no_ats_total_pooling": True,
        "twenty_twenty_five_role": {
            "used_2025_for_selection": False,
            "used_2025_for_rule_tuning": False,
            "used_2025_for_postfreeze_chronological_replay": True,
            "twenty_twenty_six_outcomes_forbidden": True,
        },
        "fit_ceiling": {
            "expected_max_paired": 262, "expected_max_individual": 524,
            "hard_ceiling_paired": MAX_PAIRED_FITS, "hard_ceiling_individual": MAX_INDIVIDUAL_FITS,
            "expected_max_batches_per_horizon": {"TUE": EXPECTED_MAX_TUE_BATCHES, "FRI": EXPECTED_MAX_FRI_BATCHES},
        },
        "success_criteria_summary": (
            "repo/hash gates pass; Fix 7.1 V2 semantics used; TUE eligible=1693; FRI eligible=1581; "
            "chronological OOF valid; fit ceiling respected; same-horizon uncertainty valid; raw market "
            "reconstruction valid; expected market reconciliation explained/reproduced; no market leakage "
            "into Ridge; all four calibration histories chronological; all four streams ready by end 2025; "
            "2025 changed no frozen rule; no 2026 outcome used; scientific hashes deterministic; focused "
            "tests pass; full pytest passes; git diff --check clean"
        ),
        "failure_rule": "do not change frozen rules just to achieve PASS; any hard gate failure stops the run immediately, no retry",
        "membership_counts": membership["membership_counts"],
        "matrix_contracts": {
            horizon: {
                "row_count": int(len(matrix_info["matrices"][horizon])),
                "n_batches": int(matrix_info["matrices"][horizon]["target_cutoff_utc"].nunique()),
            }
            for horizon in he.HORIZONS
        },
        "expected_market_reconciliation": reconciliation,
        "expected_market_reconciliation_2025": reconciliation_2025,
    }

    pytest_tail = _run_focused_tests()

    preregistration_hash = _sha256_hex(body)
    preregistration = {**body, "fix8_official_oof_calibration_preregistration_hash": preregistration_hash}

    out_path = REPO_ROOT / "outputs" / "fix8_official_oof_calibration_preregistration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8")

    preregistration["_phase4_pretest_output_tail"] = pytest_tail
    return preregistration


def _verify_preregistration_unchanged(preregistration: dict) -> None:
    body = {
        k: v for k, v in preregistration.items()
        if not k.startswith("_") and k != "fix8_official_oof_calibration_preregistration_hash"
    }
    if _sha256_hex(body) != preregistration["fix8_official_oof_calibration_preregistration_hash"]:
        raise fd.HardGateFailure("PREREGISTRATION_HASH_MUTATED_AFTER_FREEZE")


# ===========================================================================
# PHASE 5 -- official chronological OOF Ridge (TUE, FRI) + same-horizon
# uncertainty; fit-ceiling gate.
# ===========================================================================
def phase5_official_oof(preregistration: dict, matrix_info: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    results = {}
    total_paired = 0
    total_individual = 0
    for horizon in he.HORIZONS:
        matrix = matrix_info["matrices"][horizon]
        n_batches = int(matrix["target_cutoff_utc"].nunique())
        expected_max = EXPECTED_MAX_TUE_BATCHES if horizon == "TUE" else EXPECTED_MAX_FRI_BATCHES
        if n_batches > expected_max:
            raise fd.HardGateFailure(f"{horizon}_BATCH_COUNT_EXCEEDS_EXPECTATION: {n_batches} > {expected_max}")

        predictions, residual_ledger, fit_counts = ohf.build_official_horizon_oof(
            matrix, horizon=horizon, feature_state_hash=preregistration["fix71_hashes"]["horizon_feature_semantics_hash"],
        )
        total_paired += fit_counts["paired_fits"]
        total_individual += fit_counts["individual_fits"]
        results[horizon] = {"predictions": predictions, "residual_ledger": residual_ledger, "fit_counts": fit_counts, "n_batches": n_batches}
        n_oof = int((predictions["status"] == "OOF").sum())
        n_not_ready = int((predictions["status"] == "MODEL_NOT_READY").sum())
        print(f"Phase 5 [{horizon}] OK: n_batches={n_batches} fit_counts={fit_counts} n_oof={n_oof} n_not_ready={n_not_ready}")

    if total_paired > MAX_PAIRED_FITS or total_individual > MAX_INDIVIDUAL_FITS:
        raise fd.HardGateFailure(f"FIT_CEILING_EXCEEDED: paired={total_paired} individual={total_individual}")

    # Defensive re-assertion: the six frozen features actually used to fit
    # never include a forbidden current-market column (already enforced at
    # matrix-build time).
    fd.assert_no_forbidden_market_columns(list(ohf.ELO_FEATURE_COLUMNS))

    return {"per_horizon": results, "fit_counts": {"total_paired": total_paired, "total_individual": total_individual}}


# ===========================================================================
# PHASE 6 -- raw timestamped bookmaker-history market reconstruction +
# reconciliation hard gates (historical and 2025 subset).
# ===========================================================================
def phase6_market_reconstruction(preregistration: dict, matrix_info: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)

    quotes = rmr.load_raw_bookmaker_quotes()
    coherent_ats = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    coherent_total = rmr.build_coherent_book_observations(quotes, rmr.MARKET_TOTALS)
    coherent_by_market = {MARKET_ATS: coherent_ats, MARKET_TOTAL: coherent_total}

    results = {}
    for horizon in he.HORIZONS:
        matrix = matrix_info["matrices"][horizon]
        targets = matrix[["game_id", "target_cutoff_utc"]].copy()
        targets_2025 = matrix.loc[matrix["season"] == 2025, ["game_id", "target_cutoff_utc"]]

        for market in (MARKET_ATS, MARKET_TOTAL):
            result = rmr.reconstruct_market_at_cutoffs(coherent_by_market[market], targets, market=market)
            expected = EXPECTED_MARKET_RECONCILIATION[(horizon, market)]
            for key in ("eligible", "coherent_leq_cutoff", "ge3_before_age_filter", "market_ready"):
                if result.coverage[key] != expected[key]:
                    raise fd.HardGateFailure(
                        f"MARKET_RECONCILIATION_MISMATCH[{horizon}/{market}/{key}]: "
                        f"live={result.coverage} expected={expected}"
                    )

            result_2025 = rmr.reconstruct_market_at_cutoffs(coherent_by_market[market], targets_2025, market=market)
            expected_2025 = EXPECTED_MARKET_RECONCILIATION_2025[(horizon, market)]
            for key in ("eligible", "coherent_leq_cutoff", "ge3_before_age_filter", "market_ready"):
                if result_2025.coverage[key] != expected_2025[key]:
                    raise fd.HardGateFailure(
                        f"MARKET_RECONCILIATION_2025_MISMATCH[{horizon}/{market}/{key}]: "
                        f"live={result_2025.coverage} expected={expected_2025}"
                    )

            results[(horizon, market)] = result
            print(f"Phase 6 [{horizon}/{market}] OK: coverage={result.coverage} coverage_2025={result_2025.coverage}")

    return {"per_horizon_market": results, "quotes_row_count": int(len(quotes))}


# ===========================================================================
# PHASE 7 -- ATS/TOTAL pricing.
# ===========================================================================
def _price_market(residual_ledger: pd.DataFrame, consensus: pd.DataFrame, *, market: str, forecast_horizon: str):
    base = residual_ledger.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    if not consensus.empty:
        consensus_line = consensus.set_index("game_id")["consensus_line"]
    else:
        consensus_line = pd.Series(dtype=float)
    threshold = base["game_id"].map(consensus_line).to_numpy(dtype=float)
    raw = build_raw_probabilities(base, market=market, forecast_horizon=forecast_horizon, threshold=threshold)
    return raw, threshold


def phase7_pricing(preregistration: dict, oof_result: dict, market_result: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    priced = {}
    for stream in STREAM_NAMES:
        market = _MARKET_BY_STREAM[stream]
        horizon = _HORIZON_BY_STREAM[stream]
        residual_ledger = oof_result["per_horizon"][horizon]["residual_ledger"]
        consensus = market_result["per_horizon_market"][(horizon, market)].consensus
        forecast_horizon = _FORECAST_HORIZON_LABEL[horizon]
        raw, threshold = _price_market(residual_ledger, consensus, market=market, forecast_horizon=forecast_horizon)
        priced[stream] = {"raw": raw, "threshold": threshold}
        n_ready = int((raw["raw_status"] == "RAW_READY").sum())
        print(f"Phase 7 [{stream}] OK: n_rows={len(raw)} n_raw_ready={n_ready}")
    return priced


# ===========================================================================
# PHASE 8 -- four independent calibration streams (conditional + push).
# ===========================================================================
def phase8_calibration(preregistration: dict, priced: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    calib_cfg = ChronologicalCalibrationConfig()
    push_cfg = CalibrationConfig()
    ledgers = {}
    for stream in STREAM_NAMES:
        market = _MARKET_BY_STREAM[stream]
        raw = priced[stream]["raw"]
        threshold = priced[stream]["threshold"]
        conditional_ledger = generate_chronological_calibration(raw, config=calib_cfg)
        ledger = calibrate_chronological_push_probability(
            raw, conditional_ledger, market=market, market_line=threshold, config=push_cfg
        )
        n_cal = int((ledger["calibration_status"] == "CALIBRATED").sum())
        n_insuf = int((ledger["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY").sum())
        n_unavail = int((ledger["calibration_status"] == "RAW_UNAVAILABLE").sum())
        ledgers[stream] = ledger
        print(f"Phase 8 [{stream}] OK: n_calibrated={n_cal} n_insufficient_history={n_insuf} n_raw_unavailable={n_unavail}")
    return ledgers


# ===========================================================================
# PHASE 9 -- calibration-readiness-by-end-2025 gate + descriptive 2025
# post-freeze audit (never used to select or tune anything).
# ===========================================================================
def phase9_readiness_and_2025_audit(preregistration: dict, ledgers: dict, priced: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    readiness = {}
    audit = {}
    for stream in STREAM_NAMES:
        ledger = ledgers[stream]
        raw = priced[stream]["raw"]
        season_num = pd.to_numeric(ledger["season"], errors="coerce")
        ledger_2025 = ledger[season_num == 2025]

        if ledger_2025.empty:
            ready_by_end_2025 = False
            last_status = None
        else:
            last_row = ledger_2025.sort_values(["target_cutoff_utc", "game_id"], kind="stable").iloc[-1]
            last_status = str(last_row["calibration_status"])
            ready_by_end_2025 = (
                last_status == "CALIBRATED" and int(last_row["calibration_sample_count"]) >= MIN_CALIBRATION_SAMPLE_COUNT
            )
        readiness[stream] = {
            "ready_by_end_2025": bool(ready_by_end_2025),
            "last_2025_calibration_status": last_status,
            "n_2025_rows": int(len(ledger_2025)),
        }

        joined_2025 = ledger_2025.merge(raw[["game_id", "binary_target"]], on="game_id", how="left")
        scored = joined_2025[(joined_2025["calibration_status"] == "CALIBRATED") & joined_2025["binary_target"].notna()]
        n_pushes_excluded = int(
            joined_2025[(joined_2025["calibration_status"] == "CALIBRATED") & joined_2025["binary_target"].isna()].shape[0]
        )
        y = pd.to_numeric(scored["binary_target"], errors="coerce").to_numpy(float)
        p_raw = scored["raw_conditional_upper_probability"].to_numpy(float)
        p_cal = scored["calibrated_conditional_upper_probability"].to_numpy(float)
        if len(y):
            ece_raw = equal_mass_ece(y, p_raw)
            ece_cal = equal_mass_ece(y, p_cal)
            audit[stream] = {
                "label": "DESCRIPTIVE / NON-SELECTION ONLY",
                "n_scored": int(len(y)), "n_pushes_excluded": n_pushes_excluded,
                "raw_log_loss": binary_log_loss(y, p_raw), "calibrated_log_loss": binary_log_loss(y, p_cal),
                "raw_brier": brier(y, p_raw), "calibrated_brier": brier(y, p_cal),
                "raw_ece": ece_raw if ece_raw == NOT_ESTIMABLE else float(ece_raw),
                "calibrated_ece": ece_cal if ece_cal == NOT_ESTIMABLE else float(ece_cal),
            }
        else:
            audit[stream] = {"label": "DESCRIPTIVE / NON-SELECTION ONLY", "n_scored": 0, "n_pushes_excluded": n_pushes_excluded}
        print(f"Phase 9 [{stream}] readiness={readiness[stream]} audit_2025={audit[stream]}")

    return {"readiness": readiness, "audit_2025": audit}


# ===========================================================================
# PHASE 10 -- production calibration seed for the first 2026 forecast.
# ===========================================================================
def phase10_production_seed(preregistration: dict, priced: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    calib_cfg = ChronologicalCalibrationConfig()
    push_cfg = CalibrationConfig()
    seeds = {}
    for stream in STREAM_NAMES:
        market = _MARKET_BY_STREAM[stream]
        raw = priced[stream]["raw"]
        threshold = np.asarray(priced[stream]["threshold"], dtype=float)

        ready_mask = (raw["raw_status"] == "RAW_READY").to_numpy()
        binary_numeric = pd.to_numeric(raw["binary_target"], errors="coerce").to_numpy(float)
        labeled_mask = ready_mask & np.isfinite(binary_numeric)
        n_labeled = int(labeled_mask.sum())
        conditional_upper = raw["raw_conditional_upper_probability"].to_numpy(dtype=float)

        model = None
        if n_labeled >= MIN_CALIBRATION_SAMPLE_COUNT:
            model = _fit_conditional_calibrator(
                conditional_upper[labeled_mask], binary_numeric[labeled_mask].astype(int), calib_cfg.calibration_config
            )
        if model is not None:
            conditional_state = {
                "fitted": True, "coefficient": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
            }
        else:
            conditional_state = {"fitted": False, "reason": "INSUFFICIENT_HISTORY" if n_labeled < MIN_CALIBRATION_SAMPLE_COUNT else "CALIBRATOR_DECLINED"}

        legacy_market = _LEGACY_PUSH_MARKET_NAME[market]
        actual_push = (ready_mask & ~np.isfinite(binary_numeric)).astype(int)
        model_push = pd.to_numeric(raw["raw_push_probability"], errors="coerce").to_numpy(float)
        push_state = None
        if ready_mask.any():
            push_train = pd.DataFrame(
                {
                    "model_push_probability": model_push[ready_mask],
                    "actual_push": actual_push[ready_mask],
                    "market_line": threshold[ready_mask],
                }
            )
            global_scale, bucket_scales = _fit_push_scales(push_train, legacy_market, push_cfg)
            push_state = {"global_scale": global_scale, "bucket_scales": bucket_scales}

        max_result_available = (
            pd.to_datetime(raw.loc[ready_mask, "result_available_at_utc"], utc=True).max() if ready_mask.any() else None
        )

        seeds[stream] = {
            "market": market,
            "n_raw_ready": int(ready_mask.sum()),
            "n_labeled_non_push": n_labeled,
            "calibrator_family": CALIBRATOR_FAMILY,
            "calibrator_config_hash": compute_calibrator_config_hash(calib_cfg),
            "conditional_calibrator_state": conditional_state,
            "push_calibration_config_hash": compute_push_calibration_config_hash(push_cfg, market),
            "push_scale_state": push_state,
            "latest_included_result_available_at_utc": str(max_result_available) if max_result_available is not None else None,
            "horizon_feature_semantics_hash": preregistration["fix71_hashes"]["horizon_feature_semantics_hash"],
            "operational_model_spec_hash": preregistration["fix71_hashes"]["operational_model_spec_hash"],
            "preregistration_hash": preregistration["fix8_official_oof_calibration_preregistration_hash"],
            "twenty_twenty_six_outcome_used": False,
        }
        print(f"Phase 10 [{stream}] seed OK: n_raw_ready={seeds[stream]['n_raw_ready']} conditional_fitted={conditional_state.get('fitted')}")
    return seeds


# ===========================================================================
# PHASE 11 -- persist row-level evidence.
# ===========================================================================
def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if len(out) and isinstance(out[col].iloc[0], tuple):
            out[col] = out[col].apply(lambda v: list(v) if isinstance(v, tuple) else v)
    return out


def phase11_persist_evidence(oof_result: dict, market_result: dict, priced: dict, ledgers: dict, seeds: dict) -> dict:
    root = artifact_root() / "fix8-official-oof-calibration-2026"
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for horizon in he.HORIZONS:
        path = root / f"{horizon.lower()}_official_oof_residual_ledger.parquet"
        _parquet_safe(oof_result["per_horizon"][horizon]["residual_ledger"]).to_parquet(path, index=False)
        written[f"{horizon}_residual_ledger"] = str(path)

    for (horizon, market), result in market_result["per_horizon_market"].items():
        path = root / f"{horizon.lower()}_{market.lower()}_market_consensus.parquet"
        _parquet_safe(result.consensus).to_parquet(path, index=False)
        written[f"{horizon}_{market}_market_consensus"] = str(path)

    for stream in STREAM_NAMES:
        raw_path = root / f"{stream.lower()}_raw_probabilities.parquet"
        _parquet_safe(priced[stream]["raw"]).to_parquet(raw_path, index=False)
        written[f"{stream}_raw"] = str(raw_path)

        ledger_path = root / f"{stream.lower()}_calibration_ledger.parquet"
        _parquet_safe(ledgers[stream]).to_parquet(ledger_path, index=False)
        written[f"{stream}_ledger"] = str(ledger_path)

    seed_path = root / "production_calibration_seed.json"
    seed_path.write_text(json.dumps(seeds, indent=2, sort_keys=True, default=str), encoding="utf-8")
    written["production_calibration_seed"] = str(seed_path)

    print(f"Phase 11 OK: wrote {len(written)} evidence artifacts under {root}")
    return written


# ===========================================================================
# PHASE 12 -- final freeze + full pytest + git diff --check.
# ===========================================================================
def phase12_final_freeze_and_evidence(
    preregistration: dict, gate: dict, membership: dict, oof_result: dict, market_result: dict,
    priced: dict, ledgers: dict, readiness_and_audit: dict, seeds: dict, written_evidence: dict,
) -> dict:
    _verify_preregistration_unchanged(preregistration)

    all_ready = all(readiness_and_audit["readiness"][s]["ready_by_end_2025"] for s in STREAM_NAMES)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "fix8_official_oof_calibration_preregistration_hash": preregistration["fix8_official_oof_calibration_preregistration_hash"],
        "repo": {"branch": gate["branch"], "head_sha": gate["head_sha"], "required_base_sha": EXPECTED_BASE_SHA},
        "fix71_hashes": preregistration["fix71_hashes"],
        "fix7_final_model_spec_hash": preregistration["fix7_final_model_spec_hash"],
        "ridge_alpha_100_candidate_spec_hash": preregistration["ridge_alpha_100_candidate_spec_hash"],
        "ordered_six_elo_features": preregistration["ordered_six_elo_features"],
        "membership_counts": membership["membership_counts"],
        "fit_counts": oof_result["fit_counts"],
        "market_reconciliation": {
            f"{h}_{m}": market_result["per_horizon_market"][(h, m)].coverage
            for h in he.HORIZONS for m in (MARKET_ATS, MARKET_TOTAL)
        },
        "per_stream": {
            s: {
                "n_raw_ready": int((priced[s]["raw"]["raw_status"] == "RAW_READY").sum()),
                "n_calibrated": int((ledgers[s]["calibration_status"] == "CALIBRATED").sum()),
                "n_uncalibrated_insufficient_history": int((ledgers[s]["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY").sum()),
                "n_raw_unavailable": int((ledgers[s]["calibration_status"] == "RAW_UNAVAILABLE").sum()),
                "readiness": readiness_and_audit["readiness"][s],
                "audit_2025": readiness_and_audit["audit_2025"][s],
            }
            for s in STREAM_NAMES
        },
        "production_calibration_seed_summary": {
            s: {k: v for k, v in seeds[s].items() if k not in ("conditional_calibrator_state", "push_scale_state")}
            for s in STREAM_NAMES
        },
        "twenty_twenty_five_role": {
            "used_2025_for_selection": False, "used_2025_for_rule_tuning": False,
            "used_2025_for_postfreeze_chronological_replay": True,
        },
        "all_four_streams_ready_by_end_2025": all_ready,
        "evidence_paths": written_evidence,
    }

    if not all_ready:
        raise fd.HardGateFailure(f"NOT_ALL_STREAMS_CALIBRATION_READY_BY_END_2025: {readiness_and_audit['readiness']}")

    out_path = REPO_ROOT / "outputs" / "fix8_official_oof_calibration_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    full_pytest_tail = _run_full_pytest()

    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, capture_output=True, text=True)
    if diff_check.returncode != 0 or diff_check.stdout.strip():
        raise fd.HardGateFailure(f"GIT_DIFF_CHECK_FAILED: {diff_check.stdout}")

    summary["full_pytest_output_tail"] = full_pytest_tail
    summary["git_diff_check"] = "clean"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


# ===========================================================================
# PHASE 13 -- docs/MIKE_HANDOFF.md (success only) + status-doc pointer.
# ===========================================================================
def _write_mike_handoff(summary: dict, preregistration: dict) -> None:
    reconciliation_lines = "\n".join(
        f"- `{k}`: eligible={v['eligible']}, coherent<=cutoff={v['coherent_leq_cutoff']}, "
        f">=3 before age filter={v['ge3_before_age_filter']}, market-ready={v['market_ready']}"
        for k, v in summary["market_reconciliation"].items()
    )
    stream_lines = "\n".join(
        f"- `{s}`: n_raw_ready={v['n_raw_ready']}, n_calibrated={v['n_calibrated']}, "
        f"ready_by_end_2025={v['readiness']['ready_by_end_2025']}, "
        f"2025 audit (descriptive only): log_loss raw={v['audit_2025'].get('raw_log_loss')} "
        f"calibrated={v['audit_2025'].get('calibrated_log_loss')}, "
        f"brier raw={v['audit_2025'].get('raw_brier')} calibrated={v['audit_2025'].get('calibrated_brier')}"
        for s, v in summary["per_stream"].items()
    )
    content = f"""<!-- Fix 8 handoff -- generated by scripts/run_fix8_official_oof_calibration.py -->
# Mike Shackelford Audit Handoff -- Official TUE/FRI OOF + ATS/TOTAL Calibration

## What the model predicts

For every NFL REG/POST game eligible as a Tuesday or Friday forecast target
(card-scoped, see below), the system predicts the game's point margin
(home - away) and total points, then converts those into ATS (against-the-
spread) and TOTAL (over/under) probabilities against the real market line
available at that forecast cutoff.

## Six frozen input features

Per team side (home/away), three pregame Elo quantities, card-scoped
horizon-as-of (Fix 7.1 V2): `elo_pregame_rating`, `elo_pregame_win_probability`,
`elo_pregame_expected_margin`. No market data, box-score, or play-by-play
feature is used -- Elo state alone.

## Estimator: RIDGE_ALPHA_100

`StandardScaler(with_mean=True, with_std=True)` -> `Ridge(alpha=100.0,
fit_intercept=True, solver="svd")`, fit separately for margin and total, once
per horizon cutoff batch (one weekly card), never once per game. Selected by
Fix 7's model-family selection (`final_model_spec_hash` =
`{preregistration['fix7_final_model_spec_hash']}`).

## TUE/FRI card-scoped forecast semantics

A "card" is one `(season, week, season_type)` slate. `card_monday_date` is
derived from the card's own earliest kickoff (America/New_York local
calendar, DST-aware). TUE cutoff = card_monday + 1 day, 12:00 America/New_York;
FRI cutoff = card_monday + 4 days, 12:00 America/New_York -- both converted to
UTC. A game is a valid target for a horizon only if that horizon's card
cutoff falls strictly before the game's own kickoff; an ineligible game (most
often a Thursday/Thanksgiving/Christmas game relative to the Friday cutoff)
is excluded from that horizon entirely, never re-assigned to a different
card's cutoff.

## ATS/TOTAL primary outputs

Four independent, never-pooled calibration streams: `ATS_TUE`, `ATS_FRI`,
`TOTAL_TUE`, `TOTAL_FRI`. Each stream's raw probability comes from the
line-independent joint football state (predicted margin/total plus
same-horizon OOF residual uncertainty) priced against the real consensus
market line, then chronologically calibrated (conditional + push) using only
prior, already-resolved observations from that same stream.

## Football state computed before market pricing

The predicted margin/total and their uncertainty (`sigma_margin`,
`sigma_total`, `rho`, clipped to [-0.95, 0.95]) are produced entirely from the
six Elo features and prior game results -- no market line, current or
historical, ever enters the Ridge fit or the uncertainty estimate. The real
market consensus line is joined on only afterward, to convert that
football state into ATS/TOTAL probabilities.

## Chronology / no-leakage guarantees

- A target's training set is restricted to games whose own six-feature
  vector and result were both fixed and result-available strictly before
  that target's own forecast cutoff (`result_available_at_utc < target_cutoff_utc`,
  STRICT) -- enforced structurally and re-checked by an automated invariant
  after every OOF/calibration run.
- Same-horizon uncertainty (Fix 3, reused unchanged) and each of the four
  calibration streams maintain independent history -- never pooled across
  TUE/FRI or across ATS/TOTAL.
- Every market observation used for pricing is a raw, timestamped
  per-bookmaker quote whose `market_last_update <= returned_snapshot_utc <=
  target_cutoff_utc` and whose age is at most 48 hours, applied BEFORE
  counting books; at least 3 fresh, coherent, complete two-sided bookmaker
  quotes are required.

## Historical data coverage

2020-2025 (REG+POST). TUE eligible targets: {summary['membership_counts']['tue_eligible']}.
FRI eligible targets: {summary['membership_counts']['fri_eligible']}. 2025 subset:
TUE={summary['membership_counts']['tue_eligible_2025']}, FRI={summary['membership_counts']['fri_eligible_2025']}.

### Market reconciliation (eligible / coherent<=cutoff / >=3 before age filter / market-ready)

{reconciliation_lines}

### Per-stream summary

{stream_lines}

## 2025 post-freeze role

All scientific rules (features, estimator, horizon cutoffs, market
reconstruction, calibration family/warmup/thresholds) were frozen in the
preregistration BEFORE any 2025 metric was computed. 2025 was then
chronologically replayed under those already-frozen rules and may
contribute to later 2025 calibration histories once its own results become
available -- it never changed a frozen rule
(`used_2025_for_selection=False`, `used_2025_for_rule_tuning=False`,
`used_2025_for_postfreeze_chronological_replay=True`). 2026 outcomes were
never used anywhere in this run.

## Major scientific hashes

- `fix8_official_oof_calibration_preregistration_hash`: `{summary['fix8_official_oof_calibration_preregistration_hash']}`
- `horizon_feature_semantics_hash` (Fix 7.1 V2): `{summary['fix71_hashes']['horizon_feature_semantics_hash']}`
- `operational_model_spec_hash` (Fix 7.1 V2): `{summary['fix71_hashes']['operational_model_spec_hash']}`
- `fix7_final_model_spec_hash`: `{summary['fix7_final_model_spec_hash']}`
- `ridge_alpha_100_candidate_spec_hash`: `{summary['ridge_alpha_100_candidate_spec_hash']}`

## Test commands

```
PYTHONPATH=src python -m pytest -q tests/test_official_horizon_oof.py tests/test_raw_market_reconstruction.py \\
    tests/test_horizon_elo.py tests/test_chronological_oof.py tests/test_chronological_calibration.py \\
    tests/test_three_way_calibration.py tests/test_labels.py
PYTHONPATH=src python -m pytest -q
```

## Evidence replay command

```
NFL_MODEL_DATA_ROOT=/path/to/data NFL_MODEL_ARTIFACT_ROOT=/path/to/artifacts \\
    PYTHONPATH=src python scripts/run_fix8_official_oof_calibration.py
```

Row-level evidence (OOF residual ledgers, market consensus, raw/calibration
ledgers per stream, production calibration seed) is written under
`NFL_MODEL_ARTIFACT_ROOT/fix8-official-oof-calibration-2026/`; small
deterministic summaries are written under `outputs/`.

## External/private data note

This handoff and its evidence reference a private, licensed historical data
estate (`NFL_MODEL_DATA_ROOT`) and a private raw historical Odds API
snapshot archive, neither of which is committed to this repository. Row-level
parquet evidence is not committed either -- only the small deterministic
JSON summaries under `outputs/` are.

## Remaining operational limitations

- Not yet wired into any live scheduler/orchestration path ("Tuesday/Friday
  orchestration" remains a separate, not-yet-started roadmap item).
- The production calibration seed reflects state as of the end of the 2025
  season; it has not yet priced or seen any 2026 game.
- No guarantee of profitability is made or implied by this model, its
  calibration, or its descriptive 2025 post-freeze diagnostics.
"""
    path = REPO_ROOT / "docs" / "MIKE_HANDOFF.md"
    path.write_text(content, encoding="utf-8")

    status_path = REPO_ROOT / "REMEDIATION_STATUS_2026.md"
    text = status_path.read_text(encoding="utf-8")
    if "MIKE_HANDOFF.md" not in text:
        text = text.rstrip("\n") + (
            "\n\nFix 8 (official chronological TUE/FRI OOF + ATS/TOTAL calibration) produced the Mike "
            "Shackelford audit handoff package -- see `docs/MIKE_HANDOFF.md`.\n"
        )
        status_path.write_text(text, encoding="utf-8")


def _write_failure_evidence(classification: str, phase: str, extra: dict) -> Path:
    payload = {"classification": classification, "failed_phase": phase, **extra}
    path = REPO_ROOT / f"{FAILURE_ARTIFACT_PREFIX}{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return path


def main() -> int:
    phase = "UNSTARTED"
    preregistration: dict = {}
    try:
        phase = "PHASE1_GATE"
        gate = phase1_gate()
        print("Phase 1 (repo/base/hash gates) OK:", json.dumps(gate, indent=2, default=str))

        phase = "PHASE2_MEMBERSHIP"
        membership = phase2_membership(gate)

        phase = "PHASE3_MATRICES"
        matrix_info = phase3_build_matrices(membership)

        phase = "PHASE4_PREREGISTRATION"
        preregistration = phase4_preregistration(gate, membership, matrix_info)
        print(
            "Phase 4 (preregistration freeze) OK. preregistration_hash =",
            preregistration["fix8_official_oof_calibration_preregistration_hash"],
        )

        phase = "PHASE5_OFFICIAL_OOF"
        oof_result = phase5_official_oof(preregistration, matrix_info)

        phase = "PHASE6_MARKET_RECONSTRUCTION"
        market_result = phase6_market_reconstruction(preregistration, matrix_info)

        phase = "PHASE7_PRICING"
        priced = phase7_pricing(preregistration, oof_result, market_result)

        phase = "PHASE8_CALIBRATION"
        ledgers = phase8_calibration(preregistration, priced)

        phase = "PHASE9_READINESS_AND_2025_AUDIT"
        readiness_and_audit = phase9_readiness_and_2025_audit(preregistration, ledgers, priced)

        phase = "PHASE10_PRODUCTION_SEED"
        seeds = phase10_production_seed(preregistration, priced)

        phase = "PHASE11_PERSIST_EVIDENCE"
        written_evidence = phase11_persist_evidence(oof_result, market_result, priced, ledgers, seeds)

        phase = "PHASE12_FINAL_FREEZE_AND_EVIDENCE"
        summary = phase12_final_freeze_and_evidence(
            preregistration, gate, membership, oof_result, market_result, priced, ledgers,
            readiness_and_audit, seeds, written_evidence,
        )
        print("Phase 12 (final freeze + full pytest + git diff --check) OK.")

        phase = "PHASE13_MIKE_HANDOFF"
        _write_mike_handoff(summary, preregistration)
        print("Phase 13 (docs/MIKE_HANDOFF.md + status pointer) OK.")

        print(json.dumps(
            {
                "status": "FIX 8 COMPLETE",
                "preregistration_hash": preregistration["fix8_official_oof_calibration_preregistration_hash"],
                "membership_counts": summary["membership_counts"],
                "fit_counts": summary["fit_counts"],
                "all_four_streams_ready_by_end_2025": summary["all_four_streams_ready_by_end_2025"],
            },
            indent=2,
        ))
        print("FIX 8 COMPLETE — OFFICIAL TUE/FRI OOF + ATS/TOTAL CALIBRATION CERTIFIED FOR 2026")
        return 0

    except fd.HardGateFailure as gate_failure:
        _write_failure_evidence("HARD_GATE_FAILURE", phase, {
            "gate_name": str(gate_failure),
            "preregistration_hash": preregistration.get("fix8_official_oof_calibration_preregistration_hash"),
        })
        print("FIX 8 INCOMPLETE — DO NOT PACKAGE OR SEND FOR REVIEW")
        return 1

    except Exception as exc:
        _write_failure_evidence("UNEXPECTED_EXECUTION_ERROR", phase, {
            "exception_type": type(exc).__qualname__, "exception_message": str(exc),
            "preregistration_hash": preregistration.get("fix8_official_oof_calibration_preregistration_hash"),
        })
        print("FIX 8 INCOMPLETE — DO NOT PACKAGE OR SEND FOR REVIEW")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
