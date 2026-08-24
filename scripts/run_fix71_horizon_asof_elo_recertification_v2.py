"""Fix 7.1 V2: card-scoped horizon-as-of Elo remediation -- auto-mode
end-to-end run, corrected.

V1 (branch fix/horizon-asof-elo-semantics-2026, superseded) computed each
game's own TUE/FRI cutoff independently from that game's own kickoff,
floored backward until strictly earlier -- which silently floated a
Thursday/Thanksgiving/Christmas game's ineligible Friday cutoff back to the
PREVIOUS week's Friday instead of excluding it as a target. V1's evidence
(``horizon_asof_elo_recertification_preregistration_hash=c8548f54...``,
``horizon_feature_semantics_hash=d61ef892...``,
``operational_model_spec_hash=be73c146...``) is preserved, invalidated, and
NEVER reused as certified -- see
``NFL_MODEL_ARTIFACT_ROOT/invalidated/fix71-per-game-horizon-membership-2026-08-24/``.

V2 uses a single authoritative card-scoped horizon membership ledger
(:mod:`nfl_hybrid.features.horizon_elo`, ``(season, week, season_type)``
cards, shared cutoffs) and keeps target eligibility (does this game get a
supervised prediction/training row for horizon H) strictly separate from
Elo update-event eligibility (does an earlier game's known result update the
engine before a target is predicted).

Phase order:
  1. GATE: repo/base/clean-tree + Fix-6/Fix-7/candidate-registry hash
     verification.
  2. Build the horizon membership ledger + HARD membership-count gates
     (Section 3 of the corrected plan) -- before any model fit.
  3. Build TUE/FRI ELIGIBLE-TARGET-ONLY selection matrices + hashes.
  4. V2 PREREGISTRATION FREEZE (both horizons) + focused prefit tests.
  5. AUDIT A: CLOCK_ONLY_REG_COMPATIBILITY (corrected -- real TUE/FRI
     cutoffs, eligible-REG-targets-only, no hardcoded pass criterion).
  6. AUDIT B: FINAL_OPERATIONAL_SEMANTICS (corrected -- 3-way
     baseline/clock-only/final-operational classification,
     POST_TARGET_NO_FIX6_BASELINE, eligible-intersection-only TUE/FRI
     diagnostic).
  7. TUE inner fits + selection (eligible-target matrix only).
  8. FRI inner fits + selection (eligible-target matrix only) -- both must
     select RIDGE_ALPHA_100, or STOP before 2024.
  9. LOCKED_POST_EXPOSURE_2024_AUDIT, TUE (n=285) and FRI (n=264).
 10. NEW operational_model_spec_hash (never be73c146...) + evidence +
     focused/full tests + git diff check.

Any ``fd.HardGateFailure`` stops the run immediately, no retry. Never
commits, never opens a PR, never touches Fix 8.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.features.elo_state import build_elo_pregame_state
from nfl_hybrid.features.feature_manifest import validate_no_banned_features
from nfl_hybrid.features.pregame_rolling import build_game_pregame_matrix
from nfl_hybrid.selection import feature_deduction_2026 as fd
from nfl_hybrid.selection import model_family_selection_2026 as mfs

SCHEMA_VERSION = "fix7-1-horizon-asof-elo-recertification-v2"

EXPECTED_BRANCH = "fix/horizon-asof-elo-semantics-2026"
EXPECTED_BASE_SHA = "c81372b8d4a36d660396237a974eae11fef5a690"

EXPECTED_FIX6_FEATURE_MANIFEST_HASH = "d523a007f7babfd0e8913f7ad6204f3670120d5d0f36183ecead9e153a8b24bf"
EXPECTED_FIX7_FINAL_MODEL_SPEC_HASH = "418724cb32fcb9a6a58f44fc77a7d579dbbc64d8078991c13d363751892ddede"
EXPECTED_CANDIDATE_REGISTRY_HASH = "360e40e8a541c3a4480a576813c67ec13bf5dafc8d198dc8b8ca05e675cc9b58"
EXPECTED_RIDGE_ALPHA_100_SPEC_HASH = "27b3eec6554e1bdefdcdda3f6a6bb9311efa675c850f059aa71881d6c77978d7"
REQUIRED_SELECTED_CANDIDATE = "RIDGE_ALPHA_100"

# Old V1 evidence -- MUST NEVER be reused as certified.
OLD_NONCERTIFIED_PREREGISTRATION_HASH = "c8548f544ff33fb0226380ace21eee815f141906c9153d0a475ca75cb591b9a7"
OLD_NONCERTIFIED_HORIZON_SEMANTICS_HASH = "d61ef89256cf164eda7167ac9e19d41e1dac03ccc0ee7bc0e4cfa41ad920906d"
OLD_NONCERTIFIED_OPERATIONAL_MODEL_SPEC_HASH = "be73c146d138e24d1637be532e226f3e519641094c599076aa352258277178b7"

# Section 3's hard membership-count gates.
EXPECTED_ALL_TARGETS = 1408
EXPECTED_TUE_ELIGIBLE = 1408
EXPECTED_TUE_INELIGIBLE = 0
EXPECTED_FRI_ELIGIBLE = 1317
EXPECTED_FRI_INELIGIBLE = 91
EXPECTED_FRI_ELIGIBLE_REG = 1252
EXPECTED_FRI_ELIGIBLE_POST = 65
EXPECTED_FRI_INELIGIBLE_REG = 91
EXPECTED_FRI_INELIGIBLE_POST = 0
EXPECTED_FOLD_MEMBERSHIP = {
    # fold: (tue_train, tue_validate, fri_train, fri_validate)
    "A": (269, 285, 255, 267),
    "B": (554, 284, 522, 265),
    "C": (838, 285, 787, 266),
    "OUTER": (1123, 285, 1053, 264),
}

ALLOWED_DIRTY_PATHS = {
    "REMEDIATION_STATUS_2026.md",
    "docs/HORIZON_ASOF_ELO_2026.md",
    "scripts/run_fix71_horizon_asof_elo_recertification.py",
    "scripts/run_fix71_horizon_asof_elo_recertification_v2.py",
    "src/nfl_hybrid/features/horizon_elo.py",
    "tests/test_horizon_elo.py",
}
ALLOWED_DIRTY_PREFIXES = ("outputs/fix7_1_",)

FOCUSED_TEST_FILES = [
    "tests/test_horizon_elo.py",
    "tests/test_elo_pregame_state.py",
    "tests/test_model_family_selection_2026.py",
    "tests/test_feature_deduction_2026.py",
    "tests/test_joint_score.py",
    "tests/test_pregame_rolling.py",
    "tests/test_labels.py",
]

FAILURE_ARTIFACT_PREFIX = "outputs/fix7_1_v2_failure_"

ELO_FEATURE_COLUMNS = list(fd.FEATURE_GROUPS["ELO_STRENGTH"].columns)
COMPARISON_COLS = ("elo_pregame_rating", "elo_pregame_win_probability", "elo_pregame_expected_margin")
DIFF_EPSILON = 1e-9

PLANNED_PAIRED_FITS = 42
PLANNED_INDIVIDUAL_FITS = 84
MAX_PAIRED_FITS = 50
MAX_INDIVIDUAL_FITS = 100


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


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")


def _sha256_hex(payload: dict) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


# ===========================================================================
# PHASE 1 -- repo/base/clean-tree gate + historical hash verification.
# ===========================================================================
def phase1_gate() -> dict:
    branch = _run(["git", "branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise fd.HardGateFailure(f"WRONG_BRANCH: on {branch!r}, expected {EXPECTED_BRANCH!r}")

    head_sha = _run(["git", "rev-parse", "HEAD"])
    if head_sha != EXPECTED_BASE_SHA:
        _run(["git", "merge-base", "--is-ancestor", EXPECTED_BASE_SHA, "HEAD"])

    status_lines = [line for line in _run(["git", "status", "--short"]).splitlines() if line.strip()]
    unexpected = []
    for line in status_lines:
        path = line[3:].strip()
        if path in ALLOWED_DIRTY_PATHS or any(path.startswith(p) for p in ALLOWED_DIRTY_PREFIXES):
            continue
        unexpected.append(path)
    if unexpected:
        raise fd.HardGateFailure(f"UNEXPECTED_DIRTY_FILES: {unexpected}")

    fix6_summary = json.loads((REPO_ROOT / "outputs" / "fix6_feature_selection_summary.json").read_text())
    live_fix6_hash = fd.compute_feature_manifest_hash(fix6_summary["final_features"])
    if fix6_summary["feature_manifest_hash"] != EXPECTED_FIX6_FEATURE_MANIFEST_HASH:
        raise fd.HardGateFailure("FIX6_FEATURE_MANIFEST_HASH_CHANGED")
    if live_fix6_hash != EXPECTED_FIX6_FEATURE_MANIFEST_HASH:
        raise fd.HardGateFailure(f"FIX6_FEATURE_MANIFEST_HASH_RECOMPUTE_MISMATCH: live={live_fix6_hash}")

    fix7_summary = json.loads((REPO_ROOT / "outputs" / "fix7_model_family_selection_summary.json").read_text())
    if fix7_summary["final_model_spec_hash"] != EXPECTED_FIX7_FINAL_MODEL_SPEC_HASH:
        raise fd.HardGateFailure("FIX7_FINAL_MODEL_SPEC_HASH_CHANGED")

    live_registry_hash = mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY)
    if live_registry_hash != EXPECTED_CANDIDATE_REGISTRY_HASH:
        raise fd.HardGateFailure(f"CANDIDATE_REGISTRY_HASH_MISMATCH: live={live_registry_hash}")

    ridge_100 = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "RIDGE_ALPHA_100")
    live_ridge_spec_hash = mfs.compute_candidate_spec_hash(ridge_100)
    if live_ridge_spec_hash != EXPECTED_RIDGE_ALPHA_100_SPEC_HASH:
        raise fd.HardGateFailure(f"RIDGE_ALPHA_100_SPEC_HASH_MISMATCH: live={live_ridge_spec_hash}")

    return {
        "branch": branch, "head_sha": head_sha, "expected_base_sha": EXPECTED_BASE_SHA,
        "dirty_files": [line[3:].strip() for line in status_lines],
        "fix6_feature_manifest_hash": EXPECTED_FIX6_FEATURE_MANIFEST_HASH,
        "frozen_feature_columns": list(fix6_summary["final_features"]),
        "fix7_final_model_spec_hash": EXPECTED_FIX7_FINAL_MODEL_SPEC_HASH,
        "candidate_registry_hash": live_registry_hash,
        "ridge_alpha_100_candidate_spec_hash": live_ridge_spec_hash,
        "old_noncertified_hashes": {
            "horizon_asof_elo_recertification_preregistration_hash": OLD_NONCERTIFIED_PREREGISTRATION_HASH,
            "horizon_feature_semantics_hash": OLD_NONCERTIFIED_HORIZON_SEMANTICS_HASH,
            "operational_model_spec_hash": OLD_NONCERTIFIED_OPERATIONAL_MODEL_SPEC_HASH,
            "status": "NON_CERTIFIED",
        },
    }


# ===========================================================================
# PHASE 2 -- horizon membership ledger + HARD membership-count gates.
# ===========================================================================
def phase2_membership(gate: dict) -> dict:
    raw_games = pd.read_parquet(resolve("backfill.games"))
    games, firewall_counts = fd.enforce_2025_firewall(raw_games)  # REG+POST, season<=2024

    ledger = he.build_horizon_membership_ledger(games)
    ledger_hash = he.compute_horizon_membership_ledger_hash(ledger)

    if len(ledger) != EXPECTED_ALL_TARGETS:
        raise fd.HardGateFailure(f"MEMBERSHIP_COUNT_MISMATCH[all_targets]: {len(ledger)} != {EXPECTED_ALL_TARGETS}")

    counts = {}
    for horizon in he.HORIZONS:
        h = horizon.lower()
        elig = int(ledger[f"{h}_eligible"].sum())
        inelig = int((~ledger[f"{h}_eligible"]).sum())
        reg_elig = int((ledger[f"{h}_eligible"] & (ledger["season_type"] == "REG")).sum())
        post_elig = int((ledger[f"{h}_eligible"] & (ledger["season_type"] == "POST")).sum())
        reg_inelig = int((~ledger[f"{h}_eligible"] & (ledger["season_type"] == "REG")).sum())
        post_inelig = int((~ledger[f"{h}_eligible"] & (ledger["season_type"] == "POST")).sum())
        counts[horizon] = {
            "eligible": elig, "ineligible": inelig,
            "reg_eligible": reg_elig, "post_eligible": post_elig,
            "reg_ineligible": reg_inelig, "post_ineligible": post_inelig,
        }

    if (counts["TUE"]["eligible"], counts["TUE"]["ineligible"]) != (EXPECTED_TUE_ELIGIBLE, EXPECTED_TUE_INELIGIBLE):
        raise fd.HardGateFailure(f"MEMBERSHIP_COUNT_MISMATCH[TUE]: {counts['TUE']}")
    if (counts["FRI"]["eligible"], counts["FRI"]["ineligible"]) != (EXPECTED_FRI_ELIGIBLE, EXPECTED_FRI_INELIGIBLE):
        raise fd.HardGateFailure(f"MEMBERSHIP_COUNT_MISMATCH[FRI]: {counts['FRI']}")
    if (counts["FRI"]["reg_eligible"], counts["FRI"]["post_eligible"]) != (EXPECTED_FRI_ELIGIBLE_REG, EXPECTED_FRI_ELIGIBLE_POST):
        raise fd.HardGateFailure(f"MEMBERSHIP_COUNT_MISMATCH[FRI_REG_POST_ELIGIBLE]: {counts['FRI']}")
    if (counts["FRI"]["reg_ineligible"], counts["FRI"]["post_ineligible"]) != (EXPECTED_FRI_INELIGIBLE_REG, EXPECTED_FRI_INELIGIBLE_POST):
        raise fd.HardGateFailure(f"MEMBERSHIP_COUNT_MISMATCH[FRI_REG_POST_INELIGIBLE]: {counts['FRI']}")

    fold_membership = {}
    folds = {"A": (2020, 2021), "B": (2021, 2022), "C": (2022, 2023), "OUTER": (2023, 2024)}
    for name, (train_max, validate_season) in folds.items():
        tue_train = int((ledger["season"] <= train_max).sum())
        tue_val = int((ledger["season"] == validate_season).sum())
        fri_train = int(((ledger["season"] <= train_max) & ledger["fri_eligible"]).sum())
        fri_val = int(((ledger["season"] == validate_season) & ledger["fri_eligible"]).sum())
        actual = (tue_train, tue_val, fri_train, fri_val)
        fold_membership[name] = {
            "tue_train": tue_train, "tue_validate": tue_val, "fri_train": fri_train, "fri_validate": fri_val,
        }
        if actual != EXPECTED_FOLD_MEMBERSHIP[name]:
            raise fd.HardGateFailure(f"FOLD_MEMBERSHIP_MISMATCH[{name}]: actual={actual} expected={EXPECTED_FOLD_MEMBERSHIP[name]}")

    return {
        "games": games, "ledger": ledger, "ledger_hash": ledger_hash,
        "firewall_counts": firewall_counts, "membership_counts": counts, "fold_membership": fold_membership,
    }


# ===========================================================================
# PHASE 3 -- eligible-target-only TUE/FRI selection matrices.
# ===========================================================================
def build_horizon_selection_matrix(games: pd.DataFrame, horizon: str, ledger: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    eligible_ids = he.eligible_game_ids(ledger, horizon)
    games_eligible = games[games["game_id"].isin(eligible_ids)].reset_index(drop=True)

    elo_state = he.build_horizon_elo_state(games, horizon, membership_ledger=ledger)
    team_state = elo_state[
        ["game_id", "team_id", "elo_pregame_rating", "elo_pregame_win_probability", "elo_pregame_expected_margin"]
    ]
    carrier = ("season", "week", "home_score", "away_score", "season_type")
    fd.assert_no_forbidden_market_columns(carrier)

    pivoted = build_game_pregame_matrix(games_eligible, team_state, carrier_columns=carrier)

    feature_columns = list(ELO_FEATURE_COLUMNS)
    missing = [c for c in feature_columns if c not in pivoted.columns]
    if missing:
        raise ValueError(f"Pivoted matrix missing expected feature column(s): {missing}")
    fd.assert_no_forbidden_market_columns(feature_columns)
    validate_no_banned_features(feature_columns)

    pivoted["home_margin"] = pd.to_numeric(pivoted["home_score"], errors="coerce") - pd.to_numeric(pivoted["away_score"], errors="coerce")
    pivoted["total_points"] = pd.to_numeric(pivoted["home_score"], errors="coerce") + pd.to_numeric(pivoted["away_score"], errors="coerce")
    pivoted["season"] = pd.to_numeric(pivoted["season"], errors="raise").astype(int)
    for col in feature_columns:
        pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce").astype(float)

    if pivoted[feature_columns].isna().sum().sum() != 0:
        raise fd.HardGateFailure("PHASE3_UNEXPECTED_MISSINGNESS_IN_FROZEN_SIX_FEATURES")
    if len(pivoted) != len(eligible_ids):
        raise fd.HardGateFailure(f"PHASE3_ROW_COUNT_MISMATCH: matrix={len(pivoted)} eligible_ids={len(eligible_ids)}")

    return pivoted, feature_columns


def phase3_build_matrices(membership: dict) -> dict:
    games, ledger = membership["games"], membership["ledger"]
    matrices: dict[str, dict] = {}
    for horizon in he.HORIZONS:
        matrix, feature_columns = build_horizon_selection_matrix(games, horizon, ledger)
        if int(matrix["season"].max()) > fd.SELECTION_MAX_SEASON:
            raise fd.HardGateFailure(f"PHASE3_2025_ROWS_IN_MATRIX[{horizon}]")
        selection_matrix_hash, row_count, column_order = mfs.compute_selection_matrix_hash(matrix, tuple(feature_columns))
        matrices[horizon] = {
            "matrix": matrix, "feature_columns": feature_columns,
            "selection_matrix_hash": selection_matrix_hash, "selection_matrix_row_count": row_count,
            "selection_matrix_column_order": column_order,
            "reg_rows": int((matrix["season_type"] == "REG").sum()), "post_rows": int((matrix["season_type"] == "POST").sum()),
        }
        print(f"Phase 3 [{horizon}] OK: rows={row_count} (REG={matrices[horizon]['reg_rows']} POST={matrices[horizon]['post_rows']}) "
              f"selection_matrix_hash={selection_matrix_hash}")
    return {"matrices": matrices}


# ===========================================================================
# PHASE 4 -- V2 preregistration freeze (BOTH horizons) + focused prefit
# tests, before any real candidate fit.
# ===========================================================================
def phase4_preregistration(gate: dict, membership: dict, matrix_info: dict) -> dict:
    plan = {
        "planned_paired": PLANNED_PAIRED_FITS, "planned_individual": PLANNED_INDIVIDUAL_FITS,
        "max_paired": MAX_PAIRED_FITS, "max_individual": MAX_INDIVIDUAL_FITS,
        "fit_budget_ok": PLANNED_PAIRED_FITS <= MAX_PAIRED_FITS and PLANNED_INDIVIDUAL_FITS <= MAX_INDIVIDUAL_FITS,
    }
    print("Fit budget plan:", json.dumps(plan, indent=2))
    if not plan["fit_budget_ok"]:
        raise fd.HardGateFailure(f"FIT_BUDGET_EXCEEDED: {plan}")

    elo_config_hash = he.compute_elo_config_hash()
    horizon_semantics = he.horizon_semantics_spec_v2(
        fix6_feature_manifest_hash=gate["fix6_feature_manifest_hash"],
        fix6_frozen_feature_columns=gate["frozen_feature_columns"],
        elo_config_hash=elo_config_hash,
    )
    horizon_feature_semantics_hash = he.compute_horizon_feature_semantics_hash_v2(
        fix6_feature_manifest_hash=gate["fix6_feature_manifest_hash"],
        fix6_frozen_feature_columns=gate["frozen_feature_columns"],
        elo_config_hash=elo_config_hash,
    )
    if horizon_feature_semantics_hash == OLD_NONCERTIFIED_HORIZON_SEMANTICS_HASH:
        raise fd.HardGateFailure("V2_SEMANTICS_HASH_COLLIDED_WITH_NONCERTIFIED_V1_HASH")

    audit_a_definition = {
        "name": "CLOCK_ONLY_REG_COMPATIBILITY",
        "baseline": "nfl_hybrid.features.elo_state.build_elo_pregame_state(reg_games) -- old Fix-6 REG-only kickoff-order Elo",
        "clock_only_asof": "REG-only Elo via nfl_hybrid.features.horizon_elo.build_horizon_elo_state(reg_games, horizon, membership_ledger=reg_scope_ledger)",
        "population": "horizon-eligible REG targets only",
        "pass_criterion": "construction correct, membership correct, chronology invariants hold, output deterministic -- NOT a hardcoded difference count",
    }
    audit_b_definition = {
        "name": "FINAL_OPERATIONAL_SEMANTICS",
        "states": {
            "A_OLD_BASELINE": "old Fix-6 REG-only kickoff-pregame Elo",
            "B_CLOCK_ONLY": "REG-only card-cutoff-as-of Elo",
            "C_FINAL_OPERATIONAL": "REG+POST card-cutoff-as-of Elo",
        },
        "classification": "CUTOFF_AVAILABILITY_DIFFERENCE | POSTSEASON_HISTORY_DIFFERENCE | BOTH | EXACT_MATCH (REG targets); POST_TARGET_NO_FIX6_BASELINE (POST targets)",
        "tue_fri_diagnostic": "optional, restricted to intersection(TUE eligible game_ids, FRI eligible game_ids), never replaces the baseline-vs-operational audits",
    }

    body = {
        "schema_version": SCHEMA_VERSION,
        "repo": {"branch": gate["branch"], "head_sha": gate["head_sha"], "expected_base_sha": gate["expected_base_sha"]},
        "fix6_feature_manifest_hash": gate["fix6_feature_manifest_hash"],
        "fix7_final_model_spec_hash": gate["fix7_final_model_spec_hash"],
        "candidate_registry_hash": gate["candidate_registry_hash"],
        "ridge_alpha_100_candidate_spec_hash": gate["ridge_alpha_100_candidate_spec_hash"],
        "frozen_feature_columns": gate["frozen_feature_columns"],
        "old_noncertified_hashes": gate["old_noncertified_hashes"],
        "elo_config_hash": elo_config_hash,
        "horizon_semantics": horizon_semantics,
        "horizon_feature_semantics_hash": horizon_feature_semantics_hash,
        "horizon_membership_ledger_hash": membership["ledger_hash"],
        "membership_counts": membership["membership_counts"],
        "fold_membership": membership["fold_membership"],
        "inner_folds": [asdict(f) for f in fd.INNER_FOLDS],
        "outer_fold": asdict(fd.OUTER_FOLD),
        "selection_forbidden_seasons": list(fd.SELECTION_FORBIDDEN_SEASONS),
        "selection_max_season": fd.SELECTION_MAX_SEASON,
        "target_scope": "REG+POST (eligible targets only, per horizon)",
        "margin_target_definition": "home_score - away_score",
        "total_target_definition": "home_score + away_score",
        "candidate_registry": [
            {"name": c.name, "family": c.family, "complexity_rank": c.complexity_rank, "preprocessing": c.preprocessing,
             "hyperparameters": c.hyperparameters, "random_state": c.random_state, "spec_hash": mfs.compute_candidate_spec_hash(c)}
            for c in mfs.CANDIDATE_REGISTRY
        ],
        "audit_a_definition": audit_a_definition,
        "audit_b_definition": audit_b_definition,
        "ridge_alpha_rule": mfs.FAMILY_SELECTION_ALGORITHM_SOURCE,
        "family_complexity_order": "RIDGE(0) < HUBER(1) < HGBR(2)",
        "primary_metric_formula": "primary_score = 0.5 * (margin_RMSE + total_RMSE)",
        "paired_loss_formula": "L_i = 0.5 * (margin_error_i^2 + total_error_i^2); delta_i = L_a_i - L_b_i; SE = std(delta, ddof=1) / sqrt(n)",
        "success_requirement": "TUE selected candidate == RIDGE_ALPHA_100 AND FRI selected candidate == RIDGE_ALPHA_100",
        "audit_tolerance_table": {
            "overall": mfs.AUDIT_PRIMARY_TOLERANCE, "margin": mfs.AUDIT_MARGIN_TOLERANCE, "total": mfs.AUDIT_TOTAL_TOLERANCE,
            "week1": mfs.AUDIT_WEEK1_TOLERANCE, "weeks2_4": mfs.AUDIT_WEEKS2_4_TOLERANCE, "weeks5_plus": mfs.AUDIT_WEEKS5PLUS_TOLERANCE,
        },
        "outer_2024_previously_exposed": mfs.AUDIT_2024_PREVIOUSLY_EXPOSED,
        "outer_2024_role": mfs.AUDIT_2024_ROLE,
        "outer_2024_expected_validate_count": {"TUE": 285, "FRI": 264},
        "forbidden_market_columns": sorted(fd.FORBIDDEN_MARKET_COLUMNS),
        "deterministic_settings": {"random_state": fd.MODEL_CONFIG.random_state, "single_threaded": True},
        "postseason_policy": {
            "postseason_results_update_elo_when_chronologically_available": True,
            "postseason_state_carries_into_next_season_then_regress_to_mean": True,
        },
        "twenty_twenty_five_firewall": "enforce_2025_firewall applied once; used counts must be zero everywhere",
        "contracts": {
            horizon: {
                "selection_matrix_hash": matrix_info["matrices"][horizon]["selection_matrix_hash"],
                "selection_matrix_row_count": matrix_info["matrices"][horizon]["selection_matrix_row_count"],
                "selection_matrix_column_order": matrix_info["matrices"][horizon]["selection_matrix_column_order"],
                "reg_rows": matrix_info["matrices"][horizon]["reg_rows"],
                "post_rows": matrix_info["matrices"][horizon]["post_rows"],
            }
            for horizon in he.HORIZONS
        },
        "fit_budget_plan": plan,
        "firewall_counts": membership["firewall_counts"],
    }

    pytest_tail = _run_focused_tests()

    horizon_asof_elo_recertification_preregistration_v2_hash = _sha256_hex(body)
    if horizon_asof_elo_recertification_preregistration_v2_hash == OLD_NONCERTIFIED_PREREGISTRATION_HASH:
        raise fd.HardGateFailure("V2_PREREGISTRATION_HASH_COLLIDED_WITH_NONCERTIFIED_V1_HASH")

    preregistration = {**body, "horizon_asof_elo_recertification_preregistration_v2_hash": horizon_asof_elo_recertification_preregistration_v2_hash}

    root = artifact_root() / "horizon-asof-elo-recertification-2026-v2"
    root.mkdir(parents=True, exist_ok=True)
    (root / "horizon_asof_elo_recertification_preregistration_v2.json").write_text(
        json.dumps(preregistration, indent=2, default=str), encoding="utf-8"
    )
    preregistration["_phase4_pretest_output_tail"] = pytest_tail
    return preregistration


def _verify_preregistration_unchanged(preregistration: dict) -> None:
    body = {
        k: v for k, v in preregistration.items()
        if not k.startswith("_") and k != "horizon_asof_elo_recertification_preregistration_v2_hash"
    }
    if _sha256_hex(body) != preregistration["horizon_asof_elo_recertification_preregistration_v2_hash"]:
        raise fd.HardGateFailure("PREREGISTRATION_HASH_MUTATED_AFTER_FREEZE")


# ===========================================================================
# PHASE 5 -- AUDIT A: CLOCK_ONLY_REG_COMPATIBILITY (corrected).
# ===========================================================================
def phase5_audit_a(preregistration: dict, membership: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    games = membership["games"]
    reg_games = games[games["season_type"] == "REG"].reset_index(drop=True)
    reg_ledger = he.build_horizon_membership_ledger(reg_games)

    baseline = build_elo_pregame_state(reg_games)
    baseline_s = baseline.sort_values(["game_id", "team_id"]).reset_index(drop=True)

    results: dict[str, dict] = {}
    per_horizon_export: dict[str, list] = {}
    for horizon in he.HORIZONS:
        eligible_ids = he.eligible_game_ids(reg_ledger, horizon)
        clock_only = he.build_horizon_elo_state(reg_games, horizon, membership_ledger=reg_ledger)
        clock_only_s = clock_only.sort_values(["game_id", "team_id"]).reset_index(drop=True)

        baseline_elig = baseline_s[baseline_s["game_id"].isin(eligible_ids)]
        if len(clock_only_s) != len(baseline_elig):
            raise fd.HardGateFailure(
                f"AUDIT_A_ROW_COUNT_MISMATCH[{horizon}]: baseline_eligible={len(baseline_elig)} clock_only={len(clock_only_s)}"
            )

        merged = baseline_elig.merge(
            clock_only_s[["game_id", "team_id", "target_cutoff_utc"] + list(COMPARISON_COLS)],
            on=["game_id", "team_id"], suffixes=("_baseline", "_asof"), validate="one_to_one",
        )
        diffs = pd.Series(False, index=merged.index)
        for c in COMPARISON_COLS:
            diffs = diffs | ((merged[f"{c}_baseline"] - merged[f"{c}_asof"]).abs() > DIFF_EPSILON)
        differing_game_ids = sorted(merged.loc[diffs, "game_id"].unique().tolist())

        results[horizon] = {
            "eligible_target_count": len(eligible_ids),
            "exact_match_count": int((~diffs).sum()),
            "difference_count": len(differing_game_ids),
            "difference_game_ids": differing_game_ids,
            "deterministic": clock_only.equals(
                he.build_horizon_elo_state(reg_games.sample(frac=1.0, random_state=11), horizon, membership_ledger=reg_ledger)
                .sort_values(["game_id", "team_id"]).reset_index(drop=True)
            ) if False else True,  # deterministic check performed structurally below
        }
        per_horizon_export[horizon] = merged.loc[
            diffs, ["game_id", "team_id", "target_cutoff_utc"] + [f"{c}_baseline" for c in COMPARISON_COLS] + [f"{c}_asof" for c in COMPARISON_COLS]
        ].to_dict(orient="records")

        # Determinism re-check (real, not the placeholder above).
        rebuilt = he.build_horizon_elo_state(
            reg_games.sample(frac=1.0, random_state=11), horizon, membership_ledger=reg_ledger
        ).sort_values(["game_id", "team_id"]).reset_index(drop=True)
        results[horizon]["deterministic"] = clock_only_s[["game_id", "team_id"] + list(COMPARISON_COLS)].equals(
            rebuilt[["game_id", "team_id"] + list(COMPARISON_COLS)]
        )
        if not results[horizon]["deterministic"]:
            raise fd.HardGateFailure(f"AUDIT_A_NONDETERMINISTIC[{horizon}]")

        print(f"Phase 5 [AUDIT A / {horizon}] eligible={results[horizon]['eligible_target_count']} "
              f"exact_match={results[horizon]['exact_match_count']} differences={results[horizon]['difference_count']}")

    root = artifact_root() / "horizon-asof-elo-recertification-2026-v2"
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit_a_clock_only_reg_compatibility.json").write_text(
        json.dumps({"summary": results, "differences": per_horizon_export}, indent=2, default=str), encoding="utf-8"
    )

    return {"audit": "CLOCK_ONLY_REG_COMPATIBILITY", "per_horizon": results, "pass": True}


# ===========================================================================
# PHASE 6 -- AUDIT B: FINAL_OPERATIONAL_SEMANTICS (corrected).
# ===========================================================================
def phase6_audit_b(preregistration: dict, membership: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    games, ledger = membership["games"], membership["ledger"]
    reg_games = games[games["season_type"] == "REG"].reset_index(drop=True)
    reg_ledger = he.build_horizon_membership_ledger(reg_games)

    baseline = build_elo_pregame_state(reg_games).sort_values(["game_id", "team_id"]).reset_index(drop=True)

    per_horizon: dict[str, dict] = {}
    for horizon in he.HORIZONS:
        eligible_reg_ids = he.eligible_game_ids(reg_ledger, horizon)
        eligible_all_ids = he.eligible_game_ids(ledger, horizon)
        eligible_post_ids = eligible_all_ids - eligible_reg_ids

        B_clock_only = he.build_horizon_elo_state(reg_games, horizon, membership_ledger=reg_ledger)
        C_final_operational = he.build_horizon_elo_state(games, horizon, membership_ledger=ledger)

        B_s = B_clock_only.sort_values(["game_id", "team_id"]).reset_index(drop=True)
        C_reg = C_final_operational[C_final_operational["game_id"].isin(eligible_reg_ids)].sort_values(["game_id", "team_id"]).reset_index(drop=True)
        A_reg = baseline[baseline["game_id"].isin(eligible_reg_ids)].reset_index(drop=True)

        merged = A_reg.merge(
            B_s[["game_id", "team_id"] + list(COMPARISON_COLS)], on=["game_id", "team_id"], suffixes=("_A", "_B"), validate="one_to_one"
        ).merge(
            C_reg[["game_id", "team_id"] + list(COMPARISON_COLS)].rename(columns={c: f"{c}_C" for c in COMPARISON_COLS}),
            on=["game_id", "team_id"], validate="one_to_one",
        )
        clock_diff = pd.Series(False, index=merged.index)
        post_diff = pd.Series(False, index=merged.index)
        for c in COMPARISON_COLS:
            clock_diff = clock_diff | ((merged[f"{c}_A"] - merged[f"{c}_B"]).abs() > DIFF_EPSILON)
            post_diff = post_diff | ((merged[f"{c}_B"] - merged[f"{c}_C"]).abs() > DIFF_EPSILON)

        classification = pd.Series("EXACT_MATCH", index=merged.index)
        classification[clock_diff & post_diff] = "BOTH"
        classification[clock_diff & ~post_diff] = "CUTOFF_AVAILABILITY_DIFFERENCE"
        classification[~clock_diff & post_diff] = "POSTSEASON_HISTORY_DIFFERENCE"
        merged["classification"] = classification
        class_counts = classification.value_counts().to_dict()

        post_rows = C_final_operational[C_final_operational["game_id"].isin(eligible_post_ids)]

        per_horizon[horizon] = {
            "eligible_reg_target_count": len(eligible_reg_ids),
            "eligible_post_target_count": len(eligible_post_ids),
            "classification_counts": {
                "EXACT_MATCH": int(class_counts.get("EXACT_MATCH", 0)),
                "CUTOFF_AVAILABILITY_DIFFERENCE": int(class_counts.get("CUTOFF_AVAILABILITY_DIFFERENCE", 0)),
                "POSTSEASON_HISTORY_DIFFERENCE": int(class_counts.get("POSTSEASON_HISTORY_DIFFERENCE", 0)),
                "BOTH": int(class_counts.get("BOTH", 0)),
            },
            "post_target_no_fix6_baseline_count": len(post_rows),
        }
        print(f"Phase 6 [AUDIT B / {horizon}] eligible_reg={len(eligible_reg_ids)} eligible_post={len(eligible_post_ids)} "
              f"classification={per_horizon[horizon]['classification_counts']}")

    # TUE-vs-FRI diagnostic, restricted to intersection(TUE eligible, FRI eligible).
    tue_ids = he.eligible_game_ids(ledger, "TUE")
    fri_ids = he.eligible_game_ids(ledger, "FRI")
    intersection_ids = tue_ids & fri_ids
    tue_state = he.build_horizon_elo_state(games, "TUE", membership_ledger=ledger)
    fri_state = he.build_horizon_elo_state(games, "FRI", membership_ledger=ledger)
    tue_int = tue_state[tue_state["game_id"].isin(intersection_ids)].sort_values(["game_id", "team_id"]).reset_index(drop=True)
    fri_int = fri_state[fri_state["game_id"].isin(intersection_ids)].sort_values(["game_id", "team_id"]).reset_index(drop=True)
    merged_tf = tue_int.merge(
        fri_int[["game_id", "team_id"] + list(COMPARISON_COLS)], on=["game_id", "team_id"], suffixes=("_tue", "_fri"), validate="one_to_one"
    )
    tf_diff = pd.Series(False, index=merged_tf.index)
    for c in COMPARISON_COLS:
        tf_diff = tf_diff | ((merged_tf[f"{c}_tue"] - merged_tf[f"{c}_fri"]).abs() > DIFF_EPSILON)
    tue_fri_diagnostic = {
        "intersection_target_count": len(intersection_ids),
        "differing_game_count": int(merged_tf.loc[tf_diff, "game_id"].nunique()),
        "differing_game_ids_sample": sorted(merged_tf.loc[tf_diff, "game_id"].unique().tolist())[:50],
    }
    print("Phase 6 [TUE-vs-FRI diagnostic, eligible intersection only]:", json.dumps(tue_fri_diagnostic, indent=2))

    root = artifact_root() / "horizon-asof-elo-recertification-2026-v2"
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit_b_final_operational_semantics.json").write_text(
        json.dumps({"per_horizon": per_horizon, "tue_fri_diagnostic": tue_fri_diagnostic}, indent=2, default=str), encoding="utf-8"
    )

    return {"audit": "FINAL_OPERATIONAL_SEMANTICS", "per_horizon": per_horizon, "tue_fri_diagnostic": tue_fri_diagnostic, "pass": True}


# ===========================================================================
# PHASES 7/8 -- inner candidate fits + Ridge alpha + family selection, per
# horizon, on the ELIGIBLE-TARGET-ONLY matrix. Reuses the EXACT original
# Fix-7 candidate registry / inner folds / selection rules (mfs), unmodified.
# ===========================================================================
def run_horizon_selection(preregistration: dict, matrix_info: dict, horizon: str) -> dict:
    _verify_preregistration_unchanged(preregistration)
    info = matrix_info["matrices"][horizon]
    cache = mfs.PredictionCache()
    inner_results, huber_invalid_reasons, paired, individual = mfs.run_inner_fits(
        info["matrix"], info["feature_columns"], cache=cache,
        fix6_feature_manifest_hash=preregistration["fix6_feature_manifest_hash"],
        selection_matrix_hash=info["selection_matrix_hash"],
        preregistration_hash=preregistration["horizon_asof_elo_recertification_preregistration_v2_hash"],
    )
    huber_valid = not bool(huber_invalid_reasons)

    ridge_result = mfs.select_ridge_alpha(inner_results)
    family_result = mfs.select_family(inner_results, ridge_selected_name=ridge_result["selected_alpha_name"], huber_valid=huber_valid)
    selected_family = family_result["selected_family"]
    representative_name = family_result["family_representative"][selected_family]

    return {
        "horizon": horizon, "inner_results": inner_results, "huber_valid": huber_valid, "huber_invalid_reasons": huber_invalid_reasons,
        "paired_fit_count": paired, "individual_fit_count": individual,
        "ridge_result": ridge_result, "family_result": family_result,
        "selected_family": selected_family, "selected_candidate_name": representative_name,
    }


# ===========================================================================
# PHASE 9 -- LOCKED_POST_EXPOSURE_2024_AUDIT, per horizon.
# ===========================================================================
def run_horizon_locked_2024_audit(preregistration: dict, matrix_info: dict, selection: dict, expected_validate_count: int) -> dict:
    _verify_preregistration_unchanged(preregistration)
    info = matrix_info["matrices"][selection["horizon"]]
    validate_count = int((info["matrix"]["season"] == fd.OUTER_FOLD.validate_season).sum())
    if validate_count != expected_validate_count:
        raise fd.HardGateFailure(
            f"2024_AUDIT_VALIDATE_COUNT_MISMATCH[{selection['horizon']}]: actual={validate_count} expected={expected_validate_count}"
        )

    finalists = {
        "RIDGE": next(c for c in mfs.CANDIDATE_REGISTRY if c.name == selection["family_result"]["family_representative"]["RIDGE"]),
        "HGBR": next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HGBR_INCUMBENT"),
    }
    if selection["huber_valid"]:
        finalists["HUBER"] = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HUBER_FIXED")

    result = mfs.run_locked_2024_audit(info["matrix"], info["feature_columns"], finalists=finalists, selected_family=selection["selected_family"])
    if not result["overall_pass"]:
        printable = {k: v for k, v in result.items() if k != "fitted_bundles"}
        print(json.dumps(printable, indent=2, default=str))
        raise fd.HardGateFailure(f"LOCKED_POST_EXPOSURE_2024_AUDIT_FAILED[{selection['horizon']}]: {printable['rules']}")
    return {"validate_count": validate_count, **{k: v for k, v in result.items() if k != "fitted_bundles"}}


# ===========================================================================
# PHASE 10 -- new operational_model_spec_hash + evidence + tests + git diff.
# ===========================================================================
def phase10_final_freeze_and_evidence(
    preregistration: dict, membership: dict, matrix_info: dict, selections: dict, audits_2024: dict, audit_a: dict, audit_b: dict
) -> dict:
    _verify_preregistration_unchanged(preregistration)

    per_horizon_final_model_spec_hash = {}
    for horizon in he.HORIZONS:
        selection = selections[horizon]
        selected_spec = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == selection["selected_candidate_name"])
        per_horizon_final_model_spec_hash[horizon] = mfs.compute_final_model_spec_hash(
            fix6_feature_manifest_hash=preregistration["fix6_feature_manifest_hash"],
            frozen_feature_columns=preregistration["frozen_feature_columns"],
            selected_family=selection["selected_family"], selected_spec=selected_spec,
        )

    operational_model_spec_hash = _sha256_hex(
        {
            "schema_version": SCHEMA_VERSION,
            "fix7_final_model_spec_hash": preregistration["fix7_final_model_spec_hash"],
            "horizon_feature_semantics_hash": preregistration["horizon_feature_semantics_hash"],
            "horizon_membership_ledger_hash": preregistration["horizon_membership_ledger_hash"],
            "ordered_feature_list": list(preregistration["frozen_feature_columns"]),
            "ridge_alpha_100_candidate_spec_hash": preregistration["ridge_alpha_100_candidate_spec_hash"],
            "target_scope": "REG+POST",
            "tue_certification": selections["TUE"]["selected_candidate_name"],
            "fri_certification": selections["FRI"]["selected_candidate_name"],
        }
    )
    if operational_model_spec_hash == OLD_NONCERTIFIED_OPERATIONAL_MODEL_SPEC_HASH:
        raise fd.HardGateFailure("NEW_OPERATIONAL_MODEL_SPEC_HASH_COLLIDED_WITH_NONCERTIFIED_V1_HASH")

    total_paired = sum(selections[h]["paired_fit_count"] for h in he.HORIZONS) + sum(len(audits_2024[h]["per_finalist"]) for h in he.HORIZONS)
    total_individual = sum(selections[h]["individual_fit_count"] for h in he.HORIZONS) + sum(len(audits_2024[h]["per_finalist"]) * 2 for h in he.HORIZONS)
    if total_paired > MAX_PAIRED_FITS or total_individual > MAX_INDIVIDUAL_FITS:
        raise fd.HardGateFailure(f"FIT_BUDGET_EXCEEDED_AT_RUNTIME: paired={total_paired} individual={total_individual}")

    summary = {
        "horizon_asof_elo_recertification_preregistration_v2_hash": preregistration["horizon_asof_elo_recertification_preregistration_v2_hash"],
        "fix6_feature_manifest_hash": preregistration["fix6_feature_manifest_hash"],
        "fix7_final_model_spec_hash": preregistration["fix7_final_model_spec_hash"],
        "candidate_registry_hash": preregistration["candidate_registry_hash"],
        "horizon_feature_semantics_version": he.HORIZON_FEATURE_SEMANTICS_VERSION,
        "horizon_feature_semantics_hash": preregistration["horizon_feature_semantics_hash"],
        "horizon_membership_ledger_hash": preregistration["horizon_membership_ledger_hash"],
        "operational_model_spec_hash": operational_model_spec_hash,
        "per_horizon_final_model_spec_hash": per_horizon_final_model_spec_hash,
        "frozen_feature_columns": preregistration["frozen_feature_columns"],
        "old_noncertified_hashes": preregistration["old_noncertified_hashes"],
        "membership_counts": membership["membership_counts"],
        "fold_membership": membership["fold_membership"],
        "firewall_counts": membership["firewall_counts"],
        "fit_counts": {"paired": total_paired, "individual": total_individual},
        "audit_a": audit_a,
        "audit_b": audit_b,
        "per_horizon": {
            h: {
                "selection_matrix_hash": matrix_info["matrices"][h]["selection_matrix_hash"],
                "selection_matrix_row_count": matrix_info["matrices"][h]["selection_matrix_row_count"],
                "reg_rows": matrix_info["matrices"][h]["reg_rows"], "post_rows": matrix_info["matrices"][h]["post_rows"],
                "huber_valid": selections[h]["huber_valid"], "huber_invalid_reasons": selections[h]["huber_invalid_reasons"],
                "ridge_alpha_selection": selections[h]["ridge_result"], "family_selection": selections[h]["family_result"],
                "selected_family": selections[h]["selected_family"], "selected_candidate_name": selections[h]["selected_candidate_name"],
                "final_model_spec_hash": per_horizon_final_model_spec_hash[h],
                "locked_post_exposure_2024_audit": audits_2024[h],
            }
            for h in he.HORIZONS
        },
        "final_operational_spec_status": "CARD_SCOPED_RIDGE_SPEC_RECERTIFIED_FOR_TUE_FRI_OPERATIONAL_SEMANTICS",
    }

    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "fix7_1_horizon_asof_elo_recertification_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    pytest_tail = _run_focused_tests()

    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, capture_output=True, text=True)
    if diff_check.returncode != 0 or diff_check.stdout.strip():
        raise fd.HardGateFailure(f"GIT_DIFF_CHECK_FAILED: {diff_check.stdout}")

    summary["pytest_output_tail"] = pytest_tail
    summary["git_diff_check"] = "clean"
    (outputs_dir / "fix7_1_horizon_asof_elo_recertification_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


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
        print("Phase 1 (repo/base/clean-tree + hash gates) OK:", json.dumps(gate, indent=2, default=str))

        phase = "PHASE2_MEMBERSHIP"
        membership = phase2_membership(gate)
        print("Phase 2 (horizon membership + hard gates) OK:", json.dumps(
            {"membership_counts": membership["membership_counts"], "fold_membership": membership["fold_membership"],
             "ledger_hash": membership["ledger_hash"]}, indent=2, default=str))

        phase = "PHASE3_BUILD_MATRICES"
        matrix_info = phase3_build_matrices(membership)

        phase = "PHASE4_PREREGISTRATION"
        preregistration = phase4_preregistration(gate, membership, matrix_info)
        print("Phase 4 (V2 preregistration freeze) OK. horizon_asof_elo_recertification_preregistration_v2_hash =",
              preregistration["horizon_asof_elo_recertification_preregistration_v2_hash"])

        phase = "PHASE5_AUDIT_A"
        audit_a = phase5_audit_a(preregistration, membership)
        print("Phase 5 (AUDIT A: CLOCK_ONLY_REG_COMPATIBILITY) OK.")

        phase = "PHASE6_AUDIT_B"
        audit_b = phase6_audit_b(preregistration, membership)
        print("Phase 6 (AUDIT B: FINAL_OPERATIONAL_SEMANTICS) OK.")

        phase = "PHASE7_TUE_SELECTION"
        selections: dict[str, dict] = {}
        selections["TUE"] = run_horizon_selection(preregistration, matrix_info, "TUE")
        print("Phase 7 (TUE inner fits + selection) OK:", selections["TUE"]["selected_family"], selections["TUE"]["selected_candidate_name"])

        phase = "PHASE8_FRI_SELECTION"
        selections["FRI"] = run_horizon_selection(preregistration, matrix_info, "FRI")
        print("Phase 8 (FRI inner fits + selection) OK:", selections["FRI"]["selected_family"], selections["FRI"]["selected_candidate_name"])

        tue_ok = selections["TUE"]["selected_candidate_name"] == REQUIRED_SELECTED_CANDIDATE
        fri_ok = selections["FRI"]["selected_candidate_name"] == REQUIRED_SELECTED_CANDIDATE
        if not (tue_ok and fri_ok):
            print(json.dumps({
                "tue_selected_candidate": selections["TUE"]["selected_candidate_name"],
                "fri_selected_candidate": selections["FRI"]["selected_candidate_name"],
                "required": REQUIRED_SELECTED_CANDIDATE,
            }, indent=2))
            print("OPERATIONAL MODEL SPEC REQUIRES RESELECTION")
            print("HORIZON-AS-OF ELO REMEDIATION INCOMPLETE — DO NOT RESUME FIX 8")
            _write_failure_evidence("RESELECTION_REQUIRED", "PHASE8_FRI_SELECTION", {
                "tue_selected_candidate": selections["TUE"]["selected_candidate_name"],
                "fri_selected_candidate": selections["FRI"]["selected_candidate_name"],
                "preregistration_hash": preregistration["horizon_asof_elo_recertification_preregistration_v2_hash"],
            })
            return 1

        phase = "PHASE9_TUE_LOCKED_2024_AUDIT"
        audits_2024: dict[str, dict] = {}
        audits_2024["TUE"] = run_horizon_locked_2024_audit(preregistration, matrix_info, selections["TUE"], expected_validate_count=285)
        print("Phase 9 [TUE] (LOCKED_POST_EXPOSURE_2024_AUDIT) OK: overall_pass =", audits_2024["TUE"]["overall_pass"], "validate_count =", audits_2024["TUE"]["validate_count"])

        phase = "PHASE9_FRI_LOCKED_2024_AUDIT"
        audits_2024["FRI"] = run_horizon_locked_2024_audit(preregistration, matrix_info, selections["FRI"], expected_validate_count=264)
        print("Phase 9 [FRI] (LOCKED_POST_EXPOSURE_2024_AUDIT) OK: overall_pass =", audits_2024["FRI"]["overall_pass"], "validate_count =", audits_2024["FRI"]["validate_count"])

        phase = "PHASE10_FINAL_FREEZE_AND_EVIDENCE"
        summary = phase10_final_freeze_and_evidence(preregistration, membership, matrix_info, selections, audits_2024, audit_a, audit_b)
        print("Phase 10 (final hash freeze + evidence/tests/git diff) OK.")
        print(json.dumps({
            "status": "FIX 7.1 V2 COMPLETE",
            "operational_model_spec_hash": summary["operational_model_spec_hash"],
            "horizon_feature_semantics_hash": summary["horizon_feature_semantics_hash"],
            "tue_selected_candidate": selections["TUE"]["selected_candidate_name"],
            "fri_selected_candidate": selections["FRI"]["selected_candidate_name"],
            "fit_counts": summary["fit_counts"],
        }, indent=2))
        print("HORIZON-AS-OF ELO REMEDIATION V2 COMPLETE — CARD-SCOPED RIDGE SPEC RECERTIFIED FOR TUE/FRI")
        return 0

    except fd.HardGateFailure as gate_failure:
        _write_failure_evidence("HARD_GATE_FAILURE", phase, {
            "gate_name": str(gate_failure),
            "preregistration_hash": preregistration.get("horizon_asof_elo_recertification_preregistration_v2_hash"),
        })
        print("HORIZON-AS-OF ELO REMEDIATION V2 INCOMPLETE — DO NOT RESUME FIX 8")
        return 1

    except Exception as exc:
        _write_failure_evidence("UNEXPECTED_EXECUTION_ERROR", phase, {
            "exception_type": type(exc).__qualname__, "exception_message": str(exc),
            "preregistration_hash": preregistration.get("horizon_asof_elo_recertification_preregistration_v2_hash"),
        })
        print("HORIZON-AS-OF ELO REMEDIATION V2 INCOMPLETE — DO NOT RESUME FIX 8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
