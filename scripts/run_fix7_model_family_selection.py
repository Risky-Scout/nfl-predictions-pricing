"""Fix 7: limited model-family/hyperparameter selection + estimator spec
freeze -- auto-mode end-to-end run, executed in the exact 9-phase order the
execution-safety-corrected plan specifies:

  1. SOURCE / REPO / FIX-6 CONTRACT AUDIT
  2. FIREWALL + BUILD MATRIX + SELECTION_MATRIX_HASH
  3. PREREGISTRATION FREEZE + FOCUSED PREFIT TESTS
  4. INNER CANDIDATE FITS
  5. RIDGE ALPHA SELECTION
  6. FAMILY SELECTION + FINAL MODEL SPEC FREEZE
  7. LOCKED_POST_EXPOSURE_2024_AUDIT
  8. POST-FREEZE INTERPRETABILITY -- ZERO REFITS
  9. EVIDENCE / FOCUSED TESTS / GIT DIFF CHECK / FINAL REPORT

No real historical model fit occurs before Phase 3 completes and
``model_family_preregistration_hash`` is frozen. Any ``fd.HardGateFailure``
stops the run immediately with ``FIX 7 INCOMPLETE -- <gate>`` and no retry;
any other exception is reported as an ``UNEXPECTED_EXECUTION_ERROR`` and
re-raised so its traceback stays visible (never relabeled as a scientific
gate failure).

Run:
    NFL_MODEL_DATA_ROOT=/path/to/data NFL_MODEL_ARTIFACT_ROOT=/path/to/artifacts \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONPATH=src python scripts/run_fix7_model_family_selection.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve
from nfl_hybrid.features.week1_prior import Week1PriorConfig
from nfl_hybrid.selection import feature_deduction_2026 as fd
from nfl_hybrid.selection import model_family_selection_2026 as mfs

EXPECTED_BRANCH = "fix/model-family-selection-2026"
EXPECTED_BASE_SHA = "87bf1786ff2bbad5aa6160797eeb660ae0887e5d"  # full 40-char SHA, never abbreviated

ALLOWED_DIRTY_PATHS = {
    "REMEDIATION_STATUS_2026.md",
    "docs/MODEL_FAMILY_SELECTION_2026.md",
    "scripts/run_fix7_model_family_selection.py",
    "src/nfl_hybrid/selection/model_family_selection_2026.py",
    "tests/test_model_family_selection_2026.py",
}
ALLOWED_DIRTY_PREFIXES = ("outputs/fix7_",)

FOCUSED_TEST_FILES = [
    "tests/test_model_family_selection_2026.py",
    "tests/test_feature_deduction_2026.py",
    "tests/test_joint_score.py",
    "tests/test_elo_pregame_state.py",
    "tests/test_pregame_rolling.py",
    "tests/test_balldontlie_parity.py",
    "tests/test_external_data_resolver.py",
    "tests/test_labels.py",
]

FAILURE_ARTIFACT_PREFIX = "outputs/fix7_failure_"


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


# ===========================================================================
# PHASE 1 -- source / repo / Fix-6 contract audit
# ===========================================================================
def phase1_source_audit() -> dict:
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

    summary = json.loads((REPO_ROOT / "outputs" / "fix6_feature_selection_summary.json").read_text())
    frozen_feature_columns, fix6_feature_manifest_hash = mfs.load_and_verify_fix6_feature_contract(summary)
    mfs.assert_hgbr_random_state_matches_committed_config()

    return {
        "branch": branch,
        "head_sha": head_sha,
        "expected_base_sha": EXPECTED_BASE_SHA,
        "dirty_files": [line[3:].strip() for line in status_lines],
        "frozen_feature_columns": list(frozen_feature_columns),
        "fix6_feature_manifest_hash": fix6_feature_manifest_hash,
        "fix6_week1_blend_k": summary["week1_blend_k_selected_during_candidate_evaluation"],
        "hgbr_random_state_verified": fd.MODEL_CONFIG.random_state,
    }


# ===========================================================================
# PHASE 2 -- firewall + build frozen selection matrix + selection_matrix_hash
# (NO fits, NO comparative metrics -- correction #1)
# ===========================================================================
def phase2_build_selection_matrix(audit: dict) -> dict:
    raw_games = pd.read_parquet(resolve("backfill.games"))
    selection_games, firewall_counts = fd.enforce_2025_firewall(raw_games)

    week1_config = Week1PriorConfig(k=audit["fix6_week1_blend_k"])
    matrix, feature_columns = fd.build_candidate_matrix(selection_games, ["ELO_STRENGTH"], week1_config=week1_config)

    if list(feature_columns) != audit["frozen_feature_columns"]:
        raise fd.HardGateFailure(
            f"FIX6_FEATURE_CONTRACT_VIOLATION: order/membership mismatch: {feature_columns} != {audit['frozen_feature_columns']}"
        )
    if int(matrix["season"].max()) > fd.SELECTION_MAX_SEASON:
        raise fd.HardGateFailure("PHASE2_2025_ROWS_IN_MATRIX")
    if matrix[feature_columns].isna().sum().sum() != 0:
        raise fd.HardGateFailure("PHASE2_UNEXPECTED_MISSINGNESS_IN_FROZEN_SIX_FEATURES")

    selection_matrix_hash, selection_matrix_row_count, selection_matrix_column_order = mfs.compute_selection_matrix_hash(
        matrix, tuple(feature_columns)
    )

    return {
        "matrix": matrix,
        "feature_columns": list(feature_columns),
        "week1_config": week1_config,
        "firewall_counts": firewall_counts,
        "selection_matrix_hash": selection_matrix_hash,
        "selection_matrix_row_count": selection_matrix_row_count,
        "selection_matrix_column_order": selection_matrix_column_order,
    }


# ===========================================================================
# PHASE 3 -- preregistration freeze + focused prefit tests
# ===========================================================================
def phase3_preregistration(audit: dict, matrix_info: dict) -> dict:
    plan = mfs.plan_fit_budget()
    print("Fit budget plan:", json.dumps(plan, indent=2))
    if not plan["fit_budget_ok"]:
        raise fd.HardGateFailure(f"FIT_BUDGET_EXCEEDED: {plan}")

    candidate_registry_hash = mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY)

    preregistration_body = {
        "schema_version": mfs.SCHEMA_VERSION,
        "repo": {"branch": audit["branch"], "head_sha": audit["head_sha"], "expected_base_sha": audit["expected_base_sha"]},
        "fix6_feature_manifest_hash": audit["fix6_feature_manifest_hash"],
        "frozen_feature_columns": audit["frozen_feature_columns"],
        "selection_matrix_hash": matrix_info["selection_matrix_hash"],
        "selection_matrix_row_count": matrix_info["selection_matrix_row_count"],
        "selection_matrix_column_order": matrix_info["selection_matrix_column_order"],
        "inner_folds": [asdict(f) for f in fd.INNER_FOLDS],
        "outer_fold": asdict(fd.OUTER_FOLD),
        "selection_forbidden_seasons": list(fd.SELECTION_FORBIDDEN_SEASONS),
        "selection_max_season": fd.SELECTION_MAX_SEASON,
        "margin_target_definition": "home_score - away_score",
        "total_target_definition": "home_score + away_score",
        "candidate_registry": [
            {"name": c.name, "family": c.family, "complexity_rank": c.complexity_rank, "preprocessing": c.preprocessing,
             "hyperparameters": c.hyperparameters, "random_state": c.random_state, "spec_hash": mfs.compute_candidate_spec_hash(c)}
            for c in mfs.CANDIDATE_REGISTRY
        ],
        "candidate_registry_hash": candidate_registry_hash,
        "preprocessing_rule": "StandardScaler(with_mean=True, with_std=True) fit on TRAINING FOLD ROWS ONLY, never on validation data",
        "ridge_alpha_rule": "best_alpha = argmin pooled inner primary_score; alpha within one SE of best iff mean_delta(alpha) <= se_delta(alpha), delta_i = L_alpha_i - L_best_i; select LARGEST qualifying alpha",
        "family_complexity_order": "RIDGE(0) < HUBER(1) < HGBR(2)",
        "family_selection_algorithm": mfs.FAMILY_SELECTION_ALGORITHM_SOURCE,
        "primary_metric_formula": "primary_score = 0.5 * (margin_RMSE + total_RMSE)",
        "paired_loss_formula": "L_i = 0.5 * (margin_error_i^2 + total_error_i^2); delta_i = L_a_i - L_b_i; SE = std(delta, ddof=1) / sqrt(n)",
        "fit_budget_plan": plan,
        "audit_tolerance_table": {
            "overall": mfs.AUDIT_PRIMARY_TOLERANCE, "margin": mfs.AUDIT_MARGIN_TOLERANCE, "total": mfs.AUDIT_TOTAL_TOLERANCE,
            "week1": mfs.AUDIT_WEEK1_TOLERANCE, "weeks2_4": mfs.AUDIT_WEEKS2_4_TOLERANCE, "weeks5_plus": mfs.AUDIT_WEEKS5PLUS_TOLERANCE,
        },
        "outer_2024_previously_exposed": mfs.AUDIT_2024_PREVIOUSLY_EXPOSED,
        "outer_2024_role": mfs.AUDIT_2024_ROLE,
        "forbidden_market_columns": sorted(fd.FORBIDDEN_MARKET_COLUMNS),
        "deterministic_settings": {"random_state": fd.MODEL_CONFIG.random_state, "single_threaded": True},
        "ats_total_probability_evaluation": "DEFERRED_UNTIL_FINAL_FAMILY_CHRONOLOGICAL_OOF",
    }

    # Focused synthetic/policy tests run here, before Phase 4's first real historical fit.
    pytest_tail = _run_focused_tests()

    model_family_preregistration_hash = mfs.compute_preregistration_hash(preregistration_body)
    preregistration = {**preregistration_body, "model_family_preregistration_hash": model_family_preregistration_hash}

    root = artifact_root() / "model-family-selection-2026"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_family_selection_preregistration.json").write_text(
        json.dumps(preregistration, indent=2, default=str), encoding="utf-8"
    )
    preregistration["_phase3_pretest_output_tail"] = pytest_tail
    return preregistration


def _verify_preregistration_unchanged(preregistration: dict) -> None:
    body = {k: v for k, v in preregistration.items() if not k.startswith("_") and k != "model_family_preregistration_hash"}
    if mfs.compute_preregistration_hash(body) != preregistration["model_family_preregistration_hash"]:
        raise fd.HardGateFailure("PREREGISTRATION_HASH_MUTATED_AFTER_FREEZE")


# ===========================================================================
# PHASE 4 -- inner candidate fits
# ===========================================================================
def phase4_inner_fits(preregistration: dict, matrix_info: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    cache = mfs.PredictionCache()
    inner_results, huber_invalid_reasons, paired, individual = mfs.run_inner_fits(
        matrix_info["matrix"], matrix_info["feature_columns"], cache=cache,
        fix6_feature_manifest_hash=preregistration["fix6_feature_manifest_hash"],
        selection_matrix_hash=preregistration["selection_matrix_hash"],
        preregistration_hash=preregistration["model_family_preregistration_hash"],
    )
    plan = preregistration["fit_budget_plan"]
    if paired > plan["max_paired"] or individual > plan["max_individual"]:
        raise fd.HardGateFailure(f"FIT_BUDGET_EXCEEDED_AT_RUNTIME: paired={paired} individual={individual}")
    huber_valid = not bool(huber_invalid_reasons)
    return {
        "inner_results": inner_results, "huber_invalid_reasons": huber_invalid_reasons, "huber_valid": huber_valid,
        "paired_fit_count": paired, "individual_fit_count": individual,
    }


# ===========================================================================
# PHASE 5 -- Ridge alpha selection
# ===========================================================================
def phase5_ridge_alpha(inner: dict) -> dict:
    return mfs.select_ridge_alpha(inner["inner_results"])


# ===========================================================================
# PHASE 6 -- family selection + final model spec freeze
# ===========================================================================
def phase6_family_selection_and_freeze(preregistration: dict, inner: dict, ridge_result: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    family_result = mfs.select_family(
        inner["inner_results"], ridge_selected_name=ridge_result["selected_alpha_name"], huber_valid=inner["huber_valid"]
    )
    selected_family = family_result["selected_family"]
    representative_name = family_result["family_representative"][selected_family]
    selected_spec = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == representative_name)

    final_model_spec_hash = mfs.compute_final_model_spec_hash(
        fix6_feature_manifest_hash=preregistration["fix6_feature_manifest_hash"],
        frozen_feature_columns=preregistration["frozen_feature_columns"],
        selected_family=selected_family, selected_spec=selected_spec,
    )

    live_registry_hash = mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY)
    if live_registry_hash != preregistration["candidate_registry_hash"]:
        raise fd.HardGateFailure("CANDIDATE_REGISTRY_HASH_MISMATCH_AT_FREEZE")

    return {"family_result": family_result, "selected_family": selected_family, "selected_spec": selected_spec, "final_model_spec_hash": final_model_spec_hash}


# ===========================================================================
# PHASE 7 -- LOCKED_POST_EXPOSURE_2024_AUDIT
# ===========================================================================
def phase7_locked_audit(preregistration: dict, matrix_info: dict, inner: dict, freeze: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    finalists = {"RIDGE": next(c for c in mfs.CANDIDATE_REGISTRY if c.name == freeze["family_result"]["family_representative"]["RIDGE"]),
                 "HGBR": next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HGBR_INCUMBENT")}
    if inner["huber_valid"]:
        finalists["HUBER"] = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HUBER_FIXED")

    result = mfs.run_locked_2024_audit(
        matrix_info["matrix"], matrix_info["feature_columns"], finalists=finalists, selected_family=freeze["selected_family"]
    )
    if not result["overall_pass"]:
        printable = {k: v for k, v in result.items() if k != "fitted_bundles"}
        print(json.dumps(printable, indent=2, default=str))
        raise fd.HardGateFailure(f"LOCKED_POST_EXPOSURE_2024_AUDIT_FAILED: {printable['rules']}")
    return result


# ===========================================================================
# PHASE 8 -- post-freeze interpretability, ZERO refits
# ===========================================================================
def phase8_interpretability(preregistration: dict, matrix_info: dict, freeze: dict, audit_result: dict) -> dict:
    selected_family = freeze["selected_family"]
    bundle = audit_result["fitted_bundles"][selected_family]
    matrix = matrix_info["matrix"]
    feature_columns = matrix_info["feature_columns"]
    train_matrix = matrix[matrix["season"] <= fd.OUTER_FOLD.train_max_season]
    audit_matrix = matrix[matrix["season"] == fd.OUTER_FOLD.validate_season]

    if selected_family == "HGBR":
        global_importance = mfs.run_hgbr_permutation_importance(bundle["model"], audit_matrix, feature_columns)
        family_specific = {"method": "PERMUTATION_IMPORTANCE", "result": global_importance}
    else:
        family_specific = {"method": "COEFFICIENTS", "result": mfs.ridge_or_huber_interpretability(bundle, feature_columns)}

    local_sensitivity = mfs.local_sensitivity_example(
        bundle, freeze["selected_spec"].family, audit_matrix, train_matrix, feature_columns
    )
    return {"selected_family": selected_family, "family_specific": family_specific, "local_sensitivity": local_sensitivity, "phase8_additional_real_data_fits": 0}


# ===========================================================================
# PHASE 9 -- evidence / focused tests / git diff check / final report
# ===========================================================================
def phase9_evidence(preregistration: dict, all_results: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)

    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "fix7_model_family_selection_summary.json").write_text(
        json.dumps(all_results["summary"], indent=2, default=str), encoding="utf-8"
    )

    root = artifact_root() / "model-family-selection-2026"
    (root / "inner_fold_results.json").write_text(json.dumps(all_results["inner_fold_report"], indent=2, default=str), encoding="utf-8")
    (root / "ridge_alpha_selection.json").write_text(json.dumps(all_results["ridge_result"], indent=2, default=str), encoding="utf-8")
    (root / "family_selection.json").write_text(json.dumps(all_results["family_result"], indent=2, default=str), encoding="utf-8")
    (root / "locked_2024_audit.json").write_text(
        json.dumps({k: v for k, v in all_results["audit_result"].items() if k != "fitted_bundles"}, indent=2, default=str),
        encoding="utf-8",
    )
    (root / "interpretability.json").write_text(json.dumps(all_results["interpretability"], indent=2, default=str), encoding="utf-8")

    pytest_tail = _run_focused_tests()

    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, capture_output=True, text=True)
    if diff_check.returncode != 0 or diff_check.stdout.strip():
        raise fd.HardGateFailure(f"GIT_DIFF_CHECK_FAILED: {diff_check.stdout}")

    return {"pytest_output_tail": pytest_tail, "git_diff_check": "clean"}


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
        phase = "PHASE1_SOURCE_AUDIT"
        audit = phase1_source_audit()
        print("Phase 1 (source/repo/Fix-6 contract audit) OK:", json.dumps(audit, indent=2, default=str))

        phase = "PHASE2_BUILD_SELECTION_MATRIX"
        matrix_info = phase2_build_selection_matrix(audit)
        print(
            "Phase 2 (firewall + selection matrix) OK: rows=%d selection_matrix_hash=%s firewall=%s"
            % (matrix_info["selection_matrix_row_count"], matrix_info["selection_matrix_hash"], matrix_info["firewall_counts"])
        )

        phase = "PHASE3_PREREGISTRATION"
        preregistration = phase3_preregistration(audit, matrix_info)
        print("Phase 3 (preregistration freeze) OK. model_family_preregistration_hash =", preregistration["model_family_preregistration_hash"])

        phase = "PHASE4_INNER_FITS"
        inner = phase4_inner_fits(preregistration, matrix_info)
        print(f"Phase 4 (inner candidate fits) OK: paired={inner['paired_fit_count']} individual={inner['individual_fit_count']} huber_valid={inner['huber_valid']}")

        phase = "PHASE5_RIDGE_ALPHA_SELECTION"
        ridge_result = phase5_ridge_alpha(inner)
        print("Phase 5 (Ridge alpha selection) OK:", ridge_result["selected_alpha_name"])

        phase = "PHASE6_FAMILY_SELECTION_FREEZE"
        freeze = phase6_family_selection_and_freeze(preregistration, inner, ridge_result)
        print("Phase 6 (family selection + freeze) OK:", freeze["selected_family"], freeze["final_model_spec_hash"])

        phase = "PHASE7_LOCKED_2024_AUDIT"
        audit_result = phase7_locked_audit(preregistration, matrix_info, inner, freeze)
        print("Phase 7 (LOCKED_POST_EXPOSURE_2024_AUDIT) OK: overall_pass =", audit_result["overall_pass"])

        phase = "PHASE8_INTERPRETABILITY"
        interpretability = phase8_interpretability(preregistration, matrix_info, freeze, audit_result)
        print("Phase 8 (post-freeze interpretability, zero refits) OK.")

        summary = {
            "fix6_feature_manifest_hash": preregistration["fix6_feature_manifest_hash"],
            "frozen_feature_columns": preregistration["frozen_feature_columns"],
            "selection_matrix_hash": preregistration["selection_matrix_hash"],
            "selection_matrix_row_count": preregistration["selection_matrix_row_count"],
            "model_family_preregistration_hash": preregistration["model_family_preregistration_hash"],
            "firewall_counts": matrix_info["firewall_counts"],
            "fit_counts": {"paired": inner["paired_fit_count"], "individual": inner["individual_fit_count"]},
            "huber_valid": inner["huber_valid"],
            "huber_invalid_reasons": inner["huber_invalid_reasons"],
            "ridge_alpha_selection": ridge_result,
            "family_selection": freeze["family_result"],
            "selected_family": freeze["selected_family"],
            "selected_candidate_spec": {
                "name": freeze["selected_spec"].name, "family": freeze["selected_spec"].family,
                "hyperparameters": freeze["selected_spec"].hyperparameters, "random_state": freeze["selected_spec"].random_state,
            },
            "final_model_spec_hash": freeze["final_model_spec_hash"],
            "locked_post_exposure_2024_audit": {k: v for k, v in audit_result.items() if k != "fitted_bundles"},
            "interpretability": interpretability,
            "phase8_additional_real_data_fits": 0,
            "ats_total_probability_evaluation": "DEFERRED_UNTIL_FINAL_FAMILY_CHRONOLOGICAL_OOF",
            "final_feature_spec_status": "FROZEN_ESTIMATOR_SPEC_FOR_OFFICIAL_OOF_CALIBRATION",
        }

        phase = "PHASE9_EVIDENCE"
        evidence = phase9_evidence(
            preregistration,
            {
                "summary": summary, "inner_fold_report": {"paired": inner["paired_fit_count"], "individual": inner["individual_fit_count"], "huber_invalid_reasons": inner["huber_invalid_reasons"]},
                "ridge_result": ridge_result, "family_result": freeze["family_result"], "audit_result": audit_result, "interpretability": interpretability,
            },
        )
        summary["pytest_output_tail"] = evidence["pytest_output_tail"]
        summary["git_diff_check"] = evidence["git_diff_check"]
        (REPO_ROOT / "outputs" / "fix7_model_family_selection_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("Phase 9 (evidence/tests/git diff/final report) OK.")

        print(json.dumps({
            "status": "FIX 7 COMPLETE",
            "selected_family": freeze["selected_family"],
            "final_model_spec_hash": freeze["final_model_spec_hash"],
        }, indent=2))
        print("FIX 7 COMPLETE — MODEL FAMILY AND HYPERPARAMETER SPEC FROZEN FOR OFFICIAL CHRONOLOGICAL OOF/CALIBRATION")
        return 0

    except fd.HardGateFailure as gate_failure:
        _write_failure_evidence(
            "HARD_GATE_FAILURE", phase,
            {"gate_name": str(gate_failure), "preregistration_hash": preregistration.get("model_family_preregistration_hash")},
        )
        print("FIX 7 INCOMPLETE — DO NOT PROCEED")
        return 1  # no retry, no fallback

    except Exception as exc:
        _write_failure_evidence(
            "UNEXPECTED_EXECUTION_ERROR", phase,
            {"exception_type": type(exc).__qualname__, "exception_message": str(exc),
             "preregistration_hash": preregistration.get("model_family_preregistration_hash")},
        )
        print("FIX 7 INCOMPLETE — DO NOT PROCEED")
        raise  # re-raised: traceback stays visible, never relabeled as a scientific gate failure


if __name__ == "__main__":
    raise SystemExit(main())
