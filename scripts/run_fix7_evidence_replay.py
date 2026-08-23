"""Fix 7 POST-FREEZE EVIDENCE REPLAY.

This script does NOT perform model-family or hyperparameter selection. The
selection decision (``selected_family=RIDGE``, ``selected_candidate=RIDGE_ALPHA_100``,
``final_model_spec_hash=418724cb32fcb9a6a58f44fc77a7d579dbbc64d8078991c13d363751892ddede``)
was made and frozen by the original ``scripts/run_fix7_model_family_selection.py``
run (``outputs/fix7_model_family_selection_summary.json``). That run did not
persist two pieces of evidence: the per-candidate x per-inner-fold detailed
regression table (n/margin-RMSE/margin-MAE/total-RMSE/total-MAE/primary per
row), and the 2024 LOCKED_POST_EXPOSURE_AUDIT Ridge/Huber MAE values. This
script reconstructs exactly those two things by replaying the already-frozen
candidate specs on the already-frozen matrix/folds -- nothing else.

Every replayed pooled/paired quantity that the original run DID persist is
required to reproduce within a fixed numerical tolerance
(``max(1e-10, 1e-12*abs(original))``) before any reconstructed metric is
accepted. Any mismatch is a HARD STOP: this script never overwrites the
original artifacts, never re-selects, and never lets a replayed result imply
a different winner -- if the replay disagrees with the frozen record, that is
reported as a failure, not resolved by changing the frozen winner.

Run:
    NFL_MODEL_DATA_ROOT=/path/to/data NFL_MODEL_ARTIFACT_ROOT=/path/to/artifacts \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONPATH=src python scripts/run_fix7_evidence_replay.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve
from nfl_hybrid.features.week1_prior import Week1PriorConfig
from nfl_hybrid.selection import feature_deduction_2026 as fd
from nfl_hybrid.selection import model_family_selection_2026 as mfs

EXPECTED_BRANCH = "fix/model-family-selection-2026"
EXPECTED_BASE_SHA = "87bf1786ff2bbad5aa6160797eeb660ae0887e5d"

# Original Fix-7 dirty paths (already reviewed/accepted) plus this replay's
# own new paths -- nothing else may be dirty at Phase 0.
ALLOWED_DIRTY_PATHS = {
    "REMEDIATION_STATUS_2026.md",
    "docs/MODEL_FAMILY_SELECTION_2026.md",
    "scripts/run_fix7_model_family_selection.py",
    "src/nfl_hybrid/selection/model_family_selection_2026.py",
    "tests/test_model_family_selection_2026.py",
    "scripts/run_fix7_evidence_replay.py",
    "tests/test_fix7_evidence_replay.py",
}
ALLOWED_DIRTY_PREFIXES = ("outputs/fix7_",)

# Original frozen facts (V0 Section 2 of the replay-hardening request) --
# these are EXPECTED values checked against what Phase 1 actually loads from
# disk, never trusted on their own and never written anywhere as a substitute
# for the real artifact contents.
EXPECTED_FIX6_FEATURE_MANIFEST_HASH = "d523a007f7babfd0e8913f7ad6204f3670120d5d0f36183ecead9e153a8b24bf"
EXPECTED_SELECTION_MATRIX_HASH = "4034b9a468a8e01859581c338f1123a43544495670e443f68d3906c64d70a79f"
EXPECTED_SELECTION_MATRIX_ROW_COUNT = 1343
EXPECTED_PREREGISTRATION_HASH = "cafb64d89910c56b312b48cf140f901792c4a77ac7b075c98b903169f8879932"
EXPECTED_CANDIDATE_REGISTRY_HASH = "360e40e8a541c3a4480a576813c67ec13bf5dafc8d198dc8b8ca05e675cc9b58"
EXPECTED_FINAL_MODEL_SPEC_HASH = "418724cb32fcb9a6a58f44fc77a7d579dbbc64d8078991c13d363751892ddede"
EXPECTED_SELECTED_FAMILY = "RIDGE"
EXPECTED_SELECTED_CANDIDATE = "RIDGE_ALPHA_100"

FOCUSED_TEST_FILES = [
    "tests/test_model_family_selection_2026.py",
    "tests/test_fix7_evidence_replay.py",
    "tests/test_feature_deduction_2026.py",
    "tests/test_joint_score.py",
]

FAILURE_ARTIFACT_PREFIX = "outputs/fix7_evidence_replay_failure_"


class ReplayMismatch(Exception):
    """Any discrepancy between a replayed value and the frozen original
    record. Always a HARD STOP -- never resolved by changing the winner."""


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise ReplayMismatch(f"command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
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
        raise ReplayMismatch(f"REQUIRED_TEST_FAILED:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")
    return result.stdout[-2000:]


def _tolerance(original: float) -> float:
    return max(1e-10, 1e-12 * abs(original))


def _check_close(label: str, original: float, replayed: float, mismatches: list[dict]) -> dict:
    diff = abs(replayed - original)
    allowed = _tolerance(original)
    match = bool(diff <= allowed)
    record = {"label": label, "original_value": original, "replayed_value": replayed, "absolute_difference": diff, "allowed_tolerance": allowed, "match": match}
    if not match:
        mismatches.append(record)
    return record


def _check_equal(label: str, original, replayed, mismatches: list[dict]) -> dict:
    match = bool(original == replayed)
    record = {"label": label, "original_value": original, "replayed_value": replayed, "match": match}
    if not match:
        mismatches.append(record)
    return record


# ===========================================================================
# PHASE 0 -- repository / branch gate
# ===========================================================================
def phase0_repo_gate() -> dict:
    branch = _run(["git", "branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise ReplayMismatch(f"WRONG_BRANCH: on {branch!r}, expected {EXPECTED_BRANCH!r}")

    head_sha = _run(["git", "rev-parse", "HEAD"])
    origin_main = _run(["git", "rev-parse", "origin/main"])
    merge_base = _run(["git", "merge-base", "HEAD", "origin/main"])
    if head_sha != EXPECTED_BASE_SHA or origin_main != EXPECTED_BASE_SHA or merge_base != EXPECTED_BASE_SHA:
        raise ReplayMismatch(
            f"BASE_MISMATCH: head={head_sha} origin_main={origin_main} merge_base={merge_base} expected={EXPECTED_BASE_SHA}"
        )

    status_lines = [line for line in _run(["git", "status", "--short"]).splitlines() if line.strip()]
    unexpected = []
    for line in status_lines:
        path = line[3:].strip()
        if path in ALLOWED_DIRTY_PATHS or any(path.startswith(p) for p in ALLOWED_DIRTY_PREFIXES):
            continue
        unexpected.append(path)
    if unexpected:
        raise ReplayMismatch(f"UNEXPECTED_DIRTY_FILES: {unexpected}")

    return {"branch": branch, "head_sha": head_sha, "origin_main": origin_main, "merge_base": merge_base, "dirty_files": [l[3:].strip() for l in status_lines]}


# ===========================================================================
# PHASE 1 -- load original frozen evidence, verify against expected literals
# ===========================================================================
def phase1_load_original_evidence() -> dict:
    summary = json.loads((REPO_ROOT / "outputs" / "fix7_model_family_selection_summary.json").read_text())
    root = artifact_root() / "model-family-selection-2026"
    prereg = json.loads((root / "model_family_selection_preregistration.json").read_text())
    inner_fold_meta = json.loads((root / "inner_fold_results.json").read_text())
    ridge_selection = json.loads((root / "ridge_alpha_selection.json").read_text())
    family_selection = json.loads((root / "family_selection.json").read_text())
    locked_audit = json.loads((root / "locked_2024_audit.json").read_text())
    interpretability = json.loads((root / "interpretability.json").read_text())

    mismatches: list[dict] = []
    _check_equal("fix6_feature_manifest_hash", EXPECTED_FIX6_FEATURE_MANIFEST_HASH, summary["fix6_feature_manifest_hash"], mismatches)
    _check_equal("selection_matrix_hash", EXPECTED_SELECTION_MATRIX_HASH, summary["selection_matrix_hash"], mismatches)
    _check_equal("selection_matrix_row_count", EXPECTED_SELECTION_MATRIX_ROW_COUNT, summary["selection_matrix_row_count"], mismatches)
    _check_equal("model_family_preregistration_hash", EXPECTED_PREREGISTRATION_HASH, summary["model_family_preregistration_hash"], mismatches)
    _check_equal("model_family_preregistration_hash(prereg_file)", EXPECTED_PREREGISTRATION_HASH, prereg["model_family_preregistration_hash"], mismatches)
    _check_equal("candidate_registry_hash", EXPECTED_CANDIDATE_REGISTRY_HASH, prereg["candidate_registry_hash"], mismatches)
    _check_equal("final_model_spec_hash", EXPECTED_FINAL_MODEL_SPEC_HASH, summary["final_model_spec_hash"], mismatches)
    _check_equal("selected_family", EXPECTED_SELECTED_FAMILY, summary["selected_family"], mismatches)
    _check_equal("selected_candidate", EXPECTED_SELECTED_CANDIDATE, summary["selected_candidate_spec"]["name"], mismatches)
    _check_equal("selected_family(family_selection.json)", EXPECTED_SELECTED_FAMILY, family_selection["selected_family"], mismatches)
    _check_equal("selected_family(locked_2024_audit.json)", EXPECTED_SELECTED_FAMILY, locked_audit["selected_family"], mismatches)

    if mismatches:
        raise ReplayMismatch(f"ORIGINAL_EVIDENCE_DIVERGED_FROM_EXPECTED: {mismatches}")

    return {
        "summary": summary, "prereg": prereg, "inner_fold_meta": inner_fold_meta,
        "ridge_selection": ridge_selection, "family_selection": family_selection,
        "locked_audit": locked_audit, "interpretability": interpretability,
    }


# ===========================================================================
# PHASE 2 -- rebuild the exact frozen matrix (must reproduce, never repair)
# ===========================================================================
def phase2_rebuild_matrix(original: dict) -> dict:
    fix6_summary = json.loads((REPO_ROOT / "outputs" / "fix6_feature_selection_summary.json").read_text())
    frozen_feature_columns, fix6_hash = mfs.load_and_verify_fix6_feature_contract(fix6_summary)
    mfs.assert_hgbr_random_state_matches_committed_config()

    raw_games = pd.read_parquet(resolve("backfill.games"))
    raw_rows_2025_seen = int((raw_games["season"] >= 2025).sum())

    selection_games, firewall_counts_raw = fd.enforce_2025_firewall(raw_games)
    firewall_counts = {
        "raw_rows_season_2025_seen_by_replay_firewall": raw_rows_2025_seen,
        "rows_season_ge_2025_passed_beyond_replay_firewall": int((selection_games["season"] >= 2025).sum()),
    }

    week1_config = Week1PriorConfig(k=fix6_summary["week1_blend_k_selected_during_candidate_evaluation"])
    matrix, feature_columns = fd.build_candidate_matrix(selection_games, ["ELO_STRENGTH"], week1_config=week1_config)

    firewall_counts["rows_season_ge_2025_used_for_replay_features"] = int((matrix["season"] >= 2025).sum())

    if list(feature_columns) != list(frozen_feature_columns):
        raise ReplayMismatch(f"REPLAY_FEATURE_ORDER_MISMATCH: {feature_columns} != {frozen_feature_columns}")

    selection_matrix_hash, row_count, column_order = mfs.compute_selection_matrix_hash(matrix, tuple(feature_columns))

    mismatches: list[dict] = []
    _check_equal("replay_fix6_feature_manifest_hash", EXPECTED_FIX6_FEATURE_MANIFEST_HASH, fix6_hash, mismatches)
    _check_equal("replay_selection_matrix_hash", EXPECTED_SELECTION_MATRIX_HASH, selection_matrix_hash, mismatches)
    _check_equal("replay_selection_matrix_row_count", EXPECTED_SELECTION_MATRIX_ROW_COUNT, row_count, mismatches)
    if mismatches:
        raise ReplayMismatch(f"REPLAY_MATRIX_DID_NOT_REPRODUCE_FROZEN_MATRIX: {mismatches}")

    firewall_counts["rows_season_ge_2025_used_for_replay_fits"] = 0  # enforced structurally: matrix season max <= 2024, see below
    firewall_counts["rows_season_ge_2025_used_for_replay_metrics"] = 0
    if int(matrix["season"].max()) > fd.SELECTION_MAX_SEASON:
        raise ReplayMismatch("REPLAY_MATRIX_CONTAINS_ROWS_BEYOND_SELECTION_MAX_SEASON")

    return {
        "matrix": matrix, "feature_columns": list(feature_columns), "selection_matrix_hash": selection_matrix_hash,
        "selection_matrix_row_count": row_count, "selection_matrix_column_order": column_order, "firewall_counts": firewall_counts,
    }


# ===========================================================================
# PHASE 3 -- candidate spec-hash verification (no substitution, no new args)
# ===========================================================================
def phase3_verify_candidate_specs(original: dict) -> dict:
    original_by_name = {c["name"]: c for c in original["prereg"]["candidate_registry"]}
    mismatches: list[dict] = []
    spec_hashes: dict[str, str] = {}
    for candidate in mfs.CANDIDATE_REGISTRY:
        live_hash = mfs.compute_candidate_spec_hash(candidate)
        spec_hashes[candidate.name] = live_hash
        _check_equal(f"candidate_spec_hash[{candidate.name}]", original_by_name[candidate.name]["spec_hash"], live_hash, mismatches)

    live_registry_hash = mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY)
    _check_equal("candidate_registry_hash", EXPECTED_CANDIDATE_REGISTRY_HASH, live_registry_hash, mismatches)
    if mismatches:
        raise ReplayMismatch(f"CANDIDATE_SPEC_HASH_MISMATCH: {mismatches}")
    return {"spec_hashes": spec_hashes, "candidate_registry_hash": live_registry_hash}


# ===========================================================================
# PHASE 4 -- prediction-cache lookup (Section 5): there is no persisted raw
# prediction ledger from the original run anywhere on disk (only pooled/
# aggregated metrics were kept), so every candidate/fold below is a genuine
# cache miss and is replayed. This is checked, not assumed.
# ===========================================================================
def phase4_check_prediction_cache() -> dict:
    cache_dir = artifact_root() / "model-family-selection-2026" / "prediction_cache"
    loaded_from_cache: list[dict] = []
    to_regenerate: list[dict] = []
    for candidate in mfs.CANDIDATE_REGISTRY:
        for fold in fd.INNER_FOLDS:
            cache_file = cache_dir / f"{candidate.name}__{fold.name}.parquet"
            entry = {"candidate": candidate.name, "fold": fold.name}
            if cache_file.exists():
                loaded_from_cache.append(entry)  # not used by this run: no such file was ever written
            else:
                to_regenerate.append(entry)
    for finalist_name in ("RIDGE_ALPHA_100", "HUBER_FIXED", "HGBR_INCUMBENT"):
        cache_file = cache_dir / f"{finalist_name}__OUTER.parquet"
        entry = {"candidate": finalist_name, "fold": "OUTER"}
        if cache_file.exists():
            loaded_from_cache.append(entry)
        else:
            to_regenerate.append(entry)
    return {
        "cache_dir_checked": str(cache_dir),
        "replay_predictions_loaded_from_verified_cache": loaded_from_cache,
        "replay_predictions_regenerated": to_regenerate,
    }


# ===========================================================================
# PHASE 5 -- inner fold evidence replay: fit every preregistered candidate on
# every inner fold exactly once, persist the full per-row regression table.
# ===========================================================================
def phase5_inner_fold_replay(matrix_info: dict, spec_hashes: dict) -> dict:
    matrix = matrix_info["matrix"]
    feature_columns = matrix_info["feature_columns"]
    inner_results: dict[str, dict[str, pd.DataFrame]] = {}
    detail_rows: list[dict] = []
    huber_invalid_reasons: list[dict] = []
    paired_fit_count = 0
    individual_fit_count = 0

    for candidate in mfs.CANDIDATE_REGISTRY:
        inner_results[candidate.name] = {}
        for fold in fd.INNER_FOLDS:
            preds, invalid_reasons = mfs.fit_predict_fold(matrix, feature_columns, fold, candidate)
            inner_results[candidate.name][fold.name] = preds
            huber_invalid_reasons.extend(invalid_reasons)
            paired_fit_count += 1
            individual_fit_count += 2

            report = mfs._regression_report(preds)
            game_id_hash = mfs._sha256_hex({"game_ids": sorted(preds["game_id"].astype(str).tolist())})
            detail_rows.append({
                "candidate": candidate.name,
                "candidate_spec_hash": spec_hashes[candidate.name],
                "fold": fold.name,
                "train_max_season": fold.train_max_season,
                "validate_season": fold.validate_season,
                "n_games": len(preds),
                "margin_RMSE": report["margin"]["rmse"],
                "margin_MAE": report["margin"]["mae"],
                "total_RMSE": report["total"]["rmse"],
                "total_MAE": report["total"]["mae"],
                "primary_score": report["primary_score"],
                "validation_game_id_set_hash": game_id_hash,
                "metric_provenance": "POSTFREEZE_REPLAY_RECONSTRUCTED_METRIC",
            })

    huber_valid = not bool(huber_invalid_reasons)
    return {
        "inner_results": inner_results, "detail_rows": detail_rows, "huber_invalid_reasons": huber_invalid_reasons,
        "huber_valid": huber_valid, "paired_fit_count": paired_fit_count, "individual_fit_count": individual_fit_count,
    }


# ===========================================================================
# PHASE 6 -- pooled reproduction gate (Section 9)
# ===========================================================================
def phase6_pooled_reproduction_gate(inner: dict, original: dict) -> dict:
    inner_results = inner["inner_results"]
    original_pooled = {}
    for name, row in original["ridge_selection"]["per_alpha"].items():
        original_pooled[name] = row["pooled_primary_score"]
    original_pooled["HUBER_FIXED"] = original["family_selection"]["pooled_primary"]["HUBER"]
    original_pooled["HGBR_INCUMBENT"] = original["family_selection"]["pooled_primary"]["HGBR"]

    mismatches: list[dict] = []
    checks: dict[str, dict] = {}
    for candidate in mfs.CANDIDATE_REGISTRY:
        pooled = mfs._pooled(inner_results, candidate.name)
        replayed_score = mfs._primary_score(pooled)
        checks[candidate.name] = _check_close(candidate.name, original_pooled[candidate.name], replayed_score, mismatches)

    if mismatches:
        raise ReplayMismatch(f"POOLED_REPRODUCTION_GATE_FAILED: {mismatches}")
    return {"checks": checks, "all_match": True}


# ===========================================================================
# PHASE 7 -- Ridge alpha evidence cross-check (Section 10, audit only)
# ===========================================================================
def phase7_ridge_alpha_cross_check(inner: dict, original: dict) -> dict:
    replay_result = mfs.select_ridge_alpha(inner["inner_results"])
    original_result = original["ridge_selection"]

    mismatches: list[dict] = []
    _check_equal("best_alpha_name", original_result["best_alpha_name"], replay_result["best_alpha_name"], mismatches)
    _check_equal("selected_alpha_name", original_result["selected_alpha_name"], replay_result["selected_alpha_name"], mismatches)

    per_alpha_checks: dict[str, dict] = {}
    for name, orig in original_result["per_alpha"].items():
        rep = replay_result["per_alpha"][name]
        entry = {
            "mean_delta_vs_best": _check_close(f"{name}.mean_delta_vs_best", orig["mean_delta_vs_best"], rep["mean_delta_vs_best"], mismatches),
            "se_delta_vs_best": _check_close(f"{name}.se_delta_vs_best", orig["se_delta_vs_best"], rep["se_delta_vs_best"], mismatches),
            "within_one_se_of_best": _check_equal(f"{name}.within_one_se_of_best", orig["within_one_se_of_best"], rep["within_one_se_of_best"], mismatches),
        }
        per_alpha_checks[name] = entry

    if mismatches:
        raise ReplayMismatch(f"RIDGE_ALPHA_CROSS_CHECK_FAILED: {mismatches}")
    return {"replay_result": replay_result, "per_alpha_checks": per_alpha_checks}


# ===========================================================================
# PHASE 8 -- family selection evidence cross-check (Section 11, audit only)
# ===========================================================================
def phase8_family_selection_cross_check(inner: dict, ridge_cross_check: dict, original: dict) -> dict:
    replay_result = mfs.select_family(
        inner["inner_results"],
        ridge_selected_name=ridge_cross_check["replay_result"]["selected_alpha_name"],
        huber_valid=inner["huber_valid"],
    )
    original_result = original["family_selection"]

    mismatches: list[dict] = []
    _check_equal("best", original_result["best"], replay_result["best"], mismatches)
    _check_equal("one_se_set", sorted(original_result["one_se_set"]), sorted(replay_result["one_se_set"]), mismatches)
    _check_equal("tentative", original_result["tentative"], replay_result["tentative"], mismatches)
    _check_equal("selected_family", original_result["selected_family"], replay_result["selected_family"], mismatches)
    _check_equal("selection_reason", original_result["selection_reason"], replay_result["selection_reason"], mismatches)
    if replay_result["selected_family"] != EXPECTED_SELECTED_FAMILY:
        raise ReplayMismatch(
            f"REPLAY_CROSS_CHECK_WOULD_SELECT_DIFFERENT_FAMILY({replay_result['selected_family']}) "
            f"BUT FROZEN WINNER IS {EXPECTED_SELECTED_FAMILY} -- REPORTING, NOT CHANGING THE WINNER"
        )

    for fam, orig_cmp in original_result["comparisons_vs_best"].items():
        rep_cmp = replay_result["comparisons_vs_best"][fam]
        _check_close(f"comparisons_vs_best[{fam}].mean_delta", orig_cmp["mean_delta"], rep_cmp["mean_delta"], mismatches)
        _check_close(f"comparisons_vs_best[{fam}].se_delta", orig_cmp["se_delta"], rep_cmp["se_delta"], mismatches)
        _check_equal(f"comparisons_vs_best[{fam}].n", orig_cmp["n"], rep_cmp["n"], mismatches)

    if mismatches:
        raise ReplayMismatch(f"FAMILY_SELECTION_CROSS_CHECK_FAILED: {mismatches}")
    return {"replay_result": replay_result}


# ===========================================================================
# PHASE 9 -- 2024 finalist evidence replay (Section 12): fills the missing
# Ridge/Huber MAE evidence; every ORIGINALLY persisted quantity must
# reproduce within tolerance before the newly reconstructed MAEs are kept.
# ===========================================================================
def phase9_2024_finalist_replay(matrix_info: dict, original: dict) -> dict:
    matrix = matrix_info["matrix"]
    feature_columns = matrix_info["feature_columns"]
    finalist_specs = {
        "RIDGE": next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "RIDGE_ALPHA_100"),
        "HUBER": next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HUBER_FIXED"),
        "HGBR": next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HGBR_INCUMBENT"),
    }
    fd.assert_fold_seasons_allowed(fd.OUTER_FOLD)

    original_per_finalist = original["locked_audit"]["per_finalist"]
    mismatches: list[dict] = []
    per_finalist: dict[str, dict] = {}
    paired_fits = 0
    individual_fits = 0

    for family, spec in finalist_specs.items():
        preds, invalid_reasons = mfs.fit_predict_fold(matrix, feature_columns, fd.OUTER_FOLD, spec)
        if invalid_reasons:
            raise ReplayMismatch(f"REPLAY_2024_HUBER_INVALID_NUMERICAL: {invalid_reasons}")
        paired_fits += 1
        individual_fits += 2

        preds = preds.assign(segment=fd._segment_labels(preds["week"]))
        report = mfs._regression_report(preds)
        segment_primary = {}
        for segment in ("week1", "weeks2_4", "weeks5_plus"):
            seg = preds[preds["segment"] == segment]
            segment_primary[segment] = mfs._primary_score(seg) if len(seg) else None

        orig = original_per_finalist[family]
        _check_close(f"{family}.margin_rmse", orig["margin_rmse"], report["margin"]["rmse"], mismatches)
        _check_close(f"{family}.total_rmse", orig["total_rmse"], report["total"]["rmse"], mismatches)
        _check_close(f"{family}.primary_score", orig["primary_score"], report["primary_score"], mismatches)
        for segment in ("week1", "weeks2_4", "weeks5_plus"):
            _check_close(f"{family}.segment_primary_score.{segment}", orig["segment_primary_score"][segment], segment_primary[segment], mismatches)

        per_finalist[family] = {
            "candidate_name": spec.name,
            "n_games": len(preds),
            "margin_RMSE": report["margin"]["rmse"],
            "margin_MAE": report["margin"]["mae"],
            "total_RMSE": report["total"]["rmse"],
            "total_MAE": report["total"]["mae"],
            "primary_score": report["primary_score"],
            "segment_primary_score": segment_primary,
            "margin_MAE_provenance": "POSTFREEZE_REPLAY_RECONSTRUCTED_METRIC",
            "total_MAE_provenance": "POSTFREEZE_REPLAY_RECONSTRUCTED_METRIC",
            "rmse_and_primary_provenance": "ORIGINALLY_PERSISTED_SELECTION_METRIC (reproduced by replay)",
        }

    if mismatches:
        raise ReplayMismatch(f"2024_FINALIST_REPLAY_MISMATCH_VS_ORIGINAL: {mismatches}")

    return {"per_finalist": per_finalist, "paired_fits": paired_fits, "individual_fits": individual_fits}


# ===========================================================================
# PHASE 10 -- independently recompute the six locked 2024 audit rules
# (Section 13, evidence only -- does not feed back into selection)
# ===========================================================================
def phase10_recheck_audit_rules(finalist_replay: dict, original: dict) -> dict:
    per_finalist = finalist_replay["per_finalist"]
    selected = per_finalist[EXPECTED_SELECTED_FAMILY]

    def _best(metric_key):
        return min(v[metric_key] for v in per_finalist.values())

    def _best_segment(segment):
        return min(v["segment_primary_score"][segment] for v in per_finalist.values())

    rules = {
        "overall": {"selected": selected["primary_score"], "best": _best("primary_score"), "tolerance": mfs.AUDIT_PRIMARY_TOLERANCE},
        "margin": {"selected": selected["margin_RMSE"], "best": _best("margin_RMSE"), "tolerance": mfs.AUDIT_MARGIN_TOLERANCE},
        "total": {"selected": selected["total_RMSE"], "best": _best("total_RMSE"), "tolerance": mfs.AUDIT_TOTAL_TOLERANCE},
        "week1": {"selected": selected["segment_primary_score"]["week1"], "best": _best_segment("week1"), "tolerance": mfs.AUDIT_WEEK1_TOLERANCE},
        "weeks2_4": {"selected": selected["segment_primary_score"]["weeks2_4"], "best": _best_segment("weeks2_4"), "tolerance": mfs.AUDIT_WEEKS2_4_TOLERANCE},
        "weeks5_plus": {"selected": selected["segment_primary_score"]["weeks5_plus"], "best": _best_segment("weeks5_plus"), "tolerance": mfs.AUDIT_WEEKS5PLUS_TOLERANCE},
    }
    for rule in rules.values():
        rule["pass"] = bool(rule["selected"] <= rule["best"] * (1 + rule["tolerance"]))
    overall_pass = all(r["pass"] for r in rules.values())

    original_rules = original["locked_audit"]["rules"]
    mismatches: list[dict] = []
    for key, orig_rule in original_rules.items():
        rep_rule = rules[key]
        _check_close(f"rules.{key}.selected", orig_rule["selected"], rep_rule["selected"], mismatches)
        _check_close(f"rules.{key}.best", orig_rule["best"], rep_rule["best"], mismatches)
        _check_equal(f"rules.{key}.pass", orig_rule["pass"], rep_rule["pass"], mismatches)
        if not rep_rule["pass"]:
            mismatches.append({"label": f"rules.{key}.pass", "detail": "replay rule FAILED"})

    if not overall_pass or mismatches:
        raise ReplayMismatch(f"LOCKED_2024_AUDIT_REPLAY_RULE_MISMATCH_OR_FAILURE: {mismatches}")

    return {
        "label": "LOCKED_POST_EXPOSURE_2024_AUDIT_REPLAY",
        "rules": rules,
        "overall_pass": overall_pass,
        "outer_2024_previously_exposed": mfs.AUDIT_2024_PREVIOUSLY_EXPOSED,
        "outer_2024_used_for_model_selection": mfs.AUDIT_2024_USED_FOR_MODEL_SELECTION,
        "outer_2024_used_for_estimator_adjustment": mfs.AUDIT_2024_USED_FOR_ESTIMATOR_ADJUSTMENT,
        "selected_family_unchanged_after_audit": EXPECTED_SELECTED_FAMILY,
    }


# ===========================================================================
# PHASE 11 -- hash immutability gate (Section 15): re-read the ORIGINAL
# artifacts and confirm this replay wrote nothing into them.
# ===========================================================================
def phase11_hash_immutability_gate(original_before: dict) -> dict:
    reloaded = phase1_load_original_evidence()
    mismatches: list[dict] = []
    for key in ("summary", "prereg", "family_selection", "locked_audit"):
        before_bytes = json.dumps(original_before[key], sort_keys=True)
        after_bytes = json.dumps(reloaded[key], sort_keys=True)
        _check_equal(f"original_artifact_unchanged[{key}]", before_bytes == after_bytes, True, mismatches)
    if mismatches:
        raise ReplayMismatch(f"ORIGINAL_ARTIFACTS_WERE_MODIFIED_DURING_REPLAY: {mismatches}")
    return {
        "fix6_feature_manifest_hash": EXPECTED_FIX6_FEATURE_MANIFEST_HASH,
        "selection_matrix_hash": EXPECTED_SELECTION_MATRIX_HASH,
        "model_family_preregistration_hash": EXPECTED_PREREGISTRATION_HASH,
        "candidate_registry_hash": EXPECTED_CANDIDATE_REGISTRY_HASH,
        "final_model_spec_hash": EXPECTED_FINAL_MODEL_SPEC_HASH,
        "selected_family": EXPECTED_SELECTED_FAMILY,
        "selected_candidate": EXPECTED_SELECTED_CANDIDATE,
        "all_unchanged": True,
    }


def _write_failure_evidence(classification: str, phase: str, extra: dict) -> Path:
    payload = {"classification": classification, "failed_phase": phase, "no_selection_change_confirmation": "replay never calls a code path that can alter selected_family/final_model_spec_hash", **extra}
    path = REPO_ROOT / f"outputs/fix7_evidence_replay_failure.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    return path


def main() -> int:
    phase = "UNSTARTED"
    try:
        phase = "PHASE0_REPO_GATE"
        repo_gate = phase0_repo_gate()
        print("Phase 0 (repo/branch gate) OK:", json.dumps(repo_gate, indent=2))

        phase = "PHASE1_LOAD_ORIGINAL_EVIDENCE"
        original = phase1_load_original_evidence()
        print("Phase 1 (load + verify original frozen evidence) OK.")

        phase = "PHASE2_REBUILD_MATRIX"
        matrix_info = phase2_rebuild_matrix(original)
        print("Phase 2 (rebuild frozen matrix) OK: rows=%d hash=%s" % (matrix_info["selection_matrix_row_count"], matrix_info["selection_matrix_hash"]))

        phase = "PHASE3_CANDIDATE_SPEC_VERIFICATION"
        spec_check = phase3_verify_candidate_specs(original)
        print("Phase 3 (candidate spec-hash verification) OK.")

        phase = "PHASE4_PREDICTION_CACHE_CHECK"
        cache_check = phase4_check_prediction_cache()
        print(f"Phase 4 (prediction cache check) OK: cached={len(cache_check['replay_predictions_loaded_from_verified_cache'])} to_regenerate={len(cache_check['replay_predictions_regenerated'])}")

        phase = "PHASE5_INNER_FOLD_REPLAY"
        inner = phase5_inner_fold_replay(matrix_info, spec_check["spec_hashes"])
        print(f"Phase 5 (inner fold replay) OK: {len(inner['detail_rows'])} candidate x fold rows, huber_valid={inner['huber_valid']}")

        phase = "PHASE6_POOLED_REPRODUCTION_GATE"
        pooled_gate = phase6_pooled_reproduction_gate(inner, original)
        print("Phase 6 (pooled reproduction gate) OK: all six candidates reproduced within tolerance.")

        phase = "PHASE7_RIDGE_ALPHA_CROSS_CHECK"
        ridge_cross_check = phase7_ridge_alpha_cross_check(inner, original)
        print("Phase 7 (Ridge alpha cross-check) OK.")

        phase = "PHASE8_FAMILY_SELECTION_CROSS_CHECK"
        family_cross_check = phase8_family_selection_cross_check(inner, ridge_cross_check, original)
        print("Phase 8 (family selection cross-check) OK: selected_family unchanged =", family_cross_check["replay_result"]["selected_family"])

        phase = "PHASE9_2024_FINALIST_REPLAY"
        finalist_replay = phase9_2024_finalist_replay(matrix_info, original)
        print("Phase 9 (2024 finalist replay incl. MAE reconstruction) OK.")

        phase = "PHASE10_RECHECK_AUDIT_RULES"
        audit_replay = phase10_recheck_audit_rules(finalist_replay, original)
        print("Phase 10 (2024 audit rule replay) OK: overall_pass =", audit_replay["overall_pass"])

        phase = "PHASE11_HASH_IMMUTABILITY_GATE"
        immutability = phase11_hash_immutability_gate(original)
        print("Phase 11 (hash immutability gate) OK.")

        original_inner_fit_counts = original["summary"]["fit_counts"]
        original_total_paired = original_inner_fit_counts["paired"] + len(original["locked_audit"]["per_finalist"])
        original_total_individual = original_inner_fit_counts["individual"] + len(original["locked_audit"]["per_finalist"]) * 2

        replay_paired_total = inner["paired_fit_count"] + finalist_replay["paired_fits"]
        replay_individual_total = inner["individual_fit_count"] + finalist_replay["individual_fits"]

        firewall_counts = matrix_info["firewall_counts"]

        replay_evidence = {
            "schema_version": "fix7-postfreeze-evidence-replay-v1",
            "replay_used_for_selection": False,
            "selected_family_as_frozen": EXPECTED_SELECTED_FAMILY,
            "selected_candidate_as_frozen": EXPECTED_SELECTED_CANDIDATE,
            "original_frozen_hashes": {
                "fix6_feature_manifest_hash": EXPECTED_FIX6_FEATURE_MANIFEST_HASH,
                "selection_matrix_hash": EXPECTED_SELECTION_MATRIX_HASH,
                "model_family_preregistration_hash": EXPECTED_PREREGISTRATION_HASH,
                "candidate_registry_hash": EXPECTED_CANDIDATE_REGISTRY_HASH,
                "final_model_spec_hash": EXPECTED_FINAL_MODEL_SPEC_HASH,
            },
            "replay_matrix_hash": matrix_info["selection_matrix_hash"],
            "replay_matrix_row_count": matrix_info["selection_matrix_row_count"],
            "firewall_counts": firewall_counts,
            "cache_counts": {
                "replay_predictions_loaded_from_verified_cache": cache_check["replay_predictions_loaded_from_verified_cache"],
                "replay_predictions_regenerated": cache_check["replay_predictions_regenerated"],
            },
            "inner_fold_detail_rows": inner["detail_rows"],
            "pooled_reproduction_checks": pooled_gate["checks"],
            "ridge_alpha_cross_check": ridge_cross_check["per_alpha_checks"],
            "family_selection_cross_check": {
                "best": family_cross_check["replay_result"]["best"],
                "one_se_set": family_cross_check["replay_result"]["one_se_set"],
                "tentative": family_cross_check["replay_result"]["tentative"],
                "selected_family": family_cross_check["replay_result"]["selected_family"],
                "selection_reason": family_cross_check["replay_result"]["selection_reason"],
                "comparisons_vs_best": family_cross_check["replay_result"]["comparisons_vs_best"],
            },
            "finalist_2024_metrics": finalist_replay["per_finalist"],
            "locked_2024_audit_replay": audit_replay,
            "fit_counts": {
                "original_real_data_paired_fit_count": original_total_paired,
                "original_real_data_individual_fit_count": original_total_individual,
                "replay_real_data_paired_fit_count": replay_paired_total,
                "replay_real_data_individual_fit_count": replay_individual_total,
                "replay_pairs_loaded_from_verified_cache": len(cache_check["replay_predictions_loaded_from_verified_cache"]),
                "replay_pairs_regenerated": len(cache_check["replay_predictions_regenerated"]),
            },
            "hash_immutability_gate": immutability,
            "replay_environment_evidence": repo_gate,
        }
        replay_hash = mfs._sha256_hex(replay_evidence)
        replay_evidence_with_hash = {**replay_evidence, "fix7_postfreeze_evidence_replay_hash": replay_hash}

        root = artifact_root() / "model-family-selection-2026"
        (root / "fix7_postfreeze_evidence_replay.json").write_text(
            json.dumps(replay_evidence_with_hash, indent=2, default=str), encoding="utf-8"
        )

        repo_summary = {
            "schema_version": "fix7-postfreeze-evidence-replay-v1",
            "status": "POSTFREEZE_EVIDENCE_REPLAY",
            "replay_used_for_selection": False,
            "selected_family_as_frozen": EXPECTED_SELECTED_FAMILY,
            "selected_candidate_as_frozen": EXPECTED_SELECTED_CANDIDATE,
            "original_frozen_hashes": replay_evidence["original_frozen_hashes"],
            "fix7_postfreeze_evidence_replay_hash": replay_hash,
            "replay_matrix_hash": matrix_info["selection_matrix_hash"],
            "replay_matrix_row_count": matrix_info["selection_matrix_row_count"],
            "firewall_counts": firewall_counts,
            "pooled_reproduction_all_match": True,
            "ridge_alpha_cross_check_all_match": True,
            "family_selection_cross_check_all_match": True,
            "locked_2024_audit_replay_overall_pass": audit_replay["overall_pass"],
            "fit_counts": replay_evidence["fit_counts"],
            "inner_fold_detail_row_count": len(inner["detail_rows"]),
            "finalist_2024_metrics": finalist_replay["per_finalist"],
        }
        outputs_dir = REPO_ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (outputs_dir / "fix7_postfreeze_evidence_replay_summary.json").write_text(
            json.dumps(repo_summary, indent=2, default=str), encoding="utf-8"
        )

        phase = "PHASE_TESTS"
        pytest_tail = _run_focused_tests()
        print("Focused tests OK.")

        diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, capture_output=True, text=True)
        if diff_check.returncode != 0 or diff_check.stdout.strip():
            raise ReplayMismatch(f"GIT_DIFF_CHECK_FAILED: {diff_check.stdout}")

        print(json.dumps({
            "status": "FIX 7 EVIDENCE REPLAY HARDENED",
            "selected_family_unchanged": EXPECTED_SELECTED_FAMILY,
            "final_model_spec_hash_unchanged": EXPECTED_FINAL_MODEL_SPEC_HASH,
            "fix7_postfreeze_evidence_replay_hash": replay_hash,
        }, indent=2))
        print("FIX 7 EVIDENCE REPLAY HARDENED — FROZEN RIDGE SPEC UNCHANGED — READY FOR COMMIT")
        return 0

    except ReplayMismatch as mismatch:
        _write_failure_evidence(
            "SCIENTIFIC_EVIDENCE_MISMATCH", phase,
            {"mismatch": str(mismatch), "no_selection_change_confirmation": True},
        )
        print("FIX 7 EVIDENCE REPLAY FAILED — DO NOT COMMIT")
        return 1

    except Exception as exc:
        _write_failure_evidence(
            "UNEXPECTED_EXECUTION_ERROR", phase,
            {"exception_type": type(exc).__qualname__, "exception_message": str(exc)},
        )
        print("FIX 7 EVIDENCE REPLAY FAILED — DO NOT COMMIT")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
