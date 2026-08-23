"""Fix 6: Week 1 prior + compact production feature deduction -- auto-mode
end-to-end run, executed in the exact 9-phase order the auto-mode operating
contract specifies. No phase begins before the previous one's artifacts are
written; any hard-gate failure stops the run immediately with
``FIX 6 INCOMPLETE -- <gate>`` and no retry/scope-broadening/adjustment.

Run:
    NFL_MODEL_DATA_ROOT=/path/to/data NFL_MODEL_ARTIFACT_ROOT=/path/to/artifacts \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTHONPATH=src python scripts/run_fix6_feature_deduction.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve
from nfl_hybrid.features.week1_prior import (
    DEFAULT_NEUTRAL_FALLBACK,
    NEUTRAL_PRIOR_SOURCE,
    Week1PriorConfig,
    compute_week1_prior_config_hash,
)
from nfl_hybrid.selection import feature_deduction_2026 as fd

EXPECTED_BRANCH = "fix/week1-prior-feature-deduction"
EXPECTED_BASE_SHA = "39406c67"  # short-sha prefix of the merge commit this branch was built on

# Every path this pipeline (across both the original pass and this
# auto-mode revision) is allowed to have modified/created. Any OTHER dirty
# path is a hard STOP (Section 1).
ALLOWED_DIRTY_PATHS = {
    "REMEDIATION_STATUS_2026.md",
    "src/nfl_hybrid/data/external_data.py",
    "docs/FEATURE_DEDUCTION_2026.md",
    "docs/WEEK1_PRIOR_2026.md",
    "scripts/run_fix6_feature_deduction.py",
    "src/nfl_hybrid/features/week1_prior.py",
    "src/nfl_hybrid/selection/feature_deduction_2026.py",
    "tests/test_feature_deduction_2026.py",
    "tests/test_week1_prior_2026.py",
}
ALLOWED_DIRTY_PREFIXES = ("outputs/fix6_",)

FOCUSED_TEST_FILES = [
    "tests/test_week1_prior_2026.py",
    "tests/test_feature_deduction_2026.py",
    "tests/test_elo_pregame_state.py",
    "tests/test_pregame_rolling.py",
    "tests/test_balldontlie_parity.py",
    "tests/test_external_data_resolver.py",
    "tests/test_market_history_state.py",
    "tests/test_joint_score.py",
    "tests/test_labels.py",
    "tests/test_chronological_market_attachment.py",
]


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise fd.HardGateFailure(f"command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    # rstrip only -- `git status --short` lines carry meaningful LEADING
    # whitespace (the 2-char XY status code can start with a space), so a
    # full .strip() on the whole multi-line output would corrupt the first
    # line's status code.
    return result.stdout.rstrip("\n")


# ===========================================================================
# PHASE 1 -- source audit
# ===========================================================================
def phase1_source_audit() -> dict:
    branch = _run(["git", "branch", "--show-current"])
    if branch != EXPECTED_BRANCH:
        raise fd.HardGateFailure(f"WRONG_BRANCH: on {branch!r}, expected {EXPECTED_BRANCH!r}")

    head_sha = _run(["git", "rev-parse", "HEAD"])
    if not head_sha.startswith(EXPECTED_BASE_SHA):
        # Not sitting exactly on the expected base commit -- fine only if
        # HEAD is a descendant of it (_run raises HardGateFailure via a
        # nonzero `git merge-base --is-ancestor` exit code otherwise).
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

    # Re-verify (programmatically, not just at planning time) the four facts.
    registry_counts = fd.assert_registry_counts_match_expected()
    if "HOME_AWAY_CONTEXT" in fd.FEATURE_GROUPS:
        raise fd.HardGateFailure("FACT_CHECK_FAILED: HOME_AWAY_CONTEXT group must not exist.")
    try:
        fd.assert_production_eligible("team_box_official")
        raise fd.HardGateFailure("FACT_CHECK_FAILED: blocked family unexpectedly production-eligible.")
    except Exception as exc:  # ParityIneligibleError expected
        if type(exc).__name__ != "ParityIneligibleError":
            raise
    # Fact 3: neutral fallback constants are fixed, not outcome-derived.
    if DEFAULT_NEUTRAL_FALLBACK != {"points_scored": 22.0, "points_allowed": 22.0, "margin": 0.0, "win": 0.5}:
        raise fd.HardGateFailure(f"FACT_CHECK_FAILED: neutral fallback drifted: {DEFAULT_NEUTRAL_FALLBACK}")
    # Fact 4: new canonical T10 2020-2024-confirmation registry keys resolve.
    resolve("canonical_market.pregame_ats_t10_2020_2024_confirmation")
    resolve("canonical_market.pregame_total_t10_2020_2024_confirmation")

    return {
        "branch": branch,
        "head_sha": head_sha,
        "registry_counts": registry_counts,
        "dirty_files": [line[3:].strip() for line in status_lines],
    }


# ===========================================================================
# PHASE 2 -- preregistration
# ===========================================================================
def phase2_preregistration() -> dict:
    plan = fd.plan_fit_budget()
    print("Fit budget plan:", json.dumps(plan, indent=2))
    if not plan["fit_budget_ok"]:
        raise fd.HardGateFailure(f"FIT_BUDGET_EXCEEDED: {plan}")

    registry_hash = fd.freeze_candidate_registry()

    # Policy/leakage/poison-pill tests must pass here, before Phase 3.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"] + FOCUSED_TEST_FILES,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_single_threaded_env(),
    )
    print(result.stdout[-4000:])
    if result.returncode != 0:
        raise fd.HardGateFailure(f"REQUIRED_TEST_FAILED (Phase 2 pre-check):\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")

    preregistration = {
        "inner_folds": [asdict(f) for f in fd.INNER_FOLDS],
        "outer_fold": asdict(fd.OUTER_FOLD),
        "outer_2024_previously_exposed": fd.OUTER_2024_PREVIOUSLY_EXPOSED,
        "outer_2024_role": fd.OUTER_2024_ROLE,
        "candidate_group_registry_hash": registry_hash,
        "candidate_group_order": list(fd.CANDIDATE_GROUP_ORDER),
        "core_groups": list(fd.CORE_GROUPS),
        "registry_counts": fd.assert_registry_counts_match_expected(),
        "model_config_hash": fd.compute_model_config_hash(),
        "model_config": asdict(fd.MODEL_CONFIG),
        "k_grid": list(fd.K_GRID),
        "k_search_spec": list(fd.K_SEARCH_SPEC),
        "k_tie_break_rule": "selected_k = max(k such that mean_delta(k) <= se_delta(k), paired vs best_k, else best_k)",
        "primary_metric_formula": "primary_score = 0.5 * (margin_RMSE + total_RMSE)",
        "paired_loss_formula": "L_i = 0.5 * (margin_error_i^2 + total_error_i^2); delta_i = L_b_i - L_a_i; SE = std(delta, ddof=1) / sqrt(n)",
        "forward_retain_rule": "wins in >=2 of 3 inner folds AND mean_delta + se_delta < 0; among passers, lowest pooled primary_score wins (ties: fewer added features, then registry order)",
        "backward_remove_rule": "mean_delta <= se_delta (delta = L_reduced - L_current, no abs()); continuous update; fixed registry order; capped at 3 sweeps; hard STOP if sweep 3 still removes a group",
        "outer_tolerance_table": {
            "primary": fd.OUTER_PRIMARY_TOLERANCE,
            "margin": fd.OUTER_MARGIN_TOLERANCE,
            "total": fd.OUTER_TOTAL_TOLERANCE,
            "week1": fd.OUTER_WEEK1_TOLERANCE,
            "weeks2_4": fd.OUTER_WEEKS2_4_TOLERANCE,
            "weeks5_plus": fd.OUTER_WEEKS5PLUS_TOLERANCE,
        },
        "fit_budget_plan": plan,
        "selection_max_season": fd.SELECTION_MAX_SEASON,
        "selection_forbidden_seasons": list(fd.SELECTION_FORBIDDEN_SEASONS),
        "market_diagnostic_source": "canonical T10 (chronological_market_attachment.load_canonical_t10_market for <=2023; new 2020-2024-confirmation registry keys for 2024) -- never home_spread_reference/total_line_reference",
        "neutral_prior_source": NEUTRAL_PRIOR_SOURCE,
        "neutral_fallback": DEFAULT_NEUTRAL_FALLBACK,
        "selection_config_hash": fd.compute_selection_config_hash(),
    }
    payload = json.dumps(preregistration, sort_keys=True, default=str)
    preregistration["preregistration_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    root = artifact_root() / "feature-deduction-2026"
    root.mkdir(parents=True, exist_ok=True)
    (root / "selection_preregistration.json").write_text(json.dumps(preregistration, indent=2, default=str), encoding="utf-8")
    (root / "candidate_group_registry.json").write_text(
        json.dumps(
            {
                "order": list(fd.CANDIDATE_GROUP_ORDER),
                "core_groups": list(fd.CORE_GROUPS),
                "groups": {
                    name: {
                        "columns": list(fd.FEATURE_GROUPS[name].columns),
                        "kind": fd.FEATURE_GROUPS[name].kind,
                        "source_family": fd.FEATURE_GROUPS[name].source_family,
                        "interpretation": fd.FEATURE_GROUPS[name].interpretation,
                    }
                    for name in fd.CANDIDATE_GROUP_ORDER
                },
                "registry_hash": registry_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return preregistration


def _single_threaded_env() -> dict:
    import os

    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


# ===========================================================================
# PHASE 3 -- cache
# ===========================================================================
def phase3_cache(games: pd.DataFrame, week1_config: Week1PriorConfig) -> dict:
    assert_seasons = pd.to_numeric(games["season"], errors="raise")
    if (assert_seasons >= 2025).any():
        raise fd.HardGateFailure("PHASE3_2025_ROWS_PRESENT: firewall did not filter correctly.")
    matrix, features = fd.build_candidate_matrix(games, list(fd.CANDIDATE_GROUP_ORDER), week1_config=week1_config)
    if (matrix["season"] >= 2025).any():
        raise fd.HardGateFailure("PHASE3_2025_ROWS_IN_MATRIX")
    cache_hash = fd.compute_feature_manifest_hash(features)
    return {"full_matrix_rows": len(matrix), "full_matrix_feature_count": len(features), "full_matrix_hash": cache_hash}


# ===========================================================================
# PHASE 4 -- inner k selection
# ===========================================================================
def phase4_k_selection(games: pd.DataFrame) -> tuple[float, list[dict], int]:
    return fd.run_k_grid_search(games)


# ===========================================================================
# PHASE 5 -- inner feature selection
# ===========================================================================
def phase5_feature_selection(games: pd.DataFrame, week1_config: Week1PriorConfig) -> tuple[list[str], list[dict], int]:
    retained, trace, fit_count = fd.run_group_ablation(games, week1_config)
    matrix, features = fd.build_candidate_matrix(games, retained, week1_config=week1_config)
    if len(features) > 100:
        raise fd.HardGateFailure(f"FEATURE_SET_NOT_COMPACT_ENOUGH: {len(features)} > 100")
    return retained, trace, fit_count


# ===========================================================================
# PHASE 6 -- freeze
# ===========================================================================
def phase6_freeze(preregistration: dict, week1_config: Week1PriorConfig, retained_groups: list[str], features: list[str]) -> dict:
    manifest = {
        "model_config_hash": fd.compute_model_config_hash(),
        "week1_prior_config_hash": compute_week1_prior_config_hash(week1_config),
        "feature_manifest_hash": fd.compute_feature_manifest_hash(features),
        "selection_config_hash": fd.compute_selection_config_hash(),
        "candidate_group_registry_hash": preregistration["candidate_group_registry_hash"],
        "preregistration_hash": preregistration["preregistration_hash"],
        "retained_groups": retained_groups,
        "final_features": features,
        "final_feature_count": len(features),
    }
    live_registry_hash = fd.compute_candidate_group_registry_hash()
    if live_registry_hash != preregistration["candidate_group_registry_hash"]:
        raise fd.HardGateFailure("PREREGISTRATION_HASH_MISMATCH_AT_FREEZE")
    return manifest


# ===========================================================================
# PHASE 7 -- LOCKED_POST_EXPOSURE_2024_AUDIT
# ===========================================================================
def phase7_locked_audit(games: pd.DataFrame, retained_groups: list[str], week1_config: Week1PriorConfig, preregistration: dict) -> dict:
    live_registry_hash = fd.compute_candidate_group_registry_hash()
    if live_registry_hash != preregistration["candidate_group_registry_hash"]:
        raise fd.HardGateFailure("PREREGISTRATION_HASH_MISMATCH_BEFORE_PHASE7")
    result = fd.run_outer_confirmation(games, compact_groups=retained_groups, week1_config=week1_config)
    if not result["overall_pass"]:
        print(json.dumps(result, indent=2, default=str))
        raise fd.HardGateFailure(f"LOCKED_POST_EXPOSURE_2024_AUDIT_FAILED: {result}")
    return result


# ===========================================================================
# PHASE 8 -- post-freeze diagnostics
# ===========================================================================
def phase8_diagnostics(games: pd.DataFrame, retained_groups: list[str], week1_config: Week1PriorConfig, features: list[str]) -> dict:
    matrix, _ = fd.build_candidate_matrix(games, retained_groups, week1_config=week1_config)
    train = matrix[matrix["season"] <= fd.OUTER_FOLD.train_max_season]
    validate = matrix[matrix["season"] == fd.OUTER_FOLD.validate_season]

    model = fd.JointScoreModel(numeric_features=features, categorical_features=(), config=fd.MODEL_CONFIG)
    model.fit(train)

    x_validate = model.preprocessor.transform(validate[features])
    perm = permutation_importance(
        model.margin_model, x_validate, validate["home_margin"].to_numpy(), n_repeats=10, random_state=fd.SELECTION_MAX_SEASON
    )
    importance_by_feature = dict(zip(features, perm.importances_mean.tolist()))
    importance_by_group: dict[str, float] = {}
    group_of = {c: name for name in retained_groups for c in fd.FEATURE_GROUPS[name].columns}
    for feature, value in importance_by_feature.items():
        group = group_of[feature]
        importance_by_group[group] = importance_by_group.get(group, 0.0) + max(value, 0.0)

    # LOCAL_SENSITIVITY: reference values from TRAINING data only (season<=2023).
    example_row = validate.iloc[[0]]
    base_margin, base_total = model.predict_means(example_row)
    reference_values = {}
    for feat in features:
        if feat.endswith("__week1_blended") or feat.endswith("__last8_mean"):
            metric = feat.split("__")[0].replace("home_", "").replace("away_", "")
            reference_values[feat] = 22.0 if "points" in metric else (0.0 if metric == "margin" else 0.5)
        else:
            reference_values[feat] = float(train[feat].median())

    perturbation_deltas = {}
    for feat in features:
        perturbed = example_row.copy()
        perturbed[feat] = reference_values[feat]
        pm, pt = model.predict_means(perturbed)
        perturbation_deltas[feat] = {
            "reference_value": reference_values[feat],
            "original_value": float(example_row[feat].iloc[0]),
            "margin_delta": float(pm[0] - base_margin[0]),
            "total_delta": float(pt[0] - base_total[0]),
            "group": group_of[feat],
        }

    return {
        "label": "LOCAL_SENSITIVITY",
        "not_shapley_attribution": True,
        "not_additive_contribution": True,
        "example_game_id": str(example_row["game_id"].iloc[0]),
        "example_week": int(example_row["week"].iloc[0]),
        "predicted_margin": float(base_margin[0]),
        "predicted_total": float(base_total[0]),
        "reference_source": "outer model training data only (season<=2023) for recency features; fixed neutral constants for Week1-blended features",
        "perturbation_deltas": perturbation_deltas,
        "global_permutation_importance_by_feature": importance_by_feature,
        "global_permutation_importance_by_group": importance_by_group,
    }


# ===========================================================================
# PHASE 9 -- evidence / tests
# ===========================================================================
def phase9_evidence(all_results: dict) -> None:
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "fix6_week1_prior_proof.json").write_text(
        json.dumps(all_results["week1_prior_proof"], indent=2, default=str), encoding="utf-8"
    )
    (outputs_dir / "fix6_feature_selection_summary.json").write_text(
        json.dumps(all_results["feature_selection_summary"], indent=2, default=str), encoding="utf-8"
    )

    root = artifact_root() / "feature-deduction-2026"
    (root / "group_ablation_results.json").write_text(json.dumps(all_results["ablation_trace"], indent=2, default=str), encoding="utf-8")
    (root / "week1_prior_k_grid.json").write_text(json.dumps(all_results["k_trace"], indent=2, default=str), encoding="utf-8")
    (root / "selection_manifest.json").write_text(json.dumps(all_results["manifest"], indent=2, default=str), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"] + FOCUSED_TEST_FILES,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_single_threaded_env(),
    )
    print(result.stdout[-4000:])
    if result.returncode != 0:
        raise fd.HardGateFailure(f"REQUIRED_TEST_FAILED (Phase 9 final check):\n{result.stdout[-4000:]}")

    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, capture_output=True, text=True)
    if diff_check.returncode != 0 or diff_check.stdout.strip():
        raise fd.HardGateFailure(f"GIT_DIFF_CHECK_FAILED: {diff_check.stdout}")

    all_results["pytest_output_tail"] = result.stdout[-2000:]
    all_results["git_diff_check"] = "clean"


def main() -> dict:
    audit = phase1_source_audit()
    print("Phase 1 (source audit) OK:", json.dumps(audit, indent=2, default=str))

    preregistration = phase2_preregistration()
    print("Phase 2 (preregistration) OK. preregistration_hash =", preregistration["preregistration_hash"])

    raw_games = pd.read_parquet(resolve("backfill.games"))
    games, firewall_counts = fd.enforce_2025_firewall(raw_games)
    print("2025 firewall counts:", firewall_counts)

    cache_info = phase3_cache(games, Week1PriorConfig(k=fd.K_GRID[0]))
    print("Phase 3 (cache) OK:", cache_info)

    frozen_k, k_trace, k_fit_count = phase4_k_selection(games)
    print(f"Phase 4 (k selection) OK: frozen_k={frozen_k}, fits={k_fit_count}")
    week1_config = Week1PriorConfig(k=frozen_k)

    retained_groups, ablation_trace, ablation_fit_count = phase5_feature_selection(games, week1_config)
    matrix, features = fd.build_candidate_matrix(games, retained_groups, week1_config=week1_config)
    print(f"Phase 5 (feature selection) OK: retained={retained_groups}, n_features={len(features)}, fits={ablation_fit_count}")

    manifest = phase6_freeze(preregistration, week1_config, retained_groups, features)
    print("Phase 6 (freeze) OK:", json.dumps(manifest, indent=2, default=str))

    outer_result = phase7_locked_audit(games, retained_groups, week1_config, preregistration)
    print("Phase 7 (LOCKED_POST_EXPOSURE_2024_AUDIT) OK:", json.dumps(outer_result, indent=2, default=str))

    diagnostics = phase8_diagnostics(games, retained_groups, week1_config, features)
    print("Phase 8 (post-freeze diagnostics) OK.")

    manifest["outer_confirmation"] = outer_result
    manifest["interpretability"] = diagnostics
    manifest["firewall_counts"] = firewall_counts
    manifest["k_grid_trace"] = k_trace

    all_results = {
        "manifest": manifest,
        "ablation_trace": ablation_trace,
        "k_trace": k_trace,
        "week1_prior_proof": {
            "week1_prior_config_hash": manifest["week1_prior_config_hash"],
            "frozen_k": frozen_k,
            "k_grid": list(fd.K_GRID),
            "k_grid_trace": k_trace,
            "neutral_fallback": DEFAULT_NEUTRAL_FALLBACK,
            "neutral_prior_source": NEUTRAL_PRIOR_SOURCE,
            "neutral_prior_label": "UNTUNED_ENGINEERING_PRIOR",
        },
        "feature_selection_summary": {
            **manifest,
            "blocked_high_value_candidates": fd.BLOCKED_HIGH_VALUE_CANDIDATES,
            "blocked_no_live_registry_columns": fd.BLOCKED_NO_LIVE_REGISTRY_COLUMNS,
            "firewall_counts": firewall_counts,
        },
    }
    phase9_evidence(all_results)
    print("Phase 9 (evidence/tests) OK.")
    print(json.dumps({"status": "FIX 6 COMPLETE"}, indent=2))
    return all_results


if __name__ == "__main__":
    try:
        main()
    except fd.HardGateFailure as exc:
        print(f"FIX 6 INCOMPLETE -- {exc}")
        raise SystemExit(1)
