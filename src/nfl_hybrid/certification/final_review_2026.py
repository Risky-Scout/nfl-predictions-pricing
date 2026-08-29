"""FINAL REVIEW CERTIFICATION 2026 -- Sections 1-11 core library.

Model-family robustness (Sections 2-6) and calibration-improvement
robustness (Sections 7-11) for the certified ``v2026.1-fix8-certified``
baseline (commit ``d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7``).

This module has ZERO model-selection authority and ZERO calibration-fit
authority: it never changes the production model (RIDGE_ALPHA_100, six
frozen Elo features) and never refits/recalibrates anything. Section 2's
diagnostic fits apply the ALREADY-FROZEN Fix-7 candidate registry and the
ALREADY-FROZEN Fix-7 selection procedure (imported unchanged from
:mod:`nfl_hybrid.selection.model_family_selection_2026`) to a new,
TUE/FRI-horizon-scoped, REG+POST diagnostic population -- an audit of
whether that already-frozen procedure is stable, not a new search. Section
7's calibration analysis reuses the already-certified Fix-8 2025 raw and
calibrated probability ledgers verbatim; it never recalibrates.

Canonical JSON hashing follows repo convention (no shared utility exists;
see ``model_family_selection_2026._canonical_json`` /
``run_fix8_official_oof_calibration._canonical_json`` for the identical
pattern reused here): ``json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=True)`` -> SHA-256 hex digest. No ``default=str``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np
import pandas as pd

from nfl_hybrid.evaluation import official_horizon_oof as ohf
from nfl_hybrid.evaluation.week1_reliability import NOT_ESTIMABLE, binary_log_loss, brier, equal_mass_ece
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.selection import feature_deduction_2026 as fd
from nfl_hybrid.selection import model_family_selection_2026 as mfs

SCHEMA_VERSION = "final-review-certification-2026-v1"

CERTIFIED_BASELINE_TAG = "v2026.1-fix8-certified"
CERTIFIED_BASELINE_SHA = "d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7"
CERTIFIED_HORIZON_FEATURE_SEMANTICS_HASH = "bf0b136dd9b7c7f3741617b6c088926e539406d60b82563839eb1825de9fc72d"
CERTIFIED_HORIZON_MEMBERSHIP_LEDGER_HASH = "b095a2bf9177029eb67e1516fbc79b080ef800cfe18b86989254dd4d0a8dae49"
CERTIFIED_OPERATIONAL_MODEL_SPEC_HASH = "3b230bfeee3279c0e3ed9b6a7118931c1a5cf08203155be011f362c66ee8d722"
CERTIFIED_FIX8_PREREGISTRATION_HASH = "0d0a1eefffdbb6878e3246d2b5b0fb8f091bebc0eeb9f3869624620fb9e2847e"

BOOTSTRAP_SEED = 20260826
BOOTSTRAP_REPS = 10_000

STREAM_NAMES = ("ATS_TUE", "ATS_FRI", "TOTAL_TUE", "TOTAL_FRI")
_STREAM_FILE_STEM = {
    "ATS_TUE": "ats_tue", "ATS_FRI": "ats_fri", "TOTAL_TUE": "total_tue", "TOTAL_FRI": "total_fri",
}


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(payload: dict) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


class CertificationGateFailure(RuntimeError):
    """Hard STOP for this certification package -- never silently repaired."""


# ===========================================================================
# Section 0 -- repository gate.
# ===========================================================================
# This certification work itself is the only expected source of new/dirty
# paths while it is being built out phase by phase -- exactly the same
# "ALLOWED_DIRTY_PATHS/PREFIXES" pattern Fix 8 uses. Anything outside these
# prefixes is unexpected and still hard-stops the gate.
ALLOWED_DIRTY_PREFIXES: tuple[str, ...] = (
    "src/nfl_hybrid/certification/",
    "scripts/run_final_review_certification.py",
    "scripts/run_2026_production_card.py",
    "scripts/report_2026_prospective_performance.py",
    "scripts/verify_review_package.py",
    "src/nfl_hybrid/production/",
    "tests/test_final_review_certification_2026.py",
    "tests/test_production_card_2026.py",
    "docs/MODEL_STRENGTH_AND_LIMITATIONS.md",
    "docs/PRODUCTION_RUNBOOK_2026.md",
    "docs/MIKE_HANDOFF.md",
    "outputs/final_review_certification_summary.json",
    "outputs/review_package_manifest_2026.json",
    ".github/workflows/production-2026.yml",
)


def verify_repo_gate(*, branch: str, head_sha: str, tag_commit_sha: str, dirty_paths: Sequence[str]) -> dict:
    problems = []
    if branch != "cert/robustness-production-readiness-2026":
        problems.append(f"WRONG_BRANCH: {branch!r}")
    if head_sha != CERTIFIED_BASELINE_SHA:
        problems.append(f"HEAD_SHA_MISMATCH: {head_sha!r} != {CERTIFIED_BASELINE_SHA!r}")
    if tag_commit_sha != CERTIFIED_BASELINE_SHA:
        problems.append(f"TAG_COMMIT_MISMATCH: {tag_commit_sha!r} != {CERTIFIED_BASELINE_SHA!r}")
    unexpected = [p for p in dirty_paths if not any(p.startswith(prefix) for prefix in ALLOWED_DIRTY_PREFIXES)]
    if unexpected:
        problems.append(f"UNEXPECTED_DIRTY_FILES: {unexpected}")
    if problems:
        raise CertificationGateFailure("; ".join(problems))
    return {
        "branch": branch, "head_sha": head_sha, "tag": CERTIFIED_BASELINE_TAG,
        "tag_commit_sha": tag_commit_sha, "dirty_paths": list(dirty_paths),
    }


def verify_certified_hashes(fix71_summary: dict, fix8_preregistration: dict) -> dict:
    checks = {
        "horizon_feature_semantics_hash": (
            fix71_summary["horizon_feature_semantics_hash"], CERTIFIED_HORIZON_FEATURE_SEMANTICS_HASH,
        ),
        "horizon_membership_ledger_hash": (
            fix71_summary["horizon_membership_ledger_hash"], CERTIFIED_HORIZON_MEMBERSHIP_LEDGER_HASH,
        ),
        "operational_model_spec_hash": (
            fix71_summary["operational_model_spec_hash"], CERTIFIED_OPERATIONAL_MODEL_SPEC_HASH,
        ),
        "fix8_preregistration_hash": (
            fix8_preregistration["fix8_official_oof_calibration_preregistration_hash"],
            CERTIFIED_FIX8_PREREGISTRATION_HASH,
        ),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise CertificationGateFailure(f"HASH_MISMATCH: {mismatches}")
    return {k: v[0] for k, v in checks.items()}


# ===========================================================================
# Section 2/3 -- robustness scopes (predeclared; not editable after seeing
# results).
# ===========================================================================
@dataclass(frozen=True)
class RobustnessScope:
    name: str
    horizons: tuple[str, ...]
    folds: tuple[str, ...]
    description: str


ROBUSTNESS_SCOPES: tuple[RobustnessScope, ...] = (
    RobustnessScope("S1", ("TUE",), ("A", "B", "C"), "all folds pooled, TUE"),
    RobustnessScope("S2", ("FRI",), ("A", "B", "C"), "all folds pooled, FRI"),
    RobustnessScope("S3", ("TUE", "FRI"), ("A", "B", "C"), "all folds pooled, TUE+FRI with game_id clustering"),
    RobustnessScope("S4", ("TUE", "FRI"), ("B", "C"), "folds B+C only"),
    RobustnessScope("S5", ("TUE", "FRI"), ("A", "C"), "folds A+C only"),
    RobustnessScope("S6", ("TUE", "FRI"), ("A", "B"), "folds A+B only"),
    RobustnessScope("S7", ("TUE", "FRI"), ("A",), "validation season 2021 only"),
    RobustnessScope("S8", ("TUE", "FRI"), ("B",), "validation season 2022 only"),
    RobustnessScope("S9", ("TUE", "FRI"), ("C",), "validation season 2023 only"),
)
SCOPE_BY_NAME = {s.name: s for s in ROBUSTNESS_SCOPES}


# ===========================================================================
# Section 2 -- diagnostic fits. Reuses mfs.fit_predict_fold /
# mfs.CANDIDATE_REGISTRY / fd.INNER_FOLDS UNCHANGED, on a new TUE/FRI-scoped
# matrix (ohf.build_official_horizon_matrix, the same construction Fix 8
# itself certified). ZERO model-selection authority: RIDGE_ALPHA_100 remains
# the production model regardless of this audit's outcome.
# ===========================================================================
def load_reg_post_games_through_2025() -> pd.DataFrame:
    from nfl_hybrid.data.external_data import resolve

    raw_games = pd.read_parquet(resolve("backfill.games"))
    games = raw_games[raw_games["season_type"].isin(["REG", "POST"])].copy()
    games = games[pd.to_numeric(games["season"], errors="raise") <= 2025].reset_index(drop=True)
    games["game_id"] = games["game_id"].astype(str)
    return games


def build_membership_ledger(games: pd.DataFrame) -> pd.DataFrame:
    return he.build_horizon_membership_ledger(games)


def run_diagnostic_fits(games: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Returns ``diagnostic_results[horizon][candidate_name][fold_name] ->
    DataFrame`` -- one fixed fit per (horizon, candidate, fold), identical
    in structure to ``mfs.run_inner_fits``'s ``inner_results`` but computed
    once per horizon on that horizon's own eligible-target matrix."""
    diagnostic_results: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    huber_invalid_reasons: list[dict] = []
    fit_counts = {"paired": 0, "individual": 0}
    for horizon in he.HORIZONS:
        matrix = ohf.build_official_horizon_matrix(games, horizon, ledger)
        inner_results: dict[str, dict[str, pd.DataFrame]] = {}
        for candidate in mfs.CANDIDATE_REGISTRY:
            inner_results[candidate.name] = {}
            for fold in fd.INNER_FOLDS:
                preds, invalid_reasons = mfs.fit_predict_fold(matrix, ohf.ELO_FEATURE_COLUMNS, fold, candidate)
                preds = preds.assign(horizon=horizon)
                inner_results[candidate.name][fold.name] = preds
                for r in invalid_reasons:
                    huber_invalid_reasons.append({**r, "horizon": horizon})
                fit_counts["paired"] += 1
                fit_counts["individual"] += 2
        diagnostic_results[horizon] = inner_results
    return {
        "diagnostic_results": diagnostic_results,
        "huber_invalid_reasons": huber_invalid_reasons,
        "fit_counts": fit_counts,
    }


# ===========================================================================
# Section 3 -- scope-generalized selection. Algorithm-identical, parameterized
# copies of mfs.select_ridge_alpha / mfs.select_family (which hardcode
# fd.INNER_FOLDS as the pooling set) -- this is what "apply the frozen
# selection procedure in a different scope" REQUIRES; it deliberately calls
# the same private analytic primitives (mfs._primary_score,
# mfs.paired_delta_se) rather than reimplementing their formulas.
# ===========================================================================
def _scoped_frame(
    diagnostic_results: dict[str, dict[str, dict[str, pd.DataFrame]]],
    candidate_name: str,
    horizons: Sequence[str],
    fold_names: Sequence[str],
) -> pd.DataFrame:
    """Concatenates the requested (horizon, fold) cells for one candidate.
    ``game_id`` is made horizon-unique (``"<game_id>::<horizon>"``) so that
    ``mfs.paired_delta_se``'s ``(game_id, fold)`` one-to-one merge stays
    valid when TUE and FRI both contribute a row for the same underlying
    game -- the real ``game_id`` is preserved in a separate column for
    game_id-cluster bootstrap use (Section 4)."""
    parts = []
    for horizon in horizons:
        for fold_name in fold_names:
            part = diagnostic_results[horizon][candidate_name][fold_name].copy()
            part["real_game_id"] = part["game_id"]
            part["game_id"] = part["game_id"].astype(str) + "::" + horizon
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def select_ridge_alpha_scoped(
    diagnostic_results: dict, horizons: Sequence[str], fold_names: Sequence[str],
    registry: tuple = mfs.CANDIDATE_REGISTRY,
) -> dict:
    pooled = {name: _scoped_frame(diagnostic_results, name, horizons, fold_names) for name in mfs.RIDGE_ALPHA_NAMES}
    pooled_primary = {name: mfs._primary_score(pooled[name]) for name in mfs.RIDGE_ALPHA_NAMES}
    best_alpha_name = min(mfs.RIDGE_ALPHA_NAMES, key=lambda n: pooled_primary[n])
    alpha_value = {c.name: c.hyperparameters["alpha"] for c in registry if c.family == "RIDGE"}

    per_alpha: dict[str, dict] = {}
    qualifying = {best_alpha_name}
    for name in mfs.RIDGE_ALPHA_NAMES:
        if name == best_alpha_name:
            per_alpha[name] = {
                "alpha": alpha_value[name], "pooled_primary_score": pooled_primary[name],
                "mean_delta_vs_best": 0.0, "se_delta_vs_best": 0.0, "within_one_se_of_best": True, "is_best": True,
            }
            continue
        mean_delta, se_delta, n = mfs.paired_delta_se(pooled[name], pooled[best_alpha_name])
        within = bool(mean_delta <= se_delta)
        if within:
            qualifying.add(name)
        per_alpha[name] = {
            "alpha": alpha_value[name], "pooled_primary_score": pooled_primary[name],
            "mean_delta_vs_best": mean_delta, "se_delta_vs_best": se_delta,
            "within_one_se_of_best": within, "is_best": False, "n": n,
        }
    selected_name = max(qualifying, key=lambda n: alpha_value[n])
    for name in mfs.RIDGE_ALPHA_NAMES:
        per_alpha[name]["selected"] = name == selected_name
    return {
        "best_alpha_name": best_alpha_name, "selected_alpha_name": selected_name,
        "selected_alpha_value": alpha_value[selected_name], "per_alpha": per_alpha,
    }


_COMPLEXITY_RANK = mfs._COMPLEXITY_RANK


def select_family_scoped(
    diagnostic_results: dict, horizons: Sequence[str], fold_names: Sequence[str], *, ridge_selected_name: str,
) -> dict:
    family_representative = {"RIDGE": ridge_selected_name, "HUBER": "HUBER_FIXED", "HGBR": "HGBR_INCUMBENT"}
    valid_families_in_order = ["RIDGE", "HUBER", "HGBR"]

    pooled = {
        fam: _scoped_frame(diagnostic_results, family_representative[fam], horizons, fold_names)
        for fam in valid_families_in_order
    }
    pooled_primary = {fam: mfs._primary_score(pooled[fam]) for fam in valid_families_in_order}

    best = min(valid_families_in_order, key=lambda f: pooled_primary[f])

    one_se_set = set()
    comparisons_vs_best: dict[str, dict] = {}
    for fam in valid_families_in_order:
        if fam == best:
            one_se_set.add(fam)
            continue
        mean_delta, se_delta, n = mfs.paired_delta_se(pooled[fam], pooled[best])
        comparisons_vs_best[fam] = {"mean_delta": mean_delta, "se_delta": se_delta, "n": n}
        if mean_delta <= se_delta:
            one_se_set.add(fam)

    tentative = min(one_se_set, key=lambda f: _COMPLEXITY_RANK[f])
    lowest_complexity = min(valid_families_in_order, key=lambda f: _COMPLEXITY_RANK[f])
    lower_family_checks: dict[str, dict] = {}
    if tentative == lowest_complexity:
        selected, reason = tentative, "TENTATIVE_IS_LOWEST_COMPLEXITY"
    else:
        lower_families = sorted(
            (f for f in valid_families_in_order if _COMPLEXITY_RANK[f] < _COMPLEXITY_RANK[tentative]),
            key=lambda f: _COMPLEXITY_RANK[f],
        )
        all_passed, first_failed = True, None
        for s in lower_families:
            wins = sum(
                1 for fold_name in fold_names
                for horizon in horizons
                if mfs._primary_score(diagnostic_results[horizon][family_representative[tentative]][fold_name])
                < mfs._primary_score(diagnostic_results[horizon][family_representative[s]][fold_name])
            )
            of = len(fold_names) * len(horizons)
            passed = wins >= (of // 2 + 1) if of else False
            lower_family_checks[s] = {"wins": wins, "of": of, "passed": passed}
            if not passed:
                all_passed = False
                if first_failed is None:
                    first_failed = s
        if all_passed:
            selected, reason = tentative, "TENTATIVE_REPRODUCIBLY_BEAT_ALL_LOWER_FAMILIES"
        else:
            selected, reason = first_failed, "TENTATIVE_FAILED_MAJORITY_AGAINST_LOWEST_UNBEATEN_LOWER_FAMILY"

    return {
        "family_representative": family_representative,
        "valid_families_in_order": valid_families_in_order,
        "best": best,
        "pooled_primary": pooled_primary,
        "one_se_set": sorted(one_se_set, key=lambda f: _COMPLEXITY_RANK[f]),
        "comparisons_vs_best": comparisons_vs_best,
        "tentative": tentative,
        "lower_family_checks": lower_family_checks,
        "selected_family": selected,
        "selection_reason": reason,
    }


def evaluate_scope(diagnostic_results: dict, scope: RobustnessScope) -> dict:
    ridge_result = select_ridge_alpha_scoped(diagnostic_results, scope.horizons, scope.folds)
    family_result = select_family_scoped(
        diagnostic_results, scope.horizons, scope.folds, ridge_selected_name=ridge_result["selected_alpha_name"],
    )
    return {
        "scope": scope.name, "horizons": list(scope.horizons), "folds": list(scope.folds),
        "description": scope.description,
        "best_pooled_score_candidate": min(family_result["pooled_primary"], key=lambda f: family_result["pooled_primary"][f]),
        "one_se_family_set": family_result["one_se_set"],
        "selected_family": family_result["selected_family"],
        "selection_reason": family_result["selection_reason"],
        "selected_ridge_alpha": ridge_result["selected_alpha_name"],
        "ridge_alpha_result": ridge_result,
        "family_result": family_result,
    }


def run_all_scopes(diagnostic_results: dict) -> dict[str, dict]:
    return {scope.name: evaluate_scope(diagnostic_results, scope) for scope in ROBUSTNESS_SCOPES}


# ===========================================================================
# Section 4 -- game_id-cluster bootstrap stability. Does NOT refit inside the
# bootstrap: reuses the Section-2 diagnostic per-game predictions and
# resamples GAMES (not rows) with replacement, applying the frozen
# selection formulas (numpy reimplementation, verified byte-identical to
# ``mfs._primary_score`` / ``mfs.paired_delta_se`` on the un-resampled data
# before use -- see ``_assert_numpy_formulas_match_reference``) to every
# replicate. The bootstrap scope is S3 (TUE+FRI pooled, all 3 folds) --
# the same "primary" pooled population Section 5's gate thresholds are
# written against.
# ===========================================================================
def _rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    mask = np.isfinite(d)
    return float(np.sqrt(np.mean(d[mask] ** 2))) if mask.any() else float("nan")


def _primary_score_np(actual_margin, pred_margin, actual_total, pred_total) -> float:
    return 0.5 * (_rmse_np(actual_margin, pred_margin) + _rmse_np(actual_total, pred_total))


def _paired_delta_se_np(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float, int]:
    delta = loss_a - loss_b
    n = len(delta)
    mean_delta = float(delta.mean())
    se_delta = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean_delta, se_delta, n


def _row_loss_np(actual_margin, pred_margin, actual_total, pred_total) -> np.ndarray:
    return 0.5 * ((actual_margin - pred_margin) ** 2 + (actual_total - pred_total) ** 2)


def build_bootstrap_matrix(diagnostic_results: dict, scope: RobustnessScope) -> dict:
    """One aligned, row-ordered array set per candidate over ``scope``,
    keyed by a shared ``real_game_id`` array (a real game_id appears twice
    -- once per horizon -- when eligible for both TUE and FRI; both rows
    belong to the SAME cluster). Row order and length are identical across
    every candidate (all six candidates share the exact same fold/horizon
    population by construction), so a single index array resamples all
    candidates in alignment."""
    frames = {
        name: _scoped_frame(diagnostic_results, name, scope.horizons, scope.folds)
        for name in (c.name for c in mfs.CANDIDATE_REGISTRY)
    }
    ref_name = mfs.CANDIDATE_REGISTRY[0].name
    ref = frames[ref_name].sort_values(["fold", "real_game_id", "horizon"], kind="stable").reset_index(drop=True)
    row_order = ref[["fold", "real_game_id", "horizon"]]
    for name, frame in frames.items():
        aligned = frame.set_index(["fold", "real_game_id", "horizon"]).loc[
            list(row_order.itertuples(index=False, name=None))
        ]
        frames[name] = aligned.reset_index()

    actual_margin = frames[ref_name]["home_margin"].to_numpy(float)
    actual_total = frames[ref_name]["total_points"].to_numpy(float)
    real_game_id = frames[ref_name]["real_game_id"].to_numpy(object)
    fold_horizon = (frames[ref_name]["fold"].astype(str) + "::" + frames[ref_name]["horizon"].astype(str)).to_numpy(object)
    pred_margin = {name: f["pred_margin"].to_numpy(float) for name, f in frames.items()}
    pred_total = {name: f["pred_total"].to_numpy(float) for name, f in frames.items()}
    return {
        "actual_margin": actual_margin, "actual_total": actual_total, "real_game_id": real_game_id,
        "fold_horizon": fold_horizon, "pred_margin": pred_margin, "pred_total": pred_total,
    }


def _assert_numpy_formulas_match_reference(diagnostic_results: dict, scope: RobustnessScope, bm: dict) -> None:
    """Correctness gate: the fast numpy reimplementation used inside the
    10,000-replicate bootstrap loop must reproduce ``mfs._primary_score`` /
    ``mfs.paired_delta_se`` EXACTLY (to float tolerance) on the un-resampled
    (identity-index) data before it is trusted for any replicate."""
    name_a, name_b = "RIDGE_ALPHA_100", "HGBR_INCUMBENT"
    ref_frame_a = _scoped_frame(diagnostic_results, name_a, scope.horizons, scope.folds)
    ref_score_a = mfs._primary_score(ref_frame_a)
    np_score_a = _primary_score_np(bm["actual_margin"], bm["pred_margin"][name_a], bm["actual_total"], bm["pred_total"][name_a])
    if not np.isclose(ref_score_a, np_score_a, rtol=1e-9, atol=1e-9):
        raise CertificationGateFailure(f"BOOTSTRAP_FORMULA_DRIFT_PRIMARY_SCORE: ref={ref_score_a} np={np_score_a}")

    ref_frame_b = _scoped_frame(diagnostic_results, name_b, scope.horizons, scope.folds)
    ref_mean, ref_se, ref_n = mfs.paired_delta_se(ref_frame_a, ref_frame_b)
    loss_a = _row_loss_np(bm["actual_margin"], bm["pred_margin"][name_a], bm["actual_total"], bm["pred_total"][name_a])
    loss_b = _row_loss_np(bm["actual_margin"], bm["pred_margin"][name_b], bm["actual_total"], bm["pred_total"][name_b])
    np_mean, np_se, np_n = _paired_delta_se_np(loss_a, loss_b)
    if ref_n != np_n or not np.isclose(ref_mean, np_mean, rtol=1e-9, atol=1e-9) or not np.isclose(ref_se, np_se, rtol=1e-9, atol=1e-9):
        raise CertificationGateFailure(
            f"BOOTSTRAP_FORMULA_DRIFT_PAIRED_DELTA_SE: ref=({ref_mean},{ref_se},{ref_n}) np=({np_mean},{np_se},{np_n})"
        )


def _select_replicate(bm: dict, idx: np.ndarray) -> dict:
    """Applies the frozen Ridge-alpha-then-family selection procedure
    (Steps identical to ``mfs.select_ridge_alpha`` / ``mfs.select_family``)
    to one resampled row-index array, using the numpy primitives."""
    actual_margin = bm["actual_margin"][idx]
    actual_total = bm["actual_total"][idx]

    ridge_names = mfs.RIDGE_ALPHA_NAMES
    ridge_scores = {
        name: _primary_score_np(actual_margin, bm["pred_margin"][name][idx], actual_total, bm["pred_total"][name][idx])
        for name in ridge_names
    }
    best_alpha = min(ridge_names, key=lambda n: ridge_scores[n])
    alpha_value = {c.name: c.hyperparameters["alpha"] for c in mfs.CANDIDATE_REGISTRY if c.family == "RIDGE"}
    loss_best = _row_loss_np(actual_margin, bm["pred_margin"][best_alpha][idx], actual_total, bm["pred_total"][best_alpha][idx])
    qualifying = {best_alpha}
    for name in ridge_names:
        if name == best_alpha:
            continue
        loss_name = _row_loss_np(actual_margin, bm["pred_margin"][name][idx], actual_total, bm["pred_total"][name][idx])
        mean_delta, se_delta, _ = _paired_delta_se_np(loss_name, loss_best)
        if mean_delta <= se_delta:
            qualifying.add(name)
    selected_ridge_alpha = max(qualifying, key=lambda n: alpha_value[n])

    family_representative = {"RIDGE": selected_ridge_alpha, "HUBER": "HUBER_FIXED", "HGBR": "HGBR_INCUMBENT"}
    fam_scores = {
        fam: _primary_score_np(actual_margin, bm["pred_margin"][rep][idx], actual_total, bm["pred_total"][rep][idx])
        for fam, rep in family_representative.items()
    }
    best_family = min(fam_scores, key=lambda f: fam_scores[f])
    loss_best_fam = _row_loss_np(
        actual_margin, bm["pred_margin"][family_representative[best_family]][idx],
        actual_total, bm["pred_total"][family_representative[best_family]][idx],
    )
    one_se_set = {best_family}
    for fam in ("RIDGE", "HUBER", "HGBR"):
        if fam == best_family:
            continue
        loss_fam = _row_loss_np(
            actual_margin, bm["pred_margin"][family_representative[fam]][idx],
            actual_total, bm["pred_total"][family_representative[fam]][idx],
        )
        mean_delta, se_delta, _ = _paired_delta_se_np(loss_fam, loss_best_fam)
        if mean_delta <= se_delta:
            one_se_set.add(fam)
    tentative = min(one_se_set, key=lambda f: _COMPLEXITY_RANK[f])
    lowest = min(("RIDGE", "HUBER", "HGBR"), key=lambda f: _COMPLEXITY_RANK[f])
    if tentative == lowest:
        selected_family = tentative
    else:
        # Rare branch (RIDGE fell outside the one-SE set this replicate):
        # exact per-(fold,horizon)-cell win counting, majority rule -- same
        # generalization as select_family_scoped's Step D, computed on THIS
        # replicate's resampled rows only (never a pooled-score shortcut).
        lower_families = sorted(
            (f for f in ("RIDGE", "HUBER", "HGBR") if _COMPLEXITY_RANK[f] < _COMPLEXITY_RANK[tentative]),
            key=lambda f: _COMPLEXITY_RANK[f],
        )
        cells = np.unique(bm["fold_horizon"][idx])
        selected_family = tentative
        for s in lower_families:
            wins = 0
            for cell in cells:
                cell_mask = bm["fold_horizon"][idx] == cell
                t_rep, s_rep = family_representative[tentative], family_representative[s]
                t_score = _primary_score_np(
                    actual_margin[cell_mask], bm["pred_margin"][t_rep][idx][cell_mask],
                    actual_total[cell_mask], bm["pred_total"][t_rep][idx][cell_mask],
                )
                s_score = _primary_score_np(
                    actual_margin[cell_mask], bm["pred_margin"][s_rep][idx][cell_mask],
                    actual_total[cell_mask], bm["pred_total"][s_rep][idx][cell_mask],
                )
                if t_score < s_score:
                    wins += 1
            if wins < (len(cells) // 2 + 1):
                selected_family = s
                break

    return {
        "selected_ridge_alpha": selected_ridge_alpha,
        "ridge_one_se_set": qualifying,
        "selected_family": selected_family,
        "family_one_se_set": one_se_set,
        "final_selected_candidate": family_representative[selected_family] if selected_family == "RIDGE" else family_representative[selected_family],
    }


def run_bootstrap_stability(
    diagnostic_results: dict, *, scope: RobustnessScope = SCOPE_BY_NAME["S3"], n_reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    bm = build_bootstrap_matrix(diagnostic_results, scope)
    _assert_numpy_formulas_match_reference(diagnostic_results, scope, bm)

    unique_game_ids = np.unique(bm["real_game_id"])
    game_id_to_rows: dict[str, np.ndarray] = {
        gid: np.flatnonzero(bm["real_game_id"] == gid) for gid in unique_game_ids
    }
    n_unique = len(unique_game_ids)
    rng = np.random.default_rng(seed)

    family_counts = {"RIDGE": 0, "HUBER": 0, "HGBR": 0}
    ridge_alpha_counts = {n: 0 for n in mfs.RIDGE_ALPHA_NAMES}
    ridge_alpha_100_selected_count = 0
    ridge_family_in_one_se_count = 0
    hgbr_in_one_se_count = 0

    for _ in range(n_reps):
        sampled_ids = unique_game_ids[rng.integers(0, n_unique, size=n_unique)]
        idx = np.concatenate([game_id_to_rows[gid] for gid in sampled_ids])
        result = _select_replicate(bm, idx)
        family_counts[result["selected_family"]] += 1
        ridge_alpha_counts[result["selected_ridge_alpha"]] += 1
        if result["selected_family"] == "RIDGE" and result["selected_ridge_alpha"] == "RIDGE_ALPHA_100":
            ridge_alpha_100_selected_count += 1
        if "RIDGE" in result["family_one_se_set"]:
            ridge_family_in_one_se_count += 1
        if "HGBR" in result["family_one_se_set"]:
            hgbr_in_one_se_count += 1

    return {
        "scope": scope.name, "n_reps": n_reps, "seed": seed, "n_unique_game_id_clusters": n_unique,
        "family_selected_frequency": {k: v / n_reps for k, v in family_counts.items()},
        "ridge_alpha_selected_frequency": {k: v / n_reps for k, v in ridge_alpha_counts.items()},
        "probability_ridge_alpha_100_selected": ridge_alpha_100_selected_count / n_reps,
        "probability_ridge_family_in_one_se_set": ridge_family_in_one_se_count / n_reps,
        "probability_hgbr_in_one_se_set": hgbr_in_one_se_count / n_reps,
    }


# ===========================================================================
# Section 5 -- MODEL_FAMILY_STABILITY gate. Thresholds are NOT weakened.
# ===========================================================================
def evaluate_model_family_stability_gate(scope_results: dict[str, dict], bootstrap: dict) -> dict:
    checks: dict[str, dict] = {}

    checks["1_ridge_selected_s1_s2_s3"] = {
        "pass": all(scope_results[s]["selected_family"] == "RIDGE" for s in ("S1", "S2", "S3")),
        "detail": {s: scope_results[s]["selected_family"] for s in ("S1", "S2", "S3")},
    }
    checks["2_ridge_in_one_se_all_scopes"] = {
        "pass": all("RIDGE" in scope_results[s]["one_se_family_set"] for s in SCOPE_BY_NAME),
        "detail": {s: scope_results[s]["one_se_family_set"] for s in SCOPE_BY_NAME},
    }
    checks["3_hgbr_never_selected"] = {
        "pass": all(scope_results[s]["selected_family"] != "HGBR" for s in SCOPE_BY_NAME),
        "detail": {s: scope_results[s]["selected_family"] for s in SCOPE_BY_NAME},
    }
    checks["4_ridge_alpha_100_selected_s1_s2_s3"] = {
        "pass": all(scope_results[s]["selected_ridge_alpha"] == "RIDGE_ALPHA_100" for s in ("S1", "S2", "S3")),
        "detail": {s: scope_results[s]["selected_ridge_alpha"] for s in ("S1", "S2", "S3")},
    }
    checks["5_bootstrap_p_ridge_family_ge_0_80"] = {
        "pass": bootstrap["family_selected_frequency"]["RIDGE"] >= 0.80,
        "value": bootstrap["family_selected_frequency"]["RIDGE"],
    }
    checks["6_bootstrap_p_ridge_in_one_se_ge_0_90"] = {
        "pass": bootstrap["probability_ridge_family_in_one_se_set"] >= 0.90,
        "value": bootstrap["probability_ridge_family_in_one_se_set"],
    }
    checks["7_bootstrap_p_hgbr_le_0_05"] = {
        "pass": bootstrap["family_selected_frequency"]["HGBR"] <= 0.05,
        "value": bootstrap["family_selected_frequency"]["HGBR"],
    }

    all_pass = all(c["pass"] for c in checks.values())
    if all_pass:
        status = "MET_STRONGLY"
    else:
        # MODERATE evidence path (MET): RIDGE never worse than one-SE-competitive
        # and never dominated by HGBR, even if a strong-gate criterion fails.
        moderate_ok = (
            all("RIDGE" in scope_results[s]["one_se_family_set"] for s in SCOPE_BY_NAME)
            and all(scope_results[s]["selected_family"] != "HGBR" for s in SCOPE_BY_NAME)
        )
        status = "MET" if moderate_ok else "MIXED"

    return {"status": status, "checks": checks}


# ===========================================================================
# Section 7 -- calibration-improvement robustness. Reuses the ALREADY-
# CERTIFIED Fix-8 2025 raw/calibrated ledgers verbatim (no refit, no
# recalibration). Scoring mirrors
# run_fix8_official_oof_calibration.phase9_readiness_and_2025_audit EXACTLY
# (same join, same CALIBRATED-status + non-push filter, same
# binary_log_loss/brier/equal_mass_ece calls) so the point estimates match
# the certified per-stream audit_2025 block bit-for-bit.
# ===========================================================================
def load_stream_paired_arrays(stream: str, artifact_root) -> dict:
    stem = _STREAM_FILE_STEM[stream]
    ledger = pd.read_parquet(artifact_root / f"{stem}_calibration_ledger.parquet")
    raw = pd.read_parquet(artifact_root / f"{stem}_raw_probabilities.parquet")
    season_num = pd.to_numeric(ledger["season"], errors="coerce")
    ledger_2025 = ledger[season_num == 2025]
    joined_2025 = ledger_2025.merge(raw[["game_id", "binary_target"]], on="game_id", how="left")
    scored = joined_2025[(joined_2025["calibration_status"] == "CALIBRATED") & joined_2025["binary_target"].notna()]
    n_pushes_excluded = int(
        joined_2025[(joined_2025["calibration_status"] == "CALIBRATED") & joined_2025["binary_target"].isna()].shape[0]
    )
    y = pd.to_numeric(scored["binary_target"], errors="coerce").to_numpy(float)
    p_raw = scored["raw_conditional_upper_probability"].to_numpy(float)
    p_cal = scored["calibrated_conditional_upper_probability"].to_numpy(float)
    game_ids = scored["game_id"].astype(str).to_numpy(object)
    return {"stream": stream, "y": y, "p_raw": p_raw, "p_cal": p_cal, "game_id": game_ids, "n_pushes_excluded": n_pushes_excluded}


def stream_point_estimates(arrays: dict) -> dict:
    y, p_raw, p_cal = arrays["y"], arrays["p_raw"], arrays["p_cal"]
    ece_raw, ece_cal = equal_mass_ece(y, p_raw), equal_mass_ece(y, p_cal)
    return {
        "stream": arrays["stream"], "n_scored": int(len(y)), "n_pushes_excluded": arrays["n_pushes_excluded"],
        "raw_log_loss": binary_log_loss(y, p_raw), "calibrated_log_loss": binary_log_loss(y, p_cal),
        "raw_brier": brier(y, p_raw), "calibrated_brier": brier(y, p_cal),
        "raw_ece": ece_raw if ece_raw == NOT_ESTIMABLE else float(ece_raw),
        "calibrated_ece": ece_cal if ece_cal == NOT_ESTIMABLE else float(ece_cal),
        "log_loss_delta": binary_log_loss(y, p_cal) - binary_log_loss(y, p_raw),
        "brier_delta": brier(y, p_cal) - brier(y, p_raw),
    }


def bootstrap_calibration_deltas(arrays: dict, *, n_reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """10,000 game_id-cluster resamples (a game_id is never split across
    clusters; within one stream each game_id contributes exactly one row,
    so this degenerates to a standard per-observation bootstrap here, but
    is implemented generally via the same game_id-cluster machinery used in
    Section 4)."""
    y, p_raw, p_cal, game_ids = arrays["y"], arrays["p_raw"], arrays["p_cal"], arrays["game_id"]
    unique_ids = np.unique(game_ids)
    rows_by_id = {gid: np.flatnonzero(game_ids == gid) for gid in unique_ids}
    n_unique = len(unique_ids)
    rng = np.random.default_rng(seed)

    log_loss_deltas = np.empty(n_reps)
    brier_deltas = np.empty(n_reps)
    for i in range(n_reps):
        sampled_ids = unique_ids[rng.integers(0, n_unique, size=n_unique)]
        idx = np.concatenate([rows_by_id[gid] for gid in sampled_ids])
        yb, pr, pc = y[idx], p_raw[idx], p_cal[idx]
        log_loss_deltas[i] = binary_log_loss(yb, pc) - binary_log_loss(yb, pr)
        brier_deltas[i] = brier(yb, pc) - brier(yb, pr)

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    ll_lo, ll_hi = _ci(log_loss_deltas)
    br_lo, br_hi = _ci(brier_deltas)
    return {
        "n_reps": n_reps, "seed": seed, "n_unique_game_id_clusters": n_unique,
        "log_loss_delta_ci95": [ll_lo, ll_hi], "log_loss_delta_ci95_below_zero": bool(ll_hi < 0),
        "brier_delta_ci95": [br_lo, br_hi], "brier_delta_ci95_below_zero": bool(br_hi < 0),
    }


def pooled_arrays(streams: Sequence[dict]) -> dict:
    return {
        "stream": "+".join(s["stream"] for s in streams),
        "y": np.concatenate([s["y"] for s in streams]),
        "p_raw": np.concatenate([s["p_raw"] for s in streams]),
        "p_cal": np.concatenate([s["p_cal"] for s in streams]),
        "game_id": np.concatenate([s["stream"] + "::" + s["game_id"] for s in streams]),
        "n_pushes_excluded": sum(s["n_pushes_excluded"] for s in streams),
    }


# ===========================================================================
# Section 8 -- CALIBRATION_IMPROVES_RAW_PROBABILITIES gate.
# ===========================================================================
def evaluate_calibration_gate(per_stream_points: dict[str, dict], pooled_all_four_bootstrap: dict) -> dict:
    checks: dict[str, dict] = {}
    checks["1_log_loss_improves_all_4"] = {
        "pass": all(per_stream_points[s]["log_loss_delta"] < 0 for s in STREAM_NAMES),
        "detail": {s: per_stream_points[s]["log_loss_delta"] for s in STREAM_NAMES},
    }
    checks["2_brier_improves_all_4"] = {
        "pass": all(per_stream_points[s]["brier_delta"] < 0 for s in STREAM_NAMES),
        "detail": {s: per_stream_points[s]["brier_delta"] for s in STREAM_NAMES},
    }
    checks["3_ece_improves_all_4"] = {
        "pass": all(
            per_stream_points[s]["calibrated_ece"] != NOT_ESTIMABLE
            and per_stream_points[s]["raw_ece"] != NOT_ESTIMABLE
            and per_stream_points[s]["calibrated_ece"] < per_stream_points[s]["raw_ece"]
            for s in STREAM_NAMES
        ),
        "detail": {
            s: {"raw_ece": per_stream_points[s]["raw_ece"], "calibrated_ece": per_stream_points[s]["calibrated_ece"]}
            for s in STREAM_NAMES
        },
    }
    checks["4_all_four_pooled_log_loss_ci_below_zero"] = {
        "pass": pooled_all_four_bootstrap["log_loss_delta_ci95_below_zero"],
        "value": pooled_all_four_bootstrap["log_loss_delta_ci95"],
    }
    checks["5_all_four_pooled_brier_ci_below_zero"] = {
        "pass": pooled_all_four_bootstrap["brier_delta_ci95_below_zero"],
        "value": pooled_all_four_bootstrap["brier_delta_ci95"],
    }
    checks["6_no_materially_adverse_stream"] = {
        "pass": all(per_stream_points[s]["log_loss_delta"] < 0.01 for s in STREAM_NAMES),
        "detail": {s: per_stream_points[s]["log_loss_delta"] for s in STREAM_NAMES},
    }

    all_pass = all(c["pass"] for c in checks.values())
    if all_pass:
        status = "MET_STRONGLY"
    else:
        moderate_ok = (
            sum(1 for s in STREAM_NAMES if per_stream_points[s]["log_loss_delta"] < 0) >= 3
            and sum(1 for s in STREAM_NAMES if per_stream_points[s]["brier_delta"] < 0) >= 3
        )
        status = "MET" if moderate_ok else "MIXED"
    return {"status": status, "checks": checks}


def evidence_label(point_deltas_negative: int, total: int, pooled_ci_below_zero: bool) -> str:
    if point_deltas_negative == total and pooled_ci_below_zero:
        return "STRONG"
    if point_deltas_negative >= total - 1:
        return "SUGGESTIVE"
    return "MIXED"


# ===========================================================================
# Section 6 -- 2024/2025 family robustness, DIAGNOSTIC ONLY. Never used to
# alter the Section-5 verdict.
#
# 2024 reuses the ALREADY-CERTIFIED ``locked_post_exposure_2024_audit`` block
# from ``outputs/fix7_model_family_selection_summary.json`` verbatim (no new
# fit). 2025 requires one new fit per finalist per horizon (train<=2024,
# predict==2025) -- this deliberately does NOT go through
# ``mfs.fit_predict_fold``/``fd.assert_fold_seasons_allowed``, whose
# ``SELECTION_MAX_SEASON=2024`` firewall exists to keep 2025 out of the
# SELECTION procedure. This function performs no selection: it never calls
# ``select_ridge_alpha``/``select_family``, its output never feeds the
# Section-5 gate, and every record it writes is labeled
# ``POST_EXPOSURE_DIAGNOSTIC_ONLY``. Fitting mechanics (Ridge/Huber pipeline,
# HGBR JointScoreModel) are otherwise byte-identical to
# ``mfs.fit_predict_fold``.
# ===========================================================================
_DIAGNOSTIC_2025_FINALISTS = ("RIDGE_ALPHA_100", "HUBER_FIXED", "HGBR_INCUMBENT")


def _fit_predict_2025_diagnostic(matrix: pd.DataFrame, candidate: "mfs.CandidateSpec") -> pd.DataFrame:
    train = matrix[matrix["season"] <= 2024]
    validate = matrix[matrix["season"] == 2025]
    feature_columns = list(ohf.ELO_FEATURE_COLUMNS)
    if candidate.family == "HGBR":
        model = fd.JointScoreModel(numeric_features=feature_columns, categorical_features=(), config=fd.MODEL_CONFIG)
        model.fit(train)
        pred_margin, pred_total = model.predict_means(validate)
    else:
        pipe_cls = mfs._build_pipeline
        margin_pipe = pipe_cls(candidate.family, candidate.hyperparameters).fit(
            train[feature_columns], train["home_margin"].to_numpy(float)
        )
        total_pipe = pipe_cls(candidate.family, candidate.hyperparameters).fit(
            train[feature_columns], train["total_points"].to_numpy(float)
        )
        pred_margin, pred_total = margin_pipe.predict(validate[feature_columns]), total_pipe.predict(validate[feature_columns])
    out = validate[["game_id", "season", "week", "home_margin", "total_points"]].copy()
    out["pred_margin"], out["pred_total"] = pred_margin, pred_total
    out["candidate"] = candidate.name
    return out


def run_2025_diagnostic_family_comparison(games: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    per_horizon: dict[str, dict] = {}
    for horizon in he.HORIZONS:
        matrix = ohf.build_official_horizon_matrix(games, horizon, ledger)
        per_finalist = {}
        for name in _DIAGNOSTIC_2025_FINALISTS:
            candidate = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == name)
            preds = _fit_predict_2025_diagnostic(matrix, candidate)
            report = mfs._regression_report(preds)
            per_finalist[name] = {
                "candidate_name": name, "n": len(preds), "primary_score": report["primary_score"],
                "margin_rmse": report["margin"]["rmse"], "total_rmse": report["total"]["rmse"],
            }
        ridge_score = per_finalist["RIDGE_ALPHA_100"]["primary_score"]
        huber_score = per_finalist["HUBER_FIXED"]["primary_score"]
        hgbr_score = per_finalist["HGBR_INCUMBENT"]["primary_score"]
        per_horizon[horizon] = {
            "per_finalist": per_finalist,
            "ridge_competitive_with_huber": bool(abs(ridge_score - huber_score) <= 0.10 * min(ridge_score, huber_score)),
            "ridge_clearly_ahead_of_hgbr": bool(ridge_score < hgbr_score),
            "ridge_vs_huber_delta": ridge_score - huber_score,
            "ridge_vs_hgbr_delta": ridge_score - hgbr_score,
        }
    return {
        "role": "POST_EXPOSURE_DIAGNOSTIC_ONLY",
        "used_for_selection": False,
        "used_for_estimator_adjustment": False,
        "alters_section5_verdict": False,
        "per_horizon": per_horizon,
    }


# ===========================================================================
# Sections 9-10 -- absolute probability quality + sportsbook comparison.
# Reuses the certified Fix-8 2025 residual ledgers and market-consensus
# files verbatim -- no refit, no rerun of feature research. Sign
# convention verified numerically against actual outcomes before use (see
# scripts/run_final_review_market_comparison.py): ATS ``consensus_line`` is
# the home spread (implied_margin = -consensus_line); TOTAL
# ``consensus_line`` is the total line directly (implied_total =
# consensus_line).
# ===========================================================================
def absolute_quality_metrics(arrays: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y, p_cal = arrays["y"], arrays["p_cal"]
    p_clipped = np.clip(p_cal, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

    if len(np.unique(y)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y, p_cal))

    try:
        slope_model = LogisticRegression(C=1e6, solver="lbfgs")
        slope_model.fit(logit_p, y.astype(int))
        cal_slope = float(slope_model.coef_[0][0])
        cal_intercept = float(slope_model.intercept_[0])
    except Exception:
        cal_slope, cal_intercept = None, None

    sharpness = float(np.mean(np.abs(p_cal - 0.5)))

    if auc is None:
        label = "INCONCLUSIVE"
    elif auc < 0.52:
        label = "WEAK" if auc >= 0.50 else "INCONCLUSIVE"
    elif auc < 0.55:
        label = "MODERATE"
    elif auc < 0.60:
        label = "GOOD"
    else:
        label = "STRONG"

    return {
        "n": int(len(y)), "auc": auc, "calibration_slope": cal_slope, "calibration_intercept": cal_intercept,
        "sharpness_mean_abs_dev_from_half": sharpness, "discrimination_label": label,
    }


def market_implied_margin(consensus_line: np.ndarray) -> np.ndarray:
    return -np.asarray(consensus_line, dtype=float)


def market_implied_total(consensus_line: np.ndarray) -> np.ndarray:
    return np.asarray(consensus_line, dtype=float)


def point_forecast_vs_market(residual_ledger: pd.DataFrame, market_consensus: pd.DataFrame, *, market: str) -> dict:
    season_num = pd.to_numeric(residual_ledger["season"], errors="coerce")
    rl2025 = residual_ledger[season_num == 2025]
    merged = rl2025.merge(market_consensus[["game_id", "consensus_line"]], on="game_id", how="inner")
    if market == "ATS":
        model_pred, actual, market_implied = merged["predicted_margin"], merged["actual_margin"], market_implied_margin(merged["consensus_line"])
    else:
        model_pred, actual, market_implied = merged["predicted_total"], merged["actual_total"], market_implied_total(merged["consensus_line"])
    model_rmse = float(np.sqrt(np.mean((model_pred.to_numpy(float) - actual.to_numpy(float)) ** 2)))
    market_rmse = float(np.sqrt(np.mean((market_implied - actual.to_numpy(float)) ** 2)))
    return {"n": int(len(merged)), "model_rmse": model_rmse, "market_rmse": market_rmse, "model_minus_market_rmse": model_rmse - market_rmse}


def probability_vs_market_paired(
    ledger_2025_calibrated: pd.DataFrame, market_consensus: pd.DataFrame, *, n_reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED,
) -> dict:
    merged = ledger_2025_calibrated.merge(market_consensus[["game_id", "consensus_novig_probability"]], on="game_id", how="inner")
    y = pd.to_numeric(merged["binary_target"], errors="coerce").to_numpy(float)
    p_model = merged["calibrated_conditional_upper_probability"].to_numpy(float)
    p_market = merged["consensus_novig_probability"].to_numpy(float)
    game_ids = merged["game_id"].astype(str).to_numpy(object)

    log_loss_delta = binary_log_loss(y, p_model) - binary_log_loss(y, p_market)
    brier_delta = brier(y, p_model) - brier(y, p_market)

    unique_ids = np.unique(game_ids)
    rows_by_id = {gid: np.flatnonzero(game_ids == gid) for gid in unique_ids}
    n_unique = len(unique_ids)
    rng = np.random.default_rng(seed)
    ll_deltas, br_deltas = np.empty(n_reps), np.empty(n_reps)
    for i in range(n_reps):
        sampled = unique_ids[rng.integers(0, n_unique, size=n_unique)]
        idx = np.concatenate([rows_by_id[g] for g in sampled])
        yb, pm, pk = y[idx], p_model[idx], p_market[idx]
        ll_deltas[i] = binary_log_loss(yb, pm) - binary_log_loss(yb, pk)
        br_deltas[i] = brier(yb, pm) - brier(yb, pk)

    return {
        "n": int(len(merged)), "log_loss_delta": log_loss_delta, "brier_delta": brier_delta,
        "log_loss_delta_ci95": [float(np.percentile(ll_deltas, 2.5)), float(np.percentile(ll_deltas, 97.5))],
        "brier_delta_ci95": [float(np.percentile(br_deltas, 2.5)), float(np.percentile(br_deltas, 97.5))],
        "model_beats_market_ci95_entirely_below_zero": bool(np.percentile(ll_deltas, 97.5) < 0),
    }


def load_2024_locked_post_exposure_audit(fix7_summary: dict) -> dict:
    audit = dict(fix7_summary["locked_post_exposure_2024_audit"])
    audit["role"] = "LOCKED_POST_EXPOSURE_DIAGNOSTIC"
    audit["source"] = "outputs/fix7_model_family_selection_summary.json (already-certified, reused verbatim, no new fit)"
    audit.pop("fitted_bundles", None)
    return audit
