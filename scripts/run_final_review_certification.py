"""FINAL REVIEW CERTIFICATION 2026 -- Sections 0-11 orchestrator.

Model-family robustness audit + calibration-improvement robustness for the
frozen ``v2026.1-fix8-certified`` baseline. Performs NO feature selection,
NO change to the production model (RIDGE_ALPHA_100 stays production
regardless of this audit's outcome), and NO recalibration.

Phase order:
  0. Repo gate + certified-hash verification (HARD STOP on mismatch).
  1. Freeze ``final_review_certification_preregistration.json`` BEFORE any
     diagnostic fit.
  2. Diagnostic per-game fits: frozen Fix-7 candidate registry, frozen
     Fix-7 chronological folds (A/B/C), TUE/FRI-horizon-scoped.
  3. Robustness scopes S1-S9 (frozen selection procedure applied to
     predeclared pooled subsets).
  4. Game_id-cluster bootstrap stability (10,000 reps, seed 20260826).
  5. MODEL_FAMILY_STABILITY gate.
  6. 2024/2025 family robustness -- DIAGNOSTIC ONLY, never alters Section 5.
  7-11. Calibration-improvement robustness (reuses certified Fix-8 2025
     ledgers verbatim; never recalibrates).

Never commits, never opens a PR.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.certification import final_review_2026 as cert  # noqa: E402
from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root  # noqa: E402
from nfl_hybrid.selection import feature_deduction_2026 as fd  # noqa: E402
from nfl_hybrid.selection import model_family_selection_2026 as mfs  # noqa: E402

OUTPUT_DIR = artifact_root() / "final-review-certification-2026"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostic_fits"


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise cert.CertificationGateFailure(f"command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout.rstrip("\n")


def phase0_gate() -> dict:
    branch = _run(["git", "branch", "--show-current"])
    head_sha = _run(["git", "rev-parse", "HEAD"])
    tag_commit_sha = _run(["git", "rev-parse", f"{cert.CERTIFIED_BASELINE_TAG}^{{commit}}"])
    dirty_paths = [ln[3:].strip() for ln in _run(["git", "status", "--short"]).splitlines() if ln.strip()]
    gate = cert.verify_repo_gate(branch=branch, head_sha=head_sha, tag_commit_sha=tag_commit_sha, dirty_paths=dirty_paths)

    fix71_summary = json.loads((REPO_ROOT / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text())
    fix8_prereg = json.loads((REPO_ROOT / "outputs" / "fix8_official_oof_calibration_preregistration.json").read_text())
    fix7_summary = json.loads((REPO_ROOT / "outputs" / "fix7_model_family_selection_summary.json").read_text())
    hashes = cert.verify_certified_hashes(fix71_summary, fix8_prereg)
    print(f"Phase 0 OK: gate={gate} hashes={hashes}")
    return {"gate": gate, "hashes": hashes, "fix71_summary": fix71_summary, "fix8_prereg": fix8_prereg, "fix7_summary": fix7_summary}


def phase1_preregistration(phase0: dict) -> dict:
    body = {
        "schema_version": cert.SCHEMA_VERSION,
        "certified_baseline": {
            "tag": cert.CERTIFIED_BASELINE_TAG, "commit": cert.CERTIFIED_BASELINE_SHA,
            "horizon_feature_semantics_hash": cert.CERTIFIED_HORIZON_FEATURE_SEMANTICS_HASH,
            "horizon_membership_ledger_hash": cert.CERTIFIED_HORIZON_MEMBERSHIP_LEDGER_HASH,
            "operational_model_spec_hash": cert.CERTIFIED_OPERATIONAL_MODEL_SPEC_HASH,
            "fix8_preregistration_hash": cert.CERTIFIED_FIX8_PREREGISTRATION_HASH,
        },
        "six_elo_features": list(ohf_elo_columns()),
        "fix7_candidate_registry": [
            {"name": c.name, "family": c.family, "hyperparameters": c.hyperparameters} for c in mfs.CANDIDATE_REGISTRY
        ],
        "fix7_candidate_registry_hash": mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY),
        "operational_tue_fri_semantics_version": "HORIZON_CUTOFF_ASOF_ELO_V2_CARD_SCOPED",
        "robustness_scopes": [
            {"name": s.name, "horizons": list(s.horizons), "folds": list(s.folds), "description": s.description}
            for s in cert.ROBUSTNESS_SCOPES
        ],
        "chronological_folds": [
            {"name": f.name, "train_max_season": f.train_max_season, "validate_season": f.validate_season}
            for f in fd.INNER_FOLDS
        ],
        "bootstrap_method": {
            "resampling_unit": "game_id cluster (a real game_id's TUE and FRI rows are always resampled together)",
            "n_reps": cert.BOOTSTRAP_REPS, "seed": cert.BOOTSTRAP_SEED,
            "applies_to": ["Section 4: model-family bootstrap stability (scope S3)", "Section 7: calibration-improvement bootstrap (per stream and pooled)"],
        },
        "model_stability_success_rules": {
            "MET_STRONGLY": [
                "RIDGE selected by frozen procedure in S1, S2, S3",
                "RIDGE remains inside the one-SE family set in ALL S1-S9",
                "HGBR not selected in ANY S1-S9",
                "RIDGE_ALPHA_100 selected Ridge alpha in S1, S2, S3",
                "bootstrap P(RIDGE family selected) >= 0.80",
                "bootstrap P(RIDGE in one-SE set) >= 0.90",
                "bootstrap P(HGBR selected) <= 0.05",
            ],
            "thresholds_not_weakened": True,
        },
        "calibration_robustness_success_rules": {
            "MET_STRONGLY": [
                "calibrated log-loss point estimate improves in all 4 streams",
                "calibrated Brier point estimate improves in all 4 streams",
                "calibrated ECE improves in all 4 streams",
                "ALL_FOUR pooled 95% CI for calibrated-minus-raw log loss entirely below zero",
                "ALL_FOUR pooled 95% CI for calibrated-minus-raw Brier entirely below zero",
                "no individual stream has a materially adverse point estimate (log-loss delta >= +0.01)",
            ],
        },
        "production_readiness_gates": [
            "single canonical production entrypoint", "certified hashes hard-gated", "DST-aware TUE/FRI due logic tested",
            "preflight passes", "fail-closed behavior tested", "immutable forecast ledger tested",
            "idempotent replay tested", "conflicting rewrite rejected", "result attachment cannot mutate forecasts",
            "prospective reporter works on fixture", "no credential leakage", "historical end-to-end integration test passes",
            "current 2026 preflight passes", "full pytest passes", "git diff --check clean",
        ],
        "prospective_evidence_limitations": (
            "No valid prospective executable-price betting study exists as of this certification. "
            "Sportsbook probability edge and profitability are NOT established from historical/backtest evidence "
            "alone; they require the 2026 prospective shadow evidence ledger (forecast-before-result, "
            "immutable) accumulating a sufficient sample after real 2026 games are played."
        ),
        "packaging_success_contract": (
            "This certification labels each of the 9 FINAL_STATUS_SCORECARD rows using only evidence computed "
            "in this run or already-certified upstream evidence; no status is forced to MET_STRONGLY, and "
            "NOT_ESTABLISHED/NOT_DEMONSTRATED are legitimate, expected terminal statuses for the profitability "
            "and sportsbook-edge rows."
        ),
    }
    prereg_hash = cert._sha256_hex(body)
    body_with_hash = {**body, "final_review_certification_preregistration_hash": prereg_hash}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "final_review_certification_preregistration.json").write_text(
        json.dumps(body_with_hash, indent=2, sort_keys=True, default=str)
    )
    print(f"Phase 1 OK: final_review_certification_preregistration_hash={prereg_hash}")
    return {"body": body, "hash": prereg_hash}


def ohf_elo_columns():
    from nfl_hybrid.evaluation import official_horizon_oof as ohf
    return ohf.ELO_FEATURE_COLUMNS


def _verify_preregistration_unchanged(preregistration: dict) -> None:
    recomputed = cert._sha256_hex(preregistration["body"])
    if recomputed != preregistration["hash"]:
        raise cert.CertificationGateFailure(
            f"PREREGISTRATION_HASH_MUTATED_AFTER_FREEZE: recomputed={recomputed} frozen={preregistration['hash']}"
        )


def phase2_diagnostic_fits(preregistration: dict) -> dict:
    _verify_preregistration_unchanged(preregistration)
    games = cert.load_reg_post_games_through_2025()
    ledger = cert.build_membership_ledger(games)
    result = cert.run_diagnostic_fits(games, ledger)
    print(f"Phase 2 OK: fit_counts={result['fit_counts']} huber_invalid={len(result['huber_invalid_reasons'])}")

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    for horizon, inner_results in result["diagnostic_results"].items():
        frames = []
        for candidate_name, by_fold in inner_results.items():
            for fold_name, df in by_fold.items():
                frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(DIAGNOSTIC_DIR / f"{horizon.lower()}_diagnostic_predictions.parquet")
    return {**result, "games": games, "ledger": ledger}


def phase3_scopes(diagnostic: dict) -> dict:
    scope_results = cert.run_all_scopes(diagnostic["diagnostic_results"])
    for name, r in scope_results.items():
        print(f"Phase 3 [{name}] best={r['best_pooled_score_candidate']} one_se={r['one_se_family_set']} "
              f"selected_family={r['selected_family']} selected_alpha={r['selected_ridge_alpha']}")
    return scope_results


def phase4_bootstrap(diagnostic: dict) -> dict:
    t0 = time.time()
    bootstrap = cert.run_bootstrap_stability(diagnostic["diagnostic_results"])
    print(f"Phase 4 OK in {time.time()-t0:.1f}s: {bootstrap}")
    return bootstrap


def phase5_gate(scope_results: dict, bootstrap: dict) -> dict:
    gate = cert.evaluate_model_family_stability_gate(scope_results, bootstrap)
    print(f"Phase 5 OK: MODEL_FAMILY_STABILITY={gate['status']}")
    return gate


def phase6_2024_2025_diagnostic(phase0: dict, diagnostic: dict) -> dict:
    audit_2024 = cert.load_2024_locked_post_exposure_audit(phase0["fix7_summary"])
    audit_2025 = cert.run_2025_diagnostic_family_comparison(diagnostic["games"], diagnostic["ledger"])
    print(f"Phase 6 OK: 2024 selected_family={audit_2024['selected_family']} overall_pass={audit_2024['overall_pass']}")
    for horizon, r in audit_2025["per_horizon"].items():
        print(f"Phase 6 [2025 {horizon}] ridge_competitive_with_huber={r['ridge_competitive_with_huber']} "
              f"ridge_clearly_ahead_of_hgbr={r['ridge_clearly_ahead_of_hgbr']}")
    return {"2024": audit_2024, "2025": audit_2025}


def phase7_calibration_robustness() -> dict:
    root = artifact_root() / "fix8-official-oof-calibration-2026"
    per_stream_arrays = {s: cert.load_stream_paired_arrays(s, root) for s in cert.STREAM_NAMES}
    per_stream_points = {s: cert.stream_point_estimates(a) for s, a in per_stream_arrays.items()}
    per_stream_bootstrap = {}
    for s, a in per_stream_arrays.items():
        t0 = time.time()
        per_stream_bootstrap[s] = cert.bootstrap_calibration_deltas(a)
        print(f"Phase 7 [{s}] bootstrap OK in {time.time()-t0:.1f}s: {per_stream_bootstrap[s]}")

    ats_pooled_arrays = cert.pooled_arrays([per_stream_arrays["ATS_TUE"], per_stream_arrays["ATS_FRI"]])
    total_pooled_arrays = cert.pooled_arrays([per_stream_arrays["TOTAL_TUE"], per_stream_arrays["TOTAL_FRI"]])
    all_four_arrays = cert.pooled_arrays(list(per_stream_arrays.values()))

    ats_pooled_points = cert.stream_point_estimates(ats_pooled_arrays)
    total_pooled_points = cert.stream_point_estimates(total_pooled_arrays)
    all_four_points = cert.stream_point_estimates(all_four_arrays)

    ats_pooled_bootstrap = cert.bootstrap_calibration_deltas(ats_pooled_arrays)
    total_pooled_bootstrap = cert.bootstrap_calibration_deltas(total_pooled_arrays)
    t0 = time.time()
    all_four_bootstrap = cert.bootstrap_calibration_deltas(all_four_arrays)
    print(f"Phase 7 [ALL_FOUR] bootstrap OK in {time.time()-t0:.1f}s: {all_four_bootstrap}")

    return {
        "per_stream_points": per_stream_points, "per_stream_bootstrap": per_stream_bootstrap,
        "ats_pooled_points": ats_pooled_points, "ats_pooled_bootstrap": ats_pooled_bootstrap,
        "total_pooled_points": total_pooled_points, "total_pooled_bootstrap": total_pooled_bootstrap,
        "all_four_points": all_four_points, "all_four_bootstrap": all_four_bootstrap,
    }


def phase8_calibration_gate(calibration: dict) -> dict:
    gate = cert.evaluate_calibration_gate(calibration["per_stream_points"], calibration["all_four_bootstrap"])
    ats_neg = sum(1 for s in ("ATS_TUE", "ATS_FRI") if calibration["per_stream_points"][s]["log_loss_delta"] < 0)
    total_neg = sum(1 for s in ("TOTAL_TUE", "TOTAL_FRI") if calibration["per_stream_points"][s]["log_loss_delta"] < 0)
    ats_evidence = cert.evidence_label(ats_neg, 2, calibration["ats_pooled_bootstrap"]["log_loss_delta_ci95_below_zero"])
    total_evidence = cert.evidence_label(total_neg, 2, calibration["total_pooled_bootstrap"]["log_loss_delta_ci95_below_zero"])
    print(f"Phase 8 OK: CALIBRATION_IMPROVES_RAW_PROBABILITIES={gate['status']} "
          f"ATS_CALIBRATION_EVIDENCE={ats_evidence} TOTAL_CALIBRATION_EVIDENCE={total_evidence}")
    return {"gate": gate, "ats_calibration_evidence": ats_evidence, "total_calibration_evidence": total_evidence}


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def phase_persist(preregistration, phase0, scope_results, bootstrap, gate5, family_2024_2025, calibration, gate8) -> None:
    model_family_summary = {
        "schema_version": cert.SCHEMA_VERSION,
        "final_review_certification_preregistration_hash": preregistration["hash"],
        "scopes": scope_results,
        "bootstrap": bootstrap,
        "model_family_stability_gate": gate5,
        "family_2024_2025_diagnostic": family_2024_2025,
    }
    (OUTPUT_DIR / "model_family_robustness_summary.json").write_text(
        json.dumps(model_family_summary, indent=2, sort_keys=True, default=_json_default)
    )

    calibration_summary = {
        "schema_version": cert.SCHEMA_VERSION,
        "final_review_certification_preregistration_hash": preregistration["hash"],
        **calibration,
        "calibration_gate": gate8,
    }
    (OUTPUT_DIR / "calibration_robustness_summary.json").write_text(
        json.dumps(calibration_summary, indent=2, sort_keys=True, default=_json_default)
    )

    small_summary = {
        "schema_version": cert.SCHEMA_VERSION,
        "final_review_certification_preregistration_hash": preregistration["hash"],
        "repo_gate": phase0["gate"], "certified_hashes": phase0["hashes"],
        "model_family_stability_status": gate5["status"],
        "calibration_status": gate8["gate"]["status"],
        "ats_calibration_evidence": gate8["ats_calibration_evidence"],
        "total_calibration_evidence": gate8["total_calibration_evidence"],
        "scope_summary": {
            name: {
                "selected_family": r["selected_family"], "one_se_family_set": r["one_se_family_set"],
                "selected_ridge_alpha": r["selected_ridge_alpha"],
            } for name, r in scope_results.items()
        },
        "bootstrap_summary": {
            "family_selected_frequency": bootstrap["family_selected_frequency"],
            "probability_ridge_alpha_100_selected": bootstrap["probability_ridge_alpha_100_selected"],
            "probability_ridge_family_in_one_se_set": bootstrap["probability_ridge_family_in_one_se_set"],
            "probability_hgbr_in_one_se_set": bootstrap["probability_hgbr_in_one_se_set"],
        },
    }
    (REPO_ROOT / "outputs" / "final_review_certification_summary.json").write_text(
        json.dumps(small_summary, indent=2, sort_keys=True, default=_json_default)
    )
    print(f"Persisted: {OUTPUT_DIR}/model_family_robustness_summary.json, "
          f"{OUTPUT_DIR}/calibration_robustness_summary.json, outputs/final_review_certification_summary.json")


def main() -> int:
    phase0 = phase0_gate()
    preregistration = phase1_preregistration(phase0)
    diagnostic = phase2_diagnostic_fits(preregistration)
    scope_results = phase3_scopes(diagnostic)
    bootstrap = phase4_bootstrap(diagnostic)
    gate5 = phase5_gate(scope_results, bootstrap)
    family_2024_2025 = phase6_2024_2025_diagnostic(phase0, diagnostic)
    calibration = phase7_calibration_robustness()
    gate8 = phase8_calibration_gate(calibration)
    phase_persist(preregistration, phase0, scope_results, bootstrap, gate5, family_2024_2025, calibration, gate8)
    print("FINAL REVIEW CERTIFICATION Sections 0-11: DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
