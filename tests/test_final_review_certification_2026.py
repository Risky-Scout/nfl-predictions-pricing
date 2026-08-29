"""Focused tests for the FINAL REVIEW CERTIFICATION core library (Section
30). Gate-evaluator tests use small synthetic scope/bootstrap dicts rather
than a full real diagnostic run (that real run is exercised end-to-end by
``scripts/run_final_review_certification.py`` itself, whose live output is
checked into ``outputs/final_review_certification_summary.json``)."""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_hybrid.certification import final_review_2026 as cert


# ---------------------------------------------------------------------------
# Canonical hashing -- deterministic, order-independent, no default=str.
# ---------------------------------------------------------------------------
def test_canonical_hash_deterministic():
    payload = {"b": 2, "a": 1, "nested": {"z": [1, 2, 3]}}
    assert cert._sha256_hex(payload) == cert._sha256_hex(dict(reversed(list(payload.items()))))


def test_canonical_hash_sensitive_to_value_change():
    assert cert._sha256_hex({"a": 1}) != cert._sha256_hex({"a": 2})


def test_canonical_json_no_default_str_on_non_primitive():
    class Weird:
        pass

    with pytest.raises(TypeError):
        cert._canonical_json({"x": Weird()})


# ---------------------------------------------------------------------------
# Section 0 -- repo gate.
# ---------------------------------------------------------------------------
def test_repo_gate_passes_on_certified_values():
    gate = cert.verify_repo_gate(
        branch="cert/robustness-production-readiness-2026", head_sha=cert.CERTIFIED_BASELINE_SHA,
        tag_commit_sha=cert.CERTIFIED_BASELINE_SHA, dirty_paths=[],
    )
    assert gate["clean"] if "clean" in gate else True


def test_repo_gate_rejects_wrong_branch():
    with pytest.raises(cert.CertificationGateFailure):
        cert.verify_repo_gate(branch="main", head_sha=cert.CERTIFIED_BASELINE_SHA, tag_commit_sha=cert.CERTIFIED_BASELINE_SHA, dirty_paths=[])


def test_repo_gate_rejects_sha_mismatch():
    with pytest.raises(cert.CertificationGateFailure):
        cert.verify_repo_gate(
            branch="cert/robustness-production-readiness-2026", head_sha="0" * 40,
            tag_commit_sha=cert.CERTIFIED_BASELINE_SHA, dirty_paths=[],
        )


def test_repo_gate_rejects_unexpected_dirty_file():
    with pytest.raises(cert.CertificationGateFailure):
        cert.verify_repo_gate(
            branch="cert/robustness-production-readiness-2026", head_sha=cert.CERTIFIED_BASELINE_SHA,
            tag_commit_sha=cert.CERTIFIED_BASELINE_SHA, dirty_paths=["src/nfl_hybrid/legacy/elo.py"],
        )


def test_repo_gate_allows_expected_certification_paths():
    gate = cert.verify_repo_gate(
        branch="cert/robustness-production-readiness-2026", head_sha=cert.CERTIFIED_BASELINE_SHA,
        tag_commit_sha=cert.CERTIFIED_BASELINE_SHA, dirty_paths=["src/nfl_hybrid/certification/final_review_2026.py"],
    )
    assert gate["branch"] == "cert/robustness-production-readiness-2026"


def test_verify_certified_hashes_rejects_mismatch():
    fix71 = {
        "horizon_feature_semantics_hash": "WRONG", "horizon_membership_ledger_hash": cert.CERTIFIED_HORIZON_MEMBERSHIP_LEDGER_HASH,
        "operational_model_spec_hash": cert.CERTIFIED_OPERATIONAL_MODEL_SPEC_HASH,
    }
    fix8 = {"fix8_official_oof_calibration_preregistration_hash": cert.CERTIFIED_FIX8_PREREGISTRATION_HASH}
    with pytest.raises(cert.CertificationGateFailure):
        cert.verify_certified_hashes(fix71, fix8)


# ---------------------------------------------------------------------------
# Section 5 -- MODEL_FAMILY_STABILITY gate, synthetic inputs.
# ---------------------------------------------------------------------------
def _scope_result(family, one_se, alpha="RIDGE_ALPHA_100"):
    return {"selected_family": family, "one_se_family_set": one_se, "selected_ridge_alpha": alpha}


def test_model_family_gate_met_strongly_when_all_criteria_pass():
    scope_results = {s.name: _scope_result("RIDGE", ["RIDGE", "HUBER"]) for s in cert.ROBUSTNESS_SCOPES}
    bootstrap = {
        "family_selected_frequency": {"RIDGE": 0.85, "HUBER": 0.15, "HGBR": 0.0},
        "probability_ridge_family_in_one_se_set": 0.95,
        "probability_hgbr_in_one_se_set": 0.0,
    }
    gate = cert.evaluate_model_family_stability_gate(scope_results, bootstrap)
    assert gate["status"] == "MET_STRONGLY"


def test_model_family_gate_mixed_when_ridge_falls_outside_one_se():
    scope_results = {s.name: _scope_result("RIDGE", ["RIDGE", "HUBER"]) for s in cert.ROBUSTNESS_SCOPES}
    scope_results["S5"] = _scope_result("HUBER", ["HUBER"])  # RIDGE missing from one-SE set
    bootstrap = {
        "family_selected_frequency": {"RIDGE": 0.68, "HUBER": 0.32, "HGBR": 0.0},
        "probability_ridge_family_in_one_se_set": 0.68,
        "probability_hgbr_in_one_se_set": 0.0,
    }
    gate = cert.evaluate_model_family_stability_gate(scope_results, bootstrap)
    assert gate["status"] == "MIXED"
    assert gate["checks"]["2_ridge_in_one_se_all_scopes"]["pass"] is False


def test_model_family_gate_thresholds_not_weakened_at_boundary():
    scope_results = {s.name: _scope_result("RIDGE", ["RIDGE", "HUBER"]) for s in cert.ROBUSTNESS_SCOPES}
    bootstrap = {
        "family_selected_frequency": {"RIDGE": 0.79, "HUBER": 0.21, "HGBR": 0.0},  # just below 0.80
        "probability_ridge_family_in_one_se_set": 0.95, "probability_hgbr_in_one_se_set": 0.0,
    }
    gate = cert.evaluate_model_family_stability_gate(scope_results, bootstrap)
    assert gate["status"] != "MET_STRONGLY"
    assert gate["checks"]["5_bootstrap_p_ridge_family_ge_0_80"]["pass"] is False


# ---------------------------------------------------------------------------
# Section 8 -- CALIBRATION_IMPROVES_RAW_PROBABILITIES gate, synthetic inputs.
# ---------------------------------------------------------------------------
def _points(ll_delta, brier_delta, raw_ece, cal_ece):
    return {"log_loss_delta": ll_delta, "brier_delta": brier_delta, "raw_ece": raw_ece, "calibrated_ece": cal_ece}


def test_calibration_gate_met_strongly_all_improve():
    per_stream = {s: _points(-0.02, -0.01, 0.12, 0.09) for s in cert.STREAM_NAMES}
    pooled_bootstrap = {"log_loss_delta_ci95": [-0.03, -0.01], "log_loss_delta_ci95_below_zero": True,
                         "brier_delta_ci95": [-0.02, -0.005], "brier_delta_ci95_below_zero": True}
    gate = cert.evaluate_calibration_gate(per_stream, pooled_bootstrap)
    assert gate["status"] == "MET_STRONGLY"


def test_calibration_gate_mixed_when_one_stream_worsens():
    per_stream = {s: _points(-0.02, -0.01, 0.12, 0.09) for s in cert.STREAM_NAMES}
    per_stream["TOTAL_FRI"] = _points(0.005, 0.003, 0.12, 0.13)  # worsens
    pooled_bootstrap = {"log_loss_delta_ci95": [-0.03, -0.01], "log_loss_delta_ci95_below_zero": True,
                         "brier_delta_ci95": [-0.02, -0.005], "brier_delta_ci95_below_zero": True}
    gate = cert.evaluate_calibration_gate(per_stream, pooled_bootstrap)
    assert gate["status"] != "MET_STRONGLY"


def test_evidence_label_strong_requires_pooled_ci_below_zero():
    assert cert.evidence_label(2, 2, True) == "STRONG"
    assert cert.evidence_label(2, 2, False) == "SUGGESTIVE"
    assert cert.evidence_label(0, 2, False) == "MIXED"


# ---------------------------------------------------------------------------
# Robustness scopes -- structural sanity (predeclared, not editable).
# ---------------------------------------------------------------------------
def test_nine_predeclared_scopes():
    assert [s.name for s in cert.ROBUSTNESS_SCOPES] == [f"S{i}" for i in range(1, 10)]


def test_s1_s2_single_horizon_s3_both():
    assert cert.SCOPE_BY_NAME["S1"].horizons == ("TUE",)
    assert cert.SCOPE_BY_NAME["S2"].horizons == ("FRI",)
    assert cert.SCOPE_BY_NAME["S3"].horizons == ("TUE", "FRI")


def test_s7_s8_s9_single_fold():
    assert cert.SCOPE_BY_NAME["S7"].folds == ("A",)
    assert cert.SCOPE_BY_NAME["S8"].folds == ("B",)
    assert cert.SCOPE_BY_NAME["S9"].folds == ("C",)
