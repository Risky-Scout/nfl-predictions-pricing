"""Focused tests for ``scripts/finalize_fix7_replay_provenance.py``.

NO historical model fits anywhere in this file: every test operates on small
synthetic dicts shaped like the already-persisted replay artifact, never on
real backfill data, and never imports anything capable of fitting a model
(the script under test deliberately does not import
:mod:`nfl_hybrid.selection.model_family_selection_2026` either).
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finalize_fix7_replay_provenance.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("finalize_fix7_replay_provenance_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prov = _load_script()


def _sample_replay_artifact() -> dict:
    return {
        "schema_version": "fix7-postfreeze-evidence-replay-v1",
        "replay_used_for_selection": False,
        "selected_family_as_frozen": "RIDGE",
        "selected_candidate_as_frozen": "RIDGE_ALPHA_100",
        "original_frozen_hashes": {
            "fix6_feature_manifest_hash": prov.EXPECTED_FIX6_FEATURE_MANIFEST_HASH,
            "selection_matrix_hash": prov.EXPECTED_SELECTION_MATRIX_HASH,
            "model_family_preregistration_hash": prov.EXPECTED_PREREGISTRATION_HASH,
            "candidate_registry_hash": prov.EXPECTED_CANDIDATE_REGISTRY_HASH,
            "final_model_spec_hash": prov.EXPECTED_FINAL_MODEL_SPEC_HASH,
        },
        "replay_matrix_hash": prov.EXPECTED_SELECTION_MATRIX_HASH,
        "replay_matrix_row_count": 1343,
        "firewall_counts": {"raw_rows_season_2025_seen_by_replay_firewall": 285, "rows_season_ge_2025_passed_beyond_replay_firewall": 0},
        "cache_counts": {"replay_predictions_loaded_from_verified_cache": [], "replay_predictions_regenerated": [{"candidate": "RIDGE_ALPHA_1", "fold": "A"}]},
        "inner_fold_detail_rows": [
            {"candidate": "RIDGE_ALPHA_1", "candidate_spec_hash": "abc123", "fold": "A", "n_games": 272, "margin_RMSE": 14.4, "primary_score": 14.2},
            {"candidate": "HUBER_FIXED", "candidate_spec_hash": "def456", "fold": "A", "n_games": 272, "margin_RMSE": 14.3, "primary_score": 14.1},
        ],
        "pooled_reproduction_checks": {"RIDGE_ALPHA_100": {"match": True}},
        "ridge_alpha_cross_check": {"RIDGE_ALPHA_0_1": {"mean_delta_vs_best": {"match": True}}},
        "family_selection_cross_check": {"selected_family": "RIDGE"},
        "finalist_2024_metrics": {
            "RIDGE": {"margin_RMSE": 13.06, "margin_MAE": 10.04, "segment_primary_score": {"week1": 11.4}},
            "HUBER": {"margin_RMSE": 13.08, "margin_MAE": 10.05, "segment_primary_score": {"week1": 11.6}},
            "HGBR": {"margin_RMSE": 13.79, "margin_MAE": 10.76, "segment_primary_score": {"week1": 12.6}},
        },
        "locked_2024_audit_replay": {"overall_pass": True, "rules": {"overall": {"pass": True}}},
        "hash_immutability_gate": {"all_unchanged": True},
        "replay_environment_evidence": {
            "branch": "fix/model-family-selection-2026", "head_sha": prov.EXPECTED_FINAL_MODEL_SPEC_HASH[:0] + "87bf1786ff2bbad5aa6160797eeb660ae0887e5d",
            "origin_main": "87bf1786ff2bbad5aa6160797eeb660ae0887e5d", "merge_base": "87bf1786ff2bbad5aa6160797eeb660ae0887e5d",
            "dirty_files": ["outputs/fix7_model_family_selection_summary.json"],
        },
        "fit_counts": {"original_real_data_paired_fit_count": 21, "original_real_data_individual_fit_count": 42, "replay_real_data_paired_fit_count": 21, "replay_real_data_individual_fit_count": 42},
        "fix7_postfreeze_evidence_replay_hash": "legacy-volatile-hash-run1",
    }


def _sample_prereg() -> dict:
    return {
        "candidate_registry_hash": prov.EXPECTED_CANDIDATE_REGISTRY_HASH,
        "model_family_preregistration_hash": prov.EXPECTED_PREREGISTRATION_HASH,
        "deterministic_settings": {"random_state": 42, "single_threaded": True},
    }


# ---------------------------------------------------------------------------
# Replay scientific-hash determinism
# ---------------------------------------------------------------------------
def test_scientific_hash_deterministic():
    artifact = _sample_replay_artifact()
    prereg = _sample_prereg()
    payload_a = prov.build_scientific_payload(artifact, prereg)
    payload_b = prov.build_scientific_payload(artifact, prereg)
    assert prov.compute_stable_hash(payload_a) == prov.compute_stable_hash(payload_b)


def test_scientific_hash_sensitive_to_real_change():
    artifact = _sample_replay_artifact()
    prereg = _sample_prereg()
    payload_a = prov.build_scientific_payload(artifact, prereg)

    mutated = copy.deepcopy(artifact)
    mutated["finalist_2024_metrics"]["RIDGE"]["margin_RMSE"] = 999.0
    payload_b = prov.build_scientific_payload(mutated, prereg)

    assert prov.compute_stable_hash(payload_a) != prov.compute_stable_hash(payload_b)


# ---------------------------------------------------------------------------
# Dirty-metadata excluded from scientific hash: SAME scientific content,
# DIFFERENT volatile metadata (dirty files, legacy hash, branch/head churn
# simulated) => identical stable hash.
# ---------------------------------------------------------------------------
def test_dirty_metadata_excluded_from_scientific_hash():
    artifact_run1 = _sample_replay_artifact()
    artifact_run2 = copy.deepcopy(artifact_run1)

    # Simulate exactly what happened between the two real replay runs: only
    # volatile execution metadata changed (more dirty files present, a
    # different legacy hash, a later "run"), scientific content untouched.
    artifact_run2["replay_environment_evidence"]["dirty_files"] = [
        "outputs/fix7_model_family_selection_summary.json",
        "outputs/fix7_postfreeze_evidence_replay_summary.json",
        "docs/MODEL_FAMILY_SELECTION_2026.md",
    ]
    artifact_run2["fix7_postfreeze_evidence_replay_hash"] = "legacy-volatile-hash-run2-totally-different"
    artifact_run2["hash_immutability_gate"] = {"all_unchanged": True, "checked_at": "run2"}

    prereg = _sample_prereg()
    payload_1 = prov.build_scientific_payload(artifact_run1, prereg)
    payload_2 = prov.build_scientific_payload(artifact_run2, prereg)

    assert prov.compute_stable_hash(payload_1) == prov.compute_stable_hash(payload_2)


def test_scientific_payload_excludes_volatile_keys_by_construction():
    artifact = _sample_replay_artifact()
    prereg = _sample_prereg()
    payload = prov.build_scientific_payload(artifact, prereg)
    forbidden = ("dirty_files", "cache_counts", "hash_immutability_gate", "replay_environment_evidence", "fix7_postfreeze_evidence_replay_hash", "fit_counts")
    serialized = json.dumps(payload)
    for key in forbidden:
        assert key not in payload
        # also confirm no volatile VALUE (the run-2-only dirty file) leaked in as a nested value
    assert "legacy-volatile-hash" not in serialized


def test_scientific_payload_key_set_is_closed_allowlist():
    artifact = _sample_replay_artifact()
    prereg = _sample_prereg()
    payload = prov.build_scientific_payload(artifact, prereg)
    assert set(payload.keys()) == set(prov.SCIENTIFIC_PAYLOAD_KEYS)


# ---------------------------------------------------------------------------
# Canonical JSON settings
# ---------------------------------------------------------------------------
def test_canonical_json_uses_required_settings():
    payload = {"b": 1, "a": 2}
    encoded = prov._canonical_json(payload)
    assert encoded == b'{"a":2,"b":1}'  # sort_keys, no spaces


# ---------------------------------------------------------------------------
# Cumulative replay accounting serialization
# ---------------------------------------------------------------------------
def test_fit_provenance_accounting():
    fit_provenance = prov.build_fit_provenance()
    assert fit_provenance["successful_replay_run_count"] == 2
    assert fit_provenance["latest_replay_real_data_paired_fit_count"] == 21
    assert fit_provenance["latest_replay_real_data_individual_fit_count"] == 42
    assert fit_provenance["cumulative_replay_real_data_paired_fit_count"] == 42
    assert fit_provenance["cumulative_replay_real_data_individual_fit_count"] == 84
    assert fit_provenance["original_real_data_paired_fit_count"] == 21
    assert fit_provenance["original_real_data_individual_fit_count"] == 42
    assert fit_provenance["all_replay_fits_used_for_selection"] is False
    assert "duplicate_replay_explanation" in fit_provenance


def test_cumulative_equals_latest_times_run_count():
    fit_provenance = prov.build_fit_provenance()
    assert fit_provenance["cumulative_replay_real_data_paired_fit_count"] == (
        fit_provenance["latest_replay_real_data_paired_fit_count"] * fit_provenance["successful_replay_run_count"]
    )
    assert fit_provenance["cumulative_replay_real_data_individual_fit_count"] == (
        fit_provenance["latest_replay_real_data_individual_fit_count"] * fit_provenance["successful_replay_run_count"]
    )


# ---------------------------------------------------------------------------
# Original frozen-hash preservation
# ---------------------------------------------------------------------------
def test_verify_frozen_values_passes_on_matching_evidence():
    evidence = {
        "fix7_summary": {
            "fix6_feature_manifest_hash": prov.EXPECTED_FIX6_FEATURE_MANIFEST_HASH,
            "selection_matrix_hash": prov.EXPECTED_SELECTION_MATRIX_HASH,
            "model_family_preregistration_hash": prov.EXPECTED_PREREGISTRATION_HASH,
            "final_model_spec_hash": prov.EXPECTED_FINAL_MODEL_SPEC_HASH,
            "selected_family": prov.EXPECTED_SELECTED_FAMILY,
            "selected_candidate_spec": {"name": prov.EXPECTED_SELECTED_CANDIDATE},
        },
        "prereg": {"candidate_registry_hash": prov.EXPECTED_CANDIDATE_REGISTRY_HASH},
    }
    prov.verify_frozen_values_unchanged(evidence)  # must not raise


def test_verify_frozen_values_raises_on_any_divergence():
    evidence = {
        "fix7_summary": {
            "fix6_feature_manifest_hash": prov.EXPECTED_FIX6_FEATURE_MANIFEST_HASH,
            "selection_matrix_hash": prov.EXPECTED_SELECTION_MATRIX_HASH,
            "model_family_preregistration_hash": prov.EXPECTED_PREREGISTRATION_HASH,
            "final_model_spec_hash": prov.EXPECTED_FINAL_MODEL_SPEC_HASH,
            "selected_family": "HUBER",  # divergence
            "selected_candidate_spec": {"name": prov.EXPECTED_SELECTED_CANDIDATE},
        },
        "prereg": {"candidate_registry_hash": prov.EXPECTED_CANDIDATE_REGISTRY_HASH},
    }
    with pytest.raises(prov.ProvenanceMismatch):
        prov.verify_frozen_values_unchanged(evidence)


def test_verify_frozen_values_raises_on_hash_divergence():
    evidence = {
        "fix7_summary": {
            "fix6_feature_manifest_hash": "0" * 70,  # divergence
            "selection_matrix_hash": prov.EXPECTED_SELECTION_MATRIX_HASH,
            "model_family_preregistration_hash": prov.EXPECTED_PREREGISTRATION_HASH,
            "final_model_spec_hash": prov.EXPECTED_FINAL_MODEL_SPEC_HASH,
            "selected_family": prov.EXPECTED_SELECTED_FAMILY,
            "selected_candidate_spec": {"name": prov.EXPECTED_SELECTED_CANDIDATE},
        },
        "prereg": {"candidate_registry_hash": prov.EXPECTED_CANDIDATE_REGISTRY_HASH},
    }
    with pytest.raises(prov.ProvenanceMismatch):
        prov.verify_frozen_values_unchanged(evidence)


# ---------------------------------------------------------------------------
# Status/document consistency
# ---------------------------------------------------------------------------
def test_remediation_status_fix6_row_says_merged():
    text = (Path(__file__).resolve().parents[1] / "REMEDIATION_STATUS_2026.md").read_text()
    fix6_line = next(line for line in text.splitlines() if line.startswith("| Fix 6"))
    assert "**MERGED**" in fix6_line
    assert "PR #20" in fix6_line
    assert "5b50d2a" in fix6_line
    assert "IN PROGRESS" not in fix6_line


def test_remediation_status_fix7_row_still_in_progress():
    text = (Path(__file__).resolve().parents[1] / "REMEDIATION_STATUS_2026.md").read_text()
    fix7_line = next(line for line in text.splitlines() if line.startswith("| Fix 7"))
    assert "IN PROGRESS / READY FOR PR" in fix7_line
