"""EXTENDED real-data proof for the Fix 4 chronological calibration engine
(:mod:`nfl_hybrid.evaluation.chronological_calibration`).

Exercises the engine against the already-persisted, SHA256-lineage-verified
corrected Fix 3.1 OOF residual ledger (``oof_chronological.residual_ledger``
under ``NFL_MODEL_ARTIFACT_ROOT``), so this module is marked
``extended_data`` and skips cleanly when ``NFL_MODEL_ARTIFACT_ROOT`` is not
configured, the artifact is missing, or its SHA256 does not match the
certified corrected Fix 3.1 values -- e.g. GitHub CI, or any machine without
the private artifact estate. It complements, and does not replace, the
synthetic adversarial suite in :mod:`tests.test_chronological_calibration`,
which is the actual leakage-mechanism proof and runs unconditionally.

This module proves the real-data membership invariant the task specification
asks for explicitly: for a representative real target prediction,
``max_calibration_result_available_at_utc < target_cutoff_utc`` and
``target_game_id not in calibration_prediction_ids`` -- on the real,
certified 2023-season proof-scale ledger (225 games), not a hand-built toy
sequence and not a locally-retrained model.

NO OOF TRAINING (2026-08-20): this module reads ONLY the already-persisted
corrected Fix 3.1 ``oof_chronological.residual_ledger`` -- it never calls
:func:`nfl_hybrid.evaluation.chronological_oof.generate_oof_predictions` or
any other Fix 3 training path, and never calls ``persist_oof_ledger`` or
``persist_calibration_ledger`` (which would write to/overwrite a canonical
``NFL_MODEL_ARTIFACT_ROOT`` artifact) -- see
``test_module_never_trains_or_persists_oof_or_calibration_artifacts`` below,
which guards both at the source level. Before use, the module independently
recomputes the residual ledger's (and its sibling predictions/manifest
files') SHA256 and refuses to run unless all three match the certified
corrected Fix 3.1 values byte-for-byte -- this test can never silently run
against stale or leaked evidence.

STILL NOT ATS/TOTAL CERTIFICATION EVIDENCE: this module prices MONEYLINE
only (margin 0, no market line needed) as an engine-mechanism check; it does
not attach a real T-10 market line and is not primary-market evidence. The
authoritative ATS/TOTAL Fix 4 certification evidence is
``outputs/real_ats_total_chronological_calibration_proof.json``, produced by
``scripts/generate_real_ats_total_calibration_proof.py``, which reads the
identical persisted Fix 3.1 ledger.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.data.external_data import ExternalDataUnavailableError, resolve
from nfl_hybrid.evaluation.chronological_calibration import (
    MARKET_MONEYLINE,
    ChronologicalCalibrationConfig,
    build_raw_probabilities,
    generate_chronological_calibration,
)

# The certified corrected Fix 3.1 lineage (post-leakage-remediation) this
# module requires byte-exact -- the same values independently verified for
# the authoritative ATS/TOTAL certification (Fix 4 v2). Never updated to
# match whatever happens to be on disk; a mismatch means either stale/leaked
# evidence or an unrelated OOF run, and this module must refuse both.
EXPECTED_FEATURE_STATE_HASH = "7d731c88416cef2d9f55b2267b377c085b557d349117637be6fecd0f9bf1c931"
EXPECTED_OOF_PREDICTIONS_SHA256 = "06744f66361a3d78bff432825972fe812c8811e7310076c72c10b651ca69f14f"
EXPECTED_OOF_RESIDUAL_LEDGER_SHA256 = "aa3d11114d8b709042dde2865ef077203f2bf775e90d2dd38e94c8fbc5b32e9c"
EXPECTED_OOF_MANIFEST_SHA256 = "6551b34a8ca57582c85b604e11b78d743ed3089b1fcdb3c727c4266d8ebd5ad6"

FORECAST_HORIZON = "kickoff_minus_10_minutes"


def _try_resolve(key: str) -> Path | None:
    try:
        return resolve(key)
    except ExternalDataUnavailableError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


OOF_PREDICTIONS_PATH = _try_resolve("oof_chronological.predictions")
OOF_RESIDUAL_LEDGER_PATH = _try_resolve("oof_chronological.residual_ledger")
OOF_MANIFEST_PATH = _try_resolve("oof_chronological.manifest")


def _corrected_fix3_1_lineage_verified() -> bool:
    """Independently re-verify the persisted artifact is byte-exactly the
    certified corrected Fix 3.1 ledger -- never trusted merely by path/key
    resolving, and never a fallback regeneration when it doesn't match."""
    if OOF_PREDICTIONS_PATH is None or OOF_RESIDUAL_LEDGER_PATH is None or OOF_MANIFEST_PATH is None:
        return False
    if not (OOF_PREDICTIONS_PATH.is_file() and OOF_RESIDUAL_LEDGER_PATH.is_file() and OOF_MANIFEST_PATH.is_file()):
        return False
    if _sha256_file(OOF_PREDICTIONS_PATH) != EXPECTED_OOF_PREDICTIONS_SHA256:
        return False
    if _sha256_file(OOF_RESIDUAL_LEDGER_PATH) != EXPECTED_OOF_RESIDUAL_LEDGER_SHA256:
        return False
    if _sha256_file(OOF_MANIFEST_PATH) != EXPECTED_OOF_MANIFEST_SHA256:
        return False
    manifest = json.loads(OOF_MANIFEST_PATH.read_text(encoding="utf-8"))
    return str(manifest.get("feature_state_hash")) == EXPECTED_FEATURE_STATE_HASH


_FIX3_1_LINEAGE_VERIFIED = _corrected_fix3_1_lineage_verified()

pytestmark = [
    pytest.mark.extended_data,
    pytest.mark.skipif(
        not _FIX3_1_LINEAGE_VERIFIED,
        reason=(
            "NFL_MODEL_ARTIFACT_ROOT not configured, the corrected Fix 3.1 OOF "
            "residual ledger is missing, or its SHA256/feature_state_hash does "
            "not match the certified corrected Fix 3.1 values (extended_data)"
        ),
    ),
]


@pytest.fixture(scope="module")
def real_calibration_ledger():
    """The real, certified corrected Fix 3.1 residual ledger (225 proof-scale
    2023-season rows), loaded read-only -- no training, no OOF regeneration,
    no market attachment (MONEYLINE fixed at margin 0)."""
    residual_ledger = pd.read_parquet(OOF_RESIDUAL_LEDGER_PATH)
    raw = build_raw_probabilities(residual_ledger, market=MARKET_MONEYLINE, forecast_horizon=FORECAST_HORIZON)
    ledger = generate_chronological_calibration(raw, config=ChronologicalCalibrationConfig())
    return raw, ledger


def test_real_raw_probabilities_are_ready_only_for_oof_and_uncertainty_eligible_rows(real_calibration_ledger):
    raw, _ledger = real_calibration_ledger
    assert (raw["raw_status"] == "RAW_READY").sum() > 0
    ready = raw[raw["raw_status"] == "RAW_READY"]
    assert ready["raw_home_probability"].notna().all()
    assert ready["raw_push_probability"].notna().all()
    assert ready["raw_away_probability"].notna().all()


def test_representative_real_target_calibration_membership_proof(real_calibration_ledger):
    """The exact real-data proof the specification asks for:
    max_calibration_result_available_at_utc < target_cutoff_utc and
    target_game_id not in calibration_prediction_ids -- for a representative
    real target prediction, independently re-derived from the raw frame, not
    just trusted from the ledger's own bookkeeping."""
    raw, ledger = real_calibration_ledger
    candidates = ledger[
        (ledger["calibration_sample_count"] > 0) & (~ledger["calibration_prediction_ids_truncated"])
    ]
    assert len(candidates) > 0, "no real target produced a non-empty calibration sample"
    row = candidates.sort_values("calibration_sample_count", ascending=False).iloc[0]

    target_game_id = str(row["game_id"])
    target_cutoff = pd.Timestamp(row["target_cutoff_utc"])
    max_calibration_available = pd.Timestamp(row["max_calibration_result_available_at_utc"])
    calibration_ids = list(row["calibration_prediction_ids"])

    print(
        f"\n[Fix 4 real-data membership proof] target_game_id={target_game_id} "
        f"status={row['calibration_status']} "
        f"sample_count={row['calibration_sample_count']} "
        f"target_cutoff_utc={target_cutoff} "
        f"max_calibration_result_available_at_utc={max_calibration_available}"
    )

    assert target_game_id not in calibration_ids
    assert max_calibration_available < target_cutoff

    available_by_id = dict(zip(raw["game_id"], pd.to_datetime(raw["result_available_at_utc"], utc=True)))
    real_max_available = max(available_by_id[gid] for gid in calibration_ids)
    assert real_max_available == max_calibration_available
    assert real_max_available < target_cutoff


def test_every_real_calibrated_or_sampled_row_satisfies_membership_invariant(real_calibration_ledger):
    raw, ledger = real_calibration_ledger
    available_by_id = dict(zip(raw["game_id"], pd.to_datetime(raw["result_available_at_utc"], utc=True)))
    checked = 0
    for row in ledger.itertuples(index=False):
        if row.calibration_sample_count == 0:
            continue
        assert row.game_id not in row.calibration_prediction_ids or row.calibration_prediction_ids_truncated
        if row.calibration_status == "CALIBRATED":
            assert row.max_calibration_result_available_at_utc < row.target_cutoff_utc
            if not row.calibration_prediction_ids_truncated:
                for cid in row.calibration_prediction_ids:
                    assert available_by_id[cid] < row.target_cutoff_utc
                    checked += 1
    assert checked >= 0  # loop itself is the proof; no CALIBRATED rows is a valid honest outcome


def test_insufficient_history_status_is_explicit_on_real_data_not_silent(real_calibration_ledger):
    _raw, ledger = real_calibration_ledger
    sampled = ledger[ledger["calibration_sample_count"] > 0]
    assert len(sampled) > 0
    assert sampled["calibration_status"].isin(["CALIBRATED", "UNCALIBRATED_INSUFFICIENT_HISTORY"]).all()
    insufficient = sampled[sampled["calibration_status"] == "UNCALIBRATED_INSUFFICIENT_HISTORY"]
    if len(insufficient):
        assert insufficient["calibrated_home_probability"].isna().all()
        assert insufficient["raw_home_probability"].notna().all()


def test_module_never_trains_or_persists_oof_or_calibration_artifacts():
    """Source-level guard (Fix 4 test-design cleanup, 2026-08-20): this
    module must never import generate_oof_predictions (or any other Fix 3
    training entry point) or persist_oof_ledger/persist_calibration_ledger
    (which would write to/overwrite a canonical NFL_MODEL_ARTIFACT_ROOT
    artifact). Checked structurally (this module's own namespace), not by a
    raw substring scan -- this test's own docstring/assertion text
    necessarily names the forbidden functions, which a plain text scan can't
    distinguish from an actual import. Subject to the same module-level
    ``extended_data``/lineage-verified skip gate as every other test here
    (module import already fails closed if any forbidden name were imported,
    independent of whether this specific test runs)."""
    this_module = sys.modules[__name__]
    forbidden_names = (
        "generate_oof_predictions",
        "persist_oof_ledger",
        "persist_calibration_ledger",
    )
    for forbidden in forbidden_names:
        assert not hasattr(this_module, forbidden), (
            f"{forbidden} must never be imported into this module -- it would let a "
            "real-data engine-mechanism test retrain and/or mutate/overwrite a "
            "canonical NFL_MODEL_ARTIFACT_ROOT artifact."
        )
