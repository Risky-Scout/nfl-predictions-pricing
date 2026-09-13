"""Proofs for 2026 production RUNTIME readiness -- the two things that stood
between the merged automation and an unattended season.

Everything here is HERMETIC and SYNTHETIC: ``tmp_path`` estates, synthetic
capture manifests built with the real capture code's own hashing helpers, and
the repository's existing synthetic baseline/candidate builders. No network,
no real capture, no real ``$NFL_MODEL_ARTIFACT_ROOT``, no SSH, nothing
published.

PART 1 -- CAPTURE MANIFEST INTEGRITY.
  A capture manifest carries TWO different SHA256 integrity objects, and
  conflating them is what produced the Week-1 TUE "hash discrepancy":

    MANIFEST_CONTENT_SHA256  sha256 of the compact deterministic JSON of the
                             body WITHOUT ``manifest_sha256``; stored inside
                             the file by the capture script. Formatting-proof.
    MANIFEST_FILE_SHA256     sha256sum of the bytes on disk, which include the
                             ``manifest_sha256`` field and ``indent=2``.

  Proved here: the two can never be equal; the content hash always
  self-verifies from the file alone; a genuine mutation fails closed; a pure
  reformat does NOT (the content is intact); a declared hash is resolved to
  whichever object it is; a hash matching neither fails closed; and the
  manifest is never rewritten.

PART 2 -- THE DAILY PASS MUST NOT CALL A MISSING INPUT HEALTHY.
  Proved here: NOT_YET_MATURE and the other legitimate no-ops stay exit 0,
  while candidate-generation failure, a missing certified baseline seed and a
  corrupt active pointer each fail closed with their own exit code; and the
  maintenance script propagates every one of them instead of printing
  ``daily_maintenance_status=OK``.

PART 3 -- THE LEGACY PREREGISTRATION MUST NOT BLOCK THE OPERATIONAL OVERLAY.
  ``scientific_refit_authorization.authorized`` is False today and stays
  False. Proved here: it is recorded as provenance and does NOT prevent a
  valid candidate from being promoted at the existing 200-game floor.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nfl_hybrid.evaluation import prospective_strength_2026 as ps
from nfl_hybrid.production import recalibration_2026 as rc

# Reuse the existing synthetic estate builders rather than writing a second
# set; these are the same helpers the merged promotion proofs use.
from test_nfl_2026_recalibration_promotion import (  # noqa: E402
    _seed,
    _write_baseline,
    _write_candidate,
    _write_maturity_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "scripts" / "verify_capture_manifest_integrity.py"
DAILY_SCRIPT = REPO_ROOT / "ops" / "wizard" / "run_daily_maintenance.sh"
CERTIFIED_SCRIPT = REPO_ROOT / "ops" / "wizard" / "run_certified_card.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capture_mod = _load(REPO_ROOT / "scripts" / "capture_bdl_2026_asof.py", "_capture_bdl_2026_asof")
verify_mod = _load(VERIFIER, "_verify_capture_manifest_integrity")
candidate_mod = _load(
    REPO_ROOT / "scripts" / "generate_2026_recalibration_candidate.py",
    "_generate_2026_recalibration_candidate_readiness",
)


# ===========================================================================
# Synthetic capture manifests, built by the REAL capture hashing helpers.
# ===========================================================================
def _manifest_body(**overrides) -> dict:
    body = {
        "schema_version": capture_mod.SCHEMA_VERSION,
        "status": "COMPLETE",
        "season": 2026,
        "week": 1,
        "horizon": "TUE",
        "requested_horizon": "TUE",
        "nominal_cutoff_utc": "2026-09-08T15:36:15Z",
        "capture_started_at_utc": "2026-09-08T15:36:00Z",
        "capture_completed_at_utc": "2026-09-08T15:36:15Z",
        "capture_uuid": "3f1c9a2e-0000-4000-8000-000000000001",
        "request_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "requests": [{"url": "https://api.balldontlie.io/nfl/v1/games", "http_status": 200}],
        "scientific_model_files_changed": False,
    }
    body.update(overrides)
    return body


def _write_manifest(directory: Path, body: dict | None = None, *, with_self_hash: bool = True) -> Path:
    """Write a manifest exactly the way scripts/capture_bdl_2026_asof.py does:
    hash the body WITHOUT the self-hash field, insert it, then pretty-print."""
    body = dict(body if body is not None else _manifest_body())
    if with_self_hash:
        body["manifest_sha256"] = capture_mod.sha256_hex(
            capture_mod.deterministic_json(body).encode("utf-8")
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_verifier(manifest: Path, *args: str) -> tuple[int, dict[str, str]]:
    result = subprocess.run(
        [sys.executable, str(VERIFIER), str(manifest), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    fields = {}
    for line in result.stdout.splitlines():
        if line.startswith("capture_manifest") and "=" in line:
            key, _, value = line.partition("=")
            fields[key] = value
    return result.returncode, fields


# ===========================================================================
# PART 1 -- the two integrity objects.
# ===========================================================================
def test_a_capture_manifests_two_integrity_objects_can_never_be_equal(tmp_path):
    """Two independent reasons: different serialization, and a self-referential
    hash cannot cover itself."""
    manifest = _write_manifest(tmp_path / "capture")
    document = json.loads(manifest.read_text())
    content = verify_mod.manifest_content_sha256(document)
    assert document["manifest_sha256"] == content
    assert _file_sha256(manifest) != content


def test_b_the_content_hash_self_verifies_from_the_file_alone(tmp_path):
    manifest = _write_manifest(tmp_path / "capture")
    exit_code, fields = _run_verifier(manifest)
    assert exit_code == 0
    assert fields["capture_manifest_self_verified"] == "true"
    assert fields["capture_manifest_integrity"] == "OK"
    assert fields["capture_manifest_declared_object"] == verify_mod.OBJECT_NONE_DECLARED


def test_c_a_declared_file_hash_is_resolved_as_the_file_object(tmp_path):
    manifest = _write_manifest(tmp_path / "capture")
    exit_code, fields = _run_verifier(manifest, "--declared-sha256", _file_sha256(manifest))
    assert exit_code == 0
    assert fields["capture_manifest_declared_object"] == verify_mod.OBJECT_FILE


def test_d_a_declared_content_hash_is_resolved_as_the_content_object(tmp_path):
    """The case the Week-1 TUE value is most likely to be: a correctly recorded
    hash of the wrong OBJECT must not be rejected as corruption."""
    manifest = _write_manifest(tmp_path / "capture")
    content = json.loads(manifest.read_text())["manifest_sha256"]
    exit_code, fields = _run_verifier(manifest, "--declared-sha256", content)
    assert exit_code == 0
    assert fields["capture_manifest_declared_object"] == verify_mod.OBJECT_CONTENT


@pytest.mark.parametrize("declared", ["0" * 64, "f" * 64, "deadbeef"])
def test_e_a_declared_hash_matching_neither_object_fails_closed(tmp_path, declared):
    manifest = _write_manifest(tmp_path / "capture")
    exit_code, fields = _run_verifier(manifest, "--declared-sha256", declared)
    assert exit_code == 3
    assert fields["capture_manifest_declared_object"] == verify_mod.OBJECT_NO_MATCH
    assert fields["capture_manifest_integrity"] == "FAIL_CLOSED"


def test_f_a_mutated_capture_fails_closed(tmp_path):
    """Changing any recorded field breaks the self-hash. This is the check that
    did not exist before, and it needs no external reference value."""
    manifest = _write_manifest(tmp_path / "capture")
    document = json.loads(manifest.read_text())
    document["status"] = "INCOMPLETE"
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    exit_code, fields = _run_verifier(manifest)
    assert exit_code == 3
    assert fields["capture_manifest_self_verified"] == "false"
    assert fields["capture_manifest_integrity"] == "FAIL_CLOSED"


def test_g_a_pure_reformat_changes_the_file_hash_but_not_the_content_hash(tmp_path):
    """Exactly how one recorded hash can stop matching a file whose EVIDENCE is
    untouched -- and why the content hash is the durable identity."""
    manifest = _write_manifest(tmp_path / "capture")
    original_file_hash = _file_sha256(manifest)
    original_content = json.loads(manifest.read_text())["manifest_sha256"]

    document = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(document, separators=(",", ":"), sort_keys=True), encoding="utf-8")

    assert _file_sha256(manifest) != original_file_hash
    exit_code, fields = _run_verifier(manifest)
    assert exit_code == 0
    assert fields["capture_manifest_self_verified"] == "true"
    assert fields["capture_manifest_content_sha256"] == original_content


def test_h_a_manifest_without_a_self_hash_cannot_be_self_verified(tmp_path):
    manifest = _write_manifest(tmp_path / "capture", with_self_hash=False)
    exit_code, fields = _run_verifier(manifest)
    assert exit_code == 3
    assert fields["capture_manifest_self_verified"] == "false"


@pytest.mark.parametrize("payload", ["not json at all", '["a", "list"]'])
def test_i_an_unreadable_or_non_object_manifest_fails_closed(tmp_path, payload):
    directory = tmp_path / "capture"
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text(payload, encoding="utf-8")
    exit_code, _ = _run_verifier(manifest)
    assert exit_code == 3


def test_j_a_missing_manifest_fails_closed(tmp_path):
    exit_code, _ = _run_verifier(tmp_path / "nope" / "manifest.json")
    assert exit_code == 3


def test_k_verification_never_rewrites_the_capture(tmp_path):
    """The capture is immutable evidence. Verifying it must be a pure read,
    including in the fail-closed paths."""
    manifest = _write_manifest(tmp_path / "capture")
    before = manifest.read_bytes()
    _run_verifier(manifest)
    _run_verifier(manifest, "--declared-sha256", "0" * 64)
    _run_verifier(manifest, "--declared-sha256", _file_sha256(manifest))
    assert manifest.read_bytes() == before


def test_l_the_certified_card_orchestrator_uses_the_verifier(tmp_path):
    """And no longer decides capture identity with a bare sha256sum compare."""
    text = CERTIFIED_SCRIPT.read_text()
    assert "verify_capture_manifest_integrity.py" in text
    assert 'sha256sum "${CAPTURE_MANIFEST}"' not in text
    # The workflow parses this exact key out of the orchestrator's output.
    assert "capture_manifest_sha256" in (REPO_ROOT / "scripts" / "verify_capture_manifest_integrity.py").read_text()


# ===========================================================================
# PART 2 -- legitimate no-op vs missing input.
# ===========================================================================
@pytest.fixture
def estate_with_baseline(tmp_path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir(parents=True)
    _write_baseline(root)
    return root


def _report_only(artifact_root: Path) -> int:
    return candidate_mod.main(["--report-only", "--artifact-root", str(artifact_root)])


def test_m_below_the_maturity_floor_is_exit_zero(estate_with_baseline, capsys):
    """The expected steady state for most of the season: a candidate exists,
    the baseline is active, promotion is NOT_YET_MATURE. Exit 0."""
    _write_maturity_ledger(estate_with_baseline, ps.PROMOTION_ELIGIBLE_MIN_GAMES - 1)
    _write_candidate(estate_with_baseline, _seed())

    assert _report_only(estate_with_baseline) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["promotion_decision"] is None
    assert summary["active_calibrator_source"] == rc.SOURCE_BASELINE
    assert summary["certified_baseline_present"] is True
    assert summary["active_calibrator_status"] == rc.ACTIVE_BASELINE_OK


def test_n_a_missing_certified_baseline_seed_fails_closed(tmp_path, capsys):
    """The exact production state reported as healthy before this fix:
    certified_baseline_present=false, BASELINE_SEED_MISSING, exit 0. It is now
    exit 3, and it is caught even though promotion is legitimately
    NOT_YET_MATURE and never resolves a calibrator itself."""
    root = tmp_path / "artifacts"
    root.mkdir(parents=True)
    _write_maturity_ledger(root, 2)

    assert _report_only(root) == 3
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["certified_baseline_present"] is False
    assert summary["active_calibrator_status"] == rc.ACTIVE_BASELINE_SEED_MISSING


def test_o_a_corrupt_active_pointer_fails_closed(estate_with_baseline, capsys):
    """And never silently degrades to the baseline."""
    _write_maturity_ledger(estate_with_baseline, 2)
    pointer = rc.active_pointer_path(estate_with_baseline)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text("{ not json")

    assert _report_only(estate_with_baseline) == 3
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["active_calibrator_resolution"] == "FAIL_CLOSED"


def test_p_candidate_generation_failure_exits_one(estate_with_baseline, monkeypatch, capsys):
    """A host missing a required historical estate raises here. Exit 1, so the
    daily pass fails closed instead of warning and reporting OK."""
    _write_maturity_ledger(estate_with_baseline, 2)

    def _boom(**_kwargs):
        raise FileNotFoundError(
            "$NFL_MODEL_DATA_ROOT/odds-api-history-2020-2023 does not exist on this host"
        )

    monkeypatch.setattr(candidate_mod, "generate_candidate", _boom)
    assert candidate_mod.main(["--artifact-root", str(estate_with_baseline)]) == 1
    assert "CANDIDATE_GENERATION_FAILED" in capsys.readouterr().err


def test_q_an_unresolvable_artifact_root_exits_two(monkeypatch, capsys):
    def _boom():
        raise RuntimeError("NFL_MODEL_ARTIFACT_ROOT is not set")

    monkeypatch.setattr(candidate_mod, "artifact_root", _boom)
    assert candidate_mod.main([]) == 2
    assert "FAIL_CLOSED" in capsys.readouterr().err


def test_r_promotion_at_the_floor_still_exits_zero(estate_with_baseline, capsys):
    """The other end of the same contract: a real promotion is a success."""
    _write_maturity_ledger(estate_with_baseline, ps.PROMOTION_ELIGIBLE_MIN_GAMES)
    candidate = _write_candidate(estate_with_baseline, _seed())
    promotion = candidate_mod.promote_if_eligible(
        artifact_root_path=estate_with_baseline,
        operational_root=estate_with_baseline,
        repo_root=REPO_ROOT,
    )
    assert promotion["status"] == rc.DECISION_PROMOTED

    assert _report_only(estate_with_baseline) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["active_calibrator_candidate_id"] == candidate


# --- the maintenance script's own contract ---------------------------------
def test_s_daily_maintenance_fails_closed_on_every_non_zero_candidate_exit():
    text = DAILY_SCRIPT.read_text()
    assert "generate_2026_recalibration_candidate.py --promote-if-eligible" in text
    # The old behaviour -- warn, continue, then claim a healthy pass -- is gone.
    assert "reporting state only" not in text
    assert 'if [ "${recalibration_exit}" -ne 0 ]; then' in text
    assert "daily_maintenance_status=FAIL_CLOSED" in text
    # OK is echoed exactly once, and only after the fail-closed branch exits.
    assert text.count('echo "daily_maintenance_status=OK"') == 1
    ok_at = text.index('echo "daily_maintenance_status=OK"')
    fail_at = text.index('echo "daily_maintenance_status=FAIL_CLOSED"')
    assert fail_at < ok_at
    # And the daily pass still never publishes.
    assert "publish_wizard_nfl_local.py" not in text


def test_t_the_maintenance_pass_exit_codes_are_distinct():
    """Stage 4's failures must not collide with stage 2's population rejection
    (exit 3), or a scheduler log cannot say what actually broke."""
    text = DAILY_SCRIPT.read_text()
    for pass_exit in ("pass_exit=4", "pass_exit=5", "pass_exit=6", "pass_exit=7", "pass_exit=8"):
        assert pass_exit in text
    assigned = [line for line in text.splitlines() if "pass_exit=" in line and "exit" in line]
    codes = [line.split("pass_exit=")[1].strip().strip('"') for line in assigned]
    assert len(codes) == len(set(codes)), f"duplicate pass exit codes: {codes}"
    assert "3" not in codes, "stage 4 must not reuse the population-rejection exit code"


def test_u_not_yet_mature_is_never_described_as_a_failure():
    text = DAILY_SCRIPT.read_text()
    assert "NOT_YET_MATURE" in text
    assert "NOT_YET_MATURE is a success" in text or "NOT_YET_MATURE day is a success" in text


# ===========================================================================
# PART 3 -- the legacy preregistration must not block the overlay.
# ===========================================================================
def test_v_the_legacy_scientific_refit_authorization_is_still_not_authorized():
    """It must stay False. The historical preregistration is not rewritten."""
    authorization = rc.promotion_authorization(REPO_ROOT)
    assert authorization["authorized"] is False


def test_w_a_valid_candidate_promotes_despite_the_legacy_authorization(estate_with_baseline):
    """The operational overlay -- config/recalibration_promotion_2026.json --
    is what governs an OPERATIONAL promotion. The frozen preregistration's
    answer about a SCIENTIFIC refit is recorded as provenance and must not veto
    it, which is exactly what PR #44 intended."""
    assert rc.promotion_authorization(REPO_ROOT)["authorized"] is False

    _write_maturity_ledger(estate_with_baseline, ps.PROMOTION_ELIGIBLE_MIN_GAMES)
    candidate = _write_candidate(estate_with_baseline, _seed())
    result = rc.promote_candidate(
        estate_with_baseline, repo_root=REPO_ROOT, operational_root=estate_with_baseline
    )

    assert result["status"] == rc.DECISION_PROMOTED
    assert result["candidate_id"] == candidate
    assert rc.resolve_active_calibrator(estate_with_baseline).candidate_id == candidate
    # Recorded, not obeyed.
    assert result["scientific_refit_authorization"]["authorized"] is False


def test_x_the_promotion_event_retains_the_legacy_authorization_as_provenance(estate_with_baseline):
    _write_maturity_ledger(estate_with_baseline, ps.PROMOTION_ELIGIBLE_MIN_GAMES)
    _write_candidate(estate_with_baseline, _seed())
    rc.promote_candidate(
        estate_with_baseline, repo_root=REPO_ROOT, operational_root=estate_with_baseline
    )
    events = rc.list_promotion_events(estate_with_baseline)
    assert len(events) == 1
    assert events[0]["scientific_refit_authorization"]["authorized"] is False
