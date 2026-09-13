"""Proofs for AUTOMATIC 2026 recalibration promotion.

Everything here is HERMETIC and SYNTHETIC: a pytest ``tmp_path`` artifact
root, a synthetic certified baseline seed, synthetic candidates, a synthetic
prospective evaluation ledger, and the repository's own synthetic games
population. No network call, no real capture, no real
``$NFL_MODEL_ARTIFACT_ROOT``, no SSH, no deployment. Nothing in this file can
write into the real production estate.

What is proved, in the order the task listed it:

  A  below the preregistered floor: candidate generated, baseline stays
     active, promotion is an explicit NOT_YET_MATURE no-op;
  B  at/above the floor with a valid candidate: promotion happens
     automatically;
  C  the certified Fix-8 seed is byte-identical before and after a promotion;
  D  re-promoting the active candidate is idempotent -- no second pointer
     write, no second event, no seed copy;
  E  a candidate missing a stream fails closed;
  F  an unfitted conditional calibrator fails closed;
  G  a malformed push state fails closed;
  H  a non-finite coefficient/intercept/scale fails closed;
  I  a calibrator family or config hash that is not the certified one fails
     closed;
  J  a policy hash that no longer matches its lock fails closed;
  K  a tampered candidate seed or manifest fails closed;
  L  training-membership regression fails closed;
  M  a broken active pointer fails PRODUCTION closed, with no silent fallback
     to the baseline, in the resolver AND in preflight AND in a real batch;
  N  production actually prices with a promoted candidate;
  O  forecast and run evidence record the active calibrator and policy hash;
  Q  the published wizard-nfl-pricing-v2 key set is structurally unable to
     absorb the new provenance fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation import prospective_strength_2026 as ps
from nfl_hybrid.evaluation.chronological_oof import training_membership_hash
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.production import recalibration_2026 as rc
from nfl_hybrid.production import run_2026 as prod

REPO_ROOT = Path(__file__).resolve().parents[1]

_GENERATED_AT = "2026-11-24T12:00:00+00:00"
_RESULT_AVAILABLE = "2026-11-24T04:00:00+00:00"


# ===========================================================================
# Synthetic estate builders.
# ===========================================================================
def _stream_state(
    stream: str,
    *,
    game_ids: list[str],
    coefficient: float = 0.08,
    intercept: float = -0.02,
    fitted: bool = True,
    global_scale: float = 1.0,
    bucket_scales: dict | None = None,
    latest_result_available: str | None = _RESULT_AVAILABLE,
) -> dict:
    """One candidate stream state, shaped exactly like the certified seed's
    entries and carrying the REAL frozen calibrator family/config hashes, so a
    default candidate is genuinely promotable rather than promotable only
    because the test relaxed a check."""
    frozen = rc.frozen_calibration_identity()
    market = rc.MARKET_ATS if stream.startswith(rc.MARKET_ATS) else rc.MARKET_TOTAL
    ids = sorted(game_ids)
    conditional = (
        {"fitted": True, "coefficient": [[coefficient]], "intercept": [intercept]}
        if fitted
        else {"fitted": False, "reason": "INSUFFICIENT_HISTORY"}
    )
    return {
        "market": market,
        "n_raw_ready": len(ids),
        "n_labeled_non_push": len(ids),
        "calibrator_family": frozen["calibrator_family"],
        "calibrator_config_hash": frozen["calibrator_config_hash"],
        "conditional_calibrator_state": conditional,
        "push_calibration_config_hash": frozen["push_calibration_config_hash_by_market"][market],
        "push_scale_state": {
            "global_scale": global_scale,
            "bucket_scales": {"NO_PUSH": 0.0} if bucket_scales is None else bucket_scales,
        },
        "latest_included_result_available_at_utc": latest_result_available,
        "training_game_ids": ids,
        "training_game_count": len(ids),
        "training_membership_hash": training_membership_hash(ids),
    }


def _seed(*, n_games: int = 240, streams: tuple[str, ...] = rc.STREAM_NAMES, **overrides) -> dict:
    ids = [f"2026_{i:04d}" for i in range(n_games)]
    return {stream: _stream_state(stream, game_ids=ids, **overrides) for stream in streams}


def _write_candidate(aroot: Path, seed: dict, *, generated_at: str = _GENERATED_AT) -> str:
    result = rc.write_candidate(
        aroot,
        seed=seed,
        generated_for_cutoff_utc=None,
        generated_at_utc=generated_at,
        git_commit="0" * 40,
    )
    return result.candidate_id


def _write_baseline(aroot: Path) -> Path:
    """A synthetic certified Fix-8 baseline seed carrying the same frozen
    family/config hashes the real one does, so the candidate/baseline
    cross-check in :func:`rc.validate_candidate` is genuinely exercised."""
    frozen = rc.frozen_calibration_identity()
    seed = {}
    for stream in rc.STREAM_NAMES:
        market = rc.MARKET_ATS if stream.startswith(rc.MARKET_ATS) else rc.MARKET_TOTAL
        seed[stream] = {
            "market": market,
            "calibrator_family": frozen["calibrator_family"],
            "calibrator_config_hash": frozen["calibrator_config_hash"],
            "push_calibration_config_hash": frozen["push_calibration_config_hash_by_market"][market],
            "conditional_calibrator_state": {"fitted": True, "coefficient": [[0.05]], "intercept": [0.01]},
            "push_scale_state": {"global_scale": 1.0, "bucket_scales": {"NO_PUSH": 0.0}},
        }
    path = rc.certified_seed_path(aroot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seed, indent=2, sort_keys=True))
    return path


def _write_maturity_ledger(operational_root: Path, unique_completed_games: int) -> None:
    """A synthetic prospective evaluation ledger with exactly
    ``unique_completed_games`` attached, completed games -- read back by the
    FROZEN contract (:func:`ps.load_attached_evaluation_records` /
    :func:`ps._unique_completed_games`), never by a bespoke counter."""
    ledger = Path(operational_root) / "production-2026" / "evaluation-ledger" / "TUE"
    ledger.mkdir(parents=True, exist_ok=True)
    for i in range(unique_completed_games):
        game_id = f"2026_completed_{i:04d}"
        (ledger / f"{game_id}.json").write_text(json.dumps({"game_id": game_id, "horizon": "TUE"}))
        (ledger / f"{game_id}.result.json").write_text(
            json.dumps({"game_id": game_id, "result": {"home_score": 24, "away_score": 17}})
        )


@pytest.fixture
def estate(tmp_path) -> dict:
    """A synthetic artifact root holding the immutable certified baseline and
    nothing else: no pointer, no lock, no candidate."""
    aroot = tmp_path / "artifacts"
    aroot.mkdir(parents=True, exist_ok=True)
    baseline = _write_baseline(aroot)
    return {"artifact_root": aroot, "baseline": baseline, "baseline_bytes": baseline.read_bytes()}


def _promote(estate: dict, *, candidate_identifier: str | None = None) -> dict:
    return rc.promote_candidate(
        estate["artifact_root"],
        candidate_identifier=candidate_identifier,
        repo_root=REPO_ROOT,
        operational_root=estate["artifact_root"],
    )


def _evaluate(estate: dict, **kwargs) -> dict:
    return rc.evaluate_promotion(
        estate["artifact_root"],
        operational_root=estate["artifact_root"],
        repo_root=REPO_ROOT,
        **kwargs,
    )


# ===========================================================================
# The policy itself: versioned, hash-locked, and not relaxable.
# ===========================================================================
def test_the_policy_derives_its_game_floor_from_the_frozen_symbol():
    """The threshold exists in exactly ONE place. The policy records WHERE to
    read it; it must not restate the number."""
    policy = rc.load_promotion_policy(REPO_ROOT)
    assert policy.minimum_prospective_games() == ps.PROMOTION_ELIGIBLE_MIN_GAMES == 200
    block = policy.document["minimum_prospective_maturity"]
    assert block["derived_from_module"] == ps.__name__
    assert block["derived_from_symbol"] == "PROMOTION_ELIGIBLE_MIN_GAMES"
    assert block["numeric_threshold_duplicated_here"] is False
    # No literal 200 anywhere in the policy bytes.
    assert "200" not in policy.path.read_text()


def test_the_shipped_policy_enables_promotion_and_declares_the_baseline_immutable():
    policy = rc.load_promotion_policy(REPO_ROOT)
    assert policy.enabled is True
    assert policy.document["fail_closed"] is True
    assert policy.document["certified_baseline"]["immutable"] is True
    assert policy.document["certified_baseline"]["never_overwritten"] is True
    assert policy.required_streams == ("ATS_TUE", "ATS_FRI", "TOTAL_TUE", "TOTAL_FRI")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.__setitem__("schema_version", "something-else"), id="unknown_schema"),
        pytest.param(lambda d: d.__setitem__("fail_closed", False), id="not_fail_closed"),
        pytest.param(lambda d: d.__setitem__("required_streams", ["ATS_TUE"]), id="missing_stream"),
        pytest.param(
            lambda d: d["candidate_requirements"].__setitem__("require_frozen_calibrator_config_hash_match", False),
            id="relaxed_hash_requirement",
        ),
        pytest.param(
            lambda d: d["scientific_invariance"].__setitem__("ridge_alpha_unchanged", False),
            id="broken_scientific_invariance",
        ),
        pytest.param(
            lambda d: d["certified_baseline"].__setitem__("immutable", False),
            id="mutable_baseline",
        ),
        pytest.param(
            lambda d: d["minimum_prospective_maturity"].__setitem__("numeric_threshold_duplicated_here", True),
            id="duplicated_threshold",
        ),
        pytest.param(lambda d: d.__setitem__("prohibited_controls", []), id="no_prohibited_controls"),
    ],
)
def test_a_weakened_policy_is_rejected_before_any_candidate_is_examined(tmp_path, mutate):
    document = json.loads(rc.policy_path(REPO_ROOT).read_text())
    mutate(document)
    fake_repo = tmp_path / "repo"
    (fake_repo / "config").mkdir(parents=True)
    (fake_repo / rc.POLICY_RELATIVE).write_text(json.dumps(document))
    with pytest.raises(rc.PolicyError):
        rc.load_promotion_policy(fake_repo)


def test_no_environment_variable_or_flag_can_force_a_promotion(estate, monkeypatch):
    monkeypatch.setenv("ALLOW_PROMOTION", "true")
    monkeypatch.setenv("FORCE_RECALIBRATION_PROMOTION", "1")
    monkeypatch.setenv("CERTIFIED_CALIBRATOR_PROMOTION_AUTHORIZATION", "yes")
    _write_maturity_ledger(estate["artifact_root"], 10)
    _write_candidate(estate["artifact_root"], _seed())
    decision = _evaluate(estate)
    assert decision["status"] == rc.DECISION_NOT_YET_MATURE
    assert not rc.active_pointer_path(estate["artifact_root"]).is_file()


# ===========================================================================
# A. Below the preregistered floor.
# ===========================================================================
def test_below_the_floor_the_candidate_exists_and_the_baseline_stays_active(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, ps.PROMOTION_ELIGIBLE_MIN_GAMES - 1)
    candidate = _write_candidate(aroot, _seed())

    decision = _evaluate(estate)
    assert decision["status"] == rc.DECISION_NOT_YET_MATURE
    assert decision["eligible"] is False
    assert decision["status"] in rc.NO_OP_DECISIONS
    assert "199" in decision["reason"] and "200" in decision["reason"]

    # The candidate is still recorded, and the active calibrator is the
    # immutable baseline, by the ABSENCE of a pointer.
    assert candidate in [c["candidate_id"] for c in rc.list_candidates(aroot)]
    assert not rc.active_pointer_path(aroot).is_file()
    active = rc.resolve_active_calibrator(aroot)
    assert active.source == rc.SOURCE_BASELINE
    assert active.status == rc.ACTIVE_BASELINE_OK
    assert active.candidate_id is None
    assert rc.list_promotion_events(aroot) == []

    # Demanding a promotion anyway is refused, and still writes nothing.
    with pytest.raises(rc.PromotionRefused, match=rc.DECISION_NOT_YET_MATURE):
        _promote(estate)
    assert not rc.active_pointer_path(aroot).is_file()
    assert estate["baseline"].read_bytes() == estate["baseline_bytes"]


def test_a_pre_maturity_daily_pass_is_a_success_not_a_failure(estate):
    """The daily script's contract: a NOT_YET_MATURE decision is a no-op
    decision, never an exception, so the pass exits 0."""
    _write_maturity_ledger(estate["artifact_root"], 64)
    _write_candidate(estate["artifact_root"], _seed())
    decision = _evaluate(estate)
    assert decision["status"] in rc.NO_OP_DECISIONS
    assert decision["prospective_maturity"]["maturity"] == ps.MATURITY_DESCRIPTIVE


# ===========================================================================
# B / C. At the floor, a valid candidate promotes -- baseline untouched.
# ===========================================================================
def test_at_the_floor_a_valid_candidate_is_promoted_automatically(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, ps.PROMOTION_ELIGIBLE_MIN_GAMES)
    candidate = _write_candidate(aroot, _seed())

    decision = _evaluate(estate)
    assert decision["status"] == rc.DECISION_ELIGIBLE
    assert decision["candidate_id"] == candidate

    result = _promote(estate)
    assert result["status"] == rc.DECISION_PROMOTED
    assert result["promoted"] is True
    assert result["candidate_id"] == candidate

    pointer = json.loads(rc.active_pointer_path(aroot).read_text())
    assert pointer["schema_version"] == rc.ACTIVE_POINTER_SCHEMA_VERSION
    assert pointer["candidate_id"] == candidate
    assert pointer["previous_active_calibrator_source"] == rc.SOURCE_BASELINE
    assert pointer["previous_active_candidate_id"] is None
    assert pointer["prospective_unique_completed_games_at_promotion"] == 200
    assert pointer["promotion_eligible_min_games"] == 200
    assert pointer["recalibration_policy_sha256"] == rc.load_promotion_policy(REPO_ROOT).sha256
    assert sorted(pointer["training_membership_hashes"]) == sorted(rc.STREAM_NAMES)
    assert pointer["certified_baseline_overwritten"] is False
    assert pointer["certified_baseline_immutable"] is True

    active = rc.resolve_active_calibrator(aroot)
    assert active.source == rc.SOURCE_CANDIDATE
    assert active.candidate_id == candidate
    assert active.seed_path == rc.candidate_seed_path(aroot, candidate)

    events = rc.list_promotion_events(aroot)
    assert len(events) == 1
    assert events[0]["pointer"]["candidate_id"] == candidate


def test_the_certified_fix8_seed_is_byte_identical_before_and_after_promotion(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed())
    before = estate["baseline"].read_bytes()

    assert _promote(estate)["status"] == rc.DECISION_PROMOTED

    assert estate["baseline"].read_bytes() == before == estate["baseline_bytes"]
    # And the promoted seed was REFERENCED, never copied: exactly one copy of
    # the candidate seed exists in the whole estate.
    seeds = list(aroot.rglob(rc.CANDIDATE_SEED_FILENAME))
    assert len(seeds) == 1
    assert seeds[0].parent.parent == rc.candidates_dir(aroot)


def test_promotion_creates_no_duplicate_production_or_data_tree(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    candidate = _write_candidate(aroot, _seed())
    before = {p.relative_to(aroot) for p in aroot.rglob("*") if p.is_file()}

    _promote(estate)

    after = {p.relative_to(aroot) for p in aroot.rglob("*") if p.is_file()}
    created = sorted(str(p) for p in after - before)
    # Exactly three small files: the policy lock, the pointer, one event.
    # No second seed, no second calibration directory, no timestamped copies.
    assert len(created) == 3, created
    assert any(p.endswith(rc.POLICY_LOCK_FILENAME) for p in created)
    assert any(p.endswith(rc.ACTIVE_POINTER_FILENAME) for p in created)
    assert any(rc.PROMOTION_EVENTS_DIRNAME in p for p in created)
    assert not any(candidate in p and p.endswith(rc.CANDIDATE_SEED_FILENAME) for p in created)


# ===========================================================================
# D. Idempotence.
# ===========================================================================
def test_repeated_promotion_of_the_same_candidate_is_an_idempotent_noop(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    candidate = _write_candidate(aroot, _seed())

    first = _promote(estate)
    assert first["status"] == rc.DECISION_PROMOTED
    pointer_bytes = rc.active_pointer_path(aroot).read_bytes()
    files_after_first = {p.relative_to(aroot) for p in aroot.rglob("*") if p.is_file()}

    for _ in range(3):
        again = _promote(estate, candidate_identifier=candidate)
        assert again["status"] == "IDEMPOTENT_NOOP"
        assert again["promoted"] is False
        assert again["decision"] == rc.DECISION_ALREADY_ACTIVE

    assert rc.active_pointer_path(aroot).read_bytes() == pointer_bytes
    assert len(rc.list_promotion_events(aroot)) == 1
    assert {p.relative_to(aroot) for p in aroot.rglob("*") if p.is_file()} == files_after_first

    # And the automatic path (no explicit candidate) is a no-op too.
    assert _evaluate(estate)["status"] == rc.DECISION_ALREADY_ACTIVE


def test_no_candidate_at_all_is_a_no_op_decision(estate):
    _write_maturity_ledger(estate["artifact_root"], 220)
    decision = _evaluate(estate)
    assert decision["status"] == rc.DECISION_NO_CANDIDATE
    assert decision["status"] in rc.NO_OP_DECISIONS
    assert not rc.active_pointer_path(estate["artifact_root"]).is_file()


# ===========================================================================
# E - I. Candidate structural / hash defects all fail closed.
# ===========================================================================
def _mature_estate_with(estate: dict, seed: dict) -> str:
    _write_maturity_ledger(estate["artifact_root"], 220)
    return _write_candidate(estate["artifact_root"], seed)


def test_a_candidate_missing_one_stream_fails_closed(estate):
    _mature_estate_with(estate, _seed(streams=("ATS_TUE", "ATS_FRI", "TOTAL_TUE")))
    with pytest.raises(rc.PromotionRefused, match="TOTAL_FRI"):
        _evaluate(estate)
    assert not rc.active_pointer_path(estate["artifact_root"]).is_file()
    assert estate["baseline"].read_bytes() == estate["baseline_bytes"]


def test_an_unfitted_conditional_calibrator_fails_closed(estate):
    _mature_estate_with(estate, _seed(fitted=False))
    with pytest.raises(rc.PromotionRefused, match="not fitted"):
        _evaluate(estate)
    assert not rc.active_pointer_path(estate["artifact_root"]).is_file()


@pytest.mark.parametrize(
    "push_state",
    [
        pytest.param(None, id="push_state_absent"),
        pytest.param("not-a-mapping", id="push_state_not_a_mapping"),
        pytest.param({"bucket_scales": {"NO_PUSH": 0.0}}, id="no_global_scale"),
        pytest.param({"global_scale": 1.0}, id="no_bucket_scales"),
        pytest.param({"global_scale": 1.0, "bucket_scales": []}, id="bucket_scales_not_a_mapping"),
        pytest.param({"global_scale": "1.0", "bucket_scales": {}}, id="global_scale_not_numeric"),
    ],
)
def test_a_malformed_push_state_fails_closed(estate, push_state):
    seed = _seed()
    for stream in seed:
        if push_state is None:
            seed[stream].pop("push_scale_state")
        else:
            seed[stream]["push_scale_state"] = push_state
    _mature_estate_with(estate, seed)
    with pytest.raises(rc.PromotionRefused):
        _evaluate(estate)
    assert not rc.active_pointer_path(estate["artifact_root"]).is_file()


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"coefficient": float("nan")}, id="nan_coefficient"),
        pytest.param({"coefficient": float("inf")}, id="inf_coefficient"),
        pytest.param({"intercept": float("nan")}, id="nan_intercept"),
        pytest.param({"intercept": float("-inf")}, id="neg_inf_intercept"),
        pytest.param({"global_scale": float("nan")}, id="nan_global_scale"),
        pytest.param({"bucket_scales": {"NO_PUSH": float("inf")}}, id="inf_bucket_scale"),
        pytest.param({"bucket_scales": {"NO_PUSH": float("nan")}}, id="nan_bucket_scale"),
    ],
)
def test_non_finite_parameters_fail_closed(estate, overrides):
    _mature_estate_with(estate, _seed(**overrides))
    with pytest.raises(rc.PromotionRefused):
        _evaluate(estate)
    assert not rc.active_pointer_path(estate["artifact_root"]).is_file()


@pytest.mark.parametrize(
    "field",
    ["calibrator_family", "calibrator_config_hash", "push_calibration_config_hash"],
)
def test_a_calibrator_family_or_config_hash_mismatch_fails_closed(estate, field):
    """A candidate fit by a different estimator or a different config is a
    SCIENTIFIC refit and may never be promoted, however mature the sample."""
    seed = _seed()
    for stream in seed:
        seed[stream][field] = "0" * 64
    _mature_estate_with(estate, seed)
    with pytest.raises(rc.PromotionRefused, match=field):
        _evaluate(estate)
    assert not rc.active_pointer_path(estate["artifact_root"]).is_file()


def test_a_candidate_disagreeing_with_the_certified_baseline_hashes_fails_closed(estate):
    """Cross-check against the baseline seed's OWN recorded hashes, not only
    against the recomputed frozen config."""
    aroot = estate["artifact_root"]
    baseline = json.loads(estate["baseline"].read_text())
    baseline["ATS_TUE"]["calibrator_config_hash"] = "f" * 64
    estate["baseline"].write_text(json.dumps(baseline, indent=2, sort_keys=True))
    _mature_estate_with(estate, _seed())
    with pytest.raises(rc.PromotionRefused, match="certified baseline seed"):
        _evaluate(estate)
    assert not rc.active_pointer_path(aroot).is_file()


def test_a_chronologically_impossible_membership_fails_closed(estate):
    """A candidate may not include a result that became available after the
    candidate itself was generated: a future result cannot leak backward."""
    _mature_estate_with(estate, _seed(latest_result_available="2026-12-25T00:00:00+00:00"))
    with pytest.raises(rc.PromotionRefused, match="cannot leak backward"):
        _evaluate(estate)


# ===========================================================================
# J. The policy lock.
# ===========================================================================
def test_the_policy_lock_is_written_first_write_wins_and_then_verified(estate):
    aroot = estate["artifact_root"]
    policy = rc.load_promotion_policy(REPO_ROOT)

    first = rc.ensure_policy_lock(aroot, policy, git_commit="abc123")
    assert first["status"] == "LOCK_CREATED"
    lock = json.loads(rc.policy_lock_path(aroot).read_text())
    assert lock["policy_sha256"] == policy.sha256
    assert lock["policy_schema_version"] == rc.POLICY_SCHEMA_VERSION
    assert lock["git_commit"] == "abc123"
    assert lock["created_at_utc"]

    lock_bytes = rc.policy_lock_path(aroot).read_bytes()
    second = rc.ensure_policy_lock(aroot, policy, git_commit="def456")
    assert second["status"] == "LOCK_VERIFIED"
    # First write wins: a later run never rewrites the lock.
    assert rc.policy_lock_path(aroot).read_bytes() == lock_bytes


def test_a_policy_changed_after_locking_fails_closed(estate, tmp_path):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed())
    rc.ensure_policy_lock(aroot, rc.load_promotion_policy(REPO_ROOT))

    # A DIFFERENT (but still structurally valid) policy: same schema, same
    # invariants, different bytes.
    document = json.loads(rc.policy_path(REPO_ROOT).read_text())
    document["policy_id"] = "an-edited-policy"
    edited_repo = tmp_path / "edited-repo"
    (edited_repo / "config").mkdir(parents=True)
    (edited_repo / rc.POLICY_RELATIVE).write_text(json.dumps(document))
    (edited_repo / "outputs").mkdir(parents=True)
    (edited_repo / rc.PREREGISTRATION_RELATIVE).write_text(
        (REPO_ROOT / rc.PREREGISTRATION_RELATIVE).read_text()
    )

    with pytest.raises(rc.PolicyLockViolation, match="changed after it was locked"):
        rc.evaluate_promotion(aroot, operational_root=aroot, repo_root=edited_repo)
    assert not rc.active_pointer_path(aroot).is_file()


def test_a_corrupt_policy_lock_fails_closed(estate):
    aroot = estate["artifact_root"]
    path = rc.policy_lock_path(aroot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    with pytest.raises(rc.PolicyLockViolation):
        rc.ensure_policy_lock(aroot, rc.load_promotion_policy(REPO_ROOT))


# ===========================================================================
# K. Tampered candidate content.
# ===========================================================================
def test_a_tampered_candidate_seed_fails_closed(estate):
    aroot = estate["artifact_root"]
    candidate = _mature_estate_with(estate, _seed())
    seed_path = rc.candidate_seed_path(aroot, candidate)
    seed = json.loads(seed_path.read_text())
    seed["ATS_TUE"]["conditional_calibrator_state"]["coefficient"] = [[0.5]]
    seed_path.write_text(json.dumps(seed, indent=2, sort_keys=True))

    with pytest.raises(rc.PromotionRefused, match="seed content hash"):
        _evaluate(estate)
    assert not rc.active_pointer_path(aroot).is_file()


def test_a_tampered_candidate_manifest_fails_closed(estate):
    aroot = estate["artifact_root"]
    candidate = _mature_estate_with(estate, _seed())
    manifest_path = rc.candidate_manifest_path(aroot, candidate)
    manifest = json.loads(manifest_path.read_text())
    manifest["per_stream_membership"]["TOTAL_FRI"]["training_membership_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    with pytest.raises(rc.PromotionRefused, match="manifest membership hash"):
        _evaluate(estate)
    assert not rc.active_pointer_path(aroot).is_file()


def test_a_manifest_claiming_another_candidate_id_fails_closed(estate):
    aroot = estate["artifact_root"]
    candidate = _mature_estate_with(estate, _seed())
    manifest_path = rc.candidate_manifest_path(aroot, candidate)
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate_id"] = "deadbeef"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(rc.PromotionRefused, match="claiming candidate_id"):
        _evaluate(estate, candidate_identifier=candidate)


def test_a_tampered_membership_hash_inside_the_seed_fails_closed(estate):
    aroot = estate["artifact_root"]
    seed = _seed()
    seed["ATS_TUE"]["training_membership_hash"] = "9" * 64
    _mature_estate_with(estate, seed)
    with pytest.raises(rc.PromotionRefused, match="does not hash this membership"):
        _evaluate(estate)
    assert not rc.active_pointer_path(aroot).is_file()


# ===========================================================================
# L. Training-membership regression.
# ===========================================================================
def test_training_membership_regression_fails_closed(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed(n_games=240))
    assert _promote(estate)["status"] == rc.DECISION_PROMOTED
    active_before = rc.resolve_active_calibrator(aroot).candidate_id

    # A newer candidate fit on FEWER games -- evidence was lost or rewritten.
    shrunk = _write_candidate(aroot, _seed(n_games=200), generated_at="2026-11-25T12:00:00+00:00")
    with pytest.raises(rc.PromotionRefused, match="REGRESSED"):
        _evaluate(estate, candidate_identifier=shrunk)
    assert rc.resolve_active_calibrator(aroot).candidate_id == active_before


def test_a_diverging_membership_of_equal_size_fails_closed(estate):
    """Same count, different games: not a superset, so history was rewritten."""
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed(n_games=240))
    _promote(estate)

    diverging = {
        stream: _stream_state(stream, game_ids=[f"2026_other_{i:04d}" for i in range(240)])
        for stream in rc.STREAM_NAMES
    }
    other = _write_candidate(aroot, diverging, generated_at="2026-11-25T12:00:00+00:00")
    with pytest.raises(rc.PromotionRefused, match="REGRESSED"):
        _evaluate(estate, candidate_identifier=other)


def test_a_grown_membership_promotes_over_the_active_candidate(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    first = _write_candidate(aroot, _seed(n_games=240))
    _promote(estate)

    grown = _write_candidate(aroot, _seed(n_games=256), generated_at="2026-11-25T12:00:00+00:00")
    result = _promote(estate)
    assert result["status"] == rc.DECISION_PROMOTED
    assert result["candidate_id"] == grown
    pointer = json.loads(rc.active_pointer_path(aroot).read_text())
    assert pointer["previous_active_candidate_id"] == first
    assert pointer["previous_active_calibrator_source"] == rc.SOURCE_CANDIDATE
    assert result["membership_non_regression"]["ATS_TUE"]["games_added"] == 16
    # One event per ACTUAL change -- two promotions, two events.
    assert len(rc.list_promotion_events(aroot)) == 2
    assert estate["baseline"].read_bytes() == estate["baseline_bytes"]


# ===========================================================================
# M. A broken pointer fails production closed, with NO silent fallback.
# ===========================================================================
def _broken_pointer_estate(estate: dict, mutate) -> Path:
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed())
    _promote(estate)
    pointer_path = rc.active_pointer_path(aroot)
    pointer = json.loads(pointer_path.read_text())
    mutate(aroot, pointer)
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True))
    return pointer_path


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda a, p: p.__setitem__("seed_sha256", "0" * 64), id="seed_hash_mismatch"),
        pytest.param(lambda a, p: p.__setitem__("candidate_manifest_sha256", "0" * 64), id="manifest_hash_mismatch"),
        pytest.param(lambda a, p: p.__setitem__("schema_version", "older-pointer-v0"), id="unknown_pointer_schema"),
        pytest.param(lambda a, p: p.__setitem__("candidate_id", "does-not-exist"), id="missing_candidate"),
        pytest.param(lambda a, p: p.__setitem__("candidate_id", None), id="null_candidate_id"),
        pytest.param(lambda a, p: p.__setitem__("seed_path", "/tmp/a-copy/candidate_calibration_seed.json"), id="seed_path_outside_the_estate"),
        pytest.param(lambda a, p: p.__setitem__("recalibration_policy_sha256", "0" * 64), id="policy_hash_not_the_locked_one"),
        pytest.param(
            lambda a, p: rc.candidate_seed_path(a, p["candidate_id"]).unlink(),
            id="promoted_seed_deleted",
        ),
        pytest.param(
            lambda a, p: rc.candidate_manifest_path(a, p["candidate_id"]).unlink(),
            id="promoted_manifest_deleted",
        ),
    ],
)
def test_a_broken_active_pointer_fails_closed_and_never_falls_back(estate, mutate):
    aroot = estate["artifact_root"]
    _broken_pointer_estate(estate, mutate)

    with pytest.raises(rc.ActiveCalibratorError):
        rc.resolve_active_calibrator(aroot)

    # The certified baseline is present and perfectly usable -- and is still
    # NOT silently substituted. That is the whole point.
    assert estate["baseline"].is_file()
    assert estate["baseline"].read_bytes() == estate["baseline_bytes"]

    report = rc.active_calibrator_report(aroot)
    assert report["resolution"] == "FAIL_CLOSED"
    assert report["certified_baseline_present"] is True


def test_a_corrupt_pointer_file_fails_closed(estate):
    aroot = estate["artifact_root"]
    path = rc.active_pointer_path(aroot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all")
    with pytest.raises(rc.ActiveCalibratorError, match="not valid JSON"):
        rc.resolve_active_calibrator(aroot)


def test_a_promoted_candidate_that_stops_being_calibration_ready_fails_closed(estate):
    """Corruption AFTER promotion: the seed still hashes to what the pointer
    records (both are rewritten), but a stream is no longer usable. Production
    must stop rather than price part of the card uncalibrated while claiming
    the promoted calibrator."""
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    candidate = _write_candidate(aroot, _seed())
    _promote(estate)

    seed_path = rc.candidate_seed_path(aroot, candidate)
    seed = json.loads(seed_path.read_text())
    seed["TOTAL_FRI"]["conditional_calibrator_state"] = {"fitted": False}
    seed_path.write_text(json.dumps(seed, indent=2, sort_keys=True))
    pointer_path = rc.active_pointer_path(aroot)
    pointer = json.loads(pointer_path.read_text())
    pointer["seed_sha256"] = rc._sha256_file(seed_path)
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True))

    with pytest.raises(rc.ActiveCalibratorError, match="TOTAL_FRI"):
        rc.resolve_active_calibrator(aroot)


def test_preflight_blocks_on_a_broken_active_pointer(estate):
    aroot = estate["artifact_root"]
    _broken_pointer_estate(estate, lambda a, p: p.__setitem__("seed_sha256", "0" * 64))

    result = prod.run_preflight(artifact_root_path=aroot, games_population_root=aroot)
    assert result["checks"]["active_calibrator"]["status"] == "FAIL_CLOSED"
    assert "active_calibrator_invalid" in result["blocking_problems"]
    assert result["infra_ready"] is False
    assert result["overall_status"] == "NOT_READY"
    # The baseline file is still fine; preflight refuses anyway.
    assert result["checks"]["fix8_calibration_seed"]["status"] == "OK"


def test_a_batch_fails_closed_on_a_broken_active_pointer(estate, tmp_path):
    aroot = estate["artifact_root"]
    _broken_pointer_estate(estate, lambda a, p: p.__setitem__("candidate_id", "does-not-exist"))
    games = _synthetic_games()
    cutoffs = _last_card_cutoffs(games)

    manifest = prod.run_horizon_batch(
        horizon="TUE",
        as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1),
        force=True,
        operational_root=tmp_path / "ops",
        games=games,
        calibrator_root=aroot,
    )
    assert manifest["status"] == "ACTIVE_CALIBRATOR_INVALID"
    assert manifest["status"] in prod.FAIL_CLOSED_STATUSES
    assert manifest["forecast_count"] == 0
    # Fail-closed BEFORE any forecast was written.
    assert not list((tmp_path / "ops" / "production-2026" / "forecast-ledger").rglob("*.json"))


# ===========================================================================
# N / O. Production prices with the promoted candidate and records it.
# ===========================================================================
_SYNTHETIC_SEASON = 2023
_SYNTHETIC_TEAMS = ["BUF", "MIA", "NE", "NYJ", "KC", "LAC", "DEN", "LV"]


def _synthetic_games(seed: int = 11) -> pd.DataFrame:
    """The same hermetic 15-week synthetic population
    ``tests/test_production_card_2026.py`` uses, so a batch here reaches
    genuine OOF status rather than abstaining."""
    rng = np.random.default_rng(seed)
    rows = []
    game_num = 0
    for week in range(1, 16):
        monday = pd.Timestamp(f"{_SYNTHETIC_SEASON}-09-04") + pd.Timedelta(weeks=week - 1)
        for pair in range(4):
            home, away = _SYNTHETIC_TEAMS[pair * 2], _SYNTHETIC_TEAMS[pair * 2 + 1]
            kickoff = monday + pd.Timedelta(days=6, hours=17)
            rows.append({
                "game_id": f"{_SYNTHETIC_SEASON}_{week:02d}_{away}_{home}_{game_num}",
                "season": _SYNTHETIC_SEASON, "week": week, "season_type": "REG",
                "home_team_id": home, "away_team_id": away,
                "scheduled_kickoff_utc": kickoff.tz_localize("UTC"),
                "home_score": int(rng.integers(10, 35)), "away_score": int(rng.integers(10, 35)),
                "neutral_site": False,
            })
            game_num += 1
    return pd.DataFrame(rows)


def _last_card_cutoffs(games: pd.DataFrame) -> dict[str, pd.Timestamp]:
    ledger = he.build_horizon_membership_ledger(games)
    last_week = int(pd.to_numeric(ledger["week"], errors="coerce").max())
    last_card = ledger[ledger["week"] == last_week]
    return {
        "TUE": pd.Timestamp(last_card["tue_cutoff_utc"].iloc[0]),
        "FRI": pd.Timestamp(last_card["fri_cutoff_utc"].iloc[0]),
    }


def test_production_pricing_uses_the_promoted_candidates_parameters(estate):
    """The promoted seed is not merely loadable: the certified apply path
    prices with ITS coefficients, and would price differently under the
    baseline's."""
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed(coefficient=0.42, intercept=0.31))
    _promote(estate)

    active = rc.resolve_active_calibrator(aroot)
    assert active.source == rc.SOURCE_CANDIDATE
    assert active.seed["ATS_TUE"]["conditional_calibrator_state"]["coefficient"] == [[0.42]]

    residual_ledger = pd.DataFrame({
        "game_id": ["g1", "g2"], "season": [2026, 2026], "week": ["01", "01"],
        "target_cutoff_utc": pd.to_datetime(["2026-09-08T16:00:00Z"] * 2, utc=True),
        "result_available_at_utc": pd.to_datetime(["2026-09-14T22:00:00Z"] * 2, utc=True),
        "model_config_hash": ["h", "h"], "feature_state_hash": ["f", "f"],
        "status": ["OOF"] * 2, "uncertainty_eligible": [True] * 2,
        "predicted_margin": [3.0, -2.0], "predicted_total": [44.0, 47.0],
        "margin_residual_sd_oof": [13.0, 13.0], "total_residual_sd_oof": [10.0, 10.0],
        "actual_margin": [np.nan, np.nan], "actual_total": [np.nan, np.nan],
    })
    consensus = pd.DataFrame({
        "game_id": ["g1", "g2"], "consensus_line": [-2.5, 1.5],
        "consensus_novig_probability": [0.55, 0.48],
    })
    market_consensus = {prod.MARKET_ATS: consensus, prod.MARKET_TOTAL: consensus.assign(consensus_line=[44.5, 47.5])}

    via_candidate = prod.price_and_calibrate(
        residual_ledger, horizon="TUE", market_consensus=market_consensus, calibration_seed=active.seed,
    )
    assert set(via_candidate["ATS_TUE"]["calibration_status"]) == {"CALIBRATED"}
    assert via_candidate["ATS_TUE"]["calibrated_conditional_upper_probability"].notna().all()

    baseline_seed = json.loads(estate["baseline"].read_text())
    via_baseline = prod.price_and_calibrate(
        residual_ledger, horizon="TUE", market_consensus=market_consensus, calibration_seed=baseline_seed,
    )
    assert not np.allclose(
        via_candidate["ATS_TUE"]["calibrated_conditional_upper_probability"].to_numpy(float),
        via_baseline["ATS_TUE"]["calibrated_conditional_upper_probability"].to_numpy(float),
    ), "the promoted candidate must actually change the calibrated output"


def test_preflight_reports_a_promoted_candidate_as_the_active_calibrator(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    candidate = _write_candidate(aroot, _seed())
    _promote(estate)

    result = prod.run_preflight(artifact_root_path=aroot, games_population_root=aroot)
    check = result["checks"]["active_calibrator"]
    assert check["status"] == "OK"
    assert check["active_calibrator_source"] == rc.SOURCE_CANDIDATE
    assert check["active_calibrator_candidate_id"] == candidate
    assert check["recalibration_policy_sha256"] == rc.load_promotion_policy(REPO_ROOT).sha256
    assert all(check["streams_calibration_ready"].values())
    assert "active_calibrator_invalid" not in result["blocking_problems"]


def test_preflight_reports_the_baseline_when_nothing_is_promoted(estate):
    result = prod.run_preflight(
        artifact_root_path=estate["artifact_root"], games_population_root=estate["artifact_root"]
    )
    check = result["checks"]["active_calibrator"]
    assert check["status"] == "OK"
    assert check["active_calibrator_source"] == rc.SOURCE_BASELINE
    assert check["active_calibrator_candidate_id"] is None
    assert check["active_calibrator_seed_sha256"] == rc._sha256_file(estate["baseline"])


@pytest.mark.parametrize("promote_first", [False, True])
def test_forecast_and_run_evidence_record_the_active_calibrator(estate, tmp_path, promote_first):
    aroot = estate["artifact_root"]
    expected_candidate = None
    if promote_first:
        _write_maturity_ledger(aroot, 220)
        expected_candidate = _write_candidate(aroot, _seed())
        _promote(estate)
    expected_source = rc.SOURCE_CANDIDATE if promote_first else rc.SOURCE_BASELINE

    games = _synthetic_games()
    cutoffs = _last_card_cutoffs(games)
    operational_root = tmp_path / "ops"
    manifest = prod.run_horizon_batch(
        horizon="TUE",
        as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1),
        force=True,
        operational_root=operational_root,
        games=games,
        calibrator_root=aroot,
    )
    assert manifest["status"] == "SUCCESS"

    # The run manifest.
    assert manifest["input_hashes"]["active_calibrator_source"] == expected_source
    assert manifest["input_hashes"]["active_calibrator_candidate_id"] == expected_candidate
    assert manifest["input_hashes"]["active_calibrator_seed_sha256"]
    readiness = manifest["source_readiness"]["active_calibrator"]
    assert readiness["active_calibrator_source"] == expected_source
    if promote_first:
        assert readiness["recalibration_policy_sha256"] == rc.load_promotion_policy(REPO_ROOT).sha256

    # Every forecast of record.
    written = list((operational_root / "production-2026" / "forecast-ledger" / "TUE").glob("*.json"))
    assert written
    for path in written:
        payload = json.loads(path.read_text())["prediction"]
        assert payload["active_calibrator_source"] == expected_source
        assert payload["active_calibrator_candidate_id"] == expected_candidate
        assert payload["active_calibrator_seed_sha256"] == manifest["input_hashes"]["active_calibrator_seed_sha256"]
        assert payload["recalibration_policy_sha256"] == manifest["input_hashes"]["recalibration_policy_sha256"]

    # And the prospective evaluation ledger's provenance.
    evaluation = list((operational_root / "production-2026" / "evaluation-ledger" / "TUE").glob("*.json"))
    assert evaluation
    provenance = json.loads(evaluation[0].read_text())["provenance"]
    assert provenance["active_calibrator_source"] == expected_source
    assert provenance["active_calibrator_candidate_id"] == expected_candidate


def test_a_hermetic_batch_with_no_calibrator_estate_still_degrades_gracefully(tmp_path):
    """An unavailable calibrator estate is NOT a broken promotion: production
    prices uncalibrated exactly as it did before this module existed."""
    games = _synthetic_games()
    cutoffs = _last_card_cutoffs(games)
    manifest = prod.run_horizon_batch(
        horizon="TUE",
        as_of_utc=cutoffs["TUE"] + pd.Timedelta(minutes=1),
        force=True,
        operational_root=tmp_path / "ops",
        games=games,
        calibrator_root=tmp_path / "empty",
    )
    assert manifest["status"] == "SUCCESS"
    assert manifest["input_hashes"]["active_calibrator_source"] == rc.SOURCE_BASELINE
    assert manifest["input_hashes"]["active_calibrator_seed_sha256"] is None
    assert manifest["source_readiness"]["active_calibrator"]["active_calibrator_status"] == (
        rc.ACTIVE_BASELINE_SEED_MISSING
    )


# ===========================================================================
# Q. The published contract cannot absorb the new provenance fields.
# ===========================================================================
def test_the_public_card_key_order_is_a_closed_allow_list():
    """``wizard-nfl-pricing-v2`` is unchanged, and STRUCTURALLY cannot change:
    the exporter builds an explicit key list and asserts it, so new
    forecast-record provenance can never leak into the public feed."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_export_wizard_nfl_pricing", REPO_ROOT / "scripts" / "export_wizard_nfl_pricing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.SCHEMA_VERSION == "wizard-nfl-pricing-v2"
    for key in (
        "active_calibrator_source",
        "active_calibrator_candidate_id",
        "active_calibrator_seed_sha256",
        "recalibration_policy_sha256",
    ):
        assert key not in module.GAME_KEY_ORDER
    source = (REPO_ROOT / "scripts" / "export_wizard_nfl_pricing.py").read_text()
    assert "assert tuple(public_game.keys()) == GAME_KEY_ORDER" in source


# ===========================================================================
# The reported state a workflow logs every day.
# ===========================================================================
def test_the_daily_state_report_names_the_policy_maturity_and_active_calibrator(estate):
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    candidate = _write_candidate(aroot, _seed())
    _promote(estate)

    report = rc.candidate_state_report(
        artifact_root_path=aroot, operational_root=aroot, repo_root=REPO_ROOT
    )
    assert report["candidate_count"] == 1
    assert report["promotion_policy"]["policy_sha256"] == rc.load_promotion_policy(REPO_ROOT).sha256
    assert report["promotion_policy"]["minimum_prospective_games"] == 200
    assert report["prospective_maturity"]["maturity"] == ps.MATURITY_PROMOTION_ELIGIBLE
    assert report["active_calibrator"]["active_calibrator_candidate_id"] == candidate
    assert report["promotion_status"] == rc.DECISION_ALREADY_ACTIVE
    assert report["promotion_event_count"] == 1
    assert report["certified_calibrator_immutable"] is True
    assert report["certified_calibrator_sha256"] == rc._sha256_file(estate["baseline"])
    # Reporting is read-only: it never creates the lock or promotes.
    assert report["scientific_refit_authorization"]["authorized"] is False


def test_the_state_report_never_creates_the_policy_lock(estate):
    aroot = estate["artifact_root"]
    rc.candidate_state_report(artifact_root_path=aroot, operational_root=aroot, repo_root=REPO_ROOT)
    assert not rc.policy_lock_path(aroot).is_file()


def test_the_state_report_reports_fail_closed_rather_than_raising(estate):
    _broken_pointer_estate(estate, lambda a, p: p.__setitem__("seed_sha256", "0" * 64))
    report = rc.candidate_state_report(
        artifact_root_path=estate["artifact_root"],
        operational_root=estate["artifact_root"],
        repo_root=REPO_ROOT,
    )
    assert report["active_calibrator"]["resolution"] == "FAIL_CLOSED"
    assert report["promotion_decision"]["status"] == "FAIL_CLOSED"


# ===========================================================================
# The daily script's own contract.
# ===========================================================================
def _daily_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_generate_2026_recalibration_candidate",
        REPO_ROOT / "scripts" / "generate_2026_recalibration_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_daily_script_offers_promote_if_eligible_and_no_force_flag():
    parser = _daily_script().build_arg_parser()
    options = {action.dest for action in parser._actions}
    assert "promote_if_eligible" in options
    assert not any("force" in name for name in options)
    assert not any("bypass" in name for name in options)


def test_the_daily_scripts_promotion_step_promotes_when_eligible_and_no_ops_otherwise(estate):
    """The exact function ``--promote-if-eligible`` runs, end to end, against
    a synthetic estate: a no-op below the floor, a real promotion above it."""
    module = _daily_script()
    aroot = estate["artifact_root"]

    _write_maturity_ledger(aroot, ps.PROMOTION_ELIGIBLE_MIN_GAMES - 1)
    candidate = _write_candidate(aroot, _seed())
    below = module.promote_if_eligible(
        artifact_root_path=aroot, operational_root=aroot, repo_root=REPO_ROOT
    )
    assert below["status"] == rc.DECISION_NOT_YET_MATURE
    assert not rc.active_pointer_path(aroot).is_file()

    _write_maturity_ledger(aroot, ps.PROMOTION_ELIGIBLE_MIN_GAMES)
    above = module.promote_if_eligible(
        artifact_root_path=aroot, operational_root=aroot, repo_root=REPO_ROOT
    )
    assert above["status"] == rc.DECISION_PROMOTED
    assert above["candidate_id"] == candidate
    assert rc.resolve_active_calibrator(aroot).candidate_id == candidate
    assert estate["baseline"].read_bytes() == estate["baseline_bytes"]


def test_the_daily_scripts_promotion_step_raises_on_an_integrity_violation(estate):
    module = _daily_script()
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed(fitted=False))
    with pytest.raises(rc.RecalibrationError):
        module.promote_if_eligible(
            artifact_root_path=aroot, operational_root=aroot, repo_root=REPO_ROOT
        )


def test_the_daily_script_reports_without_generating_or_promoting(estate, capsys):
    module = _daily_script()
    aroot = estate["artifact_root"]
    _write_maturity_ledger(aroot, 220)
    _write_candidate(aroot, _seed())

    assert module.main(["--report-only", "--artifact-root", str(aroot)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["active_calibrator_source"] == rc.SOURCE_BASELINE
    assert payload["summary"]["recalibration_policy_sha256"] == rc.load_promotion_policy(REPO_ROOT).sha256
    assert payload["summary"]["certified_baseline_immutable"] is True
    assert payload["summary"]["promotion_eligible_min_games"] == 200
    assert not rc.active_pointer_path(aroot).is_file()
    assert not rc.policy_lock_path(aroot).is_file()


def test_daily_maintenance_promotes_and_fails_closed_on_integrity_violations():
    daily = (REPO_ROOT / "ops" / "wizard" / "run_daily_maintenance.sh").read_text()
    assert "generate_2026_recalibration_candidate.py --promote-if-eligible" in daily
    # Exit 4 (integrity/policy violation) must fail the pass.
    assert "-eq 4 ]" in daily
    assert "exit 4" in daily
    # And the daily pass still never publishes.
    assert "publish_wizard_nfl_local.py" not in daily
