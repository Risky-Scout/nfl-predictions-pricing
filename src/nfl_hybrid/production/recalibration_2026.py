"""Automatic 2026 recalibration: CANDIDATE generation + governed promotion.

WHAT THIS IS
  A candidate generator, a promotion GATE, and the ONE resolver every
  production path uses to find the currently ACTIVE calibrator. It produces
  recalibration candidates automatically (so a candidate always exists and its
  evidence is never assembled by hand at the moment someone wants to change
  production), and it promotes one only when a VERSIONED, HASH-LOCKED
  operational policy and the frozen preregistered sample-maturity firewall
  both say so.

THE TWO SEPARATE GATES, AND WHY THERE ARE TWO
  :func:`promotion_authorization` reads the historical prospective
  preregistration and answers one narrow question: does that document
  authorize a SCIENTIFIC refit -- a new calibrator family or a new calibrator
  config? It answers no, permanently, because the document asserts
  ``invariance.no_scientific_refit``. That is not rewritten here and not
  reinterpreted here.

  :func:`load_promotion_policy` reads the VERSIONED OPERATIONAL overlay
  ``config/recalibration_promotion_2026.json`` and answers a different
  question: may the ALREADY-CERTIFIED calibrator family/config be re-fitted on
  more 2026 evidence once the ALREADY-PREREGISTERED maturity firewall is
  satisfied? It answers yes -- and the policy is only accepted at all if it
  requires the candidate's calibrator family and config hashes to EQUAL the
  certified ones (:func:`_validate_policy_document`). A promotion therefore
  cannot smuggle in new science: it is the same estimator, the same config,
  more data. That is precisely what makes an automatic promotion compatible
  with ``no_scientific_refit`` instead of a violation of it.

  There is deliberately no post-hoc "the candidate must beat the active
  calibrator by X" threshold anywhere in this module. Any such threshold would
  have to be chosen after observing 2026 results.

WHAT THIS IS NOT
  Not a competing calibration engine. Every fit here is the repository's own
  already-certified machinery, called unchanged:

    * ``nfl_hybrid.evaluation.chronological_calibration.build_raw_probabilities``
      -- ATS/TOTAL raw three-way pricing;
    * ``nfl_hybrid.calibration.three_way._fit_conditional_calibrator`` /
      ``_fit_push_scales`` -- the frozen conditional + push calibrator
      families;
    * ``nfl_hybrid.evaluation.chronological_calibration.compute_calibrator_config_hash``
      / ``compute_push_calibration_config_hash`` -- the frozen config hashes;
    * ``nfl_hybrid.evaluation.prospective_strength_2026`` -- the frozen
      prospective sample-maturity and promotion contract.

  The candidate seed it writes is byte-shaped exactly like the certified
  ``production_calibration_seed.json`` (same four stream keys, same
  ``conditional_calibrator_state`` / ``push_scale_state`` structure), because
  it is built by the same calls in the same order that produced the certified
  seed (``scripts/run_fix8_official_oof_calibration.py`` phase 10). No new
  calibration math, no new calibrator family, no new threshold.

WHERE EVERYTHING GOES (no duplicated estates, no copied seeds)
  ``$NFL_MODEL_ARTIFACT_ROOT/``
    ``fix8-official-oof-calibration-2026/production_calibration_seed.json``
        The certified baseline. IMMUTABLE. Never opened for writing by this
        module, never copied, never superseded in place.
    ``production-2026/recalibration-candidates/<candidate_id>/``
        One immutable, content-addressed directory per candidate, each
        carrying its exact per-stream training MEMBERSHIP
        (``training_game_ids`` + a membership hash) and content hashes.
    ``production-2026/recalibration/active_calibrator.json``
        A tiny POINTER. Its absence means the baseline is active. A promotion
        atomically rewrites this one small file to REFERENCE a candidate seed
        that already exists above -- the seed bytes are never duplicated.
    ``production-2026/recalibration/promotion-events/<event_hash>.json``
        Append-only, content-addressed, written ONLY when the active
        calibrator actually changes. Re-promoting the active candidate is an
        idempotent no-op that writes nothing.
    ``production-2026/recalibration-policy/policy_lock.json``
        First-write-wins runtime lock recording the policy SHA256, schema, git
        commit and creation time. Every later run verifies the same hash; a
        changed policy after locking is a hard fail, never a silent accept.

HOW PROMOTION FAILS CLOSED
  :func:`evaluate_promotion` returns a plain no-op decision for the LEGITIMATE
  reasons nothing should happen (``NOT_YET_MATURE`` before the preregistered
  game floor, ``NO_CANDIDATE``, ``ALREADY_ACTIVE``, ``POLICY_DISABLED``) and
  RAISES for everything that indicates corruption, tampering or drift: a
  policy whose schema or invariants do not validate, a policy hash that no
  longer matches its lock, a candidate missing a stream, an unfitted
  conditional calibrator, a malformed push state, a non-finite parameter, a
  calibrator family/config hash that is not the certified one, a seed or
  manifest hash that does not verify, chronologically impossible membership,
  or training membership that regressed. :func:`promote_candidate` promotes
  only a decision that came back eligible.

  An operator cannot manufacture a promotion. No environment variable, CLI
  flag or filesystem marker is consulted anywhere in this module, and a policy
  that declared such a control would be rejected by
  :func:`_validate_policy_document`.

HOW PRODUCTION READS THE ACTIVE CALIBRATOR
  Through :func:`resolve_active_calibrator` and nothing else. No pointer means
  the verified baseline; a valid pointer means the referenced immutable
  candidate; an invalid pointer, a hash that does not match, or a missing or
  unusable referenced candidate raises :class:`ActiveCalibratorError`. It
  never silently falls back from a broken promoted candidate to the baseline,
  because a silent fallback would publish a card that claims one calibrator
  and used another.
"""
from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.calibration.three_way import (
    CalibrationConfig,
    _fit_conditional_calibrator,
    _fit_push_scales,
)
from nfl_hybrid.evaluation import prospective_strength_2026 as ps
from nfl_hybrid.evaluation.chronological_calibration import (
    _LEGACY_PUSH_MARKET_NAME,
    CALIBRATOR_FAMILY,
    MARKET_ATS,
    MARKET_TOTAL,
    ChronologicalCalibrationConfig,
    compute_calibrator_config_hash,
    compute_push_calibration_config_hash,
)
from nfl_hybrid.evaluation.chronological_oof import training_membership_hash
from nfl_hybrid.features import horizon_elo as he

SCHEMA_VERSION = "recalibration-candidate-2026-v1"

CANDIDATE_NAMESPACE = "production-2026/recalibration-candidates"
CANDIDATE_SEED_FILENAME = "candidate_calibration_seed.json"
CANDIDATE_MANIFEST_FILENAME = "candidate_manifest.json"

# --- The versioned operational promotion policy -----------------------------
POLICY_SCHEMA_VERSION = "recalibration-promotion-policy-2026-v1"
POLICY_RELATIVE = Path("config") / "recalibration_promotion_2026.json"
POLICY_NAMESPACE = "production-2026/recalibration-policy"
POLICY_LOCK_FILENAME = "policy_lock.json"
POLICY_LOCK_SCHEMA_VERSION = "recalibration-promotion-policy-lock-2026-v1"

# --- The active-calibrator pointer and its promotion events -----------------
ACTIVE_NAMESPACE = "production-2026/recalibration"
ACTIVE_POINTER_FILENAME = "active_calibrator.json"
ACTIVE_POINTER_SCHEMA_VERSION = "active-calibrator-pointer-2026-v1"
PROMOTION_EVENTS_DIRNAME = "promotion-events"
PROMOTION_EVENT_SCHEMA_VERSION = "recalibration-promotion-event-2026-v1"

# What ``active_calibrator_source`` says when no pointer exists: the immutable
# certified Fix-8 baseline is active. Recorded verbatim in every forecast and
# run manifest so "baseline" is an asserted fact, never an assumption.
SOURCE_BASELINE = "BASELINE"
SOURCE_CANDIDATE = "CANDIDATE"

# Resolver outcomes.
ACTIVE_BASELINE_OK = "BASELINE_OK"
ACTIVE_BASELINE_SEED_MISSING = "BASELINE_SEED_MISSING"
ACTIVE_CANDIDATE_OK = "CANDIDATE_OK"
ACTIVE_ROOT_UNAVAILABLE = "ARTIFACT_ROOT_UNAVAILABLE"

# Promotion decisions. The first four are legitimate no-ops that must NOT fail
# a workflow; anything that is not one of these five raises instead.
DECISION_PROMOTED = "PROMOTED"
DECISION_ELIGIBLE = "ELIGIBLE"
DECISION_NOT_YET_MATURE = "NOT_YET_MATURE"
DECISION_NO_CANDIDATE = "NO_CANDIDATE"
DECISION_ALREADY_ACTIVE = "ALREADY_ACTIVE"
DECISION_POLICY_DISABLED = "POLICY_DISABLED"

NO_OP_DECISIONS: frozenset[str] = frozenset(
    {DECISION_NOT_YET_MATURE, DECISION_NO_CANDIDATE, DECISION_ALREADY_ACTIVE, DECISION_POLICY_DISABLED}
)

# Every candidate requirement the policy must switch ON for this module to
# accept it. A policy that enabled automatic promotion while relaxing any of
# these is rejected: the requirement set is not negotiable per-deployment, only
# the decision to run the machinery at all is.
_MANDATORY_CANDIDATE_REQUIREMENTS: tuple[str, ...] = (
    "require_all_streams",
    "require_conditional_calibrator_fitted",
    "require_valid_push_calibration_state",
    "require_finite_parameters",
    "require_frozen_calibrator_family_match",
    "require_frozen_calibrator_config_hash_match",
    "require_frozen_push_calibration_config_hash_match",
    "require_seed_content_hash_verified",
    "require_manifest_hash_verified",
    "require_chronology_valid_membership",
    "require_training_membership_non_regression",
    "require_candidate_differs_from_active",
)

# Scientific invariants the policy must assert. These are what make an
# automatic promotion a recalibration rather than a refit.
_MANDATORY_SCIENTIFIC_INVARIANCE: tuple[str, ...] = (
    "calibrator_family_unchanged",
    "calibrator_config_unchanged",
    "push_calibration_family_and_config_unchanged",
    "six_elo_features_unchanged",
    "ridge_alpha_unchanged",
    "tue_fri_horizon_semantics_unchanged",
)

# The four independent production streams, identical to
# ``run_2026.STREAM_NAMES``. Never pooled, never collapsed.
STREAM_NAMES: tuple[str, ...] = ("ATS_TUE", "ATS_FRI", "TOTAL_TUE", "TOTAL_FRI")

# The same minimum labelled-sample floor the certified seed was built under
# (``run_2026.MIN_CALIBRATION_SAMPLE_COUNT`` /
# ``scripts/run_fix8_official_oof_calibration.py``). Not tuned here.
MIN_CALIBRATION_SAMPLE_COUNT = 100

# The certified production calibrator this module must never overwrite.
CERTIFIED_SEED_RELATIVE = Path("fix8-official-oof-calibration-2026") / "production_calibration_seed.json"

# The frozen, machine-readable prospective promotion contract. The ONLY place
# a promotion authorization may come from.
PREREGISTRATION_RELATIVE = Path("outputs") / "prospective_2026_strength_preregistration.json"

# The explicit authorization block a future preregistration would have to
# carry for automatic promotion to be legal. Absent today, deliberately.
AUTHORIZATION_KEY = "certified_calibrator_promotion_authorization"

AUTHORIZED = "PROMOTION_AUTHORIZED"
NOT_AUTHORIZED = "PROMOTION_NOT_AUTHORIZED"
AUTHORIZATION_SOURCE_MISSING = "PROMOTION_AUTHORIZATION_SOURCE_MISSING"


class RecalibrationError(RuntimeError):
    """Candidate generation could not complete without fabricating something."""


class PromotionRefused(RecalibrationError):
    """A promotion was refused. Always fail closed -- the active production
    calibrator is never replaced on a caller's word, and the certified
    baseline is never replaced at all."""


class PolicyError(RecalibrationError):
    """The operational promotion policy is missing, unparseable, of an
    unrecognized schema version, or does not assert the invariants this
    module requires before it will promote anything."""


class PolicyLockViolation(PolicyError):
    """The operational policy changed after it was locked. Fail closed rather
    than silently adopting a policy nobody has reviewed against the lock."""


class ActiveCalibratorError(RecalibrationError):
    """The ACTIVE calibrator could not be resolved and verified. Production
    must fail closed on this: never fall back from a broken promoted candidate
    to the baseline, because that would price a card with one calibrator while
    its provenance claims another."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha256_hex(payload: object) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The two frozen production helpers this module borrows. Imported lazily from
# inside the functions that need them because ``run_2026`` imports THIS module
# at the top level (it is the consumer of the active-calibrator resolver);
# reaching back for these two by name keeps that single cycle edge deferred
# without duplicating either of them.
# ---------------------------------------------------------------------------
def _stream_calibration_ready(seed_state: object) -> bool:
    """The EXACT frozen readiness predicate the certified apply path uses --
    :func:`nfl_hybrid.production.run_2026._frozen_stream_calibration_ready`.
    A candidate stream counts as fitted/finite/push-valid only if the very
    same function that decides whether production may emit a calibrated
    probability says so, so a candidate can never pass promotion and then be
    rejected at pricing time."""
    from nfl_hybrid.production.run_2026 import _frozen_stream_calibration_ready

    return bool(_frozen_stream_calibration_ready(seed_state))


def _resolve_git_commit(repo_root: Path) -> str | None:
    from nfl_hybrid.production.run_2026 import _git_commit

    return _git_commit(Path(repo_root))


# ---------------------------------------------------------------------------
# Promotion authorization -- read-only, preregistration-driven.
# ---------------------------------------------------------------------------
def promotion_authorization(repo_root: Path) -> dict:
    """Whether an EXISTING machine-readable preregistered rule authorizes
    automatically promoting a recalibration candidate over the certified
    production calibrator.

    Decided solely by the frozen prospective preregistration document. Three
    possible outcomes:

      * :data:`AUTHORIZATION_SOURCE_MISSING` -- the document is absent or
        unreadable. Fail closed; no promotion.
      * :data:`NOT_AUTHORIZED` -- the document exists and carries no
        :data:`AUTHORIZATION_KEY` block (today's state), or carries one that
        is not explicitly enabled, or asserts
        ``invariance.no_scientific_refit``. Fail closed; no promotion.
      * :data:`AUTHORIZED` -- the document carries an explicitly enabled
        authorization block AND does not assert ``no_scientific_refit``.

    Never consults an argument, an environment variable, a CLI flag or a
    filesystem marker: an operator cannot manufacture an authorization.
    """
    path = Path(repo_root) / PREREGISTRATION_RELATIVE
    if not path.is_file():
        return {
            "status": AUTHORIZATION_SOURCE_MISSING,
            "authorized": False,
            "reason": f"frozen prospective preregistration not found at {path}",
            "preregistration_path": str(path),
            "preregistration_sha256": None,
        }
    try:
        document = json.loads(path.read_text())
    except ValueError as exc:
        return {
            "status": AUTHORIZATION_SOURCE_MISSING,
            "authorized": False,
            "reason": f"frozen prospective preregistration is not valid JSON: {exc}",
            "preregistration_path": str(path),
            "preregistration_sha256": _sha256_file(path),
        }

    base = {
        "preregistration_path": str(path),
        "preregistration_sha256": _sha256_file(path),
        "preregistration_schema_version": document.get("schema_version"),
        "preregistration_hash": document.get(ps._HASH_FIELD),
        "authorization_key": AUTHORIZATION_KEY,
    }

    invariance = document.get("invariance")
    if isinstance(invariance, dict) and invariance.get("no_scientific_refit") is True:
        return {
            **base,
            "status": NOT_AUTHORIZED,
            "authorized": False,
            "reason": (
                "the frozen prospective preregistration asserts invariance.no_scientific_refit=true; "
                "replacing the certified production calibrator is explicitly outside the preregistered contract"
            ),
        }

    block = document.get(AUTHORIZATION_KEY)
    if not isinstance(block, dict) or block.get("enabled") is not True:
        return {
            **base,
            "status": NOT_AUTHORIZED,
            "authorized": False,
            "reason": (
                f"the frozen prospective preregistration carries no explicitly enabled {AUTHORIZATION_KEY!r} "
                "block; automatic promotion of the certified production calibrator is not authorized"
            ),
        }
    return {
        **base,
        "status": AUTHORIZED,
        "authorized": True,
        "reason": "explicitly enabled preregistered promotion authorization",
        "authorization": block,
    }


# ---------------------------------------------------------------------------
# Candidate generation -- existing machinery only.
# ---------------------------------------------------------------------------
def _stream_calibration_state(
    *,
    raw: pd.DataFrame,
    market: str,
    threshold: np.ndarray,
    calibration_config: ChronologicalCalibrationConfig,
    push_config: CalibrationConfig,
) -> dict:
    """One stream's candidate calibration state.

    Structurally identical to ``scripts/run_fix8_official_oof_calibration.py``
    phase 10: fit the conditional calibrator on RAW_READY rows with a resolved
    non-push binary target via the certified
    :func:`nfl_hybrid.calibration.three_way._fit_conditional_calibrator`, and
    fit the push scales on all RAW_READY rows via the certified
    :func:`nfl_hybrid.calibration.three_way._fit_push_scales`. Nothing about
    either family, its config, or the labelled-sample floor is re-decided
    here.
    """
    if "outcome_resolved" not in raw.columns:
        raise RecalibrationError(
            "raw probability frame is missing 'outcome_resolved'; without it an unplayed game "
            "cannot be told apart from a real push and would be fitted as an observed outcome"
        )
    ready_mask = (raw["raw_status"] == "RAW_READY").to_numpy()
    # Only a game that has actually been played can contribute an observed
    # label to either fit. An unplayed game is legitimately RAW_READY -- it is
    # what the forward card prices -- so readiness alone is not evidence.
    resolved_mask = raw["outcome_resolved"].fillna(False).astype(bool).to_numpy()
    fitting_mask = ready_mask & resolved_mask
    binary_numeric = pd.to_numeric(raw["binary_target"], errors="coerce").to_numpy(float)
    labeled_mask = fitting_mask & np.isfinite(binary_numeric)
    n_labeled = int(labeled_mask.sum())
    conditional_upper = raw["raw_conditional_upper_probability"].to_numpy(dtype=float)

    model = None
    if n_labeled >= MIN_CALIBRATION_SAMPLE_COUNT:
        model = _fit_conditional_calibrator(
            conditional_upper[labeled_mask],
            binary_numeric[labeled_mask].astype(int),
            calibration_config.calibration_config,
        )
    if model is not None:
        conditional_state = {
            "fitted": True,
            "coefficient": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
        }
    else:
        conditional_state = {
            "fitted": False,
            "reason": "INSUFFICIENT_HISTORY" if n_labeled < MIN_CALIBRATION_SAMPLE_COUNT else "CALIBRATOR_DECLINED",
        }

    legacy_market = _LEGACY_PUSH_MARKET_NAME[market]
    # A resolved row whose binary target is null landed exactly on the line:
    # a real push. An unresolved row is excluded from the fit entirely rather
    # than relabelled a non-push, since "not yet known" is not an observation
    # of either class.
    actual_push = (fitting_mask & ~np.isfinite(binary_numeric)).astype(int)
    model_push = pd.to_numeric(raw["raw_push_probability"], errors="coerce").to_numpy(float)
    push_state = None
    if fitting_mask.any():
        push_train = pd.DataFrame(
            {
                "model_push_probability": model_push[fitting_mask],
                "actual_push": actual_push[fitting_mask],
                "market_line": np.asarray(threshold, dtype=float)[fitting_mask],
            }
        )
        global_scale, bucket_scales = _fit_push_scales(push_train, legacy_market, push_config)
        push_state = {"global_scale": global_scale, "bucket_scales": bucket_scales}

    # Describes the rows actually fitted, never scheduled rows that merely
    # passed the readiness gate.
    training_game_ids = sorted(str(g) for g in raw.loc[fitting_mask, "game_id"])
    max_result_available = (
        pd.to_datetime(raw.loc[fitting_mask, "result_available_at_utc"], utc=True).max()
        if fitting_mask.any() and "result_available_at_utc" in raw.columns
        else None
    )

    return {
        "market": market,
        "n_raw_ready": int(ready_mask.sum()),
        "n_labeled_non_push": n_labeled,
        "calibrator_family": CALIBRATOR_FAMILY,
        "calibrator_config_hash": compute_calibrator_config_hash(calibration_config),
        "conditional_calibrator_state": conditional_state,
        "push_calibration_config_hash": compute_push_calibration_config_hash(push_config, market),
        "push_scale_state": push_state,
        "latest_included_result_available_at_utc": (
            None if max_result_available is None or pd.isna(max_result_available) else str(max_result_available)
        ),
        # The EXACT membership this candidate was fit on, plus its
        # order-independent hash. Stored so a reviewer can reproduce the
        # candidate from the ledger alone.
        "training_game_ids": training_game_ids,
        "training_game_count": len(training_game_ids),
        "training_membership_hash": training_membership_hash(training_game_ids),
    }


def build_candidate_seed(
    *,
    raw_by_stream: dict[str, pd.DataFrame],
    threshold_by_stream: dict[str, np.ndarray],
    calibration_config: ChronologicalCalibrationConfig | None = None,
    push_config: CalibrationConfig | None = None,
) -> dict:
    """A candidate calibration seed for every stream present in
    ``raw_by_stream``.

    ``raw_by_stream`` maps a stream name to the certified raw-probability
    frame produced by
    :func:`nfl_hybrid.evaluation.chronological_calibration.build_raw_probabilities`
    -- this function never builds probabilities itself, so it cannot introduce
    a competing pricing rule.
    """
    calibration_config = calibration_config or ChronologicalCalibrationConfig()
    push_config = push_config or CalibrationConfig()

    unknown = sorted(set(raw_by_stream) - set(STREAM_NAMES))
    if unknown:
        raise RecalibrationError(f"unknown calibration stream(s) {unknown}; expected a subset of {STREAM_NAMES}")

    seed: dict[str, dict] = {}
    for stream, raw in raw_by_stream.items():
        market = MARKET_ATS if stream.startswith(MARKET_ATS) else MARKET_TOTAL
        if stream not in threshold_by_stream:
            raise RecalibrationError(f"stream {stream!r} has raw probabilities but no market threshold vector")
        seed[stream] = _stream_calibration_state(
            raw=raw,
            market=market,
            threshold=threshold_by_stream[stream],
            calibration_config=calibration_config,
            push_config=push_config,
        )
    return seed


# ---------------------------------------------------------------------------
# Candidate persistence.
# ---------------------------------------------------------------------------
def candidates_dir(artifact_root_path: Path) -> Path:
    return Path(artifact_root_path) / CANDIDATE_NAMESPACE


def candidate_id(seed: dict, *, generated_for_cutoff_utc: str | None) -> str:
    """A candidate's identity: the content hash of its own deterministic seed
    plus the cutoff it was generated for. Two runs over identical evidence
    produce the same candidate id, so regeneration is idempotent rather than
    an ever-growing pile of near-duplicates."""
    return _sha256_hex({"seed": seed, "generated_for_cutoff_utc": generated_for_cutoff_utc})[:32]


@dataclass(frozen=True)
class CandidateWriteResult:
    status: str  # WRITTEN | IDEMPOTENT_NOOP
    candidate_id: str
    directory: Path
    manifest: dict


def write_candidate(
    artifact_root_path: Path,
    *,
    seed: dict,
    generated_for_cutoff_utc: str | None,
    generated_at_utc: str,
    git_commit: str | None,
    games_population_provenance: dict | None = None,
    prospective_maturity: dict | None = None,
    authorization: dict | None = None,
) -> CandidateWriteResult:
    """Persist one candidate immutably under its content-addressed id.

    Idempotent: re-writing the same candidate content returns
    ``IDEMPOTENT_NOOP``. A different payload at the same id is impossible by
    construction (the id IS the payload hash), and the certified production
    seed is never touched from here.
    """
    cid = candidate_id(seed, generated_for_cutoff_utc=generated_for_cutoff_utc)
    directory = candidates_dir(artifact_root_path) / cid
    seed_path = directory / CANDIDATE_SEED_FILENAME
    manifest_path = directory / CANDIDATE_MANIFEST_FILENAME

    seed_bytes = json.dumps(seed, indent=2, sort_keys=True, default=str).encode("utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": cid,
        "candidate_status": "CANDIDATE_ONLY",
        "generated_at_utc": generated_at_utc,
        "generated_for_cutoff_utc": generated_for_cutoff_utc,
        "git_commit": git_commit,
        "streams": sorted(seed),
        "seed_content_sha256": _sha256_hex(seed),
        "per_stream_membership": {
            stream: {
                "training_game_count": state.get("training_game_count"),
                "training_membership_hash": state.get("training_membership_hash"),
                "n_raw_ready": state.get("n_raw_ready"),
                "n_labeled_non_push": state.get("n_labeled_non_push"),
                "conditional_fitted": bool((state.get("conditional_calibrator_state") or {}).get("fitted")),
                "latest_included_result_available_at_utc": state.get("latest_included_result_available_at_utc"),
            }
            for stream, state in sorted(seed.items())
        },
        "games_population": games_population_provenance,
        "prospective_maturity": prospective_maturity,
        "promotion_authorization": authorization,
        "certified_calibrator_replaced": False,
    }

    if manifest_path.is_file() and seed_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("seed_content_sha256") == manifest["seed_content_sha256"]:
            return CandidateWriteResult("IDEMPOTENT_NOOP", cid, directory, existing)

    directory.mkdir(parents=True, exist_ok=True)
    tmp_seed = seed_path.with_name(seed_path.name + ".tmp")
    tmp_seed.write_bytes(seed_bytes)
    tmp_seed.replace(seed_path)
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    tmp_manifest.replace(manifest_path)
    return CandidateWriteResult("WRITTEN", cid, directory, manifest)


def list_candidates(artifact_root_path: Path) -> list[dict]:
    """Every recorded candidate manifest, oldest generation first."""
    root = candidates_dir(artifact_root_path)
    if not root.is_dir():
        return []
    manifests: list[dict] = []
    for manifest_path in sorted(root.glob(f"*/{CANDIDATE_MANIFEST_FILENAME}")):
        try:
            manifests.append(json.loads(manifest_path.read_text()))
        except ValueError:
            continue
    manifests.sort(key=lambda m: (str(m.get("generated_at_utc")), str(m.get("candidate_id"))))
    return manifests


# ---------------------------------------------------------------------------
# Prospective maturity context (reported, never a promotion shortcut).
# ---------------------------------------------------------------------------
def prospective_maturity_state(operational_root: Path) -> dict:
    """The frozen prospective sample-maturity label for the 2026 estate,
    computed by the existing contract
    (:mod:`nfl_hybrid.evaluation.prospective_strength_2026`) over the
    immutable evaluation ledger. The ONE maturity firewall promotion consults;
    its threshold is read from the frozen module, never restated here."""
    evaluation_root = Path(operational_root) / "production-2026" / "evaluation-ledger"
    records = ps.load_attached_evaluation_records(evaluation_root) if evaluation_root.is_dir() else []
    n_games = ps._unique_completed_games(records)
    return {
        "unique_completed_games": int(n_games),
        "maturity": ps.sample_maturity(n_games),
        "promotion_eligible_maturity": ps._promotion_eligible(n_games),
        "promotion_eligible_min_games": ps.PROMOTION_ELIGIBLE_MIN_GAMES,
        "evaluation_ledger": str(evaluation_root),
    }


def certified_seed_path(artifact_root_path: Path) -> Path:
    return Path(artifact_root_path) / CERTIFIED_SEED_RELATIVE


# ===========================================================================
# The versioned operational promotion policy.
# ===========================================================================
@dataclass(frozen=True)
class PromotionPolicy:
    """A loaded, schema-checked, invariant-checked operational policy, plus
    the SHA256 of the exact bytes on disk (what the runtime lock pins)."""

    path: Path
    sha256: str
    document: dict = field(repr=False)

    @property
    def enabled(self) -> bool:
        return self.document.get("automatic_promotion_enabled") is True

    @property
    def required_streams(self) -> tuple[str, ...]:
        return tuple(self.document["required_streams"])

    def minimum_prospective_games(self) -> int:
        """The promotion game floor, RESOLVED from the frozen symbol the
        policy names rather than from any number written in the policy.

        The policy says WHERE the threshold lives
        (``nfl_hybrid.evaluation.prospective_strength_2026``.
        ``PROMOTION_ELIGIBLE_MIN_GAMES``); this reads it from there. The
        number therefore exists in exactly one place in the whole system and
        cannot drift between the preregistered contract and the operational
        overlay."""
        block = self.document["minimum_prospective_maturity"]
        module = importlib.import_module(block["derived_from_module"])
        value = getattr(module, block["derived_from_symbol"])
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PolicyError(
                f"{block['derived_from_module']}.{block['derived_from_symbol']} is {value!r}; "
                "expected a positive integer game floor"
            )
        return int(value)

    def required_maturity_label(self) -> str:
        return str(self.document["minimum_prospective_maturity"]["required_maturity_label"])

    def summary(self) -> dict:
        """The provenance every run records: which policy, which bytes."""
        return {
            "policy_path": str(self.path),
            "policy_sha256": self.sha256,
            "policy_schema_version": self.document.get("schema_version"),
            "policy_id": self.document.get("policy_id"),
            "automatic_promotion_enabled": self.enabled,
            "required_streams": list(self.required_streams),
            "minimum_prospective_games": self.minimum_prospective_games(),
            "minimum_prospective_games_source": (
                f"{self.document['minimum_prospective_maturity']['derived_from_module']}."
                f"{self.document['minimum_prospective_maturity']['derived_from_symbol']}"
            ),
        }


def policy_path(repo_root: Path) -> Path:
    return Path(repo_root) / POLICY_RELATIVE


def _validate_policy_document(document: object, *, path: Path) -> dict:
    """Accept a policy only if it is the recognized schema AND asserts every
    invariant that makes an automatic promotion safe.

    This is the structural reason an operator cannot weaken the gate by
    editing the policy: a policy that enables promotion while switching off a
    hash check, omitting a stream, declaring itself not fail-closed, allowing
    the baseline to be overwritten, restating the maturity threshold as a
    literal, or naming a manual-bypass control is REJECTED here, before any
    candidate is even looked at."""
    if not isinstance(document, dict):
        raise PolicyError(f"{path}: promotion policy must be a JSON object, got {type(document).__name__}")

    schema = document.get("schema_version")
    if schema != POLICY_SCHEMA_VERSION:
        raise PolicyError(
            f"{path}: unrecognized promotion-policy schema_version {schema!r}; this build understands "
            f"only {POLICY_SCHEMA_VERSION!r}"
        )
    if document.get("fail_closed") is not True:
        raise PolicyError(f"{path}: promotion policy must declare fail_closed=true")

    baseline = document.get("certified_baseline")
    if not isinstance(baseline, dict):
        raise PolicyError(f"{path}: promotion policy must carry a certified_baseline block")
    if baseline.get("immutable") is not True or baseline.get("never_overwritten") is not True:
        raise PolicyError(
            f"{path}: promotion policy must declare the certified baseline immutable=true and "
            "never_overwritten=true"
        )
    if baseline.get("artifact_relative_path") != CERTIFIED_SEED_RELATIVE.as_posix():
        raise PolicyError(
            f"{path}: promotion policy names certified baseline "
            f"{baseline.get('artifact_relative_path')!r}, but this build's certified baseline is "
            f"{CERTIFIED_SEED_RELATIVE.as_posix()!r}"
        )

    streams = document.get("required_streams")
    if not isinstance(streams, list) or tuple(streams) != STREAM_NAMES:
        raise PolicyError(
            f"{path}: promotion policy required_streams must be exactly {list(STREAM_NAMES)}, got {streams!r}"
        )

    maturity = document.get("minimum_prospective_maturity")
    if not isinstance(maturity, dict):
        raise PolicyError(f"{path}: promotion policy must carry a minimum_prospective_maturity block")
    if maturity.get("derived_from_module") != ps.__name__ or maturity.get("derived_from_symbol") != "PROMOTION_ELIGIBLE_MIN_GAMES":
        raise PolicyError(
            f"{path}: the promotion game floor must be derived from {ps.__name__}."
            f"PROMOTION_ELIGIBLE_MIN_GAMES, not {maturity.get('derived_from_module')!r}."
            f"{maturity.get('derived_from_symbol')!r}"
        )
    if maturity.get("required_maturity_label") != ps.MATURITY_PROMOTION_ELIGIBLE:
        raise PolicyError(
            f"{path}: required_maturity_label must be {ps.MATURITY_PROMOTION_ELIGIBLE!r}, got "
            f"{maturity.get('required_maturity_label')!r}"
        )
    if maturity.get("numeric_threshold_duplicated_here") is not False:
        raise PolicyError(
            f"{path}: minimum_prospective_maturity must declare "
            "numeric_threshold_duplicated_here=false and must not restate the threshold"
        )

    requirements = document.get("candidate_requirements")
    if not isinstance(requirements, dict):
        raise PolicyError(f"{path}: promotion policy must carry a candidate_requirements block")
    relaxed = [name for name in _MANDATORY_CANDIDATE_REQUIREMENTS if requirements.get(name) is not True]
    if relaxed:
        raise PolicyError(
            f"{path}: promotion policy relaxes non-negotiable candidate requirement(s) {relaxed}; "
            "every one of them must be true"
        )

    invariance = document.get("scientific_invariance")
    if not isinstance(invariance, dict):
        raise PolicyError(f"{path}: promotion policy must carry a scientific_invariance block")
    broken = [name for name in _MANDATORY_SCIENTIFIC_INVARIANCE if invariance.get(name) is not True]
    if broken:
        raise PolicyError(
            f"{path}: promotion policy does not assert scientific invariance {broken}; an operational "
            "recalibration may only re-fit the already-certified family/config on more data"
        )

    prohibited = document.get("prohibited_controls")
    if not isinstance(prohibited, list) or not prohibited:
        raise PolicyError(
            f"{path}: promotion policy must list the prohibited_controls it forbids (manual force, "
            "environment override, CLI bypass, filesystem marker, results-tuned threshold)"
        )

    return document


def load_promotion_policy(repo_root: Path) -> PromotionPolicy:
    """Load, hash and validate the operational promotion policy.

    Raises :class:`PolicyError` when it is absent, unparseable, of an
    unrecognized schema version, or fails :func:`_validate_policy_document`.
    Nothing downstream can promote without a policy that got through here."""
    path = policy_path(repo_root)
    if not path.is_file():
        raise PolicyError(f"operational recalibration promotion policy not found at {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PolicyError(f"{path}: promotion policy is not valid JSON: {exc}") from exc
    _validate_policy_document(document, path=path)
    return PromotionPolicy(path=path, sha256=sha256(raw).hexdigest(), document=document)


# ===========================================================================
# The first-write-wins policy lock.
# ===========================================================================
def policy_lock_path(artifact_root_path: Path) -> Path:
    return Path(artifact_root_path) / POLICY_NAMESPACE / POLICY_LOCK_FILENAME


def ensure_policy_lock(
    artifact_root_path: Path,
    policy: PromotionPolicy,
    *,
    git_commit: str | None = None,
    now_utc: str | None = None,
    create: bool = True,
) -> dict:
    """Pin the policy on first operational use, then verify it forever.

    The first run writes one small lock recording the policy SHA256, its
    schema version, the git commit in effect and the creation timestamp. The
    write is ``O_CREAT|O_EXCL`` so two concurrent runs cannot both claim to
    have created it -- genuinely first-write-wins, not last-writer-wins.

    Every later run compares the live policy hash against the lock. A
    mismatch raises :class:`PolicyLockViolation`: the policy changed after
    being locked, and silently adopting it would let an edit to a config file
    change what production promotes with no review. ``create=False`` verifies
    an existing lock without creating one (the read-only reporting path)."""
    path = policy_lock_path(artifact_root_path)
    if path.is_file():
        try:
            lock = json.loads(path.read_text())
        except ValueError as exc:
            raise PolicyLockViolation(f"{path}: policy lock is not valid JSON: {exc}") from exc
        locked_hash = lock.get("policy_sha256")
        if locked_hash != policy.sha256:
            raise PolicyLockViolation(
                f"{path}: the operational promotion policy changed after it was locked "
                f"(locked policy_sha256={locked_hash}, current={policy.sha256} from {policy.path}). "
                "Failing closed: a reviewed policy lock is never silently replaced."
            )
        if lock.get("policy_schema_version") != policy.document.get("schema_version"):
            raise PolicyLockViolation(
                f"{path}: locked policy_schema_version {lock.get('policy_schema_version')!r} does not "
                f"match the current policy's {policy.document.get('schema_version')!r}"
            )
        return {"status": "LOCK_VERIFIED", "lock_path": str(path), "lock": lock}

    if not create:
        return {"status": "LOCK_ABSENT", "lock_path": str(path), "lock": None}

    lock = {
        "schema_version": POLICY_LOCK_SCHEMA_VERSION,
        "policy_sha256": policy.sha256,
        "policy_schema_version": policy.document.get("schema_version"),
        "policy_id": policy.document.get("policy_id"),
        "policy_path": str(policy.path),
        "git_commit": git_commit,
        "created_at_utc": now_utc or pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(lock, indent=2, sort_keys=True, default=str)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        # Another run won the race between the is_file() check above and here.
        # Re-enter to VERIFY against whatever it wrote rather than overwrite.
        return ensure_policy_lock(
            artifact_root_path, policy, git_commit=git_commit, now_utc=now_utc, create=False
        )
    with os.fdopen(handle, "w") as fh:
        fh.write(payload)
    return {"status": "LOCK_CREATED", "lock_path": str(path), "lock": lock}


# ===========================================================================
# The frozen certified calibration identity a candidate must match.
# ===========================================================================
def frozen_calibration_identity() -> dict:
    """The certified calibrator family and config hashes, recomputed from the
    frozen default configs -- the same two calls
    ``scripts/run_fix8_official_oof_calibration.py`` made when it produced the
    certified seed. A candidate whose family/config hashes differ from these
    was fit by a DIFFERENT estimator or a DIFFERENT config and may never be
    promoted, because that would be a scientific refit."""
    push_config = CalibrationConfig()
    return {
        "calibrator_family": CALIBRATOR_FAMILY,
        "calibrator_config_hash": compute_calibrator_config_hash(ChronologicalCalibrationConfig()),
        "push_calibration_config_hash_by_market": {
            MARKET_ATS: compute_push_calibration_config_hash(push_config, MARKET_ATS),
            MARKET_TOTAL: compute_push_calibration_config_hash(push_config, MARKET_TOTAL),
        },
    }


def _market_for_stream(stream: str) -> str:
    return MARKET_ATS if stream.startswith(MARKET_ATS) else MARKET_TOTAL


def _finite_parameters(state: dict) -> tuple[bool, str]:
    """Every numeric parameter the certified apply path will dereference is
    finite. :func:`_stream_calibration_ready` already covers the conditional
    coefficient/intercept and the push global scale; this additionally walks
    every push bucket scale, which the apply path indexes by market line."""
    push = state.get("push_scale_state")
    if not isinstance(push, dict):
        return False, "push_scale_state is not a mapping"
    buckets = push.get("bucket_scales")
    if not isinstance(buckets, dict):
        return False, "push_scale_state.bucket_scales is not a mapping"
    for key, value in buckets.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            return False, f"push bucket scale {key!r} is not a finite number ({value!r})"
    return True, "all parameters finite"


# ===========================================================================
# Candidate structural + hash validation.
# ===========================================================================
def candidate_seed_path(artifact_root_path: Path, candidate_identifier: str) -> Path:
    return candidates_dir(artifact_root_path) / candidate_identifier / CANDIDATE_SEED_FILENAME


def candidate_manifest_path(artifact_root_path: Path, candidate_identifier: str) -> Path:
    return candidates_dir(artifact_root_path) / candidate_identifier / CANDIDATE_MANIFEST_FILENAME


def _read_json_file(path: Path, *, what: str) -> object:
    if not path.is_file():
        raise PromotionRefused(f"{what} not found at {path}")
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise PromotionRefused(f"{path}: {what} is not valid JSON: {exc}") from exc


def validate_candidate(
    artifact_root_path: Path,
    candidate_identifier: str,
    *,
    policy: PromotionPolicy,
) -> dict:
    """Every structural, hash and chronology requirement the policy demands of
    a candidate before it may become the active calibrator.

    Raises :class:`PromotionRefused` on the FIRST failure, having written
    nothing. Returns the verified seed plus the evidence of what was checked,
    so the promotion event can record it.

    What is proved here:

      * the seed and manifest both exist and parse;
      * the manifest is this build's candidate schema and names this exact
        candidate id;
      * the seed's content hash equals the manifest's recorded
        ``seed_content_sha256`` (the seed has not been edited under the
        manifest);
      * re-deriving the content-addressed candidate id from the seed and the
        manifest's cutoff reproduces the directory name (the manifest and the
        seed belong to each other and to this id);
      * every per-stream membership hash and count in the manifest matches the
        seed it summarizes;
      * all four production streams are present;
      * each stream passes the EXACT frozen readiness predicate production
        pricing uses -- fitted conditional calibrator, finite
        coefficient/intercept, well-formed push state -- plus a finite check
        over every push bucket scale;
      * each stream's calibrator family and both config hashes equal the
        frozen certified ones (and the certified baseline seed's own recorded
        hashes, when that seed is present);
      * each stream's training membership is internally consistent (unique,
        hash-verified) and chronologically possible (nothing included whose
        result became available after the candidate was generated, and --
        when the candidate declares a cutoff -- nothing at or after it)."""
    directory = candidates_dir(artifact_root_path) / candidate_identifier
    manifest = _read_json_file(
        candidate_manifest_path(artifact_root_path, candidate_identifier), what="candidate manifest"
    )
    seed = _read_json_file(
        candidate_seed_path(artifact_root_path, candidate_identifier), what="candidate seed"
    )
    if not isinstance(manifest, dict) or not isinstance(seed, dict):
        raise PromotionRefused(f"candidate {candidate_identifier!r}: manifest and seed must both be JSON objects")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r}: manifest schema_version "
            f"{manifest.get('schema_version')!r} is not this build's {SCHEMA_VERSION!r}"
        )
    if manifest.get("candidate_id") != candidate_identifier:
        raise PromotionRefused(
            f"candidate directory {candidate_identifier!r} holds a manifest claiming candidate_id "
            f"{manifest.get('candidate_id')!r}"
        )

    seed_content_sha256 = _sha256_hex(seed)
    if manifest.get("seed_content_sha256") != seed_content_sha256:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r}: seed content hash {seed_content_sha256} does not match the "
            f"manifest's recorded {manifest.get('seed_content_sha256')}; the seed or the manifest was altered"
        )
    rederived = candidate_id(seed, generated_for_cutoff_utc=manifest.get("generated_for_cutoff_utc"))
    if rederived != candidate_identifier:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r}: re-deriving the content-addressed id from this seed and "
            f"cutoff yields {rederived!r}; the manifest and seed do not belong to this candidate id"
        )

    streams = sorted(seed)
    missing = [s for s in policy.required_streams if s not in seed]
    if missing:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} is missing required production stream(s) {missing}; "
            f"the policy requires all of {list(policy.required_streams)}"
        )
    extra = [s for s in streams if s not in policy.required_streams]
    if extra:
        raise PromotionRefused(f"candidate {candidate_identifier!r} carries unknown stream(s) {extra}")

    frozen = frozen_calibration_identity()
    certified = certified_seed_path(artifact_root_path)
    certified_seed = None
    if certified.is_file():
        try:
            loaded = json.loads(certified.read_text())
        except ValueError as exc:
            raise PromotionRefused(f"{certified}: the certified baseline seed is not valid JSON: {exc}") from exc
        certified_seed = loaded if isinstance(loaded, dict) else None

    generated_at = _coerce_utc(manifest.get("generated_at_utc"))
    declared_cutoff = _coerce_utc(manifest.get("generated_for_cutoff_utc"))
    per_stream: dict[str, dict] = {}

    for stream in policy.required_streams:
        state = seed[stream]
        if not isinstance(state, dict):
            raise PromotionRefused(f"candidate {candidate_identifier!r} stream {stream}: state is not a mapping")

        if not _stream_calibration_ready(state):
            conditional = state.get("conditional_calibrator_state")
            reason = (
                "conditional calibrator is not fitted"
                if not (isinstance(conditional, dict) and conditional.get("fitted"))
                else "conditional coefficient/intercept or push_scale_state is missing or malformed"
            )
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: {reason}. A candidate is promotable only "
                "when every stream satisfies the same frozen readiness predicate production pricing applies, "
                f"so this candidate may not become active (detail: {conditional!r})"
            )
        finite_ok, finite_detail = _finite_parameters(state)
        if not finite_ok:
            raise PromotionRefused(f"candidate {candidate_identifier!r} stream {stream}: {finite_detail}")

        market = _market_for_stream(stream)
        expected_push_hash = frozen["push_calibration_config_hash_by_market"][market]
        if state.get("calibrator_family") != frozen["calibrator_family"]:
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: calibrator_family "
                f"{state.get('calibrator_family')!r} is not the certified {frozen['calibrator_family']!r}"
            )
        if state.get("calibrator_config_hash") != frozen["calibrator_config_hash"]:
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: calibrator_config_hash "
                f"{state.get('calibrator_config_hash')} is not the certified {frozen['calibrator_config_hash']}; "
                "promoting it would be a scientific refit, which this policy can never authorize"
            )
        if state.get("push_calibration_config_hash") != expected_push_hash:
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: push_calibration_config_hash "
                f"{state.get('push_calibration_config_hash')} is not the certified {expected_push_hash}"
            )
        if isinstance(certified_seed, dict) and isinstance(certified_seed.get(stream), dict):
            baseline_state = certified_seed[stream]
            for key in ("calibrator_family", "calibrator_config_hash", "push_calibration_config_hash"):
                if key in baseline_state and baseline_state[key] != state.get(key):
                    raise PromotionRefused(
                        f"candidate {candidate_identifier!r} stream {stream}: {key} "
                        f"{state.get(key)!r} differs from the certified baseline seed's {baseline_state[key]!r}"
                    )

        membership = _validated_membership(
            state,
            candidate_identifier=candidate_identifier,
            stream=stream,
            generated_at=generated_at,
            declared_cutoff=declared_cutoff,
        )

        summary = (manifest.get("per_stream_membership") or {}).get(stream)
        if not isinstance(summary, dict):
            raise PromotionRefused(
                f"candidate {candidate_identifier!r}: manifest carries no per-stream membership summary for {stream}"
            )
        if summary.get("training_membership_hash") != membership["training_membership_hash"]:
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: manifest membership hash "
                f"{summary.get('training_membership_hash')} does not match the seed's "
                f"{membership['training_membership_hash']}"
            )
        if int(summary.get("training_game_count") or -1) != membership["training_game_count"]:
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: manifest membership count "
                f"{summary.get('training_game_count')!r} does not match the seed's "
                f"{membership['training_game_count']}"
            )
        per_stream[stream] = membership

    return {
        "candidate_id": candidate_identifier,
        "directory": str(directory),
        "seed": seed,
        "manifest": manifest,
        "seed_path": str(candidate_seed_path(artifact_root_path, candidate_identifier)),
        "seed_file_sha256": _sha256_file(candidate_seed_path(artifact_root_path, candidate_identifier)),
        "seed_content_sha256": seed_content_sha256,
        "manifest_sha256": _sha256_file(candidate_manifest_path(artifact_root_path, candidate_identifier)),
        "streams": list(policy.required_streams),
        "frozen_calibration_identity": frozen,
        "certified_baseline_cross_checked": isinstance(certified_seed, dict),
        "per_stream_membership": per_stream,
    }


def _coerce_utc(value: object) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        stamp = pd.Timestamp(str(value))
    except (ValueError, TypeError):
        return None
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _validated_membership(
    state: dict,
    *,
    candidate_identifier: str,
    stream: str,
    generated_at: pd.Timestamp | None,
    declared_cutoff: pd.Timestamp | None,
) -> dict:
    """One stream's training membership, proved internally consistent and
    chronologically possible.

    Chronology here is an ARTIFACT-level check, not a second implementation of
    the eligibility rule: the certified machinery already decided which rows
    were eligible when it fit the candidate. What is verified is that the
    recorded evidence could not have come from the future -- no result may
    have become available after the candidate was generated, and when the
    candidate declares the cutoff it was generated for, nothing may have
    become available at or after that cutoff."""
    ids = state.get("training_game_ids")
    if not isinstance(ids, list) or not all(isinstance(g, str) for g in ids):
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} stream {stream}: training_game_ids must be a list of strings"
        )
    if len(set(ids)) != len(ids):
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} stream {stream}: training_game_ids contains duplicates"
        )
    expected_hash = training_membership_hash(sorted(ids))
    if state.get("training_membership_hash") != expected_hash:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} stream {stream}: training_membership_hash "
            f"{state.get('training_membership_hash')} does not hash this membership ({expected_hash})"
        )
    if int(state.get("training_game_count") or -1) != len(ids):
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} stream {stream}: training_game_count "
            f"{state.get('training_game_count')!r} does not match {len(ids)} training_game_ids"
        )

    latest = _coerce_utc(state.get("latest_included_result_available_at_utc"))
    if ids and latest is None:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} stream {stream}: membership of {len(ids)} games carries no "
            "parseable latest_included_result_available_at_utc, so its chronology cannot be verified"
        )
    if latest is not None and generated_at is not None and latest > generated_at:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} stream {stream}: includes a result available at {latest}, "
            f"after the candidate was generated at {generated_at}; a future result cannot leak backward"
        )
    if latest is not None and declared_cutoff is not None and latest >= declared_cutoff:
        raise PromotionRefused(
            f"candidate {candidate_identifier!r} stream {stream}: includes a result available at {latest}, "
            f"at or after its declared cutoff {declared_cutoff} (the chronology rule is STRICT)"
        )

    return {
        "training_game_ids": sorted(ids),
        "training_game_count": len(ids),
        "training_membership_hash": expected_hash,
        "latest_included_result_available_at_utc": None if latest is None else str(latest),
    }


def _assert_membership_non_regression(
    *,
    candidate_identifier: str,
    candidate_membership: dict[str, dict],
    active_seed: dict,
    active_candidate_id: str,
) -> dict:
    """A promoted candidate may only ever be superseded by one fit on MORE
    evidence. For every stream the currently active candidate's training
    membership must be a subset of the new candidate's, and the count must not
    fall. The 2026 population only grows (the population updater fails closed
    on a changed score), so a shrinking or diverging membership means evidence
    was lost or rewritten, and promoting that would quietly discard history."""
    detail: dict[str, dict] = {}
    for stream, membership in candidate_membership.items():
        active_state = active_seed.get(stream)
        if not isinstance(active_state, dict):
            raise PromotionRefused(
                f"candidate {candidate_identifier!r}: the active candidate {active_candidate_id!r} has no "
                f"state for stream {stream}, so membership non-regression cannot be proved"
            )
        active_ids = active_state.get("training_game_ids")
        if not isinstance(active_ids, list):
            raise PromotionRefused(
                f"candidate {candidate_identifier!r}: the active candidate {active_candidate_id!r} stream "
                f"{stream} records no training_game_ids, so membership non-regression cannot be proved"
            )
        new_ids = set(membership["training_game_ids"])
        dropped = sorted(set(map(str, active_ids)) - new_ids)
        if dropped:
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: training membership REGRESSED against "
                f"active candidate {active_candidate_id!r} -- {len(dropped)} game(s) present in the active "
                f"membership are absent here (e.g. {dropped[:5]})"
            )
        if len(new_ids) < len(set(map(str, active_ids))):
            raise PromotionRefused(
                f"candidate {candidate_identifier!r} stream {stream}: training membership shrank from "
                f"{len(set(map(str, active_ids)))} to {len(new_ids)} games"
            )
        detail[stream] = {
            "active_training_game_count": len(set(map(str, active_ids))),
            "candidate_training_game_count": len(new_ids),
            "games_added": len(new_ids) - len(set(map(str, active_ids))),
        }
    return detail


# ===========================================================================
# The active calibrator: one pointer, one resolver.
# ===========================================================================
def active_pointer_path(artifact_root_path: Path) -> Path:
    return Path(artifact_root_path) / ACTIVE_NAMESPACE / ACTIVE_POINTER_FILENAME


def promotion_events_dir(artifact_root_path: Path) -> Path:
    return Path(artifact_root_path) / ACTIVE_NAMESPACE / PROMOTION_EVENTS_DIRNAME


@dataclass(frozen=True)
class ActiveCalibrator:
    """The resolved, verified active calibrator and the provenance every
    forecast and run manifest records about it."""

    source: str
    status: str
    candidate_id: str | None
    seed_path: Path | None
    seed_sha256: str | None
    candidate_manifest_sha256: str | None
    policy_sha256: str | None
    seed: dict = field(repr=False, default_factory=dict)

    def provenance(self) -> dict:
        """The four required reproducibility fields, plus the status that says
        whether a seed was actually found."""
        return {
            "active_calibrator_source": self.source,
            "active_calibrator_candidate_id": self.candidate_id,
            "active_calibrator_seed_sha256": self.seed_sha256,
            "recalibration_policy_sha256": self.policy_sha256,
            "active_calibrator_status": self.status,
        }


def unavailable_active_calibrator(status: str = ACTIVE_ROOT_UNAVAILABLE) -> ActiveCalibrator:
    """The effective active calibrator when the artifact root itself cannot be
    resolved (a hermetic test, or a host with no ``NFL_MODEL_ARTIFACT_ROOT``).
    Nothing is promoted and no seed exists, so production prices
    uncalibrated/fail-closed exactly as it did before this module existed --
    this is NOT a fallback from a broken promotion, which always raises."""
    return ActiveCalibrator(
        source=SOURCE_BASELINE,
        status=status,
        candidate_id=None,
        seed_path=None,
        seed_sha256=None,
        candidate_manifest_sha256=None,
        policy_sha256=None,
        seed={},
    )


def resolve_active_calibrator(artifact_root_path: Path, *, repo_root: Path | None = None) -> ActiveCalibrator:
    """THE one resolver every production path uses to find the live calibrator.

    Three outcomes, and only three:

      * no pointer -> the immutable certified Fix-8 baseline, verified to be a
        JSON object when present. A baseline that is simply absent yields
        :data:`ACTIVE_BASELINE_SEED_MISSING` with an empty seed, which is the
        pre-existing uncalibrated/fail-closed state production already
        handles and preflight already blocks on;
      * a valid pointer -> the immutable candidate it references, with the
        seed's SHA256, the manifest's SHA256, the policy hash and the full
        four-stream readiness all re-verified on every single read;
      * anything else -> :class:`ActiveCalibratorError`. A pointer that exists
        but does not verify NEVER degrades to the baseline."""
    artifact_root_path = Path(artifact_root_path)
    pointer_file = active_pointer_path(artifact_root_path)
    baseline = certified_seed_path(artifact_root_path)

    if not pointer_file.is_file():
        if not baseline.is_file():
            return unavailable_active_calibrator(ACTIVE_BASELINE_SEED_MISSING)
        try:
            seed = json.loads(baseline.read_text())
        except ValueError as exc:
            raise ActiveCalibratorError(
                f"{baseline}: the certified baseline calibration seed is not valid JSON ({exc}). Failing "
                "closed rather than pricing a card with no calibrator while claiming the baseline."
            ) from exc
        if not isinstance(seed, dict):
            raise ActiveCalibratorError(
                f"{baseline}: the certified baseline calibration seed must be a JSON object, got "
                f"{type(seed).__name__}"
            )
        return ActiveCalibrator(
            source=SOURCE_BASELINE,
            status=ACTIVE_BASELINE_OK,
            candidate_id=None,
            seed_path=baseline,
            seed_sha256=_sha256_file(baseline),
            candidate_manifest_sha256=None,
            policy_sha256=None,
            seed=seed,
        )

    try:
        pointer = json.loads(pointer_file.read_text())
    except ValueError as exc:
        raise ActiveCalibratorError(f"{pointer_file}: active-calibrator pointer is not valid JSON: {exc}") from exc
    if not isinstance(pointer, dict):
        raise ActiveCalibratorError(f"{pointer_file}: active-calibrator pointer must be a JSON object")
    if pointer.get("schema_version") != ACTIVE_POINTER_SCHEMA_VERSION:
        raise ActiveCalibratorError(
            f"{pointer_file}: active-calibrator pointer schema_version {pointer.get('schema_version')!r} is "
            f"not this build's {ACTIVE_POINTER_SCHEMA_VERSION!r}"
        )

    pointed = pointer.get("candidate_id")
    if pointed == SOURCE_BASELINE:
        # An explicit pointer BACK to the baseline. Still verified, never assumed.
        if not baseline.is_file():
            raise ActiveCalibratorError(
                f"{pointer_file}: pointer names the {SOURCE_BASELINE} calibrator, but no baseline seed exists "
                f"at {baseline}"
            )
        recorded = pointer.get("seed_sha256")
        actual = _sha256_file(baseline)
        if recorded is not None and recorded != actual:
            raise ActiveCalibratorError(
                f"{pointer_file}: pointer records baseline seed_sha256 {recorded} but {baseline} hashes to {actual}"
            )
        seed = json.loads(baseline.read_text())
        return ActiveCalibrator(
            source=SOURCE_BASELINE,
            status=ACTIVE_BASELINE_OK,
            candidate_id=None,
            seed_path=baseline,
            seed_sha256=actual,
            candidate_manifest_sha256=None,
            policy_sha256=pointer.get("recalibration_policy_sha256"),
            seed=seed if isinstance(seed, dict) else {},
        )

    if not isinstance(pointed, str) or not pointed:
        raise ActiveCalibratorError(
            f"{pointer_file}: active-calibrator pointer names candidate_id {pointed!r}; expected a candidate id "
            f"or {SOURCE_BASELINE!r}"
        )

    expected_seed = candidate_seed_path(artifact_root_path, pointed)
    recorded_path = pointer.get("seed_path")
    if recorded_path is not None and Path(str(recorded_path)) != expected_seed:
        # A pointer may only ever reference the content-addressed candidate
        # directory under THIS artifact root. Anything else -- a copy, a path
        # outside the estate, a traversal -- fails closed.
        raise ActiveCalibratorError(
            f"{pointer_file}: pointer records seed_path {recorded_path!r}, but candidate {pointed!r} must be "
            f"read from {expected_seed}; a promoted seed is never copied elsewhere"
        )
    if not expected_seed.is_file():
        raise ActiveCalibratorError(
            f"{pointer_file}: active calibrator references candidate {pointed!r}, but its immutable seed is "
            f"missing at {expected_seed}. Failing closed: production never falls back from a broken promoted "
            "candidate to the baseline."
        )

    actual_seed_sha = _sha256_file(expected_seed)
    if pointer.get("seed_sha256") != actual_seed_sha:
        raise ActiveCalibratorError(
            f"{pointer_file}: active calibrator records seed_sha256 {pointer.get('seed_sha256')} but "
            f"{expected_seed} hashes to {actual_seed_sha}; the promoted candidate seed was altered"
        )

    manifest_file = candidate_manifest_path(artifact_root_path, pointed)
    if not manifest_file.is_file():
        raise ActiveCalibratorError(
            f"{pointer_file}: active calibrator references candidate {pointed!r}, whose manifest is missing at "
            f"{manifest_file}"
        )
    actual_manifest_sha = _sha256_file(manifest_file)
    if pointer.get("candidate_manifest_sha256") != actual_manifest_sha:
        raise ActiveCalibratorError(
            f"{pointer_file}: active calibrator records candidate_manifest_sha256 "
            f"{pointer.get('candidate_manifest_sha256')} but {manifest_file} hashes to {actual_manifest_sha}"
        )

    try:
        seed = json.loads(expected_seed.read_text())
    except ValueError as exc:
        raise ActiveCalibratorError(f"{expected_seed}: promoted candidate seed is not valid JSON: {exc}") from exc
    if not isinstance(seed, dict):
        raise ActiveCalibratorError(f"{expected_seed}: promoted candidate seed must be a JSON object")
    not_ready = [s for s in STREAM_NAMES if not _stream_calibration_ready(seed.get(s))]
    if not_ready:
        raise ActiveCalibratorError(
            f"{expected_seed}: promoted candidate {pointed!r} stream(s) {not_ready} no longer satisfy the frozen "
            "calibration readiness predicate; failing closed instead of silently pricing part of the card "
            "uncalibrated under a promoted calibrator"
        )

    policy_sha = pointer.get("recalibration_policy_sha256")
    lock_file = policy_lock_path(artifact_root_path)
    if lock_file.is_file():
        try:
            lock = json.loads(lock_file.read_text())
        except ValueError as exc:
            raise ActiveCalibratorError(f"{lock_file}: policy lock is not valid JSON: {exc}") from exc
        if lock.get("policy_sha256") != policy_sha:
            raise ActiveCalibratorError(
                f"{pointer_file}: the active calibrator was promoted under policy {policy_sha}, but the locked "
                f"policy is {lock.get('policy_sha256')}; failing closed rather than serving a card promoted "
                "under a policy this estate no longer pins"
            )

    return ActiveCalibrator(
        source=SOURCE_CANDIDATE,
        status=ACTIVE_CANDIDATE_OK,
        candidate_id=pointed,
        seed_path=expected_seed,
        seed_sha256=actual_seed_sha,
        candidate_manifest_sha256=actual_manifest_sha,
        policy_sha256=policy_sha,
        seed=seed,
    )


def active_calibrator_report(artifact_root_path: Path) -> dict:
    """Human/machine readable provenance of the currently active calibrator,
    including a clear fail-closed report rather than an exception when the
    caller only wants to describe the state."""
    baseline = certified_seed_path(artifact_root_path)
    report: dict[str, object] = {
        "pointer_path": str(active_pointer_path(artifact_root_path)),
        "pointer_present": active_pointer_path(artifact_root_path).is_file(),
        "certified_baseline_path": str(baseline),
        "certified_baseline_present": baseline.is_file(),
        "certified_baseline_sha256": _sha256_file(baseline) if baseline.is_file() else None,
        "certified_baseline_immutable": True,
    }
    try:
        active = resolve_active_calibrator(artifact_root_path)
    except ActiveCalibratorError as exc:
        report["resolution"] = "FAIL_CLOSED"
        report["detail"] = str(exc)
        return report
    report["resolution"] = "OK"
    report.update(active.provenance())
    report["active_calibrator_seed_path"] = None if active.seed_path is None else str(active.seed_path)
    report["active_calibrator_manifest_sha256"] = active.candidate_manifest_sha256
    return report


# ===========================================================================
# Promotion eligibility, then atomic promotion.
# ===========================================================================
def newest_valid_candidate_id(artifact_root_path: Path) -> str | None:
    """The newest recorded candidate by generation time. ``list_candidates``
    already sorts oldest-first on (generated_at_utc, candidate_id), so this is
    its last entry. Structural validity is proved separately by
    :func:`validate_candidate` -- a candidate is never promoted just for being
    newest."""
    candidates = list_candidates(artifact_root_path)
    if not candidates:
        return None
    newest = candidates[-1].get("candidate_id")
    return None if newest is None else str(newest)


def evaluate_promotion(
    artifact_root_path: Path,
    *,
    operational_root: Path | None = None,
    repo_root: Path,
    candidate_identifier: str | None = None,
    git_commit: str | None = None,
    now_utc: str | None = None,
    create_lock: bool = True,
) -> dict:
    """Decide whether the active calibrator should change. Writes nothing.

    Returns a decision whose ``status`` is one of :data:`DECISION_ELIGIBLE` or
    a member of :data:`NO_OP_DECISIONS`. Raises :class:`PolicyError`,
    :class:`PolicyLockViolation`, :class:`PromotionRefused` or
    :class:`ActiveCalibratorError` for anything that indicates corruption,
    tampering or drift.

    The check ORDER matters. Maturity is evaluated BEFORE the candidate is
    validated, because before the preregistered game floor a candidate
    legitimately has unfitted streams (too few labelled rows yet). That must
    report ``NOT_YET_MATURE`` and succeed, not fail the daily pass. Once the
    floor is reached, the same unfitted stream is genuine corruption and
    fails closed."""
    artifact_root_path = Path(artifact_root_path)
    operational_root = Path(operational_root) if operational_root is not None else artifact_root_path

    policy = load_promotion_policy(repo_root)
    lock = ensure_policy_lock(
        artifact_root_path, policy, git_commit=git_commit, now_utc=now_utc, create=create_lock
    )
    base: dict[str, object] = {
        "policy": policy.summary(),
        "policy_lock": lock,
        "scientific_refit_authorization": promotion_authorization(repo_root),
    }

    if not policy.enabled:
        return {
            **base,
            "status": DECISION_POLICY_DISABLED,
            "eligible": False,
            "reason": f"{policy.path} sets automatic_promotion_enabled=false; the active calibrator is unchanged",
        }

    maturity = prospective_maturity_state(operational_root)
    minimum = policy.minimum_prospective_games()
    base["prospective_maturity"] = maturity
    base["minimum_prospective_games"] = minimum
    if int(maturity["unique_completed_games"]) < minimum or maturity["maturity"] != policy.required_maturity_label():
        return {
            **base,
            "status": DECISION_NOT_YET_MATURE,
            "eligible": False,
            "reason": (
                f"prospective maturity is {maturity['maturity']} at "
                f"{maturity['unique_completed_games']} unique completed 2026 games; the preregistered floor "
                f"for promotion is {minimum}. Candidate generation continues, Ridge retraining continues, and "
                "the current calibrator stays active. This is a successful no-op."
            ),
        }

    target = candidate_identifier or newest_valid_candidate_id(artifact_root_path)
    if target is None:
        return {
            **base,
            "status": DECISION_NO_CANDIDATE,
            "eligible": False,
            "reason": f"no recalibration candidate recorded under {candidates_dir(artifact_root_path)}",
        }
    base["candidate_id"] = target

    active = resolve_active_calibrator(artifact_root_path)
    base["active_calibrator"] = {
        **active.provenance(),
        "seed_path": None if active.seed_path is None else str(active.seed_path),
    }
    if active.source == SOURCE_CANDIDATE and active.candidate_id == target:
        return {
            **base,
            "status": DECISION_ALREADY_ACTIVE,
            "eligible": False,
            "reason": (
                f"candidate {target!r} is already the active calibrator; re-promoting it writes no pointer and "
                "no promotion event"
            ),
        }

    validation = validate_candidate(artifact_root_path, target, policy=policy)
    non_regression = None
    if active.source == SOURCE_CANDIDATE and active.candidate_id is not None:
        non_regression = _assert_membership_non_regression(
            candidate_identifier=target,
            candidate_membership=validation["per_stream_membership"],
            active_seed=active.seed,
            active_candidate_id=active.candidate_id,
        )

    return {
        **base,
        "status": DECISION_ELIGIBLE,
        "eligible": True,
        "reason": (
            f"candidate {target!r} satisfies every requirement in {policy.path.name} at "
            f"{maturity['unique_completed_games']} unique completed 2026 games"
        ),
        "validation": {k: v for k, v in validation.items() if k not in ("seed", "manifest")},
        "membership_non_regression": non_regression,
        "_validation": validation,
        "_active": active,
    }


def promote_candidate(
    artifact_root_path: Path,
    *,
    candidate_identifier: str | None = None,
    repo_root: Path,
    operational_root: Path | None = None,
    git_commit: str | None = None,
    now_utc: str | None = None,
) -> dict:
    """Atomically make a candidate the active calibrator, or explain why not.

    Never touches the certified baseline seed: a promotion rewrites one small
    pointer file to REFERENCE the candidate seed that already exists under
    ``recalibration-candidates/``. The seed bytes are never copied, so there
    is no second calibration estate to keep in sync.

      * eligible -> write the pointer atomically (temp file + ``os.replace``
        in the same directory) and append exactly ONE content-addressed
        promotion event. Returns :data:`DECISION_PROMOTED`.
      * already active -> writes nothing at all, no pointer rewrite and no
        second event. Returns ``IDEMPOTENT_NOOP``.
      * any other no-op decision (not yet mature, no candidate, policy
        disabled) -> :class:`PromotionRefused`, so a caller that explicitly
        demanded a promotion is told it did not happen. The automatic daily
        path calls :func:`evaluate_promotion` first and only lands here when
        the decision came back eligible, which is why a pre-maturity daily run
        is a success and not an error."""
    decision = evaluate_promotion(
        artifact_root_path,
        operational_root=operational_root,
        repo_root=repo_root,
        candidate_identifier=candidate_identifier,
        git_commit=git_commit,
        now_utc=now_utc,
    )
    public = {k: v for k, v in decision.items() if not k.startswith("_")}

    if decision["status"] == DECISION_ALREADY_ACTIVE:
        return {**public, "status": "IDEMPOTENT_NOOP", "promoted": False, "decision": DECISION_ALREADY_ACTIVE}
    if decision["status"] != DECISION_ELIGIBLE:
        raise PromotionRefused(f"{decision['status']}: {decision['reason']}")

    validation: dict = decision["_validation"]
    active: ActiveCalibrator = decision["_active"]
    policy_summary: dict = decision["policy"]
    maturity: dict = decision["prospective_maturity"]
    target = str(decision["candidate_id"])
    promoted_at = now_utc or pd.Timestamp.now(tz="UTC").isoformat()

    pointer = {
        "schema_version": ACTIVE_POINTER_SCHEMA_VERSION,
        "candidate_id": target,
        "active_calibrator_source": SOURCE_CANDIDATE,
        "seed_path": validation["seed_path"],
        "seed_sha256": validation["seed_file_sha256"],
        "seed_content_sha256": validation["seed_content_sha256"],
        "candidate_manifest_sha256": validation["manifest_sha256"],
        "recalibration_policy_sha256": policy_summary["policy_sha256"],
        "recalibration_policy_schema_version": policy_summary["policy_schema_version"],
        "promoted_at_utc": promoted_at,
        "git_commit": git_commit if git_commit is not None else _resolve_git_commit(repo_root),
        "previous_active_calibrator_source": active.source,
        "previous_active_candidate_id": active.candidate_id,
        "previous_active_seed_sha256": active.seed_sha256,
        "prospective_unique_completed_games_at_promotion": int(maturity["unique_completed_games"]),
        "prospective_maturity_at_promotion": maturity["maturity"],
        "promotion_eligible_min_games": int(decision["minimum_prospective_games"]),
        "training_membership_hashes": {
            stream: membership["training_membership_hash"]
            for stream, membership in sorted(validation["per_stream_membership"].items())
        },
        "training_game_counts": {
            stream: membership["training_game_count"]
            for stream, membership in sorted(validation["per_stream_membership"].items())
        },
        "certified_baseline_path": str(certified_seed_path(artifact_root_path)),
        "certified_baseline_sha256": _sha256_file(certified_seed_path(artifact_root_path))
        if certified_seed_path(artifact_root_path).is_file()
        else None,
        "certified_baseline_overwritten": False,
        "certified_baseline_immutable": True,
    }

    pointer_file = active_pointer_path(artifact_root_path)
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer_file.with_name(pointer_file.name + ".tmp")
    tmp.write_text(json.dumps(pointer, indent=2, sort_keys=True, default=str))
    os.replace(tmp, pointer_file)

    event = {
        "schema_version": PROMOTION_EVENT_SCHEMA_VERSION,
        "promoted_at_utc": promoted_at,
        "pointer": pointer,
        "policy": policy_summary,
        "policy_lock": decision["policy_lock"],
        "prospective_maturity": maturity,
        "membership_non_regression": decision["membership_non_regression"],
        "frozen_calibration_identity": validation["frozen_calibration_identity"],
        "certified_baseline_cross_checked": validation["certified_baseline_cross_checked"],
        "scientific_refit_authorization": decision["scientific_refit_authorization"],
    }
    event_id = _sha256_hex(event)
    events_dir = promotion_events_dir(artifact_root_path)
    events_dir.mkdir(parents=True, exist_ok=True)
    event_file = events_dir / f"{event_id}.json"
    if not event_file.is_file():
        tmp_event = event_file.with_name(event_file.name + ".tmp")
        tmp_event.write_text(json.dumps({**event, "event_id": event_id}, indent=2, sort_keys=True, default=str))
        os.replace(tmp_event, event_file)

    return {
        **public,
        "status": DECISION_PROMOTED,
        "promoted": True,
        "candidate_id": target,
        "pointer_path": str(pointer_file),
        "pointer": pointer,
        "promotion_event_id": event_id,
        "promotion_event_path": str(event_file),
    }


def list_promotion_events(artifact_root_path: Path) -> list[dict]:
    """Every recorded promotion event, oldest first. One per ACTUAL change of
    active calibrator -- never one per run."""
    root = promotion_events_dir(artifact_root_path)
    if not root.is_dir():
        return []
    events: list[dict] = []
    for path in sorted(root.glob("*.json")):
        try:
            events.append(json.loads(path.read_text()))
        except ValueError:
            continue
    events.sort(key=lambda e: (str(e.get("promoted_at_utc")), str(e.get("event_id"))))
    return events


def candidate_state_report(
    *,
    artifact_root_path: Path,
    operational_root: Path,
    repo_root: Path,
) -> dict:
    """The machine-readable state a workflow reports every run: which
    candidates exist, the current prospective maturity, the operational policy
    and its lock, which calibrator is actually active, and what the promotion
    decision would be right now. Read-only: it never creates the policy lock
    and never promotes."""
    certified = certified_seed_path(artifact_root_path)
    candidates = list_candidates(artifact_root_path)

    policy_summary: dict | None = None
    policy_detail: str | None = None
    decision: dict | None = None
    try:
        policy = load_promotion_policy(repo_root)
        policy_summary = policy.summary()
        decision = {
            k: v
            for k, v in evaluate_promotion(
                artifact_root_path,
                operational_root=operational_root,
                repo_root=repo_root,
                create_lock=False,
            ).items()
            if not k.startswith("_")
        }
    except RecalibrationError as exc:
        policy_detail = f"{type(exc).__name__}: {exc}"
        decision = {"status": "FAIL_CLOSED", "eligible": False, "reason": policy_detail}

    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(candidates),
        "candidate_ids": [c.get("candidate_id") for c in candidates],
        "latest_candidate": candidates[-1] if candidates else None,
        "prospective_maturity": prospective_maturity_state(operational_root),
        "promotion_policy": policy_summary,
        "promotion_policy_error": policy_detail,
        "promotion_decision": decision,
        "promotion_status": None if decision is None else decision.get("status"),
        "scientific_refit_authorization": promotion_authorization(repo_root),
        "active_calibrator": active_calibrator_report(artifact_root_path),
        "promotion_event_count": len(list_promotion_events(artifact_root_path)),
        "certified_calibrator_path": str(certified),
        "certified_calibrator_present": certified.is_file(),
        "certified_calibrator_sha256": _sha256_file(certified) if certified.is_file() else None,
        "certified_calibrator_immutable": True,
        "horizons": list(he.HORIZONS),
    }
