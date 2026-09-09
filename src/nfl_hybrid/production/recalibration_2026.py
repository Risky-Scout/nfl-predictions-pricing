"""Automatic 2026 recalibration CANDIDATE generation + fail-closed promotion.

WHAT THIS IS
  A candidate generator and a promotion GATE. It produces recalibration
  candidates automatically (so a candidate always exists and its evidence is
  never assembled by hand at the moment someone wants to change production),
  and it refuses to touch the certified production calibrator unless an
  existing machine-readable PREREGISTERED promotion rule explicitly authorizes
  it.

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

WHERE CANDIDATES GO
  ``$NFL_MODEL_ARTIFACT_ROOT/production-2026/recalibration-candidates/
  <candidate_id>/`` -- outside git by construction, one immutable directory
  per candidate, each carrying its exact per-stream training MEMBERSHIP
  (``training_game_ids`` + a membership hash) and content hashes.

PROMOTION FAILS CLOSED
  :func:`promote_candidate` refuses unless
  :func:`promotion_authorization` finds an EXPLICIT machine-readable
  authorization in the frozen prospective preregistration
  (``outputs/prospective_2026_strength_preregistration.json``). Today that
  document contains no authorization block and does contain
  ``invariance.no_scientific_refit = true``, so the answer is
  ``PROMOTION_NOT_AUTHORIZED`` and the certified calibrator stays active.
  Candidate generation still runs, every run reports the candidate state, and
  nothing overwrites the certified seed. An operator cannot bypass this by
  passing a flag: the authorization comes only from the preregistered
  document, never from a caller argument or an environment variable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
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
    """A promotion was attempted with no preregistered authorization, or
    against a mismatched certified calibrator. Always fail closed -- the
    certified production calibrator is never replaced on a caller's word."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha256_hex(payload: object) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
    ready_mask = (raw["raw_status"] == "RAW_READY").to_numpy()
    binary_numeric = pd.to_numeric(raw["binary_target"], errors="coerce").to_numpy(float)
    labeled_mask = ready_mask & np.isfinite(binary_numeric)
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
    actual_push = (ready_mask & ~np.isfinite(binary_numeric)).astype(int)
    model_push = pd.to_numeric(raw["raw_push_probability"], errors="coerce").to_numpy(float)
    push_state = None
    if ready_mask.any():
        push_train = pd.DataFrame(
            {
                "model_push_probability": model_push[ready_mask],
                "actual_push": actual_push[ready_mask],
                "market_line": np.asarray(threshold, dtype=float)[ready_mask],
            }
        )
        global_scale, bucket_scales = _fit_push_scales(push_train, legacy_market, push_config)
        push_state = {"global_scale": global_scale, "bucket_scales": bucket_scales}

    training_game_ids = sorted(str(g) for g in raw.loc[ready_mask, "game_id"])
    max_result_available = (
        pd.to_datetime(raw.loc[ready_mask, "result_available_at_utc"], utc=True).max()
        if ready_mask.any() and "result_available_at_utc" in raw.columns
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
    immutable evaluation ledger. Reported alongside every candidate as
    context. Reaching ``PROMOTION_ELIGIBLE`` maturity is NOT, on its own, an
    authorization to promote -- see :func:`promotion_authorization`."""
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


# ---------------------------------------------------------------------------
# Promotion -- fails closed.
# ---------------------------------------------------------------------------
def certified_seed_path(artifact_root_path: Path) -> Path:
    return Path(artifact_root_path) / CERTIFIED_SEED_RELATIVE


def promote_candidate(
    artifact_root_path: Path,
    *,
    candidate_identifier: str,
    repo_root: Path,
) -> dict:
    """Attempt to promote a candidate over the certified production
    calibrator.

    Refuses -- :class:`PromotionRefused`, nothing written, certified seed
    untouched -- unless :func:`promotion_authorization` reports
    :data:`AUTHORIZED`. Because that answer comes only from the frozen
    preregistration, and today's preregistration asserts
    ``invariance.no_scientific_refit``, this function currently always
    refuses. That is the intended behaviour, not a limitation: the certified
    calibrator remains active and the candidate remains reportable evidence.
    """
    authorization = promotion_authorization(repo_root)
    directory = candidates_dir(artifact_root_path) / candidate_identifier
    if not authorization["authorized"]:
        raise PromotionRefused(
            f"{authorization['status']}: {authorization['reason']} "
            f"(candidate {candidate_identifier!r} remains CANDIDATE_ONLY; "
            f"the certified production calibrator at "
            f"{certified_seed_path(artifact_root_path)} was not read for writing)"
        )
    if not (directory / CANDIDATE_SEED_FILENAME).is_file():
        raise PromotionRefused(f"candidate {candidate_identifier!r} has no seed at {directory}")
    # Reachable only with a future preregistration that explicitly enables
    # promotion. Deliberately still refuses to overwrite in place: a promotion
    # under a future authorized contract must record its own replacement
    # evidence, which that contract has to define.
    raise PromotionRefused(
        f"{AUTHORIZED} was granted by {authorization['preregistration_path']}, but the authorization block "
        "does not define a replacement procedure (expected keys describing how the certified seed is "
        "superseded and how the replacement is evidenced); refusing to overwrite the certified production "
        "calibrator without one"
    )


def candidate_state_report(
    *,
    artifact_root_path: Path,
    operational_root: Path,
    repo_root: Path,
) -> dict:
    """The machine-readable state a workflow reports every run: which
    candidates exist, the current prospective maturity, whether promotion is
    authorized, and confirmation that the certified calibrator is still the
    active one."""
    authorization = promotion_authorization(repo_root)
    certified = certified_seed_path(artifact_root_path)
    candidates = list_candidates(artifact_root_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(candidates),
        "candidate_ids": [c.get("candidate_id") for c in candidates],
        "latest_candidate": candidates[-1] if candidates else None,
        "prospective_maturity": prospective_maturity_state(operational_root),
        "promotion_authorization": authorization,
        "promotion_status": authorization["status"],
        "certified_calibrator_path": str(certified),
        "certified_calibrator_present": certified.is_file(),
        "certified_calibrator_sha256": _sha256_file(certified) if certified.is_file() else None,
        "certified_calibrator_active": True,
        "horizons": list(he.HORIZONS),
    }
