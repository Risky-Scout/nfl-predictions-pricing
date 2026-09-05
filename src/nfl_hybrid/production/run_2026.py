"""2026 operational production pipeline -- core library backing
``scripts/run_2026_production_card.py`` and
``scripts/report_2026_prospective_performance.py``.

Reuses the certified pipeline UNCHANGED:
  - card-scoped TUE/FRI membership + DST-aware noon cutoffs:
    :mod:`nfl_hybrid.features.horizon_elo` (``_card_monday_date``,
    ``_card_noon_cutoff_utc``, ``build_horizon_membership_ledger``,
    ``eligible_game_ids``, ``HORIZONS``).
  - six frozen Elo features + RIDGE_ALPHA_100 chronological OOF + Fix-8
    same-horizon uncertainty: :mod:`nfl_hybrid.evaluation.official_horizon_oof`
    (``build_official_horizon_matrix``, ``build_official_horizon_oof``).
  - raw timestamped bookmaker-history market reconstruction:
    :mod:`nfl_hybrid.evaluation.raw_market_reconstruction`.
  - ATS/TOTAL raw pricing:
    :func:`nfl_hybrid.evaluation.chronological_calibration.build_raw_probabilities`.
  - the frozen production calibration seed
    (``$NFL_MODEL_ARTIFACT_ROOT/fix8-official-oof-calibration-2026/production_calibration_seed.json``),
    applied via :mod:`nfl_hybrid.calibration.three_way`'s own apply-path
    primitives (``_predict_push``/``_recombine``, reused verbatim) plus a
    verified sigmoid-on-logit reconstruction of ``_predict_conditional``'s
    decision function for the stored (coefficient, intercept) pair -- see
    :func:`apply_frozen_conditional_calibrator`, whose arithmetic is
    numerically identical to reconstructing a fitted
    ``sklearn.linear_model.LogisticRegression`` and calling
    ``_predict_conditional`` on it directly (verified in
    ``tests/test_production_card_2026.py::test_frozen_calibrator_matches_three_way_apply_path``).
  - the certified hash gate: :mod:`nfl_hybrid.certification.final_review_2026`
    (``verify_certified_hashes``, the certified constants).

No duplicate reimplementation of any scientific rule. This module owns only
operational concerns: DST-aware due-run scheduling, fail-closed source
gating, the immutable forecast ledger, the append-only run manifest, the
prospective shadow evaluation ledger, result attachment, and the
prospective performance report.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from nfl_hybrid.calibration.three_way import CalibrationConfig, _predict_push, _recombine
from nfl_hybrid.certification import final_review_2026 as cert
from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.data.external_data import REPO_ROOT, artifact_root, resolve
from nfl_hybrid.evaluation import official_horizon_oof as ohf
from nfl_hybrid.evaluation import raw_market_reconstruction as rmr
from nfl_hybrid.evaluation.chronological_calibration import (
    _LEGACY_PUSH_MARKET_NAME,
    MARKET_ATS,
    MARKET_TOTAL,
    build_raw_probabilities,
)
from nfl_hybrid.evaluation.week1_reliability import NOT_ESTIMABLE, binary_log_loss, brier, equal_mass_ece
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.labels import edge_to_nullable_binary

SCHEMA_VERSION = "production-2026-v1"

NY_ZONE = he.NY_ZONE
DUE_WINDOW_MINUTES = 20

# Re-exported for callers -- the certified constants live in one place
# (nfl_hybrid.certification.final_review_2026), never re-declared here.
CERTIFIED_TAG = cert.CERTIFIED_BASELINE_TAG
CERTIFIED_SHA = cert.CERTIFIED_BASELINE_SHA

STREAM_NAMES = ("ATS_TUE", "ATS_FRI", "TOTAL_TUE", "TOTAL_FRI")
_FORECAST_HORIZON_LABEL = {"TUE": "official_card_scoped_tue", "FRI": "official_card_scoped_fri"}
MIN_CALIBRATION_SAMPLE_COUNT = 100
MIN_PROSPECTIVE_SAMPLE = 20

# ===========================================================================
# Section 16 -- fail-closed statuses. Never silently substituted for a real
# result; a status here always means "no forecast was produced this run" (or,
# for a per-market/per-stream entry, "no probability was produced for this
# market this run").
# ===========================================================================
FAIL_CLOSED_STATUSES: tuple[str, ...] = (
    "NOT_DUE",
    "SCHEDULE_UNAVAILABLE",
    "GAME_RESULT_SOURCE_UNAVAILABLE",
    "ELO_SOURCE_UNAVAILABLE",
    "MARKET_SOURCE_UNAVAILABLE",
    "MARKET_NOT_READY",
    "MODEL_NOT_READY",
    "UNCERTAINTY_NOT_READY",
    "CALIBRATION_NOT_READY",
    "SCHEMA_DRIFT",
    "IDENTIFIER_FAILURE",
    "HASH_MISMATCH",
    "FORECAST_IMMUTABILITY_VIOLATION",
)


class ProductionHardStop(RuntimeError):
    """Carries one of FAIL_CLOSED_STATUSES. Never caught-and-substituted."""

    def __init__(self, status: str, detail: str = ""):
        if status not in FAIL_CLOSED_STATUSES:
            raise ValueError(f"Unknown fail-closed status: {status!r}")
        self.status = status
        self.detail = detail
        super().__init__(f"{status}: {detail}" if detail else status)


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(payload: dict) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _as_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


# ===========================================================================
# Section 13 -- DST-aware due-run semantics. Pure calendar/time-window logic
# built ONLY on the already-certified NY_ZONE/_card_monday_date/
# _card_noon_cutoff_utc primitives from nfl_hybrid.features.horizon_elo --
# no new timezone/cutoff math is introduced here.
# ===========================================================================
_HORIZON_WEEKDAY = {"TUE": 1, "FRI": 4}  # Monday=0 (datetime/ZoneInfo convention)


def is_within_due_window(as_of_utc: pd.Timestamp, horizon: str) -> dict:
    if horizon not in _HORIZON_WEEKDAY:
        raise ValueError(f"Unknown horizon: {horizon!r}")
    as_of_utc = _as_utc(as_of_utc)
    local = as_of_utc.tz_convert(NY_ZONE)
    window_start = local.replace(hour=12, minute=0, second=0, microsecond=0)
    window_end = window_start + pd.Timedelta(minutes=DUE_WINDOW_MINUTES)
    correct_weekday = local.weekday() == _HORIZON_WEEKDAY[horizon]
    in_window = bool(window_start <= local < window_end)
    return {
        "due": bool(correct_weekday and in_window),
        "as_of_utc": as_of_utc.isoformat(),
        "as_of_local_ny": local.isoformat(),
        "horizon": horizon,
        "window_start_local_ny": window_start.isoformat(),
        "window_end_local_ny": window_end.isoformat(),
    }


def current_or_recent_cutoff(as_of_utc: pd.Timestamp, horizon: str) -> pd.Timestamp:
    """The ``horizon`` cutoff for the card whose Monday contains ``as_of_utc``
    (America/New_York calendar week) -- i.e. "the current or upcoming card",
    computed FROM the current/as-of time, not from any game's own kickoff.
    Reuses :func:`he._card_monday_date`/:func:`he._card_noon_cutoff_utc`
    verbatim (Fix 7.1 V2's own card-scoped cutoff primitives)."""
    as_of_utc = _as_utc(as_of_utc)
    monday = he._card_monday_date(as_of_utc)
    offset_days = he._HORIZON_DAY_OFFSET[horizon]
    return he._card_noon_cutoff_utc(monday, offset_days)


# ===========================================================================
# Section 17 -- live source preflight. Never prints secret VALUES -- only
# SET/UNSET, derived from .env.example (the repo's own declared var list)
# rather than a hardcoded/duplicated list.
# ===========================================================================
def env_var_names_from_env_example(repo_root: Path = REPO_ROOT) -> list[str]:
    path = repo_root / ".env.example"
    if not path.is_file():
        return []
    names: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.append(line.split("=", 1)[0].strip())
    return names


def env_var_presence(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    return {name: ("SET" if os.environ.get(name) else "UNSET") for name in env_var_names_from_env_example(repo_root)}


# Fail-closed blocking reasons for the two REQUIRED live 2026 production
# inputs. Kept distinct from the infrastructure blockers so a caller can
# always tell "the host is broken" apart from "the season's live feeds are
# not wired/available yet" -- both block live production, neither is ever
# silently absorbed into a benign "waiting" status.
SCHEDULE_2026_BLOCKER = "schedule_2026_unavailable"
LIVE_MARKET_2026_BLOCKER = "live_2026_market_source_unregistered"


# ===========================================================================
# Explicit 2026 live market source. Production NEVER auto-selects a capture:
# the exact official manifest is supplied by the operator
# (``--market-capture-manifest``). Nothing here scans a directory or picks a
# "latest" capture by mtime, directory name, capture_started_at, UUID or
# COMPLETE-ness. With no manifest supplied, every historical code path is
# untouched -- 2020-2025 reconstruction keeps working on a machine that has
# no 2026 capture at all.
# ===========================================================================
LIVE_MARKET_NOT_SUPPLIED = "NOT_SUPPLIED"
LIVE_MARKET_OK = "OK"
LIVE_MARKET_INVALID = "INVALID"


def evaluate_live_market_source(
    market_capture_manifest: Path | str | None,
    *,
    expected_season: int | None = None,
    expected_week: int | None = None,
    expected_horizon: str | None = None,
    expected_target_cutoff_utc: pd.Timestamp | str | None = None,
    artifact_root_path: Path | None = None,
) -> dict:
    """Evidence for whether a live 2026 market source is genuinely registered.

    ``registered`` is ``True`` ONLY when all four conditions hold: an explicit
    capture manifest was supplied; its validation passed
    (:func:`nfl_hybrid.data.bdl_market_bridge.validate_capture_manifest` --
    COMPLETE, production horizon, no downgrade, matching season/week/horizon/
    cutoff, verified manifest and per-page response hashes); the canonical
    market rows were successfully materialized AND re-read from the
    identity-keyed artifact; and those rows are actually consumable by the
    certified Fix-8 reconstruction (both ``spreads`` and ``totals`` yield at
    least one COMPLETE COHERENT two-sided book observation via the existing
    :func:`rmr.build_coherent_book_observations`).

    Never optimistic: any failure -- including a supplied-but-invalid
    manifest -- leaves ``registered`` ``False`` and production
    ``BLOCKED_ON_LIVE_INPUTS``. Coherence here is only the existing
    certified pairing rule being exercised as proof of consumability; the
    minimum-book, freshness and consensus gates remain exactly where they
    already live, downstream."""
    if market_capture_manifest is None:
        return {
            "registered": False,
            "status": LIVE_MARKET_NOT_SUPPLIED,
            "detail": "no --market-capture-manifest supplied; production has no live 2026 market source",
            "provenance": None,
            "source": None,
        }
    try:
        source = bridge.load_live_market_source(
            market_capture_manifest,
            expected_season=expected_season,
            expected_week=expected_week,
            expected_horizon=expected_horizon,
            expected_target_cutoff_utc=expected_target_cutoff_utc,
            artifact_root_path=artifact_root_path,
        )
        coherence = {
            raw_market: int(len(rmr.build_coherent_book_observations(source.quotes, raw_market)))
            for raw_market in (rmr.MARKET_SPREADS, rmr.MARKET_TOTALS)
        }
        empty = sorted(m for m, n in coherence.items() if n == 0)
        if empty:
            return {
                "registered": False,
                "status": LIVE_MARKET_INVALID,
                "detail": (
                    f"capture produced no coherent two-sided book observations for {empty}; "
                    "it is not consumable by the certified reconstruction"
                ),
                "provenance": {**source.provenance(), "coherent_observations": coherence},
                "source": None,
            }
    except Exception as exc:
        return {
            "registered": False,
            "status": LIVE_MARKET_INVALID,
            "detail": f"{type(exc).__name__}: {exc}",
            "provenance": {"manifest_path": str(market_capture_manifest)},
            "source": None,
        }
    return {
        "registered": True,
        "status": LIVE_MARKET_OK,
        "detail": "",
        "provenance": {**source.provenance(), "coherent_observations": coherence},
        "source": source,
    }


def summarize_preflight_readiness(
    *,
    infra_blocking: list[str],
    schedule_2026_available: bool,
    live_2026_market_source_registered: bool,
) -> dict:
    """Pure readiness verdict. INFRASTRUCTURE readiness (certified hashes,
    calibration seed, writable ledgers, git commit) is reported SEPARATELY
    from LIVE-INPUT readiness (a real 2026 schedule, a registered live 2026
    market source).

    ``production_run_ready`` is ``True`` only when infrastructure is ready
    AND every required live production input is actually available. A host
    that is missing either live input is ``BLOCKED_ON_LIVE_INPUTS`` -- it
    NEVER reports ``READY`` or any "waiting for the first due cutoff"
    status, and ``blocking_problems`` always names the missing inputs.
    This does not weaken any existing fail-closed behaviour: an
    infrastructure blocker still forces ``NOT_READY`` regardless of the
    live inputs.
    """
    infra_ready = not infra_blocking

    live_input_blocking: list[str] = []
    if not schedule_2026_available:
        live_input_blocking.append(SCHEDULE_2026_BLOCKER)
    if not live_2026_market_source_registered:
        live_input_blocking.append(LIVE_MARKET_2026_BLOCKER)

    production_run_ready = infra_ready and not live_input_blocking

    if not infra_ready:
        overall_status = "NOT_READY"
    elif production_run_ready:
        overall_status = "READY"
    else:
        overall_status = "BLOCKED_ON_LIVE_INPUTS"

    return {
        "overall_status": overall_status,
        "infra_ready": infra_ready,
        "schedule_2026_available": bool(schedule_2026_available),
        "live_2026_market_source_registered": bool(live_2026_market_source_registered),
        "production_run_ready": production_run_ready,
        "blocking_problems": list(infra_blocking) + live_input_blocking,
        "infra_blocking_problems": list(infra_blocking),
        "live_input_blocking_problems": live_input_blocking,
    }


def run_preflight(
    *,
    repo_root: Path = REPO_ROOT,
    artifact_root_path: Path | None = None,
    market_capture_manifest: Path | str | None = None,
    expected_season: int | None = None,
    expected_week: int | None = None,
    expected_horizon: str | None = None,
    expected_target_cutoff_utc: pd.Timestamp | str | None = None,
) -> dict:
    """Verifies live infrastructure without ever printing a secret value.
    ``artifact_root_path`` is injectable so a test can point EVERY
    generated-artifact lookup this call performs -- the writable-output-
    directory check AND the Fix-8 production calibration seed lookup --
    at one isolated root (``aroot``), consistently, for the whole
    invocation. Only the certified hash sources, the games/schedule
    source, and the raw market estate remain the real certified/live
    inputs regardless of ``artifact_root_path`` (they are read-only
    scientific evidence and live data feeds, not generated-artifact
    output, so there is nothing operational to isolate for them).

    ``market_capture_manifest`` is the EXACT official BallDontLie capture
    manifest, supplied explicitly; it is never auto-selected. Omitted (the
    default), there is no live 2026 market source and production stays
    ``BLOCKED_ON_LIVE_INPUTS``. The ``expected_*`` arguments are the
    requesting run's own identity, cross-checked against that capture."""
    aroot = artifact_root_path if artifact_root_path is not None else artifact_root()
    checks: dict[str, object] = {}
    blocking: list[str] = []

    checks["environment_variables"] = env_var_presence(repo_root)

    git_commit = _git_commit(repo_root)
    checks["git_commit"] = {"status": "OK" if git_commit else "UNAVAILABLE", "value": git_commit}
    if not git_commit:
        blocking.append("git_commit_unavailable")

    try:
        fix71_summary = json.loads((repo_root / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text())
        fix8_prereg = json.loads((repo_root / "outputs" / "fix8_official_oof_calibration_preregistration.json").read_text())
        hash_checks = cert.verify_certified_hashes(fix71_summary, fix8_prereg)
        checks["certified_hashes"] = {"status": "MATCH", "detail": hash_checks}
    except FileNotFoundError as exc:
        checks["certified_hashes"] = {"status": "SOURCE_MISSING", "detail": str(exc)}
        blocking.append("hash_source_missing")
    except cert.CertificationGateFailure as exc:
        checks["certified_hashes"] = {"status": "MISMATCH", "detail": str(exc)}
        blocking.append("hash_mismatch")

    seed_path = aroot / "fix8-official-oof-calibration-2026" / "production_calibration_seed.json"
    checks["fix8_calibration_seed"] = {"status": "OK" if seed_path.is_file() else "MISSING", "path": str(seed_path)}
    if not seed_path.is_file():
        blocking.append("calibration_seed_missing")

    schedule_2026_available = False
    try:
        games = pd.read_parquet(resolve("backfill.games"))
        max_season = int(pd.to_numeric(games["season"], errors="coerce").max())
        schedule_2026_available = max_season >= 2026
        checks["schedule_source"] = {"status": "OK", "max_season_available": max_season, "row_count": int(len(games))}
    except Exception as exc:
        checks["schedule_source"] = {"status": "UNAVAILABLE", "detail": str(exc)}
        blocking.append("schedule_source_unavailable")
    checks["schedule_2026_status"] = "AVAILABLE" if schedule_2026_available else "SCHEDULE_UNAVAILABLE"

    market_estate_status: dict[str, str] = {}
    for key in rmr.RAW_ODDS_HISTORY_KEYS:
        try:
            resolve(key)
            market_estate_status[key] = "AVAILABLE"
        except Exception:
            market_estate_status[key] = "UNAVAILABLE"
    checks["raw_market_estate"] = market_estate_status
    # EVIDENCE-BASED, never optimistic: a live 2026 market source counts as
    # registered only when an explicit capture manifest was supplied, passed
    # validation, materialized canonical rows that were re-read from the
    # artifact, and is consumable by the certified reconstruction. Anything
    # else -- including no manifest at all, which is the default -- keeps
    # production BLOCKED_ON_LIVE_INPUTS rather than silently degrading to a
    # fabricated/synthetic line.
    live_market = evaluate_live_market_source(
        market_capture_manifest,
        expected_season=expected_season,
        expected_week=expected_week,
        expected_horizon=expected_horizon,
        expected_target_cutoff_utc=expected_target_cutoff_utc,
        # The materialized live-market artifact IS generated-artifact output,
        # so it follows this call's injected root like every other one.
        artifact_root_path=aroot,
    )
    live_2026_market_source_registered = bool(live_market["registered"])
    checks["live_2026_market_source_registered"] = live_2026_market_source_registered
    checks["live_2026_market_source"] = {
        "status": live_market["status"],
        "detail": live_market["detail"],
        "provenance": live_market["provenance"],
    }

    for name, subdir in (
        ("forecast_ledger_dir", "forecast-ledger"), ("run_manifest_dir", "run-manifests"),
        ("evaluation_ledger_dir", "evaluation-ledger"),
    ):
        d = aroot / "production-2026" / subdir
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".preflight_write_probe"
            probe.write_text("ok")
            probe.unlink()
            writable = True
        except OSError:
            writable = False
        checks[name] = {"status": "OK" if writable else "NOT_WRITABLE", "path": str(d)}
        if not writable:
            blocking.append(f"{name}_not_writable")

    readiness = summarize_preflight_readiness(
        infra_blocking=blocking,
        schedule_2026_available=schedule_2026_available,
        live_2026_market_source_registered=live_2026_market_source_registered,
    )
    return {**readiness, "checks": checks}


# ===========================================================================
# Section 14 -- immutable forecast ledger. First successful write wins;
# an identical-payload rewrite is IDEMPOTENT_NOOP; a different-payload
# rewrite of the same identity is a hard FORECAST_IMMUTABILITY_VIOLATION.
# ===========================================================================
def _identity_path(ledger_root: Path, game_id: str, horizon: str, target_cutoff_utc: str) -> Path:
    safe_cutoff = str(target_cutoff_utc).replace(":", "").replace("+", "_")
    return ledger_root / horizon / f"{game_id}__{safe_cutoff}.json"


@dataclass
class ForecastWriteResult:
    status: str  # WRITTEN | IDEMPOTENT_NOOP
    path: Path
    prediction_hash: str


def write_forecast(ledger_root: Path, record: dict) -> ForecastWriteResult:
    """``record`` must contain ``game_id``, ``horizon``, ``target_cutoff_utc``
    (the forecast identity) and ``prediction`` (the deterministic payload the
    identity's immutability is checked against)."""
    game_id, horizon, target_cutoff_utc = str(record["game_id"]), record["horizon"], str(record["target_cutoff_utc"])
    prediction_hash = _sha256_hex(record["prediction"])
    path = _identity_path(ledger_root, game_id, horizon, target_cutoff_utc)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = json.loads(path.read_text())
        if existing["prediction_hash"] == prediction_hash:
            return ForecastWriteResult("IDEMPOTENT_NOOP", path, prediction_hash)
        raise ProductionHardStop(
            "FORECAST_IMMUTABILITY_VIOLATION",
            f"{game_id}/{horizon}/{target_cutoff_utc}: existing={existing['prediction_hash']} new={prediction_hash}",
        )

    full_record = {**record, "prediction_hash": prediction_hash}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(full_record, indent=2, sort_keys=True, default=str))
    tmp.rename(path)
    return ForecastWriteResult("WRITTEN", path, prediction_hash)


def read_forecast(ledger_root: Path, game_id: str, horizon: str, target_cutoff_utc: str) -> dict | None:
    path = _identity_path(ledger_root, game_id, horizon, target_cutoff_utc)
    return json.loads(path.read_text()) if path.exists() else None


# ===========================================================================
# Section 15 -- append-only run manifest. One file per run_id; existing
# manifest files are never edited or deleted.
# ===========================================================================
def make_run_id(as_of_utc: pd.Timestamp, horizon: str) -> str:
    ts = _as_utc(as_of_utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}__{horizon}__{uuid4().hex[:8]}"


def write_run_manifest(manifest_root: Path, manifest: dict) -> Path:
    manifest_root.mkdir(parents=True, exist_ok=True)
    path = manifest_root / f"{manifest['run_id']}.json"
    if path.exists():
        raise ProductionHardStop("IDENTIFIER_FAILURE", f"run_id already has a manifest: {manifest['run_id']}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path


def build_run_manifest(
    *, run_id: str, horizon: str, target_cutoff_utc: str | None, git_commit: str | None, source_readiness: dict,
    game_count: int, forecast_count: int, abstention_count: int, market_ready_counts: dict, calibration_ready_counts: dict,
    input_hashes: dict, output_hashes: dict, status: str, started_at_utc: pd.Timestamp, run_created_at_utc: str,
    detail: str = "",
) -> dict:
    return {
        "run_id": run_id, "started_at_utc": _as_utc(started_at_utc).isoformat(), "completed_at_utc": utc_now().isoformat(),
        # The ONE authoritative run-level creation timestamp (Section 6/7 of
        # the public-identity hardening): the exact same value already
        # computed once in run_horizon_batch from started_at_utc above, never
        # recomputed here -- present under the field name the future public
        # Wizard exporter looks up, sharing this run's started_at_utc instant
        # rather than adding a second, competing run timestamp.
        "run_created_at_utc": run_created_at_utc,
        "horizon": horizon, "target_cutoff_utc": target_cutoff_utc,
        "git_commit": git_commit, "certified_model_tag": CERTIFIED_TAG, "certified_model_sha": CERTIFIED_SHA,
        "source_readiness": source_readiness, "game_count": game_count, "forecast_count": forecast_count,
        "abstention_count": abstention_count, "market_ready_counts": market_ready_counts,
        "calibration_ready_counts": calibration_ready_counts, "input_hashes": input_hashes,
        "output_hashes": output_hashes, "status": status, "detail": detail,
    }


# ===========================================================================
# Section 18 -- prospective shadow evidence ledger. NO outcome columns at
# forecast time; enforced structurally below.
# ===========================================================================
_FORBIDDEN_EVALUATION_FIELDS = frozenset({
    "actual_margin", "actual_total", "home_score", "away_score", "binary_outcome", "result", "winner",
    "ats_binary_target", "total_binary_target",
})


def write_evaluation_record(evaluation_root: Path, record: dict) -> Path:
    leaked = _FORBIDDEN_EVALUATION_FIELDS & set(record.keys())
    if leaked:
        raise ProductionHardStop("SCHEMA_DRIFT", f"evaluation ledger record contains forbidden outcome field(s): {leaked}")
    game_id, horizon, target_cutoff_utc = str(record["game_id"]), record["horizon"], str(record["target_cutoff_utc"])
    deterministic = {k: v for k, v in record.items() if k != "provenance"}
    content_hash = _sha256_hex(deterministic)

    path = _identity_path(evaluation_root, game_id, horizon, target_cutoff_utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        existing_deterministic = {k: v for k, v in existing.items() if k not in ("provenance", "content_hash")}
        if _sha256_hex(existing_deterministic) == content_hash:
            return path
        raise ProductionHardStop(
            "FORECAST_IMMUTABILITY_VIOLATION", f"evaluation record changed for {game_id}/{horizon}/{target_cutoff_utc}",
        )
    full = {**record, "content_hash": content_hash}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(full, indent=2, sort_keys=True, default=str))
    tmp.rename(path)
    return path


# ===========================================================================
# Section 19 -- result attachment. A SEPARATE sibling file; the original
# forecast/evaluation record is never opened for writing.
# ===========================================================================
def attach_result(
    evaluation_root: Path, *, game_id: str, horizon: str, target_cutoff_utc: str,
    result: dict, result_available_at_utc: pd.Timestamp, attachment_run_time: pd.Timestamp, result_source_hash: str,
) -> Path:
    result_available_at_utc = _as_utc(result_available_at_utc)
    attachment_run_time = _as_utc(attachment_run_time)
    if not (result_available_at_utc < attachment_run_time):
        raise ProductionHardStop(
            "IDENTIFIER_FAILURE",
            f"result_available_at_utc={result_available_at_utc} not strictly before attachment_run_time={attachment_run_time}",
        )
    forecast_path = _identity_path(evaluation_root, game_id, horizon, target_cutoff_utc)
    if not forecast_path.exists():
        raise ProductionHardStop("IDENTIFIER_FAILURE", f"no evaluation-ledger forecast for {game_id}/{horizon}/{target_cutoff_utc}")

    result_path = forecast_path.with_suffix(".result.json")
    payload = {
        "game_id": game_id, "horizon": horizon, "target_cutoff_utc": target_cutoff_utc,
        "result": result, "result_available_at_utc": result_available_at_utc.isoformat(),
        "result_attached_at_utc": utc_now().isoformat(), "result_source_hash": result_source_hash,
    }
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing["result"] == result:
            return result_path
        raise ProductionHardStop("FORECAST_IMMUTABILITY_VIOLATION", f"result already attached differently for {game_id}")
    tmp = result_path.with_suffix(".result.json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.rename(result_path)
    return result_path


# ===========================================================================
# Section 12 -- forecast generation, reusing the certified pipeline
# unchanged: ohf.build_official_horizon_oof (six Elo features,
# RIDGE_ALPHA_100, Fix-8 same-horizon uncertainty), rmr (raw timestamped
# bookmaker-history market reconstruction), build_raw_probabilities (ATS/
# TOTAL raw pricing), and the FROZEN production_calibration_seed.json
# conditional + push calibrators applied via three_way.py's own apply-path
# primitives.
# ===========================================================================
_EPSILON = 1e-6

# Prospective scoring -- and therefore the production card -- is REG+POST
# only; PRESEASON never enters (contract:
# :mod:`nfl_hybrid.evaluation.prospective_strength_2026`,
# ``PROSPECTIVE_SEASON_TYPES``). The historical ``backfill.games``
# population already contains no PRESEASON rows, so this is a structural
# guard for the day a live 2026 schedule source is wired in (BDL / nflverse
# both expose preseason games), not a change to the current population or
# to any prediction on it.
REG_POST_SEASON_TYPES = ("REG", "POST")


def filter_reg_post(games: pd.DataFrame) -> pd.DataFrame:
    """Restrict a games population to ``season_type in {REG, POST}``.
    Idempotent; a no-op on the historical backfill (which has no PRESEASON
    rows). Fails closed if ``season_type`` is absent -- a schedule source
    without it cannot prove it excludes preseason."""
    if "season_type" not in games.columns:
        raise ProductionHardStop(
            "SCHEDULE_UNAVAILABLE", "games population is missing the season_type column (cannot exclude PRESEASON)"
        )
    keep = games["season_type"].astype(str).str.upper().isin(REG_POST_SEASON_TYPES)
    return games.loc[keep].reset_index(drop=True)


def load_games_population() -> pd.DataFrame:
    """The one canonical games/schedule source (per certification GROUNDING):
    :func:`nfl_hybrid.data.external_data.resolve` ``"backfill.games"``,
    restricted to REG+POST via :func:`filter_reg_post` (PRESEASON is never
    part of the 2026 production card). Currently covers seasons 2020-2025
    only -- see :func:`run_preflight`'s ``schedule_2026_status`` check, which
    reports this honestly instead of fabricating 2026 rows."""
    return filter_reg_post(pd.read_parquet(resolve("backfill.games")))


def apply_frozen_conditional_calibrator(raw_probability: np.ndarray, seed_state: dict) -> np.ndarray:
    """Applies the FROZEN ``conditional_calibrator_state`` (coefficient,
    intercept of an already-fitted one-feature logistic-on-logit calibrator,
    ``three_way.CALIBRATOR_FAMILY``) to new raw probabilities. Numerically
    identical to reconstructing a fitted
    ``sklearn.linear_model.LogisticRegression`` with these exact
    ``coef_``/``intercept_`` and calling
    ``nfl_hybrid.calibration.three_way._predict_conditional`` on it --
    verified row-for-row in
    ``tests/test_production_card_2026.py::test_frozen_calibrator_matches_three_way_apply_path``.
    Never refits."""
    p = np.clip(np.asarray(raw_probability, dtype=float), _EPSILON, 1.0 - _EPSILON)
    state = seed_state["conditional_calibrator_state"]
    if not state.get("fitted", False):
        return p
    coef = float(state["coefficient"][0][0])
    intercept = float(state["intercept"][0])
    calibrated_logit = intercept + coef * logit(p)
    return np.clip(expit(calibrated_logit), _EPSILON, 1.0 - _EPSILON)


def _frozen_stream_calibration_ready(seed_state: object) -> bool:
    """A production stream is CALIBRATED only when EVERY frozen calibration
    component the certified ``three_way`` apply path dereferences is present
    and well-formed:

      - ``seed_state`` is a mapping;
      - ``conditional_calibrator_state`` exists, is a mapping, ``fitted`` true;
      - its ``coefficient`` yields a finite ``[0][0]`` float and ``intercept``
        a finite ``[0]`` float -- exactly what
        :func:`apply_frozen_conditional_calibrator` reads;
      - ``push_scale_state`` exists, is a mapping, with a finite numeric
        ``global_scale`` and a mapping ``bucket_scales`` -- exactly what
        :func:`nfl_hybrid.calibration.three_way._predict_push` reads.

    Any missing/malformed component -> ``False``: the stream is fail-closed to
    ``CALIBRATION_NOT_READY`` with NO numeric ``calibrated_`` output, and the
    raw push probability is NEVER substituted for an absent frozen push
    calibrator while still labelling the stream CALIBRATED. Introduces no new
    calibration semantics -- it validates only the frozen state the existing
    certified apply path already requires."""
    if not isinstance(seed_state, dict):
        return False
    cond = seed_state.get("conditional_calibrator_state")
    if not isinstance(cond, dict) or not cond.get("fitted", False):
        return False
    try:
        coef = float(cond["coefficient"][0][0])
        intercept = float(cond["intercept"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    if not (np.isfinite(coef) and np.isfinite(intercept)):
        return False
    push = seed_state.get("push_scale_state")
    if not isinstance(push, dict):
        return False
    global_scale = push.get("global_scale")
    if isinstance(global_scale, bool) or not isinstance(global_scale, (int, float)):
        return False
    if not np.isfinite(float(global_scale)):
        return False
    if not isinstance(push.get("bucket_scales"), dict):
        return False
    return True


def price_and_calibrate(
    residual_ledger: pd.DataFrame, *, horizon: str, market_consensus: dict[str, pd.DataFrame],
    calibration_seed: dict[str, dict],
) -> dict[str, pd.DataFrame]:
    """``market_consensus`` maps ``"ATS"``/``"TOTAL"`` -> a consensus frame
    with the certified schema from
    :func:`nfl_hybrid.evaluation.raw_market_reconstruction.reconstruct_market_at_cutoffs`.
    ``calibration_seed`` maps the four stream names to their
    ``production_calibration_seed.json`` entries. Returns
    ``{stream_name: DataFrame}`` with raw + calibrated probabilities --
    market-dependent, so an absent/empty ``market_consensus`` entry for a
    market means that stream is entirely omitted (the caller must treat a
    missing stream as MARKET_NOT_READY for every game in this batch, never
    price at a synthetic line).

    FAIL-CLOSED: a stream is CALIBRATED only when its frozen seed passes
    :func:`_frozen_stream_calibration_ready` (fitted conditional
    coefficient/intercept AND a valid ``push_scale_state`` with
    ``global_scale`` + ``bucket_scales``). If ANY required component is
    absent/malformed, or for any individual row that is not RAW_READY, every
    ``calibrated_*`` probability column is set to NaN and
    ``calibration_status`` is ``CALIBRATION_NOT_READY``. Raw probabilities are
    never recombined into a ``calibrated_``-named field when the frozen
    calibrator is incomplete -- in particular the raw push probability is
    never substituted for a missing frozen push calibrator -- and the
    ``raw_*`` columns remain populated for diagnostics/provenance."""
    cfg = CalibrationConfig()
    priced: dict[str, pd.DataFrame] = {}
    for market in (MARKET_ATS, MARKET_TOTAL):
        consensus = market_consensus.get(market)
        stream = f"{market}_{horizon}"
        if consensus is None or consensus.empty:
            continue
        merged = residual_ledger.merge(
            consensus[["game_id", "consensus_line", "consensus_novig_probability"]],
            on="game_id", how="inner", validate="one_to_one",
        ).reset_index(drop=True)
        if merged.empty:
            continue
        raw = build_raw_probabilities(
            merged, market=market, forecast_horizon=_FORECAST_HORIZON_LABEL[horizon],
            threshold=merged["consensus_line"].to_numpy(float),
        ).reset_index(drop=True)
        if len(raw) != len(merged):
            raise ProductionHardStop("SCHEMA_DRIFT", f"build_raw_probabilities row count changed for {stream}")
        raw["market_line"] = merged["consensus_line"].to_numpy(float)
        raw["market_novig_probability"] = merged["consensus_novig_probability"].to_numpy(float)

        seed_state = calibration_seed.get(stream)
        seed_ready = _frozen_stream_calibration_ready(seed_state)
        raw["calibration_status"] = np.where(
            (raw["raw_status"] == "RAW_READY") & seed_ready, "CALIBRATED", "CALIBRATION_NOT_READY",
        )

        if seed_ready:
            raw["calibrated_conditional_upper_probability"] = apply_frozen_conditional_calibrator(
                raw["raw_conditional_upper_probability"].to_numpy(float), seed_state,
            )
            # seed_ready guarantees push_scale_state + global_scale + bucket_scales
            # are all present and valid -- the raw push probability is NEVER used
            # as a stand-in for a missing frozen push calibrator.
            push_state = seed_state["push_scale_state"]
            legacy_market = _LEGACY_PUSH_MARKET_NAME[market]
            frame_for_push = pd.DataFrame({
                "model_push_probability": raw["raw_push_probability"].to_numpy(float),
                "market_line": raw["market_line"].to_numpy(float),
            })
            calibrated_push = _predict_push(
                frame_for_push, legacy_market, push_state["global_scale"], push_state["bucket_scales"], cfg,
            )
            lower, push_final, upper = _recombine(
                raw["calibrated_conditional_upper_probability"].to_numpy(float), calibrated_push,
            )
        else:
            # FAIL-CLOSED: at least one required frozen calibration component
            # (fitted conditional coefficient/intercept, or push_scale_state's
            # global_scale/bucket_scales) is missing/malformed for this stream.
            # No field whose name begins ``calibrated_`` may carry a number, and
            # the raw push probability is NEVER substituted for an absent frozen
            # push calibrator. The raw_* columns produced above are preserved
            # untouched for diagnostics/provenance.
            nan_col = np.full(len(raw), np.nan)
            raw["calibrated_conditional_upper_probability"] = nan_col
            lower = push_final = upper = nan_col

        # Even with a fitted seed, a row that is not RAW_READY is not genuinely
        # CALIBRATED -- keep every ``calibrated_`` probability fail-closed (NaN)
        # for it too, so a numeric ``calibrated_`` value always means CALIBRATED.
        calibrated_mask = raw["calibration_status"].to_numpy() == "CALIBRATED"
        raw["calibrated_conditional_upper_probability"] = np.where(
            calibrated_mask, raw["calibrated_conditional_upper_probability"].to_numpy(float), np.nan,
        )
        raw["calibrated_lower_probability"] = np.where(calibrated_mask, lower, np.nan)
        raw["calibrated_push_probability"] = np.where(calibrated_mask, push_final, np.nan)
        raw["calibrated_upper_probability"] = np.where(calibrated_mask, upper, np.nan)

        priced[stream] = raw
    return priced


def _consensus_entry(consensus: pd.DataFrame, game_id: str) -> dict | None:
    if consensus.empty:
        return None
    match = consensus[consensus["game_id"].astype(str) == game_id]
    if match.empty:
        return None
    row = match.iloc[0]
    entry = {
        "eligible_books": int(row["eligible_books"]), "consensus_line": float(row["consensus_line"]),
        "consensus_novig_probability": float(row["consensus_novig_probability"]),
        "bookmaker_keys": list(row["bookmaker_keys"]),
        "selected_returned_snapshot_timestamps": list(row["selected_returned_snapshot_timestamps"]),
        "min_observation_age_hours": float(row["min_observation_age_hours"]),
        "max_observation_age_hours": float(row["max_observation_age_hours"]),
        "consensus_method": str(row["consensus_method"]),
    }
    return entry


# ===========================================================================
# Public forecast-of-record identity metadata (additive plumbing only -- see
# the public-identity hardening directive). home_team_id / away_team_id /
# scheduled_kickoff_utc come ONLY from the canonical production games
# population already loaded by run_horizon_batch (load_games_population /
# the injected ``games``) -- never BallDontLie, never any other external
# call, never parsed out of game_id. Joined at this production write
# boundary so the scientific OOF/Elo/Ridge frames and feature contract are
# never touched.
# ===========================================================================
def resolve_forecast_identity(games_df: pd.DataFrame, game_id: str) -> dict:
    """Return ``{"home_team_id", "away_team_id", "scheduled_kickoff_utc"}``
    for ``game_id`` from the canonical games population's matching row.
    ``scheduled_kickoff_utc`` is normalized/serialized as a timezone-aware
    UTC ISO-8601 string. FAILS CLOSED (``ProductionHardStop`` /
    ``SCHEMA_DRIFT``) -- never writes a partial record, never infers a
    replacement -- when: the required columns are absent; no row matches
    ``game_id``; ``home_team_id``/``away_team_id`` is missing or empty;
    ``home_team_id == away_team_id``; or ``scheduled_kickoff_utc`` is
    missing, unparseable, or not timezone-aware."""
    required_cols = {"home_team_id", "away_team_id", "scheduled_kickoff_utc"}
    missing_cols = required_cols - set(games_df.columns)
    if missing_cols:
        raise ProductionHardStop(
            "SCHEMA_DRIFT",
            f"{game_id}: canonical games population is missing required identity column(s): {sorted(missing_cols)}",
        )
    matches = games_df.loc[games_df["game_id"].astype(str) == str(game_id)]
    if matches.empty:
        raise ProductionHardStop("SCHEMA_DRIFT", f"{game_id}: no canonical games-population row for forecast identity")
    row = matches.iloc[0]

    def _clean_team_id(value: object, side: str) -> str:
        if pd.isna(value):
            raise ProductionHardStop("SCHEMA_DRIFT", f"{game_id}: {side}_team_id is missing in the canonical games population")
        text = str(value).strip()
        if not text:
            raise ProductionHardStop("SCHEMA_DRIFT", f"{game_id}: {side}_team_id is empty in the canonical games population")
        return text

    home_team_id = _clean_team_id(row["home_team_id"], "home")
    away_team_id = _clean_team_id(row["away_team_id"], "away")
    if home_team_id == away_team_id:
        raise ProductionHardStop("SCHEMA_DRIFT", f"{game_id}: home_team_id equals away_team_id ({home_team_id!r})")

    raw_kickoff = row["scheduled_kickoff_utc"]
    try:
        kickoff_ts = pd.NaT if pd.isna(raw_kickoff) else pd.Timestamp(raw_kickoff)
    except (ValueError, TypeError) as exc:
        raise ProductionHardStop(
            "SCHEMA_DRIFT", f"{game_id}: scheduled_kickoff_utc is unparseable ({raw_kickoff!r}): {exc}"
        ) from exc
    if pd.isna(kickoff_ts):
        raise ProductionHardStop("SCHEMA_DRIFT", f"{game_id}: scheduled_kickoff_utc is missing in the canonical games population")
    if kickoff_ts.tzinfo is None:
        raise ProductionHardStop("SCHEMA_DRIFT", f"{game_id}: scheduled_kickoff_utc is not timezone-aware ({raw_kickoff!r})")

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "scheduled_kickoff_utc": kickoff_ts.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
    }


def run_horizon_batch(
    *, horizon: str, as_of_utc: pd.Timestamp, force: bool, operational_root: Path | None = None,
    repo_root: Path = REPO_ROOT, games: pd.DataFrame | None = None,
    market_capture_manifest: Path | str | None = None,
) -> dict:
    """One attempted production batch for ``horizon``. Always writes a run
    manifest (Section 15), even on a fail-closed status, and returns it.

    ``operational_root`` is where ``production-2026/{forecast-ledger,
    run-manifests, evaluation-ledger}`` get WRITTEN -- injectable so tests
    (including the real historical integration test, Section 22) never
    touch real production state. It does not redirect any READ of a
    certified/live input: the certified hash sources, the Fix-8
    calibration seed, the games/schedule source (unless ``games`` is
    supplied), and the raw market estate always resolve against the real
    certified/live locations regardless of ``operational_root`` -- they
    are read-only scientific evidence and live data feeds, not
    generated-artifact output, so there is nothing operational to isolate
    for them. The calibration-seed lookup degrades gracefully (never
    crashes) to an empty/uncalibrated seed when the real artifact root
    itself is unavailable (e.g. a hermetic test with no
    ``NFL_MODEL_ARTIFACT_ROOT``).

    ``games`` is injectable so a test can pin the exact historical games
    population; production leaves it ``None`` (uses
    :func:`load_games_population`).

    ``market_capture_manifest`` is the EXACT official BallDontLie capture
    manifest for this card, supplied explicitly -- never auto-selected, and
    validated against this batch's own season/week/horizon/target cutoff
    before a single quote is priced. Left ``None`` (the default), the market
    estate is read exactly as before, so historical/certified runs on a
    machine with no 2026 capture behave identically to today."""
    as_of_utc = _as_utc(as_of_utc)
    aroot = operational_root if operational_root is not None else artifact_root()
    manifest_root = aroot / "production-2026" / "run-manifests"
    ledger_root = aroot / "production-2026" / "forecast-ledger"
    evaluation_root = aroot / "production-2026" / "evaluation-ledger"

    run_id = make_run_id(as_of_utc, horizon)
    started_at = utc_now()
    # ONE authoritative run-level creation timestamp: computed exactly once
    # here and reused verbatim (never recomputed) for every forecast-of-record
    # this batch writes AND for the run manifest. This IS started_at_utc's own
    # instant -- reusing that existing run-start timestamp rather than
    # minting a second, competing one -- just exposed under the field name
    # (``run_created_at_utc``) the future public Wizard exporter looks up.
    run_created_at_utc = _as_utc(started_at).isoformat()
    git_commit = _git_commit(repo_root)

    def _finish(status: str, **extra) -> dict:
        manifest = build_run_manifest(
            run_id=run_id, horizon=horizon, target_cutoff_utc=extra.get("target_cutoff_utc"),
            git_commit=git_commit, source_readiness=extra.get("source_readiness", {}),
            game_count=extra.get("game_count", 0), forecast_count=extra.get("forecast_count", 0),
            abstention_count=extra.get("abstention_count", 0), market_ready_counts=extra.get("market_ready_counts", {}),
            calibration_ready_counts=extra.get("calibration_ready_counts", {}), input_hashes=extra.get("input_hashes", {}),
            output_hashes=extra.get("output_hashes", {}), status=status, started_at_utc=started_at,
            run_created_at_utc=run_created_at_utc, detail=extra.get("detail", ""),
        )
        write_run_manifest(manifest_root, manifest)
        return manifest

    if not force:
        due = is_within_due_window(as_of_utc, horizon)
        if not due["due"]:
            return _finish("NOT_DUE", source_readiness={"due_check": due}, detail="outside the operational execution window")

    try:
        fix71_summary = json.loads((repo_root / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text())
        fix8_prereg = json.loads((repo_root / "outputs" / "fix8_official_oof_calibration_preregistration.json").read_text())
        hash_checks = cert.verify_certified_hashes(fix71_summary, fix8_prereg)
    except FileNotFoundError as exc:
        return _finish("HASH_MISMATCH", detail=f"certified hash source missing: {exc}")
    except cert.CertificationGateFailure as exc:
        return _finish("HASH_MISMATCH", detail=str(exc))
    feature_state_hash = hash_checks["horizon_feature_semantics_hash"]

    target_cutoff_utc = current_or_recent_cutoff(as_of_utc, horizon)
    if target_cutoff_utc > as_of_utc:
        return _finish(
            "NOT_DUE", target_cutoff_utc=str(target_cutoff_utc),
            detail="this horizon's current-week cutoff has not occurred yet; refusing to fabricate a future market snapshot",
        )

    try:
        games_df = load_games_population() if games is None else filter_reg_post(games)
    except Exception as exc:
        return _finish("SCHEDULE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc), detail=str(exc))

    try:
        membership_ledger = he.build_horizon_membership_ledger(games_df)
    except Exception as exc:
        return _finish("ELO_SOURCE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc), detail=str(exc))

    h = horizon.lower()
    card_rows = membership_ledger[membership_ledger[f"{h}_cutoff_utc"] == target_cutoff_utc]
    eligible_ids = set(card_rows.loc[card_rows[f"{h}_eligible"], "game_id"].astype(str))
    # Operational provenance only (no scientific-model behaviour change): the
    # canonical (season, week, season_type) card key is persisted with every
    # forecast so the prospective-strength contract can freeze its 2026
    # REG+POST population and exclude PRESEASON.
    season_type_by_game = {
        str(g): st for g, st in zip(card_rows["game_id"].astype(str), card_rows["season_type"])
    }
    if not eligible_ids:
        return _finish(
            "SCHEDULE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc),
            detail=f"no {horizon} card scheduled at cutoff {target_cutoff_utc} in the games population",
        )

    try:
        matrix = ohf.build_official_horizon_matrix(games_df, horizon, membership_ledger)
        predictions, residual_ledger, fit_counts = ohf.build_official_horizon_oof(
            matrix, horizon=horizon, feature_state_hash=feature_state_hash,
        )
    except Exception as exc:
        return _finish("MODEL_NOT_READY", target_cutoff_utc=str(target_cutoff_utc), detail=str(exc))

    cutoff_match = pd.to_datetime(predictions["target_cutoff_utc"], utc=True) == target_cutoff_utc
    batch_pred = predictions[predictions["game_id"].astype(str).isin(eligible_ids) & cutoff_match].copy()
    resid_cutoff_match = pd.to_datetime(residual_ledger["target_cutoff_utc"], utc=True) == target_cutoff_utc
    batch_resid = residual_ledger[residual_ledger["game_id"].astype(str).isin(eligible_ids) & resid_cutoff_match].copy()
    game_count = len(batch_pred)
    if game_count == 0:
        return _finish(
            "SCHEDULE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc),
            detail="eligible game_ids present in membership ledger but absent from the OOF matrix (schema drift)",
        )

    # Explicit live 2026 market source (Section: certified BDL market bridge).
    # Resolved only when the operator named the exact official capture
    # manifest, and cross-checked against THIS batch's own identity: the
    # card's (season, week), this horizon, and this target cutoff. A capture
    # frozen for a different cutoff can never be priced into this one. A
    # supplied-but-invalid capture is a hard MARKET_SOURCE_UNAVAILABLE -- it
    # is never quietly ignored so the run can proceed on historical data.
    live_quotes = None
    live_market_provenance = None
    if market_capture_manifest is not None:
        card_seasons = sorted({int(s) for s in card_rows["season"]})
        card_weeks = sorted({str(w) for w in card_rows["week"]})
        if len(card_seasons) != 1 or len(card_weeks) != 1:
            return _finish(
                "MARKET_SOURCE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc),
                detail=(
                    f"card at cutoff {target_cutoff_utc} spans seasons={card_seasons} weeks={card_weeks}; "
                    "a live market capture is scoped to exactly one (season, week)"
                ),
            )
        try:
            expected_week = int(card_weeks[0])
        except ValueError:
            expected_week = None
        live_market = evaluate_live_market_source(
            market_capture_manifest,
            expected_season=card_seasons[0],
            expected_week=expected_week,
            expected_horizon=horizon,
            expected_target_cutoff_utc=target_cutoff_utc,
            artifact_root_path=aroot if operational_root is not None else None,
        )
        if not live_market["registered"]:
            return _finish(
                "MARKET_SOURCE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc),
                source_readiness={"live_market": {
                    "status": live_market["status"], "provenance": live_market["provenance"],
                }},
                detail=f"explicit live market capture rejected: {live_market['detail']}",
            )
        live_quotes = live_market["source"].quotes
        live_market_provenance = live_market["provenance"]

    market_snapshot_available = True
    market_error = None
    try:
        quotes = rmr.load_raw_bookmaker_quotes(live_quotes=live_quotes)
    except Exception as exc:
        market_snapshot_available = False
        market_error = str(exc)
        quotes = None

    consensus_by_market: dict[str, pd.DataFrame] = {MARKET_ATS: pd.DataFrame(), MARKET_TOTAL: pd.DataFrame()}
    if market_snapshot_available:
        targets = batch_resid[["game_id"]].drop_duplicates().copy()
        targets["target_cutoff_utc"] = target_cutoff_utc
        for market, raw_key in ((MARKET_ATS, rmr.MARKET_SPREADS), (MARKET_TOTAL, rmr.MARKET_TOTALS)):
            coherent = rmr.build_coherent_book_observations(quotes, raw_key)
            result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market=raw_key)
            consensus_by_market[market] = result.consensus

    # The Fix-8 calibration seed is a read-only certified input, not
    # generated-artifact output -- unlike ``run_preflight``'s
    # ``artifact_root_path``, ``operational_root`` here exists ONLY to
    # isolate this call's forecast-ledger/run-manifest/evaluation-ledger
    # WRITES (see the docstring above), never to redirect where the real
    # certified seed is read from. Always resolved against the real
    # artifact root; gracefully degrades to an empty (uncalibrated) seed
    # -- never a crash -- when that root itself is unavailable (e.g. a
    # hermetic test, or a host with no NFL_MODEL_ARTIFACT_ROOT configured).
    try:
        seed_path = artifact_root() / "fix8-official-oof-calibration-2026" / "production_calibration_seed.json"
        calibration_seed: dict[str, dict] = json.loads(seed_path.read_text()) if seed_path.is_file() else {}
    except Exception:
        calibration_seed = {}

    priced = price_and_calibrate(
        batch_resid, horizon=horizon, market_consensus=consensus_by_market, calibration_seed=calibration_seed,
    )

    forecast_count = 0
    abstention_count = 0
    market_ready_counts = {"ATS": 0, "TOTAL": 0}
    calibration_ready_counts = {"ATS": 0, "TOTAL": 0}
    prediction_hashes: list[str] = []

    for _, row in batch_pred.sort_values("game_id").iterrows():
        game_id = str(row["game_id"])
        # Public forecast identity metadata (Sections 3-5 of the hardening):
        # ONLY from the canonical games population already loaded above,
        # never inferred/parsed/backfilled. Fails closed BEFORE any payload
        # is built or written -- never a partial forecast record.
        identity = resolve_forecast_identity(games_df, game_id)
        model_ready = row["status"] == "OOF"
        prediction_payload = {
            "model_status": str(row["status"]),
            "predicted_margin": float(row["predicted_margin"]) if pd.notna(row["predicted_margin"]) else None,
            "predicted_total": float(row["predicted_total"]) if pd.notna(row["predicted_total"]) else None,
            "model_config_hash": str(row["model_config_hash"]),
        }

        markets_payload: dict[str, dict] = {}
        for market in (MARKET_ATS, MARKET_TOTAL):
            stream = f"{market}_{horizon}"
            consensus_entry = _consensus_entry(consensus_by_market.get(market, pd.DataFrame()), game_id)
            df = priced.get(stream)
            row_match = df[df["game_id"].astype(str) == game_id] if df is not None else None

            if not market_snapshot_available:
                entry = {"status": "MARKET_SOURCE_UNAVAILABLE", "detail": market_error}
            elif consensus_entry is None:
                entry = {"status": "MARKET_NOT_READY"}
            else:
                market_ready_counts[market] += 1
                if row_match is None or row_match.empty or str(row_match.iloc[0]["raw_status"]) != "RAW_READY":
                    entry = {"status": "UNCERTAINTY_NOT_READY", "market": consensus_entry}
                else:
                    r = row_match.iloc[0]
                    calib_status = str(r["calibration_status"])
                    calibrated = calib_status == "CALIBRATED"
                    # Section 24 item 5: only a genuinely calibrated stream
                    # increments the calibration-ready count.
                    if calibrated:
                        calibration_ready_counts[market] += 1
                    entry = {
                        "status": "OK" if calibrated else "CALIBRATION_NOT_READY",
                        "market": consensus_entry,
                        # Raw probabilities stay available for diagnostics/provenance
                        # regardless of calibration readiness (Section 24 item 7).
                        "raw_home_probability": float(r["raw_home_probability"]),
                        "raw_push_probability": float(r["raw_push_probability"]),
                        "raw_away_probability": float(r["raw_away_probability"]),
                        "raw_conditional_upper_probability": float(r["raw_conditional_upper_probability"]),
                        # FAIL-CLOSED: never expose a numeric ``calibrated_``
                        # probability when the frozen calibrator is unavailable.
                        "calibrated_lower_probability": float(r["calibrated_lower_probability"]) if calibrated else None,
                        "calibrated_push_probability": float(r["calibrated_push_probability"]) if calibrated else None,
                        "calibrated_upper_probability": float(r["calibrated_upper_probability"]) if calibrated else None,
                        "calibrated_conditional_upper_probability": (
                            float(r["calibrated_conditional_upper_probability"]) if calibrated else None
                        ),
                        "calibration_status": calib_status,
                    }
            markets_payload[market] = entry

        # Operational provenance only (no scientific-model behaviour change):
        # the explicit deterministic market-state hash, persisted AT FORECAST
        # TIME over the EXACT {ATS, TOTAL} consensus market payload used for
        # pricing -- SHA-256 of canonical_json({"ATS": <consensus market dict
        # or null>, "TOTAL": <consensus market dict or null>}). Volatile
        # provenance (created_at_utc / run_id / git_commit), the model's
        # probabilities and any outcome are excluded by construction. The
        # prospective-strength reporter reconstructs this payload from the
        # immutable record alone and requires exact equality before any
        # market-relative metric. ``None`` when no market state exists.
        market_state_payload = {
            m: (markets_payload[m].get("market")
                if isinstance(markets_payload.get(m), dict) else None)
            for m in (MARKET_ATS, MARKET_TOTAL)
        }
        market_state_hash = (
            _sha256_hex(market_state_payload)
            if any(v is not None for v in market_state_payload.values()) else None
        )

        deterministic_payload = {
            "game_id": game_id, "horizon": horizon, "target_cutoff_utc": str(target_cutoff_utc),
            "season": int(row["season"]), "week": (None if pd.isna(row["week"]) else str(row["week"])),
            "season_type": season_type_by_game.get(game_id),
            "home_team_id": identity["home_team_id"], "away_team_id": identity["away_team_id"],
            "scheduled_kickoff_utc": identity["scheduled_kickoff_utc"],
            "prediction": prediction_payload, "markets": markets_payload,
            "market_state_hash": market_state_hash,
            "certified_baseline_sha": CERTIFIED_SHA, "horizon_feature_semantics_hash": feature_state_hash,
            "operational_model_spec_hash": hash_checks["operational_model_spec_hash"],
            "fix8_preregistration_hash": hash_checks["fix8_preregistration_hash"],
        }

        record = {
            "game_id": game_id, "horizon": horizon, "target_cutoff_utc": str(target_cutoff_utc),
            "created_at_utc": utc_now().isoformat(), "git_commit": git_commit, "run_id": run_id,
            # Shared run-level creation timestamp (Section 6/7): the SAME
            # value, computed exactly once above, for every forecast this
            # batch writes -- never recomputed per game. created_at_utc
            # above is unchanged and keeps representing this individual
            # ledger write's own time.
            "run_created_at_utc": run_created_at_utc,
            "certified_baseline_tag": CERTIFIED_TAG, "certified_baseline_sha": CERTIFIED_SHA,
            "prediction": deterministic_payload,
        }
        write_result = write_forecast(ledger_root, record)
        prediction_hashes.append(write_result.prediction_hash)

        write_evaluation_record(evaluation_root, {
            "game_id": game_id, "horizon": horizon, "target_cutoff_utc": str(target_cutoff_utc),
            "season": int(row["season"]), "season_type": season_type_by_game.get(game_id),
            "forecast": prediction_payload, "markets": markets_payload,
            "market_state_hash": market_state_hash,
            "market_state": {"snapshot_available": market_snapshot_available, "error": market_error},
            "provenance": {"git_commit": git_commit, "run_id": run_id, "created_at_utc": utc_now().isoformat(), **hash_checks},
        })

        if model_ready:
            forecast_count += 1
        else:
            abstention_count += 1

    output_hash = _sha256_hex({"prediction_hashes": sorted(prediction_hashes)})
    return _finish(
        "SUCCESS",
        target_cutoff_utc=str(target_cutoff_utc), game_count=game_count,
        forecast_count=forecast_count, abstention_count=abstention_count,
        market_ready_counts=market_ready_counts, calibration_ready_counts=calibration_ready_counts,
        source_readiness={
            "schedule": True, "elo": True, "market_snapshot_available": market_snapshot_available,
            "market_error": market_error, "live_market": live_market_provenance,
        },
        input_hashes={
            **hash_checks, "games_population_row_count": int(len(games_df)),
            "live_market_capture_sha256": (
                live_market_provenance["manifest_sha256"] if live_market_provenance else None
            ),
        },
        output_hashes={"forecast_batch_hash": output_hash},
    )


# ===========================================================================
# Section 20 -- prospective performance reporter. Reads ONLY immutable
# forecast+attached-result records; never optimizes a threshold, never mines
# a "best cutoff" strategy.
# ===========================================================================
def load_prospective_records(evaluation_root: Path) -> list[dict]:
    """Every evaluation-ledger forecast that has a sibling ``*.result.json``
    attached -- the only rows eligible for prospective scoring. A forecast
    with no attached result is silently excluded (not an error, not yet
    scoreable)."""
    if not evaluation_root.exists():
        return []
    records = []
    for forecast_path in sorted(evaluation_root.glob("*/*.json")):
        name = forecast_path.name
        if name.endswith(".result.json") or name.endswith(".json.tmp"):
            continue
        result_path = forecast_path.with_suffix(".result.json")
        if not result_path.exists():
            continue
        forecast = json.loads(forecast_path.read_text())
        result = json.loads(result_path.read_text())
        records.append({**forecast, "result_record": result})
    return records


def _labeled_stream_rows(records: list[dict], market: str) -> dict:
    y, p_model, p_market, margin_errs_model, margin_errs_market = [], [], [], [], []
    total_errs_model, total_errs_market = [], []
    for rec in records:
        box = rec["result_record"]["result"]
        home_score, away_score = box.get("home_score"), box.get("away_score")
        if home_score is None or away_score is None:
            continue
        actual_margin = float(home_score) - float(away_score)
        actual_total = float(home_score) + float(away_score)

        forecast = rec["forecast"]
        if forecast.get("predicted_margin") is not None:
            margin_errs_model.append(forecast["predicted_margin"] - actual_margin)
        if forecast.get("predicted_total") is not None:
            total_errs_model.append(forecast["predicted_total"] - actual_total)

        entry = rec.get("markets", {}).get(market, {})
        market_snapshot = entry.get("market")
        if market_snapshot is not None:
            line = market_snapshot["consensus_line"]
            if market == MARKET_ATS:
                margin_errs_market.append(-line - actual_margin)
                edge = actual_margin + line
            else:
                total_errs_market.append(line - actual_total)
                edge = actual_total - line
            binary = edge_to_nullable_binary(pd.Series([edge])).iloc[0]
            if pd.isna(binary):
                continue  # push -- excluded from binary scoring, matching the repo's one tie/push policy
            p_cal = entry.get("calibrated_conditional_upper_probability")
            p_raw = entry.get("raw_conditional_upper_probability")
            p_model_value = p_cal if p_cal is not None else p_raw
            if p_model_value is None:
                continue
            y.append(int(binary))
            p_model.append(float(p_model_value))
            p_market.append(float(market_snapshot["consensus_novig_probability"]))

    return {
        "y": np.asarray(y, dtype=float), "p_model": np.asarray(p_model, dtype=float),
        "p_market": np.asarray(p_market, dtype=float),
        "margin_errors_model": margin_errs_model, "margin_errors_market": margin_errs_market,
        "total_errors_model": total_errs_model, "total_errors_market": total_errs_market,
    }


def _rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    arr = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2)))


def compute_prospective_performance(records: list[dict], *, min_sample: int = MIN_PROSPECTIVE_SAMPLE) -> dict:
    """Descriptive only: model-vs-sportsbook margin/total RMSE, ATS/TOTAL
    model log loss/Brier vs market no-vig log loss/Brier with paired deltas,
    calibration ECE, and sample sizes. No threshold search, no "best
    cutoff," no strategy mining. Reports INSUFFICIENT_PROSPECTIVE_SAMPLE
    (overall, and per market/horizon) rather than force a conclusion from
    too few games."""
    if len(records) < min_sample:
        return {"status": "INSUFFICIENT_PROSPECTIVE_SAMPLE", "n": len(records), "min_sample_required": min_sample}

    report: dict = {"status": "OK", "n": len(records), "streams": {}}
    for horizon in he.HORIZONS:
        horizon_records = [r for r in records if r["horizon"] == horizon]
        for market in (MARKET_ATS, MARKET_TOTAL):
            stream = f"{market}_{horizon}"
            rows = _labeled_stream_rows(horizon_records, market)
            n = len(rows["y"])
            if n < min_sample:
                report["streams"][stream] = {"status": "INSUFFICIENT_PROSPECTIVE_SAMPLE", "n": n, "min_sample_required": min_sample}
                continue
            y, p_model, p_market = rows["y"], rows["p_model"], rows["p_market"]
            model_ll, market_ll = binary_log_loss(y, p_model), binary_log_loss(y, p_market)
            model_br, market_br = brier(y, p_model), brier(y, p_market)
            model_ece = equal_mass_ece(y, p_model)
            report["streams"][stream] = {
                "status": "OK", "n": n,
                "margin_rmse_model": _rmse(rows["margin_errors_model"]),
                "margin_rmse_market": _rmse(rows["margin_errors_market"]) if market == MARKET_ATS else None,
                "total_rmse_model": _rmse(rows["total_errors_model"]),
                "total_rmse_market": _rmse(rows["total_errors_market"]) if market == MARKET_TOTAL else None,
                "model_log_loss": model_ll, "market_no_vig_log_loss": market_ll,
                "log_loss_paired_delta": model_ll - market_ll,
                "model_brier": model_br, "market_no_vig_brier": market_br,
                "brier_paired_delta": model_br - market_br,
                "model_ece": model_ece if model_ece == NOT_ESTIMABLE else float(model_ece),
            }
    return report
