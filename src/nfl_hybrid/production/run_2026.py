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


def run_preflight(*, repo_root: Path = REPO_ROOT, artifact_root_path: Path | None = None) -> dict:
    """Verifies live infrastructure without ever printing a secret value.
    ``artifact_root_path`` is injectable so a test can point EVERY
    generated-artifact lookup this call performs -- the writable-output-
    directory check AND the Fix-8 production calibration seed lookup --
    at one isolated root (``aroot``), consistently, for the whole
    invocation. Only the certified hash sources, the games/schedule
    source, and the raw market estate remain the real certified/live
    inputs regardless of ``artifact_root_path`` (they are read-only
    scientific evidence and live data feeds, not generated-artifact
    output, so there is nothing operational to isolate for them)."""
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
    # No "odds_history.2026" (or any other live/current-season) key is
    # registered in nfl_hybrid.data.external_data -- there is currently no
    # live market source wired into the certified rmr machinery. Reported
    # honestly rather than silently degrading to a fabricated/synthetic line.
    live_2026_market_source_registered = False
    checks["live_2026_market_source_registered"] = live_2026_market_source_registered

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

    infra_ready = not blocking
    market_2026_ready = schedule_2026_available and live_2026_market_source_registered
    if infra_ready and market_2026_ready:
        overall_status = "READY"
    elif infra_ready:
        overall_status = "READY_WAITING_FOR_FIRST_DUE_CUTOFF"
    else:
        overall_status = "NOT_READY"

    return {
        "overall_status": overall_status,
        "infra_ready": infra_ready,
        "schedule_2026_available": schedule_2026_available,
        "live_2026_market_source_registered": live_2026_market_source_registered,
        "blocking_problems": blocking,
        "checks": checks,
    }


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
    input_hashes: dict, output_hashes: dict, status: str, started_at_utc: pd.Timestamp, detail: str = "",
) -> dict:
    return {
        "run_id": run_id, "started_at_utc": _as_utc(started_at_utc).isoformat(), "completed_at_utc": utc_now().isoformat(),
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


def load_games_population() -> pd.DataFrame:
    """The one canonical games/schedule source (per certification GROUNDING):
    :func:`nfl_hybrid.data.external_data.resolve` ``"backfill.games"``.
    Currently covers seasons 2020-2025 only -- see :func:`run_preflight`'s
    ``schedule_2026_status`` check, which reports this honestly instead of
    fabricating 2026 rows."""
    return pd.read_parquet(resolve("backfill.games"))


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
    price at a synthetic line)."""
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
        seed_fitted = bool(seed_state and seed_state.get("conditional_calibrator_state", {}).get("fitted"))
        if seed_fitted:
            raw["calibrated_conditional_upper_probability"] = apply_frozen_conditional_calibrator(
                raw["raw_conditional_upper_probability"].to_numpy(float), seed_state,
            )
        else:
            raw["calibrated_conditional_upper_probability"] = np.nan
        raw["calibration_status"] = np.where(
            (raw["raw_status"] == "RAW_READY") & seed_fitted, "CALIBRATED", "CALIBRATION_NOT_READY",
        )

        push_state = seed_state.get("push_scale_state") if seed_fitted else None
        if push_state is not None:
            legacy_market = _LEGACY_PUSH_MARKET_NAME[market]
            frame_for_push = pd.DataFrame({
                "model_push_probability": raw["raw_push_probability"].to_numpy(float),
                "market_line": raw["market_line"].to_numpy(float),
            })
            calibrated_push = _predict_push(
                frame_for_push, legacy_market, push_state["global_scale"], push_state["bucket_scales"], cfg,
            )
        else:
            calibrated_push = raw["raw_push_probability"].to_numpy(float)

        conditional_for_recombine = raw["calibrated_conditional_upper_probability"].to_numpy(float)
        raw_conditional = raw["raw_conditional_upper_probability"].to_numpy(float)
        conditional_for_recombine = np.where(np.isnan(conditional_for_recombine), raw_conditional, conditional_for_recombine)
        lower, push_final, upper = _recombine(conditional_for_recombine, calibrated_push)
        raw["calibrated_lower_probability"] = lower
        raw["calibrated_push_probability"] = push_final
        raw["calibrated_upper_probability"] = upper

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


def run_horizon_batch(
    *, horizon: str, as_of_utc: pd.Timestamp, force: bool, operational_root: Path | None = None,
    repo_root: Path = REPO_ROOT, games: pd.DataFrame | None = None,
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
    :func:`load_games_population`)."""
    as_of_utc = _as_utc(as_of_utc)
    aroot = operational_root if operational_root is not None else artifact_root()
    manifest_root = aroot / "production-2026" / "run-manifests"
    ledger_root = aroot / "production-2026" / "forecast-ledger"
    evaluation_root = aroot / "production-2026" / "evaluation-ledger"

    run_id = make_run_id(as_of_utc, horizon)
    started_at = utc_now()
    git_commit = _git_commit(repo_root)

    def _finish(status: str, **extra) -> dict:
        manifest = build_run_manifest(
            run_id=run_id, horizon=horizon, target_cutoff_utc=extra.get("target_cutoff_utc"),
            git_commit=git_commit, source_readiness=extra.get("source_readiness", {}),
            game_count=extra.get("game_count", 0), forecast_count=extra.get("forecast_count", 0),
            abstention_count=extra.get("abstention_count", 0), market_ready_counts=extra.get("market_ready_counts", {}),
            calibration_ready_counts=extra.get("calibration_ready_counts", {}), input_hashes=extra.get("input_hashes", {}),
            output_hashes=extra.get("output_hashes", {}), status=status, started_at_utc=started_at,
            detail=extra.get("detail", ""),
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
        games_df = games if games is not None else load_games_population()
    except Exception as exc:
        return _finish("SCHEDULE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc), detail=str(exc))

    try:
        membership_ledger = he.build_horizon_membership_ledger(games_df)
    except Exception as exc:
        return _finish("ELO_SOURCE_UNAVAILABLE", target_cutoff_utc=str(target_cutoff_utc), detail=str(exc))

    h = horizon.lower()
    card_rows = membership_ledger[membership_ledger[f"{h}_cutoff_utc"] == target_cutoff_utc]
    eligible_ids = set(card_rows.loc[card_rows[f"{h}_eligible"], "game_id"].astype(str))
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

    market_snapshot_available = True
    market_error = None
    try:
        quotes = rmr.load_raw_bookmaker_quotes()
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
                    if calib_status == "CALIBRATED":
                        calibration_ready_counts[market] += 1
                    entry = {
                        "status": "OK" if calib_status == "CALIBRATED" else "CALIBRATION_NOT_READY",
                        "market": consensus_entry,
                        "raw_home_probability": float(r["raw_home_probability"]),
                        "raw_push_probability": float(r["raw_push_probability"]),
                        "raw_away_probability": float(r["raw_away_probability"]),
                        "raw_conditional_upper_probability": float(r["raw_conditional_upper_probability"]),
                        "calibrated_lower_probability": float(r["calibrated_lower_probability"]),
                        "calibrated_push_probability": float(r["calibrated_push_probability"]),
                        "calibrated_upper_probability": float(r["calibrated_upper_probability"]),
                        "calibrated_conditional_upper_probability": (
                            float(r["calibrated_conditional_upper_probability"])
                            if pd.notna(r["calibrated_conditional_upper_probability"]) else None
                        ),
                        "calibration_status": calib_status,
                    }
            markets_payload[market] = entry

        deterministic_payload = {
            "game_id": game_id, "horizon": horizon, "target_cutoff_utc": str(target_cutoff_utc),
            "season": int(row["season"]), "week": (None if pd.isna(row["week"]) else str(row["week"])),
            "prediction": prediction_payload, "markets": markets_payload,
            "certified_baseline_sha": CERTIFIED_SHA, "horizon_feature_semantics_hash": feature_state_hash,
            "operational_model_spec_hash": hash_checks["operational_model_spec_hash"],
            "fix8_preregistration_hash": hash_checks["fix8_preregistration_hash"],
        }

        record = {
            "game_id": game_id, "horizon": horizon, "target_cutoff_utc": str(target_cutoff_utc),
            "created_at_utc": utc_now().isoformat(), "git_commit": git_commit, "run_id": run_id,
            "certified_baseline_tag": CERTIFIED_TAG, "certified_baseline_sha": CERTIFIED_SHA,
            "prediction": deterministic_payload,
        }
        write_result = write_forecast(ledger_root, record)
        prediction_hashes.append(write_result.prediction_hash)

        write_evaluation_record(evaluation_root, {
            "game_id": game_id, "horizon": horizon, "target_cutoff_utc": str(target_cutoff_utc),
            "season": int(row["season"]), "forecast": prediction_payload, "markets": markets_payload,
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
            "market_error": market_error,
        },
        input_hashes={**hash_checks, "games_population_row_count": int(len(games_df))},
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
