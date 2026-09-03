"""NFL 2026 BALLDONTLIE as-of capture -- minimal, append-only raw pregame data layer.

DATA CAPTURE ONLY. This script never touches the v2026.1 model, features,
calibration, thresholds, prediction formulas, certified tags, market-residual
research, or WizardOfOdds output. No model fitting, no backtesting, no feature
engineering.

It preserves, for a single target 2026 week, the exact raw BALLDONTLIE
response bytes for the ephemeral pregame sources listed below, captured
conservatively BEFORE the certified TUE/FRI noon-Eastern card cutoff:

  A  GET /games                          (paginated)
  B  GET /teams/{id}/roster?season=YYYY  (paginated, once per franchise)
  C  GET /player_injuries               (paginated)
  D  GET /fantasy/projections           (paginated; season+week, positions[]=QB)
  E  GET /odds                          (once per target game)
  F  GET /odds/opening                  (once per target game)
  G  GET /odds/player_props             (once per target game)

Raw response bodies are stored verbatim (never reduced) outside the git repo
under ``$NFL_MODEL_DATA_ROOT/live-observation-log/balldontlie-2026/`` in an
immutable, per-capture directory. An existing capture directory is a
FAIL-CLOSED error -- captures are never overwritten.

DURABLE APPEND-ONLY EVIDENCE RULE (C1):
Once an evidence capture directory is created and a request attempt has been
persisted, it MUST NEVER be deleted, overwritten, or reused by this script.
COMPLETE and INCOMPLETE captures are both durable evidence. This module
therefore contains NO delete / rmtree / unlink / cleanup path of any kind:
it only ``mkdir(exist_ok=False)`` a brand-new timestamped directory and
``write_bytes`` / ``write_text`` fresh files into it. A later run always
creates a new ``capture=<UTC timestamp>`` directory and never touches an
earlier one. No temporary/pre-capture working files are written under the
live-observation-log path.

Usage:
  python scripts/capture_bdl_2026_asof.py \
      --season 2026 --week 1 --season-type REG --horizon SMOKE

TUE/FRI captures must genuinely run inside the capture window; a run outside
it can only proceed with --allow-off-window-smoke, which forces the recorded
horizon to SMOKE (it can never masquerade as a production TUE/FRI capture).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.providers.balldontlie.client import BDLConfig  # noqa: E402
from nfl_hybrid.providers.balldontlie.team_crosswalk import registry  # noqa: E402

SCHEMA_VERSION = "bdl-2026-asof-capture-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.balldontlie.io/nfl/v1"
EASTERN = ZoneInfo("America/New_York")

# season_type: production capture must exclude preseason. REG -> 2, POST -> 3.
SEASON_TYPE_CODES: dict[str, int] = {"REG": 2, "POST": 3}

HORIZON_OFFSET_DAYS: dict[str, int] = {"TUE": 1, "FRI": 4}  # from the card's Monday
CUTOFF_LOCAL_TIME = dtime(12, 0)  # 12:00 PM America/New_York

# Conservative capture window for a production TUE/FRI run, relative to the
# nominal cutoff: acquire in the hours just before it, never long after.
WINDOW_LEAD = timedelta(hours=8)
WINDOW_TRAIL = timedelta(minutes=15)

RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0)  # <=2 retries, short exp backoff
MAX_RETRIES = 2
RETRY_SLEEP_CAP_SECONDS = 5.0
MAX_PAGES = 500

_SECRET_MARKERS = ("key", "token", "authorization", "secret", "password", "apikey", "bearer")

# logical source -> ordered candidate keys for its own record-level timestamp.
TIMESTAMP_CANDIDATES: dict[str, tuple[str, ...]] = {
    "odds_current": ("updated_at", "last_updated_at", "last_update"),
    "odds_opening": ("opened_at", "opening_at", "updated_at"),
    "player_props": ("updated_at", "last_updated_at", "last_update"),
    "fantasy_qb_projections": ("collected_at", "updated_at", "created_at"),
}

REQUIRED_SOURCES: tuple[str, ...] = (
    "games",
    "rosters",
    "injuries",
    "fantasy_qb_projections",
    "odds_current",
    "odds_opening",
    "player_props",
)


# ---------------------------------------------------------------------------
# Pure helpers (all independently unit-tested; no network, no clock).
# ---------------------------------------------------------------------------
def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def deterministic_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop any query parameter whose name looks like a credential. The API
    key travels only in the Authorization header, which is never recorded."""
    return {
        k: v
        for k, v in params.items()
        if not any(marker in str(k).lower() for marker in _SECRET_MARKERS)
    }


def encode_params(params: dict[str, Any]) -> list[tuple[str, str]]:
    """BDL array params use repeated ``key[]=value`` query args."""
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                pairs.append((f"{key}[]", str(item)))
        else:
            pairs.append((key, str(value)))
    return pairs


def resolve_season_type(label: str) -> tuple[str, int]:
    """REG/POST only. Preseason is never a valid production capture target."""
    key = str(label).strip().upper()
    if key not in SEASON_TYPE_CODES:
        raise ValueError(
            f"season_type must be REG or POST (regular/postseason only); got {label!r}. "
            "Preseason capture is deliberately unsupported."
        )
    return key, SEASON_TYPE_CODES[key]


def parse_iso_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 string to a tz-aware UTC datetime. A naive string is
    assumed UTC. Returns None for anything unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text), dtime(0, 0))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def nominal_cutoff_from_card_monday(card_monday: str, horizon: str) -> datetime:
    """Certified cutoff: card Monday + offset days, 12:00 America/New_York,
    DST-aware, returned as UTC. TUE = Monday+1, FRI = Monday+4."""
    if horizon not in HORIZON_OFFSET_DAYS:
        raise ValueError(f"card-monday derivation only defined for TUE/FRI, got {horizon!r}")
    monday = date.fromisoformat(card_monday)
    local = datetime.combine(
        monday + timedelta(days=HORIZON_OFFSET_DAYS[horizon]), CUTOFF_LOCAL_TIME, tzinfo=EASTERN
    )
    return local.astimezone(timezone.utc)


def parse_nominal_cutoff(value: str) -> datetime:
    """Accept a full ISO timestamp or a bare date. A bare date or a naive
    timestamp is interpreted in America/New_York (the certified cutoff zone),
    then converted to UTC."""
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=EASTERN)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass
    day = date.fromisoformat(text)  # raises ValueError on garbage -- caller handles
    return datetime.combine(day, CUTOFF_LOCAL_TIME, tzinfo=EASTERN).astimezone(timezone.utc)


def within_capture_window(now_utc: datetime, cutoff_utc: datetime) -> bool:
    return (cutoff_utc - WINDOW_LEAD) <= now_utc <= (cutoff_utc + WINDOW_TRAIL)


def resolve_effective_horizon(
    requested: str, now_utc: datetime, nominal_cutoff_utc: datetime | None, allow_off_window: bool
) -> tuple[str, str | None]:
    """Return (effective_horizon, downgrade_reason). SMOKE stays SMOKE. A
    TUE/FRI request run outside the capture window is either refused (hard
    error) or, with allow_off_window, downgraded to SMOKE so it can never be
    recorded as a production cutoff capture."""
    if requested == "SMOKE":
        return "SMOKE", None
    if requested not in HORIZON_OFFSET_DAYS:
        raise ValueError(f"unknown horizon {requested!r}")
    if nominal_cutoff_utc is None:
        raise ValueError(f"{requested} capture requires --nominal-cutoff or --card-monday")
    if within_capture_window(now_utc, nominal_cutoff_utc):
        return requested, None
    if allow_off_window:
        return "SMOKE", (
            f"requested {requested} but run at {now_utc.isoformat()} is outside the capture "
            f"window for nominal cutoff {nominal_cutoff_utc.isoformat()} "
            f"([-{WINDOW_LEAD}, +{WINDOW_TRAIL}]); forced to SMOKE"
        )
    raise OffWindowError(
        f"{requested} capture at {now_utc.isoformat()} is outside the window for nominal cutoff "
        f"{nominal_cutoff_utc.isoformat()}. Re-run inside the window, or pass "
        "--allow-off-window-smoke (which records horizon=SMOKE, not {requested})."
    )


def eligibility_counts(records: list[Any], candidate_keys: tuple[str, ...], cutoff_utc: datetime) -> dict:
    """Count records whose own source timestamp is <= the nominal cutoff vs
    > it. Raw rows are never dropped or mutated -- this is a summary only."""
    at_or_before = after = missing = 0
    resolved_key: str | None = None
    for row in records:
        value = None
        if isinstance(row, dict):
            for key in candidate_keys:
                candidate = row.get(key)
                if candidate not in (None, ""):
                    value = candidate
                    resolved_key = resolved_key or key
                    break
        parsed = parse_iso_utc(value)
        if parsed is None:
            missing += 1
        elif parsed <= cutoff_utc:
            at_or_before += 1
        else:
            after += 1
    return {
        "candidate_keys": list(candidate_keys),
        "resolved_key": resolved_key,
        "record_count": len(records),
        "at_or_before_cutoff": at_or_before,
        "after_cutoff": after,
        "missing_or_unparseable_timestamp": missing,
    }


class OffWindowError(RuntimeError):
    """A TUE/FRI capture attempted outside its window without an explicit smoke override."""


class FailClosedError(RuntimeError):
    """The target capture directory already exists -- captures are immutable."""


# ---------------------------------------------------------------------------
# HTTP layer -- exact raw bytes, bounded retries, no hidden loops.
# ---------------------------------------------------------------------------
@dataclass
class HttpResult:
    endpoint: str
    method: str
    query_parameters: dict[str, Any]
    request_started_at_utc: str
    response_received_at_utc: str
    http_status: int | None
    body_bytes: bytes | None
    attempts: int
    error: str | None

    @property
    def ok(self) -> bool:
        return self.http_status is not None and 200 <= self.http_status < 300


@dataclass
class CaptureContext:
    session: Any
    base_url: str
    timeout_seconds: float
    _api_key: str
    capture_dir: Path
    manifest_requests: list[dict] = field(default_factory=list)
    pagination_notes: dict[str, str] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        # Built inline on every call; never stored anywhere serialisable.
        return {"Authorization": self._api_key}

    def record(self, logical_name: str, result: HttpResult, output_filename: str | None, page_index: int | None) -> None:
        body = result.body_bytes or b""
        self.manifest_requests.append(
            {
                "logical_name": logical_name,
                "endpoint": result.endpoint,
                "method": result.method,
                "query_parameters": redact_params(result.query_parameters),
                "request_started_at_utc": result.request_started_at_utc,
                "response_received_at_utc": result.response_received_at_utc,
                "http_status": result.http_status,
                "response_byte_count": len(body),
                "response_sha256": sha256_hex(body) if result.body_bytes is not None else None,
                "output_filename": output_filename,
                "page_index": page_index,
                "attempts": result.attempts,
                "error": result.error,
            }
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def request_with_retries(ctx: CaptureContext, path: str, params: dict[str, Any]) -> HttpResult:
    url = f"{ctx.base_url.rstrip('/')}/{path.lstrip('/')}"
    redacted = redact_params(params)
    started = _now_iso()
    last_error: str | None = None
    attempts = 0
    for attempt in range(1 + MAX_RETRIES):
        attempts = attempt + 1
        try:
            response = ctx.session.get(
                url, params=encode_params(params), headers=ctx.headers(), timeout=ctx.timeout_seconds
            )
            status = int(response.status_code)
            content = response.content if isinstance(response.content, (bytes, bytearray)) else bytes(response.content)
            content = bytes(content)
            if status == 429 or status >= 500:
                last_error = f"HTTP {status}"
                if attempt < MAX_RETRIES:
                    retry_after = _retry_after_seconds(response)
                    time.sleep(min(retry_after or RETRY_BACKOFF_SECONDS[attempt], RETRY_SLEEP_CAP_SECONDS))
                    continue
                return HttpResult(url, "GET", redacted, started, _now_iso(), status, content, attempts, last_error)
            error = None if 200 <= status < 300 else f"HTTP {status}"
            return HttpResult(url, "GET", redacted, started, _now_iso(), status, content, attempts, error)
        except Exception as exc:  # noqa: BLE001 -- network/timeout/transport; bounded retries only
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                continue
            return HttpResult(url, "GET", redacted, started, _now_iso(), None, None, attempts, last_error)
    # unreachable
    return HttpResult(url, "GET", redacted, started, _now_iso(), None, None, attempts, last_error)


def _retry_after_seconds(response: Any) -> float | None:
    try:
        value = response.headers.get("Retry-After")
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _parse_body(result: HttpResult) -> dict | None:
    if not result.ok or result.body_bytes is None:
        return None
    try:
        parsed = json.loads(result.body_bytes)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _records_from_body(body: dict | None) -> list[Any]:
    if body is None:
        return []
    data = body.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def capture_paginated(
    ctx: CaptureContext, logical_name: str, path: str, params: dict[str, Any], stem: str
) -> tuple[bool, list[Any]]:
    """Follow ``meta.next_cursor`` to exhaustion, persisting every page's exact
    raw bytes as a numbered file. Never assumes the first page is complete."""
    page_params = dict(params)
    page_params.setdefault("per_page", 100)
    cursor: Any = None
    seen: set[Any] = set()
    page_index = 0
    results: list[HttpResult] = []
    records: list[Any] = []
    while True:
        page_index += 1
        this_params = dict(page_params)
        if cursor is not None:
            this_params["cursor"] = cursor
        result = request_with_retries(ctx, path, this_params)
        filename = f"{stem}.p{page_index:03d}.json"
        (ctx.capture_dir / filename).write_bytes(result.body_bytes or b"")
        ctx.record(logical_name, result, filename, page_index)
        results.append(result)
        if not result.ok:
            break
        body = _parse_body(result)
        records.extend(_records_from_body(body))
        next_cursor = (body or {}).get("meta", {}).get("next_cursor") if body else None
        if not next_cursor:
            break
        if next_cursor in seen:
            ctx.pagination_notes[logical_name] = f"cursor repeated ({next_cursor!r}); stopped to avoid a loop"
            break
        if page_index >= MAX_PAGES:
            ctx.pagination_notes[logical_name] = f"hit max_pages={MAX_PAGES}; stopped"
            break
        seen.add(next_cursor)
        cursor = next_cursor
    ok = len(results) > 0 and all(r.ok for r in results)
    return ok, records


def capture_single(
    ctx: CaptureContext, logical_name: str, path: str, params: dict[str, Any], filename: str
) -> tuple[bool, list[Any]]:
    """One non-paginated GET; persist the exact raw bytes verbatim."""
    result = request_with_retries(ctx, path, params)
    (ctx.capture_dir / filename).write_bytes(result.body_bytes or b"")
    ctx.record(logical_name, result, filename, None)
    return result.ok, _records_from_body(_parse_body(result))


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptureRequest:
    season: int
    week: int
    season_type_label: str
    requested_horizon: str
    nominal_cutoff_utc: datetime | None
    allow_off_window_smoke: bool = False
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0
    data_root: str | os.PathLike | None = None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _resolve_data_root(override: str | os.PathLike | None) -> Path:
    raw = override if override is not None else os.environ.get("NFL_MODEL_DATA_ROOT")
    if not raw:
        raise FailClosedError(
            "NFL_MODEL_DATA_ROOT is not set and no data_root override was given. Point it at the "
            "private data estate root before capturing."
        )
    return Path(raw).expanduser()


def run_capture(
    request: CaptureRequest,
    *,
    session: Any | None = None,
    api_key: str | None = None,
    now_utc: datetime | None = None,
) -> dict:
    season_type_label, season_type_code = resolve_season_type(request.season_type_label)
    now = now_utc or datetime.now(timezone.utc)
    effective_horizon, downgrade_reason = resolve_effective_horizon(
        request.requested_horizon, now, request.nominal_cutoff_utc, request.allow_off_window_smoke
    )
    nominal_cutoff = request.nominal_cutoff_utc or now  # SMOKE with no cutoff -> the run time
    capture_started = now
    stamp = capture_started.strftime("%Y%m%dT%H%M%SZ")

    root = _resolve_data_root(request.data_root)
    capture_dir = (
        root
        / "live-observation-log"
        / "balldontlie-2026"
        / f"season={request.season}"
        / f"week={request.week:02d}"
        / f"horizon={effective_horizon}"
        / f"capture={stamp}"
    )
    # Durable append-only evidence rule (C1): a pre-existing capture directory
    # is NEVER deleted, cleared, or reused -- it is another run's durable
    # evidence (COMPLETE or INCOMPLETE alike). Fail closed and let the caller
    # start a fresh capture=<timestamp> directory instead. The only filesystem
    # mutations anywhere below this point are mkdir(exist_ok=False) of this
    # brand-new directory and write_bytes/write_text of fresh files inside it.
    if capture_dir.exists():
        raise FailClosedError(f"FAIL CLOSED: capture directory already exists: {capture_dir}")
    capture_dir.mkdir(parents=True, exist_ok=False)

    resolved_key = api_key or BDLConfig().resolved_api_key()
    if session is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise FailClosedError("Install the data extras (`pip install -e '.[data]'`) to run a live capture.") from exc
        session = requests.Session()

    ctx = CaptureContext(
        session=session,
        base_url=request.base_url,
        timeout_seconds=request.timeout_seconds,
        _api_key=resolved_key,
        capture_dir=capture_dir,
    )

    source_ok: dict[str, bool] = {}
    timestamped_records: dict[str, list[Any]] = {}

    # A. games -----------------------------------------------------------
    games_ok, game_rows = capture_paginated(
        ctx,
        "games",
        "/games",
        {"seasons": [request.season], "weeks": [request.week], "season_type": season_type_code},
        "games",
    )
    source_ok["games"] = games_ok
    target_games = _target_game_summary(game_rows)

    # B. team rosters --------------------------------------------------
    roster_results: list[bool] = []
    for bdl_team_id in sorted(registry().keys()):
        r_ok, _ = capture_paginated(
            ctx,
            "rosters",
            f"/teams/{bdl_team_id}/roster",
            {"season": request.season},
            f"roster_team_{bdl_team_id}",
        )
        roster_results.append(r_ok)
    source_ok["rosters"] = len(roster_results) > 0 and all(roster_results)

    # C. player injuries --------------------------------------------
    injuries_ok, _ = capture_paginated(ctx, "injuries", "/player_injuries", {}, "injuries")
    source_ok["injuries"] = injuries_ok

    # D. weekly QB fantasy projections ------------------------------
    proj_ok, proj_rows = capture_paginated(
        ctx,
        "fantasy_qb_projections",
        "/fantasy/projections",
        {"season": request.season, "week": request.week, "positions": ["QB"]},
        "fantasy_qb_projections",
    )
    source_ok["fantasy_qb_projections"] = proj_ok
    timestamped_records["fantasy_qb_projections"] = proj_rows

    # E. current game odds -- whole target-week slate in one call.
    #    BDL contract (verified live 2026-09-03): /odds requires
    #    (season AND week) OR game_ids[]; a singular game_id is rejected 400.
    odds_now_ok, odds_now_rows = capture_paginated(
        ctx, "odds_current", "/odds", {"season": request.season, "week": request.week}, "odds_current"
    )
    source_ok["odds_current"] = odds_now_ok
    timestamped_records["odds_current"] = odds_now_rows

    # F. opening game odds -- same contract as /odds.
    odds_open_ok, odds_open_rows = capture_paginated(
        ctx, "odds_opening", "/odds/opening", {"season": request.season, "week": request.week}, "odds_opening"
    )
    source_ok["odds_opening"] = odds_open_ok
    timestamped_records["odds_opening"] = odds_open_rows

    # G. live player props -- once per target game (game_id is honoured here).
    game_ids = [g["bdl_game_id"] for g in target_games["games"] if g["bdl_game_id"] is not None]
    if not game_ids:
        ctx.manifest_requests.append(
            {
                "logical_name": "player_props",
                "endpoint": "/odds/player_props",
                "method": "GET",
                "query_parameters": {},
                "request_started_at_utc": _now_iso(),
                "response_received_at_utc": _now_iso(),
                "http_status": None,
                "response_byte_count": 0,
                "response_sha256": None,
                "output_filename": None,
                "page_index": None,
                "attempts": 0,
                "error": "SKIPPED_NO_TARGET_GAMES",
            }
        )
        source_ok["player_props"] = False
        timestamped_records["player_props"] = []
    else:
        prop_ok: list[bool] = []
        prop_rows: list[Any] = []
        for gid in game_ids:
            g_ok, rows = capture_single(
                ctx, "player_props", "/odds/player_props", {"game_id": gid}, f"player_props_game_{gid}.json"
            )
            prop_ok.append(g_ok)
            prop_rows.extend(rows)
        source_ok["player_props"] = len(prop_ok) > 0 and all(prop_ok)
        timestamped_records["player_props"] = prop_rows

    # Eligibility summary (raw rows are never mutated -- summary only).
    eligibility = {
        name: eligibility_counts(timestamped_records.get(name, []), TIMESTAMP_CANDIDATES[name], nominal_cutoff)
        for name in TIMESTAMP_CANDIDATES
    }
    (capture_dir / "eligibility_summary.json").write_text(
        json.dumps(eligibility, indent=2, sort_keys=True), encoding="utf-8"
    )

    required_ok = {name: source_ok.get(name, False) for name in REQUIRED_SOURCES}
    status = "COMPLETE" if all(required_ok.values()) else "INCOMPLETE"
    request_count = len(ctx.manifest_requests)
    success_count = sum(1 for r in ctx.manifest_requests if r["http_status"] is not None and 200 <= r["http_status"] < 300)
    failure_count = request_count - success_count

    manifest_body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "capture_dir": str(capture_dir),
        "season": request.season,
        "week": request.week,
        "season_type": season_type_label,
        "season_type_code": season_type_code,
        "horizon": effective_horizon,
        "requested_horizon": request.requested_horizon,
        "horizon_downgrade_reason": downgrade_reason,
        "allow_off_window_smoke": request.allow_off_window_smoke,
        "nominal_cutoff_utc": nominal_cutoff.isoformat().replace("+00:00", "Z"),
        "nominal_cutoff_provided": request.nominal_cutoff_utc is not None,
        "capture_started_at_utc": capture_started.isoformat().replace("+00:00", "Z"),
        "capture_completed_at_utc": _now_iso(),
        "git_commit": _git_commit(),
        "capture_uuid": str(uuid.uuid4()),
        "base_url": request.base_url,
        "request_count": request_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "required_source_ok": required_ok,
        "pagination_notes": ctx.pagination_notes,
        "target_games": target_games,
        "eligibility_summary": eligibility,
        "requests": ctx.manifest_requests,
        "scientific_model_files_changed": False,
    }
    manifest_body["manifest_sha256"] = sha256_hex(deterministic_json(manifest_body).encode("utf-8"))
    (capture_dir / "manifest.json").write_text(
        json.dumps(manifest_body, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_body


def _target_game_summary(game_rows: list[Any]) -> dict:
    games: list[dict] = []
    reg = post = 0
    for row in game_rows:
        if not isinstance(row, dict):
            continue
        postseason = bool(row.get("postseason"))
        post += int(postseason)
        reg += int(not postseason)
        games.append(
            {
                "bdl_game_id": row.get("id"),
                "date": row.get("date"),
                "week": row.get("week"),
                "season": row.get("season"),
                "postseason": postseason,
                "status_state": row.get("status_state"),
            }
        )
    return {
        "count": len(games),
        "regular_season_rows": reg,
        "postseason_rows": post,
        "games": games,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture raw BALLDONTLIE pregame data for one 2026 week.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season-type", choices=sorted(SEASON_TYPE_CODES), required=True,
                        help="REG or POST only -- preseason capture is unsupported.")
    parser.add_argument("--horizon", choices=["TUE", "FRI", "SMOKE"], required=True)
    parser.add_argument("--nominal-cutoff", default=None,
                        help="ISO timestamp or bare date of the certified cutoff (naive/date => America/New_York).")
    parser.add_argument("--card-monday", default=None,
                        help="Card week's Monday (YYYY-MM-DD); TUE/FRI cutoff derived at 12:00 America/New_York.")
    parser.add_argument("--allow-off-window-smoke", action="store_true",
                        help="Permit a TUE/FRI request outside its window; forces recorded horizon to SMOKE.")
    parser.add_argument("--data-root", default=None, help="Override NFL_MODEL_DATA_ROOT (tests / dry runs).")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def _resolve_cutoff(args: argparse.Namespace) -> datetime | None:
    if args.nominal_cutoff:
        return parse_nominal_cutoff(args.nominal_cutoff)
    if args.card_monday and args.horizon in HORIZON_OFFSET_DAYS:
        return nominal_cutoff_from_card_monday(args.card_monday, args.horizon)
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cutoff = _resolve_cutoff(args)
    if args.horizon in HORIZON_OFFSET_DAYS and cutoff is None:
        print(f"ERROR: {args.horizon} capture requires --nominal-cutoff or --card-monday.", file=sys.stderr)
        return 2

    request = CaptureRequest(
        season=args.season,
        week=args.week,
        season_type_label=args.season_type,
        requested_horizon=args.horizon,
        nominal_cutoff_utc=cutoff,
        allow_off_window_smoke=args.allow_off_window_smoke,
        timeout_seconds=args.timeout,
        data_root=args.data_root,
    )
    try:
        manifest = run_capture(request)
    except OffWindowError as exc:
        print(f"ERROR (off-window): {exc}", file=sys.stderr)
        return 2
    except FailClosedError as exc:
        print(f"ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2

    tg = manifest["target_games"]
    elig = manifest["eligibility_summary"]
    print(f"capture directory : {manifest['capture_dir']}")
    print(f"manifest status   : {manifest['status']}")
    print(f"horizon           : {manifest['horizon']} (requested {manifest['requested_horizon']})")
    print(f"nominal cutoff utc : {manifest['nominal_cutoff_utc']}")
    print(f"games found        : {tg['count']} (REG rows {tg['regular_season_rows']}, POST rows {tg['postseason_rows']})")
    print(f"requests           : {manifest['request_count']} ok={manifest['success_count']} failed={manifest['failure_count']}")
    for name in REQUIRED_SOURCES:
        print(f"  required {name:<24} ok={manifest['required_source_ok'][name]}")
    for name, summary in sorted(elig.items()):
        print(f"  eligibility {name:<22} <=cutoff={summary['at_or_before_cutoff']} "
              f">cutoff={summary['after_cutoff']} missing_ts={summary['missing_or_unparseable_timestamp']}")
    print(f"manifest sha256    : {manifest['manifest_sha256']}")
    return 0 if manifest["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
