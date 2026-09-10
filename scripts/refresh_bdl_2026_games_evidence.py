"""DAILY: refresh immutable 2026 completed-game (and schedule) evidence from
BallDontLie.

DATA CAPTURE ONLY. This script fits nothing, prices nothing, calibrates
nothing, publishes nothing, and never touches a forecast. It acquires the
``/games`` endpoint for season 2026 and persists the exact raw response bytes
into a new, immutable, append-only capture directory -- then leaves the
canonical normalization and the durable population update to
``scripts/update_2026_games_population.py``.

NOTHING HERE IS NEW MACHINERY
  The HTTP layer, bounded retries, cursor pagination, per-page raw-byte
  persistence, request recording, redaction, deterministic manifest encoding
  and manifest hashing are all imported verbatim from the existing capture
  script ``scripts/capture_bdl_2026_asof.py``. There is no second HTTP client,
  no second retry policy and no second manifest schema: the manifest written
  here is ``bdl-2026-asof-capture-v1``, the same schema
  :func:`nfl_hybrid.data.bdl_market_bridge.validate_capture_manifest` already
  verifies.

  The recorded ``horizon`` is
  :data:`nfl_hybrid.data.games_population_2026.GAMES_EVIDENCE_HORIZON`, never
  TUE/FRI/SMOKE. That label carries no cutoff and no market, so this capture
  can never be mistaken for -- or accepted as -- an official TUE/FRI market
  capture: the certified market path keeps the validator's TUE/FRI default.

ONE CAPTURE DIRECTORY PER SEASON TYPE
  BDL's ``/games`` response cannot itself distinguish preseason from regular
  season (verified live; week numbering restarts at 1 for both), so the
  season-type filter used for a batch IS the evidence for that batch's season
  type. REG and POST are therefore captured as separate immutable captures,
  each recording the filter it was fetched with, which is what the existing
  canonicalizer consumes as its ``season_type_hint``. PRESEASON is never
  requested.

APPEND-ONLY
  Every run creates a brand-new ``capture=<UTC timestamp>`` directory. An
  existing directory is a fail-closed error; nothing is ever deleted,
  overwritten or reused (the same durable-evidence rule the pregame capture
  script enforces).

Usage:
  python scripts/refresh_bdl_2026_games_evidence.py --season 2026
  python scripts/refresh_bdl_2026_games_evidence.py --season 2026 --season-type REG
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.data import games_population_2026 as gp26  # noqa: E402

# The existing capture script is a sibling script, not an importable package
# module, so it is loaded by explicit path -- the same way the v2 public
# exporter loads the v1 exporter's helpers. Loading it defines constants and
# functions only (guarded ``__main__``), performs no I/O and no network call.
_CAPTURE_SCRIPT_PATH = Path(__file__).resolve().parent / "capture_bdl_2026_asof.py"


_CAPTURE_MODULE_NAME = "_bdl_2026_asof_capture"


def _load_capture_module():
    existing = sys.modules.get(_CAPTURE_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_CAPTURE_MODULE_NAME, _CAPTURE_SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the capture helpers from {_CAPTURE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE execution: the capture script declares @dataclass
    # types, and dataclasses resolves a class's annotations through
    # ``sys.modules[cls.__module__]``.
    sys.modules[_CAPTURE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_CAPTURE_MODULE_NAME, None)
        raise
    return module


_cap = _load_capture_module()

SCHEMA_VERSION = _cap.SCHEMA_VERSION
GAMES_LOGICAL_NAME = "games"
EVIDENCE_NAMESPACE = "balldontlie-2026-games-evidence"

# Only the games source is required for schedule/results evidence. Recorded in
# the manifest under the same ``required_source_ok`` field the existing
# validator reads.
REQUIRED_SOURCES: tuple[str, ...] = (GAMES_LOGICAL_NAME,)


class GamesEvidenceError(RuntimeError):
    """The evidence capture cannot proceed without overwriting durable
    evidence or fabricating a value."""


def evidence_capture_dir(
    data_root: Path, *, season: int, season_type: str, stamp: str
) -> Path:
    """``$NFL_MODEL_DATA_ROOT/live-observation-log/
    balldontlie-2026-games-evidence/season=YYYY/season_type=REG/
    capture=<stamp>/``.

    A sibling of -- never inside -- the pregame ``balldontlie-2026``
    observation log, so a daily results refresh can never be confused with an
    official TUE/FRI pregame capture.
    """
    return (
        Path(data_root)
        / "live-observation-log"
        / EVIDENCE_NAMESPACE
        / f"season={season}"
        / f"season_type={season_type}"
        / f"capture={stamp}"
    )


def capture_games_evidence(
    *,
    season: int,
    season_type: str,
    session=None,
    api_key: str | None = None,
    now_utc: datetime | None = None,
    data_root: str | Path | None = None,
    base_url: str = _cap.DEFAULT_BASE_URL,
    timeout_seconds: float = 30.0,
) -> dict:
    """Acquire and persist one immutable ``/games`` evidence capture."""
    season_type_label, season_type_code = _cap.resolve_season_type(season_type)
    now = now_utc or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    root = _cap._resolve_data_root(data_root)
    capture_dir = evidence_capture_dir(root, season=season, season_type=season_type_label, stamp=stamp)
    if capture_dir.exists():
        raise GamesEvidenceError(f"FAIL CLOSED: evidence capture directory already exists: {capture_dir}")
    capture_dir.mkdir(parents=True, exist_ok=False)

    resolved_key = api_key or _cap.BDLConfig().resolved_api_key()
    if session is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise GamesEvidenceError(
                "Install the data extras (`pip install -e '.[data]'`) to run a live capture."
            ) from exc
        session = requests.Session()

    ctx = _cap.CaptureContext(
        session=session,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        _api_key=resolved_key,
        capture_dir=capture_dir,
    )

    # The whole season in one paginated sweep: completed games arrive with
    # final scores, upcoming games with null scores. No week filter -- a daily
    # refresh must see every result that has landed since the last one.
    games_ok, game_rows = _cap.capture_paginated(
        ctx,
        GAMES_LOGICAL_NAME,
        "/games",
        {"seasons": [season], "season_type": season_type_code},
        GAMES_LOGICAL_NAME,
    )

    required_ok = {GAMES_LOGICAL_NAME: games_ok}
    status = "COMPLETE" if all(required_ok.values()) else "INCOMPLETE"
    request_count = len(ctx.manifest_requests)
    success_count = sum(
        1 for r in ctx.manifest_requests if r["http_status"] is not None and 200 <= r["http_status"] < 300
    )

    manifest_body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "capture_dir": str(capture_dir),
        "capture_purpose": "GAMES_SCHEDULE_AND_RESULTS_EVIDENCE",
        "season": season,
        # A season-wide games sweep is not scoped to one week. Recorded as 0
        # (an impossible NFL week) so nothing can mistake this evidence for a
        # week-scoped market capture, which must match a real card week.
        "week": 0,
        "season_type": season_type_label,
        "season_type_code": season_type_code,
        "horizon": gp26.GAMES_EVIDENCE_HORIZON,
        "requested_horizon": gp26.GAMES_EVIDENCE_HORIZON,
        "horizon_downgrade_reason": None,
        "allow_off_window_smoke": False,
        "nominal_cutoff_utc": now.isoformat().replace("+00:00", "Z"),
        "nominal_cutoff_provided": False,
        "capture_started_at_utc": now.isoformat().replace("+00:00", "Z"),
        "capture_completed_at_utc": _cap._now_iso(),
        "git_commit": _cap._git_commit(),
        "capture_uuid": str(uuid.uuid4()),
        "base_url": base_url,
        "request_count": request_count,
        "success_count": success_count,
        "failure_count": request_count - success_count,
        "required_source_ok": required_ok,
        "pagination_notes": ctx.pagination_notes,
        "target_games": _cap._target_game_summary(game_rows),
        "eligibility_summary": {},
        "requests": ctx.manifest_requests,
        "scientific_model_files_changed": False,
    }
    manifest_body["manifest_sha256"] = _cap.sha256_hex(
        _cap.deterministic_json(manifest_body).encode("utf-8")
    )
    (capture_dir / "manifest.json").write_text(
        json.dumps(manifest_body, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_body


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", type=int, default=gp26.PRODUCTION_SEASON)
    parser.add_argument(
        "--season-type",
        action="append",
        choices=sorted(_cap.SEASON_TYPE_CODES),
        default=None,
        help="Repeatable. REG or POST only (preseason is never captured). Default: REG and POST.",
    )
    parser.add_argument("--data-root", default=None, help="Override NFL_MODEL_DATA_ROOT (tests / dry runs).")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    season_types = args.season_type or list(gp26.PRODUCTION_SEASON_TYPES)

    manifests: list[dict] = []
    exit_code = 0
    for season_type in season_types:
        try:
            manifest = capture_games_evidence(
                season=args.season,
                season_type=season_type,
                data_root=args.data_root,
                timeout_seconds=args.timeout,
            )
        except (GamesEvidenceError, _cap.FailClosedError) as exc:
            print(f"ERROR (fail-closed) [{season_type}]: {exc}", file=sys.stderr)
            return 2
        manifests.append(manifest)
        if manifest["status"] != "COMPLETE":
            exit_code = 1
        target = manifest["target_games"]
        print(
            json.dumps(
                {
                    "season_type": season_type,
                    "status": manifest["status"],
                    "capture_dir": manifest["capture_dir"],
                    "manifest_path": str(Path(manifest["capture_dir"]) / "manifest.json"),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "games_found": target["count"],
                    "regular_season_rows": target["regular_season_rows"],
                    "postseason_rows": target["postseason_rows"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
