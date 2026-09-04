"""wizard-nfl-predictions-v1 -- public publishing exporter.

PUBLISHING TRANSLATION ONLY. Reads ONE already-created official 2026 TUE/FRI
production run -- an EXPLICIT run manifest (written by
``src/nfl_hybrid/production/run_2026.py::build_run_manifest``) plus the
EXPLICIT forecast-ledger horizon directory it produced -- and writes the
public ``wizard-nfl-predictions-v1`` JSON contract.

This script does NOT generate predictions, fit any model, call BallDontLie,
call The Odds API, call any other external API, compute ATS/TOTAL/moneyline
probabilities, or read prospective QB-shadow data. It reads no environment
variable and requires no credential. It never guesses which run is
"current" -- both the manifest and the forecast directory are explicit CLI
arguments; there is no "find latest" behavior anywhere in this module.

Frozen source field mapping (established by the public-identity hardening,
PR #33 -- do not rediscover):

    public game_id            <- top-level game_id
    public kickoff_utc        <- prediction.scheduled_kickoff_utc
    public away_team          <- prediction.away_team_id   (already a
    public home_team          <- prediction.home_team_id      canonical
                                  NFL abbreviation -- no further mapping)
    public predicted_home_margin <- prediction.prediction.predicted_margin
    public predicted_game_total  <- prediction.prediction.predicted_total
    public season              <- prediction.season
    public week                <- prediction.week
    public horizon             <- top-level horizon
    public generated_at_utc    <- run_created_at_utc (the ONE authoritative
                                   shared run-creation timestamp; never
                                   created_at_utc, mtime, target_cutoff_utc,
                                   or the export's own clock)

Usage:
    python scripts/export_wizard_nfl_predictions.py \\
        --run-manifest $NFL_MODEL_ARTIFACT_ROOT/production-2026/run-manifests/<run_id>.json \\
        --forecast-dir $NFL_MODEL_ARTIFACT_ROOT/production-2026/forecast-ledger/<TUE|FRI> \\
        --output $NFL_MODEL_ARTIFACT_ROOT/public/wizardofodds/nfl/latest.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "wizard-nfl-predictions-v1"
ALLOWED_HORIZONS = ("TUE", "FRI")
SUPPORTED_SEASONS = (2026,)

TOP_LEVEL_KEY_ORDER = ("schema_version", "season", "week", "horizon", "generated_at_utc", "games")
GAME_KEY_ORDER = ("game_id", "kickoff_utc", "away_team", "home_team", "predicted_home_margin", "predicted_game_total")

_REQUIRED_MANIFEST_FIELDS = ("run_id", "run_created_at_utc", "horizon", "status", "game_count", "output_hashes")


class WizardExportError(RuntimeError):
    """Fail-closed export error. Never caught-and-substituted -- aborts the
    whole export with no output written (no partial card, no partial
    latest.json, no differing archive overwrite)."""


def _fail(message: str) -> None:
    raise WizardExportError(message)


# --------------------------------------------------------------------------- #
# canonical-json / sha256 -- byte-identical to
# nfl_hybrid.production.run_2026._canonical_json / _sha256_hex, reimplemented
# here (not imported) so this publishing utility carries no dependency on the
# model-fitting module. Used ONLY to re-verify existing, already-persisted
# hashes -- never to compute a new one that changes what's published.
# --------------------------------------------------------------------------- #
def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(payload: dict) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


# --------------------------------------------------------------------------- #
# timestamp handling
# --------------------------------------------------------------------------- #
def _parse_utc_instant(raw: Any, *, field_name: str) -> datetime:
    """Parse an absolute instant. FAILS CLOSED unless it is a non-empty
    string, parseable, and timezone-aware."""
    if not isinstance(raw, str) or not raw.strip():
        _fail(f"{field_name} is missing")
        raise AssertionError("unreachable")  # pragma: no cover
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        _fail(f"{field_name} is not a parseable timestamp: {raw!r}")
        raise AssertionError("unreachable")  # pragma: no cover
    if parsed.tzinfo is None:
        _fail(f"{field_name} is not timezone-aware: {raw!r}")
    return parsed.astimezone(timezone.utc)


def _format_utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# season / week
# --------------------------------------------------------------------------- #
def _validate_season(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"season must be an integer, got {value!r}")
    if value not in SUPPORTED_SEASONS:
        _fail(f"season {value} is not supported by {SCHEMA_VERSION} (only {SUPPORTED_SEASONS})")
    return value


def _parse_week(raw: Any) -> int:
    """Accept the canonical production week representation ONLY if it is
    losslessly interpretable as an integer NFL week: an int, or a digits-only
    string (``"1"``, ``"01"``). Reject ``None``, ``"Week 1"``, ``"1.5"``,
    floats, booleans -- never guess."""
    if isinstance(raw, bool):
        _fail(f"week must be an integer, got boolean {raw!r}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    _fail(f"week value {raw!r} is not losslessly interpretable as an integer NFL week")
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# team identity -- verbatim from prediction.home_team_id / away_team_id.
# These are ALREADY canonical NFL abbreviations (PR #33); no mapping helper
# is called again here, and the value is never altered before being emitted.
# --------------------------------------------------------------------------- #
def _validate_team(value: Any, *, game_id: str, side: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{game_id}: {side}_team_id is missing/empty in the forecast record")
    return value


# --------------------------------------------------------------------------- #
# numeric prediction fields -- preserved EXACTLY as produced by the model.
# No rounding, clamping, or rescaling ever happens here.
# --------------------------------------------------------------------------- #
def _validate_numeric(value: Any, *, game_id: str, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{game_id}: {field_name} must be a numeric value, got {value!r}")
    if not math.isfinite(value):
        _fail(f"{game_id}: {field_name} must be finite, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# run manifest (read-only; this module never writes one)
# --------------------------------------------------------------------------- #
def load_run_manifest(path: Path) -> dict:
    if not path.is_file():
        _fail(f"run manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        _fail(f"run manifest is not valid JSON: {path} ({exc})")
        raise AssertionError("unreachable")  # pragma: no cover

    missing = [f for f in _REQUIRED_MANIFEST_FIELDS if f not in manifest]
    if missing:
        _fail(f"run manifest {path} is missing required field(s) {missing}")
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        _fail(f"run manifest {path}: run_id must be a non-empty string")
    if manifest["horizon"] not in ALLOWED_HORIZONS:
        _fail(f"run manifest {path}: horizon {manifest['horizon']!r} is not one of {ALLOWED_HORIZONS}")
    if not isinstance(manifest["run_created_at_utc"], str) or not manifest["run_created_at_utc"]:
        _fail(f"run manifest {path}: run_created_at_utc must be a non-empty string")
    if manifest["status"] != "SUCCESS":
        _fail(f"run manifest {path}: status is {manifest['status']!r}, not SUCCESS -- refusing to publish")
    if isinstance(manifest["game_count"], bool) or not isinstance(manifest["game_count"], int):
        _fail(f"run manifest {path}: game_count must be an integer")
    output_hashes = manifest["output_hashes"]
    if not isinstance(output_hashes, dict) or not isinstance(output_hashes.get("forecast_batch_hash"), str):
        _fail(f"run manifest {path}: output_hashes.forecast_batch_hash is missing -- cannot prove card completeness")
    return manifest


# --------------------------------------------------------------------------- #
# forecast selection -- ONLY files inside the explicit --forecast-dir, ONLY
# those whose run_id matches the explicit manifest.
# --------------------------------------------------------------------------- #
def _load_forecast_records(forecast_dir: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in sorted(forecast_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            _fail(f"forecast-ledger file is not valid JSON: {path} ({exc})")
            raise AssertionError("unreachable")  # pragma: no cover
        records.append((path, record))
    return records


def select_run_forecasts(manifest: dict, forecast_dir: Path) -> list[dict]:
    """Select exactly the forecast-of-record files that belong to
    ``manifest['run_id']``. A file belonging to another run is excluded, not
    an error (the directory legitimately accumulates every run's evidence).
    A file that CLAIMS this run_id but disagrees with the manifest on
    horizon or run_created_at_utc is a genuine integrity contradiction and
    FAILS CLOSED. Duplicate game_id within the run FAILS CLOSED."""
    run_id = manifest["run_id"]
    selected: list[dict] = []
    seen_game_ids: set[str] = set()
    for path, record in _load_forecast_records(forecast_dir):
        if record.get("run_id") != run_id:
            continue  # belongs to another run -- excluded, never mixed in
        if record.get("horizon") != manifest["horizon"]:
            _fail(f"{path}: run_id matches {run_id!r} but horizon {record.get('horizon')!r} != manifest horizon {manifest['horizon']!r}")
        if record.get("run_created_at_utc") != manifest["run_created_at_utc"]:
            _fail(f"{path}: run_id matches {run_id!r} but run_created_at_utc disagrees with the manifest")
        game_id = record.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            _fail(f"{path}: missing/empty top-level game_id")
        if game_id in seen_game_ids:
            _fail(f"duplicate game_id {game_id!r} within run {run_id!r}")
        seen_game_ids.add(game_id)
        selected.append(record)

    expected_count = manifest["game_count"]
    if len(selected) != expected_count:
        _fail(
            f"selected {len(selected)} forecast record(s) for run {run_id!r} but the run manifest "
            f"declares game_count={expected_count} -- refusing to publish a partial or over-selected card"
        )
    return selected


# --------------------------------------------------------------------------- #
# per-game public translation (no field beyond the frozen six ever enters
# the returned dict -- market/QB-shadow data cannot leak because it is
# never read in the first place)
# --------------------------------------------------------------------------- #
def _build_public_game(record: dict) -> tuple[datetime, dict]:
    game_id = record.get("game_id")
    prediction = record.get("prediction")
    if not isinstance(prediction, dict):
        _fail(f"{game_id}: forecast record has no prediction payload")

    kickoff_dt = _parse_utc_instant(prediction.get("scheduled_kickoff_utc"), field_name=f"{game_id}: scheduled_kickoff_utc")

    home_team = _validate_team(prediction.get("home_team_id"), game_id=game_id, side="home")
    away_team = _validate_team(prediction.get("away_team_id"), game_id=game_id, side="away")
    if home_team == away_team:
        _fail(f"{game_id}: home_team_id equals away_team_id ({home_team!r})")

    inner = prediction.get("prediction")
    if not isinstance(inner, dict):
        _fail(f"{game_id}: forecast record has no nested prediction.prediction payload")
    margin = _validate_numeric(inner.get("predicted_margin"), game_id=game_id, field_name="predicted_margin")
    total = _validate_numeric(inner.get("predicted_total"), game_id=game_id, field_name="predicted_total")

    public_game = {
        "game_id": game_id,
        "kickoff_utc": _format_utc_z(kickoff_dt),
        "away_team": away_team,
        "home_team": home_team,
        "predicted_home_margin": margin,
        "predicted_game_total": total,
    }
    assert tuple(public_game.keys()) == GAME_KEY_ORDER
    return kickoff_dt, public_game


# --------------------------------------------------------------------------- #
# card assembly
# --------------------------------------------------------------------------- #
def build_public_card(manifest: dict, forecast_dir: Path) -> dict:
    selected = select_run_forecasts(manifest, forecast_dir)
    if not selected:
        _fail(
            f"run {manifest['run_id']!r} status=SUCCESS but selected zero forecasts -- "
            "season/week cannot be established for an empty card"
        )

    seasons: set[int] = set()
    weeks: set[Any] = set()
    prediction_hashes: list[str] = []
    dated_games: list[tuple[datetime, dict]] = []

    for record in selected:
        prediction = record["prediction"]

        stored_hash = record.get("prediction_hash")
        if not isinstance(stored_hash, str) or not stored_hash:
            _fail(f"{record.get('game_id')}: forecast record has no prediction_hash")
        if _sha256_hex(prediction) != stored_hash:
            _fail(f"{record.get('game_id')}: prediction_hash does not match its own prediction payload")
        prediction_hashes.append(stored_hash)

        seasons.add(prediction.get("season"))
        weeks.add(prediction.get("week"))

        dated_games.append(_build_public_game(record))

    if len(seasons) > 1:
        _fail(f"mixed season across selected forecasts for run {manifest['run_id']!r}: {sorted(seasons, key=str)}")
    if len(weeks) > 1:
        _fail(f"mixed week across selected forecasts for run {manifest['run_id']!r}: {sorted(weeks, key=str)}")

    recomputed_batch_hash = _sha256_hex({"prediction_hashes": sorted(prediction_hashes)})
    if recomputed_batch_hash != manifest["output_hashes"]["forecast_batch_hash"]:
        _fail(
            f"recomputed forecast_batch_hash {recomputed_batch_hash} does not match run manifest "
            f"output_hashes.forecast_batch_hash {manifest['output_hashes']['forecast_batch_hash']}"
        )

    season = _validate_season(next(iter(seasons)))
    week = _parse_week(next(iter(weeks)))

    dated_games.sort(key=lambda pair: (pair[0], pair[1]["game_id"]))
    games = [g for _, g in dated_games]

    generated_at_utc = _format_utc_z(_parse_utc_instant(manifest["run_created_at_utc"], field_name="run_created_at_utc"))

    card = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "week": week,
        "horizon": manifest["horizon"],
        "generated_at_utc": generated_at_utc,
        "games": games,
    }
    assert tuple(card.keys()) == TOP_LEVEL_KEY_ORDER
    return card


def serialize_card(card: dict) -> bytes:
    text = json.dumps(card, indent=2, allow_nan=False, ensure_ascii=True)
    return (text + "\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# output: immutable archive first, then atomic latest.json
# --------------------------------------------------------------------------- #
def archive_path(archive_root: Path, *, season: int, week: int, horizon: str) -> Path:
    return archive_root / f"season={season}" / f"week={week:02d}" / f"horizon={horizon}.json"


def write_archive(archive_root: Path, *, season: int, week: int, horizon: str, payload_bytes: bytes) -> Path:
    path = archive_path(archive_root, season=season, week=week, horizon=horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload_bytes:
            return path  # idempotent success
        _fail(f"archive file already exists with different content -- refusing to overwrite: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload_bytes)
    tmp.rename(path)
    return path


def write_latest(output_path: Path, payload_bytes: bytes) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    tmp.write_bytes(payload_bytes)
    os.replace(tmp, output_path)  # atomic on POSIX
    return output_path


def run_export(*, run_manifest_path: Path, forecast_dir: Path, output_path: Path) -> dict:
    """The whole pipeline: manifest validation -> card completeness ->
    forecast validation -> deterministic serialization -> immutable archive
    -> atomic latest.json. Raises WizardExportError (never partially writes
    anything) on any failure."""
    manifest = load_run_manifest(run_manifest_path)
    if not forecast_dir.is_dir():
        _fail(f"forecast-dir does not exist or is not a directory: {forecast_dir}")
    card = build_public_card(manifest, forecast_dir)
    payload_bytes = serialize_card(card)

    archive_root = output_path.parent / "archive"
    archived_path = write_archive(
        archive_root, season=card["season"], week=card["week"], horizon=card["horizon"], payload_bytes=payload_bytes,
    )
    latest = write_latest(output_path, payload_bytes)
    return {
        "status": "OK",
        "run_id": manifest["run_id"],
        "season": card["season"],
        "week": card["week"],
        "horizon": card["horizon"],
        "generated_at_utc": card["generated_at_utc"],
        "game_count": len(card["games"]),
        "archive_path": str(archived_path),
        "latest_path": str(latest),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-manifest", required=True, type=str, help="explicit run-manifest JSON path (no auto-selection)")
    parser.add_argument("--forecast-dir", required=True, type=str, help="explicit forecast-ledger/<horizon> directory")
    parser.add_argument("--output", required=True, type=str, help="explicit latest.json output path")
    args = parser.parse_args(argv)

    try:
        result = run_export(
            run_manifest_path=Path(args.run_manifest),
            forecast_dir=Path(args.forecast_dir),
            output_path=Path(args.output),
        )
    except WizardExportError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
