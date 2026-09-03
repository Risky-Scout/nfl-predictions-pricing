"""V2026.7 — Prospective sparse QB-projection-shock shadow signal.

ONE prospective 2026 shadow signal, built ONLY from the immutable append-only
BallDontLie TUE/FRI production captures under

    $NFL_MODEL_DATA_ROOT/live-observation-log/balldontlie-2026/

Question: does the Tuesday->Friday change in the primary QB's projected passing
yards carry information beyond the Friday sportsbook spread?

This script does NOT touch the production model, calibration, or any external
API. It locates the official persisted TUE and FRI captures for one week,
keeps only games whose kickoff is STRICTLY after the official Friday nominal
cutoff (v2026.7.1 chronology gate), resolves each team's primary projected QB
(max projected passing attempts), computes the single numerical predictor
``QB_PROJECTION_SHOCK_FRI``, and appends immutable rows to a shadow ledger
OUTSIDE Git under

    $NFL_MODEL_ARTIFACT_ROOT/prospective-v2026-7-qb-projection-shock/

If both production captures are not yet available it exits cleanly with
``NOT_READY_WAITING_FOR_TUE_FRI_CAPTURES`` (a success state, not a failure).

Usage:
    python scripts/update_v2026_7_qb_projection_shock_shadow.py --season 2026 --week 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIGNAL_NAME = "QB_PROJECTION_SHOCK_FRI"
SIGNAL_VERSION = "v2026.7.1"
PRODUCTION_HORIZONS = ("TUE", "FRI")
PROJECTIONS_GLOB = "fantasy_qb_projections.p*.json"
GAMES_GLOB = "games.p*.json"
WARMUP_GAMES = 64
LEDGER_SUBDIR = "prospective-v2026-7-qb-projection-shock"
LEDGER_FILENAME = "shadow_signal_ledger.jsonl"
PREREGISTRATION_FILENAME = "v2026_7_prospective_qb_projection_shock_preregistration.json"

STATUS_NOT_READY = "NOT_READY_WAITING_FOR_TUE_FRI_CAPTURES"
STATUS_OK = "OK"

# v2026.7.1 Friday kickoff eligibility classifications (Section 3 of the fix).
KICKOFF_FUTURE = "FUTURE_AT_FRI_CUTOFF"
KICKOFF_ALREADY_STARTED = "ALREADY_STARTED_AT_FRI_CUTOFF"
KICKOFF_INVALID = "INVALID_KICKOFF"


class FailClosedError(RuntimeError):
    """Raised when provenance/integrity is ambiguous — never guess, stop."""


# --------------------------------------------------------------------------- #
# paths / roots
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def preregistration_path() -> Path:
    return repo_root() / "outputs" / PREREGISTRATION_FILENAME


def _resolve_root(override: str | None, env_var: str) -> Path:
    raw = override if override is not None else os.environ.get(env_var)
    if not raw:
        raise FailClosedError(f"{env_var} is not set and no override was given.")
    return Path(raw).expanduser()


def data_root(override: str | None = None) -> Path:
    return _resolve_root(override, "NFL_MODEL_DATA_ROOT")


def artifact_root(override: str | None = None) -> Path:
    return _resolve_root(override, "NFL_MODEL_ARTIFACT_ROOT")


def week_capture_base(root: Path, season: int, week: int) -> Path:
    return (
        root
        / "live-observation-log"
        / "balldontlie-2026"
        / f"season={season}"
        / f"week={week:02d}"
    )


# --------------------------------------------------------------------------- #
# preregistration hash
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_preregistration_hash(frozen: dict) -> str:
    return hashlib.sha256(canonical_json(frozen).encode("utf-8")).hexdigest()


def load_preregistration() -> dict:
    doc = json.loads(preregistration_path().read_text())
    frozen = doc["frozen"]
    recomputed = compute_preregistration_hash(frozen)
    if recomputed != doc.get("preregistration_sha256"):
        raise FailClosedError(
            "preregistration hash mismatch: file has "
            f"{doc.get('preregistration_sha256')!r}, recomputed {recomputed!r}"
        )
    return doc


# --------------------------------------------------------------------------- #
# capture resolution
# --------------------------------------------------------------------------- #
def _read_manifest(capture_dir: Path) -> dict | None:
    manifest = capture_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        return json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return None


def resolve_official_capture(root: Path, season: int, week: int, horizon: str) -> Path | None:
    """Return the single official COMPLETE production capture directory for
    ``horizon`` (``TUE`` or ``FRI``), or ``None`` if none exists yet.

    A capture qualifies only if its own manifest records ``status == "COMPLETE"``
    AND ``horizon == <horizon>`` — so SMOKE captures, and any TUE/FRI request
    that was downgraded to SMOKE, are excluded. If more than one COMPLETE
    capture exists for a production horizon and nothing in the manifest
    designates one unambiguously, fail closed rather than pick.
    """
    if horizon not in PRODUCTION_HORIZONS:
        raise ValueError(f"{horizon!r} is not a production horizon {PRODUCTION_HORIZONS}")
    horizon_dir = week_capture_base(root, season, week) / f"horizon={horizon}"
    if not horizon_dir.is_dir():
        return None
    qualifying: list[Path] = []
    for capture_dir in sorted(p for p in horizon_dir.glob("capture=*") if p.is_dir()):
        manifest = _read_manifest(capture_dir)
        if manifest is None:
            continue
        if manifest.get("status") != "COMPLETE":
            continue
        if manifest.get("horizon") != horizon:
            continue
        qualifying.append(capture_dir)
    if not qualifying:
        return None
    if len(qualifying) > 1:
        raise FailClosedError(
            f"{len(qualifying)} COMPLETE {horizon} captures for season={season} week={week} "
            f"and no unambiguous designation: {[p.name for p in qualifying]}"
        )
    return qualifying[0]


def manifest_sha256(capture_dir: Path) -> str:
    manifest = _read_manifest(capture_dir)
    if manifest is None:
        raise FailClosedError(f"no readable manifest.json in {capture_dir}")
    value = manifest.get("manifest_sha256")
    if not value:
        raise FailClosedError(f"manifest.json in {capture_dir} has no manifest_sha256")
    return str(value)


# --------------------------------------------------------------------------- #
# projections / primary QB resolution
# --------------------------------------------------------------------------- #
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_qb_projection_records(capture_dir: Path) -> list[dict]:
    pages = sorted(capture_dir.glob(PROJECTIONS_GLOB))
    if not pages:
        raise FailClosedError(f"no {PROJECTIONS_GLOB} in {capture_dir}")
    records: list[dict] = []
    for page in pages:
        payload = json.loads(page.read_text())
        records.extend(payload.get("data", []))
    return records


def parse_utc(raw: Any) -> datetime | None:
    """Parse an ISO-8601 instant as timezone-aware UTC. Returns ``None`` for a
    missing, non-string, naive, or unparseable value (caller fails closed)."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_games(capture_dir: Path) -> list[dict]:
    pages = sorted(capture_dir.glob(GAMES_GLOB))
    if not pages:
        raise FailClosedError(f"no {GAMES_GLOB} in {capture_dir}")
    games: list[dict] = []
    for page in pages:
        payload = json.loads(page.read_text())
        for row in payload.get("data", []):
            home = (row.get("home_team") or {}).get("abbreviation")
            away = (row.get("visitor_team") or {}).get("abbreviation")
            if row.get("id") is None or not home or not away:
                continue
            games.append(
                {
                    "game_id": str(row["id"]),
                    "home_team": home,
                    "away_team": away,
                    "kickoff_raw": row.get("date"),
                    "scheduled_kickoff_utc": parse_utc(row.get("date")),
                }
            )
    return games


def friday_nominal_cutoff_utc(fri_dir: Path) -> datetime:
    """The official Friday nominal cutoff, taken verbatim from the Friday
    capture's manifest.json ``nominal_cutoff_utc`` — never recomputed or
    guessed. Fail closed if it is absent or unparseable."""
    manifest = _read_manifest(fri_dir)
    if manifest is None:
        raise FailClosedError(f"no readable manifest.json in {fri_dir}")
    cutoff = parse_utc(manifest.get("nominal_cutoff_utc"))
    if cutoff is None:
        raise FailClosedError(
            f"Friday manifest in {fri_dir} has no parseable nominal_cutoff_utc "
            f"({manifest.get('nominal_cutoff_utc')!r})"
        )
    return cutoff


def classify_friday_kickoff(kickoff_utc: datetime | None, fri_cutoff_utc: datetime) -> str:
    """A v2026.7.1 Friday shadow row is eligible only when
    ``scheduled_kickoff_utc > official FRI nominal_cutoff_utc`` (STRICTLY
    greater). Games at or before the cutoff, or with no valid kickoff, are
    omitted before QB projection resolution."""
    if kickoff_utc is None:
        return KICKOFF_INVALID
    if kickoff_utc > fri_cutoff_utc:
        return KICKOFF_FUTURE
    return KICKOFF_ALREADY_STARTED


def team_primary_qb(records: list[dict], team_abbr: str) -> dict:
    """Resolve a team's PRIMARY_PROJECTED_QB = the QB with the greatest projected
    passing_attempts. Returns ``{"status": "OK", ...}`` on success,
    ``{"status": "TIE"}`` when the greatest passing_attempts is shared, or
    ``{"status": "MISSING"}`` when no QB record has the required numeric fields.

    Realized starters, depth charts, injuries and player props are never
    consulted.
    """
    candidates = []
    for rec in records:
        if rec.get("position") != "QB":
            continue
        if (rec.get("team") or {}).get("abbreviation") != team_abbr:
            continue
        stats = rec.get("stats") or {}
        attempts = stats.get("passing_attempts")
        yards = stats.get("passing_yards")
        if not _is_number(attempts) or not _is_number(yards):
            continue
        candidates.append(
            {
                "player_id": (rec.get("player") or {}).get("id"),
                "passing_attempts": float(attempts),
                "passing_yards": float(yards),
            }
        )
    if not candidates:
        return {"status": "MISSING"}
    candidates.sort(key=lambda c: c["passing_attempts"], reverse=True)
    if len(candidates) >= 2 and candidates[0]["passing_attempts"] == candidates[1]["passing_attempts"]:
        return {"status": "TIE"}
    winner = candidates[0]
    return {
        "status": "OK",
        "player_id": winner["player_id"],
        "passing_attempts": winner["passing_attempts"],
        "passing_yards": winner["passing_yards"],
    }


# --------------------------------------------------------------------------- #
# the ONE signal
# --------------------------------------------------------------------------- #
def team_qb_projection_change(tue_primary: dict, fri_primary: dict) -> float:
    return fri_primary["passing_yards"] - tue_primary["passing_yards"]


def qb_projection_shock_fri(home_change: float, away_change: float) -> float:
    """The ONLY numerical predictor. Positive => the home team's expected QB
    passing production improved relative to the away team's from Tue to Fri."""
    return home_change - away_change


def build_signal_rows(
    tue_dir: Path,
    fri_dir: Path,
    *,
    season: int,
    week: int,
    preregistration_hash: str,
) -> tuple[list[dict], list[dict]]:
    """Return (ledger_rows, omitted) for one week. The FRIDAY capture's game
    list is authoritative for the card (Friday-only scope). No realized outcome
    is read or required."""
    tue_records = load_qb_projection_records(tue_dir)
    fri_records = load_qb_projection_records(fri_dir)
    games = load_games(fri_dir)

    tue_sha = manifest_sha256(tue_dir)
    fri_sha = manifest_sha256(fri_dir)
    fri_cutoff = friday_nominal_cutoff_utc(fri_dir)
    fri_cutoff_iso = _iso_z(fri_cutoff)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[dict] = []
    omitted: list[dict] = []
    for game in games:
        home, away = game["home_team"], game["away_team"]

        # v2026.7.1 chronology gate: only games whose kickoff is STRICTLY after
        # the official Friday nominal cutoff may create a shadow row. Classify
        # BEFORE resolving QB projections.
        kickoff_class = classify_friday_kickoff(game["scheduled_kickoff_utc"], fri_cutoff)
        if kickoff_class != KICKOFF_FUTURE:
            omitted.append(
                {
                    "game_id": game["game_id"],
                    "reason": kickoff_class,
                    "kickoff_raw": game["kickoff_raw"],
                    "fri_nominal_cutoff_utc": fri_cutoff_iso,
                }
            )
            continue
        kickoff_iso = _iso_z(game["scheduled_kickoff_utc"])

        h_tue = team_primary_qb(tue_records, home)
        h_fri = team_primary_qb(fri_records, home)
        a_tue = team_primary_qb(tue_records, away)
        a_fri = team_primary_qb(fri_records, away)
        unresolved = {
            "home_tue": h_tue["status"],
            "home_fri": h_fri["status"],
            "away_tue": a_tue["status"],
            "away_fri": a_fri["status"],
        }
        if any(v != "OK" for v in unresolved.values()):
            omitted.append(
                {
                    "game_id": game["game_id"],
                    "reason": "PRIMARY_QB_UNRESOLVED",
                    "detail": unresolved,
                    "scheduled_kickoff_utc": kickoff_iso,
                }
            )
            continue

        home_change = team_qb_projection_change(h_tue, h_fri)
        away_change = team_qb_projection_change(a_tue, a_fri)
        rows.append(
            {
                "signal_version": SIGNAL_VERSION,
                "signal_name": SIGNAL_NAME,
                "season": season,
                "week": week,
                "game_id": game["game_id"],
                "home_team": home,
                "away_team": away,
                "scheduled_kickoff_utc": kickoff_iso,
                "fri_nominal_cutoff_utc": fri_cutoff_iso,
                "tue_capture_path": str(tue_dir),
                "fri_capture_path": str(fri_dir),
                "tue_manifest_sha256": tue_sha,
                "fri_manifest_sha256": fri_sha,
                "home_tue_primary_qb_id": h_tue["player_id"],
                "home_fri_primary_qb_id": h_fri["player_id"],
                "away_tue_primary_qb_id": a_tue["player_id"],
                "away_fri_primary_qb_id": a_fri["player_id"],
                "home_tue_projected_passing_yards": h_tue["passing_yards"],
                "home_fri_projected_passing_yards": h_fri["passing_yards"],
                "away_tue_projected_passing_yards": a_tue["passing_yards"],
                "away_fri_projected_passing_yards": a_fri["passing_yards"],
                "home_qb_projection_change": home_change,
                "away_qb_projection_change": away_change,
                SIGNAL_NAME: qb_projection_shock_fri(home_change, away_change),
                "signal_created_at_utc": created_at,
                "preregistration_hash": preregistration_hash,
            }
        )
    return rows, omitted


# --------------------------------------------------------------------------- #
# append-only shadow ledger (outside Git)
# --------------------------------------------------------------------------- #
def ledger_path(art_root: Path) -> Path:
    return art_root / LEDGER_SUBDIR / LEDGER_FILENAME


def read_ledger(art_root: Path) -> list[dict]:
    path = ledger_path(art_root)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_ledger_rows(art_root: Path, rows: list[dict]) -> int:
    """Append rows. A duplicate (game_id, signal_version) — already on the
    ledger or repeated within this batch — fails closed. Existing lines are
    never rewritten or reordered."""
    path = ledger_path(art_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = {(r["game_id"], r["signal_version"]) for r in read_ledger(art_root)}
    batch_keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["game_id"], row["signal_version"])
        if key in existing_keys or key in batch_keys:
            raise FailClosedError(f"duplicate ledger row for game_id={key[0]} version={key[1]}")
        batch_keys.add(key)
    if not rows:
        return 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return len(rows)


# --------------------------------------------------------------------------- #
# Friday market baseline + prospective online model (used only once outcomes
# are attached to ledger rows; no fitting during the 64-game warmup)
# --------------------------------------------------------------------------- #
def market_implied_home_margin(consensus_home_spread: float) -> float:
    return -consensus_home_spread


def margin_residual(actual_home_margin: float, consensus_home_spread: float) -> float:
    return actual_home_margin - market_implied_home_margin(consensus_home_spread)


def online_forecast(prior_settled_rows: list[dict]) -> dict:
    """Frozen estimator: StandardScaler + Ridge(alpha=100, fit_intercept=True,
    solver="svd") on ``QB_PROJECTION_SHOCK_FRI`` -> Friday margin market
    residual, trained ONLY on previously settled eligible rows. Returns
    ``{"status": "WARMUP"}`` until 64 unique settled eligible games exist."""
    settled = [r for r in prior_settled_rows if r.get("margin_residual") is not None]
    if len({r["game_id"] for r in settled}) < WARMUP_GAMES:
        return {"status": "WARMUP", "n_settled": len(settled)}
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.array([[float(r[SIGNAL_NAME])] for r in settled], dtype=float)
    y = np.array([float(r["margin_residual"]) for r in settled], dtype=float)
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100, fit_intercept=True, solver="svd"),
    )
    model.fit(x, y)
    return {"status": "READY", "model": model, "n_settled": len(settled)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="compute the signal but do not write the ledger"
    )
    args = parser.parse_args(argv)

    prereg = load_preregistration()
    prereg_hash = prereg["preregistration_sha256"]

    try:
        d_root = data_root(args.data_root)
        tue_dir = resolve_official_capture(d_root, args.season, args.week, "TUE")
        fri_dir = resolve_official_capture(d_root, args.season, args.week, "FRI")
    except FailClosedError as exc:
        _emit({"status": "FAIL_CLOSED", "detail": str(exc)})
        return 2

    if tue_dir is None or fri_dir is None:
        _emit(
            {
                "status": STATUS_NOT_READY,
                "season": args.season,
                "week": args.week,
                "tue_capture_available": tue_dir is not None,
                "fri_capture_available": fri_dir is not None,
                "preregistration_hash": prereg_hash,
            }
        )
        return 0

    try:
        rows, omitted = build_signal_rows(
            tue_dir, fri_dir, season=args.season, week=args.week, preregistration_hash=prereg_hash
        )
        written = 0
        if not args.dry_run:
            written = append_ledger_rows(artifact_root(args.artifact_root), rows)
    except FailClosedError as exc:
        _emit({"status": "FAIL_CLOSED", "detail": str(exc)})
        return 2

    _emit(
        {
            "status": STATUS_OK,
            "season": args.season,
            "week": args.week,
            "tue_capture_path": str(tue_dir),
            "fri_capture_path": str(fri_dir),
            "signal_rows_built": len(rows),
            "signal_rows_written": written,
            "dry_run": bool(args.dry_run),
            "omitted_games": omitted,
            "numerical_predictors": [SIGNAL_NAME],
            "preregistration_hash": prereg_hash,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
