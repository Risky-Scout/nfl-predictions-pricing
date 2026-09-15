"""Append-only season performance record for OPEN / MID / CLOSE snapshots.

WHAT IT IS FOR. Answering, all season, "how did the model do against the
sportsbook OPEN and the sportsbook CLOSE?" -- from evidence recorded at the
moment each snapshot was taken, not reconstructed afterwards from whatever
the market looks like later.

WHERE IT LIVES. Under the EXISTING production artifact tree,
``$NFL_MODEL_ARTIFACT_ROOT/production-2026/snapshot-performance-ledger/``,
alongside the forecast and evaluation ledgers. No new data estate, no second
production tree, and no copy of any market or model artifact -- a snapshot
row references the capture and forecast it came from by hash and stores only
the handful of numbers a performance comparison needs.

Storage is bounded by the season's shape rather than by how often anything
runs: one small JSON per ``(season, week, game_id, stage)``, so a full
18-week season of 16 games in 3 stages is under a thousand files.

TWO RECORD KINDS, NEVER ONE.

    snapshot   what was true BEFORE the game: the model's projection and the
               exact market it was priced against. Immutable, first-write-wins.
    outcome    what happened AFTER the game went provider-final, plus every
               error and result derived from it.

They are separate files on purpose. Results must never be written back into a
snapshot: a pregame record that gets edited once the answer is known is no
longer evidence of what was known pregame. Reporting JOINS them instead.

NOTHING IS INVENTED. A snapshot whose game has not finished has no outcome
file, and reporting says so rather than defaulting a score to zero or
carrying a previous week's line forward. An ATS or total result is only
computed where the market value it needs was actually recorded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from nfl_hybrid.production.snapshot_stages_2026 import SNAPSHOT_STAGES, STAGE_CLOSE, STAGE_OPEN, validate_stage

LEDGER_NAMESPACE = "snapshot-performance-ledger"

SNAPSHOT_KIND = "snapshot"
OUTCOME_KIND = "outcome"

# ATS / total result vocabulary. PUSH is a real outcome, not a missing one.
RESULT_HOME_COVER = "HOME_COVER"
RESULT_AWAY_COVER = "AWAY_COVER"
RESULT_OVER = "OVER"
RESULT_UNDER = "UNDER"
RESULT_PUSH = "PUSH"

# The identity every snapshot row is keyed by.
SNAPSHOT_IDENTITY_FIELDS = ("season", "week", "game_id", "snapshot_stage")

# What each snapshot must record. Enforced so a row cannot silently lose a
# provenance field and still look like evidence.
REQUIRED_SNAPSHOT_FIELDS = (
    # identity / timing
    "season",
    "week",
    "game_id",
    "snapshot_stage",
    "snapshot_at_utc",
    "scheduled_kickoff_utc",
    # model projection
    "model_home_margin",
    "model_total",
    # market observed at that instant
    "market_consensus_home_spread",
    "market_consensus_total",
    "market_book_quotes",
    "market_observed_at_utc",
    # provenance -- which model, which code, which calibrator, which evidence
    "model_tag",
    "model_sha",
    "production_git_commit",
    "active_calibrator_source",
    "active_calibrator_candidate_id",
    "active_calibrator_seed_sha256",
    "capture_manifest_sha256",
    "forecast_run_id",
    "forecast_prediction_hash",
)

REQUIRED_OUTCOME_FIELDS = (
    "season",
    "week",
    "game_id",
    "final_home_score",
    "final_away_score",
    "realized_margin",
    "realized_total",
    "result_attached_at_utc",
)


class PerformanceLedgerError(RuntimeError):
    """Fail-closed ledger error. A row that cannot be trusted is not written."""


class SnapshotImmutabilityViolation(PerformanceLedgerError):
    """A recorded snapshot was contradicted by a later write.

    Distinct from a plain error because it means evidence disagreed with
    itself -- an operator has to look, and no automatic reconciliation is
    attempted.
    """


def ledger_root(artifact_root: Path) -> Path:
    return Path(artifact_root) / "production-2026" / LEDGER_NAMESPACE


def _week_dir(artifact_root: Path, *, season: int, week: Any) -> Path:
    return ledger_root(artifact_root) / f"season={int(season)}" / f"week={_week_label(week)}"


def _week_label(week: Any) -> str:
    """Zero-padded when numeric, verbatim otherwise (POST weeks are labels)."""
    try:
        return f"{int(week):02d}"
    except (TypeError, ValueError):
        label = str(week).strip()
        if not label:
            raise PerformanceLedgerError("week is required and is never defaulted")
        return label


def _safe(token: str) -> str:
    return str(token).replace("/", "_").replace(":", "").replace("+", "_")


def snapshot_path(artifact_root: Path, *, season: int, week: Any, game_id: str, stage: str) -> Path:
    validate_stage(stage)
    return _week_dir(artifact_root, season=season, week=week) / f"{_safe(game_id)}__{stage}.json"


def outcome_path(artifact_root: Path, *, season: int, week: Any, game_id: str) -> Path:
    return _week_dir(artifact_root, season=season, week=week) / f"{_safe(game_id)}__OUTCOME.json"


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def content_sha256(payload: dict) -> str:
    return sha256(_canonical(payload)).hexdigest()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _require(payload: dict, required: tuple[str, ...], *, what: str) -> None:
    missing = [name for name in required if name not in payload]
    if missing:
        raise PerformanceLedgerError(f"{what} is missing required field(s) {missing}")


def _finite_or_none(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerformanceLedgerError(f"{field_name} must be a real number or null, got {value!r}")
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        raise PerformanceLedgerError(f"{field_name} must be finite, got {value!r}")
    return numeric


# ---------------------------------------------------------------------------
# recording a snapshot -- immutable, first-write-wins
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WriteResult:
    status: str  # WRITTEN | IDEMPOTENT_NOOP
    path: Path
    content_sha256: str


def record_snapshot(artifact_root: Path, snapshot: dict) -> WriteResult:
    """Record one pregame snapshot, once.

    An identical re-record is an IDEMPOTENT_NOOP so a retried automation pass
    is safe. A DIFFERENT payload for the same identity is a violation, not an
    update: the pregame truth for a stage cannot change after the fact.
    """
    _require(snapshot, REQUIRED_SNAPSHOT_FIELDS, what="snapshot")
    validate_stage(snapshot["snapshot_stage"])
    for numeric in ("model_home_margin", "model_total", "market_consensus_home_spread", "market_consensus_total"):
        _finite_or_none(snapshot.get(numeric), field_name=numeric)
    if not isinstance(snapshot["market_book_quotes"], list):
        raise PerformanceLedgerError(
            "market_book_quotes must be the list of exact per-book quotes the consensus was "
            "derived from -- a consensus with no book evidence behind it is not reviewable"
        )

    path = snapshot_path(
        artifact_root,
        season=snapshot["season"],
        week=snapshot["week"],
        game_id=snapshot["game_id"],
        stage=snapshot["snapshot_stage"],
    )
    digest = content_sha256(snapshot)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if content_sha256(existing) == digest:
            return WriteResult("IDEMPOTENT_NOOP", path, digest)
        raise SnapshotImmutabilityViolation(
            f"{snapshot['game_id']} {snapshot['snapshot_stage']}: a different snapshot is already "
            f"recorded at {path} -- a pregame snapshot is immutable and is never revised"
        )
    _write_atomic(path, snapshot)
    return WriteResult("WRITTEN", path, digest)


def read_snapshot(artifact_root: Path, *, season: int, week: Any, game_id: str, stage: str) -> dict | None:
    path = snapshot_path(artifact_root, season=season, week=week, game_id=game_id, stage=stage)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# ---------------------------------------------------------------------------
# attaching a result -- a SEPARATE record; snapshots are never touched
# ---------------------------------------------------------------------------
def ats_result(realized_margin: float | None, market_home_spread: float | None) -> str | None:
    """Home-perspective ATS result, or ``None`` where it is not defined.

    ``market_home_spread`` is sportsbook notation as captured: -3.5 means the
    home team is favoured by 3.5, so home covers by winning by MORE than 3.5.
    """
    if realized_margin is None or market_home_spread is None:
        return None
    edge = realized_margin + market_home_spread
    if edge > 0:
        return RESULT_HOME_COVER
    if edge < 0:
        return RESULT_AWAY_COVER
    return RESULT_PUSH


def total_result(realized_total: float | None, market_total: float | None) -> str | None:
    if realized_total is None or market_total is None:
        return None
    if realized_total > market_total:
        return RESULT_OVER
    if realized_total < market_total:
        return RESULT_UNDER
    return RESULT_PUSH


def _abs_error(projection: float | None, realized: float | None) -> float | None:
    if projection is None or realized is None:
        return None
    return abs(float(projection) - float(realized))


def attach_result(
    artifact_root: Path,
    *,
    season: int,
    week: Any,
    game_id: str,
    final_home_score: int,
    final_away_score: int,
    result_attached_at_utc: str,
) -> WriteResult:
    """Record the provider-final result for a game.

    Written as its own record next to -- never into -- that game's snapshots.
    Only call this once the provider says the game is final; this module does
    not infer finality and stores no in-progress score.
    """
    home = _finite_or_none(final_home_score, field_name="final_home_score")
    away = _finite_or_none(final_away_score, field_name="final_away_score")
    if home is None or away is None:
        raise PerformanceLedgerError(
            "a result requires both final scores -- a half-known result is not a result"
        )

    outcome = {
        "season": int(season),
        "week": week,
        "game_id": game_id,
        "final_home_score": home,
        "final_away_score": away,
        "realized_margin": home - away,
        "realized_total": home + away,
        "result_attached_at_utc": result_attached_at_utc,
    }
    _require(outcome, REQUIRED_OUTCOME_FIELDS, what="outcome")

    path = outcome_path(artifact_root, season=season, week=week, game_id=game_id)
    digest = content_sha256(outcome)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if content_sha256(existing) == digest:
            return WriteResult("IDEMPOTENT_NOOP", path, digest)
        raise SnapshotImmutabilityViolation(
            f"{game_id}: a different final result is already recorded at {path} -- "
            "a provider-final result is immutable"
        )
    _write_atomic(path, outcome)
    return WriteResult("WRITTEN", path, digest)


def read_outcome(artifact_root: Path, *, season: int, week: Any, game_id: str) -> dict | None:
    path = outcome_path(artifact_root, season=season, week=week, game_id=game_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# ---------------------------------------------------------------------------
# reporting -- model versus the sportsbook OPEN and CLOSE
# ---------------------------------------------------------------------------
REPORT_COLUMNS = (
    "season",
    "week",
    "game_id",
    "snapshot_stage",
    "snapshot_at_utc",
    "scheduled_kickoff_utc",
    "model_home_margin",
    "model_total",
    "market_consensus_home_spread",
    "market_consensus_total",
    "market_book_count",
    "final_home_score",
    "final_away_score",
    "realized_margin",
    "realized_total",
    "model_margin_abs_error",
    "model_total_abs_error",
    "market_margin_abs_error",
    "market_total_abs_error",
    "ats_result",
    "total_result",
    "open_close_spread_movement",
    "open_close_total_movement",
    "closing_line_value_points",
    "outcome_status",
)

OUTCOME_PENDING = "PENDING"
OUTCOME_FINAL = "FINAL"


@dataclass
class _GameRows:
    snapshots: dict[str, dict] = field(default_factory=dict)
    outcome: dict | None = None


def _discover(artifact_root: Path) -> dict[tuple, _GameRows]:
    root = ledger_root(artifact_root)
    games: dict[tuple, _GameRows] = {}
    if not root.is_dir():
        return games
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("."):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload.get("season"), str(payload.get("week")), payload.get("game_id"))
        bucket = games.setdefault(key, _GameRows())
        if path.name.endswith("__OUTCOME.json"):
            bucket.outcome = payload
        else:
            bucket.snapshots[validate_stage(payload["snapshot_stage"])] = payload
    return games


def closing_line_value_points(
    *, model_home_margin_at_open: float | None, open_spread: float | None, close_spread: float | None
) -> float | None:
    """Points gained or lost by pricing at OPEN instead of CLOSE.

    Signed in the direction the model leaned at OPEN. Backing the home team at
    -3 when the line closes -4 beats the close by a point; backing the away
    team at +3 when it closes +4 loses one. ``None`` when the model abstained,
    when either line is missing, or when the model had no side (an exact
    pick against the open).
    """
    if model_home_margin_at_open is None or open_spread is None or close_spread is None:
        return None
    model_edge = model_home_margin_at_open + open_spread
    if model_edge == 0:
        return None
    return open_spread - close_spread if model_edge > 0 else close_spread - open_spread


def build_report_rows(artifact_root: Path) -> list[dict]:
    """One row per recorded snapshot, joined to its game's result if final.

    A pending game is reported as pending with empty derived columns rather
    than omitted, so a season report never silently under-counts the games it
    is judging the model on.
    """
    discovered = sorted(
        _discover(artifact_root).items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), str(item[0][2])),
    )
    rows: list[dict] = []
    for (season, _week_key, game_id), bucket in discovered:
        outcome = bucket.outcome
        realized_margin = None if outcome is None else outcome.get("realized_margin")
        realized_total = None if outcome is None else outcome.get("realized_total")

        open_snap = bucket.snapshots.get(STAGE_OPEN)
        close_snap = bucket.snapshots.get(STAGE_CLOSE)
        open_spread = None if open_snap is None else open_snap.get("market_consensus_home_spread")
        close_spread = None if close_snap is None else close_snap.get("market_consensus_home_spread")
        open_total = None if open_snap is None else open_snap.get("market_consensus_total")
        close_total = None if close_snap is None else close_snap.get("market_consensus_total")

        movement_spread = (
            None if open_spread is None or close_spread is None else close_spread - open_spread
        )
        movement_total = None if open_total is None or close_total is None else close_total - open_total
        clv = closing_line_value_points(
            model_home_margin_at_open=None if open_snap is None else open_snap.get("model_home_margin"),
            open_spread=open_spread,
            close_spread=close_spread,
        )

        for stage in SNAPSHOT_STAGES:
            snapshot = bucket.snapshots.get(stage)
            if snapshot is None:
                continue
            market_spread = snapshot.get("market_consensus_home_spread")
            market_total = snapshot.get("market_consensus_total")
            rows.append(
                {
                    "season": season,
                    "week": snapshot.get("week"),
                    "game_id": game_id,
                    "snapshot_stage": stage,
                    "snapshot_at_utc": snapshot.get("snapshot_at_utc"),
                    "scheduled_kickoff_utc": snapshot.get("scheduled_kickoff_utc"),
                    "model_home_margin": snapshot.get("model_home_margin"),
                    "model_total": snapshot.get("model_total"),
                    "market_consensus_home_spread": market_spread,
                    "market_consensus_total": market_total,
                    "market_book_count": len(snapshot.get("market_book_quotes") or []),
                    "final_home_score": None if outcome is None else outcome.get("final_home_score"),
                    "final_away_score": None if outcome is None else outcome.get("final_away_score"),
                    "realized_margin": realized_margin,
                    "realized_total": realized_total,
                    "model_margin_abs_error": _abs_error(snapshot.get("model_home_margin"), realized_margin),
                    "model_total_abs_error": _abs_error(snapshot.get("model_total"), realized_total),
                    # The market's own error, in the same home-margin sign
                    # convention as the model's, so the two are comparable:
                    # a home spread of -3 is a predicted home margin of +3.
                    "market_margin_abs_error": _abs_error(
                        None if market_spread is None else -market_spread, realized_margin
                    ),
                    "market_total_abs_error": _abs_error(market_total, realized_total),
                    "ats_result": ats_result(realized_margin, market_spread),
                    "total_result": total_result(realized_total, market_total),
                    "open_close_spread_movement": movement_spread,
                    "open_close_total_movement": movement_total,
                    "closing_line_value_points": clv,
                    "outcome_status": OUTCOME_PENDING if outcome is None else OUTCOME_FINAL,
                }
            )
    for row in rows:
        assert tuple(row.keys()) == REPORT_COLUMNS
    return rows


def report_json(artifact_root: Path) -> dict:
    rows = build_report_rows(artifact_root)
    final_rows = [r for r in rows if r["outcome_status"] == OUTCOME_FINAL]

    def _mean(values: list[float]) -> float | None:
        usable = [v for v in values if v is not None]
        return None if not usable else sum(usable) / len(usable)

    by_stage = {}
    for stage in SNAPSHOT_STAGES:
        staged = [r for r in final_rows if r["snapshot_stage"] == stage]
        by_stage[stage] = {
            "graded_snapshots": len(staged),
            "model_margin_mae": _mean([r["model_margin_abs_error"] for r in staged]),
            "market_margin_mae": _mean([r["market_margin_abs_error"] for r in staged]),
            "model_total_mae": _mean([r["model_total_abs_error"] for r in staged]),
            "market_total_mae": _mean([r["market_total_abs_error"] for r in staged]),
        }
    return {
        "schema_version": "nfl-snapshot-performance-report-v1",
        "snapshot_rows": len(rows),
        "graded_rows": len(final_rows),
        "pending_rows": len(rows) - len(final_rows),
        "by_stage": by_stage,
        "rows": rows,
    }


def report_csv(artifact_root: Path) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(REPORT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in build_report_rows(artifact_root):
        writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    return buffer.getvalue()
