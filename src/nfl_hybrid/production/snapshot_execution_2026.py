"""Executing OPEN / MID / CLOSE snapshots -- the production call site.

WHAT RUNS HERE. For each due stage instant: refresh what is known, refit the
EXISTING chronological Ridge at that instant, reconstruct the market at that
instant, write the forecast of record, and append the performance row. The
stage contract in :mod:`snapshot_stages_2026` says *when*; this module makes
it *happen*.

NO SECOND PIPELINE. Every scientific step is the certified one, reached
through :func:`run_2026.run_horizon_batch` with its ``stage_cutoffs``
override. The six Elo features, the Ridge alpha, the paired margin/total
fits, the certified hashes, the active-calibrator resolution, the capture
validation and the forecast-ledger immutability rule are all exactly what a
TUE/FRI card run uses. The only difference is the as-of instant the fit is
performed at -- which is the definition of a snapshot stage.

POINT-IN-TIME IS INHERITED, NOT REIMPLEMENTED. The certified fitter already
trains on precisely the rows with ``result_available_at_utc`` strictly before
the batch cutoff that carry observed outcomes. So a provider-final Thursday
game becomes eligible for a later Sunday CLOSE automatically, and an
unresolved or in-progress game never enters training, because that is already
the rule -- this module does not get a vote.

A STAGE BATCH IS ONE INSTANT. Games that share a kickoff share a CLOSE, so
the 1:00pm ET group is one batch. Games with different kickoffs are different
batches. Grouping is explicit, never "near enough".

DUE MEANS PAST. A stage instant in the future is never executed: pricing a
market that has not been observed yet would be fabrication. A stage instant
at or after its game's kickoff is not a pregame snapshot and is refused.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.evaluation import raw_market_reconstruction as rmr
from nfl_hybrid.production import run_2026 as prod
from nfl_hybrid.production import snapshot_performance_ledger_2026 as pl
from nfl_hybrid.production import snapshot_stages_2026 as st

# Reasons a stage is not executed now. All are normal, not failures.
SKIP_NOT_DUE = "NOT_DUE"
SKIP_POST_KICKOFF = "POST_KICKOFF"
SKIP_UNRESOLVED_STAGE = "UNRESOLVED_STAGE"
SKIP_ALREADY_RECORDED = "ALREADY_RECORDED"
SKIP_NO_ARCHIVED_EVIDENCE = "NO_ARCHIVED_EVIDENCE_AT_OR_BEFORE_CUTOFF"


class SnapshotExecutionError(RuntimeError):
    """Fail-closed execution error."""


# Where the immutable STAGE observation archive lives, relative to
# NFL_MODEL_DATA_ROOT. Same tree capture_bdl_2026_asof.py writes to; this
# module only ever READS it.
STAGE_ARCHIVE_NAMESPACE = "live-observation-log/balldontlie-2026"


@dataclass(frozen=True)
class ArchivedCapture:
    """One archived STAGE observation: where it is, and when it observed."""

    manifest_path: Path
    observed_at_utc: pd.Timestamp


def archived_stage_captures(data_root: Path, *, season: int, week) -> list[ArchivedCapture]:
    """Every archived STAGE capture for one card, oldest observation first.

    Selection reads only each manifest's ``nominal_cutoff_utc`` -- the instant
    it observed the board -- so listing the archive stays cheap even with
    hundreds of captures. A manifest that cannot state when it observed is
    skipped here rather than raising: it simply cannot be placed on the
    timeline, and the callers below verify every capture they actually use.
    """
    week_label = f"{int(week):02d}" if str(week).isdigit() else str(week)
    pattern = (
        Path(STAGE_ARCHIVE_NAMESPACE)
        / f"season={int(season)}"
        / f"week={week_label}"
        / f"horizon={st.STAGE_CAPTURE_HORIZON}"
        / "capture=*"
        / "manifest.json"
    )
    found: list[ArchivedCapture] = []
    for path in sorted(Path(data_root).glob(str(pattern))):
        try:
            observed = pd.Timestamp(json.loads(path.read_text())["nominal_cutoff_utc"])
        except Exception:
            continue
        if observed is pd.NaT or pd.isna(observed):
            continue
        observed = observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
        found.append(ArchivedCapture(manifest_path=path, observed_at_utc=observed))
    return sorted(found, key=lambda c: (c.observed_at_utc, str(c.manifest_path)))


def newest_capture_at_or_before(
    captures: list[ArchivedCapture], cutoff
) -> ArchivedCapture | None:
    """The freshest observation that is still not after ``cutoff``.

    This is what makes a HISTORICAL stage reachable. A sweep running today
    holds a capture observed today, which ``observation_not_after_utc``
    correctly refuses for a cutoff that has already passed -- so MID and CLOSE
    were unreachable the moment their instant went by. Reaching back to the
    last capture taken before the cutoff prices the stage from evidence that
    genuinely predates it, with the no-lookahead rule fully intact rather than
    relaxed.
    """
    limit = prod._as_utc(cutoff)
    eligible = [c for c in captures if c.observed_at_utc <= limit]
    return max(eligible, key=lambda c: c.observed_at_utc) if eligible else None


def accumulated_stage_quotes(
    captures: list[ArchivedCapture], *, artifact_root_path: Path | None = None
) -> tuple[pd.DataFrame | None, dict]:
    """Every archived observation's quotes, concatenated oldest first.

    OPEN means "the FIRST valid market observed", which is a fact about the
    whole season's evidence, not about whichever capture a sweep happens to be
    holding. Resolving it from one capture made OPEN move forward every sweep
    and collide with the immutable ledger; resolving it from the accumulated
    archive makes it deterministic -- the same archive yields the same answer
    on every future sweep.

    A capture that fails validation is skipped and counted, not fatal: an
    archive spanning months should not be held hostage by one bad file, and
    the same row-level reasoning already governs a single capture's odds rows.
    """
    frames: list[pd.DataFrame] = []
    rejected: list[dict] = []
    for capture in captures:
        try:
            validated = bridge.validate_capture_manifest(
                capture.manifest_path,
                expected_horizon=bridge.STAGE_HORIZON,
                allowed_horizons=bridge.STAGE_HORIZONS,
            )
            frames.append(bridge.build_bookmaker_quotes(validated))
        except Exception as exc:
            rejected.append(
                {"manifest_path": str(capture.manifest_path), "reason": f"{type(exc).__name__}: {exc}"}
            )
    provenance = {
        "archived_capture_count": len(captures),
        "usable_capture_count": len(frames),
        "rejected_capture_count": len(rejected),
        "rejected_captures": rejected[:ARCHIVE_REJECTION_EXAMPLE_LIMIT],
    }
    if not frames:
        return None, provenance

    accumulated = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=list(bridge.BOOKMAKER_QUOTE_COLUMNS)
    )
    accumulated["returned_snapshot_utc"] = pd.to_datetime(
        accumulated["returned_snapshot_utc"], utc=True
    )
    accumulated = accumulated.sort_values(
        ["returned_snapshot_utc", "game_id", "bookmaker_key"], kind="stable"
    ).reset_index(drop=True)
    provenance["accumulated_quote_rows"] = int(len(accumulated))
    return accumulated, provenance


ARCHIVE_REJECTION_EXAMPLE_LIMIT = 5


@dataclass(frozen=True)
class StageBatch:
    """One stage instant and the games whose stage it is."""

    stage: str
    cutoff_utc: pd.Timestamp
    game_ids: tuple[str, ...]


@dataclass(frozen=True)
class StageExecution:
    """What one batch did."""

    stage: str
    cutoff_utc: pd.Timestamp
    game_ids: tuple[str, ...]
    run_status: str
    run_id: str
    ledger_writes: tuple[str, ...]
    skipped: str | None = None


# ---------------------------------------------------------------------------
# planning -- which stage instants are due
# ---------------------------------------------------------------------------
def plan_stage_batches(
    *,
    stage: str,
    card: pd.DataFrame,
    as_of_utc,
    open_observations: dict[str, pd.Timestamp] | None = None,
    operational_root: Path | None = None,
    season: int | None = None,
    week=None,
) -> tuple[list[StageBatch], dict[str, str]]:
    """Group the card's games into the due batches for ``stage``.

    ``card`` needs ``game_id`` and ``scheduled_kickoff_utc``. Returns the due
    batches plus, for every game not in one, the reason -- so an automation
    pass can log why a game was left alone instead of silently omitting it.

    When ``operational_root`` (with ``season``/``week``) is supplied, a stage
    already recorded in the performance ledger is treated as COMPLETE and
    skipped. A pregame snapshot is immutable, so recomputing one every sweep
    can only either waste the work or -- as it did in production -- collide
    with the record it wrote earlier. Skipping keeps
    ``SnapshotImmutabilityViolation`` for what it is meant to catch: a genuine
    contradiction, not the routine act of sweeping twice.
    """
    st.validate_stage(stage)
    as_of = prod._as_utc(as_of_utc)
    observations = open_observations or {}
    earliest = st.card_earliest_kickoff_utc(card["scheduled_kickoff_utc"])
    check_recorded = operational_root is not None and season is not None and week is not None

    by_instant: dict[pd.Timestamp, list[str]] = {}
    skipped: dict[str, str] = {}
    for row in card.itertuples(index=False):
        game_id = str(row.game_id)
        if check_recorded and pl.read_snapshot(
            operational_root, season=season, week=week, game_id=game_id, stage=stage
        ) is not None:
            skipped[game_id] = SKIP_ALREADY_RECORDED
            continue
        schedule = st.resolve_game_snapshots(
            game_id=game_id,
            scheduled_kickoff_utc=row.scheduled_kickoff_utc,
            card_earliest_kickoff_utc=earliest,
            open_observed_at_utc=observations.get(game_id),
        )
        cutoff = schedule.cutoff_for(stage)
        if cutoff is None:
            skipped[game_id] = SKIP_UNRESOLVED_STAGE
            continue
        if not st.is_strictly_pregame(cutoff, row.scheduled_kickoff_utc):
            skipped[game_id] = SKIP_POST_KICKOFF
            continue
        if cutoff > as_of:
            skipped[game_id] = SKIP_NOT_DUE
            continue
        by_instant.setdefault(cutoff, []).append(game_id)

    batches = [
        StageBatch(stage=stage, cutoff_utc=instant, game_ids=tuple(sorted(ids)))
        for instant, ids in sorted(by_instant.items())
    ]
    return batches, skipped


def discover_open_instant(quotes: pd.DataFrame, *, game_id: str, scheduled_kickoff_utc) -> pd.Timestamp | None:
    """The first instant at which this game's board is genuinely priceable.

    "First valid market" means the earliest observation at which BOTH
    certified markets can actually be reconstructed -- a spread alone is not a
    board this system can price. Validity uses the existing eligible-book
    floor rather than a new one, so OPEN cannot be earlier than the instant
    the certified reconstruction would first have produced a consensus.

    Returns ``None`` when the board was never priceable pregame; that game has
    no OPEN and the caller records it as missing rather than back-dating one.
    """
    kickoff = prod._as_utc(scheduled_kickoff_utc)
    per_market: list[pd.Timestamp] = []
    for raw_market in (rmr.MARKET_SPREADS, rmr.MARKET_TOTALS):
        coherent = rmr.build_coherent_book_observations(quotes, raw_market)
        if coherent.empty:
            return None
        mine = coherent[coherent["game_id"].astype(str) == str(game_id)].copy()
        if mine.empty:
            return None
        mine["returned_snapshot_utc"] = pd.to_datetime(mine["returned_snapshot_utc"], utc=True)
        mine = mine[mine["returned_snapshot_utc"] < kickoff].sort_values("returned_snapshot_utc")
        first_valid = None
        for instant in mine["returned_snapshot_utc"].drop_duplicates():
            seen = mine[mine["returned_snapshot_utc"] <= instant]["bookmaker_key"].nunique()
            if seen >= rmr.MINIMUM_FRESH_COHERENT_BOOKS:
                first_valid = instant
                break
        if first_valid is None:
            return None
        per_market.append(first_valid)
    # Both markets must be priceable, so the board opens at the later of the
    # two -- never the earlier, which would claim a board that could not yet
    # be fully priced.
    return st.validate_open_observation(max(per_market), scheduled_kickoff_utc=kickoff)


# ---------------------------------------------------------------------------
# the exact per-book quotes a consensus was derived from
# ---------------------------------------------------------------------------
def selected_book_quotes(
    quotes: pd.DataFrame, *, game_id: str, market_entry: dict, raw_market: str
) -> list[dict]:
    """The individual book observations behind one consensus line.

    Read-only, and derived from the consensus's OWN record of which books and
    which returned-snapshot instants it selected, so these are provably the
    quotes that produced the published number rather than a re-selection that
    might differ.
    """
    market = market_entry.get("market") or {}
    books = set(market.get("bookmaker_keys") or ())
    instants = {pd.Timestamp(t) for t in (market.get("selected_returned_snapshot_timestamps") or ())}
    if not books or not instants:
        return []

    coherent = rmr.build_coherent_book_observations(quotes, raw_market)
    if coherent.empty:
        return []
    frame = coherent[coherent["game_id"].astype(str) == str(game_id)].copy()
    frame["returned_snapshot_utc"] = pd.to_datetime(frame["returned_snapshot_utc"], utc=True)
    frame = frame[frame["bookmaker_key"].isin(books) & frame["returned_snapshot_utc"].isin(instants)]
    return [
        {
            "bookmaker_key": str(row.bookmaker_key),
            "market": raw_market,
            "line": float(row.line),
            "home_or_over_price_decimal": float(row.home_or_over_price_decimal),
            "away_or_under_price_decimal": float(row.away_or_under_price_decimal),
            "returned_snapshot_utc": pd.Timestamp(row.returned_snapshot_utc).isoformat().replace("+00:00", "Z"),
            "market_last_update": str(row.market_last_update),
        }
        for row in frame.sort_values(["bookmaker_key", "returned_snapshot_utc"]).itertuples(index=False)
    ]


# ---------------------------------------------------------------------------
# feeding the performance ledger from the immutable forecast of record
# ---------------------------------------------------------------------------
def _consensus_line(prediction: dict, market_name: str) -> float | None:
    entry = (prediction.get("markets") or {}).get(market_name) or {}
    market = entry.get("market") or {}
    value = market.get("consensus_line")
    return None if value is None else float(value)


def _market_observed_at(prediction: dict) -> str | None:
    instants: list[pd.Timestamp] = []
    for market_name in ("ATS", "TOTAL"):
        entry = (prediction.get("markets") or {}).get(market_name) or {}
        market = entry.get("market") or {}
        instants += [pd.Timestamp(t) for t in (market.get("selected_returned_snapshot_timestamps") or ())]
    return None if not instants else max(instants).isoformat().replace("+00:00", "Z")


def performance_row_from_forecast(
    record: dict, *, stage: str, manifest: dict, book_quotes: list[dict]
) -> dict:
    """One performance-ledger row, built FROM the forecast of record.

    Deriving it from the immutable forecast rather than from loose run-time
    variables is what guarantees the performance record and the published
    projection can never disagree about what the model said.

    Run and commit identity come from the RECORD, not from this invocation's
    manifest. The forecast ledger is first-write-wins, so a replay's manifest
    carries a fresh ``run_id`` over rows it did not write; taking it from
    there would make the same snapshot serialize differently every pass and
    collide with its own immutability check. The row names the run that
    actually produced the forecast -- which is also the more useful answer.
    """
    prediction = record["prediction"]
    inner = prediction.get("prediction") or {}
    return {
        "season": int(prediction["season"]),
        "week": prediction.get("week"),
        "game_id": str(record["game_id"]),
        "snapshot_stage": st.validate_stage(stage),
        "snapshot_at_utc": str(record["target_cutoff_utc"]),
        "scheduled_kickoff_utc": prediction.get("scheduled_kickoff_utc"),
        "model_home_margin": inner.get("predicted_margin"),
        "model_total": inner.get("predicted_total"),
        "market_consensus_home_spread": _consensus_line(prediction, "ATS"),
        "market_consensus_total": _consensus_line(prediction, "TOTAL"),
        "market_book_quotes": book_quotes,
        "market_observed_at_utc": _market_observed_at(prediction),
        "model_tag": record.get("certified_baseline_tag"),
        "model_sha": record.get("certified_baseline_sha"),
        "production_git_commit": record.get("git_commit"),
        "active_calibrator_source": prediction.get("active_calibrator_source"),
        "active_calibrator_candidate_id": prediction.get("active_calibrator_candidate_id"),
        "active_calibrator_seed_sha256": prediction.get("active_calibrator_seed_sha256"),
        "capture_manifest_sha256": (manifest.get("input_hashes") or {}).get("live_market_capture_sha256"),
        "forecast_run_id": record.get("run_id"),
        "forecast_prediction_hash": record.get("prediction_hash"),
    }


# ---------------------------------------------------------------------------
# executing one batch
# ---------------------------------------------------------------------------
def execute_stage_batch(
    batch: StageBatch,
    *,
    horizon: str,
    operational_root: Path,
    as_of_utc,
    market_capture_manifest: Path | str | None = None,
    quotes: pd.DataFrame | None = None,
    **batch_kwargs,
) -> StageExecution:
    """Run one stage instant end to end.

    The certified batch runner does the science and writes the immutable
    forecast of record; this function then appends the performance rows for
    exactly the forecasts it wrote. A non-SUCCESS batch appends nothing --
    there is no half-recorded snapshot.
    """
    manifest = prod.run_horizon_batch(
        horizon=horizon,
        as_of_utc=as_of_utc,
        force=True,
        operational_root=operational_root,
        market_capture_manifest=market_capture_manifest,
        snapshot_stage=batch.stage,
        stage_cutoffs={game_id: batch.cutoff_utc for game_id in batch.game_ids},
        **batch_kwargs,
    )
    if manifest["status"] != "SUCCESS":
        return StageExecution(
            stage=batch.stage,
            cutoff_utc=batch.cutoff_utc,
            game_ids=batch.game_ids,
            run_status=manifest["status"],
            run_id=manifest["run_id"],
            ledger_writes=(),
        )

    ledger_root = Path(operational_root) / "production-2026" / "forecast-ledger"
    writes: list[str] = []
    for game_id in batch.game_ids:
        record = prod.read_forecast(ledger_root, game_id, horizon, str(batch.cutoff_utc))
        if record is None:
            raise SnapshotExecutionError(
                f"{batch.stage} batch reported SUCCESS but no forecast of record exists for {game_id} "
                f"at {batch.cutoff_utc} -- refusing to record performance for a forecast that is not there"
            )
        prediction = record["prediction"]
        book_quotes: list[dict] = []
        if quotes is not None:
            for market_name, raw_market in (("ATS", rmr.MARKET_SPREADS), ("TOTAL", rmr.MARKET_TOTALS)):
                entry = (prediction.get("markets") or {}).get(market_name) or {}
                book_quotes += selected_book_quotes(
                    quotes, game_id=game_id, market_entry=entry, raw_market=raw_market
                )
        result = pl.record_snapshot(
            operational_root,
            performance_row_from_forecast(
                record, stage=batch.stage, manifest=manifest, book_quotes=book_quotes
            ),
        )
        writes.append(f"{game_id}:{result.status}")

    return StageExecution(
        stage=batch.stage,
        cutoff_utc=batch.cutoff_utc,
        game_ids=batch.game_ids,
        run_status=manifest["status"],
        run_id=manifest["run_id"],
        ledger_writes=tuple(writes),
    )


def run_due_stage_snapshots(
    *,
    stage: str,
    horizon: str,
    card: pd.DataFrame,
    as_of_utc,
    operational_root: Path,
    open_observations: dict[str, pd.Timestamp] | None = None,
    market_capture_manifest: Path | str | None = None,
    quotes: pd.DataFrame | None = None,
    archived_captures: list[ArchivedCapture] | None = None,
    season: int | None = None,
    week=None,
    **batch_kwargs,
) -> dict:
    """Every due batch for one stage. The entry point automation calls.

    With ``archived_captures`` supplied, each batch is priced from the newest
    archived observation at or before ITS OWN cutoff, rather than from one
    capture shared by every stage. That is what makes a historical MID or
    CLOSE reachable at all. Without it, behaviour is exactly as before.
    """
    batches, skipped = plan_stage_batches(
        stage=stage,
        card=card,
        as_of_utc=as_of_utc,
        open_observations=open_observations,
        operational_root=operational_root,
        season=season,
        week=week,
    )

    executions: list[StageExecution] = []
    for batch in batches:
        batch_manifest, batch_quotes = market_capture_manifest, quotes
        if archived_captures is not None:
            chosen = newest_capture_at_or_before(archived_captures, batch.cutoff_utc)
            if chosen is None:
                # No observation predates this stage instant, so there is
                # nothing that could price it without looking ahead.
                executions.append(
                    StageExecution(
                        stage=batch.stage,
                        cutoff_utc=batch.cutoff_utc,
                        game_ids=batch.game_ids,
                        run_status="MARKET_SOURCE_UNAVAILABLE",
                        run_id="",
                        ledger_writes=(),
                        skipped=SKIP_NO_ARCHIVED_EVIDENCE,
                    )
                )
                continue
            batch_manifest = chosen.manifest_path
            batch_quotes, _ = accumulated_stage_quotes([chosen])
        executions.append(
            execute_stage_batch(
                batch,
                horizon=horizon,
                operational_root=operational_root,
                as_of_utc=as_of_utc,
                market_capture_manifest=batch_manifest,
                quotes=batch_quotes,
                **batch_kwargs,
            )
        )
    return {
        "stage": stage,
        "as_of_utc": str(prod._as_utc(as_of_utc)),
        "due_batches": len(batches),
        "executions": [
            {
                "cutoff_utc": str(execution.cutoff_utc),
                "game_ids": list(execution.game_ids),
                "run_status": execution.run_status,
                "run_id": execution.run_id,
                "ledger_writes": list(execution.ledger_writes),
            }
            for execution in executions
        ],
        "skipped": skipped,
        "status": (
            "OK"
            if all(e.run_status == "SUCCESS" for e in executions)
            else "FAIL_CLOSED"
        ),
    }


def attach_final_results(
    *, operational_root: Path, games: pd.DataFrame, result_attached_at_utc: str
) -> dict:
    """Attach provider-final results to every snapshot game that has one.

    Only genuinely final games: a row without both scores is skipped, never
    defaulted. Snapshots are not touched -- the result becomes its own record.
    """
    attached, pending = [], []
    for row in games.itertuples(index=False):
        home, away = getattr(row, "home_score", None), getattr(row, "away_score", None)
        if home is None or away is None or pd.isna(home) or pd.isna(away):
            pending.append(str(row.game_id))
            continue
        result = pl.attach_result(
            operational_root,
            season=int(row.season),
            week=row.week,
            game_id=str(row.game_id),
            final_home_score=float(home),
            final_away_score=float(away),
            result_attached_at_utc=result_attached_at_utc,
        )
        attached.append(f"{row.game_id}:{result.status}")
    return {"attached": attached, "pending": pending}


def write_season_reports(*, operational_root: Path, output_dir: Path) -> dict:
    """Season CSV + JSON performance reporting from the real ledger rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = pl.report_json(operational_root)
    json_path = output_dir / "snapshot_performance_report.json"
    csv_path = output_dir / "snapshot_performance_report.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    csv_path.write_text(pl.report_csv(operational_root), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "snapshot_rows": report["snapshot_rows"],
        "graded_rows": report["graded_rows"],
        "pending_rows": report["pending_rows"],
    }
