"""Execute whatever OPEN / MID / CLOSE snapshots have become due.

THE ENTRY POINT AUTOMATION CALLS. A sweep, not a timetable: it asks the stage
contract which instants have passed for the current week's games, executes
those, and no-ops on everything else. That is why one polling schedule can
drive all three stages -- CLOSE is per game (kickoff minus 60 minutes), so
there is no single clock time to schedule it at in the first place.

THE CRON IS NEVER THE GATE, exactly as it is never the gate for the certified
TUE/FRI pass. Which stage instants are due is decided by each game's own
kickoff and by ``America/New_York`` card arithmetic borrowed from
``horizon_elo``, so a sweep that fires at an uninteresting moment resolves
nothing due and exits 0. DST needs no special case because no hour is ever
hard-coded.

EXECUTION IS AFTER THE INSTANT, AND THE INSTANT IS EXACT. A sweep cannot run
*at* kickoff minus 60 minutes to the second, and it must not: the market at
that instant has to have been observed before it can be priced. So the
snapshot's cutoff is exactly kickoff minus 60 minutes -- that is the instant
the model is fitted at and the market is reconstructed at -- while the
execution happens on the next sweep after it. A future instant is never
executed.

HISTORICAL REPAIR. By default the sweep asks which card is CURRENT, which is
right for unattended operation and wrong for repair: once the card advances a
past week becomes unreachable, so any gap in it can never be filled. Naming a
season/week explicitly selects that card instead. Only the selection changes --
every stage rule, cutoff, market gate and ledger guard downstream is the same
code the scheduled sweep runs.

This entry point never publishes and never writes ``latest.json`` in either
mode; publication is a separate script the orchestrator calls afterwards. So
repairing a past week structurally cannot disturb the current public feed.

Usage:
  python scripts/run_2026_stage_snapshots.py --stage ALL
  python scripts/run_2026_stage_snapshots.py --stage CLOSE --as-of 2026-09-20T16:05:00Z

  # Historical repair -- plan first, then write.
  python scripts/run_2026_stage_snapshots.py --stage ALL --season 2026 --week 2 --dry-run
  python scripts/run_2026_stage_snapshots.py --stage ALL --season 2026 --week 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.data import bdl_market_bridge as bridge  # noqa: E402
from nfl_hybrid.features import horizon_elo as he  # noqa: E402
from nfl_hybrid.production import run_2026 as prod  # noqa: E402
from nfl_hybrid.production import snapshot_execution_2026 as ex  # noqa: E402
from nfl_hybrid.production import snapshot_stages_2026 as st  # noqa: E402

STAGE_ALL = "ALL"


def resolve_current_card(games: pd.DataFrame, as_of_utc) -> tuple[pd.DataFrame, dict]:
    """The current week's card, via the certified card arithmetic.

    The card is identified by its own TUE cutoff -- the same value
    ``current_or_recent_cutoff`` gives the certified pass -- so the snapshot
    sweep and the certified card can never disagree about which week it is.
    """
    as_of = prod._as_utc(as_of_utc)
    ledger = he.build_horizon_membership_ledger(games)
    tue_cutoff = prod.current_or_recent_cutoff(as_of, "TUE")
    card = ledger[ledger["tue_cutoff_utc"] == tue_cutoff]
    if card.empty:
        return pd.DataFrame(columns=["game_id", "scheduled_kickoff_utc"]), {
            "status": "NO_CURRENT_CARD",
            "tue_cutoff_utc": str(tue_cutoff),
        }
    seasons = sorted({int(s) for s in card["season"]})
    weeks = sorted({str(w) for w in card["week"]})
    kickoffs = games[games["game_id"].astype(str).isin(set(card["game_id"].astype(str)))][
        ["game_id", "scheduled_kickoff_utc"]
    ].reset_index(drop=True)
    return kickoffs, {
        "status": "OK",
        "tue_cutoff_utc": str(tue_cutoff),
        "season": seasons[0] if len(seasons) == 1 else seasons,
        "week": weeks[0] if len(weeks) == 1 else weeks,
        "card_game_count": int(len(kickoffs)),
    }


def resolve_historical_card(games: pd.DataFrame, *, season: int, week) -> tuple[pd.DataFrame, dict]:
    """One EXPLICITLY named card, for repairing a week that has gone by.

    The sweep normally asks "which card is current?", which is right for
    unattended operation and wrong for repair: once the card advances, a
    past week becomes unreachable and any gap in it can never be filled.
    Week 2's OPEN rows were quarantined for correction while Week 2 was
    current, and the card moved to Week 3 before the corrected rows were
    written -- so they would have stayed missing forever.

    The card still comes from the SAME certified membership ledger the
    current-card path uses; the only difference is that the operator names
    the week instead of the clock choosing it. Nothing about stage cutoffs,
    market evidence or model behaviour changes as a result.
    """
    ledger = he.build_horizon_membership_ledger(games)
    match = ledger[
        (ledger["season"].astype(int) == int(season))
        & (ledger["week"].astype(str) == str(week))
    ]
    if match.empty:
        return pd.DataFrame(columns=["game_id", "scheduled_kickoff_utc"]), {
            "status": "NO_SUCH_CARD",
            "season": int(season),
            "week": str(week),
        }
    cutoffs = sorted({str(c) for c in match["tue_cutoff_utc"]})
    kickoffs = games[games["game_id"].astype(str).isin(set(match["game_id"].astype(str)))][
        ["game_id", "scheduled_kickoff_utc"]
    ].reset_index(drop=True)
    return kickoffs, {
        "status": "OK",
        "mode": "HISTORICAL_REPAIR",
        "tue_cutoff_utc": cutoffs[0] if len(cutoffs) == 1 else cutoffs,
        "season": int(season),
        "week": str(week),
        "card_game_count": int(len(kickoffs)),
    }


def load_stage_quotes(
    market_capture_manifest, *, not_after_utc, artifact_root_path: Path | None = None
) -> pd.DataFrame | None:
    """The canonical per-book quotes from a STAGE capture, or ``None``.

    STAGE is deliberately NOT in ``bridge.PRODUCTION_HORIZONS``, so loading a
    stage capture through the default market-source contract rejects it as a
    non-production horizon. That is what made OPEN discovery hand ``None`` to
    the discoverer and report every game UNRESOLVED_STAGE even though the
    capture was COMPLETE with 125 odds observations.

    The stage expectations are stated EXPLICITLY here, exactly as the stage
    batch path already states them, rather than by widening the default. A
    certified TUE/FRI caller is unaffected, and a SMOKE or corrupt capture is
    still refused.
    """
    evidence = prod.evaluate_live_market_source(
        market_capture_manifest,
        expected_horizon=bridge.STAGE_HORIZON,
        allowed_horizons=bridge.STAGE_HORIZONS,
        # Bounds the observation to this sweep's own instant, which matters
        # for a replay with an explicit past --as-of: a capture taken later
        # than the instant being replayed must not inform it.
        observation_not_after_utc=not_after_utc,
        artifact_root_path=artifact_root_path,
    )
    if not evidence["registered"] or evidence["source"] is None:
        return None
    return evidence["source"].quotes


def discover_open_observations(
    card: pd.DataFrame, *, quotes: pd.DataFrame | None
) -> dict[str, pd.Timestamp]:
    """Each game's first priceable board instant, or nothing recorded for it.

    Without market evidence no OPEN is claimed. A game whose board has not
    been seen simply has no OPEN, which leaves its MID unresolved too if it is
    the Thursday game -- missing evidence stays missing rather than being
    replaced by an assumed opening instant.
    """
    if quotes is None or quotes.empty:
        return {}
    observations: dict[str, pd.Timestamp] = {}
    for row in card.itertuples(index=False):
        instant = ex.discover_open_instant(
            quotes, game_id=str(row.game_id), scheduled_kickoff_utc=row.scheduled_kickoff_utc
        )
        if instant is not None:
            observations[str(row.game_id)] = instant
    return observations


def run(
    *,
    stage: str,
    as_of: str | None,
    operational_root: Path,
    market_capture_manifest: Path | None,
    data_root: Path | None = None,
    games: pd.DataFrame | None = None,
    historical_season: int | None = None,
    historical_week=None,
    dry_run: bool = False,
    **batch_kwargs,
) -> dict:
    """One sweep. Current-card by default; an explicitly named card on repair.

    ``historical_season``/``historical_week`` switch ONLY which card is
    selected. Every stage rule downstream is the same code the scheduled
    sweep runs, so a repaired week is resolved exactly as it would have been
    at the time.

    This entry point never publishes and never writes ``latest.json`` in
    either mode -- publication is a separate script the sweep orchestrator
    calls afterwards -- so repairing a past week structurally cannot disturb
    the current public feed.
    """
    as_of_utc = prod._as_utc(as_of) if as_of else prod.utc_now()
    historical = historical_season is not None or historical_week is not None
    if historical and (historical_season is None or historical_week is None):
        raise SystemExit("historical repair requires BOTH --season and --week")
    if games is None:
        games, _ = prod.load_games_population_with_provenance()
        games = prod.filter_reg_post(games)

    if historical:
        card, card_info = resolve_historical_card(
            games, season=historical_season, week=historical_week
        )
    else:
        card, card_info = resolve_current_card(games, as_of_utc)
    if card_info["status"] != "OK":
        return {
            "status": card_info["status"],
            "as_of_utc": str(as_of_utc),
            "card": card_info,
            "stages": [],
        }

    # OPEN is a fact about the season's ACCUMULATED evidence, and a historical
    # MID/CLOSE can only be priced from an observation that predates it. Both
    # need the archive, not just the capture this sweep happens to hold.
    # Resolved only once a card exists, and tolerant of an unset root so a
    # hermetic caller can run with no observation archive at all.
    resolved_data_root = data_root if data_root is not None else os.environ.get("NFL_MODEL_DATA_ROOT")
    archived = (
        []
        if resolved_data_root is None
        else ex.archived_stage_captures(
            Path(resolved_data_root), season=card_info["season"], week=card_info["week"]
        )
    )
    accumulated, archive_provenance = (
        (None, {"archived_capture_count": 0}) if not archived else ex.accumulated_stage_quotes(archived)
    )
    observations = discover_open_observations(card, quotes=accumulated)

    stages = list(st.SNAPSHOT_STAGES) if stage == STAGE_ALL else [st.validate_stage(stage)]

    if dry_run:
        # Plan only. Names every stage that WOULD be written, and why each
        # game is otherwise skipped, without touching the ledger.
        planned = []
        for one in stages:
            batches, skipped = ex.plan_stage_batches(
                stage=one,
                card=card,
                as_of_utc=as_of_utc,
                open_observations=observations,
                operational_root=operational_root,
                season=card_info["season"],
                week=card_info["week"],
            )
            planned.append(
                {
                    "stage": one,
                    "due_batches": len(batches),
                    "would_write": [
                        {"cutoff_utc": str(b.cutoff_utc), "game_ids": list(b.game_ids)}
                        for b in batches
                    ],
                    "skipped": skipped,
                }
            )
        return {
            "status": "DRY_RUN",
            "as_of_utc": str(as_of_utc),
            "card": card_info,
            "stage_archive": archive_provenance,
            "open_observations": {k: str(v) for k, v in sorted(observations.items())},
            "stages": planned,
            "wrote_nothing": True,
        }

    results = [
        ex.run_due_stage_snapshots(
            stage=one,
            horizon="TUE",
            card=card,
            as_of_utc=as_of_utc,
            operational_root=operational_root,
            open_observations=observations,
            archived_captures=archived,
            season=card_info["season"],
            week=card_info["week"],
            games=games,
            **batch_kwargs,
        )
        for one in stages
    ]
    return {
        "status": "OK" if all(r["status"] == "OK" for r in results) else "FAIL_CLOSED",
        "as_of_utc": str(as_of_utc),
        "card": card_info,
        "stage_archive": archive_provenance,
        "open_observations": {k: str(v) for k, v in sorted(observations.items())},
        "stages": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default=STAGE_ALL, choices=[STAGE_ALL, *st.SNAPSHOT_STAGES])
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--operational-root", default=None)
    parser.add_argument("--market-capture-manifest", default=None)
    parser.add_argument("--data-root", default=None, help="Override NFL_MODEL_DATA_ROOT (tests).")
    parser.add_argument(
        "--season", type=int, default=None,
        help="Repair an explicitly named past card. Requires --week. Default: the current card.",
    )
    parser.add_argument("--week", default=None, help="Repair this week. Requires --season.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be written and write nothing.",
    )
    args = parser.parse_args(argv)

    root = Path(args.operational_root) if args.operational_root else prod.artifact_root()
    result = run(
        stage=args.stage,
        as_of=args.as_of,
        operational_root=root,
        market_capture_manifest=Path(args.market_capture_manifest) if args.market_capture_manifest else None,
        data_root=Path(args.data_root) if args.data_root else None,
        historical_season=args.season,
        historical_week=args.week,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    # Exit 0 for a clean no-op (nothing due) as well as for successful
    # execution; a scheduled sweep that finds nothing to do is not a failure.
    return 0 if result["status"] in ("OK", "DRY_RUN", "NO_CURRENT_CARD") else 3


if __name__ == "__main__":
    raise SystemExit(main())
