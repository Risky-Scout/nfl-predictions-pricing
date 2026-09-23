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
from nfl_hybrid.production import snapshot_performance_ledger_2026 as pl  # noqa: E402
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


def card_still_stage_active(
    card: pd.DataFrame, *, season: int, week, operational_root: Path
) -> bool:
    """Whether any stage on this card can still legitimately be recorded.

    A card is finished when every game's resolvable stage is already in the
    performance ledger. Until then it stays active -- including for stages
    whose cutoff has not arrived yet, which is the whole point: a Monday-night
    CLOSE is still ahead of us on the Monday morning the next card becomes
    current.

    OPEN is deliberately NOT counted here. It depends on observed market
    evidence that may never have existed for a game, so an unobservable OPEN
    would keep a finished card active forever. MID and CLOSE are derived from
    the schedule and are therefore always resolvable for a pregame game.
    """
    for row in card.itertuples(index=False):
        kickoff = prod._as_utc(row.scheduled_kickoff_utc)
        for stage in (st.STAGE_MID, st.STAGE_CLOSE):
            if pl.read_snapshot(
                operational_root, season=season, week=week,
                game_id=str(row.game_id), stage=stage,
            ) is not None:
                continue
            cutoff = (
                st.close_cutoff_utc(kickoff)
                if stage == st.STAGE_CLOSE
                else st.mid_anchor_utc(st.card_earliest_kickoff_utc(card["scheduled_kickoff_utc"]))
            )
            if st.is_strictly_pregame(cutoff, kickoff):
                return True
    return False


def resolve_active_cards(
    games: pd.DataFrame, as_of_utc, *, operational_root: Path | None = None
) -> list[tuple[pd.DataFrame, dict]]:
    """Every card a sweep must process right now, oldest first.

    THE ROLLOVER GAP THIS CLOSES. The card becomes "current" on its own TUE
    cutoff, but the PREVIOUS card is not finished then -- its Monday-night
    game still closes at kickoff minus 60, which in 2026 Week 2 was
    ``2026-09-21T23:15:00Z``, roughly 17 hours AFTER the sweep had already
    advanced to Week 3. A sweep that only ever looks at the current card
    therefore walks away from a CLOSE that has not happened yet, and once it
    does the stage can never be recorded by any scheduled run.

    Returning the previous card alongside the current one fixes that without
    delaying anything: next week's OPEN evidence keeps accumulating on the
    current card in the same sweep. The previous card drops out as soon as its
    remaining stages are recorded, so this is bounded to a real overlap rather
    than an ever-growing backlog.
    """
    current_card, current_info = resolve_current_card(games, as_of_utc)
    if current_info["status"] != "OK":
        return [(current_card, current_info)]

    active: list[tuple[pd.DataFrame, dict]] = []
    if operational_root is not None:
        previous = _previous_card(games, current_info["tue_cutoff_utc"])
        if previous is not None:
            previous_card, previous_info = previous
            if card_still_stage_active(
                previous_card,
                season=previous_info["season"],
                week=previous_info["week"],
                operational_root=operational_root,
            ):
                previous_info = {**previous_info, "role": "PREVIOUS_STILL_ACTIVE"}
                active.append((previous_card, previous_info))

    active.append((current_card, {**current_info, "role": "CURRENT"}))
    return active


def resolve_publication_card(games: pd.DataFrame, as_of_utc) -> tuple[pd.DataFrame, dict]:
    """The card the PUBLIC feed should show right now.

    The feed carries exactly one ``(season, week)``, so a rollover forces a
    choice, and "whichever card is current" is the wrong one. The card turns
    over on its TUE cutoff, which in 2026 was the Monday MORNING -- so the
    public page would have dropped Week 2's Monday-night game roughly 18
    hours before it kicked off, and its CLOSE, landing at 23:15Z, would never
    have reached the feed at all.

    Publish the OLDEST still-active card that has a game yet to kick off.
    That keeps a week on the page for exactly as long as it still has
    something to show, and hands over the moment it does not:

        Mon 05:50Z  Week 2 -- Monday-nighter still ahead
        Mon 23:30Z  Week 2 -- its CLOSE is now recorded and publishable
        Tue 01:00Z  Week 3 -- Week 2 has no pregame game left

    A previous card whose games have all kicked off can never pin the page,
    even if some stage of it is still unrecorded, because the rule asks about
    KICKOFFS rather than about ledger state. Those are deliberately different
    questions: execution activity says whether a stage can still be recorded,
    while publication liveness says whether the public still has a game to
    look at. A week whose stages are all written is finished for the sweep
    but still live on the page until its last game starts.
    """
    as_of = prod._as_utc(as_of_utc)
    current_card, current_info = resolve_current_card(games, as_of)
    if current_info["status"] != "OK":
        return current_card, current_info

    candidates = []
    previous = _previous_card(games, current_info["tue_cutoff_utc"])
    if previous is not None:
        candidates.append(previous)
    candidates.append((current_card, current_info))

    for card, info in candidates:  # oldest first
        if any(prod._as_utc(k) > as_of for k in card["scheduled_kickoff_utc"]):
            return card, {**info, "publication_role": "LIVE_CARD"}
    # Nothing anywhere is still pregame; keep serving the current card.
    return current_card, {**current_info, "publication_role": "CURRENT_NOTHING_PREGAME"}


def _previous_card(games: pd.DataFrame, current_cutoff) -> tuple[pd.DataFrame, dict] | None:
    """The card immediately before the current one, or ``None``.

    Exactly one card back. A longer lookback would be a backlog sweeper, which
    is a different thing and would quietly re-open weeks that were closed
    deliberately.
    """
    ledger = he.build_horizon_membership_ledger(games)
    cutoffs = sorted({pd.Timestamp(c) for c in ledger["tue_cutoff_utc"]})
    current = pd.Timestamp(current_cutoff)
    earlier = [c for c in cutoffs if c < current]
    if not earlier:
        return None

    match = ledger[ledger["tue_cutoff_utc"] == max(earlier)]
    seasons = sorted({int(s) for s in match["season"]})
    weeks = sorted({str(w) for w in match["week"]})
    kickoffs = games[games["game_id"].astype(str).isin(set(match["game_id"].astype(str)))][
        ["game_id", "scheduled_kickoff_utc"]
    ].reset_index(drop=True)
    return kickoffs, {
        "status": "OK",
        "tue_cutoff_utc": str(max(earlier)),
        "season": seasons[0] if len(seasons) == 1 else seasons,
        "week": weeks[0] if len(weeks) == 1 else weeks,
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
        active = [resolve_historical_card(games, season=historical_season, week=historical_week)]
    else:
        # EVERY card still stage-active, not just the current one -- see
        # resolve_active_cards for the Monday-night rollover this closes.
        active = resolve_active_cards(games, as_of_utc, operational_root=operational_root)

    if all(info["status"] != "OK" for _, info in active):
        card_info = active[0][1]
        return {
            "status": card_info["status"],
            "as_of_utc": str(as_of_utc),
            "card": card_info,
            "stages": [],
        }

    stages = list(st.SNAPSHOT_STAGES) if stage == STAGE_ALL else [st.validate_stage(stage)]
    resolved_data_root = data_root if data_root is not None else os.environ.get("NFL_MODEL_DATA_ROOT")

    processed = [
        _process_card(
            card=card,
            card_info=card_info,
            stages=stages,
            as_of_utc=as_of_utc,
            operational_root=operational_root,
            data_root=resolved_data_root,
            games=games,
            dry_run=dry_run,
            **batch_kwargs,
        )
        for card, card_info in active
        if card_info["status"] == "OK"
    ]

    everything = [stage_result for card in processed for stage_result in card["stages"]]
    return {
        "status": (
            "DRY_RUN"
            if dry_run
            else ("OK" if all(s["status"] == "OK" for s in everything) else "FAIL_CLOSED")
        ),
        "as_of_utc": str(as_of_utc),
        # The card this sweep considers primary, kept for readers that only
        # ever expected one.
        "card": processed[-1]["card"],
        "active_cards": [card["card"] for card in processed],
        "cards": processed,
        "stage_archive": processed[-1]["stage_archive"],
        "open_observations": processed[-1]["open_observations"],
        "stages": everything,
        **({"wrote_nothing": True} if dry_run else {}),
    }


def _process_card(
    *,
    card: pd.DataFrame,
    card_info: dict,
    stages: list[str],
    as_of_utc,
    operational_root: Path,
    data_root,
    games: pd.DataFrame | None,
    dry_run: bool,
    **batch_kwargs,
) -> dict:
    """One card, end to end. Identical to the single-card path it replaces.

    Each card resolves its OWN observation archive, so processing two cards in
    one sweep shares no evidence between them and creates no duplicate
    capture, dataset or artifact -- each simply reads the archive that already
    exists for its week.
    """
    # OPEN is a fact about the season's ACCUMULATED evidence, and a historical
    # MID/CLOSE can only be priced from an observation that predates it. Both
    # need the archive, not just the capture this sweep happens to hold.
    # Tolerant of an unset root so a hermetic caller can run with no archive.
    archived = (
        []
        if data_root is None
        else ex.archived_stage_captures(
            Path(data_root), season=card_info["season"], week=card_info["week"]
        )
    )
    accumulated, archive_provenance = (
        (None, {"archived_capture_count": 0}) if not archived else ex.accumulated_stage_quotes(archived)
    )
    observations = discover_open_observations(card, quotes=accumulated)

    if dry_run:
        # Plan only. Names every stage that WOULD be written, and why each
        # game is otherwise skipped, without touching the ledger.
        results = []
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
            results.append(
                {
                    "stage": one,
                    "status": "DRY_RUN",
                    "week": card_info["week"],
                    "due_batches": len(batches),
                    "would_write": [
                        {"cutoff_utc": str(b.cutoff_utc), "game_ids": list(b.game_ids)}
                        for b in batches
                    ],
                    "skipped": skipped,
                }
            )
    else:
        results = [
            {
                **ex.run_due_stage_snapshots(
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
                ),
                "week": card_info["week"],
            }
            for one in stages
        ]

    return {
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
