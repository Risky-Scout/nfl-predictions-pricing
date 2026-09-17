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

Usage:
  python scripts/run_2026_stage_snapshots.py --stage ALL
  python scripts/run_2026_stage_snapshots.py --stage CLOSE --as-of 2026-09-20T16:05:00Z
"""
from __future__ import annotations

import argparse
import json
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
    games: pd.DataFrame | None = None,
    **batch_kwargs,
) -> dict:
    as_of_utc = prod._as_utc(as_of) if as_of else prod.utc_now()
    if games is None:
        games, _ = prod.load_games_population_with_provenance()
        games = prod.filter_reg_post(games)

    card, card_info = resolve_current_card(games, as_of_utc)
    if card_info["status"] != "OK":
        return {"status": "NO_CURRENT_CARD", "as_of_utc": str(as_of_utc), "card": card_info, "stages": []}

    quotes = (
        None
        if market_capture_manifest is None
        else load_stage_quotes(
            market_capture_manifest, not_after_utc=as_of_utc, artifact_root_path=operational_root
        )
    )
    observations = discover_open_observations(card, quotes=quotes)

    stages = list(st.SNAPSHOT_STAGES) if stage == STAGE_ALL else [st.validate_stage(stage)]
    results = [
        ex.run_due_stage_snapshots(
            stage=one,
            horizon="TUE",
            card=card,
            as_of_utc=as_of_utc,
            operational_root=operational_root,
            open_observations=observations,
            market_capture_manifest=market_capture_manifest,
            quotes=quotes,
            games=games,
            **batch_kwargs,
        )
        for one in stages
    ]
    return {
        "status": "OK" if all(r["status"] == "OK" for r in results) else "FAIL_CLOSED",
        "as_of_utc": str(as_of_utc),
        "card": card_info,
        "open_observations": {k: str(v) for k, v in sorted(observations.items())},
        "stages": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default=STAGE_ALL, choices=[STAGE_ALL, *st.SNAPSHOT_STAGES])
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--operational-root", default=None)
    parser.add_argument("--market-capture-manifest", default=None)
    args = parser.parse_args(argv)

    root = Path(args.operational_root) if args.operational_root else prod.artifact_root()
    result = run(
        stage=args.stage,
        as_of=args.as_of,
        operational_root=root,
        market_capture_manifest=Path(args.market_capture_manifest) if args.market_capture_manifest else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    # Exit 0 for a clean no-op (nothing due) as well as for successful
    # execution; a scheduled sweep that finds nothing to do is not a failure.
    return 0 if result["status"] in ("OK", "NO_CURRENT_CARD") else 3


if __name__ == "__main__":
    raise SystemExit(main())
