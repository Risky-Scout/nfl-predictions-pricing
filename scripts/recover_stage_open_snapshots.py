"""Audited correction of OPEN snapshots written by the current-capture resolver.

WHAT WENT WRONG. Before the historical resolver, ``discover_open_instant`` saw
only the capture a sweep happened to be holding, so "the first valid market
observed" was computed as "the earliest observation in THIS capture" -- which
is always roughly now. The 2026 Week-2 OPEN rows were therefore recorded at
``2026-09-19 04:42``, about two days and three hours after the board was in
fact first priceable at ``2026-09-17 01:24``.

WHY NOT JUST OVERWRITE THEM. They are immutable evidence of what production
recorded, and a pregame snapshot that gets quietly edited once the answer is
known is no longer evidence of anything. This tool therefore uses the
convention the repository already established for Fix-3.1, Fix-4 and Fix-7.1:
the defective artifact is MOVED, intact, under
``$NFL_MODEL_ARTIFACT_ROOT/invalidated/<defect-slug>/`` -- a tree
``scripts/prune_replaceable_artifacts.py`` explicitly protects -- and the
corrected artifact carries ``supersedes_invalidated_artifact`` naming it.
Nothing is deleted and nothing is overwritten in place.

DRY BY DEFAULT. Without ``--apply`` this writes nothing at all and simply
reports what it would do. With ``--apply`` it is idempotent: a row already
corrected is left alone, and a row whose recorded OPEN already equals the
archive's answer is reported as correct and untouched.

FAIL-CLOSED ON SURPRISE. The archive is the authority for the corrected value,
so if it cannot establish an OPEN for a game the tool refuses rather than
guessing, and it refuses a destination that is already occupied by different
content. It never invents a snapshot for a game the resolver cannot price.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.production import run_2026 as prod  # noqa: E402
from nfl_hybrid.production import snapshot_execution_2026 as ex  # noqa: E402
from nfl_hybrid.production import snapshot_performance_ledger_2026 as pl  # noqa: E402
from nfl_hybrid.production import snapshot_stages_2026 as st  # noqa: E402

DEFAULT_DEFECT_SLUG = "stage-open-current-capture-resolver-2026-09-19"
SUPERSEDES_KEY = "supersedes_invalidated_artifact"
INVALIDATION_REASON = (
    "OPEN was resolved from the single capture the sweep was holding rather than from the "
    "accumulated STAGE observation archive, so it recorded the sweep's own instant instead of "
    "the first instant the board was actually priceable."
)

# Per-game outcomes.
STATUS_CORRECT = "ALREADY_CORRECT"
STATUS_SUPERSEDE = "SUPERSEDE"
STATUS_MISSING_ROW = "MISSING_LEDGER_ROW"
STATUS_NO_ARCHIVE_OPEN = "NO_ARCHIVE_OPEN"
STATUS_ALREADY_RECOVERED = "ALREADY_RECOVERED"


class RecoveryRefused(RuntimeError):
    """Fail-closed recovery error. Nothing is moved or written."""


def invalidated_dir(artifact_root: Path, *, defect_slug: str) -> Path:
    return Path(artifact_root) / "invalidated" / defect_slug


def _true_open(accumulated, card_row) -> pd.Timestamp | None:
    return ex.discover_open_instant(
        accumulated,
        game_id=str(card_row.game_id),
        scheduled_kickoff_utc=card_row.scheduled_kickoff_utc,
    )


def plan(
    *,
    artifact_root: Path,
    data_root: Path,
    season: int,
    week,
    as_of_utc=None,
    defect_slug: str = DEFAULT_DEFECT_SLUG,
    games: pd.DataFrame | None = None,
) -> dict:
    """What recovery WOULD do. Pure: reads only, writes nothing."""
    artifact_root, data_root = Path(artifact_root), Path(data_root)
    as_of = prod._as_utc(as_of_utc) if as_of_utc is not None else prod.utc_now()

    if games is None:
        games = prod.filter_reg_post(prod.load_games_population_with_provenance()[0])
    ledger = prod.he.build_horizon_membership_ledger(games)
    tue = prod.current_or_recent_cutoff(as_of, "TUE")
    card_ids = sorted(set(ledger[ledger["tue_cutoff_utc"] == tue]["game_id"].astype(str)))
    card = games[games["game_id"].astype(str).isin(set(card_ids))][
        ["game_id", "scheduled_kickoff_utc"]
    ].reset_index(drop=True)
    if card.empty:
        raise RecoveryRefused(f"no card resolved for season={season} week={week} at {as_of.isoformat()}")

    captures = ex.archived_stage_captures(data_root, season=season, week=week)
    if not captures:
        raise RecoveryRefused(
            f"no archived STAGE captures under {data_root} for season={season} week={week}; "
            "the archive IS the authority for the corrected OPEN and is never guessed"
        )
    accumulated, archive_provenance = ex.accumulated_stage_quotes(captures)
    if accumulated is None:
        raise RecoveryRefused(
            f"none of the {len(captures)} archived captures produced usable quotes: {archive_provenance}"
        )

    destination = invalidated_dir(artifact_root, defect_slug=defect_slug)
    actions: list[dict] = []
    for row in card.itertuples(index=False):
        game_id = str(row.game_id)
        recorded = pl.read_snapshot(
            artifact_root, season=season, week=week, game_id=game_id, stage=st.STAGE_OPEN
        )
        true_open = _true_open(accumulated, row)
        entry = {
            "game_id": game_id,
            "recorded_open": None if recorded is None else recorded.get("snapshot_at_utc"),
            "archive_open": None if true_open is None else str(true_open),
        }

        if true_open is None:
            entry["status"] = STATUS_NO_ARCHIVE_OPEN
        elif recorded is None:
            entry["status"] = STATUS_MISSING_ROW
        elif pd.Timestamp(recorded["snapshot_at_utc"]) == true_open:
            entry["status"] = STATUS_CORRECT
        elif recorded.get(SUPERSEDES_KEY):
            entry["status"] = STATUS_ALREADY_RECOVERED
        else:
            entry["status"] = STATUS_SUPERSEDE
            entry["drift"] = str(pd.Timestamp(recorded["snapshot_at_utc"]) - true_open)
            entry["invalidated_path"] = str(destination / f"{game_id}__OPEN.json")
        actions.append(entry)

    counts: dict[str, int] = {}
    for action in actions:
        counts[action["status"]] = counts.get(action["status"], 0) + 1
    return {
        "season": season,
        "week": week,
        "defect_slug": defect_slug,
        "invalidated_dir": str(destination),
        "stage_archive": archive_provenance,
        "card_game_count": int(len(card)),
        "status_counts": counts,
        "actions": actions,
    }


SUPERSESSION_MANIFEST = "supersession_manifest.json"


def quarantine(decision: dict, *, artifact_root: Path) -> dict:
    """Move every defective OPEN row under ``invalidated/`` and record why.

    This is the only mutating step, and it is deliberately narrow: it does NOT
    compute a replacement. The corrected row is produced by re-running the
    ordinary fixed resolver afterwards, so the recovery cannot drift from
    production -- whatever the next sweep would write is exactly what recovery
    writes. Removing the live row is what lets it: with no ledger entry the
    already-recorded skip stands aside, and the archive resolver supplies the
    true OPEN.

    The link from corrected back to invalidated lives in the supersession
    manifest written here, following the ``invalidated/<defect-slug>/``
    convention Fix-3.1, Fix-4 and Fix-7.1 already use.
    """
    artifact_root = Path(artifact_root)
    destination = Path(decision["invalidated_dir"])
    season, week = decision["season"], decision["week"]

    moved, skipped, links = [], [], {}
    for action in decision["actions"]:
        game_id = action["game_id"]
        if action["status"] != STATUS_SUPERSEDE:
            skipped.append({"game_id": game_id, "status": action["status"]})
            continue

        source = pl.snapshot_path(
            artifact_root, season=season, week=week, game_id=game_id, stage=st.STAGE_OPEN
        )
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / f"{game_id}__OPEN.json"
        if target.exists():
            # Idempotent: a second run must not clobber preserved evidence.
            if source.is_file() and target.read_bytes() != source.read_bytes():
                raise RecoveryRefused(
                    f"{target} already holds a DIFFERENT invalidated artifact -- refusing to "
                    "overwrite preserved forensic evidence"
                )
        else:
            if not source.is_file():
                raise RecoveryRefused(f"expected a defective OPEN row at {source} and found none")
            shutil.move(str(source), str(target))
        moved.append(str(target))
        links[game_id] = {
            SUPERSEDES_KEY: str(target.relative_to(artifact_root)),
            "invalidated_open": action["recorded_open"],
            "archive_open": action["archive_open"],
            "drift": action.get("drift"),
        }

    if links:
        manifest = {
            "defect_slug": decision["defect_slug"],
            "invalidation_reason": INVALIDATION_REASON,
            "season": season,
            "week": week,
            "stage": st.STAGE_OPEN,
            "stage_archive": decision["stage_archive"],
            "superseded": links,
        }
        path = destination / SUPERSESSION_MANIFEST
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return {"moved_to_invalidated": moved, "skipped": skipped, "superseded_games": sorted(links)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--defect-slug", default=DEFAULT_DEFECT_SLUG)
    parser.add_argument("--apply", action="store_true", help="Actually move/write. Omitted, this is a dry run.")
    args = parser.parse_args(argv)

    artifact_root = Path(args.artifact_root) if args.artifact_root else prod.artifact_root()
    data_root = Path(args.data_root) if args.data_root else Path(os.environ["NFL_MODEL_DATA_ROOT"])

    try:
        decision = plan(
            artifact_root=artifact_root,
            data_root=data_root,
            season=args.season,
            week=args.week,
            as_of_utc=args.as_of,
            defect_slug=args.defect_slug,
        )
        decision["quarantine"] = (
            quarantine(decision, artifact_root=artifact_root) if args.apply else None
        )
    except RecoveryRefused as exc:
        print(json.dumps({"status": "REFUSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    decision["status"] = "QUARANTINED" if args.apply else "DRY_RUN"
    if args.apply:
        decision["next_step"] = (
            "Run the ordinary snapshot sweep. With the defective rows quarantined the fixed "
            "resolver will write the corrected OPEN -- including the missing one -- from the "
            "archive, and the supersession manifest links them back."
        )
    print(json.dumps(decision, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
