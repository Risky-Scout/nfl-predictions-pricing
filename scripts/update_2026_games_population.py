"""DAILY: update the durable 2026 canonical games population from explicitly
supplied BallDontLie evidence captures.

This script performs no network call, fits nothing, prices nothing and
publishes nothing. It reads one or more EXPLICIT capture manifests, turns them
into canonical population rows through the repository's EXISTING BDL
canonicalizer (:func:`nfl_hybrid.providers.balldontlie.canonical.normalize_games`,
reached via :func:`nfl_hybrid.data.games_population_2026.games_from_capture`),
and merges them into the durable population under
``$NFL_MODEL_ARTIFACT_ROOT/games-population-2026/``.

CAPTURE SELECTION IS NEVER AUTOMATIC BY DEFAULT
  ``--capture-manifest`` (repeatable) names the exact manifests to ingest.
  ``--latest-games-evidence`` is the ONE opt-in convenience for the unattended
  daily job: it selects, per ``season_type``, the newest COMPLETE capture
  written by ``scripts/refresh_bdl_2026_games_evidence.py`` under that
  script's own evidence namespace, by the capture directory's own recorded UTC
  timestamp -- never by filesystem mtime, and never from the pregame
  ``balldontlie-2026`` observation log (an official TUE/FRI market capture is
  only ever ingested when named explicitly).

FAIL CLOSED
  A rejected capture, a conflicting canonical identity, or a changed recorded
  score aborts the run with a non-zero exit and leaves the durable population
  exactly as it was.

Usage:
  python scripts/update_2026_games_population.py --latest-games-evidence
  python scripts/update_2026_games_population.py --capture-manifest /path/manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.data import games_population_2026 as gp26  # noqa: E402
from nfl_hybrid.data.external_data import external_root  # noqa: E402

EVIDENCE_NAMESPACE = "balldontlie-2026-games-evidence"
_CAPTURE_STAMP = re.compile(r"^capture=(\d{8}T\d{6}Z)$")


def discover_latest_games_evidence(
    *, season: int, data_root: Path | None = None
) -> list[Path]:
    """The newest COMPLETE games-evidence capture manifest per season type.

    Ordering is by the capture directory's OWN recorded UTC stamp
    (``capture=YYYYMMDDTHHMMSSZ``), which is written once at capture time and
    never changes -- not by mtime, not by directory listing order, not by a
    manifest field that a later read could reinterpret.
    """
    root = Path(data_root) if data_root is not None else external_root()
    base = root / "live-observation-log" / EVIDENCE_NAMESPACE / f"season={season}"
    if not base.is_dir():
        return []

    selected: list[Path] = []
    for season_type_dir in sorted(base.glob("season_type=*")):
        candidates: list[tuple[str, Path]] = []
        for capture_dir in sorted(season_type_dir.glob("capture=*")):
            match = _CAPTURE_STAMP.match(capture_dir.name)
            manifest = capture_dir / "manifest.json"
            if match is None or not manifest.is_file():
                continue
            try:
                if json.loads(manifest.read_text()).get("status") != "COMPLETE":
                    continue
            except ValueError:
                continue
            candidates.append((match.group(1), manifest))
        if candidates:
            candidates.sort(key=lambda pair: pair[0])
            selected.append(candidates[-1][1])
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--capture-manifest",
        action="append",
        default=None,
        help="Repeatable. Explicit BallDontLie capture manifest.json path to ingest as 2026 games evidence.",
    )
    parser.add_argument(
        "--latest-games-evidence",
        action="store_true",
        help=(
            "Ingest the newest COMPLETE capture per season_type from the daily games-evidence namespace "
            "written by scripts/refresh_bdl_2026_games_evidence.py."
        ),
    )
    parser.add_argument("--season", type=int, default=gp26.PRODUCTION_SEASON)
    parser.add_argument("--data-root", default=None, help="Override NFL_MODEL_DATA_ROOT (tests / dry runs).")
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Override NFL_MODEL_ARTIFACT_ROOT for the durable population (tests / dry runs).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    manifests: list[Path] = [Path(p) for p in (args.capture_manifest or [])]
    if args.latest_games_evidence:
        try:
            manifests += discover_latest_games_evidence(
                season=args.season, data_root=Path(args.data_root) if args.data_root else None
            )
        except Exception as exc:
            print(json.dumps({"status": "FAIL_CLOSED", "detail": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
            return 2
    if not manifests:
        print(
            json.dumps(
                {
                    "status": "NO_EVIDENCE_SUPPLIED",
                    "detail": (
                        "no --capture-manifest given and --latest-games-evidence found no COMPLETE "
                        "games-evidence capture; the durable population was not touched"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    artifact_root_path = Path(args.artifact_root) if args.artifact_root else None
    rows_frames = []
    evidence: list[dict] = []
    for manifest_path in manifests:
        try:
            rows, entry = gp26.games_from_capture(manifest_path, expected_season=args.season)
        except gp26.GamesPopulationError as exc:
            print(
                json.dumps(
                    {"status": "FAIL_CLOSED", "manifest": str(manifest_path), "detail": str(exc)}, indent=2
                ),
                file=sys.stderr,
            )
            return 2
        rows_frames.append(rows)
        evidence.append(entry)

    import pandas as pd

    incoming = pd.concat(rows_frames, ignore_index=True) if rows_frames else pd.DataFrame()
    try:
        # Merging the incoming evidence against ITSELF first surfaces a
        # cross-capture identity conflict before any durable state is read.
        combined = incoming.iloc[0:0]
        for frame in rows_frames:
            combined = gp26.merge_population_rows(combined, frame)
        update = gp26.update_durable_2026_population(
            combined, evidence=evidence, artifact_root_path=artifact_root_path
        )
    except gp26.GamesPopulationError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": update.status,
                "population_path": str(update.path),
                "row_count": update.manifest["row_count"],
                "completed_row_count": update.manifest["completed_row_count"],
                "scheduled_row_count": update.manifest["scheduled_row_count"],
                "stale_unresolved_game_ids": update.manifest["stale_unresolved_game_ids"],
                "content_sha256": update.manifest["content_sha256"],
                "evidence_as_of_utc": update.manifest["evidence_as_of_utc"],
                "ingested_manifests": [e["manifest_path"] for e in evidence],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
