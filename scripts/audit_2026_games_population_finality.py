"""AUDIT (and, on request, REPAIR) the durable 2026 canonical games population
against the newest BallDontLie games evidence.

Why this exists
---------------
Until the completion-status fix, ``canonical_games_to_population_rows`` copied
whatever score a capture carried, without consulting the provider's own
``status_state``. A capture taken while a game was still being played
therefore recorded a running scoreline as the game's RESULT -- and because a
recorded result is immutable, the real final score was afterwards rejected and
the daily pass failed closed. Observed in production on 2026-09-13 for
``2026_01_BAL_IND``: stored (23.0, 38.0), incoming (23.0, 41.0).

The code defect is fixed at the source. This script answers the two remaining
operational questions:

  1. WHICH stored rows are wrong? The merge fails closed on the FIRST conflict,
     so a single failure message says nothing about how many other rows are
     affected. The audit here classifies EVERY row and never stops early.
  2. HOW is the population corrected? By REBUILDING the derived population from
     the authoritative captures through the fixed code path -- never by
     hand-editing the parquet.

Safety
------
``--audit`` (the default) is strictly READ-ONLY: it opens the population and
the capture manifests and writes nothing at all.

``--repair`` rewrites exactly one derived artifact, the population parquet and
its manifest under ``games-population-2026/``. It touches nothing else: not the
historical Odds API estates, not the Fix-8 calibration estate, not the Week-1
TUE capture, not prospective evaluation history, not the certified baseline,
and no model configuration or hash. The superseded parquet is kept alongside
as a timestamped ``.superseded-<stamp>.parquet``, and every corrected cell --
the rejected old value and the authoritative new one -- is recorded in
``canonical_games_2026.repair.json``.

Usage:
  python scripts/audit_2026_games_population_finality.py
  python scripts/audit_2026_games_population_finality.py --json
  python scripts/audit_2026_games_population_finality.py --repair --confirm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.data import games_population_2026 as gp26  # noqa: E402
from nfl_hybrid.data.external_data import external_root  # noqa: E402
from nfl_hybrid.providers.balldontlie import finality as bdl_finality  # noqa: E402

from update_2026_games_population import (  # noqa: E402
    EVIDENCE_NAMESPACE,
    discover_latest_games_evidence,
)

_CAPTURE_STAMP = re.compile(r"^capture=(\d{8}T\d{6}Z)$")

# Per-game verdicts.
OK_FINAL = "OK_FINAL"
OK_UNRESOLVED = "OK_UNRESOLVED"
PENDING_INGEST = "PENDING_INGEST"
PREMATURELY_FROZEN_STALE = "PREMATURELY_FROZEN_STALE"
FROZEN_WITHOUT_FINAL_AUTHORITY = "FROZEN_WITHOUT_FINAL_AUTHORITY"
MISSING_FROM_EVIDENCE = "MISSING_FROM_EVIDENCE"
MISSING_FROM_POPULATION = "MISSING_FROM_POPULATION"

# The verdicts that mean the stored population is actually wrong.
DEFECTIVE_VERDICTS = (PREMATURELY_FROZEN_STALE, FROZEN_WITHOUT_FINAL_AUTHORITY)


def discover_all_games_evidence(*, season: int, data_root: Path | None = None) -> list[Path]:
    """EVERY COMPLETE games-evidence capture manifest, oldest first.

    Ordered by the capture directory's own recorded UTC stamp, exactly like
    ``discover_latest_games_evidence`` -- never by mtime. The rebuild replays
    these in that order so it reproduces the real merge history rather than
    inventing a new one.
    """
    root = Path(data_root) if data_root is not None else external_root()
    base = root / "live-observation-log" / EVIDENCE_NAMESPACE / f"season={season}"
    if not base.is_dir():
        return []

    found: list[tuple[str, Path]] = []
    for season_type_dir in sorted(base.glob("season_type=*")):
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
            found.append((match.group(1), manifest))
    found.sort(key=lambda pair: pair[0])
    return [manifest for _stamp, manifest in found]


def _evidence_truth(manifests: list[Path], *, season: int) -> dict[str, dict]:
    """What the newest evidence says about each game: its score and whether the
    provider itself calls it final. Read through the same canonicalizer the
    production path uses, never a bespoke parse."""
    truth: dict[str, dict] = {}
    for manifest_path in manifests:
        capture = gp26.bridge.validate_capture_manifest(
            manifest_path,
            expected_season=season,
            required_sources=(gp26.bridge.GAMES_LOGICAL_NAME,),
            allowed_horizons=gp26.GAMES_EVIDENCE_HORIZONS,
        )
        normalized = gp26.bdl_canonical.normalize_games(
            gp26.bridge.read_games_rows(capture), season_type_hint=capture.season_type
        )
        for record in normalized.to_dict(orient="records"):
            if str(record.get("season_type", "")).upper() not in gp26.PRODUCTION_SEASON_TYPES:
                continue
            game_id = str(record["game_id"])
            home = pd.to_numeric(record.get("home_score"), errors="coerce")
            away = pd.to_numeric(record.get("away_score"), errors="coerce")
            truth[game_id] = {
                "status": record.get("status"),
                "status_state": record.get("status_state"),
                "is_final": bdl_finality.is_final(record.get("status_state")),
                "home_score": None if pd.isna(home) else float(home),
                "away_score": None if pd.isna(away) else float(away),
                "manifest_path": str(manifest_path),
            }
    return truth


def _pair(record) -> tuple[float | None, float | None]:
    home = record.get("home_score")
    away = record.get("away_score")
    return (
        None if home is None or pd.isna(home) else float(home),
        None if away is None or pd.isna(away) else float(away),
    )


def audit(population: pd.DataFrame, truth: dict[str, dict]) -> list[dict]:
    """Classify EVERY game. Never stops at the first mismatch -- the whole
    point is to learn the blast radius, not to re-discover one failure."""
    findings: list[dict] = []
    stored_ids = set()

    for record in population.to_dict(orient="records"):
        game_id = str(record["game_id"])
        stored_ids.add(game_id)
        stored_home, stored_away = _pair(record)
        stored_has_result = stored_home is not None and stored_away is not None
        evidence = truth.get(game_id)

        if evidence is None:
            verdict = MISSING_FROM_EVIDENCE
        elif stored_has_result and evidence["is_final"]:
            same = (stored_home, stored_away) == (evidence["home_score"], evidence["away_score"])
            verdict = OK_FINAL if same else PREMATURELY_FROZEN_STALE
        elif stored_has_result and not evidence["is_final"]:
            # A score was recorded for a game the provider still does not call
            # final: it can only have come from a non-final capture.
            verdict = FROZEN_WITHOUT_FINAL_AUTHORITY
        elif not stored_has_result and evidence["is_final"]:
            verdict = PENDING_INGEST
        else:
            verdict = OK_UNRESOLVED

        findings.append({
            "game_id": game_id,
            "verdict": verdict,
            "stored_home_score": stored_home,
            "stored_away_score": stored_away,
            "evidence_home_score": None if evidence is None else evidence["home_score"],
            "evidence_away_score": None if evidence is None else evidence["away_score"],
            "evidence_status": None if evidence is None else evidence["status"],
            "evidence_status_state": None if evidence is None else evidence["status_state"],
            "evidence_is_final": None if evidence is None else evidence["is_final"],
        })

    for game_id in sorted(set(truth) - stored_ids):
        findings.append({
            "game_id": game_id,
            "verdict": MISSING_FROM_POPULATION,
            "stored_home_score": None,
            "stored_away_score": None,
            "evidence_home_score": truth[game_id]["home_score"],
            "evidence_away_score": truth[game_id]["away_score"],
            "evidence_status": truth[game_id]["status"],
            "evidence_status_state": truth[game_id]["status_state"],
            "evidence_is_final": truth[game_id]["is_final"],
        })

    findings.sort(key=lambda f: f["game_id"])
    return findings


def rebuild_from_evidence(manifests: list[Path], *, season: int) -> tuple[pd.DataFrame, list[dict]]:
    """Replay every capture, oldest first, through the FIXED code path."""
    rebuilt = None
    evidence: list[dict] = []
    for manifest_path in manifests:
        rows, entry = gp26.games_from_capture(manifest_path, expected_season=season)
        rebuilt = rows if rebuilt is None else gp26.merge_population_rows(rebuilt, rows)
        evidence.append(entry)
    if rebuilt is None:
        raise gp26.GamesPopulationError("no COMPLETE games evidence found; nothing to rebuild from")
    return rebuilt, evidence


def diff_cells(before: pd.DataFrame, after: pd.DataFrame) -> list[dict]:
    """Every score cell the rebuild changes, with the rejected old value and
    the authoritative new one -- the provenance record of the repair."""
    old = {str(r["game_id"]): _pair(r) for r in before.to_dict(orient="records")}
    new = {str(r["game_id"]): _pair(r) for r in after.to_dict(orient="records")}
    changes: list[dict] = []
    for game_id in sorted(set(old) | set(new)):
        old_pair = old.get(game_id)
        new_pair = new.get(game_id)
        if old_pair != new_pair:
            changes.append({
                "game_id": game_id,
                "rejected_home_score": None if old_pair is None else old_pair[0],
                "rejected_away_score": None if old_pair is None else old_pair[1],
                "corrected_home_score": None if new_pair is None else new_pair[0],
                "corrected_away_score": None if new_pair is None else new_pair[1],
            })
    return changes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repair", action="store_true", help="Rebuild the population from authoritative evidence.")
    parser.add_argument("--confirm", action="store_true", help="Required alongside --repair; without it nothing is written.")
    parser.add_argument("--json", action="store_true", help="Emit the full machine-readable finding list.")
    parser.add_argument("--season", type=int, default=gp26.PRODUCTION_SEASON)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--artifact-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_root = Path(args.data_root) if args.data_root else None
    artifact_root_path = Path(args.artifact_root) if args.artifact_root else None

    population, manifest = gp26.read_durable_2026_population(artifact_root_path)
    if manifest is None and population.empty:
        print(json.dumps({"status": "NO_POPULATION", "detail": "no durable 2026 population exists yet"}, indent=2))
        return 1

    latest = discover_latest_games_evidence(season=args.season, data_root=data_root)
    if not latest:
        print(json.dumps({"status": "NO_EVIDENCE", "detail": "no COMPLETE games-evidence capture found"}, indent=2))
        return 1

    try:
        truth = _evidence_truth(latest, season=args.season)
    except Exception as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2

    findings = audit(population, truth)
    by_verdict: dict[str, list[str]] = {}
    for finding in findings:
        by_verdict.setdefault(finding["verdict"], []).append(finding["game_id"])
    defective = [f for f in findings if f["verdict"] in DEFECTIVE_VERDICTS]

    report = {
        "status": "DEFECTS_FOUND" if defective else "CLEAN",
        "population_row_count": int(len(population)),
        "evidence_manifests": [str(p) for p in latest],
        "verdict_counts": {verdict: len(ids) for verdict, ids in sorted(by_verdict.items())},
        "defective_game_ids": [f["game_id"] for f in defective],
        "defects": defective,
        # An in-progress score that happened to equal the final score is
        # indistinguishable from a correctly recorded one here, and is
        # harmless: the stored value is right either way.
        "note": (
            "PREMATURELY_FROZEN_STALE = stored score contradicts the final evidence. "
            "FROZEN_WITHOUT_FINAL_AUTHORITY = a score is stored for a game the provider does not call final."
        ),
    }
    if args.json:
        report["findings"] = findings

    if not args.repair:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not defective else 3

    if not args.confirm:
        report["repair"] = "REFUSED: --repair requires --confirm"
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    all_manifests = discover_all_games_evidence(season=args.season, data_root=data_root)
    try:
        rebuilt, evidence = rebuild_from_evidence(all_manifests, season=args.season)
    except gp26.GamesPopulationError as exc:
        report["repair"] = f"FAIL_CLOSED: {exc}"
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    changes = diff_cells(population, rebuilt)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    target = gp26.population_path(artifact_root_path)
    superseded = target.with_name(f"{target.stem}.superseded-{stamp}.parquet")
    population.to_parquet(superseded, index=False)

    # Replace the derived population wholesale, through the module's own
    # atomic writer, rather than editing cells in place.
    target.unlink(missing_ok=True)
    gp26.population_manifest_path(artifact_root_path).unlink(missing_ok=True)
    update = gp26.update_durable_2026_population(
        rebuilt, evidence=evidence, artifact_root_path=artifact_root_path
    )

    provenance = {
        "repaired_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "reason": (
            "scores captured while a game was still in progress had been recorded as immutable "
            "results; the population was rebuilt from authoritative evidence after the "
            "completion-status fix"
        ),
        "superseded_population_parquet": str(superseded),
        "superseded_content_sha256": (manifest or {}).get("content_sha256"),
        "repaired_content_sha256": update.manifest["content_sha256"],
        "replayed_capture_manifests": [str(p) for p in all_manifests],
        "corrected_cells": changes,
        "audit_before_repair": report,
    }
    provenance_path = gp26.population_dir(artifact_root_path) / "canonical_games_2026.repair.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str))

    print(json.dumps({
        "status": "REPAIRED",
        "population_path": str(update.path),
        "row_count": update.manifest["row_count"],
        "completed_row_count": update.manifest["completed_row_count"],
        "scheduled_row_count": update.manifest["scheduled_row_count"],
        "content_sha256": update.manifest["content_sha256"],
        "corrected_cell_count": len(changes),
        "corrected_cells": changes,
        "superseded_population_parquet": str(superseded),
        "provenance_path": str(provenance_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
