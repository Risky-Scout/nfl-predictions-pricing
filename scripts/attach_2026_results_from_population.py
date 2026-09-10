"""DAILY: update prospective evaluation by attaching real 2026 results to
already-written production forecasts.

This is the prospective-evaluation half of the daily pass. It generates no
forecast, fits nothing, calibrates nothing, prices nothing and publishes
nothing. It only takes final scores that are ALREADY recorded in the durable
canonical BallDontLie 2026 games population and attaches them to the matching
evaluation-ledger forecasts through the EXISTING, immutability-enforcing
:func:`nfl_hybrid.production.run_2026.attach_result`.

NO RESULT IS EVER INVENTED, AND NONE CAN LEAK BACKWARD
  * A score comes only from the durable population, which itself comes only
    from verified BallDontLie captures.
  * ``result_available_at_utc`` is the CERTIFIED availability instant
    (:func:`nfl_hybrid.features.horizon_elo.compute_result_available_at_utc`),
    not the moment this script noticed the score, so attaching a result late
    cannot make it look like it was known earlier.
  * ``attach_result`` itself refuses any attachment whose
    ``result_available_at_utc`` is not strictly before the attachment instant,
    and refuses to change an already-attached result at all. A forecast is
    never mutated -- the result lands in a sibling ``*.result.json``.
  * A forecast is only ever matched by its exact
    ``game_id``/``horizon``/``target_cutoff_utc`` identity.

WHY THIS IS NOT A SECOND EVALUATION ENGINE
  Scoring stays where it already lives:
  :func:`nfl_hybrid.production.run_2026.load_prospective_records` /
  :func:`nfl_hybrid.production.run_2026.compute_prospective_performance`, which
  ``scripts/report_2026_prospective_performance.py`` already reports. This
  script only supplies the labels those functions have been waiting for.

Usage:
  python scripts/attach_2026_results_from_population.py
  python scripts/attach_2026_results_from_population.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfl_hybrid.data import games_population_2026 as gp26  # noqa: E402
from nfl_hybrid.data.external_data import artifact_root  # noqa: E402
from nfl_hybrid.features import horizon_elo as he  # noqa: E402
from nfl_hybrid.production import run_2026 as prod  # noqa: E402


def _pending_forecasts(evaluation_root: Path) -> list[dict]:
    """Evaluation-ledger forecasts with no result attached yet.

    Mirrors :func:`run_2026.load_prospective_records`'s traversal exactly, but
    keeps the complement: the records that are still unlabelled.
    """
    if not evaluation_root.exists():
        return []
    pending = []
    for forecast_path in sorted(evaluation_root.glob("*/*.json")):
        name = forecast_path.name
        if name.endswith(".result.json") or name.endswith(".json.tmp"):
            continue
        if forecast_path.with_suffix(".result.json").exists():
            continue
        pending.append(json.loads(forecast_path.read_text()))
    return pending


def attach_available_results(
    *,
    artifact_root_path: Path,
    operational_root: Path,
    dry_run: bool = False,
) -> dict:
    evaluation_root = Path(operational_root) / "production-2026" / "evaluation-ledger"
    population, manifest = gp26.read_durable_2026_population(artifact_root_path)

    completed = population.loc[population["home_score"].notna() & population["away_score"].notna()].copy()
    if len(completed):
        completed["result_available_at_utc"] = he.compute_result_available_at_utc(completed)
    scores = {
        str(row["game_id"]): row
        for row in completed.to_dict(orient="records")
    }

    pending = _pending_forecasts(evaluation_root)
    attachment_run_time = prod.utc_now()

    attached: list[dict] = []
    skipped: list[dict] = []
    failures: list[dict] = []

    for forecast in pending:
        game_id = str(forecast["game_id"])
        row = scores.get(game_id)
        if row is None:
            skipped.append({"game_id": game_id, "reason": "NO_COMPLETED_RESULT_IN_POPULATION"})
            continue

        result = {"home_score": float(row["home_score"]), "away_score": float(row["away_score"])}
        available_at = pd.Timestamp(row["result_available_at_utc"])
        if not (available_at < attachment_run_time):
            # Certified availability has not been reached yet even though a
            # score is present. Never backdate; wait for the next daily pass.
            skipped.append(
                {
                    "game_id": game_id,
                    "reason": "RESULT_NOT_YET_CHRONOLOGICALLY_AVAILABLE",
                    "result_available_at_utc": available_at.isoformat(),
                }
            )
            continue

        # Binds the attached label to the exact population content it came
        # from, so a reviewer can prove which evidence produced this result.
        source_hash = sha256(
            json.dumps(
                {
                    "population_content_sha256": (manifest or {}).get("content_sha256"),
                    "game_id": game_id,
                    "result": result,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        entry = {
            "game_id": game_id,
            "horizon": forecast["horizon"],
            "target_cutoff_utc": forecast["target_cutoff_utc"],
            "result": result,
            "result_available_at_utc": available_at.isoformat(),
            "result_source_hash": source_hash,
        }
        if dry_run:
            attached.append({**entry, "status": "WOULD_ATTACH"})
            continue
        try:
            path = prod.attach_result(
                evaluation_root,
                game_id=game_id,
                horizon=forecast["horizon"],
                target_cutoff_utc=forecast["target_cutoff_utc"],
                result=result,
                result_available_at_utc=available_at,
                attachment_run_time=attachment_run_time,
                result_source_hash=source_hash,
            )
        except prod.ProductionHardStop as exc:
            failures.append({**entry, "status": exc.status, "detail": exc.detail})
            continue
        attached.append({**entry, "status": "ATTACHED", "path": str(path)})

    return {
        "status": "FAIL_CLOSED" if failures else ("DRY_RUN" if dry_run else "OK"),
        "evaluation_root": str(evaluation_root),
        "population_content_sha256": (manifest or {}).get("content_sha256"),
        "completed_2026_games": int(len(completed)),
        "pending_forecasts": len(pending),
        "attached_count": len(attached),
        "skipped_count": len(skipped),
        "failure_count": len(failures),
        "attached": attached,
        "skipped": skipped,
        "failures": failures,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifact-root", default=None, help="Override NFL_MODEL_ARTIFACT_ROOT.")
    parser.add_argument(
        "--operational-root",
        default=None,
        help="Root holding production-2026/ ledgers (defaults to the artifact root).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be attached; write nothing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        artifact_root_path = Path(args.artifact_root) if args.artifact_root else artifact_root()
    except Exception as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2
    operational_root = Path(args.operational_root) if args.operational_root else artifact_root_path

    report = attach_available_results(
        artifact_root_path=artifact_root_path,
        operational_root=operational_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 2 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
