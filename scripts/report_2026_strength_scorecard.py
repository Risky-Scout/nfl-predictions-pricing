"""Prospective 2026 model-strength scorecard reporter (Sections 18-19).

Applies the FROZEN promotion contract
(``outputs/prospective_2026_strength_preregistration.json`` /
:mod:`nfl_hybrid.evaluation.prospective_strength_2026`) to the immutable
prospective ledgers and writes:

  $NFL_MODEL_ARTIFACT_ROOT/production-2026/prospective-strength/
      PROSPECTIVE_2026_STATUS_SCORECARD.json
      PROSPECTIVE_2026_STATUS_SCORECARD.md

Reads ONLY immutable records: the forecast ledger, the evaluation ledger +
attached ``*.result.json`` files, the shadow model-family ledger, the
executable-price ledger (if activated), the run manifests, and
``config/executable_books_2026.json`` (presence + hash only). It never
reads a retrospective 2020-2025 evaluation artifact to score a 2026 gate.

Before enough prospective results exist this does NOT fail: it prints a
scorecard whose performance rows read ``INSUFFICIENT_PROSPECTIVE_SAMPLE`` /
``NOT_DEMONSTRATED`` / ``NOT_ESTABLISHED`` and whose ``reported_status``
values are the preserved current evidence-supported labels. It also runs
against an unset ``$NFL_MODEL_ARTIFACT_ROOT`` (nothing to read, nothing to
write -- it prints the empty-estate scorecard to stdout).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.data.external_data import ExternalDataUnavailableError, REPO_ROOT, artifact_root  # noqa: E402
from nfl_hybrid.evaluation import prospective_strength_2026 as ps  # noqa: E402


def _load_run_manifests(root: Path) -> list[dict]:
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--operational-root", type=str, default=None,
        help="Override the production-2026 operational root (defaults to $NFL_MODEL_ARTIFACT_ROOT).",
    )
    parser.add_argument(
        "--data-through-utc", type=str, default=None,
        help="ISO-8601 timestamp recorded in the scorecard as the evidence horizon.",
    )
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write scorecard files.")
    args = parser.parse_args()

    if args.operational_root:
        aroot = Path(args.operational_root)
    else:
        try:
            aroot = artifact_root()
        except ExternalDataUnavailableError:
            aroot = None

    if aroot is None:
        scorecard = ps.build_scorecard(data_through_utc=args.data_through_utc)
        print(json.dumps(scorecard, indent=2, sort_keys=True, default=str))
        print(
            "\n# NFL_MODEL_ARTIFACT_ROOT is unset -- printed the empty-estate scorecard only.",
            file=sys.stderr,
        )
        return 0

    base = aroot / "production-2026"
    book_policy = ps.executable_book_policy_status(REPO_ROOT / "config" / "executable_books_2026.json")
    scorecard = ps.build_scorecard(
        forecast_records=ps.load_forecast_ledger(base / "forecast-ledger"),
        evaluation_records=ps.load_attached_evaluation_records(base / "evaluation-ledger"),
        shadow_records=ps.load_shadow_ledger(base / "shadow-model-family-ledger"),
        price_records=ps.load_executable_price_ledger(base / "executable-price-ledger"),
        run_manifests=_load_run_manifests(base / "run-manifests"),
        book_policy=book_policy,
        book_policy_lock=ps.verify_executable_book_policy_lock(
            aroot, policy_hash=book_policy.get("policy_hash")
        ),
        integrity_events=ps.load_betting_integrity_events(aroot),
        data_through_utc=args.data_through_utc,
    )

    print(json.dumps(scorecard, indent=2, sort_keys=True, default=str))

    if not args.no_write:
        out_dir = base / "prospective-strength"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "PROSPECTIVE_2026_STATUS_SCORECARD.json").write_text(
            json.dumps(scorecard, indent=2, sort_keys=True, default=str)
        )
        (out_dir / "PROSPECTIVE_2026_STATUS_SCORECARD.md").write_text(ps.render_scorecard_markdown(scorecard))
        print(f"\n# wrote {out_dir}/PROSPECTIVE_2026_STATUS_SCORECARD.{{json,md}}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
