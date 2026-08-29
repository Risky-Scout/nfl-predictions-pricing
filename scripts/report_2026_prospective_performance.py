"""Prospective 2026 performance reporter (Section 20).

Reads ONLY the immutable evaluation-ledger forecast+attached-result records
under ``$NFL_MODEL_ARTIFACT_ROOT/production-2026/evaluation-ledger/``.
Descriptive only: model-vs-sportsbook margin/total RMSE, ATS/TOTAL model
log loss/Brier vs market no-vig log loss/Brier with paired deltas,
calibration ECE, and sample sizes. No threshold optimization, no "best
cutoff," no strategy mining. Reports ``INSUFFICIENT_PROSPECTIVE_SAMPLE``
(overall and per stream) rather than force a conclusion from too few games
-- expected and fine against an empty/near-empty real 2026 ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.production import run_2026 as prod  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--evaluation-root", type=str, default=None,
        help="Override the evaluation-ledger root (defaults to $NFL_MODEL_ARTIFACT_ROOT/production-2026/evaluation-ledger).",
    )
    args = parser.parse_args()

    evaluation_root = Path(args.evaluation_root) if args.evaluation_root else (
        prod.artifact_root() / "production-2026" / "evaluation-ledger"
    )
    records = prod.load_prospective_records(evaluation_root)
    report = prod.compute_prospective_performance(records)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
