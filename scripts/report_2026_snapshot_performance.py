"""Season CSV + JSON performance reporting from the snapshot ledger.

Read-only. Joins each recorded OPEN/MID/CLOSE snapshot to its game's
provider-final result, if there is one, and reports the model's error beside
the sportsbook's at the SAME instant -- which is the only way the comparison
means anything. A game that has not finished is reported as pending rather
than dropped, so the season totals never quietly shrink to the games that
happen to flatter the model.

Writes nothing except the two report files.

Usage:
  python scripts/report_2026_snapshot_performance.py \\
      --artifact-root $NFL_MODEL_ARTIFACT_ROOT \\
      --output-dir $NFL_MODEL_ARTIFACT_ROOT/production-2026/snapshot-performance-reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.production import snapshot_execution_2026 as ex  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    result = ex.write_season_reports(
        operational_root=Path(args.artifact_root), output_dir=Path(args.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
