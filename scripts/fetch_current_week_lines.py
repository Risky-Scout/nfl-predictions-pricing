"""DEPRECATED thin wrapper. Superseded by scripts/build_week1_2026_lines.py, which
routes through the ONE canonical market-pricing path (pricing/market_pricing.py) using
the frozen artifact's de-vig/consensus methods. This wrapper delegates so no duplicate
consensus/de-vig math survives here.
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    print("NOTE: fetch_current_week_lines is deprecated; delegating to build_week1_2026_lines.")
    # forward the same CLI (--season/--week/--output) to the canonical builder
    sys.argv[0] = "build_week1_2026_lines.py"
    runpy.run_path(__file__.replace("fetch_current_week_lines.py", "build_week1_2026_lines.py"), run_name="__main__")


if __name__ == "__main__":
    main()
