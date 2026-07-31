"""DEPRECATED thin wrapper. Superseded by scripts/build_week1_2026_lines.py, which
routes through the ONE canonical market-pricing path (pricing/market_pricing.py) using
the frozen artifact's de-vig/consensus methods. This wrapper delegates via a normal
module import so no duplicate consensus/de-vig math survives here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_builder():
    path = Path(__file__).with_name("build_week1_2026_lines.py")
    spec = importlib.util.spec_from_file_location("build_week1_2026_lines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    print("NOTE: fetch_current_week_lines is deprecated; delegating to build_week1_2026_lines.")
    _load_builder().main()


if __name__ == "__main__":
    main()
