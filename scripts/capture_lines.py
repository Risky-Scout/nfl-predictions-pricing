"""Own-book line capture: append a timestamped, book-attributed odds snapshot.

Each run polls the Odds API current-odds endpoint once (regions*markets = 3
credits, returns the whole slate across h2h/spreads/totals) and appends an
immutable snapshot to data/line_history/ with a SHA-256 manifest. Files are
named by capture time and never overwritten, so a season of self-captured
open-to-close movement accumulates -- the #1 ranked acquisition from
FINAL_REPORT, built for ~free.

Recommended cadence (per week, whole slate): Tue open / Fri / Sun T-60 / T-10.
That is 4 calls x 3 credits = 12 credits per week; a full 18-week season is
~216 credits for opening-to-closing movement on every game and market.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.io import write_frame
from nfl_hybrid.data.odds import add_market_consensus, devig_two_way_groups
from nfl_hybrid.data.provenance import SourceManifest, utc_now_iso
from nfl_hybrid.data.providers.the_odds_api import OddsAPIConfig, TheOddsAPIAdapter

HIST = Path("data/line_history")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="adhoc", help="Cadence label, e.g. tue_open / fri / sun_t60 / t10.")
    args = ap.parse_args()
    HIST.mkdir(parents=True, exist_ok=True)

    adapter = TheOddsAPIAdapter(OddsAPIConfig())
    result = adapter.current_odds()
    credits = result.metadata.get("x-requests-last")
    remaining = result.metadata.get("x-requests-remaining")
    odds = result.data
    if len(odds):
        odds = devig_two_way_groups(odds)
        odds = add_market_consensus(odds)
    odds["capture_label"] = args.label
    odds["captured_at_utc"] = utc_now_iso()

    stamp = re.sub(r"[^0-9]", "", utc_now_iso())[:14]
    stem = HIST / f"nfl_odds_{stamp}_{args.label}"
    if Path(f"{stem}.parquet").exists():
        raise FileExistsError(f"{stem}.parquet exists; captures are immutable.")
    path = write_frame(odds, stem)
    SourceManifest.from_frame(
        dataset_name=f"line_capture_{stamp}_{args.label}",
        source_name="the_odds_api",
        source_locator="sports/americanfootball_nfl/odds (current)",
        frame=odds,
        request_parameters={
            "capture_label": args.label,
            "credits_spent": credits,
            "credits_remaining_after": remaining,
        },
    ).write_json(HIST / f"nfl_odds_{stamp}_{args.label}.manifest.json")

    print(f"captured {len(odds)} quote rows across {odds['provider_event_id'].nunique()} events")
    print(f"credits spent: {credits} | remaining: {remaining}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
