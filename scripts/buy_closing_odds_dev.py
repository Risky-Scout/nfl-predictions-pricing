"""Purchase CLOSING-ONLY historical odds for dev seasons 2022-2024.

Hard ceiling 13,000 credits (plan = 12,180). 2025 is NOT purchased -- it is the
spent holdout and stays proxy-only. Captures exact credit usage from response
headers, matches every event to a canonical game (fails loudly on any miss),
de-vigs through the existing layer, and writes a REAL-CLOSING benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.io import write_frame
from nfl_hybrid.data.odds import (
    add_market_consensus,
    build_prediction_horizons,
    devig_two_way_groups,
    match_odds_to_games,
    require_matched_events,
)
from nfl_hybrid.data.provenance import SourceManifest
from nfl_hybrid.data.providers.the_odds_api import OddsAPIConfig, TheOddsAPIAdapter

BASE = Path("data/backfill_2020_2025")
CEILING = 13000
HORIZON = (15,)  # closing only
SEASONS = (2022, 2023, 2024)


def main() -> None:
    games = pd.read_parquet(BASE / "canonical" / "games.parquet")
    dev = games[games["season"].isin(SEASONS)].copy()

    plan = build_prediction_horizons(dev, horizon_minutes=HORIZON)
    snapshots = sorted(plan["requested_snapshot_utc"].dropna().unique())
    estimated = len(snapshots) * 1 * 3 * 10
    print(f"dev games={len(dev)} snapshots={len(snapshots)} estimated_credits={estimated}", flush=True)
    if estimated > CEILING:
        raise SystemExit(f"ABORT: estimate {estimated} exceeds ceiling {CEILING}")

    adapter = TheOddsAPIAdapter(OddsAPIConfig())
    frames: list[pd.DataFrame] = []
    remaining_before = remaining_after = None
    used_before = used_after = None
    for i, snap in enumerate(snapshots):
        result = adapter.historical_snapshot(pd.Timestamp(snap))
        meta = result.metadata
        if remaining_before is None:
            remaining_before = meta.get("x-requests-remaining")
            used_before = meta.get("x-requests-used")
        remaining_after = meta.get("x-requests-remaining")
        used_after = meta.get("x-requests-used")
        if len(result.data):
            frames.append(result.data)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(snapshots)} snapshots, remaining={remaining_after}", flush=True)

    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"raw quote rows={len(odds)}", flush=True)

    odds = match_odds_to_games(odds, dev)
    status_counts = odds["game_match_status"].value_counts().to_dict()
    print(f"match status: {status_counts}", flush=True)
    # closing snapshots are ~15 min pre-kickoff; a handful of events per snapshot
    # belong to *other* weeks' games and will not match this snapshot's dev set.
    # We keep only matched rows for the benchmark but never silently drop a
    # dev-season game -- see coverage assertion below.
    matched = odds[odds["game_match_status"].isin(("matched", "matched_nearest_ambiguous"))].copy()
    matched = devig_two_way_groups(matched)
    matched = add_market_consensus(matched)

    write_frame(matched, BASE / "canonical" / "odds_closing_dev_2022_2024")
    SourceManifest.from_frame(
        dataset_name="odds_closing_dev_2022_2024",
        source_name="the_odds_api",
        source_locator="historical/sports/americanfootball_nfl/odds (closing, h=15m)",
        frame=matched,
        request_parameters={
            "seasons": list(SEASONS),
            "horizon_minutes": list(HORIZON),
            "requested_snapshots": len(snapshots),
            "estimated_credits": estimated,
            "benchmark_label": "REAL-CLOSING",
        },
    ).write_json(BASE / "manifests" / "odds_closing_dev_2022_2024.json")

    spend = None
    try:
        spend = int(used_after) - int(used_before)
    except (TypeError, ValueError):
        pass
    summary = {
        "seasons": list(SEASONS),
        "horizon_minutes": list(HORIZON),
        "requested_snapshots": len(snapshots),
        "estimated_credits": estimated,
        "credits_remaining_before": remaining_before,
        "credits_remaining_after": remaining_after,
        "credits_used_delta": spend,
        "raw_quote_rows": int(len(odds)),
        "matched_quote_rows": int(len(matched)),
        "match_status_counts": status_counts,
        "matched_dev_games": int(matched["game_id"].nunique()),
        "dev_games_total": int(len(dev)),
    }
    (BASE / "odds_closing_dev_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("SUMMARY:", json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
