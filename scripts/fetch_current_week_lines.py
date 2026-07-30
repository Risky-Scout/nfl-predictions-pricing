"""Fetch current-week NFL lines via The Odds API current-odds endpoint.

Emits a games CSV in the predict_week input schema (game_id, season, week,
home_team, away_team, home_spread, total_line, market_source + de-vigged market
probabilities). current-odds costs regions*markets = 3 credits per call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.data.odds import add_market_consensus, devig_two_way_groups
from nfl_hybrid.data.providers.the_odds_api import OddsAPIConfig, TheOddsAPIAdapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    adapter = TheOddsAPIAdapter(OddsAPIConfig())
    result = adapter.current_odds()
    print(f"current-odds cost (credits): {result.metadata.get('x-requests-last')}")
    odds = result.data
    odds = odds[~odds["is_live"].astype(bool)]
    odds = odds[odds["market_type"].isin(["moneyline", "spread", "total"])].copy()
    odds["provider_event_id"] = odds["provider_event_id"].astype(str)
    odds = devig_two_way_groups(odds)

    def consensus(market, side, col):
        s = odds[(odds["market_type"] == market) & (odds["outcome_side"] == side)]
        return s.groupby("provider_event_id")[col].median()

    ev = odds.groupby("provider_event_id").agg(
        home_team=("home_team_id", "first"), away_team=("away_team_id", "first"),
    )
    ev["home_spread"] = consensus("spread", "home", "line_value")
    ev["total_line"] = consensus("total", "over", "line_value")
    ev["market_ml_home_probability"] = consensus("moneyline", "home", "devig_probability")
    ev["market_cover_probability"] = consensus("spread", "home", "devig_probability")
    ev["market_over_probability"] = consensus("total", "over", "devig_probability")
    ev = ev.reset_index(drop=True)
    ev["season"] = args.season
    ev["week"] = str(args.week)
    ev["game_id"] = [f"{args.season}_wk{args.week}_{a}_{h}" for a, h in zip(ev["away_team"], ev["home_team"])]
    ev["market_source"] = "LIVE-CURRENT"
    ev = ev.dropna(subset=["home_spread", "total_line"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(args.output, index=False)
    print(f"wrote {len(ev)} games -> {args.output}")


if __name__ == "__main__":
    main()
