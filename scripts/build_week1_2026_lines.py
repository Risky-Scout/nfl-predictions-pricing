"""Build the 2026 week-1 lines CSV from the real schedule + current Odds API lines.

Games with posted lines are priced from de-vigged consensus; games without are
flagged UNPRICED-AWAITING-LINES and never invented. Emits a predict_week-format
games CSV. One current-odds call = 3 credits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.data.odds import devig_two_way_groups, match_odds_to_games
from nfl_hybrid.data.providers.nflverse import NflverseAdapter
from nfl_hybrid.data.providers.the_odds_api import OddsAPIConfig, TheOddsAPIAdapter

OUT = Path("outputs/season_2026")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", default="1")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-fetch", action="store_true", help="Skip the paid API call (all UNPRICED).")
    args = ap.parse_args()
    SEASON, WEEK = args.season, str(args.week)
    OUT.mkdir(parents=True, exist_ok=True)
    output = args.output or str(OUT / f"lines_wk{WEEK}.csv")

    sched = NflverseAdapter().load_games([SEASON]).data
    wk1 = sched[sched["week"].astype(str) == str(WEEK)].copy()
    wk1["game_id"] = wk1["game_id"].astype(str)

    priced = pd.DataFrame()
    credits = 0
    if not args.no_fetch:
        result = TheOddsAPIAdapter(OddsAPIConfig()).current_odds()
        credits = int(result.metadata.get("x-requests-last") or 0)
        odds = result.data
        odds = odds[~odds["is_live"].astype(bool)]
        odds = odds[odds["market_type"].isin(["moneyline", "spread", "total"])].copy()
        odds["provider_event_id"] = odds["provider_event_id"].astype(str)
        matched = match_odds_to_games(odds, wk1)
        matched = matched[matched["game_match_status"].isin(["matched", "matched_nearest_ambiguous"])].copy()
        if len(matched):
            matched = devig_two_way_groups(matched)

            def cons(market, side, col):
                s = matched[(matched["market_type"] == market) & (matched["outcome_side"] == side)]
                return s.groupby("game_id")[col].median()

            priced = pd.DataFrame({
                "home_spread": cons("spread", "home", "line_value"),
                "total_line": cons("total", "over", "line_value"),
                "market_ml_home_probability": cons("moneyline", "home", "devig_probability"),
                "market_cover_probability": cons("spread", "home", "devig_probability"),
                "market_over_probability": cons("total", "over", "devig_probability"),
                "n_books": matched.groupby("game_id")["bookmaker_id"].nunique(),
            })

    rows = []
    for _, g in wk1.iterrows():
        gid = g["game_id"]
        base = dict(
            game_id=gid, season=SEASON, week=str(WEEK),
            home_team=g["home_team_id"], away_team=g["away_team_id"],
        )
        if gid in getattr(priced, "index", []):
            p = priced.loc[gid]
            if pd.notna(p["home_spread"]) and pd.notna(p["total_line"]):
                rows.append({**base, "home_spread": float(p["home_spread"]), "total_line": float(p["total_line"]),
                             "market_ml_home_probability": float(p["market_ml_home_probability"]),
                             "market_cover_probability": float(p.get("market_cover_probability", np.nan)),
                             "market_over_probability": float(p.get("market_over_probability", np.nan)),
                             "market_source": "LIVE-CURRENT", "status": "PRICED",
                             "n_books": int(p["n_books"])})
                continue
        rows.append({**base, "home_spread": np.nan, "total_line": np.nan,
                     "market_source": "UNPRICED-AWAITING-LINES", "status": "UNPRICED-AWAITING-LINES", "n_books": 0})

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    n_priced = int((frame["status"] == "PRICED").sum())
    print(f"week-1 games: {len(frame)} | priced: {n_priced} | awaiting lines: {len(frame) - n_priced}")
    print(f"current-odds credits spent: {credits}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
