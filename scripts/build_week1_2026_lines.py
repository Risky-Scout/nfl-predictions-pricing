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
    ap.add_argument("--as-of-utc", default=None, help="Freshness reference / replay timestamp.")
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
        from nfl_hybrid.data.provenance import utc_now_iso
        from nfl_hybrid.pricing.artifact import load_pricing_artifact
        from nfl_hybrid.pricing.market_pricing import price_game_markets

        artifact = load_pricing_artifact()  # methods come from the frozen artifact
        as_of = args.as_of_utc or utc_now_iso()
        result = TheOddsAPIAdapter(OddsAPIConfig()).current_odds()
        credits = int(result.metadata.get("x-requests-last") or 0)
        odds = result.data
        odds = odds[~odds["is_live"].astype(bool)]
        odds = odds[odds["market_type"].isin(["moneyline", "spread", "total"])].copy()
        odds["provider_event_id"] = odds["provider_event_id"].astype(str)
        matched = match_odds_to_games(odds, wk1)
        matched = matched[matched["game_match_status"].isin(["matched", "matched_nearest_ambiguous"])].copy()

        rows_priced = []
        for gid, g in matched.groupby("game_id"):
            prices = price_game_markets(g, as_of_utc=as_of, artifact=artifact)
            ml, sp, tot = prices["moneyline"], prices["spread"], prices["total"]
            # signed reference lines at the selected reference point (deterministic)
            def signed_line(mkt, side):
                p = prices[mkt].reference_point
                if p is None:
                    return np.nan
                s = g[(g["market_type"] == mkt) & (g["outcome_side"] == side)
                      & np.isclose(pd.to_numeric(g["line_value"], errors="coerce").abs(), p, atol=1e-6)]
                return float(s["line_value"].median()) if len(s) else np.nan
            rows_priced.append({
                "game_id": str(gid),
                "home_spread": signed_line("spread", "home"),
                "total_line": signed_line("total", "over"),
                "market_ml_home_probability": ml.fair_probability,
                "market_cover_probability": sp.fair_probability,
                "market_over_probability": tot.fair_probability,
                "n_books": sp.audit.get("books_retained", 0),
                "books_rejected_stale": sp.audit.get("books_rejected_stale", 0),
                "quote_age_minutes": sp.audit.get("median_quote_age_min"),
                "consensus_dispersion": sp.audit.get("consensus_dispersion"),
                "market_status": sp.status,
                "devig_method": sp.audit.get("devig_method"),
                "consensus_method": sp.audit.get("consensus_method"),
            })
        priced = pd.DataFrame(rows_priced).set_index("game_id") if rows_priced else pd.DataFrame()

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
                             "n_books": int(p["n_books"]),
                             "books_rejected_stale": int(p.get("books_rejected_stale", 0)),
                             "quote_age_minutes": float(p.get("quote_age_minutes", np.nan)) if pd.notna(p.get("quote_age_minutes")) else np.nan,
                             "consensus_dispersion": float(p.get("consensus_dispersion", np.nan)),
                             "market_status": p.get("market_status"),
                             "devig_method": p.get("devig_method"),
                             "consensus_method": p.get("consensus_method")})
                continue
        rows.append({**base, "home_spread": np.nan, "total_line": np.nan,
                     "market_source": "UNPRICED-AWAITING-LINES", "status": "UNPRICED-AWAITING-LINES", "n_books": 0,
                     "books_rejected_stale": 0, "quote_age_minutes": np.nan, "consensus_dispersion": np.nan,
                     "market_status": "NO_PRICE", "devig_method": None, "consensus_method": None})

    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    n_priced = int((frame["status"] == "PRICED").sum())
    print(f"week-1 games: {len(frame)} | priced: {n_priced} | awaiting lines: {len(frame) - n_priced}")
    print(f"current-odds credits spent: {credits}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
