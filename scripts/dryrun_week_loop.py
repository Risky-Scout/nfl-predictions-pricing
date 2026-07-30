"""Operations rehearsal: run the full weekly loop on 2025 week-1 REAL data.

This is a rehearsal only -- no selection decision is made from 2025. It proves the
machinery end to end: card -> immutable log -> resolve with real scores -> score
vs market -> cumulative scorecard -> live-promotion gate (which HOLDS at 1 week).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl_hybrid.operations import (
    cumulative_live_scorecard,
    live_promotion_decision,
    log_week_predictions,
    score_resolved_week,
)
from nfl_hybrid.pricing.betting_card import build_betting_card, readiness_from_production_spec

SEASON, WEEK = 2025, 1
REHEARSAL_DIR = Path("outputs/season_2026/rehearsal_2025wk1")


def main() -> None:
    games = pd.read_parquet("data/backfill_2020_2025/canonical/games.parquet")
    wk = games[(games["season"] == SEASON) & (games["week"].astype(str) == str(WEEK))].copy()
    wk["home_team"] = wk["home_team_id"]
    wk["away_team"] = wk["away_team_id"]
    wk["home_spread"] = pd.to_numeric(wk["home_spread_reference"], errors="coerce")
    wk["total_line"] = pd.to_numeric(wk["total_line_reference"], errors="coerce")
    wk = wk.dropna(subset=["home_spread", "total_line"])

    readiness = readiness_from_production_spec("config/production_model_spec.json")
    print("readiness (from frozen production spec):", readiness)

    # [3] card -- all markets RETAIN_BASELINE -> stake zero
    card = build_betting_card(wk, readiness_by_market=readiness)
    print(f"card rows={len(card)} recommended_bets={int(card['should_bet'].sum())}")

    # [4] immutable pre-kickoff log
    REHEARSAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = "2025-09-05T12:00:00Z"  # fixed rehearsal timestamp (would be datetime.now(UTC) live)
    path = log_week_predictions(
        card, output_dir=REHEARSAL_DIR, season=SEASON, week=WEEK, logged_at_utc=ts, overwrite=True
    )
    print(f"logged immutable record -> {path}")

    # [5] resolve with REAL 2025 wk1 scores, then score vs market
    wk["home_margin"] = pd.to_numeric(wk["home_score"], errors="coerce") - pd.to_numeric(wk["away_score"], errors="coerce")
    wk["total_points"] = pd.to_numeric(wk["home_score"], errors="coerce") + pd.to_numeric(wk["away_score"], errors="coerce")
    resolved = pd.DataFrame(
        {
            "game_id": wk["game_id"].astype(str),
            "home_win": (wk["home_margin"] > 0).astype(int),
            "home_cover": (wk["home_margin"] + wk["home_spread"] > 0).astype(int),
            "over": (wk["total_points"] > wk["total_line"]).astype(int),
        }
    )
    logged = pd.read_csv(path)
    logged["game_id"] = logged["game_id"].astype(str)
    weekly = score_resolved_week(logged, resolved)
    print("\n[5] weekly score (model == market baseline in production):")
    print(weekly.to_string(index=False))
    weekly.to_csv(REHEARSAL_DIR / "weekly_scores.csv", index=False)

    # cumulative scorecard + live gate (1 week -> HOLD)
    cum = cumulative_live_scorecard(weekly)
    decision = live_promotion_decision(cum, drift_alarm_markets=set())
    print("\n[6] cumulative live scorecard:")
    print(cum.to_string(index=False))
    print("\n[6] live-promotion decision (expected HOLD at week 1):")
    print(decision.to_string(index=False))
    cum.to_csv(REHEARSAL_DIR / "cumulative_scorecard.csv", index=False)
    decision.to_csv(REHEARSAL_DIR / "live_promotion_decision.csv", index=False)


if __name__ == "__main__":
    main()
