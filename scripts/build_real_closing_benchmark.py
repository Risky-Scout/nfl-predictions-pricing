"""Build the REAL-CLOSING de-vigged market benchmark for dev seasons 2022-2024.

For each dev game, select its closing snapshot (smallest positive
minutes-to-kickoff), then take the cross-book consensus (median) de-vigged
probability per market, plus the consensus closing spread/total lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("data/backfill_2020_2025")
OUT = Path("outputs")


def main() -> None:
    o = pd.read_parquet(BASE / "canonical" / "odds_closing_dev_2022_2024.parquet")
    o = o[o["market_type"].isin(["moneyline", "spread", "total"])].copy()
    o = o[o["minutes_to_kickoff"].astype(float) > 0]  # exclude in-play

    # closing snapshot per game = smallest positive minutes-to-kickoff
    o["mtk"] = o["minutes_to_kickoff"].astype(float)
    closing_snap = o.groupby("game_id")["mtk"].transform("min")
    closing = o[o["mtk"] == closing_snap].copy()

    def consensus_prob(market, side):
        sub = closing[(closing["market_type"] == market) & (closing["outcome_side"] == side)]
        return sub.groupby("game_id")["devig_probability"].median()

    def consensus_line(market, side):
        sub = closing[(closing["market_type"] == market) & (closing["outcome_side"] == side)]
        return sub.groupby("game_id")["line_value"].median()

    games = pd.read_parquet(BASE / "canonical" / "games.parquet")
    dev = games[games["season"].isin([2022, 2023, 2024])][
        ["game_id", "season", "week", "home_team_id", "away_team_id"]
    ].copy()

    bench = dev.set_index("game_id")
    bench["market_ml_home_probability"] = consensus_prob("moneyline", "home")
    bench["market_cover_home_probability"] = consensus_prob("spread", "home")
    bench["market_over_probability"] = consensus_prob("total", "over")
    bench["closing_home_spread"] = consensus_line("spread", "home")
    bench["closing_total_line"] = consensus_line("total", "over")
    bench["closing_minutes_to_kickoff"] = closing.groupby("game_id")["mtk"].median()
    bench["closing_books"] = closing.groupby("game_id")["bookmaker_id"].nunique()
    bench["market_source"] = "REAL-CLOSING"
    bench = bench.reset_index()

    complete = bench.dropna(
        subset=[
            "market_ml_home_probability",
            "market_cover_home_probability",
            "market_over_probability",
            "closing_home_spread",
            "closing_total_line",
        ]
    )
    OUT.mkdir(exist_ok=True)
    complete.to_parquet(OUT / "real_closing_benchmark_2022_2024.parquet", index=False)

    summary = {
        "dev_games_total": int(len(dev)),
        "games_with_closing_benchmark": int(len(complete)),
        "coverage_fraction": round(len(complete) / len(dev), 4),
        "median_closing_minutes_to_kickoff": round(float(complete["closing_minutes_to_kickoff"].median()), 1),
        "median_closing_books": int(complete["closing_books"].median()),
        "by_season": {int(k): int(v) for k, v in complete.groupby("season").size().to_dict().items()},
        "mean_ml_home_prob": round(float(complete["market_ml_home_probability"].mean()), 4),
        "mean_cover_home_prob": round(float(complete["market_cover_home_probability"].mean()), 4),
        "mean_over_prob": round(float(complete["market_over_probability"].mean()), 4),
    }
    (OUT / "real_closing_benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
