"""Build the REAL-CLOSING exact-point market benchmark for dev seasons 2022-2024.

Thin I/O wrapper: read odds + games, delegate to the ONE authoritative
implementation in ``nfl_hybrid.markets.exact_contract`` (exact-point consensus --
each selected spread/total is an actual quoted point and its probability is the
median de-vigged probability at *that exact point*), validate, and write outputs.

There is deliberately no consensus logic in this script; see
``exact_contract.build_real_closing_benchmark``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_hybrid.markets.exact_contract import (
    build_real_closing_benchmark,
    validate_benchmark,
)

BASE = Path("data/backfill_2020_2025")
OUT = Path("outputs")
DEV_SEASONS = (2022, 2023, 2024)


def main() -> None:
    odds = pd.read_parquet(BASE / "canonical" / "odds_closing_dev_2022_2024.parquet")
    games = pd.read_parquet(BASE / "canonical" / "games.parquet")
    dev = games[games["season"].isin(DEV_SEASONS)][
        ["game_id", "season", "week", "home_team_id", "away_team_id"]
    ].copy()

    result = build_real_closing_benchmark(odds, dev)
    bench = result.benchmark
    validate_benchmark(bench)  # fail closed before writing anything

    OUT.mkdir(exist_ok=True)
    bench.to_parquet(OUT / "real_closing_benchmark_2022_2024.parquet", index=False)

    by_season = {int(k): int(v) for k, v in bench.groupby("season").size().to_dict().items()}
    summary = {
        "aggregation_method": "exact_point_consensus_v1",
        "market_source": "REAL-CLOSING",
        "dev_games_total": int(len(dev)),
        "benchmark_input_row_count": int(result.input_row_count),
        "games_with_closing_benchmark": int(len(bench)),
        "coverage_fraction": round(len(bench) / len(dev), 4) if len(dev) else 0.0,
        "by_season": by_season,
        "median_closing_minutes_to_kickoff": round(
            float(bench["closing_minutes_to_kickoff"].median()), 1
        ),
        "median_spread_consensus_books": int(bench["spread_consensus_books"].median()),
        "median_total_consensus_books": int(bench["total_consensus_books"].median()),
        "median_spread_candidate_point_count": int(
            bench["spread_candidate_point_count"].median()
        ),
        "median_total_candidate_point_count": int(
            bench["total_candidate_point_count"].median()
        ),
        "mean_ml_home_prob": round(float(bench["market_ml_home_probability"].mean()), 4),
        "mean_cover_home_prob": round(float(bench["market_cover_home_probability"].mean()), 4),
        "mean_over_prob": round(float(bench["market_over_probability"].mean()), 4),
    }
    (OUT / "real_closing_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
