import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.selection.distributional_tournament import (
    DistributionalTournamentConfig,
    run_distributional_tournament,
)


def _write_matrix(
    root: Path,
    market: str,
    variant: str,
    frame: pd.DataFrame,
    features: list[str],
) -> None:
    frame.to_parquet(
        root / f"{market}_{variant}.parquet",
        index=False,
    )
    (root / f"{market}_{variant}.manifest.json").write_text(
        json.dumps({"features": features}),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def synthetic_tournament(tmp_path_factory):
    original_to_parquet = pd.DataFrame.to_parquet
    original_read_parquet = pd.read_parquet

    def _to_pickle_with_parquet_name(self, path, index=False, **kwargs):
        frame = self if index else self.reset_index(drop=True)
        frame.to_pickle(path)

    def _read_pickle_with_parquet_name(path, **kwargs):
        return pd.read_pickle(path)

    pd.DataFrame.to_parquet = _to_pickle_with_parquet_name
    pd.read_parquet = _read_pickle_with_parquet_name

    root = tmp_path_factory.mktemp("distributional")
    compact = root / "compact"
    output = root / "output"
    compact.mkdir()

    rng = np.random.default_rng(11)
    rows = []
    for season in range(2020, 2026):
        for game in range(120):
            context = rng.normal()
            strength = rng.normal()
            qb = rng.normal()
            market_margin = 2.0 * strength + 0.8 * qb
            home_spread = np.round(-market_margin * 2.0) / 2.0
            actual_margin = int(
                np.rint(
                    market_margin
                    + 0.6 * strength
                    + 0.2 * context
                    + rng.normal(0.0, 10.0)
                )
            )
            total_line = np.round(
                (44.0 + 0.5 * context) * 2.0
            ) / 2.0
            actual_total = int(
                np.rint(
                    total_line
                    + 0.8 * context
                    + 0.5 * qb
                    + rng.normal(0.0, 9.0)
                )
            )
            margin_residual = actual_margin + home_spread
            total_residual = actual_total - total_line
            ml_prob = 1.0 / (
                1.0 + np.exp(-(-home_spread) / 6.5)
            )
            rows.append(
                {
                    "game_id": f"{season}_{(game % 18) + 1:02d}_{game:03d}",
                    "season": season,
                    "home_team_id": "KC",
                    "away_team_id": "BUF",
                    "playoff_flag": float(game >= 110),
                    "matchup_epa_per_play_net": strength,
                    "qb_epa_per_dropback_diff": qb,
                    "target_home_margin": actual_margin,
                    "target_home_win": (
                        float(actual_margin > 0)
                        if actual_margin != 0
                        else np.nan
                    ),
                    "target_tie": float(actual_margin == 0),
                    "target_home_cover": float(margin_residual > 0),
                    "target_ats_push": float(
                        np.isclose(margin_residual, 0.0)
                    ),
                    "target_margin_residual": margin_residual,
                    "target_total_points": actual_total,
                    "target_over": float(total_residual > 0),
                    "target_total_push": float(
                        np.isclose(total_residual, 0.0)
                    ),
                    "target_total_residual": total_residual,
                    "market_home_spread": home_spread,
                    "market_implied_margin": -home_spread,
                    "market_home_ml_novig_prob": ml_prob,
                    "market_home_cover_novig_prob": 0.5,
                    "market_total_line": total_line,
                    "market_over_novig_prob": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    football_features = [
        "playoff_flag",
        "matchup_epa_per_play_net",
        "qb_epa_per_dropback_diff",
    ]

    ml_football = frame[
        [
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "target_home_win",
            "target_tie",
            "target_home_margin",
            *football_features,
        ]
    ].copy()
    ml_market_features = [
        *football_features,
        "market_home_ml_novig_prob",
        "market_implied_margin",
    ]
    ml_market = frame[
        [
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "target_home_win",
            "target_tie",
            "target_home_margin",
            *ml_market_features,
        ]
    ].copy()

    ats_football = frame[
        [
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "target_home_cover",
            "target_ats_push",
            "target_margin_residual",
            "target_home_margin",
            *football_features,
        ]
    ].copy()
    ats_market_features = [
        *football_features,
        "market_home_spread",
        "market_home_cover_novig_prob",
    ]
    ats_market = frame[
        [
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "target_home_cover",
            "target_ats_push",
            "target_margin_residual",
            "target_home_margin",
            *ats_market_features,
        ]
    ].copy()

    total_football = frame[
        [
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "target_over",
            "target_total_push",
            "target_total_residual",
            "target_total_points",
            *football_features,
        ]
    ].copy()
    total_market_features = [
        *football_features,
        "market_total_line",
        "market_over_novig_prob",
    ]
    total_market = frame[
        [
            "game_id",
            "season",
            "home_team_id",
            "away_team_id",
            "target_over",
            "target_total_push",
            "target_total_residual",
            "target_total_points",
            *total_market_features,
        ]
    ].copy()

    _write_matrix(
        compact,
        "pregame_moneyline",
        "football_only",
        ml_football,
        football_features,
    )
    _write_matrix(
        compact,
        "pregame_moneyline",
        "market_augmented",
        ml_market,
        ml_market_features,
    )
    _write_matrix(
        compact,
        "pregame_ats",
        "football_only",
        ats_football,
        football_features,
    )
    _write_matrix(
        compact,
        "pregame_ats",
        "market_augmented",
        ats_market,
        ats_market_features,
    )
    _write_matrix(
        compact,
        "pregame_total",
        "football_only",
        total_football,
        football_features,
    )
    _write_matrix(
        compact,
        "pregame_total",
        "market_augmented",
        total_market,
        total_market_features,
    )

    config = DistributionalTournamentConfig(
        ridge_alphas=(10.0,),
        offset_alphas=(0.1,),
        inner_time_splits=2,
        minimum_inner_train_rows=20,
        minimum_residual_samples=20,
        minimum_outer_train_rows=100,
        minimum_outer_test_rows=100,
        bootstrap_repetitions=20,
    )
    try:
        folds, aggregate, selected = run_distributional_tournament(
            compact,
            output,
            config=config,
        )
        yield output, folds, aggregate, selected
    finally:
        pd.DataFrame.to_parquet = original_to_parquet
        pd.read_parquet = original_read_parquet


def test_distributional_tournament_never_scores_2024_or_2025(
    synthetic_tournament,
):
    output, folds, _, selected = synthetic_tournament
    assert set(folds["test_season"]) == {2021, 2022, 2023}
    assert not folds["test_season"].isin([2024, 2025]).any()
    assert len(selected) == 6
    assert (output / "distributional_oof_predictions.parquet").exists()


def test_distributional_tournament_reports_three_way_metrics(
    synthetic_tournament,
):
    output, _, aggregate, _ = synthetic_tournament
    required = {
        "pooled_binary_log_loss",
        "pooled_binary_brier",
        "pooled_three_way_log_loss",
        "pooled_three_way_brier",
        "mean_predicted_push",
        "actual_push_rate",
    }
    assert required.issubset(aggregate.columns)
    bootstrap = pd.read_csv(output / "distributional_bootstrap.csv")
    assert set(bootstrap["bootstrap_method"]) == {
        "season_stratified_week_cluster"
    }
