import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.selection.compact_tournament import (
    TournamentConfig,
    _candidate_feature_sets,
    _probability_metrics,
    run_compact_tournament,
)


def test_candidate_sets_never_include_unlisted_features():
    features = (
        "home_field_indicator",
        "matchup_epa_per_play_net",
        "qb_epa_per_dropback_diff",
        "market_home_ml_novig_prob",
    )
    candidates = _candidate_feature_sets(features)
    for selected in candidates.values():
        assert set(selected).issubset(features)


def test_probability_metrics_are_finite():
    metrics = _probability_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.8, 0.3, 0.7]),
        TournamentConfig(minimum_train_rows=2, minimum_test_rows=2),
    )
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["brier"])
    assert np.isfinite(metrics["ece"])



def test_tournament_never_uses_2024_or_2025():
    from nfl_hybrid.selection.compact_tournament import (
        DEFAULT_MARKET_SPECS,
        _evaluate_matrix,
    )

    rng = np.random.default_rng(4)
    rows = []
    for season in range(2020, 2026):
        for game in range(120):
            signal = rng.normal()
            margin = 3.0 * signal + rng.normal(0, 8)
            rows.append(
                {
                    "game_id": f"{season}_{game}",
                    "season": season,
                    "home_team_id": "KC",
                    "away_team_id": "BUF",
                    "home_field_indicator": 1.0,
                    "matchup_epa_per_play_net": signal,
                    "qb_epa_per_dropback_diff": signal,
                    "market_home_ml_novig_prob": 1 / (1 + np.exp(-0.25 * margin)),
                    "market_implied_margin": 0.0,
                    "target_home_win": float(margin > 0),
                    "target_tie": 0,
                    "target_home_margin": margin,
                }
            )
    data = pd.DataFrame(rows)
    features = (
        "home_field_indicator",
        "matchup_epa_per_play_net",
        "qb_epa_per_dropback_diff",
        "market_home_ml_novig_prob",
        "market_implied_margin",
    )

    folds = _evaluate_matrix(
        data,
        features,
        spec=DEFAULT_MARKET_SPECS["pregame_moneyline"],
        variant="market_augmented",
        config=TournamentConfig(
            minimum_train_rows=100,
            minimum_test_rows=100,
        ),
    )
    assert set(folds["test_season"]) == {2021, 2022, 2023}
    assert not folds["test_season"].isin([2024, 2025]).any()
