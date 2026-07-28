from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.feature_manifest import (
    build_manifest_matrix,
    load_feature_manifest,
    validate_no_banned_features,
)
from nfl_hybrid.features.market_compact import (
    OA_METRICS,
    QB_METRICS,
    engineer_compact_market_features,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "features"



def _stage_two_fixture(rows: int = 8) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "game_id": [f"G{i}" for i in range(rows)],
        "season": [2020 + (i % 4) for i in range(rows)],
        "week": [i + 1 for i in range(rows)],
        "home_team_id": ["KC"] * rows,
        "away_team_id": ["BUF"] * rows,
        "home_score": 24 + (index % 6),
        "away_score": 20 + ((index * 2) % 7),
        "neutral_site": [False, False, True, False] * 2,
        "playoff": [False, False, False, True] * 2,
        "division_game": [False, True, False, False] * 2,
        "home_rest_days": 7 + (index % 3),
        "away_rest_days": 6 + ((index + 1) % 3),
        "home_bye_flag": [False, False, True, False] * 2,
        "away_bye_flag": [False, True, False, False] * 2,
        "roof": ["outdoors", "closed", "outdoors", "dome"] * 2,
        "home_team_prior_games": index + 1,
        "away_team_prior_games": index + 2,
        "home_spread_reference": np.array([-3.5, 2.5, -1.5, 4.0] * 2),
        "total_line_reference": 43.5 + (index % 5),
        "home_moneyline_reference": -150 - 5 * index,
        "away_moneyline_reference": 130 + 5 * index,
        "home_spread_price_reference": -110 + (index % 3),
        "away_spread_price_reference": -110 - (index % 3),
        "over_price_reference": -108 + (index % 2),
        "under_price_reference": -112 - (index % 2),
        "spread_reference_difference": (index % 3) * 0.5,
        "total_reference_difference": ((index + 1) % 3) * 0.5,
        "home_qb_max_starter_probability": 0.90 - 0.01 * index,
        "away_qb_max_starter_probability": 0.88 - 0.01 * index,
        "home_qb_starter_candidates": 1 + (index % 2),
        "away_qb_starter_candidates": 1 + ((index + 1) % 2),
        "home_qb_starter_entropy": 0.05 + 0.01 * index,
        "away_qb_starter_entropy": 0.04 + 0.01 * index,
    }

    for metric_number, metric in enumerate(OA_METRICS):
        base = 0.01 * (metric_number + 1)
        prefix = f"oa_{metric}"
        data[f"home_{prefix}_offense_mean"] = base + 0.002 * index
        data[f"away_{prefix}_offense_mean"] = base - 0.001 * index
        data[f"home_{prefix}_defense_allowed_mean"] = base + 0.0015 * index
        data[f"away_{prefix}_defense_allowed_mean"] = base - 0.0012 * index
        data[f"home_{prefix}_league_mean"] = np.full(rows, base)
        data[f"away_{prefix}_league_mean"] = np.full(rows, base)
        data[f"home_{prefix}_offense_sd"] = 0.10 + 0.001 * index
        data[f"away_{prefix}_offense_sd"] = 0.11 + 0.001 * index
        data[f"home_{prefix}_defense_sd"] = 0.12 + 0.001 * index
        data[f"away_{prefix}_defense_sd"] = 0.13 + 0.001 * index
        data[f"home_{prefix}_offense_reliability"] = 0.4 + 0.02 * index
        data[f"away_{prefix}_offense_reliability"] = 0.42 + 0.02 * index
        data[f"home_{prefix}_defense_reliability"] = 0.44 + 0.02 * index
        data[f"away_{prefix}_defense_reliability"] = 0.46 + 0.02 * index

    for metric_number, metric in enumerate(QB_METRICS):
        base = 0.02 * (metric_number + 1)
        data[f"home_qb_{metric}_mean"] = base + 0.002 * index
        data[f"away_qb_{metric}_mean"] = base - 0.001 * index
        data[f"home_qb_{metric}_sd"] = 0.15 + 0.001 * index
        data[f"away_qb_{metric}_sd"] = 0.16 + 0.001 * index
        data[f"home_qb_{metric}_reliability"] = 0.45 + 0.02 * index
        data[f"away_qb_{metric}_reliability"] = 0.47 + 0.02 * index
        data[f"home_qb_{metric}_delta_vs_team"] = 0.01 + 0.001 * index
        data[f"away_qb_{metric}_delta_vs_team"] = -0.01 - 0.001 * index

    for metric_number, metric in enumerate(
        (
            "red_zone_td_rate",
            "neutral_situation_pass_rate",
            "scrimmage_plays_per_drive",
        )
    ):
        base = 0.4 + 0.03 * metric_number
        data[f"home_offense_{metric}__ewm_hl4"] = base + 0.002 * index
        data[f"away_offense_{metric}__ewm_hl4"] = base - 0.001 * index
        data[f"home_defense_allowed_{metric}__ewm_hl4"] = (
            base + 0.0015 * index
        )
        data[f"away_defense_allowed_{metric}__ewm_hl4"] = (
            base - 0.0012 * index
        )
    return pd.DataFrame(data)


def test_banned_feature_gate_rejects_targets():
    with pytest.raises(ValueError, match="Banned"):
        validate_no_banned_features(["target_home_win"])


def test_compact_engineering_never_returns_source_warehouse_columns():
    engineered = engineer_compact_market_features(_stage_two_fixture())
    assert len(engineered) == 8
    assert "home_oa_epa_per_play_offense_mean" not in engineered.columns
    assert "matchup_epa_per_play_net" in engineered.columns
    assert "qb_epa_per_dropback_diff" in engineered.columns


@pytest.mark.parametrize(
    "manifest_name",
    (
        "pregame_moneyline.yaml",
        "pregame_ats.yaml",
        "pregame_total.yaml",
    ),
)
def test_manifests_build_compact_football_and_market_matrices(manifest_name):
    engineered = engineer_compact_market_features(_stage_two_fixture())
    manifest = load_feature_manifest(CONFIG_ROOT / manifest_name)

    football, football_features = build_manifest_matrix(
        engineered,
        manifest,
        include_market=False,
    )
    market, market_features = build_manifest_matrix(
        engineered,
        manifest,
        include_market=True,
    )

    assert len(football) == 8
    assert len(market) == 8
    assert len(football_features) <= manifest.max_football_features
    assert len(market_features) <= manifest.max_market_augmented_features
    assert len(market_features) < 60
    assert set(football_features).issubset(set(market_features))


def test_targets_are_derived_from_scores_and_exact_lines():
    engineered = engineer_compact_market_features(_stage_two_fixture())
    row = engineered.iloc[0]
    expected_margin = 24 - 20
    expected_total = 24 + 20
    assert row["target_home_margin"] == expected_margin
    assert row["target_total_points"] == expected_total
    assert row["target_margin_residual"] == pytest.approx(
        expected_margin - row["market_implied_margin"]
    )
    assert row["target_total_residual"] == pytest.approx(
        expected_total - row["market_total_line"]
    )
