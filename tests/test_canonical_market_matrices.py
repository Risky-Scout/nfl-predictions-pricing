import json

import pandas as pd
import pytest

from nfl_hybrid.features.canonical_market_matrices import (
    canonicalize,
    load_roles,
)


@pytest.fixture
def roles(tmp_path):
    payload = {
        "pregame_moneyline": {
            "CANONICAL_ANCHOR": ["market_t10_novig_probability"],
            "CHALLENGER_FEATURE": [
                "market_probability_movement",
                "market_t10_probability_sd",
                "market_opening_horizon_minutes",
            ],
            "QUALITY_GATE": [
                "market_t10_eligible_books",
                "market_t10_hold",
                "market_t10_snapshot_lag_minutes",
            ],
            "AUDIT_ONLY": [
                "market_t10_requested_snapshot_utc",
                "market_t10_returned_snapshot_utc",
                "market_opening_available",
                "market_opening_missing",
                "market_opening_novig_probability",
                "market_opening_line",
            ],
            "PROHIBITED": [
                "market_home_ml_novig_prob",
                "market_home_ml_raw_prob",
                "market_moneyline_hold",
                "market_implied_margin",
                "market_total_line",
                "market_implied_home_points",
                "market_implied_away_points",
                "market_spread_source_disagreement",
                "market_total_source_disagreement",
                "market_t10_consensus_line",
                "market_t10_line_sd",
                "market_line_movement",
            ],
        },
        "pregame_ats": {
            "CANONICAL_ANCHOR": [
                "market_t10_consensus_line",
                "market_t10_novig_probability",
            ],
            "CHALLENGER_FEATURE": [
                "market_line_movement",
                "market_probability_movement",
                "market_t10_line_sd",
                "market_t10_probability_sd",
                "market_opening_horizon_minutes",
                "ats_t10_spread_magnitude",
                "ats_t10_spread_distance_to_3",
                "ats_t10_spread_distance_to_7",
                "ats_t10_spread_integer_flag",
                "ats_t10_spread_half_point_flag",
                "ats_t10_home_favorite_flag",
            ],
            "QUALITY_GATE": [
                "market_t10_eligible_books",
                "market_t10_hold",
                "market_t10_snapshot_lag_minutes",
            ],
            "AUDIT_ONLY": [
                "market_t10_requested_snapshot_utc",
                "market_t10_returned_snapshot_utc",
                "market_opening_available",
                "market_opening_missing",
                "market_opening_novig_probability",
                "market_opening_line",
            ],
            "PROHIBITED": [
                "market_home_spread",
                "market_home_cover_novig_prob",
                "market_spread_hold",
                "market_total_line",
                "market_implied_margin",
                "spread_magnitude",
                "spread_distance_to_3",
                "spread_distance_to_7",
                "spread_integer_flag",
                "spread_half_point_flag",
                "home_favorite_flag",
                "market_spread_source_disagreement",
                "market_total_source_disagreement",
            ],
        },
        "pregame_total": {
            "CANONICAL_ANCHOR": [
                "market_t10_consensus_line",
                "market_t10_novig_probability",
            ],
            "CHALLENGER_FEATURE": [
                "market_line_movement",
                "market_probability_movement",
                "market_t10_line_sd",
                "market_t10_probability_sd",
                "market_opening_horizon_minutes",
            ],
            "QUALITY_GATE": [
                "market_t10_eligible_books",
                "market_t10_hold",
                "market_t10_snapshot_lag_minutes",
            ],
            "AUDIT_ONLY": [
                "market_t10_requested_snapshot_utc",
                "market_t10_returned_snapshot_utc",
                "market_opening_available",
                "market_opening_missing",
                "market_opening_novig_probability",
                "market_opening_line",
            ],
            "PROHIBITED": [
                "market_total_line",
                "market_over_novig_prob",
                "market_total_hold",
                "market_implied_home_points",
                "market_implied_away_points",
                "market_implied_margin",
                "market_home_spread",
                "market_total_source_disagreement",
                "market_spread_source_disagreement",
            ],
        },
    }
    path = tmp_path / "roles.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_roles(path)


def _frame():
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "season": [2021, 2021],
            "target_home_margin": [3.0, -3.0],
            "football_feature": [0.1, -0.2],
            "market_home_spread": [-3.0, 2.5],
            "market_home_cover_novig_prob": [0.51, 0.49],
            "market_spread_hold": [0.04, 0.05],
            "market_total_line": [47.5, 44.0],
            "market_implied_margin": [3.0, -2.5],
            "spread_magnitude": [3.0, 2.5],
            "spread_distance_to_3": [0.0, 0.5],
            "spread_distance_to_7": [4.0, 4.5],
            "spread_integer_flag": [1, 0],
            "spread_half_point_flag": [0, 1],
            "home_favorite_flag": [1, 0],
            "market_spread_source_disagreement": [0.0, 0.1],
            "market_total_source_disagreement": [0.0, 0.2],
            "market_t10_consensus_line": [-3.5, 3.0],
            "market_t10_novig_probability": [0.52, 0.48],
            "market_line_movement": [-0.5, 0.5],
            "market_probability_movement": [0.01, -0.01],
            "market_t10_line_sd": [0.2, 0.3],
            "market_t10_probability_sd": [0.01, 0.02],
            "market_opening_horizon_minutes": [10080.0, 4320.0],
            "market_t10_eligible_books": [12, 14],
            "market_t10_hold": [0.04, 0.05],
            "market_t10_snapshot_lag_minutes": [5.0, 4.0],
            "market_t10_requested_snapshot_utc": ["a", "b"],
            "market_t10_returned_snapshot_utc": ["c", "d"],
            "market_opening_available": [1, 1],
            "market_opening_missing": [0, 0],
            "market_opening_novig_probability": [0.51, 0.49],
            "market_opening_line": [-3.0, 2.5],
        }
    )


def _manifest():
    return {
        "features": [
            "football_feature",
            "market_home_spread",
            "market_home_cover_novig_prob",
            "market_spread_hold",
            "market_total_line",
            "market_implied_margin",
            "spread_magnitude",
            "spread_distance_to_3",
            "spread_distance_to_7",
            "spread_integer_flag",
            "spread_half_point_flag",
            "home_favorite_flag",
            "market_spread_source_disagreement",
            "market_total_source_disagreement",
            "market_t10_novig_probability",
            "market_t10_hold",
            "market_t10_eligible_books",
            "market_t10_probability_sd",
            "market_opening_horizon_minutes",
            "market_probability_movement",
            "market_t10_snapshot_lag_minutes",
            "market_t10_consensus_line",
            "market_t10_line_sd",
            "market_opening_line",
            "market_line_movement",
        ]
    }


def test_legacy_fields_removed_and_t10_anchors_retained(roles):
    _, manifest, audit = canonicalize(
        _frame(), _manifest(), market="pregame_ats", roles=roles
    )
    features = set(manifest["features"])
    assert "market_home_spread" not in features
    assert "market_home_cover_novig_prob" not in features
    assert "market_t10_consensus_line" in features
    assert "market_t10_novig_probability" in features
    assert audit["blocked_features_in_estimator"] == 0


def test_quality_gates_removed(roles):
    _, manifest, _ = canonicalize(
        _frame(), _manifest(), market="pregame_ats", roles=roles
    )
    features = set(manifest["features"])
    assert "market_t10_eligible_books" not in features
    assert "market_t10_hold" not in features
    assert "market_t10_snapshot_lag_minutes" not in features


def test_ats_features_recomputed_from_t10_line(roles):
    frame, manifest, _ = canonicalize(
        _frame(), _manifest(), market="pregame_ats", roles=roles
    )
    assert frame.iloc[0]["ats_t10_spread_magnitude"] == pytest.approx(3.5)
    assert frame.iloc[0]["ats_t10_spread_half_point_flag"] == 1
    assert frame.iloc[0]["ats_t10_home_favorite_flag"] == 1
    assert "ats_t10_spread_magnitude" in manifest["features"]


def test_unknown_market_field_rejected(roles):
    frame = _frame()
    frame["market_unknown_leak"] = 1.0
    manifest = _manifest()
    manifest["features"].append("market_unknown_leak")
    with pytest.raises(ValueError, match="unclassified"):
        canonicalize(
            frame, manifest, market="pregame_ats", roles=roles
        )


def test_duplicate_games_rejected(roles):
    frame = _frame()
    frame.loc[1, "game_id"] = "G1"
    with pytest.raises(ValueError, match="duplicate game_id"):
        canonicalize(
            frame, _manifest(), market="pregame_ats", roles=roles
        )


def test_missing_anchor_rejected(roles):
    frame = _frame().drop(columns=["market_t10_novig_probability"])
    manifest = _manifest()
    manifest["features"] = [
        name
        for name in manifest["features"]
        if name != "market_t10_novig_probability"
    ]
    with pytest.raises(ValueError, match="missing canonical fields"):
        canonicalize(
            frame, manifest, market="pregame_ats", roles=roles
        )


def test_ats_targets_are_derived_from_t10_line():
    from nfl_hybrid.features.canonical_market_matrices import (
        _add_t10_targets,
    )

    frame = pd.DataFrame(
        {
            "target_home_margin": [3, 3, -3],
            "market_t10_consensus_line": [-3.5, -3.0, 3.0],
        }
    )

    result = _add_t10_targets(frame, "pregame_ats")

    assert result["target_t10_home_cover"].tolist() == [0, 0, 0]
    assert result["target_t10_ats_push"].tolist() == [0, 1, 1]
    assert result["target_t10_margin_residual"].tolist() == [-0.5, 0.0, 0.0]


def test_total_targets_are_derived_from_t10_line():
    from nfl_hybrid.features.canonical_market_matrices import (
        _add_t10_targets,
    )

    frame = pd.DataFrame(
        {
            "target_total_points": [47, 48, 47],
            "market_t10_consensus_line": [47.5, 47.0, 47.0],
        }
    )

    result = _add_t10_targets(frame, "pregame_total")

    assert result["target_t10_over"].tolist() == [0, 1, 0]
    assert result["target_t10_total_push"].tolist() == [0, 0, 1]
    assert result["target_t10_total_residual"].tolist() == [-0.5, 1.0, 0.0]
