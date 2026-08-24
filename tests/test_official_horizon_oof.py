import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation import official_horizon_oof as ohf
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.selection import feature_deduction_2026 as fd


def _synthetic_matrix() -> pd.DataFrame:
    rows = [
        # cutoff1 batch -- no prior games at all.
        {"game_id": "G1", "season": 2024, "week": 1, "target_cutoff_utc": "2024-09-03T12:00:00Z",
         "result_available_at_utc": "2024-09-08T01:00:00Z", "home_margin": 3.0, "total_points": 45.0},
        {"game_id": "G2", "season": 2024, "week": 1, "target_cutoff_utc": "2024-09-03T12:00:00Z",
         "result_available_at_utc": "2024-09-11T01:00:00Z", "home_margin": -7.0, "total_points": 52.0},
        # cutoff2 batch -- only G1's result is available before cutoff2 (G2's is not, it resolves after).
        {"game_id": "G3", "season": 2024, "week": 2, "target_cutoff_utc": "2024-09-10T12:00:00Z",
         "result_available_at_utc": "2024-09-16T01:00:00Z", "home_margin": 10.0, "total_points": 41.0},
        {"game_id": "G4", "season": 2024, "week": 2, "target_cutoff_utc": "2024-09-10T12:00:00Z",
         "result_available_at_utc": "2024-09-16T02:00:00Z", "home_margin": -2.0, "total_points": 47.0},
        # cutoff3 batch -- G1, G2, G3, G4 all resolved by now.
        {"game_id": "G5", "season": 2024, "week": 3, "target_cutoff_utc": "2024-09-17T12:00:00Z",
         "result_available_at_utc": "2024-09-23T01:00:00Z", "home_margin": 1.0, "total_points": 50.0},
        {"game_id": "G6", "season": 2024, "week": 3, "target_cutoff_utc": "2024-09-17T12:00:00Z",
         "result_available_at_utc": "2024-09-23T02:00:00Z", "home_margin": -5.0, "total_points": 38.0},
    ]
    frame = pd.DataFrame(rows)
    frame["target_cutoff_utc"] = pd.to_datetime(frame["target_cutoff_utc"], utc=True)
    frame["result_available_at_utc"] = pd.to_datetime(frame["result_available_at_utc"], utc=True)
    rng = np.random.default_rng(7)
    for col in ohf.ELO_FEATURE_COLUMNS:
        frame[col] = rng.normal(size=len(frame))
    return frame


def _tiny_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["H1_THU", "H2_SUN", "H3_SUN"],
            "season": [2024, 2024, 2024],
            "season_type": ["REG", "REG", "REG"],
            "week": [1, 1, 2],
            "home_team_id": ["BUF", "MIA", "NE"],
            "away_team_id": ["KC", "NE", "BUF"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2024-09-05T00:20:00Z", "2024-09-08T17:00:00Z", "2024-09-15T17:00:00Z"], utc=True
            ),
            "home_score": [20, 24, 27],
            "away_score": [17, 21, 24],
            "neutral_site": [False, False, False],
        }
    )


def test_future_result_excluded_from_training():
    matrix = _synthetic_matrix()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)
    cutoff2_rows = predictions[predictions["game_id"].isin(["G3", "G4"])]
    assert (cutoff2_rows["training_game_count"] == 1).all()
    assert set(cutoff2_rows.iloc[0]["training_game_ids"]) == {"G1"}


def test_strict_equality_at_cutoff_excluded():
    matrix = _synthetic_matrix()
    matrix = matrix.copy()
    matrix.loc[matrix["game_id"] == "G2", "result_available_at_utc"] = pd.Timestamp("2024-09-10T12:00:00Z")
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)
    cutoff2_rows = predictions[predictions["game_id"].isin(["G3", "G4"])]
    # G2's result_available_at_utc now EQUALS cutoff2 exactly -- must still be excluded (strict <).
    assert set(cutoff2_rows.iloc[0]["training_game_ids"]) == {"G1"}


def test_model_not_ready_below_min_training_games():
    matrix = _synthetic_matrix()
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE")  # default min_training_games=48
    assert (predictions["status"] == "MODEL_NOT_READY").all()
    assert predictions["predicted_margin"].isna().all()

    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1)
    predictions2, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)
    cutoff1_rows = predictions2[predictions2["game_id"].isin(["G1", "G2"])]
    assert (cutoff1_rows["status"] == "MODEL_NOT_READY").all()
    cutoff2_rows = predictions2[predictions2["game_id"].isin(["G3", "G4"])]
    assert (cutoff2_rows["status"] == "OOF").all()
    assert cutoff2_rows["predicted_margin"].notna().all()


def test_fit_once_per_shared_cutoff_batch():
    matrix = _synthetic_matrix()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1)
    predictions, fit_counts = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)
    n_ready_batches = predictions.loc[predictions["status"] == "OOF", "target_cutoff_utc"].nunique()
    assert n_ready_batches == 2  # cutoff2 and cutoff3
    assert fit_counts["paired_fits"] == n_ready_batches
    assert fit_counts["individual_fits"] == 2 * n_ready_batches


def test_target_never_trains_on_itself_structural_invariant():
    games = _tiny_games()
    ledger = he.build_horizon_membership_ledger(games)
    matrix = ohf.build_official_horizon_matrix(games, "TUE", ledger)
    # result_available_at_utc (kickoff + floor) is always strictly after
    # target_cutoff_utc (always strictly before kickoff) -- so a target can
    # never appear in its own training mask.
    assert (matrix["result_available_at_utc"] > matrix["target_cutoff_utc"]).all()


def test_no_cross_horizon_residual_pooling_order_independent():
    matrix_a = _synthetic_matrix()
    matrix_b = _synthetic_matrix().assign(
        home_margin=lambda d: d["home_margin"] + 100.0, total_points=lambda d: d["total_points"] + 100.0
    )
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)

    _, ledger_a1, _ = ohf.build_official_horizon_oof(matrix_a, horizon="TUE", config=cfg)
    _, ledger_b1, _ = ohf.build_official_horizon_oof(matrix_b, horizon="FRI", config=cfg)
    _, ledger_b2, _ = ohf.build_official_horizon_oof(matrix_b, horizon="FRI", config=cfg)
    _, ledger_a2, _ = ohf.build_official_horizon_oof(matrix_a, horizon="TUE", config=cfg)

    pd.testing.assert_series_equal(
        ledger_a1["margin_residual_sd_oof"].reset_index(drop=True),
        ledger_a2["margin_residual_sd_oof"].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        ledger_b1["margin_residual_sd_oof"].reset_index(drop=True),
        ledger_b2["margin_residual_sd_oof"].reset_index(drop=True),
    )
    for _, row in ledger_a1.iterrows():
        assert set(row["training_game_ids"]) <= set(ledger_a1["game_id"])
    for _, row in ledger_b1.iterrows():
        assert set(row["training_game_ids"]) <= set(ledger_b1["game_id"])


def test_uncertainty_warmup_gate():
    matrix = _synthetic_matrix()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=100)
    _, ledger, _ = ohf.build_official_horizon_oof(matrix, horizon="TUE", config=cfg)
    assert not ledger["uncertainty_eligible"].any()

    cfg2 = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    _, ledger2, _ = ohf.build_official_horizon_oof(matrix, horizon="TUE", config=cfg2)
    assert ledger2["uncertainty_eligible"].any()


def test_rho_clip_reused_from_fix3():
    matrix = _synthetic_matrix()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    _, ledger, _ = ohf.build_official_horizon_oof(matrix, horizon="TUE", config=cfg)
    corr = ledger["residual_correlation_oof"].dropna()
    assert (corr >= -0.95).all() and (corr <= 0.95).all()


def test_official_model_config_hash_deterministic():
    cfg = ohf.OfficialHorizonOOFConfig()
    h1 = ohf.compute_official_model_config_hash(cfg)
    h2 = ohf.compute_official_model_config_hash(cfg)
    assert h1 == h2
    cfg2 = ohf.OfficialHorizonOOFConfig(min_training_games=10)
    assert ohf.compute_official_model_config_hash(cfg2) != h1


def test_no_market_feature_leakage_in_frozen_six_features():
    fd.assert_no_forbidden_market_columns(list(ohf.ELO_FEATURE_COLUMNS))
    assert set(ohf.ELO_FEATURE_COLUMNS).isdisjoint(fd.FORBIDDEN_MARKET_COLUMNS)


def test_frozen_six_features_are_elo_only_home_away_pivoted():
    expected = (
        "home_elo_pregame_rating", "home_elo_pregame_win_probability", "home_elo_pregame_expected_margin",
        "away_elo_pregame_rating", "away_elo_pregame_win_probability", "away_elo_pregame_expected_margin",
    )
    assert ohf.ELO_FEATURE_COLUMNS == expected


def test_ridge_alpha_100_hyperparameters_match_frozen_spec():
    assert ohf.RIDGE_HYPERPARAMETERS == {"alpha": 100.0, "fit_intercept": True, "solver": "svd"}
    assert ohf.RIDGE_PREPROCESSING == {"type": "StandardScaler", "with_mean": True, "with_std": True}


def test_deterministic_hashes_and_availability_invariant_enforced():
    matrix = _synthetic_matrix()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions1, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)
    predictions2, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)
    pd.testing.assert_frame_equal(
        predictions1.drop(columns=["predicted_margin", "predicted_total"]),
        predictions2.drop(columns=["predicted_margin", "predicted_total"]),
    )
    np.testing.assert_allclose(predictions1["predicted_margin"].dropna(), predictions2["predicted_margin"].dropna())
