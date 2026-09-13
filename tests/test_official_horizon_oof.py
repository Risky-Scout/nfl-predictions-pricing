import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation import official_horizon_oof as ohf
from nfl_hybrid.evaluation.chronological_oof import training_membership_hash
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


def _matrix_with_unresolved_rows() -> pd.DataFrame:
    """``_synthetic_matrix`` plus two SCHEDULED-but-unplayed games, shaped
    exactly like the composite 2026 population: a projected
    ``result_available_at_utc`` that precedes a later card's cutoff, and no
    observed outcome at all. Scores are absent, never imputed."""
    frame = _synthetic_matrix()
    unresolved = pd.DataFrame(
        [
            # Resolves (projected) before cutoff3 -- so a purely temporal mask
            # would sweep it into cutoff3's Ridge fit with a NaN label.
            {"game_id": "U1", "season": 2024, "week": 2, "target_cutoff_utc": "2024-09-10T12:00:00Z",
             "result_available_at_utc": "2024-09-16T03:00:00Z",
             "home_margin": np.nan, "total_points": np.nan},
            {"game_id": "U2", "season": 2024, "week": 2, "target_cutoff_utc": "2024-09-10T12:00:00Z",
             "result_available_at_utc": "2024-09-16T04:00:00Z",
             "home_margin": np.nan, "total_points": np.nan},
        ]
    )
    unresolved["target_cutoff_utc"] = pd.to_datetime(unresolved["target_cutoff_utc"], utc=True)
    unresolved["result_available_at_utc"] = pd.to_datetime(unresolved["result_available_at_utc"], utc=True)
    rng = np.random.default_rng(19)
    for col in ohf.ELO_FEATURE_COLUMNS:
        unresolved[col] = rng.normal(size=len(unresolved))
    return pd.concat([frame, unresolved], ignore_index=True)


def _composite_population(*, completed_weeks: int = 15, scheduled_weeks: int = 6) -> pd.DataFrame:
    """A population shaped like the live one: resolved history followed by
    scheduled 2026 games carrying no score."""
    teams = ["BUF", "MIA", "NE", "NYJ", "KC", "LAC", "DEN", "LV"]
    rng = np.random.default_rng(11)
    rows, n = [], 0
    for week in range(1, completed_weeks + scheduled_weeks + 1):
        monday = pd.Timestamp("2026-09-07") + pd.Timedelta(weeks=week - 1)
        resolved = week <= completed_weeks
        for pair in range(4):
            kickoff = monday + pd.Timedelta(days=6, hours=17)
            rows.append({
                "game_id": f"2026_{week:02d}_{teams[pair * 2 + 1]}_{teams[pair * 2]}_{n}",
                "season": 2026, "week": week, "season_type": "REG",
                "home_team_id": teams[pair * 2], "away_team_id": teams[pair * 2 + 1],
                "scheduled_kickoff_utc": kickoff.tz_localize("UTC"),
                "home_score": int(rng.integers(10, 35)) if resolved else np.nan,
                "away_score": int(rng.integers(10, 35)) if resolved else np.nan,
                "neutral_site": False,
            })
            n += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regression: an unresolved game must never supply a TRAINING LABEL.
#
# The composite 2026 population is the first estate where "this game's result
# is available" and "this game has an observed outcome" diverge: a scheduled,
# unplayed game has a projected result_available_at_utc (kickoff + the frozen
# settlement offset) that can precede a later card's cutoff while its score is
# still absent. A purely temporal training mask therefore handed NaN labels to
# Ridge ("Input y contains NaN"), and NaN residuals to the uncertainty pool.
# ---------------------------------------------------------------------------
def test_unresolved_scheduled_rows_never_enter_training():
    matrix = _matrix_with_unresolved_rows()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)

    # cutoff3's temporal window contains U1 and U2, which have no outcome.
    cutoff3 = predictions[predictions["game_id"].isin(["G5", "G6"])]
    assert len(cutoff3) == 2
    for ids in cutoff3["training_game_ids"]:
        assert {"U1", "U2"}.isdisjoint(set(ids))


def test_resolved_prior_games_still_enter_training():
    """The other half of the contract: the fix must not shrink a legitimate
    training set."""
    matrix = _matrix_with_unresolved_rows()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)

    cutoff3 = predictions[predictions["game_id"].isin(["G5", "G6"])]
    assert set(cutoff3.iloc[0]["training_game_ids"]) == {"G1", "G2", "G3", "G4"}
    assert (cutoff3["training_game_count"] == 4).all()


def test_unresolved_rows_remain_prediction_targets():
    """An unplayed game is exactly what production needs a forecast FOR. It
    must still be predicted -- it simply must not vote in the fit."""
    matrix = _matrix_with_unresolved_rows()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)

    unresolved = predictions[predictions["game_id"].isin(["U1", "U2"])]
    assert len(unresolved) == 2
    assert (unresolved["status"] == "OOF").all()
    assert unresolved["predicted_margin"].notna().all()
    assert unresolved["predicted_total"].notna().all()


def test_training_membership_describes_only_labeled_rows():
    """training_game_count / training_game_ids / training_membership_hash must
    describe the rows actually in the fit, not unresolved scheduled rows."""
    matrix = _matrix_with_unresolved_rows()
    labeled = set(
        matrix.loc[matrix["home_margin"].notna() & matrix["total_points"].notna(), "game_id"].astype(str)
    )
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)

    for _, row in predictions.iterrows():
        ids = set(row["training_game_ids"])
        assert ids <= labeled
        if not row["training_game_ids_truncated"]:
            assert row["training_game_count"] == len(ids)
        assert row["training_membership_hash"] == training_membership_hash(sorted(ids))


def test_no_nan_reaches_either_ridge_target_vector():
    """The exact production failure, end to end from a composite population:
    ``ValueError: Input y contains NaN``."""
    games = _composite_population()
    ledger = he.build_horizon_membership_ledger(games)
    matrix = ohf.build_official_horizon_matrix(games, "TUE", ledger)
    assert matrix["home_margin"].isna().any(), "fixture must contain unresolved games"

    predictions, residual_ledger, _ = ohf.build_official_horizon_oof(
        matrix, horizon="TUE", feature_state_hash="f" * 64
    )

    labeled = set(
        matrix.loc[matrix["home_margin"].notna() & matrix["total_points"].notna(), "game_id"].astype(str)
    )
    for ids in predictions["training_game_ids"]:
        assert set(ids) <= labeled
    # Every unresolved game is still carried as a target row.
    assert len(predictions) == len(matrix)
    assert len(residual_ledger) == len(matrix)


def test_unresolved_residuals_never_pollute_the_uncertainty_pool():
    """``status == "OOF"`` means a forecast exists, not that the outcome does.
    An unresolved row's NaN residual entering the pool would turn every later
    SD/correlation into NaN silently, because np.std propagates NaN."""
    games = _composite_population(completed_weeks=8, scheduled_weeks=4)
    ledger = he.build_horizon_membership_ledger(games)
    matrix = ohf.build_official_horizon_matrix(games, "TUE", ledger)
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=4, min_uncertainty_warmup=4)

    _, residual_ledger, _ = ohf.build_official_horizon_oof(matrix, horizon="TUE", config=cfg)

    oof = residual_ledger[residual_ledger["status"] == "OOF"]
    assert (~np.isfinite(oof["margin_residual"].to_numpy(dtype=float))).any(), (
        "fixture must contain unresolved OOF rows, otherwise this proves nothing"
    )
    eligible = residual_ledger[residual_ledger["uncertainty_eligible"]]
    assert len(eligible) > 0
    assert np.isfinite(eligible["margin_residual_sd_oof"].to_numpy(dtype=float)).all()
    assert np.isfinite(eligible["total_residual_sd_oof"].to_numpy(dtype=float)).all()
    assert np.isfinite(eligible["residual_correlation_oof"].to_numpy(dtype=float)).all()


def test_chronological_no_leakage_rules_remain_intact():
    """Strict ``result_available_at_utc < target_cutoff_utc``, no target in its
    own fit, and the availability invariant still enforced."""
    matrix = _matrix_with_unresolved_rows()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)

    resolution = dict(
        zip(matrix["game_id"].astype(str), pd.to_datetime(matrix["result_available_at_utc"], utc=True))
    )
    for _, row in predictions.iterrows():
        cutoff = pd.Timestamp(row["target_cutoff_utc"])
        assert row["game_id"] not in set(row["training_game_ids"])
        for train_id in row["training_game_ids"]:
            assert resolution[train_id] < cutoff
        if row["training_game_count"]:
            assert pd.Timestamp(row["max_training_result_available_at_utc"]) < cutoff


def test_the_label_filter_is_a_no_op_on_a_fully_resolved_matrix():
    """Backward compatibility with the certified 2020-2025 results: where every
    row is labelled, the training mask is exactly the purely temporal one."""
    matrix = _synthetic_matrix()
    assert matrix["home_margin"].notna().all()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE", config=cfg)

    resolution = pd.to_datetime(matrix["result_available_at_utc"], utc=True)
    for _, row in predictions.iterrows():
        expected = int((resolution < pd.Timestamp(row["target_cutoff_utc"])).sum())
        assert row["training_game_count"] == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_no_non_finite_label_of_any_kind_enters_training(bad):
    """NaN is what production hit, but an infinity is no more trainable. The
    eligibility mask is finiteness, not merely not-null."""
    matrix = _synthetic_matrix()
    cfg = ohf.OfficialHorizonOOFConfig(min_training_games=1, min_uncertainty_warmup=2)
    poisoned = matrix.copy()
    poisoned.loc[poisoned["game_id"] == "G1", "home_margin"] = bad

    predictions, _ = ohf.generate_official_horizon_oof_predictions(poisoned, horizon="TUE", config=cfg)
    for ids in predictions["training_game_ids"]:
        assert "G1" not in set(ids)
    # G1 is still predicted; it just cannot vote.
    assert len(predictions[predictions["game_id"] == "G1"]) == 1


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
