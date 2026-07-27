import json

import numpy as np
import pandas as pd

from nfl_hybrid.selection.compact_tournament import (
    DEFAULT_MARKET_SPECS,
    TournamentConfig,
    _aggregate_results,
    _build_stability_report,
    _candidate_feature_sets,
    _evaluate_matrix,
    _market_anchor_mode,
    _probability_metrics,
    _select_models,
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
                    "market_home_ml_novig_prob": 1
                    / (1 + np.exp(-0.25 * margin)),
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
            bootstrap_repetitions=10,
            run_integrity_checks=False,
        ),
    )
    assert set(folds["test_season"]) == {2021, 2022, 2023}
    assert not folds["test_season"].isin([2024, 2025]).any()


def test_market_augmented_classifier_without_anchor_is_flagged():
    spec = DEFAULT_MARKET_SPECS["pregame_ats"]
    classifier_mode = _market_anchor_mode(
        spec=spec,
        variant="market_augmented",
        model_family="classifier",
        selected_features=("matchup_epa_per_play_net",),
    )
    residual_mode = _market_anchor_mode(
        spec=spec,
        variant="market_augmented",
        model_family="residual_regression",
        selected_features=("matchup_epa_per_play_net",),
    )
    assert classifier_mode == "unanchored"
    assert residual_mode == "market_residual_target_anchor"


def _fold_row(
    season: int,
    *,
    family: str,
    log_loss_value: float,
    brier_value: float,
) -> dict[str, object]:
    baseline = family == "baseline"
    return {
        "market": "pregame_ats",
        "variant": "market_augmented",
        "test_season": season,
        "feature_set": "market_baseline" if baseline else "strength_only",
        "model_family": family,
        "model_name": "baseline" if baseline else "logistic_l2_c0.05",
        "feature_count": 0 if baseline else 15,
        "features": "[]" if baseline else json.dumps(["matchup_x"]),
        "market_anchor_mode": (
            "market_baseline" if baseline else "explicit_market_feature"
        ),
        "n": 250,
        "log_loss": log_loss_value,
        "brier": brier_value,
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "ece": 0.02,
        "mean_probability": 0.5,
        "event_rate": 0.5,
    }


def test_baseline_veto_retains_baseline_when_challenger_loses():
    rows = []
    for season in (2021, 2022, 2023):
        rows.append(
            _fold_row(
                season,
                family="baseline",
                log_loss_value=0.692,
                brier_value=0.249,
            )
        )
        rows.append(
            _fold_row(
                season,
                family="classifier",
                log_loss_value=0.699,
                brier_value=0.253,
            )
        )
    folds = pd.DataFrame(rows)
    aggregate = _aggregate_results(folds)
    stability = _build_stability_report(folds, TournamentConfig())
    selected, _ = _select_models(aggregate, stability)
    assert selected.iloc[0]["model_family"] == "baseline"
    assert (
        selected.iloc[0]["selection_status"]
        == "RETAIN_BASELINE_NO_QUALIFYING_MODEL"
    )
