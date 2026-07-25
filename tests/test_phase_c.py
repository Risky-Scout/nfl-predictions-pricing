import numpy as np
import pandas as pd

from nfl_hybrid.modern.market_residual import MarketResidualJointScoreModel
from nfl_hybrid.priors.quarterback import QuarterbackPriorBuilder, starter_mixture
from nfl_hybrid.priors.roster import RosterAdjustmentModel
from nfl_hybrid.priors.team import (
    EmpiricalBayesTeamPrior,
    TeamPriorConfig,
    tune_team_prior_hyperparameters,
)


def test_small_team_sample_shrinks_more():
    history = pd.DataFrame(
        {
            "entity_id": ["BIG", "SMALL", "BIG", "SMALL", "BIG", "SMALL"],
            "season": [2025, 2025, 2024, 2024, 2023, 2023],
            "metric": ["epa"] * 6,
            "value": [0.20, 0.20, 0.15, 0.15, 0.10, 0.10],
            "sample_size": [1000, 50, 900, 40, 800, 30],
        }
    )
    priors = EmpiricalBayesTeamPrior(
        TeamPriorConfig(prior_strength=400.0)
    ).build(history, target_season=2026)
    by_team = priors.set_index("entity_id")
    assert by_team.loc["BIG", "regression_weight"] > by_team.loc["SMALL", "regression_weight"]
    assert (
        by_team.loc["BIG", "prior_standard_deviation"]
        < by_team.loc["SMALL", "prior_standard_deviation"]
    )


def test_team_prior_tuning_returns_candidate():
    rows = []
    for season in range(2021, 2026):
        for team, base in (("A", 0.2), ("B", -0.1), ("C", 0.0)):
            rows.append(
                {
                    "entity_id": team,
                    "season": season,
                    "metric": "epa",
                    "value": base + 0.01 * (season - 2021),
                    "sample_size": 900,
                }
            )
    history = pd.DataFrame(rows)
    config, results = tune_team_prior_hyperparameters(
        history,
        target_seasons=[2024, 2025],
        season_weight_candidates=[(0.7, 0.3, 0.0), (0.5, 0.3, 0.2)],
        prior_strength_candidates=[100.0, 500.0],
    )
    assert len(results) == 4
    assert config.prior_strength in {100.0, 500.0}


def test_qb_mixture_includes_starter_uncertainty():
    history = pd.DataFrame(
        {
            "player_id": ["QB1", "QB1", "QB2", "QB2", "QB3"],
            "value": [0.20, 0.15, -0.10, -0.05, 0.00],
            "dropbacks": [500, 400, 300, 200, 100],
            "recency_dropbacks": [0, 500, 0, 300, 0],
            "years_experience": [4, 4, 2, 2, 1],
        }
    )
    priors = QuarterbackPriorBuilder().build(history)
    probabilities = pd.DataFrame(
        {
            "team_id": ["T", "T"],
            "player_id": ["QB1", "QB2"],
            "starter_probability": [0.6, 0.4],
        }
    )
    mix = starter_mixture(priors, probabilities).iloc[0]
    candidate_sd = priors[priors["player_id"].isin(["QB1", "QB2"])][
        "prior_standard_deviation"
    ].min()
    assert mix["starter_candidates"] == 2
    assert mix["starter_entropy"] > 0
    assert mix["qb_prior_standard_deviation"] > candidate_sd


def test_roster_adjustment_model_runs():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "returning_offense": rng.uniform(0.4, 1.0, 80),
            "returning_defense": rng.uniform(0.4, 1.0, 80),
        }
    )
    frame["early_season_performance_residual"] = (
        2.0 * frame["returning_offense"]
        + 1.0 * frame["returning_defense"]
        + rng.normal(0, 0.2, len(frame))
    )
    model = RosterAdjustmentModel(
        ["returning_offense", "returning_defense"]
    ).fit(frame)
    prediction = model.predict(frame.iloc[:3])
    assert len(prediction) == 3
    assert prediction["roster_adjustment_standard_deviation"].gt(0).all()


def _market_data(n=320, seed=7):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    market_margin = 2.5 + 3.0 * signal + rng.normal(0, 1.0, n)
    market_total = 44.0 + 1.5 * signal + rng.normal(0, 1.0, n)
    true_margin = market_margin + 1.2 * signal + rng.normal(0, 9.5, n)
    true_total = market_total + 0.8 * signal + rng.normal(0, 10.0, n)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "team": np.where(signal >= 0, "A", "B"),
            "home_spread": -market_margin,
            "total_line": market_total,
            "home_margin": true_margin,
            "total_points": true_total,
        }
    )
    frame["home_win"] = (frame["home_margin"] > 0).astype(float)
    ats_edge = frame["home_margin"] + frame["home_spread"]
    frame["home_cover"] = (ats_edge > 0).astype(float)
    frame["over"] = (frame["total_points"] > frame["total_line"]).astype(float)
    return frame


def test_market_residual_joint_model_is_coherent():
    data = _market_data()
    model = MarketResidualJointScoreModel(["signal"], ["team"])
    model.fit(data.iloc[:230], calibration=data.iloc[230:280])
    output = model.predict_markets(data.iloc[280:])
    for column in (
        "home_win_probability",
        "away_win_probability",
        "tie_probability",
        "home_cover_probability",
        "away_cover_probability",
        "ats_push_probability",
        "over_probability",
        "under_probability",
        "total_push_probability",
    ):
        assert output[column].between(0, 1).all()

    assert np.allclose(
        output["home_win_probability"]
        + output["away_win_probability"]
        + output["tie_probability"],
        1.0,
    )
    assert np.allclose(
        output["home_cover_probability"]
        + output["away_cover_probability"]
        + output["ats_push_probability"],
        1.0,
    )
    assert np.allclose(
        output["over_probability"]
        + output["under_probability"]
        + output["total_push_probability"],
        1.0,
    )
