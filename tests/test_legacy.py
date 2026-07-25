from math import isclose

import pytest

from nfl_hybrid.legacy.elo import EloContext, LegacyElo
from nfl_hybrid.legacy.ensemble import EnsembleInputs, legacy_ensemble_models
from nfl_hybrid.legacy.market import american_to_implied_probability, devig_two_way
from nfl_hybrid.legacy.qb import quarterback_value
from nfl_hybrid.legacy.scoring import legacy_wind_factor, predict_legacy_score


def test_elo_neutral_equal_teams():
    model = LegacyElo({"H": 1500.0, "A": 1500.0})
    p = model.predict("H", "A", EloContext(neutral_site=True))
    assert isclose(p.home_win_probability, 0.5)
    assert isclose(p.expected_home_margin, 0.0)


def test_elo_workbook_home_field():
    model = LegacyElo({"H": 1500.0, "A": 1500.0})
    p = model.predict("H", "A")
    assert isclose(p.adjusted_difference, 55.0)
    assert isclose(p.expected_home_margin, 2.2)


def test_unplayed_game_cannot_update():
    model = LegacyElo()
    with pytest.raises(ValueError):
        model.update("H", "A", None, None, completed=False)


def test_travel_direction_rewards_home_when_away_travels_farther():
    model = LegacyElo({"H": 1500.0, "A": 1500.0})
    p = model.predict(
        "H",
        "A",
        EloContext(neutral_site=True, home_travel_miles=0, away_travel_miles=2000),
    )
    assert isclose(p.travel_adjustment, 8.0)


def test_qb_interception_is_negative():
    base = quarterback_value(30, 20, 250, 2, 0, 2, 3, 15, 0)
    with_pick = quarterback_value(30, 20, 250, 2, 1, 2, 3, 15, 0)
    assert isclose(base - with_pick, 14.1)


def test_moneyline_conversion_and_devig():
    assert isclose(american_to_implied_probability(-150), 0.6)
    result = devig_two_way(-150, 130)
    assert isclose(result.home_probability + result.away_probability, 1.0)


def test_legacy_score_formula():
    p = predict_legacy_score(
        23.0, 21.0, 1.1, 0.9, 1.05, 0.95, h2h_total=44.0, wind_mph=10.0
    )
    assert p.expected_home_points > p.expected_away_points
    assert p.adjusted_total < p.blended_total
    assert 0.65 <= legacy_wind_factor(100.0) <= 1.05


def test_all_32_ensemble_models_are_available():
    models = legacy_ensemble_models(EnsembleInputs(0.60, 0.58, 0.55))
    assert set(models) == set(range(1, 33))
    assert models[19] == 0.55
    assert models[32] == 0.52
