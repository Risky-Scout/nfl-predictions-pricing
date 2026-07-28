import math

import pytest

from nfl_hybrid.spreadsheet_baselines import (
    TeamScoringProfile,
    adjusted_elo_difference,
    adjusted_qb_game_value,
    blend_base_and_head_to_head,
    brier_error,
    contest_points,
    elo_expected_margin,
    elo_point_change,
    elo_win_probability,
    exponential_decay_weight,
    implied_team_points,
    margin_of_victory_multiplier,
    offseason_regression,
    predict_points,
    probability_average,
    qb_game_value,
    reference_cases,
    starter_qb_adjustment,
    team_attack_strength,
    team_defense_weakness,
    update_qb_rating,
    wind_multiplier,
)


def test_all_reference_cells_match_exactly():
    for case in reference_cases():
        assert case.calculate() == pytest.approx(
            case.expected,
            abs=case.tolerance,
        )


def test_elo_probability_is_complementary():
    probability = elo_win_probability(80)
    assert probability + elo_win_probability(-80) == pytest.approx(1.0)


def test_elo_expected_margin():
    assert elo_expected_margin(325.5) == pytest.approx(13.02)


def test_adjusted_elo_difference_contract():
    value = adjusted_elo_difference(
        home_rating=1500,
        away_rating=1450,
        home_travel_distance=0,
        away_travel_distance=750,
        neutral_site=False,
        home_had_rest_week=True,
        away_had_rest_week=False,
        home_qb_adjustment=8,
        away_qb_adjustment=3,
    )
    assert value == pytest.approx(
        50 + 55 + 3 + 25 + 5
    )


def test_margin_multiplier_and_point_change():
    multiplier = margin_of_victory_multiplier(7, -325.5)
    change = elo_point_change(
        k_factor=20,
        actual_result=1,
        expected_probability=0.133110879399282,
        margin_multiplier=multiplier,
    )
    assert multiplier == pytest.approx(2.4405288832732133)
    assert change == pytest.approx(42.31335874842736)


def test_offseason_regression():
    assert offseason_regression(1600, 1500) == pytest.approx(
        1566.6666666666667
    )


def test_qb_composite_and_adjustments():
    value = qb_game_value(
        pass_attempts=24,
        completions=14,
        passing_yards=248,
        passing_touchdowns=3,
        interceptions=1,
        sacks=2,
        rush_attempts=3,
        rushing_yards=24,
        rushing_touchdowns=0,
    )
    adjusted = adjusted_qb_game_value(
        value,
        league_average_qb_value_allowed=48.92,
        opponent_qb_value_allowed=55,
    )
    assert value == pytest.approx(63.5)
    assert adjusted == pytest.approx(57.42)
    assert update_qb_rating(80.91, adjusted) == pytest.approx(78.561)
    assert starter_qb_adjustment(46.67, 44) == pytest.approx(8.811)


def test_total_strength_formulas():
    attack = team_attack_strength(
        home_points_for=21.926829268292682,
        away_points_for=22.682926829268293,
        league_home_points_for=23.59211461028993,
        league_away_points_for=21.807661671464317,
    )
    defense = team_defense_weakness(
        home_points_against=22.341463414634145,
        away_points_against=25.75609756097561,
        league_home_points_against=21.852083890025305,
        league_away_points_against=23.68380512292469,
    )
    assert attack == pytest.approx(0.9847745745123948)
    assert defense == pytest.approx(1.0549466929550688)


def test_total_points_prediction():
    result = predict_points(
        league_home_average=23.647940074906366,
        league_away_average=21.768539325842696,
        home_profile=TeamScoringProfile(
            attack_strength=1.239977423814897,
            defense_weakness=0.920292838580524,
        ),
        away_profile=TeamScoringProfile(
            attack_strength=0.941349399244411,
            defense_weakness=0.9939020774618309,
        ),
    )
    assert result[0] == pytest.approx(29.14410296778459)
    assert result[1] == pytest.approx(18.858458093504805)
    assert result[3] == pytest.approx(48.0025610612894)


def test_total_blend_and_wind():
    blended = blend_base_and_head_to_head(
        48.0025610612894,
        53.5,
    )
    assert blended == pytest.approx(49.65179274290257)
    assert wind_multiplier(0) == pytest.approx(1.0)
    assert wind_multiplier(10) < 1.0


def test_brier_and_contest_points():
    assert brier_error(0.64, 1) == pytest.approx(0.1296)
    assert contest_points(0.64, 1, week=1) == pytest.approx(12.0)
    assert contest_points(0.64, 1, week=18) == pytest.approx(24.0)


def test_probability_averages():
    assert probability_average(0.71, 0.64) == pytest.approx(0.675)
    assert probability_average(0.71, 0.64, 0.60) == pytest.approx(0.65)


def test_exponential_decay_half_life():
    assert exponential_decay_weight(
        initial_weight=10,
        age=1,
    ) == pytest.approx(9.980764435756287)
    assert exponential_decay_weight(
        initial_weight=10,
        age=360,
    ) == pytest.approx(5.0)


def test_market_implied_team_points():
    home, away = implied_team_points(
        total_line=47.5,
        home_spread=-3.5,
    )
    assert home == pytest.approx(25.5)
    assert away == pytest.approx(22.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        elo_win_probability(0, scale=0)
    with pytest.raises(ValueError):
        wind_multiplier(-1)
    with pytest.raises(ValueError):
        probability_average()



def test_qb_game_value_allows_negative_yardage():
    value = qb_game_value(
        pass_attempts=0,
        completions=0,
        passing_yards=-5,
        passing_touchdowns=0,
        interceptions=0,
        sacks=0,
        rush_attempts=0,
        rushing_yards=-3,
        rushing_touchdowns=0,
    )

    assert value == pytest.approx(-2.8)


def test_qb_game_value_rejects_negative_count_fields():
    with pytest.raises(ValueError, match="pass_attempts"):
        qb_game_value(
            pass_attempts=-1,
            completions=0,
            passing_yards=0,
            passing_touchdowns=0,
            interceptions=0,
            sacks=0,
            rush_attempts=0,
            rushing_yards=0,
            rushing_touchdowns=0,
        )
