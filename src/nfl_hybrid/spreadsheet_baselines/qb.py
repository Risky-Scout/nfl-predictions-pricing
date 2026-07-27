from __future__ import annotations


QB_ELO_MULTIPLIER = 3.3


def qb_game_value(
    *,
    pass_attempts: float,
    completions: float,
    passing_yards: float,
    passing_touchdowns: float,
    interceptions: float,
    sacks: float,
    rush_attempts: float,
    rushing_yards: float,
    rushing_touchdowns: float,
) -> float:
    """
    Exact spreadsheet composite:

    -2.2*Att + 3.7*Comp + PassYds/5 + 11.3*PassTD
    -14.1*INT - 8*Sacks -1.1*RushAtt + 0.6*RushYds
    +15.9*RushTD
    """
    values = [
        pass_attempts,
        completions,
        passing_yards,
        passing_touchdowns,
        interceptions,
        sacks,
        rush_attempts,
        rushing_yards,
        rushing_touchdowns,
    ]
    if any(float(value) < 0 for value in values):
        raise ValueError("QB counting statistics cannot be negative.")
    return (
        -2.2 * float(pass_attempts)
        + 3.7 * float(completions)
        + float(passing_yards) / 5.0
        + 11.3 * float(passing_touchdowns)
        - 14.1 * float(interceptions)
        - 8.0 * float(sacks)
        - 1.1 * float(rush_attempts)
        + 0.6 * float(rushing_yards)
        + 15.9 * float(rushing_touchdowns)
    )


def adjusted_qb_game_value(
    game_value: float,
    *,
    league_average_qb_value_allowed: float,
    opponent_qb_value_allowed: float,
) -> float:
    return (
        float(game_value)
        + float(league_average_qb_value_allowed)
        - float(opponent_qb_value_allowed)
    )


def update_qb_rating(
    old_rating: float,
    adjusted_game_value: float,
    *,
    new_game_weight: float = 0.10,
) -> float:
    if not 0.0 <= new_game_weight <= 1.0:
        raise ValueError("new_game_weight must be in [0, 1].")
    return (
        float(old_rating) * (1.0 - new_game_weight)
        + float(adjusted_game_value) * new_game_weight
    )


def starter_qb_adjustment(
    starter_rating: float,
    team_qb_rating: float,
    *,
    multiplier: float = QB_ELO_MULTIPLIER,
) -> float:
    return float(multiplier) * (
        float(starter_rating) - float(team_qb_rating)
    )
