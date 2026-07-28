from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_hybrid.data.team_ids import canonical_team_id


@dataclass(frozen=True)
class AdvancedPBPConfig:
    """Initial research definitions for advanced play-level aggregation."""

    neutral_score_margin: float = 8.0
    neutral_min_seconds_remaining: float = 300.0

    competitive_first_half_margin: float = 24.0
    competitive_third_quarter_margin: float = 20.0
    competitive_early_fourth_margin: float = 16.0
    competitive_final_five_margin: float = 8.0

    explosive_pass_yards: float = 20.0
    explosive_rush_yards: float = 10.0



def _canonicalize_team_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize provider aliases such as LA -> LAR."""

    result = frame.copy()

    for column in (
        "posteam",
        "defteam",
        "home_team",
        "away_team",
    ):
        if column in result.columns:
            result[column] = result[column].map(
                canonical_team_id
            )

    return result


SPECIAL_METRIC_PREFIXES = (
    "special_teams_",
    "field_goal_",
    "extra_point_",
    "punt_",
    "kickoff_",
)


def _numeric(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)

    return pd.to_numeric(frame[column], errors="coerce")


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    return _numeric(frame, column, 0.0).fillna(0.0).eq(1.0)


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")

    return frame[column].astype("string").fillna("")


def _rate(numerator: float, denominator: float) -> float:
    if denominator is None or denominator <= 0:
        return np.nan

    return float(numerator) / float(denominator)


def _masked_mean(
    frame: pd.DataFrame,
    column: str,
    mask: pd.Series,
) -> float:
    values = _numeric(frame, column)
    selected = values[mask]

    if selected.notna().sum() == 0:
        return np.nan

    return float(selected.mean())


def _masked_sum(
    frame: pd.DataFrame,
    column: str,
    mask: pd.Series,
) -> float:
    values = _numeric(frame, column)
    selected = values[mask]

    if selected.notna().sum() == 0:
        return 0.0

    return float(selected.sum())


def _competitive_mask(
    frame: pd.DataFrame,
    config: AdvancedPBPConfig,
) -> pd.Series:
    seconds = _numeric(
        frame,
        "game_seconds_remaining",
        0.0,
    ).fillna(0.0)

    margin = _numeric(
        frame,
        "score_differential",
        0.0,
    ).fillna(0.0).abs()

    limits = np.select(
        [
            seconds.gt(1800),
            seconds.gt(900),
            seconds.gt(300),
        ],
        [
            config.competitive_first_half_margin,
            config.competitive_third_quarter_margin,
            config.competitive_early_fourth_margin,
        ],
        default=config.competitive_final_five_margin,
    )

    return pd.Series(
        margin.to_numpy() <= limits,
        index=frame.index,
    )


def _play_masks(
    frame: pd.DataFrame,
    config: AdvancedPBPConfig,
) -> dict[str, pd.Series]:
    play_type = _text(frame, "play_type")

    no_play = _flag(frame, "no_play") | play_type.eq("no_play")
    kneel = _flag(frame, "qb_kneel") | play_type.eq("qb_kneel")
    spike = play_type.eq("qb_spike")

    dropback = _flag(frame, "qb_dropback")
    rush_attempt = _flag(frame, "rush_attempt")
    pass_attempt = _flag(frame, "pass_attempt")
    scramble = _flag(frame, "scramble")

    scrimmage = (
        (dropback | rush_attempt)
        & ~no_play
        & ~kneel
        & ~spike
    )

    designed_rush = (
        rush_attempt
        & ~scramble
        & ~kneel
        & ~spike
        & ~no_play
    )

    special_teams = (
        _flag(frame, "special_teams_play")
        | play_type.isin(
            [
                "field_goal",
                "extra_point",
                "punt",
                "kickoff",
            ]
        )
    ) & ~no_play

    down = _numeric(frame, "down")
    score_difference = _numeric(
        frame,
        "score_differential",
        0.0,
    ).fillna(0.0)

    seconds = _numeric(
        frame,
        "game_seconds_remaining",
        0.0,
    ).fillna(0.0)

    half_seconds = _numeric(
        frame,
        "half_seconds_remaining",
    )

    yardline_100 = _numeric(frame, "yardline_100")
    goal_to_go = _flag(frame, "goal_to_go")

    competitive = (
        scrimmage
        & _competitive_mask(frame, config)
    )

    neutral = (
        scrimmage
        & score_difference.abs().le(
            config.neutral_score_margin
        )
        & seconds.gt(
            config.neutral_min_seconds_remaining
        )
    )

    early_down = scrimmage & down.le(2)
    late_down = scrimmage & down.ge(3)

    two_minute = (
        scrimmage
        & half_seconds.gt(0)
        & half_seconds.le(120)
    )

    red_zone = scrimmage & yardline_100.le(20)

    return {
        "play_type": play_type,
        "no_play": no_play,
        "kneel": kneel,
        "spike": spike,
        "scrimmage": scrimmage,
        "competitive": competitive,
        "neutral": neutral,
        "early_down": early_down,
        "late_down": late_down,
        "two_minute": two_minute,
        "red_zone": red_zone,
        "dropback": dropback & scrimmage,
        "pass_attempt": pass_attempt & scrimmage,
        "rush_attempt": rush_attempt & scrimmage,
        "designed_rush": designed_rush,
        "scramble": scramble & scrimmage,
        "special_teams": special_teams,
        "third_down": scrimmage & down.eq(3),
        "fourth_down": scrimmage & down.eq(4),
        "goal_to_go": scrimmage & goal_to_go,
    }


def _drive_metrics(
    frame: pd.DataFrame,
    scrimmage_mask: pd.Series,
) -> dict[str, float]:
    if "drive" not in frame.columns:
        return {
            "drives": 0,
            "points_per_drive": np.nan,
            "scrimmage_plays_per_drive": np.nan,
            "average_starting_field_position": np.nan,
            "red_zone_drives": 0,
            "red_zone_td_rate": np.nan,
            "goal_to_go_drives": 0,
            "goal_to_go_td_rate": np.nan,
        }

    valid_drive = frame["drive"].notna()

    if not valid_drive.any():
        return {
            "drives": 0,
            "points_per_drive": np.nan,
            "scrimmage_plays_per_drive": np.nan,
            "average_starting_field_position": np.nan,
            "red_zone_drives": 0,
            "red_zone_td_rate": np.nan,
            "goal_to_go_drives": 0,
            "goal_to_go_td_rate": np.nan,
        }

    drive_frame = frame.loc[valid_drive].copy()
    drives = int(drive_frame["drive"].nunique())

    points = 0.0

    if {
        "posteam_score",
        "posteam_score_post",
    }.issubset(drive_frame.columns):
        for _, drive in drive_frame.groupby(
            "drive",
            sort=False,
            dropna=True,
        ):
            before = pd.to_numeric(
                drive["posteam_score"],
                errors="coerce",
            ).dropna()

            after = pd.to_numeric(
                drive["posteam_score_post"],
                errors="coerce",
            ).dropna()

            if len(before) and len(after):
                points += max(
                    float(after.max() - before.min()),
                    0.0,
                )

    starting_field_positions: list[float] = []

    if "yardline_100" in drive_frame.columns:
        for _, drive in drive_frame.groupby(
            "drive",
            sort=False,
            dropna=True,
        ):
            if "game_seconds_remaining" in drive.columns:
                drive = drive.sort_values(
                    "game_seconds_remaining",
                    ascending=False,
                )

            yardline = pd.to_numeric(
                drive["yardline_100"],
                errors="coerce",
            ).dropna()

            if len(yardline):
                starting_field_positions.append(
                    100.0 - float(yardline.iloc[0])
                )

    yardline = _numeric(frame, "yardline_100")
    touchdown = _flag(frame, "touchdown")
    goal_to_go = _flag(frame, "goal_to_go")

    red_zone_drives = int(
        frame.loc[
            valid_drive & yardline.le(20),
            "drive",
        ].nunique()
    )

    red_zone_td_drives = int(
        frame.loc[
            valid_drive
            & yardline.le(20)
            & touchdown,
            "drive",
        ].nunique()
    )

    goal_to_go_drives = int(
        frame.loc[
            valid_drive & goal_to_go,
            "drive",
        ].nunique()
    )

    goal_to_go_td_drives = int(
        frame.loc[
            valid_drive
            & goal_to_go
            & touchdown,
            "drive",
        ].nunique()
    )

    return {
        "drives": drives,
        "points_per_drive": _rate(points, drives),
        "scrimmage_plays_per_drive": _rate(
            int(scrimmage_mask.sum()),
            drives,
        ),
        "average_starting_field_position": (
            float(np.mean(starting_field_positions))
            if starting_field_positions
            else np.nan
        ),
        "red_zone_drives": red_zone_drives,
        "red_zone_td_rate": _rate(
            red_zone_td_drives,
            red_zone_drives,
        ),
        "goal_to_go_drives": goal_to_go_drives,
        "goal_to_go_td_rate": _rate(
            goal_to_go_td_drives,
            goal_to_go_drives,
        ),
    }


def _offensive_metric_name(metric: str) -> str:
    if metric.startswith(SPECIAL_METRIC_PREFIXES):
        return metric

    return f"offense_{metric}"


def _opponent_metric_name(metric: str) -> str:
    if metric.startswith(SPECIAL_METRIC_PREFIXES):
        return f"opponent_{metric}"

    return f"defense_allowed_{metric}"


def aggregate_advanced_team_game(
    play_by_play: pd.DataFrame,
    *,
    config: AdvancedPBPConfig | None = None,
) -> pd.DataFrame:
    """Create one advanced team-game row from many play-level rows.

    Raw play-by-play remains unchanged. This table is a postgame fact table and
    must be shifted before being used for pregame prediction.
    """

    cfg = config or AdvancedPBPConfig()

    required = {
        "game_id",
        "posteam",
        "defteam",
    }

    missing = required - set(play_by_play.columns)

    if missing:
        raise ValueError(
            f"Play-by-play missing required columns: {sorted(missing)}"
        )

    work = play_by_play.loc[
        play_by_play["game_id"].notna()
        & play_by_play["posteam"].notna()
        & play_by_play["defteam"].notna()
    ].copy()

    work = _canonicalize_team_columns(work)

    rows: list[dict[str, object]] = []

    for (
        game_id,
        team_id,
        opponent_id,
    ), frame in work.groupby(
        ["game_id", "posteam", "defteam"],
        sort=False,
        dropna=False,
    ):
        masks = _play_masks(frame, cfg)

        scrimmage = masks["scrimmage"]
        competitive = masks["competitive"]
        neutral = masks["neutral"]
        early = masks["early_down"]
        late = masks["late_down"]
        two_minute = masks["two_minute"]
        red_zone = masks["red_zone"]
        dropback = masks["dropback"]
        pass_attempt = masks["pass_attempt"]
        rush_attempt = masks["rush_attempt"]
        designed_rush = masks["designed_rush"]
        scramble = masks["scramble"]
        special_teams = masks["special_teams"]
        third_down = masks["third_down"]
        fourth_down = masks["fourth_down"]

        passing_yards = _numeric(
            frame,
            "passing_yards",
            0.0,
        ).fillna(0.0)

        rushing_yards = _numeric(
            frame,
            "rushing_yards",
            0.0,
        ).fillna(0.0)

        interception = _flag(frame, "interception")
        fumble_lost = _flag(frame, "fumble_lost")
        sack = _flag(frame, "sack")
        touchdown = _flag(frame, "touchdown")
        completion = _flag(frame, "complete_pass")
        penalty = _flag(frame, "penalty")

        turnovers = (
            interception | fumble_lost
        ) & scrimmage

        explosive_pass = (
            pass_attempt
            & passing_yards.ge(
                cfg.explosive_pass_yards
            )
        )

        explosive_rush = (
            designed_rush
            & rushing_yards.ge(
                cfg.explosive_rush_yards
            )
        )

        play_type = masks["play_type"]

        field_goal = (
            special_teams
            & play_type.eq("field_goal")
        )

        extra_point = (
            special_teams
            & play_type.eq("extra_point")
        )

        punt = (
            special_teams
            & play_type.eq("punt")
        )

        kickoff = (
            special_teams
            & play_type.eq("kickoff")
        )

        third_converted = (
            _flag(frame, "third_down_converted")
            & third_down
        )

        fourth_converted = (
            _flag(frame, "fourth_down_converted")
            & fourth_down
        )

        row: dict[str, object] = {
            "game_id": game_id,
            "team_id": team_id,
            "opponent_id": opponent_id,
        }

        for metadata_column in (
            "season",
            "season_type",
            "week",
            "home_team",
            "away_team",
        ):
            if metadata_column in frame.columns:
                row[metadata_column] = (
                    frame[metadata_column].iloc[0]
                )

        metrics: dict[str, float | int] = {
            "raw_play_rows": len(frame),
            "scrimmage_plays": int(scrimmage.sum()),
            "competitive_scrimmage_plays": int(
                competitive.sum()
            ),
            "neutral_scrimmage_plays": int(
                neutral.sum()
            ),
            "neutral_dropbacks": int(
                (neutral & dropback).sum()
            ),
            "neutral_situation_pass_rate": _rate(
                int((neutral & dropback).sum()),
                int(neutral.sum()),
            ),
            "early_down_plays": int(early.sum()),
            "late_down_plays": int(late.sum()),
            "two_minute_plays": int(
                two_minute.sum()
            ),
            "red_zone_plays": int(red_zone.sum()),
            "dropbacks": int(dropback.sum()),
            "pass_attempts": int(
                pass_attempt.sum()
            ),
            "rush_attempts": int(
                rush_attempt.sum()
            ),
            "designed_rushes": int(
                designed_rush.sum()
            ),
            "scrambles": int(scramble.sum()),
            "epa_per_play": _masked_mean(
                frame,
                "epa",
                scrimmage,
            ),
            "success_rate": _masked_mean(
                frame,
                "success",
                scrimmage,
            ),
            "competitive_epa_per_play": _masked_mean(
                frame,
                "epa",
                competitive,
            ),
            "competitive_success_rate": _masked_mean(
                frame,
                "success",
                competitive,
            ),
            "neutral_epa_per_play": _masked_mean(
                frame,
                "epa",
                neutral,
            ),
            "neutral_success_rate": _masked_mean(
                frame,
                "success",
                neutral,
            ),
            "early_down_epa": _masked_mean(
                frame,
                "epa",
                early,
            ),
            "early_down_success_rate": _masked_mean(
                frame,
                "success",
                early,
            ),
            "late_down_epa": _masked_mean(
                frame,
                "epa",
                late,
            ),
            "late_down_success_rate": _masked_mean(
                frame,
                "success",
                late,
            ),
            "two_minute_epa": _masked_mean(
                frame,
                "epa",
                two_minute,
            ),
            "two_minute_success_rate": _masked_mean(
                frame,
                "success",
                two_minute,
            ),
            "red_zone_epa": _masked_mean(
                frame,
                "epa",
                red_zone,
            ),
            "red_zone_success_rate": _masked_mean(
                frame,
                "success",
                red_zone,
            ),
            "dropback_epa": _masked_mean(
                frame,
                "epa",
                dropback,
            ),
            "dropback_success_rate": _masked_mean(
                frame,
                "success",
                dropback,
            ),
            "pass_attempt_epa": _masked_mean(
                frame,
                "epa",
                pass_attempt,
            ),
            "completion_rate": _rate(
                int((completion & pass_attempt).sum()),
                int(pass_attempt.sum()),
            ),
            "cpoe": _masked_mean(
                frame,
                "cpoe",
                pass_attempt,
            ),
            "air_yards_per_attempt": _masked_mean(
                frame,
                "air_yards",
                pass_attempt,
            ),
            "rush_epa": _masked_mean(
                frame,
                "epa",
                rush_attempt,
            ),
            "rush_success_rate": _masked_mean(
                frame,
                "success",
                rush_attempt,
            ),
            "designed_rush_epa": _masked_mean(
                frame,
                "epa",
                designed_rush,
            ),
            "designed_rush_success_rate": _masked_mean(
                frame,
                "success",
                designed_rush,
            ),
            "scramble_epa": _masked_mean(
                frame,
                "epa",
                scramble,
            ),
            "scramble_success_rate": _masked_mean(
                frame,
                "success",
                scramble,
            ),
            "explosive_pass_rate": _rate(
                int(explosive_pass.sum()),
                int(pass_attempt.sum()),
            ),
            "explosive_rush_rate": _rate(
                int(explosive_rush.sum()),
                int(designed_rush.sum()),
            ),
            "sacks": int((sack & dropback).sum()),
            "sack_rate": _rate(
                int((sack & dropback).sum()),
                int(dropback.sum()),
            ),
            "interceptions": int(
                (interception & scrimmage).sum()
            ),
            "interception_rate": _rate(
                int(
                    (
                        interception
                        & pass_attempt
                    ).sum()
                ),
                int(pass_attempt.sum()),
            ),
            "fumbles_lost": int(
                (fumble_lost & scrimmage).sum()
            ),
            "turnovers": int(turnovers.sum()),
            "turnover_rate": _rate(
                int(turnovers.sum()),
                int(scrimmage.sum()),
            ),
            "third_down_attempts": int(
                third_down.sum()
            ),
            "third_down_conversion_rate": _rate(
                int(third_converted.sum()),
                int(third_down.sum()),
            ),
            "fourth_down_attempts": int(
                fourth_down.sum()
            ),
            "fourth_down_conversion_rate": _rate(
                int(fourth_converted.sum()),
                int(fourth_down.sum()),
            ),
            "no_huddle_rate": _rate(
                int(
                    (
                        _flag(frame, "no_huddle")
                        & scrimmage
                    ).sum()
                ),
                int(scrimmage.sum()),
            ),
            "shotgun_rate": _rate(
                int(
                    (
                        _flag(frame, "shotgun")
                        & scrimmage
                    ).sum()
                ),
                int(scrimmage.sum()),
            ),
            "penalty_events": int(penalty.sum()),
            "penalty_rate": _rate(
                int(penalty.sum()),
                int(scrimmage.sum()),
            ),
            "special_teams_plays": int(
                special_teams.sum()
            ),
            "special_teams_epa_per_play": _masked_mean(
                frame,
                "epa",
                special_teams,
            ),
            "field_goal_attempts": int(
                field_goal.sum()
            ),
            "field_goal_epa_per_attempt": _masked_mean(
                frame,
                "epa",
                field_goal,
            ),
            "extra_point_attempts": int(
                extra_point.sum()
            ),
            "extra_point_epa_per_attempt": _masked_mean(
                frame,
                "epa",
                extra_point,
            ),
            "punt_attempts": int(punt.sum()),
            "punt_epa_per_attempt": _masked_mean(
                frame,
                "epa",
                punt,
            ),
            "kickoff_attempts": int(
                kickoff.sum()
            ),
            "kickoff_epa_per_attempt": _masked_mean(
                frame,
                "epa",
                kickoff,
            ),
            "touchdowns": int(
                (touchdown & scrimmage).sum()
            ),
        }

        metrics.update(
            _drive_metrics(
                frame,
                scrimmage,
            )
        )

        for metric, value in metrics.items():
            row[
                _offensive_metric_name(metric)
            ] = value

        rows.append(row)

    offense = pd.DataFrame(rows)

    if offense.empty:
        return offense

    id_columns = [
        column
        for column in (
            "game_id",
            "season",
            "season_type",
            "week",
            "home_team",
            "away_team",
            "team_id",
            "opponent_id",
        )
        if column in offense.columns
    ]

    metric_columns = [
        column
        for column in offense.columns
        if column not in id_columns
    ]

    mirror = offense[
        ["game_id", "team_id", "opponent_id"]
        + metric_columns
    ].copy()

    mirror = mirror.rename(
        columns={
            "team_id": "opponent_id",
            "opponent_id": "team_id",
            **{
                column: _opponent_metric_name(
                    column.removeprefix("offense_")
                )
                if column.startswith("offense_")
                else f"opponent_{column}"
                for column in metric_columns
            },
        }
    )

    return offense.merge(
        mirror,
        on=["game_id", "team_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )


def aggregate_qb_game_efficiency(
    play_by_play: pd.DataFrame,
    *,
    config: AdvancedPBPConfig | None = None,
) -> pd.DataFrame:
    """Create one quarterback-game row per passer with at least one dropback."""

    cfg = config or AdvancedPBPConfig()

    required = {
        "game_id",
        "posteam",
        "defteam",
        "passer_player_id",
    }

    missing = required - set(play_by_play.columns)

    if missing:
        raise ValueError(
            f"Play-by-play missing required QB columns: {sorted(missing)}"
        )

    work = play_by_play.loc[
        play_by_play["game_id"].notna()
        & play_by_play["posteam"].notna()
        & play_by_play["defteam"].notna()
        & play_by_play["passer_player_id"].notna()
    ].copy()

    work = _canonicalize_team_columns(work)

    rows: list[dict[str, object]] = []

    for (
        game_id,
        team_id,
        opponent_id,
        player_id,
    ), frame in work.groupby(
        [
            "game_id",
            "posteam",
            "defteam",
            "passer_player_id",
        ],
        sort=False,
        dropna=False,
    ):
        masks = _play_masks(frame, cfg)

        dropback = masks["dropback"]
        pass_attempt = masks["pass_attempt"]
        competitive = (
            masks["competitive"]
            & dropback
        )
        neutral = (
            masks["neutral"]
            & dropback
        )

        if int(dropback.sum()) == 0:
            continue

        completion = _flag(frame, "complete_pass")
        interception = _flag(frame, "interception")
        sack = _flag(frame, "sack")
        scramble = _flag(frame, "scramble")
        touchdown = _flag(frame, "touchdown")

        passing_yards = _numeric(
            frame,
            "passing_yards",
            0.0,
        ).fillna(0.0)

        explosive_pass = (
            pass_attempt
            & passing_yards.ge(
                cfg.explosive_pass_yards
            )
        )

        row: dict[str, object] = {
            "game_id": game_id,
            "team_id": team_id,
            "opponent_id": opponent_id,
            "player_id": player_id,
        }

        for metadata_column in (
            "season",
            "season_type",
            "week",
            "home_team",
            "away_team",
        ):
            if metadata_column in frame.columns:
                row[metadata_column] = (
                    frame[metadata_column].iloc[0]
                )

        row.update(
            {
                "dropbacks": int(dropback.sum()),
                "attempts": int(pass_attempt.sum()),
                "completions": int(
                    (
                        completion
                        & pass_attempt
                    ).sum()
                ),
                "passing_yards": float(
                    passing_yards[
                        pass_attempt
                    ].sum()
                ),
                "passing_touchdowns": int(
                    (
                        touchdown
                        & pass_attempt
                    ).sum()
                ),
                "interceptions": int(
                    (
                        interception
                        & pass_attempt
                    ).sum()
                ),
                "sacks": int(
                    (
                        sack
                        & dropback
                    ).sum()
                ),
                "scrambles": int(
                    (
                        scramble
                        & dropback
                    ).sum()
                ),
                "epa_per_dropback": _masked_mean(
                    frame,
                    "epa",
                    dropback,
                ),
                "success_rate": _masked_mean(
                    frame,
                    "success",
                    dropback,
                ),
                "competitive_epa_per_dropback": _masked_mean(
                    frame,
                    "epa",
                    competitive,
                ),
                "neutral_epa_per_dropback": _masked_mean(
                    frame,
                    "epa",
                    neutral,
                ),
                "completion_rate": _rate(
                    int(
                        (
                            completion
                            & pass_attempt
                        ).sum()
                    ),
                    int(pass_attempt.sum()),
                ),
                "cpoe": _masked_mean(
                    frame,
                    "cpoe",
                    pass_attempt,
                ),
                "air_yards_per_attempt": _masked_mean(
                    frame,
                    "air_yards",
                    pass_attempt,
                ),
                "explosive_pass_rate": _rate(
                    int(explosive_pass.sum()),
                    int(pass_attempt.sum()),
                ),
                "sack_rate": _rate(
                    int(
                        (
                            sack
                            & dropback
                        ).sum()
                    ),
                    int(dropback.sum()),
                ),
                "interception_rate": _rate(
                    int(
                        (
                            interception
                            & pass_attempt
                        ).sum()
                    ),
                    int(pass_attempt.sum()),
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)
