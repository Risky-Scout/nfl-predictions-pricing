from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and denominator > 0 else np.nan


def _drive_points(frame: pd.DataFrame) -> tuple[float, int]:
    if "drive" not in frame or frame["drive"].isna().all():
        return np.nan, 0
    drive_count = int(frame["drive"].nunique(dropna=True))
    if not {"posteam_score", "posteam_score_post"}.issubset(frame.columns):
        return np.nan, drive_count

    points = 0.0
    for _, drive in frame.groupby("drive", dropna=True, sort=False):
        before = pd.to_numeric(drive["posteam_score"], errors="coerce").dropna()
        after = pd.to_numeric(drive["posteam_score_post"], errors="coerce").dropna()
        if len(before) and len(after):
            points += max(float(after.max() - before.min()), 0.0)
    return points, drive_count


def aggregate_team_game_efficiency(
    play_by_play: pd.DataFrame,
    *,
    exclude_no_plays: bool = True,
) -> pd.DataFrame:
    """Aggregate canonical nflverse plays into one leakage-ready team-game row.

    The result contains offense metrics for `team_id` and the same metrics
    allowed by its defense. It is a postgame table; pregame features must
    lag or otherwise enforce `available_at_utc`.
    """
    required = {"game_id", "posteam", "defteam"}
    missing = required - set(play_by_play.columns)
    if missing:
        raise ValueError(f"Play-by-play missing: {sorted(missing)}")

    work = play_by_play.copy()
    work = work[work["posteam"].notna() & work["defteam"].notna()].copy()
    if exclude_no_plays and "no_play" in work:
        work = work[_series(work, "no_play", 0).fillna(0).eq(0)]
    if "play_type" in work:
        work = work[~work["play_type"].isin(["no_play", "qb_kneel"])].copy()

    rows: list[dict[str, object]] = []
    grouping = ["game_id", "posteam", "defteam"]
    for keys, frame in work.groupby(grouping, sort=False, dropna=False):
        game_id, team_id, opponent_id = keys
        epa = _series(frame, "epa", np.nan)
        success = _series(frame, "success", np.nan)
        down = _series(frame, "down", np.nan)
        score_diff = _series(frame, "score_differential", np.nan)
        seconds = _series(frame, "game_seconds_remaining", np.nan)
        dropback = _series(frame, "qb_dropback", 0).fillna(0).eq(1)
        rush = _series(frame, "rush_attempt", 0).fillna(0).eq(1)
        pass_attempt = _series(frame, "pass_attempt", 0).fillna(0).eq(1)
        sack = _series(frame, "sack", 0).fillna(0).eq(1)
        interception = _series(frame, "interception", 0).fillna(0).eq(1)
        fumble_lost = _series(frame, "fumble_lost", 0).fillna(0).eq(1)
        touchdown = _series(frame, "touchdown", 0).fillna(0).eq(1)
        penalty = _series(frame, "penalty", 0).fillna(0).eq(1)
        early = down.le(2)
        late = down.ge(3)
        neutral = score_diff.abs().le(8) & seconds.gt(300)
        red_zone = _series(frame, "yardline_100", np.nan).le(20)
        goal_to_go = _series(frame, "goal_to_go", 0).fillna(0).eq(1)
        third = down.eq(3)
        fourth = down.eq(4)
        third_converted = _series(frame, "third_down_converted", 0).fillna(0).eq(1)
        fourth_converted = _series(frame, "fourth_down_converted", 0).fillna(0).eq(1)
        passing_yards = _series(frame, "passing_yards", 0).fillna(0)
        rushing_yards = _series(frame, "rushing_yards", 0).fillna(0)

        points, drives = _drive_points(frame)
        if "drive" in frame and drives:
            red_zone_drives = int(frame.loc[red_zone, "drive"].nunique(dropna=True))
            red_zone_td_drives = int(
                frame.loc[red_zone & touchdown, "drive"].nunique(dropna=True)
            )
            goal_to_go_drives = int(
                frame.loc[goal_to_go, "drive"].nunique(dropna=True)
            )
            goal_to_go_td_drives = int(
                frame.loc[goal_to_go & touchdown, "drive"].nunique(dropna=True)
            )
            starting_yardline = (
                frame.sort_values(["drive", "game_seconds_remaining"], ascending=[True, False])
                .groupby("drive", dropna=True)["yardline_100"]
                .first()
            )
            avg_starting_field_position = (
                float((100.0 - pd.to_numeric(starting_yardline, errors="coerce")).mean())
                if len(starting_yardline)
                else np.nan
            )
        else:
            red_zone_drives = goal_to_go_drives = 0
            red_zone_td_drives = goal_to_go_td_drives = 0
            avg_starting_field_position = np.nan

        metadata = {}
        for column in (
            "season",
            "season_type",
            "week",
            "gameday",
            "home_team",
            "away_team",
        ):
            if column in frame:
                metadata[column] = frame[column].iloc[0]

        plays = len(frame)
        dropbacks = int(dropback.sum())
        rushes = int(rush.sum())
        pass_attempts = int(pass_attempt.sum())
        turnovers = int((interception | fumble_lost).sum())

        rows.append(
            {
                **metadata,
                "game_id": game_id,
                "team_id": team_id,
                "opponent_id": opponent_id,
                "plays": plays,
                "drives": drives,
                "points_per_drive": _rate(points, drives) if np.isfinite(points) else np.nan,
                "epa_per_play": float(epa.mean()),
                "success_rate": float(success.mean()),
                "dropbacks": dropbacks,
                "dropback_epa": float(epa[dropback].mean()) if dropbacks else np.nan,
                "dropback_success_rate": float(success[dropback].mean()) if dropbacks else np.nan,
                "rushes": rushes,
                "rush_epa": float(epa[rush].mean()) if rushes else np.nan,
                "rush_success_rate": float(success[rush].mean()) if rushes else np.nan,
                "early_down_epa": float(epa[early].mean()) if early.any() else np.nan,
                "late_down_epa": float(epa[late].mean()) if late.any() else np.nan,
                "early_down_pass_rate": _rate(int((early & dropback).sum()), int(early.sum())),
                "neutral_situation_pass_rate": _rate(
                    int((neutral & dropback).sum()), int(neutral.sum())
                ),
                "explosive_pass_rate": _rate(
                    int((pass_attempt & passing_yards.ge(20)).sum()), pass_attempts
                ),
                "explosive_rush_rate": _rate(
                    int((rush & rushing_yards.ge(10)).sum()), rushes
                ),
                "turnovers": turnovers,
                "turnover_rate": _rate(turnovers, plays),
                "interception_rate": _rate(int(interception.sum()), dropbacks),
                "fumble_lost_rate": _rate(int(fumble_lost.sum()), plays),
                "sacks": int(sack.sum()),
                "sack_rate": _rate(int(sack.sum()), dropbacks),
                "third_down_attempts": int(third.sum()),
                "third_down_conversion_rate": _rate(
                    int(third_converted.sum()), int(third.sum())
                ),
                "fourth_down_attempts": int(fourth.sum()),
                "fourth_down_conversion_rate": _rate(
                    int(fourth_converted.sum()), int(fourth.sum())
                ),
                "red_zone_drives": red_zone_drives,
                "red_zone_td_rate": _rate(red_zone_td_drives, red_zone_drives),
                "goal_to_go_drives": goal_to_go_drives,
                "goal_to_go_td_rate": _rate(goal_to_go_td_drives, goal_to_go_drives),
                "plays_per_drive": _rate(plays, drives),
                "average_starting_field_position": avg_starting_field_position,
                "penalty_rate": _rate(int(penalty.sum()), plays),
                "penalty_epa": float(epa[penalty].sum()) if penalty.any() else 0.0,
                "no_huddle_rate": _rate(
                    int(_series(frame, "no_huddle", 0).fillna(0).eq(1).sum()), plays
                ),
                "shotgun_rate": _rate(
                    int(_series(frame, "shotgun", 0).fillna(0).eq(1).sum()), plays
                ),
            }
        )

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
            "gameday",
            "home_team",
            "away_team",
            "team_id",
            "opponent_id",
        )
        if column in offense.columns
    ]
    metric_columns = [column for column in offense.columns if column not in id_columns]
    offense_prefixed = offense.rename(
        columns={column: f"offense_{column}" for column in metric_columns}
    )

    defense = offense[["game_id", "team_id", "opponent_id"] + metric_columns].copy()
    defense = defense.rename(
        columns={
            "team_id": "opponent_id",
            "opponent_id": "team_id",
            **{column: f"defense_allowed_{column}" for column in metric_columns},
        }
    )

    return offense_prefixed.merge(
        defense,
        on=["game_id", "team_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )
