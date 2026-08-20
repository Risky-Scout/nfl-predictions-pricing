from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from nfl_hybrid.data.team_ids import canonical_team_id


@dataclass(frozen=True)
class PregameRollingConfig:
    """Configuration for leakage-safe pregame team features."""

    windows: tuple[int, ...] = (4, 8)
    ewm_halflife_games: float = 4.0
    min_periods: int = 1


OWN_METRIC_PREFIXES = (
    "offense_",
    "defense_allowed_",
    "special_teams_",
    "field_goal_",
    "extra_point_",
    "punt_",
    "kickoff_",
)

IDENTIFIER_COLUMNS = {
    "game_id",
    "team_id",
    "opponent_id",
    "season",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "event_time",
}


def _resolve_event_time(games: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        "kickoff_utc",
        "scheduled_kickoff_utc",
        "gameday",
        "schedule_date",
    )

    time_column = next(
        (column for column in candidates if column in games.columns),
        None,
    )

    if time_column is None:
        raise ValueError(
            "Games table must contain one of: "
            "kickoff_utc, scheduled_kickoff_utc, gameday, schedule_date."
        )

    schedule = games[["game_id", time_column]].copy()
    schedule["event_time"] = pd.to_datetime(
        schedule[time_column],
        errors="coerce",
        utc=True,
    )

    if schedule["event_time"].isna().all():
        raise ValueError(
            f"Could not parse any timestamps from {time_column}."
        )

    return schedule[["game_id", "event_time"]]


def select_team_metric_columns(
    team_game: pd.DataFrame,
) -> list[str]:
    numeric_columns = set(
        team_game.select_dtypes(
            include=[np.number, "bool"]
        ).columns
    )

    return sorted(
        column
        for column in numeric_columns
        if column not in IDENTIFIER_COLUMNS
        and column.startswith(OWN_METRIC_PREFIXES)
        and not column.startswith("opponent_")
    )


def build_team_pregame_features(
    team_game: pd.DataFrame,
    games: pd.DataFrame,
    *,
    config: PregameRollingConfig | None = None,
    metric_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build team features using only games completed before each game.

    Each current-game metric is shifted one game before any expanding,
    rolling, season-to-date, or exponentially weighted calculation.
    """

    cfg = config or PregameRollingConfig()

    required_team_columns = {
        "game_id",
        "team_id",
        "opponent_id",
        "season",
    }

    missing = required_team_columns - set(team_game.columns)

    if missing:
        raise ValueError(
            f"Team-game table missing columns: {sorted(missing)}"
        )

    if "game_id" not in games.columns:
        raise ValueError("Games table is missing game_id.")

    schedule = _resolve_event_time(games)

    work = team_game.merge(
        schedule,
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    for column in (
        "team_id",
        "opponent_id",
        "home_team",
        "away_team",
    ):
        if column in work.columns:
            work[column] = work[column].map(
                canonical_team_id
            )

    if work["event_time"].isna().any():
        missing_games = (
            work.loc[work["event_time"].isna(), "game_id"]
            .drop_duplicates()
            .head(20)
            .tolist()
        )
        raise ValueError(
            f"Missing event time for games: {missing_games}"
        )

    work["season"] = pd.to_numeric(
        work["season"],
        errors="raise",
    ).astype(int)

    metrics = (
        list(metric_columns)
        if metric_columns is not None
        else select_team_metric_columns(work)
    )

    if not metrics:
        raise ValueError("No team-game metric columns were selected.")

    unknown_metrics = sorted(set(metrics) - set(work.columns))

    if unknown_metrics:
        raise ValueError(
            f"Requested metrics are missing: {unknown_metrics}"
        )

    work = work.sort_values(
        ["team_id", "event_time", "game_id"],
        kind="stable",
    ).reset_index(drop=True)

    id_columns = [
        column
        for column in (
            "game_id",
            "team_id",
            "opponent_id",
            "season",
            "season_type",
            "week",
            "home_team",
            "away_team",
            "event_time",
        )
        if column in work.columns
    ]

    team_frames: list[pd.DataFrame] = []

    for _, team_frame in work.groupby(
        "team_id",
        sort=False,
    ):
        team_frame = team_frame.sort_values(
            ["event_time", "game_id"],
            kind="stable",
        ).reset_index(drop=True)

        feature_data: dict[str, pd.Series | np.ndarray] = {
            "team_prior_games": np.arange(
                len(team_frame),
                dtype=int,
            ),
            "team_season_prior_games": (
                team_frame.groupby(
                    "season",
                    sort=False,
                ).cumcount()
            ),
            "days_since_last_game": (
                team_frame["event_time"]
                .diff()
                .dt.total_seconds()
                .div(86400.0)
            ),
        }

        for metric in metrics:
            values = pd.to_numeric(
                team_frame[metric],
                errors="coerce",
            )

            # Critical anti-leakage step.
            prior = values.shift(1)

            feature_data[
                f"{metric}__all_mean"
            ] = prior.expanding(
                min_periods=cfg.min_periods
            ).mean()

            feature_data[
                f"{metric}__all_count"
            ] = prior.expanding(
                min_periods=1
            ).count()

            feature_data[
                f"{metric}__ewm_hl{cfg.ewm_halflife_games:g}"
            ] = prior.ewm(
                halflife=cfg.ewm_halflife_games,
                adjust=False,
                min_periods=cfg.min_periods,
            ).mean()

            season_mean = (
                team_frame.assign(_metric=values)
                .groupby(
                    "season",
                    sort=False,
                )["_metric"]
                .transform(
                    lambda series: (
                        series.shift(1)
                        .expanding(
                            min_periods=cfg.min_periods
                        )
                        .mean()
                    )
                )
            )

            season_count = (
                team_frame.assign(_metric=values)
                .groupby(
                    "season",
                    sort=False,
                )["_metric"]
                .transform(
                    lambda series: (
                        series.shift(1)
                        .expanding(min_periods=1)
                        .count()
                    )
                )
            )

            feature_data[
                f"{metric}__season_mean"
            ] = season_mean

            feature_data[
                f"{metric}__season_count"
            ] = season_count

            for window in cfg.windows:
                feature_data[
                    f"{metric}__last{window}_mean"
                ] = prior.rolling(
                    window=window,
                    min_periods=cfg.min_periods,
                ).mean()

                feature_data[
                    f"{metric}__last{window}_count"
                ] = prior.rolling(
                    window=window,
                    min_periods=1,
                ).count()

        features = pd.DataFrame(feature_data)

        team_frames.append(
            pd.concat(
                [
                    team_frame[id_columns].reset_index(drop=True),
                    features.reset_index(drop=True),
                ],
                axis=1,
            )
        )

    output = pd.concat(
        team_frames,
        ignore_index=True,
    )

    duplicate_keys = int(
        output.duplicated(
            ["game_id", "team_id"],
            keep=False,
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(
            f"Duplicate pregame team keys found: {duplicate_keys}"
        )

    return output


def build_game_pregame_matrix(
    games: pd.DataFrame,
    team_pregame: pd.DataFrame,
    *,
    carrier_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Pivot canonicalized team features to one home/away game row.

    ``carrier_columns``: the exact NATIVE ``games`` columns to carry onto
    the base of the returned matrix, beyond the join keys (``game_id`` and
    whichever of ``home_team_id``/``home_team`` and ``away_team_id``/
    ``away_team`` are present) that are always kept because the pivot itself
    needs them. Default is ``()`` -- no native ``games`` column, safe or
    otherwise, is carried through unless a caller explicitly names it.
    ``None`` is accepted and treated identically to ``()``; it is NOT, and
    must never become, a way to request "every native column" -- there is no
    such request this function will honor.

    A caller that wants a native ``games`` column in its output (e.g. a
    schedule field like ``gameday``) must name it explicitly in
    ``carrier_columns``. No caller may request "all columns" -- there is no
    wildcard/None escape hatch. This is a deliberate API-level fix for the
    Fix 3.1 OOF target-feature-leakage incident (see
    :mod:`nfl_hybrid.evaluation.chronological_oof`): the pre-fix default
    silently copied the full, un-narrowed ``games`` table as the pivot base,
    which let native postgame/current-market columns (``home_score``,
    ``home_moneyline_reference``, ``home_spread_reference``, ...) survive
    into what a blind ``home_*``/``away_*`` sweep downstream then mistook for
    genuine pivoted pregame-state features. A safe-by-default empty
    carrier set means a newly added native ``games`` column (of any name,
    including one that looks pregame-safe) can never again reach a caller's
    output without that caller explicitly asking for it by name.
    """

    home_candidates = (
        "home_team_id",
        "home_team",
    )
    away_candidates = (
        "away_team_id",
        "away_team",
    )

    home_column = next(
        (
            column
            for column in home_candidates
            if column in games.columns
        ),
        None,
    )

    away_column = next(
        (
            column
            for column in away_candidates
            if column in games.columns
        ),
        None,
    )

    if home_column is None or away_column is None:
        raise ValueError(
            "Games table must contain home and away "
            "team identifiers."
        )

    # None is accepted only as a synonym for () -- never as "all columns".
    # There is no wildcard/full-passthrough request this function honors.
    if carrier_columns is None:
        carrier_columns = ()

    keep = list(
        dict.fromkeys(
            ["game_id", home_column, away_column, *carrier_columns]
        )
    )
    missing_carrier = [c for c in keep if c not in games.columns]
    if missing_carrier:
        raise ValueError(
            f"Games table missing requested carrier_columns: {missing_carrier}"
        )
    game_work = games[keep].copy()

    if game_work["game_id"].duplicated().any():
        raise ValueError(
            "Games table contains duplicate game_id rows."
        )

    game_work["_home_join_team_id"] = (
        game_work[home_column]
        .map(canonical_team_id)
    )

    game_work["_away_join_team_id"] = (
        game_work[away_column]
        .map(canonical_team_id)
    )

    team_work = team_pregame.copy()

    team_work["_join_team_id"] = (
        team_work["team_id"]
        .map(canonical_team_id)
    )

    duplicate_team_keys = team_work.duplicated(
        ["game_id", "_join_team_id"],
        keep=False,
    )

    if duplicate_team_keys.any():
        duplicates = team_work.loc[
            duplicate_team_keys,
            ["game_id", "team_id", "_join_team_id"],
        ].head(20)

        raise ValueError(
            "Canonicalization produced duplicate team-game keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    feature_columns = [
        column
        for column in team_work.columns
        if column not in IDENTIFIER_COLUMNS
        and column != "_join_team_id"
    ]

    home = team_work[
        ["game_id", "_join_team_id"]
        + feature_columns
    ].copy()

    home = home.rename(
        columns={
            "_join_team_id": "_home_feature_team_id",
            **{
                column: f"home_{column}"
                for column in feature_columns
            },
        }
    )

    away = team_work[
        ["game_id", "_join_team_id"]
        + feature_columns
    ].copy()

    away = away.rename(
        columns={
            "_join_team_id": "_away_feature_team_id",
            **{
                column: f"away_{column}"
                for column in feature_columns
            },
        }
    )

    matrix = game_work.merge(
        home,
        left_on=[
            "game_id",
            "_home_join_team_id",
        ],
        right_on=[
            "game_id",
            "_home_feature_team_id",
        ],
        how="left",
        validate="one_to_one",
    )

    matrix = matrix.merge(
        away,
        left_on=[
            "game_id",
            "_away_join_team_id",
        ],
        right_on=[
            "game_id",
            "_away_feature_team_id",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_home = (
        matrix["_home_feature_team_id"].isna()
    )

    missing_away = (
        matrix["_away_feature_team_id"].isna()
    )

    if missing_home.any() or missing_away.any():
        bad_games = matrix.loc[
            missing_home | missing_away,
            [
                "game_id",
                home_column,
                away_column,
                "_home_join_team_id",
                "_away_join_team_id",
            ],
        ].head(30)

        raise ValueError(
            "Could not attach both canonical team rows:\n"
            f"{bad_games.to_string(index=False)}"
        )

    matrix = matrix.drop(
        columns=[
            "_home_join_team_id",
            "_away_join_team_id",
            "_home_feature_team_id",
            "_away_feature_team_id",
        ]
    )

    return matrix
