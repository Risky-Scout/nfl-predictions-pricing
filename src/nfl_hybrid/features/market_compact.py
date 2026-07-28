from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


OA_METRICS = (
    "epa_per_play",
    "success_rate",
    "dropback_epa",
    "dropback_success_rate",
    "designed_rush_epa",
    "designed_rush_success_rate",
    "explosive_pass_rate",
    "explosive_rush_rate",
    "sack_rate",
    "turnover_rate",
    "points_per_drive",
    "special_teams_epa_per_play",
)

QB_METRICS = (
    "epa_per_dropback",
    "success_rate",
    "competitive_epa_per_dropback",
    "neutral_epa_per_dropback",
    "completion_rate",
    "cpoe",
    "air_yards_per_attempt",
    "explosive_pass_rate",
    "sack_rate",
    "interception_rate",
)

LAGGED_MATCHUP_METRICS = (
    "red_zone_td_rate",
    "neutral_situation_pass_rate",
    "scrimmage_plays_per_drive",
)


def _first_existing(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
    default: float | str | bool | None = np.nan,
) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    if required:
        raise ValueError(f"None of the required columns exist: {tuple(candidates)}")
    return pd.Series(default, index=frame.index)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(float)
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y", "neutral", "post"}).astype(float)


def american_implied_probability(odds: pd.Series) -> pd.Series:
    values = _numeric(odds)
    positive = values.gt(0)
    probability = pd.Series(np.nan, index=values.index, dtype=float)
    probability.loc[positive] = 100.0 / (values.loc[positive] + 100.0)
    probability.loc[~positive & values.notna()] = (
        -values.loc[~positive & values.notna()]
        / (-values.loc[~positive & values.notna()] + 100.0)
    )
    return probability


def _devig_two_way(
    first_price: pd.Series,
    second_price: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    first_raw = american_implied_probability(first_price)
    second_raw = american_implied_probability(second_price)
    hold = first_raw + second_raw - 1.0
    denominator = first_raw + second_raw
    first_novig = first_raw / denominator
    second_novig = second_raw / denominator
    return first_novig, second_novig, hold


def _derive_targets(frame: pd.DataFrame, out: dict[str, object]) -> None:
    home_score = _numeric(
        _first_existing(frame, ("home_score", "score_home"))
    )
    away_score = _numeric(
        _first_existing(frame, ("away_score", "score_away"))
    )
    margin = home_score - away_score
    total = home_score + away_score
    spread = out["market_home_spread"]
    total_line = out["market_total_line"]

    out["target_home_margin"] = margin
    out["target_total_points"] = total
    out["target_home_win"] = np.where(
        margin.gt(0), 1.0, np.where(margin.lt(0), 0.0, np.nan)
    )
    out["target_tie"] = margin.eq(0).astype(float)
    ats_edge = margin + spread
    out["target_home_cover"] = np.where(
        ats_edge.gt(0), 1.0, np.where(ats_edge.lt(0), 0.0, np.nan)
    )
    out["target_ats_push"] = ats_edge.eq(0).astype(float)
    total_edge = total - total_line
    out["target_over"] = np.where(
        total_edge.gt(0), 1.0, np.where(total_edge.lt(0), 0.0, np.nan)
    )
    out["target_total_push"] = total_edge.eq(0).astype(float)
    out["target_margin_residual"] = margin - out["market_implied_margin"]
    out["target_total_residual"] = total - total_line


def _derive_context(frame: pd.DataFrame, out: dict[str, object]) -> None:
    out["home_field_indicator"] = 1.0 - _boolean(
        _first_existing(
            frame,
            ("neutral_site", "stadium_neutral"),
            required=False,
            default=False,
        )
    )
    playoff_source = _first_existing(
        frame,
        ("playoff", "schedule_playoff"),
        required=False,
        default=np.nan,
    )
    playoff_flag = _boolean(playoff_source)

    # The canonical nflverse game table commonly stores postseason status as
    # season_type/game_type == "POST" rather than a dedicated boolean.
    # Fall back to that representation when the direct flag is unavailable or
    # non-informative.
    season_type = _first_existing(
        frame,
        ("season_type", "game_type"),
        required=False,
        default="",
    ).astype("string").str.strip().str.upper()

    postseason_from_type = season_type.isin(
        {
            "POST",
            "POSTSEASON",
            "PLAYOFF",
            "WC",
            "WILD_CARD",
            "DIV",
            "DIVISIONAL",
            "CON",
            "CONFERENCE",
            "SB",
            "SUPER_BOWL",
        }
    ).astype(float)

    if playoff_source.isna().all() or playoff_flag.nunique(dropna=True) <= 1:
        playoff_flag = np.maximum(playoff_flag, postseason_from_type)

    out["playoff_flag"] = playoff_flag
    out["division_game_flag"] = _boolean(
        _first_existing(
            frame,
            ("division_game", "div_game"),
            required=False,
            default=False,
        )
    )
    home_rest = _numeric(
        _first_existing(
            frame,
            ("home_rest_days", "home_rest"),
            required=False,
        )
    )
    away_rest = _numeric(
        _first_existing(
            frame,
            ("away_rest_days", "away_rest"),
            required=False,
        )
    )
    out["rest_days_diff"] = home_rest - away_rest
    home_bye_source = _first_existing(
        frame,
        ("home_bye_flag",),
        required=False,
        default=np.nan,
    )
    away_bye_source = _first_existing(
        frame,
        ("away_bye_flag",),
        required=False,
        default=np.nan,
    )

    home_bye = _boolean(home_bye_source)
    away_bye = _boolean(away_bye_source)

    # Some canonical schedule tables retain rest days but not explicit bye
    # flags. In that case, use an extended-rest indicator (10+ days) as the
    # reproducible bye proxy. This captures standard NFL bye weeks while
    # leaving the continuous rest differential available separately.
    home_bye_from_rest = home_rest.ge(10).fillna(False).astype(float)
    away_bye_from_rest = away_rest.ge(10).fillna(False).astype(float)

    if home_bye_source.isna().all() or home_bye.nunique(dropna=True) <= 1:
        home_bye = np.maximum(home_bye, home_bye_from_rest)

    if away_bye_source.isna().all() or away_bye.nunique(dropna=True) <= 1:
        away_bye = np.maximum(away_bye, away_bye_from_rest)

    out["bye_diff"] = home_bye - away_bye

    roof = _first_existing(
        frame, ("roof", "weather_detail"), required=False, default=""
    ).astype("string").str.lower()
    explicit_indoor = _boolean(
        _first_existing(
            frame, ("indoor_game", "dome"), required=False, default=False
        )
    )
    out["indoor_flag"] = np.maximum(
        explicit_indoor,
        roof.str.contains("dome|closed|indoor", regex=True, na=False).astype(float),
    )

    home_prior = _numeric(
        _first_existing(
            frame,
            ("home_team_prior_games", "home_games_prior"),
            required=False,
            default=0.0,
        )
    ).fillna(0.0)
    away_prior = _numeric(
        _first_existing(
            frame,
            ("away_team_prior_games", "away_games_prior"),
            required=False,
            default=0.0,
        )
    ).fillna(0.0)
    out["prior_games_min"] = np.minimum(home_prior, away_prior)
    out["early_season_uncertainty"] = 1.0 / np.sqrt(
        1.0 + out["prior_games_min"]
    )


def _derive_market(frame: pd.DataFrame, out: dict[str, object]) -> None:
    out["market_home_spread"] = _numeric(
        _first_existing(
            frame,
            ("home_spread_reference", "spreadspoke_home_spread_reference"),
        )
    )
    out["market_total_line"] = _numeric(
        _first_existing(
            frame,
            ("total_line_reference", "spreadspoke_total_line_reference"),
        )
    )
    out["market_implied_margin"] = -out["market_home_spread"]
    out["market_implied_home_points"] = (
        out["market_total_line"] + out["market_implied_margin"]
    ) / 2.0
    out["market_implied_away_points"] = (
        out["market_total_line"] - out["market_implied_margin"]
    ) / 2.0

    home_ml = _first_existing(frame, ("home_moneyline_reference",))
    away_ml = _first_existing(frame, ("away_moneyline_reference",))
    (
        out["market_home_ml_novig_prob"],
        out["market_away_ml_novig_prob"],
        out["market_moneyline_hold"],
    ) = _devig_two_way(home_ml, away_ml)
    out["market_home_ml_raw_prob"] = american_implied_probability(home_ml)

    home_spread_price = _first_existing(
        frame, ("home_spread_price_reference",)
    )
    away_spread_price = _first_existing(
        frame, ("away_spread_price_reference",)
    )
    (
        out["market_home_cover_novig_prob"],
        out["market_away_cover_novig_prob"],
        out["market_spread_hold"],
    ) = _devig_two_way(home_spread_price, away_spread_price)

    over_price = _first_existing(frame, ("over_price_reference",))
    under_price = _first_existing(frame, ("under_price_reference",))
    (
        out["market_over_novig_prob"],
        out["market_under_novig_prob"],
        out["market_total_hold"],
    ) = _devig_two_way(over_price, under_price)

    magnitude = out["market_home_spread"].abs()
    out["spread_magnitude"] = magnitude
    out["spread_distance_to_3"] = (magnitude - 3.0).abs()
    out["spread_distance_to_7"] = (magnitude - 7.0).abs()
    out["spread_integer_flag"] = np.isclose(
        out["market_home_spread"],
        np.round(out["market_home_spread"]),
    ).astype(float)
    out["spread_half_point_flag"] = np.isclose(
        (out["market_home_spread"].abs() * 2.0) % 2.0,
        1.0,
    ).astype(float)
    out["home_favorite_flag"] = out["market_home_spread"].lt(0).astype(float)

    out["market_spread_source_disagreement"] = _numeric(
        _first_existing(
            frame,
            ("spread_reference_difference",),
            required=False,
            default=0.0,
        )
    ).fillna(0.0)
    out["market_total_source_disagreement"] = _numeric(
        _first_existing(
            frame,
            ("total_reference_difference",),
            required=False,
            default=0.0,
        )
    ).fillna(0.0)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Required compact-feature source columns missing: {missing}")


def _derive_oa_matchups(frame: pd.DataFrame, out: dict[str, object]) -> None:
    for metric in OA_METRICS:
        prefix = f"oa_{metric}"
        columns = {
            "home_off": f"home_{prefix}_offense_mean",
            "away_off": f"away_{prefix}_offense_mean",
            "home_def_allowed": f"home_{prefix}_defense_allowed_mean",
            "away_def_allowed": f"away_{prefix}_defense_allowed_mean",
            "home_league": f"home_{prefix}_league_mean",
            "away_league": f"away_{prefix}_league_mean",
            "home_off_sd": f"home_{prefix}_offense_sd",
            "away_off_sd": f"away_{prefix}_offense_sd",
            "home_def_sd": f"home_{prefix}_defense_sd",
            "away_def_sd": f"away_{prefix}_defense_sd",
            "home_off_rel": f"home_{prefix}_offense_reliability",
            "away_off_rel": f"away_{prefix}_offense_reliability",
            "home_def_rel": f"home_{prefix}_defense_reliability",
            "away_def_rel": f"away_{prefix}_defense_reliability",
        }
        _require_columns(frame, columns.values())
        league = (
            _numeric(frame[columns["home_league"]])
            + _numeric(frame[columns["away_league"]])
        ) / 2.0
        home_expected = (
            _numeric(frame[columns["home_off"]])
            + _numeric(frame[columns["away_def_allowed"]])
            - league
        )
        away_expected = (
            _numeric(frame[columns["away_off"]])
            + _numeric(frame[columns["home_def_allowed"]])
            - league
        )
        out[f"matchup_{metric}_home_expected"] = home_expected
        out[f"matchup_{metric}_away_expected"] = away_expected
        out[f"matchup_{metric}_net"] = home_expected - away_expected
        out[f"matchup_{metric}_sum"] = home_expected + away_expected
        out[f"matchup_{metric}_uncertainty"] = np.sqrt(
            _numeric(frame[columns["home_off_sd"]]) ** 2
            + _numeric(frame[columns["away_off_sd"]]) ** 2
            + _numeric(frame[columns["home_def_sd"]]) ** 2
            + _numeric(frame[columns["away_def_sd"]]) ** 2
        )
        out[f"matchup_{metric}_reliability"] = pd.concat(
            [
                _numeric(frame[columns["home_off_rel"]]),
                _numeric(frame[columns["away_off_rel"]]),
                _numeric(frame[columns["home_def_rel"]]),
                _numeric(frame[columns["away_def_rel"]]),
            ],
            axis=1,
        ).mean(axis=1)


def _derive_qb_matchups(frame: pd.DataFrame, out: dict[str, object]) -> None:
    for metric in QB_METRICS:
        home_mean = f"home_qb_{metric}_mean"
        away_mean = f"away_qb_{metric}_mean"
        home_sd = f"home_qb_{metric}_sd"
        away_sd = f"away_qb_{metric}_sd"
        home_rel = f"home_qb_{metric}_reliability"
        away_rel = f"away_qb_{metric}_reliability"
        home_delta = f"home_qb_{metric}_delta_vs_team"
        away_delta = f"away_qb_{metric}_delta_vs_team"
        _require_columns(
            frame,
            (
                home_mean,
                away_mean,
                home_sd,
                away_sd,
                home_rel,
                away_rel,
                home_delta,
                away_delta,
            ),
        )
        out[f"qb_{metric}_diff"] = (
            _numeric(frame[home_mean]) - _numeric(frame[away_mean])
        )
        out[f"qb_{metric}_sum"] = (
            _numeric(frame[home_mean]) + _numeric(frame[away_mean])
        )
        out[f"qb_{metric}_combined_sd"] = np.sqrt(
            _numeric(frame[home_sd]) ** 2 + _numeric(frame[away_sd]) ** 2
        )
        out[f"qb_{metric}_min_reliability"] = np.minimum(
            _numeric(frame[home_rel]), _numeric(frame[away_rel])
        )
        out[f"qb_{metric}_delta_vs_team_diff"] = (
            _numeric(frame[home_delta]) - _numeric(frame[away_delta])
        )

    starter_columns = (
        "home_qb_max_starter_probability",
        "away_qb_max_starter_probability",
        "home_qb_starter_candidates",
        "away_qb_starter_candidates",
        "home_qb_starter_entropy",
        "away_qb_starter_entropy",
    )
    _require_columns(frame, starter_columns)
    out["qb_min_max_starter_probability"] = np.minimum(
        _numeric(frame["home_qb_max_starter_probability"]),
        _numeric(frame["away_qb_max_starter_probability"]),
    )
    out["qb_starter_candidates_sum"] = (
        _numeric(frame["home_qb_starter_candidates"])
        + _numeric(frame["away_qb_starter_candidates"])
    )
    out["qb_starter_entropy_sum"] = (
        _numeric(frame["home_qb_starter_entropy"])
        + _numeric(frame["away_qb_starter_entropy"])
    )


def _derive_lagged_matchups(frame: pd.DataFrame, out: dict[str, object]) -> None:
    for metric in LAGGED_MATCHUP_METRICS:
        columns = (
            f"home_offense_{metric}__ewm_hl4",
            f"away_offense_{metric}__ewm_hl4",
            f"home_defense_allowed_{metric}__ewm_hl4",
            f"away_defense_allowed_{metric}__ewm_hl4",
        )
        if not all(column in frame.columns for column in columns):
            continue
        home_expected = (
            _numeric(frame[columns[0]]) + _numeric(frame[columns[3]])
        ) / 2.0
        away_expected = (
            _numeric(frame[columns[1]]) + _numeric(frame[columns[2]])
        ) / 2.0
        out[f"lagged_{metric}_net"] = home_expected - away_expected
        out[f"lagged_{metric}_sum"] = home_expected + away_expected


def engineer_compact_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a small, pre-approved feature surface from the feature warehouse."""
    if "game_id" not in frame.columns:
        raise ValueError("Stage-two matrix is missing game_id.")

    out: dict[str, object] = {}
    metadata_candidates = (
        "game_id",
        "season",
        "week",
        "gameday",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
    )
    for column in metadata_candidates:
        if column in frame.columns:
            out[column] = frame[column]

    _derive_context(frame, out)
    _derive_market(frame, out)
    _derive_oa_matchups(frame, out)
    _derive_qb_matchups(frame, out)
    _derive_lagged_matchups(frame, out)
    _derive_targets(frame, out)

    engineered = pd.DataFrame(out, index=frame.index)
    if engineered["game_id"].duplicated().any():
        raise ValueError("Compact engineered table contains duplicate games.")
    return engineered
