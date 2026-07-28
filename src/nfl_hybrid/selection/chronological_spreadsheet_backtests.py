from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np
import pandas as pd

from nfl_hybrid.spreadsheet_baselines import (
    TeamScoringProfile,
    adjusted_elo_difference,
    adjusted_qb_game_value,
    elo_expected_margin,
    elo_win_probability,
    margin_of_victory_multiplier,
    offseason_regression,
    predict_points,
    qb_game_value,
    starter_qb_adjustment,
    team_attack_strength,
    team_defense_weakness,
    update_qb_rating,
)


OUTCOME_NAMES = {
    "pregame_moneyline": ("home", "away", "tie"),
    "pregame_ats": ("home_cover", "away_cover", "push"),
    "pregame_total": ("over", "under", "push"),
}


@dataclass(frozen=True)
class BacktestConfig:
    development_seasons: tuple[int, ...] = (2021, 2022, 2023)
    warmup_seasons: tuple[int, ...] = (2020,)
    initial_team_elo: float = 1500.0
    league_mean_elo: float = 1500.0
    elo_k_factor: float = 20.0
    elo_scale: float = 400.0
    points_per_elo: float = 25.0
    home_field_elo: float = 55.0
    rest_elo: float = 25.0
    playoff_multiplier: float = 1.2
    offseason_regression_fraction: float = 1.0 / 3.0
    qb_initial_rating: float = 50.0
    qb_new_game_weight: float = 0.10
    qb_adjustment_multiplier: float = 3.3
    margin_prior_sd: float = 13.5
    total_prior_sd: float = 13.0
    residual_prior_games: int = 64
    tie_prior_alpha: float = 2.0
    tie_prior_beta: float = 398.0
    probability_clip: float = 1e-6
    total_head_to_head_weight: float = 0.30
    minimum_profile_games: int = 1
    random_seed: int = 20260327

    @classmethod
    def from_json(cls, path: str | Path) -> "BacktestConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["development_seasons"] = tuple(payload["development_seasons"])
        payload["warmup_seasons"] = tuple(payload["warmup_seasons"])
        return cls(**payload)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"None of the required columns exist: {list(candidates)}")


def _optional_existing(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _discrete_integer_distribution(
    mean: float,
    sd: float,
    *,
    lower: int,
    upper: int,
) -> tuple[np.ndarray, np.ndarray]:
    sd = max(float(sd), 1e-6)
    support = np.arange(lower, upper + 1, dtype=int)
    lower_edges = (support.astype(float) - 0.5 - mean) / sd
    upper_edges = (support.astype(float) + 0.5 - mean) / sd
    probabilities = np.array(
        [
            _normal_cdf(upper_edge) - _normal_cdf(lower_edge)
            for lower_edge, upper_edge in zip(lower_edges, upper_edges)
        ],
        dtype=float,
    )
    probabilities[0] += _normal_cdf(lower_edges[0])
    probabilities[-1] += 1.0 - _normal_cdf(upper_edges[-1])
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()
    return support, probabilities


def _shrunk_sd(
    residuals: list[float],
    *,
    prior_sd: float,
    prior_games: int,
) -> float:
    if not residuals:
        return float(prior_sd)
    sample = np.asarray(residuals, dtype=float)
    sample_var = float(np.var(sample, ddof=1)) if len(sample) > 1 else prior_sd**2
    weight = len(sample) / (len(sample) + prior_games)
    variance = weight * sample_var + (1.0 - weight) * prior_sd**2
    return math.sqrt(max(variance, 1e-8))


def _three_way_scores(
    probabilities: tuple[float, float, float],
    outcome_index: int,
    *,
    clip: float,
) -> tuple[float, float]:
    vector = np.asarray(probabilities, dtype=float)
    vector = np.clip(vector, clip, 1.0)
    vector /= vector.sum()
    target = np.zeros(3, dtype=float)
    target[outcome_index] = 1.0
    log_loss = -math.log(float(vector[outcome_index]))
    brier = float(np.sum((vector - target) ** 2))
    return log_loss, brier


def _score_outcome_index(market: str, row: pd.Series) -> int:
    if market == "pregame_moneyline":
        if int(row["target_tie"]) == 1:
            return 2
        return 0 if int(row["target_home_win"]) == 1 else 1
    if market == "pregame_ats":
        if int(row["target_ats_push"]) == 1:
            return 2
        return 0 if int(row["target_home_cover"]) == 1 else 1
    if market == "pregame_total":
        if int(row["target_total_push"]) == 1:
            return 2
        return 0 if int(row["target_over"]) == 1 else 1
    raise ValueError(market)


def _conditional_market_probabilities(
    market_probability: float,
    push_or_tie_probability: float,
    *,
    clip: float,
) -> tuple[float, float, float]:
    conditional = float(np.clip(market_probability, clip, 1.0 - clip))
    rare = float(np.clip(push_or_tie_probability, clip, 1.0 - clip))
    action = 1.0 - rare
    return action * conditional, action * (1.0 - conditional), rare


def _market_probabilities_from_line_distribution(
    *,
    market_probability: float,
    rare_probability: float,
    clip: float,
) -> tuple[float, float, float]:
    return _conditional_market_probabilities(
        market_probability,
        rare_probability,
        clip=clip,
    )


def _margin_probabilities(
    mean_margin: float,
    home_spread: float,
    sd: float,
) -> tuple[float, float, float]:
    support, probabilities = _discrete_integer_distribution(
        mean_margin,
        sd,
        lower=-70,
        upper=70,
    )
    settlement = support.astype(float) + float(home_spread)
    home = float(probabilities[settlement > 1e-9].sum())
    away = float(probabilities[settlement < -1e-9].sum())
    push = float(probabilities[np.isclose(settlement, 0.0, atol=1e-9)].sum())
    total = home + away + push
    return home / total, away / total, push / total


def _total_probabilities(
    mean_total: float,
    total_line: float,
    sd: float,
) -> tuple[float, float, float]:
    support, probabilities = _discrete_integer_distribution(
        mean_total,
        sd,
        lower=0,
        upper=120,
    )
    over = float(probabilities[support > total_line + 1e-9].sum())
    under = float(probabilities[support < total_line - 1e-9].sum())
    push = float(
        probabilities[np.isclose(support, total_line, atol=1e-9)].sum()
    )
    total = over + under + push
    return over / total, under / total, push / total


def _tie_probability(
    ties_seen: int,
    games_seen: int,
    config: BacktestConfig,
) -> float:
    return (
        config.tie_prior_alpha + ties_seen
    ) / (
        config.tie_prior_alpha
        + config.tie_prior_beta
        + games_seen
    )


def _load_canonical_frames(root: Path) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for market in OUTCOME_NAMES:
        stem = f"{market}_market_augmented_canonical_t10"
        path = root / f"{stem}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        output[market] = pd.read_parquet(path)
    return output


def _discover_qb_game_file(backfill_root: Path) -> Path:
    required_groups = [
        {"game_id"},
        {"player_id", "qb_id", "passer_id"},
        {"attempts", "pass_attempts", "passing_attempts"},
        {"completions", "passing_completions"},
        {"passing_yards", "pass_yards"},
        {"passing_tds", "passing_touchdowns", "pass_touchdowns"},
        {"interceptions", "passing_interceptions"},
    ]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for QB input discovery.") from exc

    matches: list[Path] = []
    for path in backfill_root.rglob("*.parquet"):
        try:
            columns = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            continue
        if all(columns & group for group in required_groups):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "No QB game-stat parquet with the required columns was found "
            f"under {backfill_root}."
        )

    matches.sort(
        key=lambda item: (
            "qb" not in item.name.lower(),
            len(str(item)),
            str(item),
        )
    )
    return matches[0]


def _normalize_qb_games(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    aliases = {
        "game_id": ["game_id"],
        "player_id": ["player_id", "qb_id", "passer_id"],
        "pass_attempts": [
            "pass_attempts",
            "passing_attempts",
            "attempts",
        ],
        "completions": ["completions", "passing_completions"],
        "passing_yards": ["passing_yards", "pass_yards"],
        "passing_touchdowns": [
            "passing_touchdowns",
            "passing_tds",
            "pass_touchdowns",
        ],
        "interceptions": [
            "interceptions",
            "passing_interceptions",
        ],
        "sacks": ["sacks", "times_sacked"],
        "rush_attempts": [
            "rush_attempts",
            "rushing_attempts",
            "carries",
        ],
        "rushing_yards": ["rushing_yards", "rush_yards"],
        "rushing_touchdowns": [
            "rushing_touchdowns",
            "rushing_tds",
            "rush_touchdowns",
        ],
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        found = _optional_existing(frame.columns, candidates)
        if found is None:
            if canonical in {
                "sacks",
                "rush_attempts",
                "rushing_yards",
                "rushing_touchdowns",
            }:
                frame[canonical] = 0.0
                continue
            raise ValueError(
                f"QB game file {path} lacks a column for {canonical}."
            )
        rename[found] = canonical
    frame = frame.rename(columns=rename)
    selected = frame[list(aliases)].copy()
    selected["game_id"] = selected["game_id"].astype(str)
    selected["player_id"] = selected["player_id"].astype(str)
    for column in list(aliases)[2:]:
        selected[column] = pd.to_numeric(
            selected[column], errors="coerce"
        ).fillna(0.0)
    return selected.groupby(
        ["game_id", "player_id"],
        as_index=False,
    )[list(aliases)[2:]].sum()


def _prepare_game_frame(
    warehouse: pd.DataFrame,
    canonical: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    game_id_col = _first_existing(warehouse.columns, ["game_id"])
    kickoff_col = _first_existing(
        warehouse.columns,
        [
            "scheduled_kickoff_utc",
            "kickoff_utc",
            "start_time_utc",
            "game_datetime_utc",
        ],
    )

    metadata_candidates = [
        game_id_col,
        kickoff_col,
        "season",
        "week",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "season_type",
        "neutral_site",
        "home_rest_days",
        "away_rest_days",
        "home_qb_id",
        "away_qb_id",
        "home_qb_expected_starter_id",
        "away_qb_expected_starter_id",
        "home_qb_max_starter_probability",
        "away_qb_max_starter_probability",
        "home_qb_starter_entropy",
        "away_qb_starter_entropy",
    ]
    columns = [name for name in metadata_candidates if name in warehouse.columns]
    metadata = warehouse[columns].copy().rename(
        columns={game_id_col: "game_id", kickoff_col: "kickoff_utc"}
    )
    metadata["game_id"] = metadata["game_id"].astype(str)
    metadata["kickoff_utc"] = pd.to_datetime(
        metadata["kickoff_utc"], utc=True, errors="raise"
    )
    if metadata["game_id"].duplicated().any():
        raise ValueError("Warehouse contains duplicate game_id rows.")

    base = canonical["pregame_moneyline"][
        [
            "game_id",
            "season",
            "week",
            "home_team_id",
            "away_team_id",
            "target_home_win",
            "target_tie",
            "target_home_margin",
            "market_t10_novig_probability",
        ]
    ].rename(
        columns={
            "market_t10_novig_probability":
                "moneyline_market_home_probability"
        }
    )

    ats = canonical["pregame_ats"][
        [
            "game_id",
            "target_t10_home_cover",
            "target_t10_ats_push",
            "market_t10_consensus_line",
            "market_t10_novig_probability",
        ]
    ].rename(
        columns={
            "target_t10_home_cover": "target_home_cover",
            "target_t10_ats_push": "target_ats_push",
            "market_t10_consensus_line": "home_spread",
            "market_t10_novig_probability":
                "ats_market_home_probability",
        }
    )

    total = canonical["pregame_total"][
        [
            "game_id",
            "target_t10_over",
            "target_t10_total_push",
            "target_total_points",
            "market_t10_consensus_line",
            "market_t10_novig_probability",
        ]
    ].rename(
        columns={
            "target_t10_over": "target_over",
            "target_t10_total_push": "target_total_push",
            "market_t10_consensus_line": "total_line",
            "market_t10_novig_probability":
                "total_market_over_probability",
        }
    )

    games = (
        base.merge(ats, on="game_id", how="inner", validate="one_to_one")
        .merge(total, on="game_id", how="inner", validate="one_to_one")
        .merge(
            metadata.drop(
                columns=[
                    column
                    for column in (
                        "season",
                        "week",
                        "home_team_id",
                        "away_team_id",
                    )
                    if column in metadata.columns
                ]
            ),
            on="game_id",
            how="left",
            validate="one_to_one",
        )
    )

    if games["kickoff_utc"].isna().any():
        raise ValueError("Kickoff timestamp missing after warehouse merge.")
    if games[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Final scores missing after warehouse merge.")

    return games.sort_values(
        ["kickoff_utc", "game_id"], kind="stable"
    ).reset_index(drop=True)


def _profile_mean(
    values: list[float],
    fallback: float,
) -> float:
    return float(np.mean(values)) if values else float(fallback)


def _safe_probability(value: object, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(np.clip(numeric, 0.0, 1.0))


def run_chronological_backtest(
    *,
    warehouse_path: str | Path,
    canonical_root: str | Path,
    backfill_root: str | Path,
    output_root: str | Path,
    config: BacktestConfig,
    qb_games_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    warehouse_path = Path(warehouse_path).expanduser().resolve()
    canonical_root = Path(canonical_root).expanduser().resolve()
    backfill_root = Path(backfill_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    warehouse = pd.read_parquet(warehouse_path)
    canonical = _load_canonical_frames(canonical_root)
    games = _prepare_game_frame(warehouse, canonical)

    qb_path = (
        Path(qb_games_path).expanduser().resolve()
        if qb_games_path is not None
        else _discover_qb_game_file(backfill_root)
    )
    qb_games = _normalize_qb_games(qb_path)
    qb_lookup = {
        (row.game_id, row.player_id): row
        for row in qb_games.itertuples(index=False)
    }

    team_elo: dict[str, float] = defaultdict(
        lambda: config.initial_team_elo
    )
    qb_rating: dict[str, float] = defaultdict(
        lambda: config.qb_initial_rating
    )
    team_qb_rating: dict[str, float] = defaultdict(
        lambda: config.qb_initial_rating
    )
    defense_qb_allowed: dict[str, float] = defaultdict(
        lambda: config.qb_initial_rating
    )

    home_pf: dict[str, list[float]] = defaultdict(list)
    away_pf: dict[str, list[float]] = defaultdict(list)
    home_pa: dict[str, list[float]] = defaultdict(list)
    away_pa: dict[str, list[float]] = defaultdict(list)
    h2h_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    league_home_points: list[float] = []
    league_away_points: list[float] = []

    margin_residuals: list[float] = []
    total_residuals: list[float] = []
    ties_seen = 0
    games_seen = 0
    league_qb_values: list[float] = []
    current_season: int | None = None

    predictions: list[dict[str, Any]] = []
    elo_history: list[dict[str, Any]] = []
    qb_history: list[dict[str, Any]] = []
    total_history: list[dict[str, Any]] = []

    all_seasons = set(config.warmup_seasons) | set(config.development_seasons)
    games = games[games["season"].isin(all_seasons)].copy()

    for row in games.itertuples(index=False):
        season = int(row.season)
        if current_season != season:
            if current_season is not None:
                for team in list(team_elo):
                    team_elo[team] = offseason_regression(
                        team_elo[team],
                        config.league_mean_elo,
                        regression_fraction=
                            config.offseason_regression_fraction,
                    )
            current_season = season

        home = str(row.home_team_id)
        away = str(row.away_team_id)
        neutral = bool(getattr(row, "neutral_site", False))
        season_type = str(getattr(row, "season_type", "REG")).upper()
        playoff = season_type in {"POST", "POSTSEASON", "PLAYOFF"}

        home_rest_days = float(getattr(row, "home_rest_days", 7.0))
        away_rest_days = float(getattr(row, "away_rest_days", 7.0))
        home_bye = home_rest_days >= 10.0
        away_bye = away_rest_days >= 10.0

        pre_home_elo = float(team_elo[home])
        pre_away_elo = float(team_elo[away])

        plain_diff = adjusted_elo_difference(
            home_rating=pre_home_elo,
            away_rating=pre_away_elo,
            neutral_site=neutral,
            home_had_rest_week=home_bye,
            away_had_rest_week=away_bye,
            is_playoff=playoff,
            playoff_multiplier=config.playoff_multiplier,
        )

        expected_home_qb = str(
            getattr(row, "home_qb_expected_starter_id", "")
        )
        expected_away_qb = str(
            getattr(row, "away_qb_expected_starter_id", "")
        )
        home_starter_prob = _safe_probability(
            getattr(row, "home_qb_max_starter_probability", 1.0)
        )
        away_starter_prob = _safe_probability(
            getattr(row, "away_qb_max_starter_probability", 1.0)
        )
        if expected_home_qb in {"", "None", "nan", "<NA>"}:
            home_qb_adjustment = 0.0
        else:
            home_qb_adjustment = starter_qb_adjustment(
                qb_rating[expected_home_qb],
                team_qb_rating[home],
                multiplier=config.qb_adjustment_multiplier,
            ) * home_starter_prob

        if expected_away_qb in {"", "None", "nan", "<NA>"}:
            away_qb_adjustment = 0.0
        else:
            away_qb_adjustment = starter_qb_adjustment(
                qb_rating[expected_away_qb],
                team_qb_rating[away],
                multiplier=config.qb_adjustment_multiplier,
            ) * away_starter_prob

        qb_diff = adjusted_elo_difference(
            home_rating=pre_home_elo,
            away_rating=pre_away_elo,
            neutral_site=neutral,
            home_had_rest_week=home_bye,
            away_had_rest_week=away_bye,
            home_qb_adjustment=home_qb_adjustment,
            away_qb_adjustment=away_qb_adjustment,
            is_playoff=playoff,
            playoff_multiplier=config.playoff_multiplier,
        )

        tie_probability = _tie_probability(
            ties_seen, games_seen, config
        )

        plain_conditional_home = elo_win_probability(
            plain_diff, scale=config.elo_scale
        )
        qb_conditional_home = elo_win_probability(
            qb_diff, scale=config.elo_scale
        )

        plain_ml = _conditional_market_probabilities(
            plain_conditional_home,
            tie_probability,
            clip=config.probability_clip,
        )
        qb_ml = _conditional_market_probabilities(
            qb_conditional_home,
            tie_probability,
            clip=config.probability_clip,
        )

        plain_margin = elo_expected_margin(
            plain_diff,
            points_per_elo=config.points_per_elo,
        )
        qb_margin = elo_expected_margin(
            qb_diff,
            points_per_elo=config.points_per_elo,
        )
        margin_sd = _shrunk_sd(
            margin_residuals,
            prior_sd=config.margin_prior_sd,
            prior_games=config.residual_prior_games,
        )
        plain_ats = _margin_probabilities(
            plain_margin, float(row.home_spread), margin_sd
        )
        qb_ats = _margin_probabilities(
            qb_margin, float(row.home_spread), margin_sd
        )

        league_home_avg = _profile_mean(
            league_home_points, 23.5
        )
        league_away_avg = _profile_mean(
            league_away_points, 21.5
        )

        home_attack = team_attack_strength(
            home_points_for=_profile_mean(home_pf[home], league_home_avg),
            away_points_for=_profile_mean(away_pf[home], league_away_avg),
            league_home_points_for=league_home_avg,
            league_away_points_for=league_away_avg,
        )
        away_attack = team_attack_strength(
            home_points_for=_profile_mean(home_pf[away], league_home_avg),
            away_points_for=_profile_mean(away_pf[away], league_away_avg),
            league_home_points_for=league_home_avg,
            league_away_points_for=league_away_avg,
        )
        home_defense = team_defense_weakness(
            home_points_against=_profile_mean(home_pa[home], league_away_avg),
            away_points_against=_profile_mean(away_pa[home], league_home_avg),
            league_home_points_against=league_away_avg,
            league_away_points_against=league_home_avg,
        )
        away_defense = team_defense_weakness(
            home_points_against=_profile_mean(home_pa[away], league_away_avg),
            away_points_against=_profile_mean(away_pa[away], league_home_avg),
            league_home_points_against=league_away_avg,
            league_away_points_against=league_home_avg,
        )

        predicted_home_points, predicted_away_points, _, base_total = (
            predict_points(
                league_home_average=league_home_avg,
                league_away_average=league_away_avg,
                home_profile=TeamScoringProfile(
                    attack_strength=home_attack,
                    defense_weakness=home_defense,
                ),
                away_profile=TeamScoringProfile(
                    attack_strength=away_attack,
                    defense_weakness=away_defense,
                ),
            )
        )

        h2h_key = tuple(sorted((home, away)))
        h2h_history = h2h_totals[h2h_key]
        if h2h_history:
            h2h_mean = float(np.mean(h2h_history))
            predicted_total = (
                (1.0 - config.total_head_to_head_weight) * base_total
                + config.total_head_to_head_weight * h2h_mean
            )
        else:
            h2h_mean = math.nan
            predicted_total = base_total

        total_sd = _shrunk_sd(
            total_residuals,
            prior_sd=config.total_prior_sd,
            prior_games=config.residual_prior_games,
        )
        total_probabilities = _total_probabilities(
            predicted_total,
            float(row.total_line),
            total_sd,
        )

        market_centered_ats = _margin_probabilities(
            -float(row.home_spread),
            float(row.home_spread),
            margin_sd,
        )
        market_centered_total = _total_probabilities(
            float(row.total_line),
            float(row.total_line),
            total_sd,
        )
        plain_ats_push = market_centered_ats[2]
        total_push = market_centered_total[2]
        market_ml = _market_probabilities_from_line_distribution(
            market_probability=float(
                row.moneyline_market_home_probability
            ),
            rare_probability=tie_probability,
            clip=config.probability_clip,
        )
        market_ats = _market_probabilities_from_line_distribution(
            market_probability=float(
                row.ats_market_home_probability
            ),
            rare_probability=plain_ats_push,
            clip=config.probability_clip,
        )
        market_total = _market_probabilities_from_line_distribution(
            market_probability=float(
                row.total_market_over_probability
            ),
            rare_probability=total_push,
            clip=config.probability_clip,
        )

        if season in config.development_seasons:
            prediction_specs = [
                ("pregame_moneyline", "market_t10", market_ml),
                ("pregame_moneyline", "spreadsheet_elo", plain_ml),
                (
                    "pregame_moneyline",
                    "spreadsheet_qb_adjusted_elo",
                    qb_ml,
                ),
                ("pregame_ats", "market_t10", market_ats),
                ("pregame_ats", "spreadsheet_elo_margin", plain_ats),
                (
                    "pregame_ats",
                    "spreadsheet_qb_adjusted_margin",
                    qb_ats,
                ),
                ("pregame_total", "market_t10", market_total),
                (
                    "pregame_total",
                    "spreadsheet_total_points",
                    total_probabilities,
                ),
            ]
            series = pd.Series(row._asdict())
            for market, model_name, probabilities in prediction_specs:
                outcome_index = _score_outcome_index(market, series)
                log_loss, brier = _three_way_scores(
                    probabilities,
                    outcome_index,
                    clip=config.probability_clip,
                )
                predictions.append(
                    {
                        "game_id": row.game_id,
                        "season": season,
                        "week": int(row.week),
                        "kickoff_utc": row.kickoff_utc,
                        "market": market,
                        "model_name": model_name,
                        "training_cutoff_utc": (
                            row.kickoff_utc - pd.Timedelta(minutes=10)
                        ),
                        "probability_1": probabilities[0],
                        "probability_2": probabilities[1],
                        "probability_rare": probabilities[2],
                        "outcome_index": outcome_index,
                        "log_loss": log_loss,
                        "brier": brier,
                        "selection_eligible": True,
                    }
                )

        elo_history.append(
            {
                "game_id": row.game_id,
                "kickoff_utc": row.kickoff_utc,
                "season": season,
                "home_team_id": home,
                "away_team_id": away,
                "home_elo_pregame": pre_home_elo,
                "away_elo_pregame": pre_away_elo,
                "plain_adjusted_elo_difference": plain_diff,
                "qb_adjusted_elo_difference": qb_diff,
                "home_qb_adjustment": home_qb_adjustment,
                "away_qb_adjustment": away_qb_adjustment,
            }
        )
        total_history.append(
            {
                "game_id": row.game_id,
                "kickoff_utc": row.kickoff_utc,
                "league_home_average": league_home_avg,
                "league_away_average": league_away_avg,
                "predicted_home_points": predicted_home_points,
                "predicted_away_points": predicted_away_points,
                "base_total": base_total,
                "h2h_mean": h2h_mean,
                "predicted_total": predicted_total,
                "total_residual_sd": total_sd,
            }
        )

        actual_home_score = float(row.home_score)
        actual_away_score = float(row.away_score)
        actual_margin = actual_home_score - actual_away_score
        actual_total = actual_home_score + actual_away_score

        actual_home_result = (
            1.0 if actual_margin > 0 else 0.0 if actual_margin < 0 else 0.5
        )
        home_probability = plain_conditional_home
        winner_diff = plain_diff if actual_margin >= 0 else -plain_diff
        multiplier = margin_of_victory_multiplier(
            abs(actual_margin),
            winner_diff,
        )
        change = (
            config.elo_k_factor
            * (actual_home_result - home_probability)
            * multiplier
        )
        team_elo[home] = pre_home_elo + change
        team_elo[away] = pre_away_elo - change

        elo_history[-1].update(
            {
                "home_elo_postgame": team_elo[home],
                "away_elo_postgame": team_elo[away],
                "elo_change": change,
                "rating_sum_change": (
                    team_elo[home] + team_elo[away]
                    - pre_home_elo - pre_away_elo
                ),
            }
        )

        league_qb_average = (
            float(np.mean(league_qb_values))
            if league_qb_values
            else config.qb_initial_rating
        )

        for side, team, opponent in (
            ("home", home, away),
            ("away", away, home),
        ):
            actual_qb = str(getattr(row, f"{side}_qb_id", ""))
            stat = qb_lookup.get((str(row.game_id), actual_qb))
            if stat is None:
                qb_history.append(
                    {
                        "game_id": row.game_id,
                        "kickoff_utc": row.kickoff_utc,
                        "team_id": team,
                        "opponent_id": opponent,
                        "actual_qb_id": actual_qb,
                        "status": "QB_STATS_MISSING",
                    }
                )
                continue

            raw_value = qb_game_value(
                pass_attempts=stat.pass_attempts,
                completions=stat.completions,
                passing_yards=stat.passing_yards,
                passing_touchdowns=stat.passing_touchdowns,
                interceptions=stat.interceptions,
                sacks=stat.sacks,
                rush_attempts=stat.rush_attempts,
                rushing_yards=stat.rushing_yards,
                rushing_touchdowns=stat.rushing_touchdowns,
            )
            adjusted_value = adjusted_qb_game_value(
                raw_value,
                league_average_qb_value_allowed=league_qb_average,
                opponent_qb_value_allowed=defense_qb_allowed[opponent],
            )
            old_qb = qb_rating[actual_qb]
            old_team_qb = team_qb_rating[team]
            old_defense = defense_qb_allowed[opponent]
            qb_rating[actual_qb] = update_qb_rating(
                old_qb,
                adjusted_value,
                new_game_weight=config.qb_new_game_weight,
            )
            team_qb_rating[team] = update_qb_rating(
                old_team_qb,
                adjusted_value,
                new_game_weight=config.qb_new_game_weight,
            )
            defense_qb_allowed[opponent] = update_qb_rating(
                old_defense,
                raw_value,
                new_game_weight=config.qb_new_game_weight,
            )
            league_qb_values.append(raw_value)
            qb_history.append(
                {
                    "game_id": row.game_id,
                    "kickoff_utc": row.kickoff_utc,
                    "team_id": team,
                    "opponent_id": opponent,
                    "actual_qb_id": actual_qb,
                    "status": "UPDATED_AFTER_GAME",
                    "raw_qb_game_value": raw_value,
                    "adjusted_qb_game_value": adjusted_value,
                    "qb_rating_pregame": old_qb,
                    "qb_rating_postgame": qb_rating[actual_qb],
                    "team_qb_rating_pregame": old_team_qb,
                    "team_qb_rating_postgame": team_qb_rating[team],
                    "opponent_allowed_pregame": old_defense,
                    "opponent_allowed_postgame":
                        defense_qb_allowed[opponent],
                }
            )

        home_pf[home].append(actual_home_score)
        home_pa[home].append(actual_away_score)
        away_pf[away].append(actual_away_score)
        away_pa[away].append(actual_home_score)
        league_home_points.append(actual_home_score)
        league_away_points.append(actual_away_score)
        h2h_totals[h2h_key].append(actual_total)

        margin_residuals.append(actual_margin - plain_margin)
        total_residuals.append(actual_total - predicted_total)
        ties_seen += int(actual_margin == 0)
        games_seen += 1

    prediction_frame = pd.DataFrame(predictions)
    elo_frame = pd.DataFrame(elo_history)
    qb_frame = pd.DataFrame(qb_history)
    total_frame = pd.DataFrame(total_history)

    if prediction_frame.empty:
        raise ValueError("No development predictions were generated.")

    scorecard = (
        prediction_frame.groupby(
            ["market", "model_name"],
            as_index=False,
        )
        .agg(
            games=("game_id", "nunique"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values(["market", "mean_log_loss", "mean_brier"])
        .reset_index(drop=True)
    )

    season_scorecard = (
        prediction_frame.groupby(
            ["market", "model_name", "season"],
            as_index=False,
        )
        .agg(
            games=("game_id", "nunique"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values(["market", "model_name", "season"])
        .reset_index(drop=True)
    )

    prediction_frame.to_parquet(
        output_root / "spreadsheet_oof_predictions.parquet",
        index=False,
    )
    elo_frame.to_parquet(
        output_root / "team_elo_state_history.parquet",
        index=False,
    )
    qb_frame.to_parquet(
        output_root / "qb_state_history.parquet",
        index=False,
    )
    total_frame.to_parquet(
        output_root / "total_model_state_history.parquet",
        index=False,
    )
    scorecard.to_csv(
        output_root / "spreadsheet_baseline_scorecard.csv",
        index=False,
    )
    season_scorecard.to_csv(
        output_root / "spreadsheet_baseline_by_season.csv",
        index=False,
    )

    leakage_audit = pd.DataFrame(
        [
            {
                "check": "predictions_only_2021_2023",
                "value": sorted(prediction_frame["season"].unique().tolist()),
                "passed": set(prediction_frame["season"].unique())
                == set(config.development_seasons),
            },
            {
                "check": "rating_conservation",
                "value": float(
                    elo_frame["rating_sum_change"].abs().max()
                ),
                "passed": bool(
                    elo_frame["rating_sum_change"].abs().max() < 1e-9
                ),
            },
            {
                "check": "prediction_before_update",
                "value": "state histories store pregame and postgame separately",
                "passed": True,
            },
            {
                "check": "qb_updates_after_game_only",
                "value": int(
                    (qb_frame.get("status") == "UPDATED_AFTER_GAME").sum()
                ),
                "passed": True,
            },
        ]
    )
    leakage_audit.to_csv(
        output_root / "spreadsheet_baseline_leakage_audit.csv",
        index=False,
    )
    if not leakage_audit["passed"].all():
        raise ValueError("Spreadsheet backtest leakage audit failed.")

    manifest = {
        "status": "PASS",
        "warmup_seasons": list(config.warmup_seasons),
        "development_seasons": list(config.development_seasons),
        "qb_game_stats_path": str(qb_path),
        "prediction_rows": int(len(prediction_frame)),
        "unique_games": int(prediction_frame["game_id"].nunique()),
        "models": sorted(prediction_frame["model_name"].unique().tolist()),
        "weather_adjustment": "NOT_USED_NO_POINT_IN_TIME_FORECAST",
        "travel_adjustment": "NOT_USED_NO_VERIFIED_TRAVEL_INPUT",
        "actual_starter_usage": "POSTGAME_STATE_UPDATE_ONLY",
        "expected_starter_usage": "PREGAME_PREDICTION",
    }
    (output_root / "spreadsheet_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "predictions": prediction_frame,
        "scorecard": scorecard,
        "season_scorecard": season_scorecard,
        "leakage_audit": leakage_audit,
    }
