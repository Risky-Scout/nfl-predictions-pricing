from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nfl_hybrid.legacy.elo import EloContext, LegacyElo


TEAM_ALIASES = {
    "Arizona Cardinals": "ARI",
    "Phoenix Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Houston Oilers": "TEN",
    "Tennessee Oilers": "TEN",
    "Tennessee Titans": "TEN",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "St. Louis Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Oakland Raiders": "LV",
    "Los Angeles Raiders": "LV",
    "Las Vegas Raiders": "LV",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Washington Redskins": "WAS",
    "Washington Football Team": "WAS",
    "Washington Commanders": "WAS",
}


def canonical_team_id(name: str) -> str:
    name = str(name)
    return TEAM_ALIASES.get(name, name.upper().replace(" ", "_"))


def derive_home_spread(
    favorite_id: str | None,
    spread_favorite: float | None,
    home_team_id: str,
    away_team_id: str | None = None,
) -> float:
    if spread_favorite is None or pd.isna(spread_favorite):
        return np.nan
    spread = float(spread_favorite)
    favorite = "" if favorite_id is None else str(favorite_id).strip().upper()

    if favorite in {"PICK", "PK", "EVEN", ""} or spread == 0:
        return 0.0

    # Match the raw ID first. This preserves Houston Texans (HOU), while still
    # allowing Houston Oilers (also HOU in the workbook) to map to Tennessee.
    if favorite == home_team_id:
        return spread
    if away_team_id is not None and favorite == away_team_id:
        return -spread

    id_aliases = {
        "OAK": "LV",
        "SD": "LAC",
        "STL": "LAR",
        "WSH": "WAS",
        "HOU": "TEN",
    }
    favorite = id_aliases.get(favorite, favorite)

    if favorite == home_team_id:
        return spread
    if away_team_id is not None and favorite == away_team_id:
        return -spread

    # The source spread is always the favorite's negative spread. If the
    # favorite ID is unavailable or stale, default to away-favorite sign only
    # after explicit matching has failed.
    return -spread


@dataclass
class RunningTeamState:
    games: int = 0
    home_games: int = 0
    away_games: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    home_points_for: float = 0.0
    home_points_against: float = 0.0
    away_points_for: float = 0.0
    away_points_against: float = 0.0
    ewma_points_for: float = 22.0
    ewma_points_against: float = 22.0
    ewma_margin: float = 0.0

    def update(self, points_for: float, points_against: float, *, at_home: bool, alpha: float) -> None:
        self.games += 1
        self.points_for += points_for
        self.points_against += points_against
        if at_home:
            self.home_games += 1
            self.home_points_for += points_for
            self.home_points_against += points_against
        else:
            self.away_games += 1
            self.away_points_for += points_for
            self.away_points_against += points_against

        self.ewma_points_for = alpha * points_for + (1.0 - alpha) * self.ewma_points_for
        self.ewma_points_against = alpha * points_against + (1.0 - alpha) * self.ewma_points_against
        self.ewma_margin = alpha * (points_for - points_against) + (1.0 - alpha) * self.ewma_margin


@dataclass
class LeagueState:
    games: int = 0
    home_points: float = 0.0
    away_points: float = 0.0

    @property
    def avg_home(self) -> float:
        return self.home_points / self.games if self.games else 22.0

    @property
    def avg_away(self) -> float:
        return self.away_points / self.games if self.games else 20.5


@dataclass
class PregameFeatureBuilder:
    """Sequential, leakage-safe feature builder.

    Every feature is emitted before the current game's score updates any state.
    """

    ewma_alpha: float = 0.18
    factor_prior_games: float = 8.0
    elo: LegacyElo = field(default_factory=LegacyElo)

    def _shrunk_ratio(self, numerator: float, denominator_games: int, league_average: float) -> float:
        prior = self.factor_prior_games
        estimate = (numerator + prior * league_average) / (denominator_games + prior)
        return estimate / league_average if league_average > 0 else 1.0

    def build(self, games: pd.DataFrame) -> pd.DataFrame:
        required = {
            "season", "home_team", "away_team", "home_score", "away_score",
            "playoff", "neutral_site", "favorite_id", "spread_favorite",
            "total_line", "temperature", "wind_mph", "humidity",
        }
        missing = required - set(games.columns)
        if missing:
            raise ValueError(f"Missing required game columns: {sorted(missing)}")

        ordered = games.copy()
        if "game_index" not in ordered:
            ordered["game_index"] = np.arange(len(ordered))
        ordered = ordered.sort_values(["season", "game_index"], kind="stable").reset_index(drop=True)

        team_states: dict[str, RunningTeamState] = defaultdict(RunningTeamState)
        league = LeagueState()
        h2h_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
        rows: list[dict[str, Any]] = []
        previous_season: int | None = None

        for game in ordered.itertuples(index=False):
            season = int(game.season)
            if previous_season is not None and season != previous_season:
                self.elo.regress_to_mean()
            previous_season = season

            home = canonical_team_id(game.home_team)
            away = canonical_team_id(game.away_team)
            hs = team_states[home]
            aws = team_states[away]

            context = EloContext(
                neutral_site=bool(game.neutral_site),
                playoff=bool(game.playoff),
            )
            elo_pred = self.elo.predict(home, away, context)

            league_home = league.avg_home
            league_away = league.avg_away

            home_attack = 0.5 * (
                self._shrunk_ratio(hs.home_points_for, hs.home_games, league_home)
                + self._shrunk_ratio(hs.away_points_for, hs.away_games, league_away)
            )
            away_attack = 0.5 * (
                self._shrunk_ratio(aws.home_points_for, aws.home_games, league_home)
                + self._shrunk_ratio(aws.away_points_for, aws.away_games, league_away)
            )

            # Points allowed at home are compared with the league away baseline.
            home_def_weak = 0.5 * (
                self._shrunk_ratio(hs.home_points_against, hs.home_games, league_away)
                + self._shrunk_ratio(hs.away_points_against, hs.away_games, league_home)
            )
            away_def_weak = 0.5 * (
                self._shrunk_ratio(aws.home_points_against, aws.home_games, league_away)
                + self._shrunk_ratio(aws.away_points_against, aws.away_games, league_home)
            )

            legacy_home_points = league_home * home_attack * away_def_weak
            legacy_away_points = league_away * away_attack * home_def_weak
            matchup_key = tuple(sorted((home, away)))
            previous_h2h = h2h_totals[matchup_key]
            h2h_mean = float(np.mean(previous_h2h)) if previous_h2h else np.nan

            home_spread = derive_home_spread(
                getattr(game, "favorite_id"),
                getattr(game, "spread_favorite"),
                home,
                away,
            )

            home_score = float(game.home_score)
            away_score = float(game.away_score)
            margin = home_score - away_score
            total = home_score + away_score

            row = {
                "game_index": int(game.game_index),
                "season": season,
                "week": getattr(game, "week", None),
                "playoff": int(bool(game.playoff)),
                "neutral_site": int(bool(game.neutral_site)),
                "home_team_id": home,
                "away_team_id": away,
                "home_spread": home_spread,
                "total_line": float(game.total_line) if not pd.isna(game.total_line) else np.nan,
                "temperature": float(game.temperature) if not pd.isna(game.temperature) else np.nan,
                "wind_mph": float(game.wind_mph) if not pd.isna(game.wind_mph) else np.nan,
                "humidity": float(game.humidity) if not pd.isna(game.humidity) else np.nan,
                "dome": int(str(getattr(game, "weather_detail", "")).upper() == "DOME"),
                "home_elo": elo_pred.home_rating,
                "away_elo": elo_pred.away_rating,
                "elo_difference_adjusted": elo_pred.adjusted_difference,
                "legacy_home_win_probability": elo_pred.home_win_probability,
                "legacy_expected_margin": elo_pred.expected_home_margin,
                "league_home_points": league_home,
                "league_away_points": league_away,
                "home_attack": home_attack,
                "away_attack": away_attack,
                "home_defense_weakness": home_def_weak,
                "away_defense_weakness": away_def_weak,
                "legacy_expected_home_points": legacy_home_points,
                "legacy_expected_away_points": legacy_away_points,
                "legacy_expected_total": legacy_home_points + legacy_away_points,
                "h2h_prior_games": len(previous_h2h),
                "h2h_prior_total": h2h_mean,
                "home_games_prior": hs.games,
                "away_games_prior": aws.games,
                "home_ewma_points_for": hs.ewma_points_for,
                "home_ewma_points_against": hs.ewma_points_against,
                "home_ewma_margin": hs.ewma_margin,
                "away_ewma_points_for": aws.ewma_points_for,
                "away_ewma_points_against": aws.ewma_points_against,
                "away_ewma_margin": aws.ewma_margin,
                "home_score": home_score,
                "away_score": away_score,
                "home_margin": margin,
                "total_points": total,
                "home_win": float(margin > 0) if margin != 0 else 0.5,
                "home_cover": np.nan if pd.isna(home_spread) or margin + home_spread == 0
                    else float(margin + home_spread > 0),
                "ats_push": int(not pd.isna(home_spread) and margin + home_spread == 0),
                "over": np.nan if pd.isna(game.total_line) or total - float(game.total_line) == 0
                    else float(total > float(game.total_line)),
                "total_push": int(not pd.isna(game.total_line) and total - float(game.total_line) == 0),
            }
            rows.append(row)

            hs.update(home_score, away_score, at_home=True, alpha=self.ewma_alpha)
            aws.update(away_score, home_score, at_home=False, alpha=self.ewma_alpha)
            league.games += 1
            league.home_points += home_score
            league.away_points += away_score
            h2h_totals[matchup_key].append(total)
            self.elo.update(home, away, home_score, away_score, context, completed=True)

        return pd.DataFrame(rows)
