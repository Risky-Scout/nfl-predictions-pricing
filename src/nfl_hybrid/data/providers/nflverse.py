from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nfl_hybrid.data.provenance import utc_now_iso
from nfl_hybrid.data.team_ids import canonical_team_id, try_canonical_team_id
from nfl_hybrid.data.providers.base import ProviderResult


def _to_pandas(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _series(frame: pd.DataFrame, column: str, default: object = pd.NA) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _normalize_season_type(value: object) -> str:
    text = str(value).strip().upper()
    return {"WC": "POST", "DIV": "POST", "CON": "POST", "SB": "POST", "POST": "POST",
            "REG": "REG", "PRE": "PRE"}.get(text, text)


@dataclass
class NflverseAdapter:
    """Lazy wrapper around nflreadpy.

    nflreadpy returns Polars frames; this adapter converts them to pandas and
    adds source/as-of metadata. The package is imported only when a load method
    is called, keeping the model core usable without data-ingestion extras.
    """

    module: Any | None = None
    source_name: str = "nflverse"

    def _module(self) -> Any:
        if self.module is not None:
            return self.module
        try:
            import nflreadpy as nfl
        except ImportError as exc:
            raise RuntimeError(
                "Install data dependencies with `pip install -e '.[data]'` "
                "to use nflverse ingestion."
            ) from exc
        self.module = nfl
        return nfl

    def _load(self, function_name: str, seasons: Iterable[int] | bool) -> ProviderResult:
        module = self._module()
        loader = getattr(module, function_name)
        raw = _to_pandas(loader(list(seasons) if seasons is not True else True))
        retrieved = utc_now_iso()
        raw["source_name"] = self.source_name
        raw["source_retrieved_at_utc"] = retrieved
        return ProviderResult(
            data=raw,
            metadata={
                "source_name": self.source_name,
                "loader": function_name,
                "seasons": True if seasons is True else list(seasons),
                "source_retrieved_at_utc": retrieved,
            },
        )

    def load_schedules_raw(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_schedules", seasons)

    def load_games(self, seasons: Iterable[int]) -> ProviderResult:
        result = self.load_schedules_raw(seasons)
        raw = result.data
        if "datetime" in raw.columns:
            kickoff = pd.to_datetime(raw["datetime"], errors="coerce", utc=True)
            kickoff_basis = "nflverse_datetime"
        else:
            # nflverse schedules publish gameday plus gametime in U.S. Eastern
            # clock time. Localize before converting to UTC; parsing a bare
            # `20:20` value as a date would silently assign today's date.
            combined = (
                _series(raw, "gameday").astype("string").fillna("")
                + " "
                + _series(raw, "gametime").astype("string").fillna("")
            )
            local = pd.to_datetime(combined, errors="coerce")
            try:
                kickoff = local.dt.tz_localize(
                    "America/New_York",
                    ambiguous="NaT",
                    nonexistent="shift_forward",
                ).dt.tz_convert("UTC")
            except (TypeError, AttributeError):
                kickoff = pd.to_datetime(combined, errors="coerce", utc=True)
            kickoff_basis = "nflverse_gameday_gametime_eastern"

        frame = pd.DataFrame(
            {
                "game_id": _series(raw, "game_id"),
                "source_game_id": _series(raw, "game_id"),
                "season": pd.to_numeric(_series(raw, "season"), errors="coerce").astype("Int64"),
                "season_type": _series(raw, "game_type").map(_normalize_season_type),
                "week": _series(raw, "week"),
                "scheduled_kickoff_utc": kickoff,
                "kickoff_time_basis": kickoff_basis,
                "schedule_date": pd.to_datetime(_series(raw, "gameday"), errors="coerce"),
                "home_team_id": _series(raw, "home_team").map(try_canonical_team_id),
                "away_team_id": _series(raw, "away_team").map(try_canonical_team_id),
                "home_team_name": _series(raw, "home_team"),
                "away_team_name": _series(raw, "away_team"),
                "neutral_site": _series(raw, "location").astype("string").str.lower().eq("neutral"),
                "stadium_id": _series(raw, "stadium_id"),
                "stadium_name": _series(raw, "stadium"),
                "roof": _series(raw, "roof"),
                "surface": _series(raw, "surface"),
                "home_score": pd.to_numeric(_series(raw, "home_score"), errors="coerce"),
                "away_score": pd.to_numeric(_series(raw, "away_score"), errors="coerce"),
                "overtime": _series(raw, "overtime"),
                "home_rest_days": pd.to_numeric(_series(raw, "home_rest"), errors="coerce"),
                "away_rest_days": pd.to_numeric(_series(raw, "away_rest"), errors="coerce"),
                "division_game": _series(raw, "div_game"),
                "home_moneyline_reference": pd.to_numeric(
                    _series(raw, "home_moneyline"), errors="coerce"
                ),
                "away_moneyline_reference": pd.to_numeric(
                    _series(raw, "away_moneyline"), errors="coerce"
                ),
                # nflverse defines positive spread_line as home favored.
                # Canonical sportsbook home spread uses favorites as negative.
                "nflverse_spread_line_home_favored_positive": pd.to_numeric(
                    _series(raw, "spread_line"), errors="coerce"
                ),
                "home_spread_reference": -pd.to_numeric(
                    _series(raw, "spread_line"), errors="coerce"
                ),
                "home_spread_price_reference": pd.to_numeric(
                    _series(raw, "home_spread_odds"), errors="coerce"
                ),
                "away_spread_price_reference": pd.to_numeric(
                    _series(raw, "away_spread_odds"), errors="coerce"
                ),
                "total_line_reference": pd.to_numeric(
                    _series(raw, "total_line"), errors="coerce"
                ),
                "over_price_reference": pd.to_numeric(
                    _series(raw, "over_odds"), errors="coerce"
                ),
                "under_price_reference": pd.to_numeric(
                    _series(raw, "under_odds"), errors="coerce"
                ),
                "temperature_f_observed": pd.to_numeric(
                    _series(raw, "temp"), errors="coerce"
                ),
                "wind_mph_observed": pd.to_numeric(
                    _series(raw, "wind"), errors="coerce"
                ),
                "home_qb_id": _series(raw, "home_qb_id"),
                "away_qb_id": _series(raw, "away_qb_id"),
                "home_qb_name": _series(raw, "home_qb_name"),
                "away_qb_name": _series(raw, "away_qb_name"),
                "home_coach": _series(raw, "home_coach"),
                "away_coach": _series(raw, "away_coach"),
                "referee": _series(raw, "referee"),
                "line_timestamp_utc": pd.NaT,
                "line_timestamp_known": False,
                "line_reference_type": "nflverse_schedule_reference",
                "source_name": self.source_name,
                "source_retrieved_at_utc": result.metadata["source_retrieved_at_utc"],
            }
        )
        return ProviderResult(data=frame, metadata=result.metadata)

    def load_pbp(self, seasons: Iterable[int]) -> ProviderResult:
        result = self._load("load_pbp", seasons)
        keep = [
            "game_id", "play_id", "season", "season_type", "week", "posteam", "defteam",
            "home_team", "away_team", "qtr", "game_seconds_remaining",
            "half_seconds_remaining", "drive", "series", "down", "ydstogo",
            "yardline_100", "goal_to_go", "score_differential", "play_type",
            "no_play", "qb_kneel", "posteam_score", "posteam_score_post",
            "pass_attempt", "rush_attempt", "qb_dropback", "sack", "scramble",
            "complete_pass", "interception", "fumble", "fumble_lost", "touchdown",
            "first_down", "third_down_converted", "fourth_down_converted",
            "air_yards", "yards_after_catch", "passing_yards", "rushing_yards",
            "return_yards", "penalty", "penalty_yards", "epa", "wpa", "success",
            "cp", "cpoe", "passer_player_id", "rusher_player_id",
            "receiver_player_id", "kicker_player_id", "punter_player_id",
            "special_teams_play", "no_huddle", "shotgun", "time_of_possession",
            "source_name", "source_retrieved_at_utc",
        ]
        available = [column for column in keep if column in result.data.columns]
        return ProviderResult(data=result.data[available].copy(), metadata=result.metadata)

    def load_player_stats(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_player_stats", seasons)

    def load_team_stats(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_team_stats", seasons)

    def load_rosters(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_rosters", seasons)

    def load_weekly_rosters(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_rosters_weekly", seasons)

    def load_snap_counts(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_snap_counts", seasons)

    def load_depth_charts(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_depth_charts", seasons)

    def load_injuries(self, seasons: Iterable[int]) -> ProviderResult:
        return self._load("load_injuries", seasons)

    def load_players(self) -> ProviderResult:
        module = self._module()
        raw = _to_pandas(module.load_players())
        retrieved = utc_now_iso()
        raw["source_name"] = self.source_name
        raw["source_retrieved_at_utc"] = retrieved
        return ProviderResult(
            data=raw,
            metadata={
                "source_name": self.source_name,
                "loader": "load_players",
                "source_retrieved_at_utc": retrieved,
            },
        )
