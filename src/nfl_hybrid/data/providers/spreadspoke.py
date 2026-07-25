from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_hybrid.data.io import read_tabular
from nfl_hybrid.data.provenance import SourceManifest, utc_now_iso
from nfl_hybrid.data.team_ids import canonical_team_id, try_canonical_team_id
from nfl_hybrid.data.providers.base import ProviderResult


_REQUIRED_CORE = {
    "schedule_date",
    "schedule_season",
    "team_home",
    "score_home",
    "team_away",
    "score_away",
    "team_favorite_id",
    "spread_favorite",
    "over_under_line",
}


def _as_bool(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "neutral", "post"}
    return bool(value)


def _season_type(playoff: object) -> str:
    return "POST" if _as_bool(playoff) else "REG"


def _week_value(row: pd.Series) -> object:
    for column in ("schedule_week", "schedule_week_str"):
        if column in row and pd.notna(row[column]):
            return row[column]
    return pd.NA


def _home_spread(
    favorite_raw: object,
    favorite_spread: object,
    home_id: str,
    away_id: str,
) -> float:
    if pd.isna(favorite_spread):
        return np.nan
    spread = float(favorite_spread)
    favorite = try_canonical_team_id(
        None if pd.isna(favorite_raw) else str(favorite_raw),
        allow_pick=True,
    )
    if favorite == "PICK" or spread == 0:
        return 0.0
    if favorite == home_id:
        return spread
    if favorite == away_id:
        return -spread
    return np.nan


@dataclass
class SpreadspokeAdapter:
    locator: str | Path
    source_name: str = "spreadspoke"

    def load_raw(self) -> ProviderResult:
        frame = read_tabular(self.locator)
        missing = sorted(_REQUIRED_CORE - set(frame.columns))
        if missing:
            raise ValueError(f"Spreadspoke file is missing required columns: {missing}")
        retrieved = utc_now_iso()
        return ProviderResult(
            data=frame,
            metadata={
                "source_name": self.source_name,
                "source_locator": str(self.locator),
                "source_retrieved_at_utc": retrieved,
            },
        )

    def load_games(self, seasons: Iterable[int] | None = None) -> ProviderResult:
        result = self.load_raw()
        raw = result.data.copy()
        if seasons is not None:
            seasons_set = {int(x) for x in seasons}
            raw = raw[pd.to_numeric(raw["schedule_season"], errors="coerce").isin(seasons_set)]

        retrieved = str(result.metadata["source_retrieved_at_utc"])
        rows: list[dict[str, object]] = []
        for source_index, row in raw.iterrows():
            home_id = (
                try_canonical_team_id(row.get("home_team_id"))
                or canonical_team_id(str(row["team_home"]))
            )
            away_id = (
                try_canonical_team_id(row.get("away_team_id"))
                or canonical_team_id(str(row["team_away"]))
            )
            season = int(row["schedule_season"])
            date = pd.to_datetime(row["schedule_date"], errors="coerce")
            week = _week_value(row)
            home_score = pd.to_numeric(row["score_home"], errors="coerce")
            away_score = pd.to_numeric(row["score_away"], errors="coerce")
            home_spread = (
                pd.to_numeric(row.get("spread_home"), errors="coerce")
                if "spread_home" in raw.columns
                else _home_spread(
                    row.get("team_favorite_id"),
                    row.get("spread_favorite"),
                    home_id,
                    away_id,
                )
            )
            game_id = f"spreadspoke_{season}_{source_index:05d}"
            if pd.notna(date):
                game_id = f"spreadspoke_{date.strftime('%Y%m%d')}_{away_id}_{home_id}"

            rows.append(
                {
                    "game_id": game_id,
                    "source_game_id": row.get("game_id", pd.NA),
                    "season": season,
                    "season_type": _season_type(row.get("schedule_playoff", False)),
                    "week": week,
                    "scheduled_kickoff_utc": pd.NaT,
                    "schedule_date": date,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_team_name": row["team_home"],
                    "away_team_name": row["team_away"],
                    "neutral_site": _as_bool(row.get("stadium_neutral", False)),
                    "stadium_name": row.get("stadium", pd.NA),
                    "surface": row.get("stadium_surface", pd.NA),
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_spread_reference": home_spread,
                    "total_line_reference": pd.to_numeric(
                        row.get("over_under_line"), errors="coerce"
                    ),
                    "favorite_team_id": try_canonical_team_id(
                        row.get("team_favorite_id"), allow_pick=True
                    ),
                    "temperature_f_observed": pd.to_numeric(
                        row.get("weather_temperature"), errors="coerce"
                    ),
                    "wind_mph_observed": pd.to_numeric(
                        row.get("weather_wind_mph"), errors="coerce"
                    ),
                    "humidity_observed": pd.to_numeric(
                        row.get("weather_humidity"), errors="coerce"
                    ),
                    "weather_detail_observed": row.get("weather_detail", pd.NA),
                    "line_timestamp_utc": pd.NaT,
                    "line_timestamp_known": False,
                    "line_reference_type": "spreadspoke_game_reference",
                    "source_row_number": int(source_index),
                    "source_name": self.source_name,
                    "source_retrieved_at_utc": retrieved,
                }
            )

        frame = pd.DataFrame(rows)
        return ProviderResult(data=frame, metadata=result.metadata)
