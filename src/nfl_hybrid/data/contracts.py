from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import pandas as pd


class ContractViolation(ValueError):
    """Raised when a provider frame violates a canonical table contract."""


@dataclass(frozen=True)
class DataContract:
    name: str
    required_columns: tuple[str, ...]
    unique_key: tuple[str, ...] = ()
    non_null_columns: tuple[str, ...] = ()
    allowed_values: Mapping[str, frozenset[object]] = field(default_factory=dict)

    def validate(self, frame: pd.DataFrame, *, strict: bool = True) -> list[str]:
        problems: list[str] = []
        missing = sorted(set(self.required_columns) - set(frame.columns))
        if missing:
            problems.append(f"missing columns: {missing}")
            if strict:
                raise ContractViolation(f"{self.name}: " + "; ".join(problems))
            return problems

        for column in self.non_null_columns:
            null_count = int(frame[column].isna().sum())
            if null_count:
                problems.append(f"{column}: {null_count} null values")

        if self.unique_key:
            duplicate_count = int(frame.duplicated(list(self.unique_key), keep=False).sum())
            if duplicate_count:
                problems.append(
                    f"unique key {self.unique_key}: {duplicate_count} rows participate in duplicates"
                )

        for column, allowed in self.allowed_values.items():
            values = set(frame[column].dropna().unique())
            unexpected = sorted(values - set(allowed), key=str)
            if unexpected:
                problems.append(f"{column}: unexpected values {unexpected[:20]}")

        if strict and problems:
            raise ContractViolation(f"{self.name}: " + "; ".join(problems))
        return problems


GAMES = DataContract(
    name="games",
    required_columns=(
        "game_id", "season", "season_type", "week", "scheduled_kickoff_utc",
        "home_team_id", "away_team_id", "neutral_site", "home_score", "away_score",
        "source_name", "source_retrieved_at_utc",
    ),
    unique_key=("game_id",),
    non_null_columns=("game_id", "season", "home_team_id", "away_team_id", "source_name"),
    allowed_values={"season_type": frozenset({"PRE", "REG", "POST"})},
)

ODDS_SNAPSHOTS = DataContract(
    name="odds_snapshots",
    required_columns=(
        "provider_event_id", "game_id", "sport_key", "bookmaker_id",
        "market_type", "outcome_side", "outcome_name", "line_value",
        "price_american", "snapshot_utc", "commence_time_utc",
        "bookmaker_last_update_utc", "is_live", "source_name",
        "source_retrieved_at_utc",
    ),
    unique_key=(
        "provider_event_id", "bookmaker_id", "market_type", "outcome_name",
        "line_value", "snapshot_utc",
    ),
    non_null_columns=(
        "provider_event_id", "bookmaker_id", "market_type", "outcome_name",
        "price_american", "snapshot_utc",
    ),
    allowed_values={
        "market_type": frozenset({"moneyline", "spread", "total"}),
        "outcome_side": frozenset({"home", "away", "over", "under", "draw", "unknown"}),
    },
)

INJURIES = DataContract(
    name="injuries",
    required_columns=(
        "season", "week", "team_id", "player_id", "player_name", "position",
        "injury_primary", "practice_status", "game_status", "status_date_utc",
        "source_name", "source_retrieved_at_utc",
    ),
    non_null_columns=("season", "week", "team_id", "player_name", "source_name"),
)

PLAYS = DataContract(
    name="plays",
    required_columns=(
        "game_id", "play_id", "season", "week", "posteam", "defteam", "qtr",
        "game_seconds_remaining", "down", "ydstogo", "yardline_100", "play_type",
        "epa", "wpa", "success", "source_name", "source_retrieved_at_utc",
    ),
    unique_key=("game_id", "play_id"),
    non_null_columns=("game_id", "play_id", "season", "source_name"),
)

CANONICAL_CONTRACTS = {
    "games": GAMES,
    "odds_snapshots": ODDS_SNAPSHOTS,
    "injuries": INJURIES,
    "plays": PLAYS,
}


def validate_frame(
    table_name: str,
    frame: pd.DataFrame,
    *,
    strict: bool = True,
) -> list[str]:
    if table_name not in CANONICAL_CONTRACTS:
        raise KeyError(f"Unknown canonical table: {table_name}")
    return CANONICAL_CONTRACTS[table_name].validate(frame, strict=strict)
