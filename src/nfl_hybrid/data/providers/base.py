from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class ProviderResult:
    data: pd.DataFrame
    metadata: dict[str, object]


class GameProvider(Protocol):
    def load_games(self, seasons: list[int]) -> ProviderResult: ...


class OddsProvider(Protocol):
    def current_odds(self) -> ProviderResult: ...
    def historical_snapshot(self, snapshot_utc: str) -> ProviderResult: ...


class InjuryProvider(Protocol):
    def load_injuries(self, season: int, week: str | int) -> ProviderResult: ...
