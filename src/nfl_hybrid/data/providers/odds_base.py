from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

import pandas as pd

from nfl_hybrid.data.providers.base import ProviderResult


class ProviderAgnosticOddsAdapter(ABC):
    """Canonical interface for timestamped Moneyline, spread, and total odds."""

    @abstractmethod
    def current_odds(self) -> ProviderResult:
        raise NotImplementedError

    @abstractmethod
    def historical_snapshot(self, snapshot_utc: str | datetime) -> ProviderResult:
        raise NotImplementedError

    @abstractmethod
    def flatten_response(self, payload: dict[str, object] | list[object]) -> pd.DataFrame:
        raise NotImplementedError


def american_to_decimal(price_american: float | int | None) -> float | None:
    if price_american is None or pd.isna(price_american):
        return None
    price = float(price_american)
    if price == 0:
        return None
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def american_to_implied(price_american: float | int | None) -> float | None:
    decimal = american_to_decimal(price_american)
    if decimal is None or decimal <= 1.0:
        return None
    return 1.0 / decimal
