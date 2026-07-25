from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Iterable

import pandas as pd

from nfl_hybrid.data.provenance import utc_now_iso
from nfl_hybrid.data.providers.base import ProviderResult
from nfl_hybrid.data.providers.odds_base import (
    ProviderAgnosticOddsAdapter,
    american_to_decimal,
    american_to_implied,
)
from nfl_hybrid.data.team_ids import try_canonical_team_id


@dataclass(frozen=True)
class OddsAPIConfig:
    api_key: str | None = None
    api_key_env: str = "THE_ODDS_API_KEY"
    base_url: str = "https://api.the-odds-api.com/v4"
    sport_key: str = "americanfootball_nfl"
    regions: tuple[str, ...] = ("us",)
    markets: tuple[str, ...] = ("h2h", "spreads", "totals")
    bookmakers: tuple[str, ...] = ()
    odds_format: str = "american"
    date_format: str = "iso"
    timeout_seconds: int = 60

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"The Odds API key is missing. Set {self.api_key_env} or pass api_key."
            )
        return key


def _as_iso(value: str | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _market_type(market_key: str) -> str:
    return {"h2h": "moneyline", "spreads": "spread", "totals": "total"}.get(
        market_key, market_key
    )


def _outcome_side(
    *,
    market_key: str,
    outcome_name: str,
    home_team: str,
    away_team: str,
) -> str:
    if market_key == "totals":
        text = outcome_name.strip().lower()
        if text == "over":
            return "over"
        if text == "under":
            return "under"
    if outcome_name == home_team:
        return "home"
    if outcome_name == away_team:
        return "away"
    if outcome_name.strip().lower() in {"draw", "tie"}:
        return "draw"
    return "unknown"


class TheOddsAPIAdapter(ProviderAgnosticOddsAdapter):
    source_name = "the_odds_api"

    def __init__(self, config: OddsAPIConfig | None = None, session: Any | None = None) -> None:
        self.config = config or OddsAPIConfig()
        self._session = session

    def _requests(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Install data dependencies with `pip install -e '.[data]'` "
                "to call The Odds API."
            ) from exc
        self._session = requests.Session()
        return self._session

    def _params(self) -> dict[str, str]:
        params = {
            "apiKey": self.config.resolved_api_key(),
            "regions": ",".join(self.config.regions),
            "markets": ",".join(self.config.markets),
            "oddsFormat": self.config.odds_format,
            "dateFormat": self.config.date_format,
        }
        if self.config.bookmakers:
            params["bookmakers"] = ",".join(self.config.bookmakers)
        return params

    def _get(self, path: str, params: dict[str, str]) -> tuple[Any, dict[str, str]]:
        response = self._requests().get(
            f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}",
            params=params,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower().startswith("x-requests-")
        }
        return response.json(), headers

    def current_odds(self) -> ProviderResult:
        payload, headers = self._get(
            f"sports/{self.config.sport_key}/odds",
            self._params(),
        )
        frame = self.flatten_response(payload)
        return ProviderResult(
            data=frame,
            metadata={
                "source_name": self.source_name,
                "endpoint": "current_odds",
                "sport_key": self.config.sport_key,
                **headers,
            },
        )

    def historical_snapshot(self, snapshot_utc: str | datetime) -> ProviderResult:
        requested = _as_iso(snapshot_utc)
        params = self._params()
        params["date"] = requested
        payload, headers = self._get(
            f"historical/sports/{self.config.sport_key}/odds",
            params,
        )
        frame = self.flatten_response(payload)
        return ProviderResult(
            data=frame,
            metadata={
                "source_name": self.source_name,
                "endpoint": "historical_odds",
                "requested_snapshot_utc": requested,
                "returned_snapshot_utc": payload.get("timestamp")
                if isinstance(payload, dict) else None,
                "previous_snapshot_utc": payload.get("previous_timestamp")
                if isinstance(payload, dict) else None,
                "next_snapshot_utc": payload.get("next_timestamp")
                if isinstance(payload, dict) else None,
                "sport_key": self.config.sport_key,
                **headers,
            },
        )

    def flatten_response(self, payload: dict[str, object] | list[object]) -> pd.DataFrame:
        retrieved = utc_now_iso()
        if isinstance(payload, dict) and "data" in payload:
            snapshot = payload.get("timestamp")
            events = payload.get("data") or []
        else:
            snapshot = retrieved
            events = payload

        rows: list[dict[str, object]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("id")
            home_team = str(event.get("home_team") or "")
            away_team = str(event.get("away_team") or "")
            commence = event.get("commence_time")
            event_snapshot = snapshot or event.get("last_update") or retrieved
            for bookmaker in event.get("bookmakers", []) or []:
                if not isinstance(bookmaker, dict):
                    continue
                bookmaker_id = bookmaker.get("key")
                bookmaker_name = bookmaker.get("title")
                bookmaker_last_update = bookmaker.get("last_update")
                for market in bookmaker.get("markets", []) or []:
                    if not isinstance(market, dict):
                        continue
                    market_key = str(market.get("key") or "")
                    market_last_update = market.get("last_update")
                    for outcome in market.get("outcomes", []) or []:
                        if not isinstance(outcome, dict):
                            continue
                        outcome_name = str(outcome.get("name") or "")
                        price = pd.to_numeric(outcome.get("price"), errors="coerce")
                        line_value = pd.to_numeric(outcome.get("point"), errors="coerce")
                        side = _outcome_side(
                            market_key=market_key,
                            outcome_name=outcome_name,
                            home_team=home_team,
                            away_team=away_team,
                        )
                        snapshot_ts = pd.to_datetime(event_snapshot, utc=True, errors="coerce")
                        commence_ts = pd.to_datetime(commence, utc=True, errors="coerce")
                        is_live = (
                            bool(snapshot_ts >= commence_ts)
                            if pd.notna(snapshot_ts) and pd.notna(commence_ts)
                            else False
                        )
                        rows.append(
                            {
                                "provider_event_id": event_id,
                                "game_id": pd.NA,
                                "sport_key": event.get("sport_key", self.config.sport_key),
                                "sport_title": event.get("sport_title"),
                                "home_team_name": home_team,
                                "away_team_name": away_team,
                                "home_team_id": try_canonical_team_id(home_team),
                                "away_team_id": try_canonical_team_id(away_team),
                                "commence_time_utc": commence_ts,
                                "snapshot_utc": snapshot_ts,
                                "minutes_to_kickoff": (
                                    (commence_ts - snapshot_ts).total_seconds() / 60.0
                                    if pd.notna(snapshot_ts) and pd.notna(commence_ts)
                                    else pd.NA
                                ),
                                "bookmaker_id": bookmaker_id,
                                "bookmaker_name": bookmaker_name,
                                "bookmaker_region": ",".join(self.config.regions),
                                "bookmaker_last_update_utc": pd.to_datetime(
                                    market_last_update or bookmaker_last_update,
                                    utc=True,
                                    errors="coerce",
                                ),
                                "market_key_raw": market_key,
                                "market_type": _market_type(market_key),
                                "outcome_name": outcome_name,
                                "outcome_side": side,
                                "line_value": line_value,
                                "price_american": price,
                                "price_decimal": american_to_decimal(price),
                                "raw_implied_probability": american_to_implied(price),
                                "is_live": is_live,
                                "source_name": self.source_name,
                                "source_retrieved_at_utc": retrieved,
                            }
                        )
        return pd.DataFrame(rows)

    @staticmethod
    def estimate_historical_credits(
        number_of_snapshots: int,
        *,
        region_count: int = 1,
        market_count: int = 3,
        credits_per_region_market: int = 10,
    ) -> int:
        return (
            int(number_of_snapshots)
            * int(region_count)
            * int(market_count)
            * int(credits_per_region_market)
        )
