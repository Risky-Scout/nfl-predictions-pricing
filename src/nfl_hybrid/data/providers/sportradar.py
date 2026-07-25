from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable

import pandas as pd

from nfl_hybrid.data.provenance import utc_now_iso
from nfl_hybrid.data.team_ids import try_canonical_team_id
from nfl_hybrid.data.providers.base import ProviderResult


@dataclass(frozen=True)
class SportradarConfig:
    api_key: str | None = None
    api_key_env: str = "SPORTRADAR_API_KEY"
    base_url: str = "https://api.sportradar.com/nfl/official"
    access_level: str = "trial"
    version: str = "v7"
    language_code: str = "en"
    timeout_seconds: int = 60

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Sportradar key is missing. Set {self.api_key_env} or pass api_key."
            )
        return key


def _iter_teams(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    # Common Sportradar responses use `teams`. This recursive fallback keeps
    # the parser compatible with wrappers that nest teams under week/league.
    teams = payload.get("teams")
    if isinstance(teams, list):
        yield from (team for team in teams if isinstance(team, dict))
        return
    for value in payload.values():
        if isinstance(value, dict):
            yield from _iter_teams(value)


def _players_for_team(team: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("players", "injuries"):
        value = team.get(key)
        if isinstance(value, list):
            return [player for player in value if isinstance(player, dict)]
    return []


def _normalize_practice(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    mapping = {
        "DID NOT PARTICIPATE IN PRACTICE": "DNP",
        "DID NOT PARTICIPATE": "DNP",
        "LIMITED PARTICIPATION IN PRACTICE": "LIMITED",
        "LIMITED PARTICIPATION": "LIMITED",
        "FULL PARTICIPATION IN PRACTICE": "FULL",
        "FULL PARTICIPATION": "FULL",
        "": "NOT_LISTED",
    }
    return mapping.get(text, text.replace(" ", "_"))


def _normalize_game(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    return {
        "": "NO_DESIGNATION",
        "QUESTIONABLE": "QUESTIONABLE",
        "DOUBTFUL": "DOUBTFUL",
        "OUT": "OUT",
    }.get(text, text.replace(" ", "_"))


class SportradarInjuryAdapter:
    source_name = "sportradar"

    def __init__(self, config: SportradarConfig | None = None, session: Any | None = None) -> None:
        self.config = config or SportradarConfig()
        self._session = session

    def _requests(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Install data dependencies with `pip install -e '.[data]'` "
                "to call Sportradar."
            ) from exc
        self._session = requests.Session()
        return self._session

    def endpoint(self, season: int, season_type: str, week: int | str) -> str:
        return (
            f"{self.config.base_url.rstrip('/')}/{self.config.access_level}/"
            f"{self.config.version}/{self.config.language_code}/seasons/"
            f"{int(season)}/{season_type.upper()}/{week}/injuries.json"
        )

    def load_injuries(
        self,
        season: int,
        week: int | str,
        *,
        season_type: str = "REG",
    ) -> ProviderResult:
        url = self.endpoint(season, season_type, week)
        response = self._requests().get(
            url,
            params={"api_key": self.config.resolved_api_key()},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        frame = self.flatten_response(
            response.json(),
            season=season,
            week=week,
            season_type=season_type,
        )
        return ProviderResult(
            data=frame,
            metadata={
                "source_name": self.source_name,
                "endpoint": url,
                "season": int(season),
                "season_type": season_type.upper(),
                "week": str(week),
                "source_retrieved_at_utc": frame["source_retrieved_at_utc"].iloc[0]
                if len(frame) else utc_now_iso(),
            },
        )

    def flatten_response(
        self,
        payload: dict[str, Any],
        *,
        season: int | None = None,
        week: int | str | None = None,
        season_type: str | None = None,
    ) -> pd.DataFrame:
        retrieved = utc_now_iso()
        season_node = payload.get("season") if isinstance(payload.get("season"), dict) else {}
        week_node = payload.get("week") if isinstance(payload.get("week"), dict) else {}
        season_value = season if season is not None else season_node.get("year")
        week_value = week if week is not None else week_node.get("sequence")
        season_type_value = season_type or season_node.get("type") or season_node.get("name")

        rows: list[dict[str, object]] = []
        for team in _iter_teams(payload):
            team_alias = team.get("alias") or team.get("abbreviation")
            team_id = try_canonical_team_id(team_alias)
            for player in _players_for_team(team):
                injury = player.get("injury") if isinstance(player.get("injury"), dict) else {}
                practice = (
                    injury.get("practice")
                    if isinstance(injury.get("practice"), dict)
                    else {}
                )
                game_status = injury.get("status") or player.get("status")
                rows.append(
                    {
                        "season": int(season_value) if season_value is not None else pd.NA,
                        "season_type": season_type_value,
                        "week": str(week_value) if week_value is not None else pd.NA,
                        "team_id": team_id,
                        "provider_team_id": team.get("id"),
                        "provider_team_sr_id": team.get("sr_id"),
                        "team_market": team.get("market"),
                        "team_name": team.get("name"),
                        "player_id": player.get("id"),
                        "player_sr_id": player.get("sr_id"),
                        "player_name": player.get("name"),
                        "jersey": player.get("jersey"),
                        "position": player.get("position"),
                        "injury_primary": injury.get("primary"),
                        "practice_status_raw": practice.get("status"),
                        "practice_status": _normalize_practice(practice.get("status")),
                        "game_status_raw": game_status,
                        "game_status": _normalize_game(game_status),
                        "status_date_utc": pd.to_datetime(
                            injury.get("status_date"),
                            utc=True,
                            errors="coerce",
                        ),
                        "estimated_return_date": pd.to_datetime(
                            injury.get("estimated_return_date"),
                            utc=True,
                            errors="coerce",
                        ),
                        "source_name": self.source_name,
                        "source_retrieved_at_utc": retrieved,
                    }
                )
        return pd.DataFrame(rows)
