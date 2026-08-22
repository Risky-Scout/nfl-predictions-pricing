"""A small, boring BALLDONTLIE (BDL) NFL API client.

Verified live against the current production API on 2026-08-21 (games,
team_stats, player_injuries, plays, teams/roster, advanced_stats, stats,
odds all reachable with a GOAT-tier key). Base URL, cursor pagination shape
(``meta.next_cursor`` / ``meta.prev_cursor``), and the ``status_state``
lifecycle enum below all match that live verification -- this is not a
best-guess mapping from third-party documentation alone.

Responsibilities kept deliberately narrow (see the module docstring in
``__init__.py`` for where each later stage lives): authenticated requests,
pagination to exhaustion, timeouts, clear HTTP/rate-limit exceptions, and
JSON shape sanity checks. No canonicalization, no finality logic, no
retries that can loop indefinitely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import os

from nfl_hybrid.data.provenance import utc_now_iso

DEFAULT_BASE_URL = "https://api.balldontlie.io/nfl/v1"
DEFAULT_API_KEY_ENV = "BALLDONTLIE_API_KEY"

# The exact status_state enum BDL documents and that a live /games call
# returned on 2026-08-21 ("scheduled" for 2026 week 1 fixtures, "final" for
# completed 2025 games). See finality.py for how this drives result
# eligibility -- this module only transports the raw value.
KNOWN_STATUS_STATES = frozenset(
    {
        "scheduled",
        "in_progress",
        "final",
        "postponed",
        "canceled",
        "delayed",
        "suspended",
        "abandoned",
        "unknown",
    }
)


class BDLError(RuntimeError):
    """Base class for all BDL client errors."""


class BDLConfigError(BDLError):
    """Client misconfiguration (e.g. missing API key)."""


class BDLHTTPError(BDLError):
    """A non-2xx HTTP response from BDL that is not a rate-limit error."""

    def __init__(self, status_code: int, endpoint: str, body: str) -> None:
        self.status_code = status_code
        self.endpoint = endpoint
        self.body = body
        super().__init__(f"BDL request to {endpoint!r} failed with HTTP {status_code}: {body[:500]}")


class BDLRateLimitError(BDLHTTPError):
    """HTTP 429. No hidden retry -- the caller decides whether/when to retry."""

    def __init__(self, endpoint: str, body: str, retry_after_seconds: float | None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(429, endpoint, body)


class BDLSchemaError(BDLError):
    """The response body was not the JSON shape this client expects."""


class BDLPaginationError(BDLError):
    """Pagination did not terminate cleanly (repeated/looping cursor, or the
    configured max-page safety cap was exceeded without exhausting the
    provider's pagination)."""


@dataclass(frozen=True)
class BDLResponse:
    """One fully-paginated, raw (not yet canonicalized) BDL result."""

    endpoint: str
    request_params: dict[str, Any]
    data: list[dict[str, Any]]
    pages_fetched: int
    requested_at_utc: str
    received_at_utc: str


@dataclass(frozen=True)
class BDLConfig:
    api_key: str | None = None
    api_key_env: str = DEFAULT_API_KEY_ENV
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0
    max_page_size: int = 100
    # Hard safety cap on pages per logical request -- pagination must be
    # PROVEN to exhaust, but an adversarial/broken cursor must never spin
    # forever. 500 pages * 100/page = 50,000 rows, far beyond any single
    # season/game query this client issues.
    max_pages: int = 500

    def resolved_api_key(self) -> str:
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise BDLConfigError(
                f"BALLDONTLIE key is missing. Set {self.api_key_env} or pass api_key. "
                "Never hardcode it in source."
            )
        return key


def _flatten_params(params: dict[str, Any]) -> list[tuple[str, str]]:
    """BDL array params use repeated ``key[]=value`` query args."""
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                pairs.append((f"{key}[]", str(item)))
        else:
            pairs.append((key, str(value)))
    return pairs


class BDLClient:
    """Authenticated, paginating BDL NFL client. Returns raw JSON rows --
    canonicalization is a separate stage (see ``canonical.py`` / ``plays.py``)."""

    provider_name = "balldontlie"

    def __init__(self, config: BDLConfig | None = None, session: Any | None = None) -> None:
        self.config = config or BDLConfig()
        self._session = session

    def _requests(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:
            raise BDLConfigError(
                "Install data dependencies with `pip install -e '.[data]'` to call BALLDONTLIE."
            ) from exc
        self._session = requests.Session()
        return self._session

    def _headers(self) -> dict[str, str]:
        # Never log or return this dict anywhere it could be printed.
        return {"Authorization": self.config.resolved_api_key()}

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = self._requests().get(
            url,
            params=_flatten_params(params),
            headers=self._headers(),
            timeout=self.config.timeout_seconds,
        )
        status = response.status_code
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            raise BDLRateLimitError(
                endpoint,
                response.text,
                float(retry_after) if retry_after else None,
            )
        if status >= 400:
            raise BDLHTTPError(status, endpoint, response.text)
        try:
            body = response.json()
        except ValueError as exc:
            raise BDLSchemaError(f"Non-JSON response from {endpoint!r}: {response.text[:200]}") from exc
        if not isinstance(body, dict) or "data" not in body:
            raise BDLSchemaError(
                f"Unexpected BDL response shape from {endpoint!r}: expected an object with a "
                f"'data' key, got {type(body).__name__}"
            )
        return body

    def _paginate(self, endpoint: str, params: dict[str, Any]) -> BDLResponse:
        requested_at = utc_now_iso()
        page_params = dict(params)
        page_params.setdefault("per_page", self.config.max_page_size)

        rows: list[dict[str, Any]] = []
        seen_cursors: set[object] = set()
        pages_fetched = 0
        cursor: object | None = None

        while True:
            if cursor is not None:
                page_params["cursor"] = cursor
            body = self._get(endpoint, page_params)
            pages_fetched += 1

            data = body.get("data")
            if not isinstance(data, list):
                raise BDLSchemaError(f"Unexpected 'data' type from {endpoint!r}: {type(data).__name__}")
            rows.extend(item for item in data if isinstance(item, dict))

            if pages_fetched > self.config.max_pages:
                raise BDLPaginationError(
                    f"{endpoint!r} pagination exceeded max_pages={self.config.max_pages} "
                    "without exhausting -- refusing to loop further."
                )

            meta = body.get("meta") or {}
            next_cursor = meta.get("next_cursor")
            if next_cursor is None:
                break
            if next_cursor in seen_cursors:
                raise BDLPaginationError(
                    f"{endpoint!r} pagination cursor repeated ({next_cursor!r}) -- "
                    "provider pagination did not terminate cleanly."
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        return BDLResponse(
            endpoint=endpoint,
            request_params=dict(params),
            data=rows,
            pages_fetched=pages_fetched,
            requested_at_utc=requested_at,
            received_at_utc=utc_now_iso(),
        )

    def _get_single(self, endpoint: str) -> dict[str, Any]:
        """Non-paginated single-resource fetch (e.g. GET /games/{id})."""
        requested_at = utc_now_iso()
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = self._requests().get(url, headers=self._headers(), timeout=self.config.timeout_seconds)
        status = response.status_code
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            raise BDLRateLimitError(endpoint, response.text, float(retry_after) if retry_after else None)
        if status >= 400:
            raise BDLHTTPError(status, endpoint, response.text)
        try:
            body = response.json()
        except ValueError as exc:
            raise BDLSchemaError(f"Non-JSON response from {endpoint!r}: {response.text[:200]}") from exc
        if not isinstance(body, dict) or "data" not in body or not isinstance(body["data"], dict):
            raise BDLSchemaError(f"Unexpected single-resource shape from {endpoint!r}")
        return body

    # -- public, provider-scoped surface -----------------------------------
    # Deliberately only the endpoints Fix 5 needs (Section 4): games, plays,
    # team_stats, player_stats (`/stats`), player_injuries, team roster.

    def get_games(
        self,
        *,
        seasons: Iterable[int] | None = None,
        weeks: Iterable[int] | None = None,
        team_ids: Iterable[int] | None = None,
        dates: Iterable[str] | None = None,
        season_type: int | None = None,
        game_ids: Iterable[int] | None = None,
    ) -> BDLResponse:
        return self._paginate(
            "/games",
            {
                "seasons": list(seasons) if seasons is not None else None,
                "weeks": list(weeks) if weeks is not None else None,
                "team_ids": list(team_ids) if team_ids is not None else None,
                "dates": list(dates) if dates is not None else None,
                "season_type": season_type,
                "game_ids": list(game_ids) if game_ids is not None else None,
            },
        )

    def get_game(self, game_id: int) -> dict[str, Any]:
        body = self._get_single(f"/games/{int(game_id)}")
        return body["data"]

    def get_plays(self, game_id: int) -> BDLResponse:
        return self._paginate("/plays", {"game_id": int(game_id)})

    def get_team_stats(
        self,
        *,
        seasons: Iterable[int] | None = None,
        team_ids: Iterable[int] | None = None,
        game_ids: Iterable[int] | None = None,
    ) -> BDLResponse:
        # NOTE: /team_stats does NOT support a `weeks` filter -- verified live
        # on 2026-08-21: passing weeks=[1] silently returned all 18 weeks of
        # the season unfiltered rather than erroring or filtering. There is
        # deliberately no `weeks` parameter here; callers must filter the
        # returned rows on `row["game"]["week"]` themselves (or pass
        # `game_ids`, which IS an honored filter). See
        # docs/BDL_LIVE_INTEGRATION.md "Verified API discrepancies".
        return self._paginate(
            "/team_stats",
            {
                "seasons": list(seasons) if seasons is not None else None,
                "team_ids": list(team_ids) if team_ids is not None else None,
                "game_ids": list(game_ids) if game_ids is not None else None,
            },
        )

    def get_player_stats(
        self,
        *,
        seasons: Iterable[int] | None = None,
        player_ids: Iterable[int] | None = None,
        game_ids: Iterable[int] | None = None,
    ) -> BDLResponse:
        return self._paginate(
            "/stats",
            {
                "seasons": list(seasons) if seasons is not None else None,
                "player_ids": list(player_ids) if player_ids is not None else None,
                "game_ids": list(game_ids) if game_ids is not None else None,
            },
        )

    def get_player_injuries(
        self,
        *,
        team_ids: Iterable[int] | None = None,
        player_ids: Iterable[int] | None = None,
    ) -> BDLResponse:
        return self._paginate(
            "/player_injuries",
            {
                "team_ids": list(team_ids) if team_ids is not None else None,
                "player_ids": list(player_ids) if player_ids is not None else None,
            },
        )

    def get_team_roster(self, team_id: int, *, season: int | None = None) -> BDLResponse:
        params: dict[str, Any] = {}
        if season is not None:
            params["season"] = season
        return self._paginate(f"/teams/{int(team_id)}/roster", params)
