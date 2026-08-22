"""BDL raw JSON -> canonical, provider-neutral games / team_stats /
player_stats / injuries / roster DataFrames.

Field lists below were captured from live, authenticated BDL NFL API calls
on 2026-08-21 (season 2025 games, team_stats, player_injuries, stats,
teams/{id}/roster) -- not guessed from third-party summaries. Where the
live response disagreed with prior documentation, the live response wins
(Section 2) and the discrepancy is noted inline.

Every normalizer here is a pure function: raw list[dict] in, a canonical
``pandas.DataFrame`` out, plus per-row ``source_payload_hash`` for
idempotence proof (Section 22). No network access, no finality decisions
(see ``finality.py``), no PBP (see ``plays.py``).
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import pandas as pd

from nfl_hybrid.data.provenance import utc_now_iso
from nfl_hybrid.providers.balldontlie.team_crosswalk import (
    TeamCrosswalkError,
    canonical_team_from_team_object,
)

PROVIDER_NAME = "balldontlie"


class CanonicalizationError(ValueError):
    """A raw BDL row could not be normalized without fabricating a value."""


def _payload_hash(raw: dict[str, Any]) -> str:
    encoded = json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_game_id(season: int, week: int, away_team: str, home_team: str) -> str:
    # Matches the existing nflverse convention exactly (see
    # NflverseAdapter.load_games / tests/test_data_adapters.py
    # `"2020_01_HOU_KC"`): {season}_{week:02d}_{away}_{home}. Using the same
    # convention is what lets a live 2026 game key line up with the same
    # game's historical-style identifier with no separate crosswalk table.
    return f"{season}_{week:02d}_{away_team}_{home_team}"


# -- games (Section 7) ---------------------------------------------------------

GAMES_COLUMNS = (
    "provider", "provider_game_id", "game_id", "season", "week", "season_type",
    "season_type_basis", "postseason", "kickoff_utc", "home_team", "away_team",
    "home_score", "away_score", "status", "status_state",
    "source_collected_at_utc", "source_payload_hash",
)


def normalize_games(
    raw_games: list[dict[str, Any]],
    *,
    season_type_hint: str | None = None,
    retrieved_at_utc: str | None = None,
) -> pd.DataFrame:
    """``season_type_hint`` ("PRE" or "REG") disambiguates non-postseason
    games.

    BDL's own ``/games`` response never says whether a non-postseason game
    is preseason or regular season -- verified live on 2026-08-21: a
    ``season_type=1`` (preseason) query and a ``season_type=2`` (regular)
    query both return rows shaped identically (``postseason: false``, week
    numbering restarting at 1 in both). The distinction only exists in
    which query filter was used to fetch the batch. Pass the season_type
    filter you queried with as ``season_type_hint``; postseason rows are
    always unambiguous (``postseason: true`` -> ``"POST"``) regardless of
    the hint.

    FAILS CLOSED (Fix 5 certification Section 6): a non-postseason row with
    a missing or unrecognized ``season_type_hint`` raises
    ``CanonicalizationError`` rather than silently defaulting to ``"REG"``.
    Preseason week numbering restarts at 1 exactly like regular season, so
    inferring regular season "by default" from an ambiguous batch would risk
    silently conflating a preseason Week 1 game with a regular-season Week 1
    game of the same matchup. Callers MUST always pass the ``season_type``
    filter they queried BDL with as this hint -- there is no safe default.
    """
    retrieved = retrieved_at_utc or utc_now_iso()
    rows: list[dict[str, Any]] = []
    for raw in raw_games:
        try:
            home = canonical_team_from_team_object(raw.get("home_team"))
            away = canonical_team_from_team_object(raw.get("visitor_team"))
        except TeamCrosswalkError as exc:
            raise CanonicalizationError(f"game {raw.get('id')!r}: {exc}") from exc

        season = raw.get("season")
        week = raw.get("week")
        if season is None or week is None:
            raise CanonicalizationError(f"game {raw.get('id')!r} missing season/week")

        postseason = bool(raw.get("postseason"))
        if postseason:
            season_type, basis = "POST", "postseason_flag"
        elif season_type_hint in ("PRE", "REG"):
            season_type, basis = season_type_hint, "query_hint"
        else:
            raise CanonicalizationError(
                f"game {raw.get('id')!r} (season={season}, week={week}, postseason=False) has an "
                "ambiguous season type: BDL does not distinguish preseason from regular season on "
                "non-postseason /games rows, and week numbering restarts at 1 for both. Pass "
                "season_type_hint='PRE' or 'REG' matching the query filter used to fetch this batch "
                "-- never inferred from week number, and never defaulted."
            )

        rows.append(
            {
                "provider": PROVIDER_NAME,
                "provider_game_id": raw.get("id"),
                "game_id": _canonical_game_id(int(season), int(week), away, home),
                "season": int(season),
                "week": int(week),
                "season_type": season_type,
                "season_type_basis": basis,
                "postseason": postseason,
                "kickoff_utc": pd.to_datetime(raw.get("date"), utc=True, errors="coerce"),
                "home_team": home,
                "away_team": away,
                "home_score": pd.to_numeric(raw.get("home_team_score"), errors="coerce"),
                "away_score": pd.to_numeric(raw.get("visitor_team_score"), errors="coerce"),
                "status": raw.get("status"),
                "status_state": raw.get("status_state"),
                "source_collected_at_utc": retrieved,
                "source_payload_hash": _payload_hash(raw),
            }
        )
    return pd.DataFrame(rows, columns=list(GAMES_COLUMNS))


# -- team_stats (Section 11) ---------------------------------------------------

# Native BDL field names (verified live, game 423945) are kept as-is -- they
# are already clear snake_case; renaming them would just be indirection.
_TEAM_STATS_NUMERIC_FIELDS = (
    "first_downs", "first_downs_passing", "first_downs_rushing", "first_downs_penalty",
    "third_down_conversions", "third_down_attempts",
    "fourth_down_conversions", "fourth_down_attempts",
    "total_offensive_plays", "total_yards", "yards_per_play", "total_drives",
    "net_passing_yards", "passing_completions", "passing_attempts", "yards_per_pass",
    "sacks", "sack_yards_lost", "rushing_yards", "rushing_attempts", "yards_per_rush_attempt",
    "red_zone_scores", "red_zone_attempts", "penalties", "penalty_yards",
    "turnovers", "fumbles_lost", "interceptions_thrown", "defensive_touchdowns",
    "possession_time_seconds",
)


def normalize_team_stats(
    raw_team_stats: list[dict[str, Any]],
    *,
    retrieved_at_utc: str | None = None,
) -> pd.DataFrame:
    retrieved = retrieved_at_utc or utc_now_iso()
    rows: list[dict[str, Any]] = []
    for raw in raw_team_stats:
        game = raw.get("game") or {}
        try:
            team = canonical_team_from_team_object(raw.get("team"))
        except TeamCrosswalkError as exc:
            raise CanonicalizationError(f"team_stats for game {game.get('id')!r}: {exc}") from exc

        row: dict[str, Any] = {
            "provider": PROVIDER_NAME,
            "provider_game_id": game.get("id"),
            "team": team,
            "home_away": raw.get("home_away"),
            "third_down_efficiency_text": raw.get("third_down_efficiency"),
            "fourth_down_efficiency_text": raw.get("fourth_down_efficiency"),
            "possession_time_text": raw.get("possession_time"),
            "source_collected_at_utc": retrieved,
            "source_payload_hash": _payload_hash(raw),
        }
        for field in _TEAM_STATS_NUMERIC_FIELDS:
            row[field] = pd.to_numeric(raw.get(field), errors="coerce")
        rows.append(row)
    columns = (
        ["provider", "provider_game_id", "team", "home_away"]
        + list(_TEAM_STATS_NUMERIC_FIELDS)
        + ["third_down_efficiency_text", "fourth_down_efficiency_text", "possession_time_text",
           "source_collected_at_utc", "source_payload_hash"]
    )
    return pd.DataFrame(rows, columns=columns)


def team_stats_completeness(team_stats: pd.DataFrame, game: dict[str, str]) -> dict[str, Any]:
    """Section 11 reconciliation: exactly one canonical row per participating
    team, both teams matching the canonical game's own teams, no duplicates.

    ``game`` is a mapping with ``provider_game_id`` (or ``game_id``),
    ``home_team``, ``away_team``. Returns a report dict including
    ``box_complete: bool`` -- never raises; callers decide what a failed
    reconciliation means for their own gate.
    """
    provider_game_id = game.get("provider_game_id")
    subset = team_stats.loc[team_stats["provider_game_id"] == provider_game_id]
    expected = {game["home_team"], game["away_team"]}
    present_teams = list(subset["team"])
    duplicate_teams = sorted({t for t in present_teams if present_teams.count(t) > 1})
    present_set = set(present_teams)
    missing = sorted(expected - present_set)
    unexpected = sorted(present_set - expected)
    box_complete = (
        len(subset) == 2
        and present_set == expected
        and not duplicate_teams
        and not unexpected
    )
    return {
        "provider_game_id": provider_game_id,
        "rows_found": len(subset),
        "teams_present": sorted(present_set),
        "teams_missing": missing,
        "duplicate_teams": duplicate_teams,
        "unexpected_teams": unexpected,
        "box_complete": box_complete,
    }


# -- player_stats (Section 12) --------------------------------------------------

_PLAYER_STATS_NUMERIC_FIELDS = (
    "passing_completions", "passing_attempts", "passing_yards", "yards_per_pass_attempt",
    "passing_touchdowns", "passing_interceptions", "sacks", "sacks_loss", "qbr", "qb_rating",
    "rushing_attempts", "rushing_yards", "yards_per_rush_attempt", "rushing_touchdowns", "long_rushing",
    "receptions", "receiving_yards", "yards_per_reception", "receiving_touchdowns", "long_reception",
    "receiving_targets", "fumbles", "fumbles_lost", "fumbles_recovered",
    "total_tackles", "defensive_sacks", "solo_tackles", "tackles_for_loss", "passes_defended",
    "qb_hits", "fumbles_touchdowns", "defensive_interceptions", "interception_yards",
    "interception_touchdowns", "kick_returns", "kick_return_yards",
)

# Player identity parity (Section 12): BDL player ids are a BDL-native
# namespace with no published crosswalk to nflverse's `gsis_id`/`pfr_id`
# space in this repo today. Treat player-identity parity as PARTIAL until an
# explicit crosswalk layer exists -- never silently assume the two ids
# coincide.
PLAYER_IDENTITY_PARITY = "PARTIAL"


def normalize_player_stats(
    raw_player_stats: list[dict[str, Any]],
    *,
    retrieved_at_utc: str | None = None,
) -> pd.DataFrame:
    retrieved = retrieved_at_utc or utc_now_iso()
    rows: list[dict[str, Any]] = []
    for raw in raw_player_stats:
        game = raw.get("game") or {}
        player = raw.get("player") or {}
        try:
            team = canonical_team_from_team_object(raw.get("team"))
        except TeamCrosswalkError as exc:
            raise CanonicalizationError(f"player_stats for game {game.get('id')!r}: {exc}") from exc

        row: dict[str, Any] = {
            "provider": PROVIDER_NAME,
            "provider_game_id": game.get("id"),
            "provider_player_id": player.get("id"),
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "position": player.get("position_abbreviation") or player.get("position"),
            "team": team,
            "source_collected_at_utc": retrieved,
            "source_payload_hash": _payload_hash(raw),
        }
        for field in _PLAYER_STATS_NUMERIC_FIELDS:
            row[field] = pd.to_numeric(raw.get(field), errors="coerce")
        rows.append(row)
    columns = (
        ["provider", "provider_game_id", "provider_player_id", "player_name", "position", "team"]
        + list(_PLAYER_STATS_NUMERIC_FIELDS)
        + ["source_collected_at_utc", "source_payload_hash"]
    )
    return pd.DataFrame(rows, columns=columns)


def player_stats_completeness(player_stats: pd.DataFrame, provider_game_id: int) -> dict[str, Any]:
    """Conservative: complete iff at least one player-stat row exists for
    each team side is not verifiable from `/stats` alone (BDL does not flag
    "box score finalized" independent of the game's own status_state), so
    this only proves non-emptiness, never full-roster completeness. Callers
    needing a stronger contract should treat this as a floor, not a
    guarantee -- see docs/BDL_LIVE_INTEGRATION.md completeness section."""
    subset = player_stats.loc[player_stats["provider_game_id"] == provider_game_id]
    return {
        "provider_game_id": provider_game_id,
        "rows_found": len(subset),
        "player_stats_complete": len(subset) > 0,
    }


# -- injuries (Section 14) ------------------------------------------------------

def normalize_injuries(
    raw_injuries: list[dict[str, Any]],
    *,
    retrieved_at_utc: str | None = None,
) -> pd.DataFrame:
    """AS-OF pregame snapshot family. Persists status/comment exactly as
    provided -- never infers inactive status, practice participation,
    expected snaps, or severity, since BDL does not supply those fields on
    this endpoint (verified live 2026-08-21: `player_injuries` rows are
    `{player, status, comment, date}` only)."""
    retrieved = retrieved_at_utc or utc_now_iso()
    rows: list[dict[str, Any]] = []
    for raw in raw_injuries:
        player = raw.get("player") or {}
        team_obj = player.get("team")
        try:
            team = canonical_team_from_team_object(team_obj) if team_obj else None
        except TeamCrosswalkError:
            team = None  # fail closed on the team join only; the injury row itself is still kept
        rows.append(
            {
                "provider": PROVIDER_NAME,
                "provider_player_id": player.get("id"),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "position": player.get("position_abbreviation") or player.get("position"),
                "team": team,
                "status_raw": raw.get("status"),
                "comment_raw": raw.get("comment"),
                "report_date_utc": pd.to_datetime(raw.get("date"), utc=True, errors="coerce"),
                "collected_at_utc": retrieved,
                "source_payload_hash": _payload_hash(raw),
            }
        )
    columns = [
        "provider", "provider_player_id", "player_name", "position", "team",
        "status_raw", "comment_raw", "report_date_utc", "collected_at_utc", "source_payload_hash",
    ]
    return pd.DataFrame(rows, columns=columns)


# -- roster / depth (Section 13) -------------------------------------------------

def normalize_roster(
    raw_roster: list[dict[str, Any]],
    *,
    team: str,
    season: int | None = None,
    retrieved_at_utc: str | None = None,
) -> pd.DataFrame:
    """A LIVE pregame snapshot source, never a completed-game result. No
    future snapshot may be used for an earlier forecast -- callers must
    filter on ``collected_at_utc`` themselves relative to their own forecast
    cutoff; this function only stamps the snapshot time, it does not
    enforce as-of correctness (that belongs to the caller's forecast
    pipeline, not the provider layer)."""
    retrieved = retrieved_at_utc or utc_now_iso()
    rows: list[dict[str, Any]] = []
    for raw in raw_roster:
        player = raw.get("player") or {}
        rows.append(
            {
                "provider": PROVIDER_NAME,
                "team": team,
                "season": season,
                "provider_player_id": player.get("id"),
                "player_name": raw.get("player_name")
                or f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "position": raw.get("position") or player.get("position_abbreviation"),
                "depth": raw.get("depth"),
                "injury_status": raw.get("injury_status"),
                "collected_at_utc": retrieved,
                "source_payload_hash": _payload_hash(raw),
            }
        )
    columns = [
        "provider", "team", "season", "provider_player_id", "player_name", "position",
        "depth", "injury_status", "collected_at_utc", "source_payload_hash",
    ]
    return pd.DataFrame(rows, columns=columns)
