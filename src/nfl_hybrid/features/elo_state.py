"""Postgame-Elo pregame state: sequential, leakage-safe by construction.

This adds no new rating math. It reuses :class:`nfl_hybrid.legacy.elo.LegacyElo`
(the Elo engine) and :func:`nfl_hybrid.data.team_ids.canonical_team_id` (the
single canonical team-id policy) unchanged, and only sequences the existing
engine over the canonical games table in strict chronological order.

``nfl_hybrid.features.pregame.PregameFeatureBuilder`` already contains a
correctly-ordered (predict-before-update) Elo sequencer, but it runs on the raw
SpreadSpoke games shape and its own separate, weaker team-alias table -- it has
no production callers (see the Fix 2 completion report) and is left untouched
rather than folded in here, to avoid silently changing its behavior. This
module is a fresh, minimal adapter over the canonical ``games`` schema
(``game_id``, ``home_team_id``, ``away_team_id``, ``scheduled_kickoff_utc``,
``home_score``, ``away_score``) and the canonical team-id policy, built for the
authoritative pregame-state interface.
"""
from __future__ import annotations

import pandas as pd

from nfl_hybrid.data.team_ids import canonical_team_id
from nfl_hybrid.legacy.elo import EloConfig, EloContext, LegacyElo

TRANSFORM_NAME = "build_elo_pregame_state"
TRANSFORM_VERSION = "1.0"

REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "home_team_id",
    "away_team_id",
    "scheduled_kickoff_utc",
    "home_score",
    "away_score",
}


def build_elo_pregame_state(
    games: pd.DataFrame,
    *,
    config: EloConfig | None = None,
) -> pd.DataFrame:
    """One row per (game_id, team_id): that team's Elo state PRE-game.

    Each row is emitted by calling ``LegacyElo.predict`` before that same
    game's ``LegacyElo.update`` -- the current game's own score never enters
    its own emitted row. A game with a missing score (an upcoming/unplayed
    target) is scored using ratings as of its kickoff but never updates them,
    so inserting a target game cannot perturb any other game's state.
    """
    missing = REQUIRED_GAME_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"Games table missing Elo-required columns: {sorted(missing)}")

    work = games.copy()
    work["game_id"] = work["game_id"].astype(str)
    work["scheduled_kickoff_utc"] = pd.to_datetime(work["scheduled_kickoff_utc"], utc=True, errors="coerce")
    if work["scheduled_kickoff_utc"].isna().any():
        raise ValueError("Elo pregame state requires a parseable scheduled_kickoff_utc for every game.")
    if work["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id rows in games table.")

    work["home_team_id"] = work["home_team_id"].map(canonical_team_id)
    work["away_team_id"] = work["away_team_id"].map(canonical_team_id)
    work["neutral_site"] = work["neutral_site"].fillna(False).astype(bool) if "neutral_site" in work else False
    work["playoff"] = work["playoff"].fillna(False).astype(bool) if "playoff" in work else False

    work = work.sort_values(["scheduled_kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)

    elo = LegacyElo(config=config or EloConfig())
    previous_season: int | None = None
    rows: list[dict[str, object]] = []

    for game in work.itertuples(index=False):
        season = int(game.season)
        if previous_season is not None and season != previous_season:
            elo.regress_to_mean()
        previous_season = season

        context = EloContext(neutral_site=bool(game.neutral_site), playoff=bool(game.playoff))
        prediction = elo.predict(game.home_team_id, game.away_team_id, context)

        for team_id, opponent_id, side, rating, win_probability, expected_margin in (
            (
                game.home_team_id, game.away_team_id, "home",
                prediction.home_rating, prediction.home_win_probability, prediction.expected_home_margin,
            ),
            (
                game.away_team_id, game.home_team_id, "away",
                prediction.away_rating, 1.0 - prediction.home_win_probability, -prediction.expected_home_margin,
            ),
        ):
            rows.append(
                {
                    "game_id": game.game_id,
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "season": season,
                    "event_time": game.scheduled_kickoff_utc,
                    "elo_pregame_side": side,
                    "elo_pregame_rating": rating,
                    "elo_pregame_win_probability": win_probability,
                    "elo_pregame_expected_margin": expected_margin,
                }
            )

        home_score = game.home_score
        away_score = game.away_score
        played = pd.notna(home_score) and pd.notna(away_score)
        if played:
            elo.update(
                game.home_team_id, game.away_team_id,
                float(home_score), float(away_score),
                context, completed=True,
            )

    output = pd.DataFrame(rows)
    if output.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate Elo pregame rows produced.")
    return output
