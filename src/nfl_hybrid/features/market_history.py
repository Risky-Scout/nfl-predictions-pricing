"""Rolling prior-game ATS/OU market-outcome state: leakage-safe by construction.

This is a thin adapter, not a parallel rolling-math implementation: it builds a
long team-perspective outcome frame from the canonical games table and hands it
directly to :func:`nfl_hybrid.features.pregame_rolling.build_team_pregame_features`
-- the same shift-then-roll primitive the EPA state family uses -- with an
explicit ``metric_columns`` list, so it inherits that builder's shift(1)-before-
rolling anti-leakage guarantee unmodified rather than re-deriving it.

No family in this repository built a *history* of prior ATS/OU results before
Fix 2 -- ``features/augmented_matrix.py`` attaches the *current* game's own
market line (``home_spread``, ``total_line``), which is a legitimate pregame
feature but a different thing entirely from "how has this team fared against
the spread in its prior games". This module adds that missing state family
using only existing primitives: :func:`nfl_hybrid.labels.edge_to_nullable_binary`
(the repository's single tie/push policy -- a push is excluded, never scored as
a loss, for either team) and :func:`build_team_pregame_features`.
"""
from __future__ import annotations

import pandas as pd

from nfl_hybrid.data.team_ids import canonical_team_id
from nfl_hybrid.labels import edge_to_nullable_binary
from nfl_hybrid.features.pregame_rolling import (
    PregameRollingConfig,
    build_team_pregame_features,
)

TRANSFORM_NAME = "build_team_pregame_features[market_history]"
TRANSFORM_VERSION = "1.0"

REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "home_spread_reference",
    "total_line_reference",
}

MARKET_METRIC_COLUMNS = ("market_ats_win", "market_over_win")


def _team_market_outcomes(games: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_GAME_COLUMNS - set(games.columns)
    if missing:
        raise ValueError(f"Games table missing market-history-required columns: {sorted(missing)}")

    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    g["home_team_id"] = g["home_team_id"].map(canonical_team_id)
    g["away_team_id"] = g["away_team_id"].map(canonical_team_id)

    home_margin = pd.to_numeric(g["home_score"], errors="coerce") - pd.to_numeric(g["away_score"], errors="coerce")
    home_spread = pd.to_numeric(g["home_spread_reference"], errors="coerce")
    total_points = pd.to_numeric(g["home_score"], errors="coerce") + pd.to_numeric(g["away_score"], errors="coerce")
    total_line = pd.to_numeric(g["total_line_reference"], errors="coerce")

    home_ats_edge = home_margin + home_spread
    over_edge = total_points - total_line

    home_rows = pd.DataFrame(
        {
            "game_id": g["game_id"],
            "team_id": g["home_team_id"],
            "opponent_id": g["away_team_id"],
            "season": g["season"],
            "market_ats_win": edge_to_nullable_binary(home_ats_edge),
            "market_over_win": edge_to_nullable_binary(over_edge),
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_id": g["game_id"],
            "team_id": g["away_team_id"],
            "opponent_id": g["home_team_id"],
            "season": g["season"],
            "market_ats_win": edge_to_nullable_binary(-home_ats_edge),
            "market_over_win": edge_to_nullable_binary(over_edge),
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


def build_market_history_pregame_state(
    games: pd.DataFrame,
    *,
    config: PregameRollingConfig | None = None,
) -> pd.DataFrame:
    """Each team's rolling prior-game ATS-cover / over rate.

    A push (edge exactly zero) is excluded (``pd.NA``), never scored as a loss
    for either side, per the repository's single tie/push policy.
    """
    team_outcomes = _team_market_outcomes(games)
    return build_team_pregame_features(
        team_outcomes,
        games,
        config=config or PregameRollingConfig(),
        metric_columns=list(MARKET_METRIC_COLUMNS),
    )
