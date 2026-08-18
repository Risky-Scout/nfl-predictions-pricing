"""The single authoritative pregame-state interface (Fix 2).

This module implements no new state math for the EPA, opponent-adjusted, or QB
families -- each was already leakage-safe by construction (shift-then-roll, or
a strict ``snapshot_date <`` cut) before Fix 2. It orchestrates the existing
builders and adds what none of them individually provided: a unified,
per-family lineage record (which prior games generated each row) and a
cross-family check that no state family smuggles in a target/outcome column.

    aggregate_advanced_team_game (postgame team-game facts)
        -> build_team_pregame_features            (EPA / box-score state)
        -> build_opponent_adjusted_team_features    (opponent-adjusted state)
    qb_game
        -> build_qb_pregame_team_features            (QB pregame state)
    games
        -> build_elo_pregame_state                   (Elo state; new adapter, Fix 2)
        -> build_market_history_pregame_state         (ATS/OU history; new adapter, Fix 2)

Two state families did not exist before Fix 2 (Elo state and market/ATS-OU
history) -- both are new adapter modules that reuse existing math (
:class:`nfl_hybrid.legacy.elo.LegacyElo`, :func:`build_team_pregame_features`)
rather than introducing any of their own. See ``elo_state.py`` and
``market_history.py`` for exactly what is new code vs. reused.

Every family's state columns are checked against the repository's single
banned-column policy (:mod:`nfl_hybrid.features.feature_manifest`), and every
family's lineage is checked against
:func:`nfl_hybrid.features.state_lineage.verify_pregame_state_lineage` before
this function returns -- a target/outcome column, a self-referencing lineage
row, or a future-game lineage reference all raise here rather than reaching a
caller.

This module does not change, wire into production, or select a feature set for
any existing pipeline (``predict_week``, ``fundamental_shadow``, the frozen
``FROZEN_FEATURES`` contract in ``augmented_matrix.py`` are all untouched). It
is an additive, auditable composition layer over what already existed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nfl_hybrid.features.elo_state import (
    TRANSFORM_NAME as ELO_TRANSFORM_NAME,
    TRANSFORM_VERSION as ELO_TRANSFORM_VERSION,
    build_elo_pregame_state,
)
from nfl_hybrid.features.feature_manifest import validate_no_banned_features
from nfl_hybrid.features.market_history import (
    TRANSFORM_NAME as MARKET_TRANSFORM_NAME,
    TRANSFORM_VERSION as MARKET_TRANSFORM_VERSION,
    build_market_history_pregame_state,
)
from nfl_hybrid.features.opponent_pregame import (
    OpponentAdjustedConfig,
    build_opponent_adjusted_team_features,
)
from nfl_hybrid.features.pbp_advanced import aggregate_advanced_team_game
from nfl_hybrid.features.pregame_rolling import (
    PregameRollingConfig,
    build_team_pregame_features,
)
from nfl_hybrid.features.qb_pregame import (
    QBPregameConfig,
    build_qb_pregame_team_features,
)
from nfl_hybrid.features.state_lineage import (
    sequential_prior_game_lineage,
    snapshot_prior_game_lineage,
    verify_pregame_state_lineage,
)

EPA_TRANSFORM_NAME = "build_team_pregame_features[epa]"
EPA_TRANSFORM_VERSION = "1.0"
OPPONENT_TRANSFORM_NAME = "build_opponent_adjusted_team_features"
OPPONENT_TRANSFORM_VERSION = "1.0"
QB_TRANSFORM_NAME = "build_qb_pregame_team_features"
QB_TRANSFORM_VERSION = "1.0"

STATE_FAMILIES = ("epa", "opponent_adjusted", "qb", "elo", "market_history")

# Identifier/metadata columns every family may legitimately carry; never
# checked against the banned-column policy (they are keys, not features).
_ID_COLUMNS = {
    "game_id", "team_id", "opponent_id", "player_id",
    "season", "season_type", "week", "home_team", "away_team", "event_time",
}


@dataclass(frozen=True)
class PregameStateBundle:
    """One authoritative pregame-state result: a state table and a lineage
    table per family that was built."""

    state: dict[str, pd.DataFrame] = field(default_factory=dict)
    lineage: dict[str, pd.DataFrame] = field(default_factory=dict)

    def combined_lineage(self) -> pd.DataFrame:
        frames = [df for df in self.lineage.values() if len(df)]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def build_authoritative_pregame_state(
    games: pd.DataFrame,
    play_by_play: pd.DataFrame,
    *,
    qb_game: pd.DataFrame | None = None,
    rolling_config: PregameRollingConfig | None = None,
    opponent_config: OpponentAdjustedConfig | None = None,
    qb_config: QBPregameConfig | None = None,
    lineage_id_cap: int = 250,
) -> PregameStateBundle:
    """Build every registered pregame-state family plus its lineage record.

    Reuses each family's own existing, independently leakage-tested builder
    unchanged; adds nothing but chronological lineage and cross-family
    banned-column validation. Raises on any target/outcome column found in a
    state family's columns, or on any lineage invariant violation (see
    :func:`nfl_hybrid.features.state_lineage.verify_pregame_state_lineage`).

    ``qb_game`` is optional: without it the QB state family is simply omitted
    (not fabricated) from the returned bundle.
    """
    rcfg = rolling_config or PregameRollingConfig()
    ocfg = opponent_config or OpponentAdjustedConfig()

    team_game = aggregate_advanced_team_game(play_by_play)

    epa_state = build_team_pregame_features(team_game, games, config=rcfg)
    epa_lineage = sequential_prior_game_lineage(
        epa_state,
        state_family="epa",
        transform_name=EPA_TRANSFORM_NAME,
        transform_version=EPA_TRANSFORM_VERSION,
    )

    opponent_state = build_opponent_adjusted_team_features(team_game, games, config=ocfg)
    opponent_history = opponent_state[["game_id", "team_id", "event_time"]].assign(
        snapshot_date=pd.to_datetime(opponent_state["event_time"], utc=True).dt.floor(ocfg.snapshot_floor)
    )
    opponent_lineage = snapshot_prior_game_lineage(
        opponent_history,
        opponent_history,
        state_family="opponent_adjusted",
        transform_name=OPPONENT_TRANSFORM_NAME,
        transform_version=OPPONENT_TRANSFORM_VERSION,
        max_ids_stored=lineage_id_cap,
    )

    state: dict[str, pd.DataFrame] = {"epa": epa_state, "opponent_adjusted": opponent_state}
    lineage: dict[str, pd.DataFrame] = {"epa": epa_lineage, "opponent_adjusted": opponent_lineage}

    if qb_game is not None:
        qcfg = qb_config or QBPregameConfig()
        qb_state = build_qb_pregame_team_features(qb_game, games, config=qcfg)
        qb_history = qb_state[["game_id", "team_id", "event_time"]].assign(
            snapshot_date=pd.to_datetime(qb_state["event_time"], utc=True).dt.floor(qcfg.snapshot_floor)
        )
        qb_lineage = snapshot_prior_game_lineage(
            qb_history,
            qb_history,
            state_family="qb",
            transform_name=QB_TRANSFORM_NAME,
            transform_version=QB_TRANSFORM_VERSION,
            max_ids_stored=lineage_id_cap,
        )
        state["qb"] = qb_state
        lineage["qb"] = qb_lineage

    elo_state = build_elo_pregame_state(games)
    elo_lineage = sequential_prior_game_lineage(
        elo_state,
        state_family="elo",
        transform_name=ELO_TRANSFORM_NAME,
        transform_version=ELO_TRANSFORM_VERSION,
    )
    state["elo"] = elo_state
    lineage["elo"] = elo_lineage

    market_state = build_market_history_pregame_state(games, config=rcfg)
    market_lineage = sequential_prior_game_lineage(
        market_state,
        state_family="market_history",
        transform_name=MARKET_TRANSFORM_NAME,
        transform_version=MARKET_TRANSFORM_VERSION,
    )
    state["market_history"] = market_state
    lineage["market_history"] = market_lineage

    for frame in state.values():
        validate_no_banned_features([c for c in frame.columns if c not in _ID_COLUMNS])

    for frame in lineage.values():
        verify_pregame_state_lineage(frame, games)

    return PregameStateBundle(state=state, lineage=lineage)
