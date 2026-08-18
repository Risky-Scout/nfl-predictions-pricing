"""Fix 2 mandatory adversarial proof suite for the authoritative pregame-state
interface (:mod:`nfl_hybrid.features.pregame_state`).

Three sequential games A -> B -> C (same two teams, alternating home/away).
Every proof required by the Fix 2 specification is a separate, named test:

  Target-game invariance (game B):
    - mutate B's score / margin / outcome                 -> B unchanged
    - mutate B's ATS/OU market line                        -> B unchanged
    - mutate B's own postgame Elo input (its score)         -> B unchanged
    - mutate B's PBP/EPA                                    -> B unchanged
    - mutate future game C                                  -> B (and A) unchanged

  Legitimate-history sensitivity (game A -> game B):
    - mutate A's score/state           -> B's lagged state MAY change
    - mutate A's score (postgame Elo)  -> B changes when the Elo family is used
    - mutate A's PBP/EPA               -> B's EPA (and opponent-adjusted) state changes
    - mutate A's closing/ATS history   -> ONLY B's lagged market-history state changes

  Structural invariants:
    - target game never appears in its own contributing-game list
    - future games never appear in any contributing-game list
    - lag occurs before rolling aggregation (first game has no history)
    - duplicate game_id fails
    - chronology is deterministic; identical inputs produce identical state
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_hybrid.features.pregame_state import build_authoritative_pregame_state

NON_FEATURE_COLUMNS = {
    "game_id", "team_id", "opponent_id", "player_id",
    "season", "season_type", "week", "home_team", "away_team", "event_time",
}


def _games():
    return pd.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "home_team_id": ["KC", "BUF", "KC"],
            "away_team_id": ["BUF", "KC", "BUF"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2024-09-01T17:00:00Z", "2024-09-08T17:00:00Z", "2024-09-15T17:00:00Z"], utc=True
            ),
            "home_score": [24, 20, 27],
            "away_score": [17, 23, 24],
            "home_spread_reference": [-3.0, -2.5, -3.5],
            "total_line_reference": [44.0, 45.0, 46.0],
            "neutral_site": [False, False, False],
            "playoff": [False, False, False],
        }
    )


def _pbp(epa_by_game=None):
    epa_by_game = epa_by_game or {"A": 0.10, "B": 0.20, "C": 0.30}
    rows = []
    for gid, home, away in [("A", "KC", "BUF"), ("B", "BUF", "KC"), ("C", "KC", "BUF")]:
        epa = epa_by_game[gid]
        for team, opp, sign in [(home, away, 1), (away, home, -1)]:
            for _ in range(15):
                rows.append(
                    dict(
                        game_id=gid, posteam=team, defteam=opp, season=2024, week=1,
                        qb_dropback=1, pass_attempt=1, rush_attempt=0,
                        epa=epa * sign, success=1, down=1,
                        game_seconds_remaining=1800, score_differential=0,
                        passing_yards=8, complete_pass=1,
                    )
                )
    return pd.DataFrame(rows)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def _rows_for_game(frame: pd.DataFrame, game_id: str) -> pd.DataFrame:
    return frame[frame["game_id"].astype(str) == game_id].sort_values("team_id").reset_index(drop=True)


def _assert_game_unchanged(before: pd.DataFrame, after: pd.DataFrame, game_id: str) -> None:
    b = _rows_for_game(before, game_id)
    a = _rows_for_game(after, game_id)
    cols = _feature_columns(b)
    pd.testing.assert_frame_equal(b[cols], a[cols], check_dtype=False)


def _any_column_differs(before: pd.DataFrame, after: pd.DataFrame, game_id: str) -> bool:
    b = _rows_for_game(before, game_id)
    a = _rows_for_game(after, game_id)
    cols = _feature_columns(b)
    try:
        pd.testing.assert_frame_equal(b[cols], a[cols], check_dtype=False)
        return False
    except AssertionError:
        return True


# ---------------------------------------------------------------------------
# Target-game invariance (game B)
# ---------------------------------------------------------------------------

def test_mutate_B_score_and_margin_does_not_change_B_pregame_state():
    base = build_authoritative_pregame_state(_games(), _pbp())
    games = _games()
    games.loc[games["game_id"] == "B", ["home_score", "away_score"]] = [3, 41]  # flips the winner too
    mutated = build_authoritative_pregame_state(games, _pbp())
    for family in base.state:
        _assert_game_unchanged(base.state[family], mutated.state[family], "B")


def test_mutate_B_ats_ou_line_does_not_change_B_pregame_state():
    base = build_authoritative_pregame_state(_games(), _pbp())
    games = _games()
    games.loc[games["game_id"] == "B", ["home_spread_reference", "total_line_reference"]] = [21.0, 5.0]
    mutated = build_authoritative_pregame_state(games, _pbp())
    for family in base.state:
        _assert_game_unchanged(base.state[family], mutated.state[family], "B")


def test_mutate_B_own_postgame_elo_input_does_not_change_B_pregame_elo():
    """B's own score determines its postgame Elo update, which only affects games
    strictly after B -- never B's own already-emitted pregame row."""
    base = build_authoritative_pregame_state(_games(), _pbp())
    games = _games()
    games.loc[games["game_id"] == "B", ["home_score", "away_score"]] = [55, 0]
    mutated = build_authoritative_pregame_state(games, _pbp())
    _assert_game_unchanged(base.state["elo"], mutated.state["elo"], "B")


def test_mutate_B_pbp_epa_does_not_change_B_pregame_state():
    base = build_authoritative_pregame_state(_games(), _pbp())
    mutated = build_authoritative_pregame_state(_games(), _pbp({"A": 0.10, "B": 9.99, "C": 0.30}))
    for family in base.state:
        _assert_game_unchanged(base.state[family], mutated.state[family], "B")


def test_mutate_future_game_C_does_not_change_A_or_B_state():
    base = build_authoritative_pregame_state(_games(), _pbp())
    games = _games()
    games.loc[games["game_id"] == "C", ["home_score", "away_score", "home_spread_reference", "total_line_reference"]] = [
        1, 99, 40.0, -20.0,
    ]
    mutated = build_authoritative_pregame_state(games, _pbp({"A": 0.10, "B": 0.20, "C": -9.99}))
    for family in base.state:
        _assert_game_unchanged(base.state[family], mutated.state[family], "A")
        _assert_game_unchanged(base.state[family], mutated.state[family], "B")


# ---------------------------------------------------------------------------
# Legitimate-history sensitivity (game A -> game B)
# ---------------------------------------------------------------------------

def test_mutate_A_score_changes_B_lagged_score_driven_state_only():
    base = build_authoritative_pregame_state(_games(), _pbp())
    games = _games()
    games.loc[games["game_id"] == "A", ["home_score", "away_score"]] = [3, 41]
    mutated = build_authoritative_pregame_state(games, _pbp())

    # score-driven families change for B ...
    assert _any_column_differs(base.state["elo"], mutated.state["elo"], "B")
    assert _any_column_differs(base.state["market_history"], mutated.state["market_history"], "B")
    # ... but EPA/opponent-adjusted state, which never consume final score, do not.
    _assert_game_unchanged(base.state["epa"], mutated.state["epa"], "B")
    _assert_game_unchanged(base.state["opponent_adjusted"], mutated.state["opponent_adjusted"], "B")


def test_mutate_A_postgame_elo_changes_B_when_elo_family_is_used():
    base = build_authoritative_pregame_state(_games(), _pbp())
    games = _games()
    games.loc[games["game_id"] == "A", ["home_score", "away_score"]] = [45, 3]
    mutated = build_authoritative_pregame_state(games, _pbp())
    assert _any_column_differs(base.state["elo"], mutated.state["elo"], "B")


def _pbp_team_epa_override(game_id: str, team_id: str, new_epa: float) -> pd.DataFrame:
    """Change one team's EPA in one game only -- the base fixture mirrors each
    game's EPA symmetrically (+edge / -edge) across the two teams, so a
    same-sign mutation to both sides would cancel out in any league-wide mean
    and mask a real, detectable state change."""
    frame = _pbp()
    mask = (frame["game_id"] == game_id) & (frame["posteam"] == team_id)
    frame.loc[mask, "epa"] = new_epa
    return frame


def test_mutate_A_pbp_epa_changes_B_epa_and_opponent_adjusted_state_only():
    base = build_authoritative_pregame_state(_games(), _pbp())
    mutated = build_authoritative_pregame_state(_games(), _pbp_team_epa_override("A", "KC", 9.99))

    assert _any_column_differs(base.state["epa"], mutated.state["epa"], "B")
    assert _any_column_differs(base.state["opponent_adjusted"], mutated.state["opponent_adjusted"], "B")
    # score/market-derived families never consume PBP and must be untouched.
    _assert_game_unchanged(base.state["elo"], mutated.state["elo"], "B")
    _assert_game_unchanged(base.state["market_history"], mutated.state["market_history"], "B")


def test_mutate_A_closing_ats_history_changes_only_lagged_market_state():
    base = build_authoritative_pregame_state(_games(), _pbp())
    games = _games()
    games.loc[games["game_id"] == "A", "home_spread_reference"] = -20.0  # flips A's ATS result for KC
    mutated = build_authoritative_pregame_state(games, _pbp())

    assert _any_column_differs(base.state["market_history"], mutated.state["market_history"], "B")
    _assert_game_unchanged(base.state["epa"], mutated.state["epa"], "B")
    _assert_game_unchanged(base.state["opponent_adjusted"], mutated.state["opponent_adjusted"], "B")
    _assert_game_unchanged(base.state["elo"], mutated.state["elo"], "B")


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_target_game_never_in_own_source_list():
    bundle = build_authoritative_pregame_state(_games(), _pbp())
    for family, lineage in bundle.lineage.items():
        for row in lineage.itertuples(index=False):
            assert row.game_id not in row.contributing_game_ids, (family, row.game_id)


def test_future_games_never_appear_in_source_list():
    bundle = build_authoritative_pregame_state(_games(), _pbp())
    allowed_prior = {"A": set(), "B": {"A"}, "C": {"A", "B"}}
    for family, lineage in bundle.lineage.items():
        for row in lineage.itertuples(index=False):
            if row.contributing_ids_truncated:
                continue
            assert set(row.contributing_game_ids) <= allowed_prior[row.game_id], (
                family, row.game_id, row.contributing_game_ids,
            )


def test_lag_occurs_before_rolling_aggregation_first_game_has_no_history():
    bundle = build_authoritative_pregame_state(_games(), _pbp())
    epa_first = _rows_for_game(bundle.state["epa"], "A")
    count_cols = [c for c in epa_first.columns if c.endswith("__all_count")]
    assert epa_first[count_cols].fillna(0).to_numpy().sum() == 0

    market_first = _rows_for_game(bundle.state["market_history"], "A")
    count_cols = [c for c in market_first.columns if c.endswith("__all_count")]
    assert market_first[count_cols].fillna(0).to_numpy().sum() == 0

    for family, lineage in bundle.lineage.items():
        for row in lineage.itertuples(index=False):
            if row.game_id == "A" and row.contributing_scope == "team_direct":
                assert row.contributing_game_count == 0


def test_duplicate_game_id_fails():
    games = pd.concat([_games(), _games().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_authoritative_pregame_state(games, _pbp())


def test_chronology_is_deterministic_regardless_of_input_row_order():
    games = _games()
    pbp = _pbp()
    first = build_authoritative_pregame_state(games, pbp)
    second = build_authoritative_pregame_state(
        games.sample(frac=1.0, random_state=3).reset_index(drop=True),
        pbp.sample(frac=1.0, random_state=5).reset_index(drop=True),
    )
    for family in first.state:
        a = first.state[family].sort_values(["game_id", "team_id"]).reset_index(drop=True)
        b = second.state[family].sort_values(["game_id", "team_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_identical_inputs_produce_identical_state():
    games = _games()
    pbp = _pbp()
    first = build_authoritative_pregame_state(games, pbp)
    second = build_authoritative_pregame_state(games, pbp)
    for family in first.state:
        pd.testing.assert_frame_equal(first.state[family], second.state[family])
    for family in first.lineage:
        pd.testing.assert_frame_equal(first.lineage[family], second.lineage[family])
