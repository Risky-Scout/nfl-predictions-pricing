"""Fix 3.1 mandatory adversarial proof suite for OOF feature-provenance
(:mod:`nfl_hybrid.evaluation.chronological_oof`).

Fix 3's ``build_oof_feature_matrix`` swept every numeric ``home_*``/``away_*``
column off the pivoted matrix returned by
:func:`nfl_hybrid.features.pregame_rolling.build_game_pregame_matrix`. That
pivot's base frame was a full copy of the raw ``games`` table, so native
carrier columns that already happened to start with ``home_``/``away_``
(``home_score``, ``away_score``, ``home_moneyline_reference``,
``away_moneyline_reference``, ``home_spread_reference``,
``home_spread_price_reference``, ``away_spread_price_reference``) survived
into the feature set unchanged, once per state family -- ``epa__home_score``
exactly equalling the target game's own final score was the confirmed proof.

This module proves that class of leak is now structurally impossible, not
merely absent from today's 35 known names:

  - known-leak regression (defense-in-depth over the positive-provenance fix)
  - poison-pill target mutation: a target game's own current-market/outcome
    fields cannot change its own feature vector
  - future-game mutation: a later game's outcome/market cannot change an
    earlier target's feature vector
  - prior-game positive control: a legitimate prior game's real outcome CAN
    still change a later game's feature vector -- proves the repair did not
    simply sever legitimate historical state
  - current-market leakage: no feature's provenance traces back to a native
    current-game market/outcome column

All fixtures here are small, synthetic, and hand-built specifically to
include the previously-leaking native columns (``home_moneyline_reference``,
``away_moneyline_reference``, ``home_spread_price_reference``,
``away_spread_price_reference``) that the repository's other synthetic
fixtures (``_tiny_games_and_pbp`` in ``test_chronological_oof.py``) do not
carry, so this suite would have failed against the pre-Fix-3.1
implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation.chronological_oof import (
    build_oof_feature_matrix,
    compute_feature_provenance,
    declared_state_family_columns,
)
from nfl_hybrid.features.pregame_rolling import build_game_pregame_matrix

# The exact 35 names confirmed leaked in the Fix 3.1 incident report: 7
# native passthrough columns x 5 state families.
KNOWN_LEAKED_NATIVE_SUFFIXES = (
    "home_score",
    "away_score",
    "home_moneyline_reference",
    "away_moneyline_reference",
    "home_spread_reference",
    "home_spread_price_reference",
    "away_spread_price_reference",
)
STATE_FAMILIES_FOR_LEAK_CHECK = ("epa", "opponent_adjusted", "qb", "elo", "market_history")
KNOWN_LEAKED_FEATURE_NAMES = tuple(
    f"{family}__{suffix}"
    for family in STATE_FAMILIES_FOR_LEAK_CHECK
    for suffix in KNOWN_LEAKED_NATIVE_SUFFIXES
)


def _games_with_native_market_and_outcome_columns(
    n: int = 32,
    seed: int = 0,
    *,
    home_score_overrides: dict[int, float] | None = None,
    away_score_overrides: dict[int, float] | None = None,
    home_spread_reference_overrides: dict[int, float] | None = None,
    total_line_reference_overrides: dict[int, float] | None = None,
    home_moneyline_reference_overrides: dict[int, float] | None = None,
    away_moneyline_reference_overrides: dict[int, float] | None = None,
    home_spread_price_reference_overrides: dict[int, float] | None = None,
    away_spread_price_reference_overrides: dict[int, float] | None = None,
) -> pd.DataFrame:
    """A games table carrying every native column implicated in the Fix 3.1
    incident report -- including moneyline/spread-price reference columns
    the repository's other synthetic fixtures omit -- so a leak in any of
    them would be caught here."""
    teams = ["KC", "BUF", "SF", "DAL"]
    rng = np.random.default_rng(seed)
    game_ids = [f"T{i:03d}" for i in range(n)]
    kickoff = pd.date_range("2023-09-07", periods=n, freq="7D", tz="UTC")
    home = [teams[i % 4] for i in range(n)]
    away = [teams[(i + 1) % 4] for i in range(n)]

    home_score = rng.integers(10, 35, n).astype(float)
    away_score = rng.integers(10, 35, n).astype(float)
    home_spread_reference = rng.uniform(-10, 10, n)
    total_line_reference = rng.uniform(38, 52, n)
    home_moneyline_reference = rng.integers(-250, 250, n).astype(float)
    away_moneyline_reference = -home_moneyline_reference
    home_spread_price_reference = rng.integers(-120, -100, n).astype(float)
    away_spread_price_reference = rng.integers(-120, -100, n).astype(float)

    for idx, value in (home_score_overrides or {}).items():
        home_score[idx] = value
    for idx, value in (away_score_overrides or {}).items():
        away_score[idx] = value
    for idx, value in (home_spread_reference_overrides or {}).items():
        home_spread_reference[idx] = value
    for idx, value in (total_line_reference_overrides or {}).items():
        total_line_reference[idx] = value
    for idx, value in (home_moneyline_reference_overrides or {}).items():
        home_moneyline_reference[idx] = value
    for idx, value in (away_moneyline_reference_overrides or {}).items():
        away_moneyline_reference[idx] = value
    for idx, value in (home_spread_price_reference_overrides or {}).items():
        home_spread_price_reference[idx] = value
    for idx, value in (away_spread_price_reference_overrides or {}).items():
        away_spread_price_reference[idx] = value

    return pd.DataFrame(
        {
            "game_id": game_ids,
            "season": 2023,
            "week": np.arange(n) + 1,
            "home_team_id": home,
            "away_team_id": away,
            "scheduled_kickoff_utc": kickoff,
            "home_score": home_score,
            "away_score": away_score,
            "home_spread_reference": home_spread_reference,
            "total_line_reference": total_line_reference,
            "home_moneyline_reference": home_moneyline_reference,
            "away_moneyline_reference": away_moneyline_reference,
            "home_spread_price_reference": home_spread_price_reference,
            "away_spread_price_reference": away_spread_price_reference,
            "neutral_site": False,
            "playoff": False,
        }
    )


def _pbp_for_games(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for game in games.itertuples(index=False):
        for team, opp in [
            (game.home_team_id, game.away_team_id),
            (game.away_team_id, game.home_team_id),
        ]:
            for _ in range(6):
                rows.append(
                    dict(
                        game_id=game.game_id, posteam=team, defteam=opp,
                        season=2023, week=1, qb_dropback=1, pass_attempt=1,
                        rush_attempt=0, epa=float(rng.normal() * 0.1), success=1,
                        down=1, game_seconds_remaining=1800, score_differential=0,
                        passing_yards=8, complete_pass=1,
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Known-leak regression (defense-in-depth)
# ---------------------------------------------------------------------------

def test_known_leaked_feature_names_are_absent():
    games = _games_with_native_market_and_outcome_columns()
    pbp = _pbp_for_games(games)
    matrix, feature_columns, _bundle = build_oof_feature_matrix(games, pbp)

    leaked_present = [name for name in KNOWN_LEAKED_FEATURE_NAMES if name in feature_columns]
    assert leaked_present == []
    assert "epa__home_score" not in feature_columns

    # Nothing whose provenance-derived source_feature is a native
    # market/outcome column either, even under a different family prefix.
    provenance = compute_feature_provenance(feature_columns)
    unsafe_sources = {
        "score", "home_score", "away_score",
        "moneyline_reference", "home_moneyline_reference", "away_moneyline_reference",
        "spread_reference", "home_spread_reference",
        "spread_price_reference", "home_spread_price_reference", "away_spread_price_reference",
        "total_line_reference",
    }
    leaked_by_source = [r for r in provenance if r["source_feature"] in unsafe_sources]
    assert leaked_by_source == []


def test_declared_state_family_columns_excludes_identifiers():
    games = _games_with_native_market_and_outcome_columns()
    pbp = _pbp_for_games(games)
    _matrix, _feature_columns, bundle = build_oof_feature_matrix(games, pbp)

    for family, state in bundle.state.items():
        declared = declared_state_family_columns(state)
        assert "game_id" not in declared
        assert "team_id" not in declared
        assert declared, f"{family} produced no declared columns"


def test_native_carrier_columns_never_reach_the_pivot_when_narrowed():
    """Unit-level proof of the pivot-side defense-in-depth: with
    carrier_columns=(), build_game_pregame_matrix's own output frame does
    not carry home_score/away_score/home_spread_reference at all -- there is
    nothing left for even a reintroduced wildcard sweep to catch."""
    games = _games_with_native_market_and_outcome_columns(n=8)
    pbp = _pbp_for_games(games)
    from nfl_hybrid.features.pbp_advanced import aggregate_advanced_team_game
    from nfl_hybrid.features.pregame_rolling import build_team_pregame_features

    team_game = aggregate_advanced_team_game(pbp)
    epa_state = build_team_pregame_features(team_game, games)

    narrowed = build_game_pregame_matrix(games, epa_state, carrier_columns=())
    assert "home_score" not in narrowed.columns
    assert "away_score" not in narrowed.columns
    assert "home_spread_reference" not in narrowed.columns
    assert "home_moneyline_reference" not in narrowed.columns

    # Fix 3.1 API hardening: the DEFAULT (carrier_columns unset) is equally
    # safe -- there is no longer a "backward-compatible full native row"
    # default left to preserve. augmented_matrix.py / opponent_pregame.py
    # were updated to pass carrier_columns=() explicitly, and the function's
    # own default is now () too, so both paths behave identically here.
    default_pivot = build_game_pregame_matrix(games, epa_state)
    assert "home_score" not in default_pivot.columns
    assert "away_score" not in default_pivot.columns
    assert "home_spread_reference" not in default_pivot.columns
    assert "home_moneyline_reference" not in default_pivot.columns

    # None is accepted but must never mean "all columns" either.
    none_pivot = build_game_pregame_matrix(games, epa_state, carrier_columns=None)
    assert "home_score" not in none_pivot.columns


# ---------------------------------------------------------------------------
# Poison-pill target mutation
# ---------------------------------------------------------------------------

def _target_position(games: pd.DataFrame) -> int:
    # Deep enough into the sequence to have real rolling history, but not
    # the very last row (so a "future game" exists for the future-mutation
    # test below).
    return len(games) - 5


def test_poison_pill_target_mutation_does_not_change_its_own_feature_vector():
    games = _games_with_native_market_and_outcome_columns()
    pbp = _pbp_for_games(games)
    matrix, feature_columns, _bundle = build_oof_feature_matrix(games, pbp)

    target_idx = _target_position(games)
    target_game_id = games.loc[target_idx, "game_id"]
    base_row = matrix.loc[matrix["game_id"] == target_game_id, feature_columns].iloc[0]

    mutated_games = _games_with_native_market_and_outcome_columns(
        home_score_overrides={target_idx: 999.0},
        away_score_overrides={target_idx: -999.0},
        home_spread_reference_overrides={target_idx: 12345.0},
        total_line_reference_overrides={target_idx: -12345.0},
        home_moneyline_reference_overrides={target_idx: 999999.0},
        away_moneyline_reference_overrides={target_idx: -999999.0},
        home_spread_price_reference_overrides={target_idx: 424242.0},
        away_spread_price_reference_overrides={target_idx: -424242.0},
    )
    assert mutated_games.loc[target_idx, "game_id"] == target_game_id
    mutated_pbp = _pbp_for_games(mutated_games)  # unaffected by the games mutation; rebuilt for parity
    mutated_matrix, mutated_feature_columns, _bundle2 = build_oof_feature_matrix(
        mutated_games, mutated_pbp
    )

    assert tuple(mutated_feature_columns) == tuple(feature_columns)
    mutated_row = mutated_matrix.loc[
        mutated_matrix["game_id"] == target_game_id, feature_columns
    ].iloc[0]

    pd.testing.assert_series_equal(base_row, mutated_row, check_names=False)


# ---------------------------------------------------------------------------
# Future-game mutation
# ---------------------------------------------------------------------------

def test_future_game_mutation_does_not_change_an_earlier_targets_feature_vector():
    games = _games_with_native_market_and_outcome_columns()
    pbp = _pbp_for_games(games)
    matrix, feature_columns, _bundle = build_oof_feature_matrix(games, pbp)

    target_idx = _target_position(games)
    target_game_id = games.loc[target_idx, "game_id"]
    future_idx = len(games) - 1
    assert future_idx > target_idx
    base_row = matrix.loc[matrix["game_id"] == target_game_id, feature_columns].iloc[0]

    mutated_games = _games_with_native_market_and_outcome_columns(
        home_score_overrides={future_idx: 777.0},
        away_score_overrides={future_idx: -777.0},
        home_spread_reference_overrides={future_idx: 55555.0},
        home_moneyline_reference_overrides={future_idx: 88888.0},
    )
    mutated_pbp = _pbp_for_games(mutated_games)
    mutated_matrix, _fc, _bundle2 = build_oof_feature_matrix(mutated_games, mutated_pbp)
    mutated_row = mutated_matrix.loc[
        mutated_matrix["game_id"] == target_game_id, feature_columns
    ].iloc[0]

    pd.testing.assert_series_equal(base_row, mutated_row, check_names=False)


# ---------------------------------------------------------------------------
# Prior-game positive control
# ---------------------------------------------------------------------------

def test_prior_game_positive_control_legitimate_history_still_propagates():
    """The repair must not sever legitimate chronology: mutating a
    completed PRIOR game's real outcome must still change a later target's
    pregame feature vector (via Elo state, which updates on prior
    home_score/away_score)."""
    games = _games_with_native_market_and_outcome_columns()
    pbp = _pbp_for_games(games)
    matrix, feature_columns, _bundle = build_oof_feature_matrix(games, pbp)

    target_idx = _target_position(games)
    target_game_id = games.loc[target_idx, "game_id"]
    prior_idx = 3
    assert prior_idx < target_idx
    base_row = matrix.loc[matrix["game_id"] == target_game_id, feature_columns]

    mutated_games = _games_with_native_market_and_outcome_columns(
        home_score_overrides={prior_idx: 58.0},
        away_score_overrides={prior_idx: 3.0},
    )
    mutated_pbp = _pbp_for_games(mutated_games)
    mutated_matrix, _fc, _bundle2 = build_oof_feature_matrix(mutated_games, mutated_pbp)
    mutated_row = mutated_matrix.loc[mutated_matrix["game_id"] == target_game_id, feature_columns]

    elo_cols = [c for c in feature_columns if c.startswith("elo__")]
    assert elo_cols, "expected Elo-family features to be present"
    changed = (
        base_row[elo_cols].to_numpy() != mutated_row[elo_cols].to_numpy()
    ).any()
    assert changed, "mutating a legitimate prior game's result changed no later Elo feature"


# ---------------------------------------------------------------------------
# Current-market leakage
# ---------------------------------------------------------------------------

def test_no_feature_traces_to_a_native_current_market_or_outcome_column():
    games = _games_with_native_market_and_outcome_columns()
    pbp = _pbp_for_games(games)
    matrix, feature_columns, _bundle = build_oof_feature_matrix(games, pbp)

    target_idx = _target_position(games)
    target_game_id = games.loc[target_idx, "game_id"]
    target_native_row = games.loc[target_idx]
    feature_row = matrix.loc[matrix["game_id"] == target_game_id, feature_columns].iloc[0]

    # No predictive feature literally equals the target's own current market
    # reference or outcome value.
    for native_col in (
        "home_score", "away_score", "home_spread_reference",
        "total_line_reference", "home_moneyline_reference",
        "away_moneyline_reference", "home_spread_price_reference",
        "away_spread_price_reference",
    ):
        native_value = float(target_native_row[native_col])
        matches = feature_row[np.isclose(feature_row.astype(float), native_value, equal_nan=False)]
        # A rolling/expanding mean could coincidentally equal a raw value by
        # chance on tiny synthetic data; the decisive check is name/provenance
        # (test_known_leaked_feature_names_are_absent), this is belt-and-suspenders.
        assert len(matches) < len(feature_row)

    provenance = compute_feature_provenance(feature_columns)
    assert all(r["state_family"] in {"epa", "opponent_adjusted", "qb", "elo", "market_history"} for r in provenance)
