"""Fix 6 Section 11 acceptance suite for the Week 1 / early-season prior
(:mod:`nfl_hybrid.features.week1_prior`).

CI-safe: synthetic fixtures only, no private backfill, no network.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.pregame_rolling import build_team_pregame_features
from nfl_hybrid.features.week1_prior import (
    DEFAULT_NEUTRAL_FALLBACK,
    GAME_RESULT_METRICS,
    NEUTRAL_FALLBACK,
    NEUTRAL_MARGIN,
    NEUTRAL_POINTS_ALLOWED,
    NEUTRAL_POINTS_SCORED,
    NEUTRAL_PRIOR_SOURCE,
    NEUTRAL_WIN_RATE,
    PRIOR_SEASON_AVAILABLE,
    Week1PriorConfig,
    apply_week1_blend,
    build_game_result_team_game,
    compute_week1_prior_config_hash,
)

_NEUTRAL = dict(DEFAULT_NEUTRAL_FALLBACK)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
            "season": [2023, 2023, 2023, 2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 1, 2, 3, 1],
            "home_team_id": ["KC", "BUF", "KC", "KC", "BUF", "KC", "NE"],
            "away_team_id": ["BUF", "KC", "BUF", "BUF", "KC", "BUF", "SEA"],
            "scheduled_kickoff_utc": pd.to_datetime(
                [
                    "2023-09-03T17:00:00Z",
                    "2023-09-10T17:00:00Z",
                    "2023-09-17T17:00:00Z",
                    "2024-09-01T17:00:00Z",
                    "2024-09-08T17:00:00Z",
                    "2024-09-15T17:00:00Z",
                    "2024-09-01T17:00:00Z",
                ],
                utc=True,
            ),
            "home_score": [24, 23, 27, 30, 14, 17, 10],
            "away_score": [17, 20, 24, 10, 21, 16, 24],
        }
    )


def _pipeline(games: pd.DataFrame, *, k: float = 4.0) -> pd.DataFrame:
    team_game = build_game_result_team_game(games)
    team_pregame = build_team_pregame_features(
        team_game, games, metric_columns=list(GAME_RESULT_METRICS)
    )
    config = Week1PriorConfig(k=k, neutral_fallback=dict(_NEUTRAL))
    return apply_week1_blend(team_pregame, team_game, config=config)


# ---------------------------------------------------------------------------
# A. Week 1 contains no current-season results (poison pill on its own score).
# ---------------------------------------------------------------------------
def test_a_week1_own_score_mutation_does_not_change_its_own_blend():
    original = _games()
    mutated = original.copy()
    mutated.loc[mutated["game_id"] == "G4", ["home_score", "away_score"]] = [3, 55]

    base = _pipeline(original)
    changed = _pipeline(mutated)

    cols = [f"{m}__week1_blended" for m in GAME_RESULT_METRICS]
    base_g4 = base[base["game_id"] == "G4"].set_index("team_id")[cols]
    changed_g4 = changed[changed["game_id"] == "G4"].set_index("team_id")[cols]
    pd.testing.assert_frame_equal(base_g4.sort_index(), changed_g4.sort_index())


# ---------------------------------------------------------------------------
# B. Prior-season final state may affect next-season Week 1.
# ---------------------------------------------------------------------------
def test_b_prior_season_final_state_sets_next_season_week1():
    out = _pipeline(_games())
    kc_g4 = out[(out["game_id"] == "G4") & (out["team_id"] == "KC")].iloc[0]

    # KC's full 2023 average: margins +7, -3, +3 -> mean 7/3.
    assert kc_g4["prior_status"] == PRIOR_SEASON_AVAILABLE
    assert kc_g4["margin__week1_blended"] == pytest.approx((7 - 3 + 3) / 3)
    assert kc_g4["points_scored__week1_blended"] == pytest.approx((24 + 20 + 27) / 3)
    assert kc_g4["win__week1_blended"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# C. Future/current-season games cannot affect Week 1.
# ---------------------------------------------------------------------------
def test_c_later_current_season_game_does_not_affect_week1():
    original = _games()
    mutated = original.copy()
    mutated.loc[mutated["game_id"] == "G6", ["home_score", "away_score"]] = [55, 0]

    base = _pipeline(original)
    changed = _pipeline(mutated)

    cols = [f"{m}__week1_blended" for m in GAME_RESULT_METRICS]
    base_g4 = base[base["game_id"] == "G4"].set_index("team_id")[cols]
    changed_g4 = changed[changed["game_id"] == "G4"].set_index("team_id")[cols]
    pd.testing.assert_frame_equal(base_g4.sort_index(), changed_g4.sort_index())


# ---------------------------------------------------------------------------
# D. A team with no prior season receives the declared neutral prior.
# ---------------------------------------------------------------------------
def test_d_no_prior_season_uses_neutral_fallback():
    out = _pipeline(_games())

    kc_g1 = out[(out["game_id"] == "G1") & (out["team_id"] == "KC")].iloc[0]
    assert kc_g1["prior_status"] == NEUTRAL_FALLBACK
    for metric, value in _NEUTRAL.items():
        assert kc_g1[f"{metric}__week1_blended"] == pytest.approx(value)

    ne_g7 = out[(out["game_id"] == "G7") & (out["team_id"] == "NE")].iloc[0]
    assert ne_g7["prior_status"] == NEUTRAL_FALLBACK
    sea_g7 = out[(out["game_id"] == "G7") & (out["team_id"] == "SEA")].iloc[0]
    assert sea_g7["prior_status"] == NEUTRAL_FALLBACK


def test_neutral_fallback_is_fixed_untuned_and_never_outcome_derived():
    """No `derive_neutral_fallback`-style function exists in this module --
    the neutral constants are hardcoded module constants, labeled exactly
    as an untuned engineering prior, never computed from repository data."""
    assert not hasattr(
        __import__("nfl_hybrid.features.week1_prior", fromlist=["derive_neutral_fallback"]),
        "derive_neutral_fallback",
    )
    assert DEFAULT_NEUTRAL_FALLBACK == {
        "points_scored": NEUTRAL_POINTS_SCORED,
        "points_allowed": NEUTRAL_POINTS_ALLOWED,
        "margin": NEUTRAL_MARGIN,
        "win": NEUTRAL_WIN_RATE,
    }
    assert NEUTRAL_POINTS_SCORED == 22.0
    assert NEUTRAL_POINTS_ALLOWED == 22.0
    assert NEUTRAL_MARGIN == 0.0
    assert NEUTRAL_WIN_RATE == 0.5
    assert "not estimated from 2020-2024 data" in NEUTRAL_PRIOR_SOURCE

    config = Week1PriorConfig(k=4.0)  # relies on the default factory
    assert config.neutral_fallback == DEFAULT_NEUTRAL_FALLBACK
    assert config.neutral_prior_source == NEUTRAL_PRIOR_SOURCE


# ---------------------------------------------------------------------------
# E. As completed games accumulate, current-season evidence weight rises
#    monotonically (checked via the exact blended values it produces).
# ---------------------------------------------------------------------------
def test_e_current_season_weight_rises_monotonically():
    out = _pipeline(_games(), k=4.0)
    kc = out[out["team_id"] == "KC"].set_index("game_id")

    prior = (7 - 3 + 3) / 3  # KC's 2023 margin average
    w1 = 0.0
    w2 = 1 / (1 + 4)
    w3 = 2 / (2 + 4)
    assert w1 < w2 < w3

    expected_g4 = prior  # n=0 -> explicit branch, pure prior
    expected_g5 = w2 * 20.0 + (1 - w2) * prior  # only prior game: G4 margin = +20
    expected_g6 = w3 * 13.5 + (1 - w3) * prior  # mean(G4=+20, G5 KC margin=+7) = 13.5

    assert kc.loc["G4", "margin__week1_blended"] == pytest.approx(expected_g4)
    assert kc.loc["G5", "margin__week1_blended"] == pytest.approx(expected_g5)
    assert kc.loc["G6", "margin__week1_blended"] == pytest.approx(expected_g6)


def test_n_equals_zero_is_exactly_prior_reference_no_nan_anywhere():
    """Plus-sign blend formula, explicit n=0 branch: every Week-1 (n=0) row's
    blended value must equal prior_reference exactly, and the full output
    must contain zero NaN values across every `__week1_blended` column."""
    out = _pipeline(_games(), k=4.0)
    cols = [f"{m}__week1_blended" for m in GAME_RESULT_METRICS]

    assert out[cols].isna().sum().sum() == 0

    week1_rows = out[out["team_season_prior_games"] == 0]
    assert len(week1_rows) > 0
    for _, row in week1_rows.iterrows():
        for metric in GAME_RESULT_METRICS:
            if row["prior_status"] == NEUTRAL_FALLBACK:
                expected = _NEUTRAL[metric]
            else:
                # PRIOR_SEASON_AVAILABLE: exact team/season prior-season mean,
                # recomputed independently here for the assertion.
                team_game = build_game_result_team_game(_games())
                prior_season = team_game[
                    (team_game["team_id"] == row["team_id"]) & (team_game["season"] == row["season"] - 1)
                ]
                expected = prior_season[metric].mean()
            assert row[f"{metric}__week1_blended"] == pytest.approx(expected), (row["team_id"], metric)


# ---------------------------------------------------------------------------
# F. No blend parameter changes after the outer 2024 evaluation -- enforced
#    at the type level (frozen dataclass) plus a stable content hash.
# ---------------------------------------------------------------------------
def test_f_config_is_frozen_and_hash_is_stable():
    config = Week1PriorConfig(k=4.0, neutral_fallback=dict(_NEUTRAL))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.k = 6.0  # type: ignore[misc]

    first = compute_week1_prior_config_hash(config)
    second = compute_week1_prior_config_hash(Week1PriorConfig(k=4.0, neutral_fallback=dict(_NEUTRAL)))
    assert first == second

    different = compute_week1_prior_config_hash(Week1PriorConfig(k=6.0, neutral_fallback=dict(_NEUTRAL)))
    assert different != first


# ---------------------------------------------------------------------------
# G. Historical replay is deterministic.
# ---------------------------------------------------------------------------
def test_g_deterministic_replay_regardless_of_row_order():
    games = _games()
    first = _pipeline(games)
    second = _pipeline(games.sample(frac=1.0, random_state=11).reset_index(drop=True))

    sort_cols = ["game_id", "team_id"]
    pd.testing.assert_frame_equal(
        first.sort_values(sort_cols).reset_index(drop=True),
        second.sort_values(sort_cols).reset_index(drop=True),
    )


def test_week1_config_rejects_non_positive_k():
    with pytest.raises(ValueError):
        Week1PriorConfig(k=0.0, neutral_fallback=dict(_NEUTRAL))


def test_week1_config_requires_every_metric_fallback():
    with pytest.raises(ValueError):
        Week1PriorConfig(k=4.0, neutral_fallback={"margin": 0.0})


def test_build_game_result_team_game_tie_is_excluded_from_win():
    games = _games()
    games.loc[games["game_id"] == "G4", ["home_score", "away_score"]] = [17, 17]
    team_game = build_game_result_team_game(games)
    g4 = team_game[team_game["game_id"] == "G4"]
    assert g4["win"].isna().all()
    assert (g4["margin"] == 0).all()
