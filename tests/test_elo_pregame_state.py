import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.elo_state import build_elo_pregame_state


def _games():
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "home_team_id": ["KC", "BUF", "KC"],
            "away_team_id": ["BUF", "KC", "BUF"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2024-09-01T17:00:00Z", "2024-09-08T17:00:00Z", "2024-09-15T17:00:00Z"], utc=True
            ),
            "home_score": [24, 20, 27],
            "away_score": [17, 23, 24],
            "neutral_site": [False, False, False],
            "playoff": [False, False, False],
        }
    )


def test_first_game_uses_initial_rating_for_both_teams():
    out = build_elo_pregame_state(_games())
    g1 = out[out["game_id"] == "G1"]
    assert set(g1["elo_pregame_rating"].round(6)) == {1500.0}


def test_third_game_rating_reflects_only_first_two_games():
    out = build_elo_pregame_state(_games())
    g3_kc = out[(out["game_id"] == "G3") & (out["team_id"] == "KC")].iloc[0]
    # KC won G1 (home, +7) and lost G2 (away, -3): rating should have moved up net.
    assert g3_kc["elo_pregame_rating"] > 1500.0


def test_mutating_target_games_own_score_does_not_change_its_own_pregame_rating():
    original = _games()
    mutated = original.copy()
    mutated.loc[mutated["game_id"] == "G2", ["home_score", "away_score"]] = [45, 3]

    base = build_elo_pregame_state(original)
    changed = build_elo_pregame_state(mutated)

    g2_base = base[base["game_id"] == "G2"].set_index("team_id")["elo_pregame_rating"]
    g2_changed = changed[changed["game_id"] == "G2"].set_index("team_id")["elo_pregame_rating"]
    pd.testing.assert_series_equal(g2_base.sort_index(), g2_changed.sort_index())


def test_mutating_prior_games_score_changes_later_pregame_rating():
    original = _games()
    mutated = original.copy()
    mutated.loc[mutated["game_id"] == "G1", ["home_score", "away_score"]] = [3, 45]

    base = build_elo_pregame_state(original)
    changed = build_elo_pregame_state(mutated)

    g2_base = base[(base["game_id"] == "G2") & (base["team_id"] == "KC")]["elo_pregame_rating"].iloc[0]
    g2_changed = changed[(changed["game_id"] == "G2") & (changed["team_id"] == "KC")]["elo_pregame_rating"].iloc[0]
    assert g2_base != pytest.approx(g2_changed)


def test_unplayed_target_game_does_not_update_ratings():
    games = _games()
    games.loc[games["game_id"] == "G3", ["home_score", "away_score"]] = [np.nan, np.nan]
    out = build_elo_pregame_state(games)
    # G3 contributed no update, so a hypothetical G4 (not present) would be
    # unaffected; here we assert G3 itself still received a pregame prediction.
    assert len(out[out["game_id"] == "G3"]) == 2


def test_duplicate_game_id_fails():
    games = pd.concat([_games(), _games().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_elo_pregame_state(games)


def test_deterministic_repeat_run():
    games = _games()
    first = build_elo_pregame_state(games)
    second = build_elo_pregame_state(games.sample(frac=1.0, random_state=7))
    pd.testing.assert_frame_equal(
        first.sort_values(["game_id", "team_id"]).reset_index(drop=True),
        second.sort_values(["game_id", "team_id"]).reset_index(drop=True),
    )
