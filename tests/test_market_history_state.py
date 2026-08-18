import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.market_history import build_market_history_pregame_state


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
            # KC (home) covers -3.0 in G1 (margin +7); KC (away) fails to cover
            # +3.5 in G2 (margin -3, i.e. lost by 3, needed to lose by < 3.5:
            # away_margin = -3 > -3.5 so KC DOES cover here -- picked deliberately
            # non-trivial values below instead, see test assertions.
            "home_score": [24, 20, 27],
            "away_score": [17, 23, 24],
            "home_spread_reference": [-3.0, -2.5, -3.5],
            "total_line_reference": [44.0, 45.0, 60.0],
        }
    )


def test_first_game_has_no_market_history():
    out = build_market_history_pregame_state(_games())
    first = out.sort_values(["team_id", "event_time"]).groupby("team_id").head(1)
    assert first["market_ats_win__all_count"].fillna(0).eq(0).all()
    assert first["market_ats_win__all_mean"].isna().all()


def test_second_game_reflects_only_first_games_ats_result():
    out = build_market_history_pregame_state(_games())
    # G1: home_margin = 7, home_spread = -3.0 -> edge = 4.0 > 0 -> KC covered.
    kc_g2 = out[(out["game_id"] == "G2") & (out["team_id"] == "KC")].iloc[0]
    assert kc_g2["market_ats_win__all_count"] == 1
    assert kc_g2["market_ats_win__all_mean"] == pytest.approx(1.0)

    buf_g2 = out[(out["game_id"] == "G2") & (out["team_id"] == "BUF")].iloc[0]
    assert buf_g2["market_ats_win__all_mean"] == pytest.approx(0.0)


def test_push_is_excluded_not_scored_as_loss():
    games = _games()
    # Make G1 an exact push: home_margin(7) + home_spread == 0 -> home_spread = -7.0
    games.loc[games["game_id"] == "G1", "home_spread_reference"] = -7.0
    out = build_market_history_pregame_state(games)
    kc_g2 = out[(out["game_id"] == "G2") & (out["team_id"] == "KC")].iloc[0]
    assert kc_g2["market_ats_win__all_count"] == 0
    assert np.isnan(kc_g2["market_ats_win__all_mean"])


def test_mutating_targets_own_line_does_not_change_its_own_state():
    original = _games()
    mutated = original.copy()
    mutated.loc[mutated["game_id"] == "G2", "home_spread_reference"] = 12.0

    base = build_market_history_pregame_state(original)
    changed = build_market_history_pregame_state(mutated)

    base_g2 = base[base["game_id"] == "G2"].set_index("team_id")["market_ats_win__all_mean"]
    changed_g2 = changed[changed["game_id"] == "G2"].set_index("team_id")["market_ats_win__all_mean"]
    pd.testing.assert_series_equal(base_g2.sort_index(), changed_g2.sort_index())


def test_mutating_prior_games_line_changes_later_state():
    original = _games()
    mutated = original.copy()
    # Flip G1 from a KC cover to a KC non-cover (home_margin=7, so spread must
    # be more negative than -7 for the home favorite to fail to cover).
    mutated.loc[mutated["game_id"] == "G1", "home_spread_reference"] = -20.0

    base = build_market_history_pregame_state(original)
    changed = build_market_history_pregame_state(mutated)

    base_val = base[(base["game_id"] == "G2") & (base["team_id"] == "KC")]["market_ats_win__all_mean"].iloc[0]
    changed_val = changed[(changed["game_id"] == "G2") & (changed["team_id"] == "KC")]["market_ats_win__all_mean"].iloc[0]
    assert base_val == pytest.approx(1.0)
    assert changed_val == pytest.approx(0.0)


def test_duplicate_game_id_fails():
    games = pd.concat([_games(), _games().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_market_history_pregame_state(games)
