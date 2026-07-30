"""Fast unit tests for the augmented feature matrix helpers + leakage invariants.

The full real-PBP pipeline is exercised in examples/ and the tournament; here we
keep tests fast with a small synthetic team-game frame, focusing on the
leakage-critical shift behaviour and the engineered differentials/rest features.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import _diff_features, _rest_features
from nfl_hybrid.features.pregame_rolling import (
    PregameRollingConfig,
    build_team_pregame_features,
)


def test_diff_features_signs():
    matrix = pd.DataFrame(
        {
            "home_offense_epa_per_play__season_mean": [0.10],
            "away_defense_allowed_epa_per_play__season_mean": [0.02],
            "away_offense_epa_per_play__season_mean": [0.05],
            "home_defense_allowed_epa_per_play__season_mean": [0.03],
            "home_offense_success_rate__season_mean": [0.50],
            "away_offense_success_rate__season_mean": [0.45],
            "home_offense_dropback_epa__season_mean": [0.12],
            "away_offense_dropback_epa__season_mean": [0.08],
            "home_offense_rush_epa__season_mean": [-0.02],
            "away_offense_rush_epa__season_mean": [0.01],
        }
    )
    out, names = _diff_features(matrix)
    assert out["epa_matchup_home_season_mean"].iloc[0] == pytest.approx(0.08)
    assert out["epa_matchup_away_season_mean"].iloc[0] == pytest.approx(0.02)
    assert out["epa_net_edge_season_mean"].iloc[0] == pytest.approx(0.06)
    assert out["success_rate_net_season_mean"].iloc[0] == pytest.approx(0.05)
    assert "dropback_epa_net_season_mean" in names


def test_rest_features_short_week_and_diff():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2024, 2024, 2024],
            "home_team_id": ["KC", "KC", "BUF"],
            "away_team_id": ["BUF", "DET", "KC"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2024-09-08T17:00:00Z", "2024-09-12T00:15:00Z", "2024-09-15T17:00:00Z"],
                utc=True,
            ),
        }
    )
    out, cols = _rest_features(games)
    # KC plays g1 (Sun) then g2 (Thu) -> short week for KC at g2
    assert out.loc[games.index[games["game_id"] == "g2"][0], "home_short_week"] == 1.0
    # first game has NaN rest
    assert np.isnan(out.loc[games.index[games["game_id"] == "g1"][0], "home_rest_days"])
    assert set(cols) == {"home_rest_days", "away_rest_days", "rest_diff", "home_short_week", "away_short_week"}


def _synthetic_team_game(n_seasons=1, teams=("KC", "BUF"), games_per=6):
    rows = []
    gid = 0
    for season in range(2020, 2020 + n_seasons):
        for wk in range(1, games_per + 1):
            gid += 1
            a, b = teams
            rows.append(dict(game_id=f"G{gid}", season=season, week=wk, team_id=a, opponent_id=b,
                             offense_epa_per_play=0.1 * wk, defense_allowed_epa_per_play=-0.05 * wk))
            rows.append(dict(game_id=f"G{gid}", season=season, week=wk, team_id=b, opponent_id=a,
                             offense_epa_per_play=-0.1 * wk, defense_allowed_epa_per_play=0.05 * wk))
    return pd.DataFrame(rows)


def _synthetic_games(team_game):
    g = team_game[team_game["team_id"] == "KC"][["game_id", "season", "week"]].copy()
    g["home_team_id"] = "KC"
    g["away_team_id"] = "BUF"
    g["scheduled_kickoff_utc"] = pd.to_datetime("2020-09-01T17:00:00Z", utc=True) + pd.to_timedelta(
        7 * (g["week"] - 1), unit="D"
    )
    return g.reset_index(drop=True)


def test_first_game_has_no_rolling_history():
    tg = _synthetic_team_game()
    games = _synthetic_games(tg)
    tp = build_team_pregame_features(tg, games, config=PregameRollingConfig(windows=(4,)))
    tp = tp.merge(games[["game_id", "scheduled_kickoff_utc"]], on="game_id")
    tp = tp.sort_values(["team_id", "scheduled_kickoff_utc"], kind="stable")
    count_cols = [c for c in tp.columns if c.endswith("__season_count") or c.endswith("__last4_count")]
    first = tp.groupby("team_id").head(1)
    assert float(first[count_cols].fillna(0).to_numpy().sum()) == 0.0


def test_future_corruption_does_not_change_earlier_features():
    tg = _synthetic_team_game()
    games = _synthetic_games(tg)
    cfg = PregameRollingConfig(windows=(4,))
    tp = build_team_pregame_features(tg, games, config=cfg)

    corrupt = tg.copy()
    last_idx = corrupt.groupby("team_id").tail(1).index
    corrupt.loc[last_idx, ["offense_epa_per_play", "defense_allowed_epa_per_play"]] = -999.0
    tp_c = build_team_pregame_features(corrupt, games, config=cfg)

    last_games = set(tg.groupby("team_id").tail(1)["game_id"])
    m = tp.merge(tp_c, on=["game_id", "team_id"], suffixes=("_o", "_c"))
    m = m[~m["game_id"].isin(last_games)]
    feat = [c for c in tp.columns if c.endswith(("_mean", "_hl4"))]
    for c in feat:
        a, b = m[f"{c}_o"].to_numpy(float), m[f"{c}_c"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            assert np.abs(a[both] - b[both]).max() < 1e-9
