"""Fast unit tests for the augmented feature matrix helpers + leakage invariants.

The full real-PBP pipeline is exercised in examples/ and the tournament; here we
keep tests fast with a small synthetic team-game frame, focusing on the
leakage-critical shift behaviour and the engineered differentials/rest features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation.market_relative import (
    MarketRelativeConfig,
    evaluate_market_relative,
)
from nfl_hybrid.features.augmented_matrix import (
    _diff_features,
    _edge_to_nullable_binary,
    _outcome_columns,
    _rest_features,
    build_augmented_feature_matrix,
)
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


# ================================================================================
# R1: tie/push labels are excluded (pd.NA), never class zero.
# ================================================================================
def _games(specs):
    """Minimal games frame for ``_outcome_columns`` from per-game score/line specs."""
    rows = []
    for i, s in enumerate(specs):
        rows.append(
            {
                "game_id": f"g{i}", "season": 2022, "week": 1,
                "home_team_id": "AAA", "away_team_id": "BBB",
                "home_spread_reference": s.get("home_spread"),
                "total_line_reference": s.get("total_line"),
                "home_score": s.get("home_score"),
                "away_score": s.get("away_score"),
            }
        )
    return pd.DataFrame(rows)


def test_edge_helper_tie_push_and_nonfinite():
    out = _edge_to_nullable_binary(pd.Series([2.0, -2.0, 0.0, 1e-12, np.inf, np.nan]))
    assert str(out.dtype) == "Int8"
    assert out.iloc[0] == 1                 # positive edge
    assert out.iloc[1] == 0                 # negative edge
    assert pd.isna(out.iloc[2])             # exact zero -> NA
    assert pd.isna(out.iloc[3])             # within tolerance of zero -> NA
    assert pd.isna(out.iloc[4])             # +inf -> NA
    assert pd.isna(out.iloc[5])             # NaN -> NA


def test_home_win_tie_is_null():
    g = _outcome_columns(
        _games(
            [
                {"home_score": 24, "away_score": 17, "home_spread": -3.0, "total_line": 44.0},
                {"home_score": 17, "away_score": 24, "home_spread": -3.0, "total_line": 44.0},
                {"home_score": 20, "away_score": 20, "home_spread": -3.0, "total_line": 44.0},
            ]
        )
    )
    assert g["home_win"].iloc[0] == 1       # margin +7
    assert g["home_win"].iloc[1] == 0       # margin -7
    assert pd.isna(g["home_win"].iloc[2])   # tie -> NA (not an away win)
    assert str(g["home_win"].dtype) == "Int8"


def test_ats_push_is_null_and_half_point_never_pushes():
    g = _outcome_columns(
        _games(
            [
                {"home_score": 24, "away_score": 21, "home_spread": -3.0, "total_line": 44.0},   # m=3, push
                {"home_score": 24, "away_score": 20, "home_spread": -3.0, "total_line": 44.0},   # m=4, cover
                {"home_score": 22, "away_score": 20, "home_spread": -3.0, "total_line": 44.0},   # m=2, no cover
                {"home_score": 24, "away_score": 21, "home_spread": -3.5, "total_line": 44.0},   # m=3, half-point
                {"home_score": 24, "away_score": 20, "home_spread": -3.5, "total_line": 44.0},   # m=4, half-point
            ]
        )
    )
    assert pd.isna(g["home_cover"].iloc[0])  # exact ATS push -> NA (not an away cover)
    assert g["home_cover"].iloc[1] == 1
    assert g["home_cover"].iloc[2] == 0
    # a half-point spread can never push an integer final margin
    assert g["home_cover"].iloc[3] == 0      # edge -0.5
    assert g["home_cover"].iloc[4] == 1      # edge +0.5
    assert g["home_cover"].notna().iloc[3] and g["home_cover"].notna().iloc[4]


def test_total_push_is_null_and_half_point_never_pushes():
    g = _outcome_columns(
        _games(
            [
                {"home_score": 24, "away_score": 20, "home_spread": -3.0, "total_line": 44.0},   # total 44, push
                {"home_score": 24, "away_score": 21, "home_spread": -3.0, "total_line": 44.0},   # total 45, over
                {"home_score": 23, "away_score": 20, "home_spread": -3.0, "total_line": 44.0},   # total 43, under
                {"home_score": 24, "away_score": 20, "home_spread": -3.0, "total_line": 44.5},   # total 44, half-point
                {"home_score": 24, "away_score": 21, "home_spread": -3.0, "total_line": 44.5},   # total 45, half-point
            ]
        )
    )
    assert pd.isna(g["over"].iloc[0])        # exact total push -> NA (not an under)
    assert g["over"].iloc[1] == 1
    assert g["over"].iloc[2] == 0
    assert g["over"].iloc[3] == 0            # edge -0.5
    assert g["over"].iloc[4] == 1            # edge +0.5


def test_missing_inputs_null_only_applicable_targets():
    g = _outcome_columns(
        _games(
            [
                {"home_score": None, "away_score": 20, "home_spread": -3.0, "total_line": 44.0},  # missing score
                {"home_score": 24, "away_score": 20, "home_spread": None, "total_line": 45.0},    # missing spread
                {"home_score": 27, "away_score": 20, "home_spread": -3.0, "total_line": None},     # missing total line
            ]
        )
    )
    # missing score -> every applicable binary target is null
    assert pd.isna(g["home_win"].iloc[0])
    assert pd.isna(g["home_cover"].iloc[0])
    assert pd.isna(g["over"].iloc[0])
    # missing spread -> only ATS is null (moneyline + total still resolve)
    assert pd.isna(g["home_cover"].iloc[1])
    assert g["home_win"].iloc[1] == 1        # margin +4
    assert g["over"].iloc[1] == 0            # total 44 vs line 45
    # missing total line -> only totals is null (moneyline + ATS still resolve)
    assert pd.isna(g["over"].iloc[2])
    assert g["home_win"].iloc[2] == 1        # margin +7
    assert g["home_cover"].iloc[2] == 1      # edge +4


def test_augmented_matrix_retains_ats_push_row():
    """An ATS-push game stays in the matrix for regression; only home_cover is null."""
    fixture_dir = Path(__file__).parent / "fixtures" / "live_features"
    games = pd.read_csv(fixture_dir / "games.csv")
    pbp = pd.read_csv(fixture_dir / "pbp.csv")
    # training path = completed games only (upcoming games have no play-by-play).
    completed = games[games["home_score"].notna()].copy()

    gid = "2021_05_BUF_NYJ"  # a completed game with full rolling history
    row = completed["game_id"].astype(str) == gid
    assert row.any()
    # force an exact ATS push (margin +4, spread -4) without pushing the total.
    completed.loc[row, "home_score"] = 24
    completed.loc[row, "away_score"] = 20
    completed.loc[row, "home_spread_reference"] = -4.0

    matrix, _ = build_augmented_feature_matrix(completed, pbp)
    sub = matrix[matrix["game_id"].astype(str) == gid]
    assert len(sub) == 1                                 # push row preserved
    r = sub.iloc[0]
    assert pd.isna(r["home_cover"])                      # ATS push -> null target
    assert pd.notna(r["home_margin"]) and float(r["home_margin"]) == 4.0
    assert pd.notna(r["total_points"]) and float(r["total_points"]) == 44.0
    assert pd.notna(r["home_win"]) and int(r["home_win"]) == 1   # non-pushed markets kept
    assert pd.notna(r["over"])
    assert str(matrix["home_cover"].dtype) == "Int8"


def test_evaluation_excludes_only_the_pushed_market():
    """A push removes the game from that market's graded sample only."""
    hs = np.array([-3.0, -3.0, -3.0, -7.0])
    tl = np.array([44.0, 44.0, 44.0, 44.0])
    margin = np.array([7.0, -7.0, 3.0, 10.0])   # g2: margin 3 + spread -3 -> ATS push
    total = np.array([50.0, 40.0, 45.0, 48.0])  # no total pushes
    frame = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(4)], "season": 2022,
            "home_spread": hs, "total_line": tl,
            "home_margin": margin, "total_points": total,
            "predicted_margin": -hs,
            "home_win_probability_no_tie": 0.6,
            "home_cover_probability_no_push": 0.55,
            "over_probability_no_push": 0.5,
        }
    )
    frame["home_win"] = _edge_to_nullable_binary(frame["home_margin"])
    frame["home_cover"] = _edge_to_nullable_binary(frame["home_margin"] + frame["home_spread"])
    frame["over"] = _edge_to_nullable_binary(frame["total_points"] - frame["total_line"])

    res = evaluate_market_relative(
        frame, config=MarketRelativeConfig(bootstrap_repetitions=50)
    )
    pooled = res["probability_scorecard"]
    pooled = pooled[pooled["segment"] == "pooled"].set_index("market")
    assert int(pooled.loc["home_win", "n"]) == 4      # no tie -> all four graded
    assert int(pooled.loc["over", "n"]) == 4          # no total push -> all four graded
    assert int(pooled.loc["home_cover", "n"]) == 3    # exactly the one ATS push excluded


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
