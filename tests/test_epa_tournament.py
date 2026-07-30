"""Fast tests for the EPA tournament candidates on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.selection.epa_tournament import (
    CANDIDATES,
    PROB_COLS,
    candidate_gbm,
    candidate_jointscore_epa,
    candidate_logistic,
    candidate_market_residual,
    candidate_stacked,
    run_walk_forward,
)

FEATURES = ["home_spread", "total_line", "epa_net_edge_season_mean", "rest_diff"]


def _matrix(seed=0, n_per_season=120, seasons=(2020, 2021, 2022)):
    rng = np.random.default_rng(seed)
    rows = []
    gid = 0
    for s in seasons:
        for i in range(n_per_season):
            gid += 1
            hs = float(np.round(rng.normal(0, 6)))
            tl = float(np.round(rng.normal(45, 5)))
            edge = float(rng.normal(0, 0.1))
            margin = -hs + rng.normal(0, 13)
            total = tl + rng.normal(0, 13)
            rows.append(
                dict(
                    game_id=f"g{gid}", season=s, week=(i % 18) + 1,
                    home_spread=hs, total_line=tl,
                    epa_net_edge_season_mean=edge, rest_diff=float(rng.integers(-4, 5)),
                    home_margin=margin, total_points=total,
                    home_win=int(margin > 0), home_cover=int(margin + hs > 0),
                    over=int(total > tl),
                    ref_ml_home_prob=0.5, ref_cover_prob=0.5, ref_over_prob=0.5,
                )
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fn", [candidate_market_residual, candidate_logistic])
def test_simple_candidates_valid_probs(fn):
    m = _matrix()
    train = m[m["season"] < 2022]
    test = m[m["season"] == 2022]
    out = fn(train, test, FEATURES)
    for col in PROB_COLS:
        assert col in out.columns
        assert ((out[col] > 0) & (out[col] < 1)).all()


def test_gbm_and_jointscore_valid_probs():
    m = _matrix()
    train = m[m["season"] < 2022]
    test = m[m["season"] == 2022]
    for out in (
        candidate_gbm(train, test, FEATURES, calibration_season=2021),
        candidate_jointscore_epa(train, test, FEATURES, calibration_season=2021),
    ):
        for col in PROB_COLS:
            assert ((out[col] >= 0) & (out[col] <= 1)).all()


def test_stacked_valid_probs():
    m = _matrix()
    train = m[m["season"] < 2022]
    test = m[m["season"] == 2022]
    out = candidate_stacked(
        train, test, FEATURES, calibration_season=2021,
        market_prob_col_map=["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"],
    )
    assert out is not None
    for col in PROB_COLS:
        assert ((out[col] >= 0) & (out[col] <= 1)).all()


def test_walk_forward_produces_all_candidates():
    m = _matrix(seasons=(2020, 2021, 2022, 2023))
    results = run_walk_forward(
        m, FEATURES, test_seasons=[2022, 2023],
        market_prob_cols=["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"],
    )
    # all five candidates should produce pooled predictions
    assert set(results.keys()) == set(CANDIDATES)
    for name, df in results.items():
        assert len(df) > 0
        assert "home_win_probability_no_tie" in df.columns
