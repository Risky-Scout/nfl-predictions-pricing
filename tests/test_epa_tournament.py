"""Fast tests for the EPA tournament candidates on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import _edge_to_nullable_binary
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


def _matrix(seed=0, n_per_season=120, seasons=(2020, 2021, 2022), force_pushes=False):
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
            if force_pushes and i % 40 == 0:
                # exact tie + exact ATS push + exact total push, all in one game
                margin = -hs  # ATS push (margin + hs == 0)
                if hs == 0.0:
                    margin = 0.0  # also a moneyline tie
                total = tl  # total push
            rows.append(
                dict(
                    game_id=f"g{gid}", season=s, week=(i % 18) + 1,
                    home_spread=hs, total_line=tl,
                    epa_net_edge_season_mean=edge, rest_diff=float(rng.integers(-4, 5)),
                    home_margin=margin, total_points=total,
                    ref_ml_home_prob=0.5, ref_cover_prob=0.5, ref_over_prob=0.5,
                )
            )
    frame = pd.DataFrame(rows)
    # Use the canonical label policy: ties/pushes become nullable Int8 pd.NA, never
    # a spurious class zero. This mirrors build_augmented_feature_matrix exactly.
    frame["home_win"] = _edge_to_nullable_binary(frame["home_margin"])
    frame["home_cover"] = _edge_to_nullable_binary(frame["home_margin"] + frame["home_spread"])
    frame["over"] = _edge_to_nullable_binary(frame["total_points"] - frame["total_line"])
    return frame


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


# --- R1: nullable-label consumption ------------------------------------------- #
def test_matrix_targets_are_nullable_int8_with_pushes():
    """The synthetic matrix carries real nullable pushes/ties (pd.NA), not zeros."""
    m = _matrix(force_pushes=True)
    for col in ("home_win", "home_cover", "over"):
        assert str(m[col].dtype) == "Int8"
    # forced ATS pushes exist and are null (not class zero)
    assert m["home_cover"].isna().any()
    assert m["over"].isna().any()


def test_gbm_and_logistic_accept_nullable_targets_finite_probs():
    """Direct GBM and logistic candidates train past nullable labels and return
    finite probabilities for every test row (full coverage)."""
    m = _matrix(force_pushes=True)
    train = m[m["season"] < 2022]
    test = m[m["season"] == 2022]
    for out in (
        candidate_gbm(train, test, FEATURES, calibration_season=2021),
        candidate_logistic(train, test, FEATURES),
    ):
        assert len(out) == len(test)  # a push in one market never drops a test row
        for col in PROB_COLS:
            p = out[col].to_numpy(float)
            assert np.isfinite(p).all()
            assert ((p >= 0) & (p <= 1)).all()


def test_jointscore_fits_through_ties_and_pushes():
    """JointScore continuous fitting accepts rows containing a tie or push and the
    calibrators (which must not see nullable labels) still yield finite probs."""
    m = _matrix(force_pushes=True)
    train = m[m["season"] < 2022]
    test = m[m["season"] == 2022]
    out = candidate_jointscore_epa(train, test, FEATURES, calibration_season=2021)
    assert len(out) == len(test)
    for col in PROB_COLS:
        assert np.isfinite(out[col].to_numpy(float)).all()


def test_stacked_does_not_cast_nullable_meta_targets():
    """The stacked candidate filters nullable meta-targets rather than casting them
    to int (which would raise on pd.NA)."""
    m = _matrix(force_pushes=True)
    train = m[m["season"] < 2022]
    test = m[m["season"] == 2022]
    out = candidate_stacked(
        train, test, FEATURES, calibration_season=2021,
        market_prob_col_map=["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"],
    )
    assert out is not None
    assert len(out) == len(test)
    for col in PROB_COLS:
        assert ((out[col] >= 0) & (out[col] <= 1)).all()


# =============================================================================
# R1: degenerate (nullable) sparse-target fallback behaviour for each candidate.
#
# Each market's target is nullable Int8 (pd.NA marks ties/pushes). When a market
# is left with zero valid fitting rows or a single observed class, the classifier
# is undefined and the candidate must return the validated per-test-row market
# baseline -- one probability per test row, never an empty / training-length /
# scalar array. These tests drive those branches directly on tiny synthetic data.
# =============================================================================
BASELINES = ["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"]
_BASELINE_VALUE = {"ref_ml_home_prob": 0.40, "ref_cover_prob": 0.60, "ref_over_prob": 0.55}
_BASE_BY_PROB = dict(zip(PROB_COLS, [0.40, 0.60, 0.55]))


def _mode_edges(mode, n):
    """Decision-edge sequence for a market: 'two' -> both classes, 'one0'/'one1'
    -> a single class, 'none' -> all zero edges (ties/pushes -> pd.NA)."""
    if mode == "two":
        return [6.0 if i % 2 == 0 else -6.0 for i in range(n)]
    if mode == "one1":
        return [6.0] * n
    if mode == "one0":
        return [-6.0] * n
    if mode == "none":
        return [0.0] * n
    raise ValueError(mode)


def _season_rows(rng, season, n, ml, ats, tot, gid_start):
    mle, ae, te = _mode_edges(ml, n), _mode_edges(ats, n), _mode_edges(tot, n)
    rows = []
    for i in range(n):
        m = mle[i]
        tp = 44.0 + float(rng.integers(-7, 8))   # vary totals so residuals are non-degenerate
        rows.append(
            dict(
                game_id=f"g{gid_start + i}", season=season, week=(i % 18) + 1,
                home_spread=ae[i] - m,        # home_margin + home_spread == ats edge
                total_line=tp - te[i],        # total_points - total_line == tot edge
                epa_net_edge_season_mean=float(rng.normal(0, 0.1)),
                rest_diff=float(rng.integers(-4, 5)),
                home_margin=float(m), total_points=tp,
                ref_ml_home_prob=0.40, ref_cover_prob=0.60, ref_over_prob=0.55,
            )
        )
    return rows


def _make(spec, seed=7):
    """spec: list of (season, n, ml_mode, ats_mode, tot_mode)."""
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for season, n, ml, ats, tot in spec:
        rows += _season_rows(rng, season, n, ml, ats, tot, gid)
        gid += n
    frame = pd.DataFrame(rows)
    frame["home_win"] = _edge_to_nullable_binary(frame["home_margin"])
    frame["home_cover"] = _edge_to_nullable_binary(frame["home_margin"] + frame["home_spread"])
    frame["over"] = _edge_to_nullable_binary(frame["total_points"] - frame["total_line"])
    return frame


@pytest.fixture
def fit_spies(monkeypatch):
    """Count fit() calls on the GBM, isotonic and logistic estimators."""
    import nfl_hybrid.selection.epa_tournament as ep

    counts = {"gbm": 0, "iso": 0, "logit": 0}
    for key, cls in (
        ("gbm", ep.HistGradientBoostingClassifier),
        ("iso", ep.IsotonicRegression),
        ("logit", ep.LogisticRegression),
    ):
        orig = cls.fit

        def make(orig, key):
            def fit(self, *a, **k):
                counts[key] += 1
                return orig(self, *a, **k)
            return fit

        monkeypatch.setattr(cls, "fit", make(orig, key))
    return counts


def _assert_full_valid_probs(out, test):
    assert len(out) == len(test)
    assert out.index.equals(test.index)
    for col in PROB_COLS:
        p = out[col].to_numpy(float)
        assert p.shape[0] == len(test)          # never empty / training-length / scalar
        assert np.isfinite(p).all()
        assert ((p >= 0.0) & (p <= 1.0)).all()


def _assert_is_baseline(out, test):
    for col in PROB_COLS:
        assert np.allclose(out[col].to_numpy(float), _BASE_BY_PROB[col])


# --- Part 11: GBM sparse-target cases ---------------------------------------- #
@pytest.mark.parametrize("mode", ["none", "one0", "one1"])
def test_gbm_zero_or_one_class_fit_uses_market_baseline(mode, fit_spies):
    """Cases A/B: zero valid fit rows or one fit class -> no GBM fit, market baseline."""
    m = _make([(2020, 60, mode, mode, mode), (2021, 12, "two", "two", "two")])
    train, test = m[m["season"] == 2020], m[m["season"] == 2021]
    out = candidate_gbm(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)
    _assert_is_baseline(out, test)
    assert fit_spies["gbm"] == 0            # classifier never fit
    assert fit_spies["iso"] == 0            # isotonic never fit


def test_gbm_two_class_fit_zero_valid_calibration_is_uncalibrated(fit_spies):
    """Case C: both fit classes but zero valid calibration rows -> GBM, no isotonic."""
    m = _make([(2019, 60, "two", "two", "two"), (2020, 35, "none", "none", "none"),
               (2021, 12, "two", "two", "two")])
    train = m[m["season"] < 2021]
    test = m[m["season"] == 2021]
    out = candidate_gbm(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)
    assert fit_spies["gbm"] == 3            # one GBM fit per market
    assert fit_spies["iso"] == 0            # isotonic skipped (empty calibration)


def test_gbm_two_class_fit_one_class_calibration_is_uncalibrated(fit_spies):
    """Case D: both fit classes but one-class calibration -> GBM, no isotonic."""
    m = _make([(2019, 60, "two", "two", "two"), (2020, 35, "one0", "one0", "one0"),
               (2021, 12, "two", "two", "two")])
    train = m[m["season"] < 2021]
    test = m[m["season"] == 2021]
    out = candidate_gbm(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)
    assert fit_spies["gbm"] == 3
    assert fit_spies["iso"] == 0


def test_gbm_fully_valid_fit_and_calibration_is_calibrated(fit_spies):
    """Case E: both fit and calibration classes -> GBM + isotonic per market."""
    m = _make([(2019, 60, "two", "two", "two"), (2020, 35, "two", "two", "two"),
               (2021, 12, "two", "two", "two")])
    train = m[m["season"] < 2021]
    test = m[m["season"] == 2021]
    out = candidate_gbm(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)
    assert fit_spies["gbm"] == 3
    assert fit_spies["iso"] == 3


def test_gbm_push_changes_only_the_affected_market_sample(fit_spies):
    """Case 7: an ATS push only removes rows from the ATS fitting sample; moneyline
    and totals keep every row, and every test row still gets a probability."""
    m = _make([(2020, 60, "two", "two", "two"), (2021, 12, "two", "two", "two")])
    train, test = m[m["season"] == 2020], m[m["season"] == 2021]
    # inject three exact ATS pushes into training rows (margin + spread == 0) without
    # disturbing moneyline (margin != 0) or totals.
    idx = train.index[:3]
    train = train.copy()
    train.loc[idx, "home_spread"] = -train.loc[idx, "home_margin"].to_numpy(float)
    train["home_cover"] = _edge_to_nullable_binary(train["home_margin"] + train["home_spread"])
    ml_valid = train["home_win"].isin([0, 1]).sum()
    ats_valid = train["home_cover"].isin([0, 1]).sum()
    over_valid = train["over"].isin([0, 1]).sum()
    assert ml_valid == len(train)                 # moneyline untouched
    assert over_valid == len(train)               # totals untouched
    assert ats_valid == len(train) - 3            # exactly the pushes removed
    out = candidate_gbm(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)


# --- Part 12: logistic sparse-target cases ----------------------------------- #
@pytest.mark.parametrize("mode", ["none", "one0", "one1"])
def test_logistic_zero_or_one_class_uses_market_baseline(mode, fit_spies):
    m = _make([(2020, 60, mode, mode, mode), (2021, 12, "two", "two", "two")])
    train, test = m[m["season"] == 2020], m[m["season"] == 2021]
    out = candidate_logistic(train, test, FEATURES, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)
    _assert_is_baseline(out, test)
    assert fit_spies["logit"] == 0             # logistic regression never fit


def test_logistic_two_class_fits_and_scores_all_rows(fit_spies):
    m = _make([(2020, 60, "two", "two", "two"), (2021, 12, "two", "two", "two")])
    train, test = m[m["season"] == 2020], m[m["season"] == 2021]
    out = candidate_logistic(train, test, FEATURES, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)
    assert fit_spies["logit"] == 3             # one fit per market, existing behaviour


def test_logistic_push_masks_are_independent_per_market(fit_spies):
    """A totals push only drops rows from the totals fit; moneyline & ATS keep all
    rows, and the logistic candidate still scores every test row."""
    m = _make([(2020, 60, "two", "two", "two"), (2021, 12, "two", "two", "two")])
    train, test = m[m["season"] == 2020], m[m["season"] == 2021]
    idx = train.index[:2]
    train = train.copy()
    train.loc[idx, "total_line"] = train.loc[idx, "total_points"].to_numpy(float)  # exact push
    train["over"] = _edge_to_nullable_binary(train["total_points"] - train["total_line"])
    assert train["home_win"].isin([0, 1]).sum() == len(train)
    assert train["home_cover"].isin([0, 1]).sum() == len(train)
    assert train["over"].isin([0, 1]).sum() == len(train) - 2
    out = candidate_logistic(train, test, FEATURES, market_prob_col_map=BASELINES)
    _assert_full_valid_probs(out, test)


# --- Part 13: stacked sparse-target cases ------------------------------------ #
@pytest.mark.parametrize("meta_mode", ["none", "one1"])
def test_stacked_zero_or_one_class_meta_uses_test_baseline(meta_mode):
    """Zero valid meta rows or a one-class meta target -> validated TEST-row market
    baseline (never the previous [:0] empty / training-length array)."""
    m = _make([(2019, 60, "two", "two", "two"),
               (2020, 35, meta_mode, "two", "two"),   # holdout: moneyline meta degenerate
               (2021, 12, "two", "two", "two")])
    train = m[m["season"] < 2021]
    test = m[m["season"] == 2021]
    out = candidate_stacked(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    assert out is not None
    _assert_full_valid_probs(out, test)
    # the degenerate moneyline meta target falls back to the test-row baseline...
    assert np.allclose(out["home_win_probability_no_tie"].to_numpy(float), _BASE_BY_PROB["home_win_probability_no_tie"])
    # ...while the two well-formed markets keep their fitted meta predictions.
    assert len(out["home_cover_probability_no_push"]) == len(test)


def test_stacked_two_class_meta_uses_meta_model():
    m = _make([(2019, 60, "two", "two", "two"), (2020, 35, "two", "two", "two"),
               (2021, 12, "two", "two", "two")])
    train = m[m["season"] < 2021]
    test = m[m["season"] == 2021]
    out = candidate_stacked(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    assert out is not None
    _assert_full_valid_probs(out, test)


def test_stacked_never_returns_empty_training_length_or_scalar():
    """Direct regression for the invalid `[:0]` branch: the degenerate meta output
    must have exactly len(test) rows and preserve test.index -- not 0, not
    len(meta_train), not a scalar."""
    meta_len = 35
    test_len = 12
    m = _make([(2019, 60, "two", "two", "two"),
               (2020, meta_len, "none", "two", "two"),
               (2021, test_len, "two", "two", "two")])
    train = m[m["season"] < 2021]
    test = m[m["season"] == 2021]
    out = candidate_stacked(train, test, FEATURES, calibration_season=2020, market_prob_col_map=BASELINES)
    col = out["home_win_probability_no_tie"]
    assert len(col) == test_len
    assert len(col) != 0                       # not the old [:0] empty array
    assert len(col) != meta_len                # not a training-length array
    assert np.ndim(col.to_numpy()) == 1        # not a scalar
    assert out.index.equals(test.index)        # test index preserved


# --- Part 14: cross-market target independence ------------------------------- #
def test_cross_market_target_independence():
    """One tie, one ATS push, one total push, plus resolved games: each market's
    valid mask excludes only its own null, and every completed row stays available
    for continuous margin/total regression."""
    def game(m, ats_edge, tot_edge):
        return dict(
            home_spread=ats_edge - m, total_line=44.0 - tot_edge,
            home_margin=float(m), total_points=44.0,
        )

    frame = pd.DataFrame([
        game(0.0, 3.0, 3.0),     # 0: moneyline TIE only (ATS +3, total +3 resolved)
        game(6.0, 0.0, 3.0),     # 1: ATS PUSH only (ML +6, total +3 resolved)
        game(6.0, 3.0, 0.0),     # 2: TOTAL PUSH only (ML +6, ATS +3 resolved)
        game(6.0, 3.0, 3.0),     # 3: all resolved
        game(-6.0, -3.0, -3.0),  # 4: all resolved
        game(7.0, 4.0, 5.0),     # 5: all resolved
        game(-8.0, -2.0, -4.0),  # 6: all resolved
    ])
    hw = _edge_to_nullable_binary(frame["home_margin"])
    hc = _edge_to_nullable_binary(frame["home_margin"] + frame["home_spread"])
    ov = _edge_to_nullable_binary(frame["total_points"] - frame["total_line"])
    ml_valid = hw.isin([0, 1]).to_numpy()
    ats_valid = hc.isin([0, 1]).to_numpy()
    tot_valid = ov.isin([0, 1]).to_numpy()

    # moneyline fitting excludes ONLY the tie (row 0)
    assert not ml_valid[0] and ml_valid[1:].all()
    # ATS fitting excludes ONLY the ATS push (row 1)
    assert not ats_valid[1] and ats_valid[0] and ats_valid[2:].all()
    # totals fitting excludes ONLY the total push (row 2)
    assert not tot_valid[2] and tot_valid[:2].all() and tot_valid[3:].all()
    # a null in one market never removes a resolved target in another market
    assert ml_valid[1] and tot_valid[1]        # ATS push row available to ML & totals
    assert ats_valid[0] and tot_valid[0]       # tie row available to ATS & totals
    assert ml_valid[2] and ats_valid[2]        # total-push row available to ML & ATS
    # every completed row keeps finite continuous outcomes for regression
    assert np.isfinite(frame["home_margin"].to_numpy()).all()
    assert np.isfinite(frame["total_points"].to_numpy()).all()
