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


# =============================================================================
# R2: contract-strict walk-forward. Every scored test row carries a complete
# exact REAL-CLOSING contract; the contract metadata is propagated untouched
# through prediction; proxy/mismatched/duplicate rows fail closed.
# =============================================================================
import nfl_hybrid.selection.epa_tournament as ep
from nfl_hybrid.markets.exact_contract import (
    AGGREGATION_METHOD,
    MARKET_SOURCE_CLOSING,
    MARKET_SOURCE_PROXY,
    ContractError,
    make_market_contract_id,
    validate_prediction_contract,
)

CFEATURES = ["home_spread", "total_line", "epa_net_edge_season_mean", "rest_diff"]
TOUR_PROB_COLS = [
    "tournament_market_ml_home_probability",
    "tournament_market_cover_home_probability",
    "tournament_market_over_probability",
]


def _ccid(gid, market, side, line, snap=180.0):
    return make_market_contract_id(
        game_id=gid, market_type=market, outcome_side=side, line_value=line,
        market_source=MARKET_SOURCE_CLOSING, snapshot_minutes_to_kickoff=snap,
        aggregation_method=AGGREGATION_METHOD,
    )


def _cbench_row(gid, season, hs, tl, snap=180.0):
    return dict(
        game_id=gid, season=season, week=1, home_team_id="H", away_team_id="A",
        market_ml_home_probability=0.55, market_cover_home_probability=0.5,
        market_over_probability=0.5, closing_home_spread=hs, closing_total_line=tl,
        closing_minutes_to_kickoff=snap, market_source=MARKET_SOURCE_CLOSING,
        aggregation_method=AGGREGATION_METHOD,
        moneyline_contract_id=_ccid(gid, "moneyline", "home", None, snap),
        spread_contract_id=_ccid(gid, "spread", "home", hs, snap),
        total_contract_id=_ccid(gid, "total", "over", tl, snap),
        moneyline_consensus_books=5, spread_consensus_books=5, total_consensus_books=5,
        spread_candidate_point_count=1, total_candidate_point_count=1,
    )


def _crow(gid, season, hs, tl, margin, total, source, rng, snap=180.0):
    real = source == MARKET_SOURCE_CLOSING
    return dict(
        game_id=gid, season=season, week=int(rng.integers(1, 18)),
        home_spread=hs, total_line=tl, home_margin=margin, total_points=total,
        epa_net_edge_season_mean=float(rng.normal(0, 0.1)), rest_diff=float(rng.integers(-4, 5)),
        home_win=_edge_to_nullable_binary(pd.Series([margin])).iloc[0],
        home_cover=_edge_to_nullable_binary(pd.Series([margin + hs])).iloc[0],
        over=_edge_to_nullable_binary(pd.Series([total - tl])).iloc[0],
        market_contract_source=source,
        moneyline_contract_id=_ccid(gid, "moneyline", "home", None, snap) if real else "proxy",
        spread_contract_id=_ccid(gid, "spread", "home", hs, snap) if real else "proxy",
        total_contract_id=_ccid(gid, "total", "over", tl, snap) if real else "proxy",
        tournament_market_ml_home_probability=0.55 if real else 0.5,
        tournament_market_cover_home_probability=0.5,
        tournament_market_over_probability=0.5,
        closing_home_spread=hs if real else np.nan,
        closing_total_line=tl if real else np.nan,
        closing_minutes_to_kickoff=snap if real else np.nan,
    )


def _contract_scenario(n_real=30, seed=3):
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for season in (2020, 2021):
        for _ in range(70):
            gid += 1
            hs = float(np.round(rng.normal(0, 6))); tl = float(np.round(rng.normal(45, 5)))
            rows.append(_crow(f"p{gid}", season, hs, tl, -hs + rng.normal(0, 13), tl + rng.normal(0, 13), MARKET_SOURCE_PROXY, rng))
    bench = []
    for _ in range(n_real):
        gid += 1; g = f"r{gid}"
        hs = float(np.round(rng.normal(0, 6))); tl = float(np.round(rng.normal(45, 5)))
        rows.append(_crow(g, 2022, hs, tl, -hs + rng.normal(0, 13), tl + rng.normal(0, 13), MARKET_SOURCE_CLOSING, rng))
        bench.append(_cbench_row(g, 2022, hs, tl))
    return pd.DataFrame(rows), pd.DataFrame(bench)


def _run(matrix):
    return run_walk_forward(
        matrix, CFEATURES, test_seasons=[2022], market_prob_cols=TOUR_PROB_COLS,
        expected_test_contract_source=MARKET_SOURCE_CLOSING,
    )


# --- Test A: all candidates receive exact point columns ---------------------- #
def test_all_candidates_receive_exact_point_columns(monkeypatch):
    matrix, _ = _contract_scenario()
    seen = {}
    required = set(TOUR_PROB_COLS) | {
        "closing_home_spread", "closing_total_line", "spread_contract_id",
        "total_contract_id", "moneyline_contract_id",
    }
    for fn_name in ("candidate_market_residual", "candidate_gbm", "candidate_logistic",
                    "candidate_jointscore_epa", "candidate_stacked"):
        orig = getattr(ep, fn_name)

        def make(orig, name):
            def wrapper(train, test, *a, **k):
                seen[name] = set(test.columns)
                return orig(train, test, *a, **k)
            return wrapper

        monkeypatch.setattr(ep, fn_name, make(orig, fn_name))
    _run(matrix)
    assert seen  # candidates were invoked
    for name, cols in seen.items():
        assert required.issubset(cols), f"{name} missing {required - cols}"


# --- Test B: output contract propagation ------------------------------------- #
def test_output_contract_propagation():
    matrix, bench = _contract_scenario()
    results = _run(matrix)
    test_ids = list(matrix[matrix["season"] == 2022]["game_id"])
    for name, df in results.items():
        assert len(df) == len(test_ids)
        assert list(df["game_id"]) == test_ids            # same games, same order
        for col in ("moneyline_contract_id", "spread_contract_id", "total_contract_id",
                    "closing_home_spread", "closing_total_line",
                    "tournament_market_cover_home_probability"):
            assert col in df.columns
        # points equal the bound closing points
        assert np.allclose(df["home_spread"].to_numpy(float), df["closing_home_spread"].to_numpy(float))
        assert np.allclose(df["total_line"].to_numpy(float), df["closing_total_line"].to_numpy(float))


# --- Test C: line mismatch fails --------------------------------------------- #
def test_line_mismatch_fails_naming_game_and_market():
    matrix, bench = _contract_scenario()
    pred = _run(matrix)["C2_gbm"].copy()
    gid = pred.loc[0, "game_id"]
    pred.loc[0, "home_spread"] = pred.loc[0, "home_spread"] + 1.0  # mutate a point post-prediction
    with pytest.raises(ContractError) as exc:
        validate_prediction_contract(pred, bench)
    assert str(gid) in str(exc.value) and "spread" in str(exc.value)


# --- Test D: contract ID mismatch fails (point unchanged) -------------------- #
def test_contract_id_mismatch_fails():
    matrix, bench = _contract_scenario()
    pred = _run(matrix)["C2_gbm"].copy()
    pred.loc[0, "spread_contract_id"] = "tampered-but-same-point"
    with pytest.raises(ContractError):
        validate_prediction_contract(pred, bench)


# --- Test E: market probability / benchmark alignment (not positional) ------- #
def test_benchmark_alignment_is_by_game_id_not_positional():
    matrix, bench = _contract_scenario()
    pred = _run(matrix)["C2_gbm"].copy()
    shuffled = bench.sample(frac=1.0, random_state=9).reset_index(drop=True)
    # reordering benchmark rows must not break validation: it realigns by game_id
    validate_prediction_contract(pred, shuffled)


# --- Test F: duplicate game id fails ----------------------------------------- #
def test_duplicate_game_id_fails_in_validator_and_walk_forward():
    matrix, bench = _contract_scenario()
    pred = _run(matrix)["C2_gbm"].copy()
    dup_bench = pd.concat([bench, bench.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError):
        validate_prediction_contract(pred, dup_bench)
    dup_matrix = pd.concat([matrix, matrix[matrix["season"] == 2022].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        _run(dup_matrix)


# --- Test G: source enforcement ---------------------------------------------- #
def test_proxy_source_test_row_fails_before_prediction():
    matrix, bench = _contract_scenario()
    matrix = matrix.copy()
    idx = matrix.index[matrix["season"] == 2022][0]
    matrix.loc[idx, "market_contract_source"] = MARKET_SOURCE_PROXY  # illegal scored source
    with pytest.raises(ValueError):
        _run(matrix)


# =============================================================================
# R2 batch: two silent-failure guards in run_walk_forward.
#   Issue 1 - a requested historical test season must never be SILENTLY skipped.
#   Issue 2 - a registered candidate must never be SILENTLY omitted.
# Strict (contract) mode fails closed; legacy/synthetic mode keeps leniency.
# =============================================================================
def test_strict_walk_forward_raises_on_empty_test_season():
    matrix, _ = _contract_scenario()
    with pytest.raises(ValueError, match="2099"):
        run_walk_forward(
            matrix, CFEATURES, test_seasons=[2099, 2022], market_prob_cols=TOUR_PROB_COLS,
            expected_test_contract_source=MARKET_SOURCE_CLOSING,
            require_complete=True,
        )


def test_strict_walk_forward_raises_on_insufficient_training():
    matrix, _ = _contract_scenario()
    # test season 2021 has rows but only 2020 (<100) precedes it as training history
    with pytest.raises(ValueError, match="training rows"):
        run_walk_forward(
            matrix, CFEATURES, test_seasons=[2021], market_prob_cols=TOUR_PROB_COLS,
            expected_test_contract_source=MARKET_SOURCE_CLOSING,
            require_complete=True,
        )


def test_legacy_walk_forward_still_skips_unscoreable_season():
    m = _matrix(seasons=(2020, 2021, 2022))
    res = run_walk_forward(
        m, FEATURES, test_seasons=[2019, 2022],
        market_prob_cols=["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"],
    )
    # 2019 is silently skipped (legacy leniency preserved), 2022 scores every candidate
    assert set(res.keys()) == set(CANDIDATES)


def test_candidate_registry_mismatch_raises(monkeypatch):
    matrix, _ = _contract_scenario()
    monkeypatch.setattr(ep, "CANDIDATES", ep.CANDIDATES + ("C6_ghost",))
    with pytest.raises(RuntimeError, match="candidate set mismatch"):
        run_walk_forward(
            matrix, CFEATURES, test_seasons=[2022], market_prob_cols=TOUR_PROB_COLS,
            expected_test_contract_source=MARKET_SOURCE_CLOSING,
            require_complete=True,
        )


def test_strict_walk_forward_raises_on_none_candidate(monkeypatch):
    matrix, _ = _contract_scenario()
    monkeypatch.setattr(ep, "candidate_stacked", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="C5_stacked"):
        run_walk_forward(
            matrix, CFEATURES, test_seasons=[2022], market_prob_cols=TOUR_PROB_COLS,
            expected_test_contract_source=MARKET_SOURCE_CLOSING,
            require_complete=True,
        )


def test_legacy_walk_forward_skips_none_candidate(monkeypatch):
    m = _matrix(seasons=(2020, 2021, 2022))
    monkeypatch.setattr(ep, "candidate_stacked", lambda *a, **k: None)
    res = run_walk_forward(
        m, FEATURES, test_seasons=[2022],
        market_prob_cols=["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"],
    )
    # legacy mode still tolerates a None candidate (no fail-closed)
    assert "C5_stacked" not in res and "C1_market_residual" in res


# =============================================================================
# R2 batch 1 correction: require_complete is an EXPLICIT completeness contract,
# INDEPENDENT of expected_test_contract_source. Every test below monkeypatches
# all candidates to fixed-value stubs so NO real estimator is ever fit.
# =============================================================================
_STUB_MARKET_COLS = ["ref_ml_home_prob", "ref_cover_prob", "ref_over_prob"]


def stub_candidate(*args, **kwargs):
    """Lightweight candidate: one fixed-value probability row per test row,
    indexed exactly as test.index. No model fitting."""
    test = args[1]
    return pd.DataFrame(
        {
            PROB_COLS[0]: 0.51,
            PROB_COLS[1]: 0.52,
            PROB_COLS[2]: 0.53,
        },
        index=test.index,
    )


def _patch_all_candidates(monkeypatch, *, stacked=stub_candidate):
    """Monkeypatch every candidate function in the module to a lightweight stub.
    ``stacked`` overrides only candidate_stacked (e.g. to return None)."""
    monkeypatch.setattr(ep, "candidate_market_residual", stub_candidate)
    monkeypatch.setattr(ep, "candidate_gbm", stub_candidate)
    monkeypatch.setattr(ep, "candidate_logistic", stub_candidate)
    monkeypatch.setattr(ep, "candidate_jointscore_epa", stub_candidate)
    monkeypatch.setattr(ep, "candidate_stacked", stacked)


def test_require_complete_is_independent_of_contract_source(monkeypatch):
    """Contract validation does NOT imply completeness: the same input with a
    contract source but require_complete=False keeps the lenient skip, while
    require_complete=True fails closed on the empty requested season."""
    _patch_all_candidates(monkeypatch)
    m = _matrix(seasons=(2020, 2021, 2022))

    # require_complete=False: the empty requested season 2099 is silently skipped
    # even though a contract source is supplied -> no completeness exception.
    res = run_walk_forward(
        m, FEATURES, test_seasons=[2099], market_prob_cols=_STUB_MARKET_COLS,
        expected_test_contract_source="REAL-CLOSING", require_complete=False,
    )
    assert res == {}  # nothing scored, no raise

    # require_complete=True: the same empty requested season is now a hard failure.
    with pytest.raises(ValueError) as exc:
        run_walk_forward(
            m, FEATURES, test_seasons=[2099], market_prob_cols=_STUB_MARKET_COLS,
            expected_test_contract_source="REAL-CLOSING", require_complete=True,
        )
    msg = str(exc.value)
    assert "run_walk_forward" in msg and "2099" in msg and "zero eligible rows" in msg


def test_require_complete_rejects_zero_test_rows(monkeypatch):
    _patch_all_candidates(monkeypatch)
    m = _matrix(seasons=(2020, 2021, 2022))
    with pytest.raises(ValueError) as exc:
        run_walk_forward(
            m, FEATURES, test_seasons=[2099], market_prob_cols=_STUB_MARKET_COLS,
            require_complete=True,
        )
    msg = str(exc.value)
    assert "zero eligible rows" in msg and "2099" in msg


def test_require_complete_rejects_insufficient_training(monkeypatch):
    # Fewer than 100 prior training rows (2020 = 50) for a nonempty test season 2021.
    _patch_all_candidates(monkeypatch)
    m = _matrix(seasons=(2020, 2021), n_per_season=50)
    with pytest.raises(ValueError) as exc:
        run_walk_forward(
            m, FEATURES, test_seasons=[2021], market_prob_cols=_STUB_MARKET_COLS,
            require_complete=True,
        )
    msg = str(exc.value)
    assert "insufficient historical training rows" in msg
    assert "train=50" in msg and "required=100" in msg and "2021" in msg


def test_require_complete_rejects_candidate_none(monkeypatch):
    # >=100 training rows and a nonempty test season; only C5_stacked returns None.
    _patch_all_candidates(monkeypatch, stacked=lambda *a, **k: None)
    m = _matrix(seasons=(2020, 2021, 2022))
    with pytest.raises(RuntimeError) as exc:
        run_walk_forward(
            m, FEATURES, test_seasons=[2022], market_prob_cols=_STUB_MARKET_COLS,
            require_complete=True,
        )
    msg = str(exc.value)
    assert "C5_stacked" in msg and "candidate returned None" in msg and "2022" in msg


def test_require_complete_tracks_every_candidate_and_season(monkeypatch):
    _patch_all_candidates(monkeypatch)
    requested = [2022, 2023]
    # >=100 rows before the first requested season (2020+2021 = 240) and >=1 row in
    # each requested test season.
    m = _matrix(seasons=(2020, 2021, 2022, 2023))
    result = run_walk_forward(
        m, FEATURES, test_seasons=requested, market_prob_cols=_STUB_MARKET_COLS,
        require_complete=True,
    )
    assert set(result) == set(CANDIDATES)
    for name in CANDIDATES:
        assert set(result[name]["season"]) == set(requested)
        assert not result[name].empty


# =============================================================================
# R2 Batch 2A: candidate GAME COVERAGE. Every candidate must reproduce the exact
# ordered (season, game_id) grid of the eligible test frame -- a missing, extra,
# duplicated, or reordered game fails closed BEFORE pooling. Every test below
# monkeypatches all five candidates to lightweight stubs so no estimator is fit.
# =============================================================================
def _patch_c3_malformed(monkeypatch, malformed):
    """Patch every candidate to the normal Batch 1 stub, except C3_logistic which
    uses ``malformed`` (a ``*args, **kwargs`` stub returning a bad frame). No real
    estimator runs in either branch."""
    monkeypatch.setattr(ep, "candidate_market_residual", stub_candidate)
    monkeypatch.setattr(ep, "candidate_gbm", stub_candidate)
    monkeypatch.setattr(ep, "candidate_logistic", malformed)
    monkeypatch.setattr(ep, "candidate_jointscore_epa", stub_candidate)
    monkeypatch.setattr(ep, "candidate_stacked", stub_candidate)


def test_ordered_game_keys_rejects_duplicate_season_game_key():
    frame = pd.DataFrame(
        {
            "season": [2022, 2022],
            "game_id": ["g1", "g1"],
        }
    )
    with pytest.raises(ValueError) as exc:
        ep._ordered_game_keys(frame, context="unit")
    assert "duplicate season/game keys" in str(exc.value)


def test_require_complete_rejects_candidate_missing_one_game(monkeypatch):
    def c3_missing(*args, **kwargs):
        test = args[1]
        normal = stub_candidate(*args, **kwargs)
        # drop the final game -> fewer rows than the eligible test frame
        return normal.iloc[:-1].copy()

    _patch_c3_malformed(monkeypatch, c3_missing)
    m = _matrix(seasons=(2020, 2021, 2022))   # train(2020+2021)=240 >= 100
    with pytest.raises((ValueError, RuntimeError)) as exc:
        run_walk_forward(
            m, FEATURES, test_seasons=[2022], market_prob_cols=_STUB_MARKET_COLS,
            require_complete=True,
        )
    msg = str(exc.value)
    # C3 is named and the failure is a row-count / index-alignment / coverage
    # mismatch (the existing length guard fails closed before pooling).
    assert "C3_logistic" in msg
    assert ("aligned row per test row" in msg) or ("coverage mismatch" in msg)


def test_require_complete_rejects_candidate_extra_game(monkeypatch):
    def c3_extra(*args, **kwargs):
        test = args[1]
        normal = stub_candidate(*args, **kwargs)
        extra_idx = int(test.index.max()) + 1000   # index not present in test.index
        extra = pd.DataFrame(
            {PROB_COLS[0]: [0.51], PROB_COLS[1]: [0.52], PROB_COLS[2]: [0.53]},
            index=[extra_idx],
        )
        return pd.concat([normal, extra])

    _patch_c3_malformed(monkeypatch, c3_extra)
    m = _matrix(seasons=(2020, 2021, 2022))
    with pytest.raises((ValueError, RuntimeError)) as exc:
        run_walk_forward(
            m, FEATURES, test_seasons=[2022], market_prob_cols=_STUB_MARKET_COLS,
            require_complete=True,
        )
    msg = str(exc.value)
    assert "C3_logistic" in msg
    assert ("aligned row per test row" in msg) or ("coverage mismatch" in msg)


def test_require_complete_rejects_candidate_game_order_mismatch(monkeypatch):
    def c3_reversed(*args, **kwargs):
        normal = stub_candidate(*args, **kwargs)
        # reversed rows keeping the reversed index labels (never reset)
        return normal.iloc[::-1]

    _patch_c3_malformed(monkeypatch, c3_reversed)
    m = _matrix(seasons=(2020, 2021, 2022))
    with pytest.raises((ValueError, RuntimeError)) as exc:
        run_walk_forward(
            m, FEATURES, test_seasons=[2022], market_prob_cols=_STUB_MARKET_COLS,
            require_complete=True,
        )
    msg = str(exc.value)
    assert "C3_logistic" in msg
    assert ("aligned row per test row" in msg) or ("order_mismatch" in msg)


def test_require_complete_accepts_exact_candidate_game_grid(monkeypatch):
    _patch_all_candidates(monkeypatch)
    requested = [2022, 2023]
    m = _matrix(seasons=(2020, 2021, 2022, 2023))
    # expected ordered keys: test rows concatenated in requested-season order
    expected_keys = []
    for season in requested:
        part = m[m["season"] == season]
        expected_keys += ep._ordered_game_keys(part, context=f"expected {season}")

    result = run_walk_forward(
        m, FEATURES, test_seasons=requested, market_prob_cols=_STUB_MARKET_COLS,
        require_complete=True,
    )
    assert set(result) == set(CANDIDATES)
    for name in CANDIDATES:
        actual = ep._ordered_game_keys(result[name], context=name)
        assert actual == expected_keys
    # every candidate reproduced the identical expected grid
    all_keys = [ep._ordered_game_keys(result[name], context=name) for name in CANDIDATES]
    assert all(k == all_keys[0] for k in all_keys)
