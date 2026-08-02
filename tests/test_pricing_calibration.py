import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.pricing.devig import devig_proportional, devig_power, devig_shin, devig_pair
from nfl_hybrid.pricing.margin_surface import (
    MarginSurface,
    build_empirical_pmf_table,
    discretized_normal_pmf,
)


# ---- de-vig -------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [devig_proportional, devig_power, devig_shin])
def test_devig_sums_to_one(fn):
    for p1, p2 in [(0.55, 0.52), (0.70, 0.36), (0.90, 0.15), (0.50, 0.50)]:
        q1, q2 = fn(p1, p2)
        assert q1 + q2 == pytest.approx(1.0, abs=1e-6)
        assert 0 <= q1 <= 1 and 0 <= q2 <= 1


def test_devig_no_vig_is_identity_like():
    # already-fair pair sums to ~1 and stays ordered
    for fn in (devig_proportional, devig_power, devig_shin):
        q1, q2 = fn(0.6, 0.4)
        assert q1 > q2


def test_power_and_shin_correct_favourite_longshot_bias():
    # vig falls disproportionately on the longshot, so power/Shin give the
    # favourite a HIGHER fair prob than proportional (which over-shades favourites).
    p1, p2 = 0.80, 0.28  # overround 1.08, p1 is the favourite
    qp, _ = devig_proportional(p1, p2)
    qpow, _ = devig_power(p1, p2)
    qsh, _ = devig_shin(p1, p2)
    assert qpow >= qp - 1e-9
    assert qsh >= qp - 1e-9


# ---- margin surface ------------------------------------------------------ #
def _synthetic_games(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    spread = rng.choice([-7, -3, -2.5, 0, 3, 6.5], size=n)
    margin = np.rint(-spread + rng.normal(0, 13, n)).astype(int)
    total = rng.normal(45, 4, n).round()
    return pd.DataFrame({"home_margin": margin, "closing_home_spread": spread, "total_points": margin, "closing_total": total})


def test_discretized_normal_sums_to_one():
    _, p = discretized_normal_pmf(3.0, 13.5)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert (p >= 0).all()


def test_empirical_pmf_normalized_and_nonnegative():
    tbl = build_empirical_pmf_table(_synthetic_games(), bucket_width=2.0)
    sums = tbl.groupby("bucket_center")["probability"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)
    assert (tbl["probability"] >= 0).all()
    assert tbl["probability"].notna().all()


def test_cover_push_fail_identity():
    tbl = build_empirical_pmf_table(_synthetic_games(), bucket_width=2.0)
    surf = MarginSurface(method="empirical", pmf_table=tbl)
    for spread in (-7, -3, 0, 3):
        for line in (-7, -3, -2.5, 0, 3):
            h, pu, a = surf.cover_probabilities(spread, line)
            assert h + pu + a == pytest.approx(1.0, abs=1e-9)


def test_half_point_line_has_zero_push():
    tbl = build_empirical_pmf_table(_synthetic_games(), bucket_width=2.0)
    surf = MarginSurface(method="empirical", pmf_table=tbl)
    _, push, _ = surf.cover_probabilities(-3.0, -3.5)
    assert push == pytest.approx(0.0, abs=1e-12)


def test_whole_number_push_equals_margin_mass():
    tbl = build_empirical_pmf_table(_synthetic_games(), bucket_width=2.0)
    surf = MarginSurface(method="empirical", pmf_table=tbl)
    margins, probs = surf.pmf(-3.0)
    # push at line -3 => home_margin == 3
    exact = float(probs[margins == 3].sum())
    _, push, _ = surf.cover_probabilities(-3.0, -3.0)
    assert push == pytest.approx(exact, abs=1e-9)


def test_spread_ladder_monotonic_home_cover():
    tbl = build_empirical_pmf_table(_synthetic_games(), bucket_width=2.0)
    surf = MarginSurface(method="empirical", pmf_table=tbl)
    hc = [surf.cover_probabilities(-3.0, L)[0] for L in range(-14, 15)]
    assert np.all(np.diff(hc) >= -1e-12)  # more points for home -> higher cover


def test_opposite_sides_reconcile():
    tbl = build_empirical_pmf_table(_synthetic_games(), bucket_width=2.0)
    surf = MarginSurface(method="empirical", pmf_table=tbl)
    h, pu, a = surf.cover_probabilities(-3.0, -3.0)
    # away covering line +3 mirrors home failing to cover -3
    assert a == pytest.approx(1.0 - h - pu, abs=1e-9)


def test_normal_fallback_still_works():
    surf = MarginSurface(method="normal")
    h, pu, a = surf.cover_probabilities(-3.0, -3.0, sigma=13.0)
    assert h + pu + a == pytest.approx(1.0, abs=1e-9)
    assert pu > 0  # integer line has a push under normal too


# ---- R1: moneyline grading in select_margin_surface excludes ties ---------- #
# A tied game (home_margin == 0) must be pd.NA in the canonical label and dropped
# from BOTH the target and the probability vector by the same mask -- never graded
# as a class-zero away win. Missing and non-finite margins are excluded too.
def _margins_with_special_2022(seed=0):
    """Clean 2019-2021 training history plus a controlled 2022 test season with a
    tie, +inf, -inf and a missing margin alongside known class-1/class-0 games."""
    rng = np.random.default_rng(seed)
    spreads = np.array([-7.0, -6.0, -3.0, -2.5, 0.0, 3.0, 6.5])
    rows = []
    for season in (2019, 2020, 2021):
        for _ in range(150):
            s = float(rng.choice(spreads))
            m = int(np.rint(-s + rng.normal(0, 13)))
            if m == 0:
                m = 1  # keep training history free of incidental ties
            rows.append({"season": season, "closing_home_spread": s, "home_margin": float(m)})
    special = [
        (-3.0, 7.0),      # finite positive margin -> class 1
        (-3.0, 3.0),      # finite positive margin -> class 1
        (-6.0, 10.0),     # finite positive margin -> class 1
        (3.0, -7.0),      # finite negative margin -> class 0
        (2.0, -4.0),      # finite negative margin -> class 0
        (0.0, 0.0),       # tie -> pd.NA (excluded)
        (-3.0, np.inf),   # +inf -> pd.NA (excluded)
        (3.0, -np.inf),   # -inf -> pd.NA (excluded)
        (0.0, np.nan),    # missing -> pd.NA (excluded)
    ]
    for s, m in special:
        rows.append({"season": 2022, "closing_home_spread": s, "home_margin": m})
    return pd.DataFrame(rows)


def test_select_margin_surface_excludes_ties_and_nonfinite(monkeypatch):
    import nfl_hybrid.pricing.calibration as cal

    # isolate grading to a single clean-history season; keep the bootstrap cheap.
    monkeypatch.setattr(cal, "TEST_SEASONS", [2022])
    monkeypatch.setattr(cal, "BOOT_N", 50)

    ll_calls, brier_calls = [], []
    orig_ll, orig_brier = cal._pointwise_logloss, cal.brier_score_loss

    def spy_ll(y, p):
        y, p = np.asarray(y), np.asarray(p)
        ll_calls.append((y.copy(), p.copy()))
        return orig_ll(y, p)

    def spy_brier(y, p, *a, **k):
        y, p = np.asarray(y), np.asarray(p)
        brier_calls.append((y.copy(), p.copy()))
        return orig_brier(y, p, *a, **k)

    monkeypatch.setattr(cal, "_pointwise_logloss", spy_ll)
    monkeypatch.setattr(cal, "brier_score_loss", spy_brier)

    res = cal.select_margin_surface(_margins_with_special_2022(), buckets=(1.0, 2.0))

    N_VALID, N_POS = 5, 3   # 3 finite positive + 2 finite negative; 4 excluded
    # normal (1 call) + empirical buckets 1.0 & 2.0 (2 calls) = >=3 graded calls,
    # covering BOTH the normal-surface and empirical-surface evaluation paths.
    assert len(ll_calls) >= 3
    assert len(brier_calls) >= 3
    for y, p in ll_calls + brier_calls:
        assert y.shape == p.shape              # same mask on target and probabilities
        assert y.shape[0] == N_VALID           # exactly the tie/inf/-inf/missing excluded
        assert set(np.unique(y)).issubset({0, 1})   # no NA leaked; no tie graded as 0
        assert int(y.sum()) == N_POS           # positives -> 1, negatives -> 0
        assert np.isfinite(p).all()

    # reported sample size excludes exactly the four invalid games on every surface.
    assert res["candidates"]["normal_baseline"]["n"] == N_VALID
    assert res["candidates"]["empirical_bw1.0"]["n"] == N_VALID
    assert res["candidates"]["empirical_bw2.0"]["n"] == N_VALID


# ---- R1 zero-valid grading protection (Phase 8) --------------------------- #
# select_margin_surface must (a) skip a test season with zero resolved no-tie
# outcomes without ever handing an empty array to a metric, (b) raise a specific
# ValueError when EVERY fold is empty -- before any concatenate/metric/bootstrap,
# and (c) grade the normal and every empirical surface on one shared valid mask.
import nfl_hybrid.pricing.calibration as _cal  # noqa: E402


def _clean_training(rng, spreads, seasons=(2019, 2020, 2021), per=150):
    rows = []
    for season in seasons:
        for _ in range(per):
            s = float(rng.choice(spreads))
            m = int(np.rint(-s + rng.normal(0, 13)))
            if m == 0:
                m = 1  # keep training history free of incidental ties
            rows.append({"season": season, "closing_home_spread": s, "home_margin": float(m)})
    return rows


_SPREADS = np.array([-7.0, -6.0, -3.0, -2.5, 0.0, 3.0, 6.5])


def _margins_valid_then_empty(seed=0):
    """2022 = a valid fold (7 resolved); 2023 = a fully empty fold (ties/inf/missing)."""
    rng = np.random.default_rng(seed)
    rows = _clean_training(rng, _SPREADS)
    valid_2022 = [
        (-3.0, 7.0), (-3.0, 3.0), (-6.0, 10.0), (-2.5, 5.0),   # +margin -> class 1
        (3.0, -7.0), (2.0, -4.0), (6.5, -9.0),                 # -margin -> class 0
        (0.0, 0.0), (-3.0, np.inf),                            # tie / +inf -> excluded
    ]
    for s, m in valid_2022:
        rows.append({"season": 2022, "closing_home_spread": s, "home_margin": m})
    empty_2023 = [(0.0, 0.0), (-3.0, np.nan), (-3.0, np.inf), (3.0, -np.inf), (0.0, 0.0)]
    for s, m in empty_2023:
        rows.append({"season": 2023, "closing_home_spread": s, "home_margin": m})
    return pd.DataFrame(rows)


def test_zero_valid_one_empty_fold_is_skipped(monkeypatch):
    monkeypatch.setattr(_cal, "TEST_SEASONS", [2022, 2023])
    monkeypatch.setattr(_cal, "BOOT_N", 50)

    ll_calls = []
    orig_ll = _cal._pointwise_logloss

    def spy_ll(y, p):
        y, p = np.asarray(y), np.asarray(p)
        ll_calls.append((y.copy(), p.copy()))
        return orig_ll(y, p)

    monkeypatch.setattr(_cal, "_pointwise_logloss", spy_ll)

    res = _cal.select_margin_surface(_margins_valid_then_empty(), buckets=(1.0, 2.0))

    N_VALID = 7  # only 2022's resolved games; 2023 (empty) contributes nothing
    assert len(ll_calls) >= 3               # normal + 2 empirical buckets (1 valid fold each)
    for y, p in ll_calls:
        assert y.shape == p.shape           # same mask on target and probabilities
        assert y.shape[0] == N_VALID        # empty 2023 fold never reaches a metric
        assert y.shape[0] > 0               # no empty array handed to a metric
        assert set(np.unique(y)).issubset({0, 1})
        assert np.isfinite(p).all()
    # normal and every empirical path graded the IDENTICAL target values, in order
    targets = [y for y, _ in ll_calls]
    for y in targets[1:]:
        assert np.array_equal(y, targets[0])
    # reported n counts only the one valid fold's resolved observations
    assert res["candidates"]["normal_baseline"]["n"] == N_VALID
    assert res["candidates"]["empirical_bw1.0"]["n"] == N_VALID
    assert res["candidates"]["empirical_bw2.0"]["n"] == N_VALID


def _margins_all_empty():
    rows = []
    for season in (2022, 2023):
        for s, m in [(0.0, 0.0), (-3.0, np.nan), (-3.0, np.inf), (3.0, -np.inf), (0.0, 0.0)]:
            rows.append({"season": season, "closing_home_spread": s, "home_margin": m})
    return pd.DataFrame(rows)


def test_zero_valid_all_folds_empty_raises_before_any_metric(monkeypatch):
    monkeypatch.setattr(_cal, "TEST_SEASONS", [2022, 2023])
    monkeypatch.setattr(_cal, "BOOT_N", 50)

    def boom(name):
        def _f(*a, **k):
            raise AssertionError(f"{name} must not be called when all folds are empty")
        return _f

    # any metric / concatenate-dependent / bootstrap call proves the guard is too late
    monkeypatch.setattr(_cal, "_pointwise_logloss", boom("_pointwise_logloss"))
    monkeypatch.setattr(_cal, "brier_score_loss", boom("brier_score_loss"))
    monkeypatch.setattr(_cal, "_equal_mass_ece", boom("_equal_mass_ece"))
    monkeypatch.setattr(_cal, "_paired_bootstrap_ci", boom("_paired_bootstrap_ci"))

    with pytest.raises(
        ValueError,
        match="select_margin_surface: no valid non-tied moneyline outcomes remain across the configured test seasons",
    ):
        _cal.select_margin_surface(_margins_all_empty(), buckets=(1.0, 2.0))


def _margins_special_single_season(seed=1):
    """2022 test season carrying pos, neg, zero, NaN, +inf and -inf margins."""
    rng = np.random.default_rng(seed)
    rows = _clean_training(rng, _SPREADS)
    special = [
        (-3.0, 7.0),      # positive finite -> class 1
        (3.0, -5.0),      # negative finite -> class 0
        (0.0, 0.0),       # tie -> excluded
        (0.0, np.nan),    # missing -> excluded
        (-3.0, np.inf),   # +inf -> excluded
        (3.0, -np.inf),   # -inf -> excluded
    ]
    for s, m in special:
        rows.append({"season": 2022, "closing_home_spread": s, "home_margin": m})
    return pd.DataFrame(rows)


def test_zero_valid_shared_normal_empirical_mask(monkeypatch):
    monkeypatch.setattr(_cal, "TEST_SEASONS", [2022])
    monkeypatch.setattr(_cal, "BOOT_N", 50)

    ll_calls = []
    orig_ll = _cal._pointwise_logloss

    def spy_ll(y, p):
        y, p = np.asarray(y), np.asarray(p)
        ll_calls.append((y.copy(), p.copy()))
        return orig_ll(y, p)

    monkeypatch.setattr(_cal, "_pointwise_logloss", spy_ll)

    _cal.select_margin_surface(_margins_special_single_season(), buckets=(1.0, 2.0))

    assert len(ll_calls) >= 3  # normal + 2 empirical surfaces, all on the same fold
    y0 = ll_calls[0][0]
    assert np.array_equal(y0, np.array([1, 0]))          # only the two resolved rows, in order
    for y, p in ll_calls:
        assert np.array_equal(y, y0)                     # identical y for every surface
        assert y.shape == p.shape                        # identical exclusion + equal length
        assert set(np.unique(y)).issubset({0, 1})        # tie never graded as class 0
        assert np.isfinite(p).all()                      # non-finite outcomes never reach metrics


def _margins_known_mixture(seed=2):
    rng = np.random.default_rng(seed)
    rows = _clean_training(rng, _SPREADS)
    positives = [(-3.0, 7.0), (-3.0, 3.0), (-6.0, 10.0), (-2.5, 5.0), (-3.0, 2.0)]  # 5 -> 1
    negatives = [(3.0, -7.0), (2.0, -4.0), (6.5, -9.0)]                              # 3 -> 0
    excluded = [
        (0.0, 0.0),        # zero
        (0.0, 0.5e-9),     # inside +tol
        (0.0, -0.5e-9),    # inside -tol
        (0.0, np.nan),     # missing
        (-3.0, np.inf),    # +inf
        (3.0, -np.inf),    # -inf
    ]
    for s, m in positives + negatives + excluded:
        rows.append({"season": 2022, "closing_home_spread": s, "home_margin": m})
    return pd.DataFrame(rows)


def test_zero_valid_exact_graded_count(monkeypatch):
    monkeypatch.setattr(_cal, "TEST_SEASONS", [2022])
    monkeypatch.setattr(_cal, "BOOT_N", 50)

    res = _cal.select_margin_surface(_margins_known_mixture(), buckets=(1.0, 2.0))

    # n == count(finite margin > 1e-9) + count(finite margin < -1e-9); nothing else.
    N_EXPECTED = 5 + 3
    assert res["candidates"]["normal_baseline"]["n"] == N_EXPECTED
    assert res["candidates"]["empirical_bw1.0"]["n"] == N_EXPECTED
    assert res["candidates"]["empirical_bw2.0"]["n"] == N_EXPECTED
