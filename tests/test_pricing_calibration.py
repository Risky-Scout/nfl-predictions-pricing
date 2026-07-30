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
