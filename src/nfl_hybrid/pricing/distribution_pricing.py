"""Fix 4 item 5 (+ certification-blocker repair): the one canonical
distributional-pricing definition.

Two independent pieces of code in this repository price a three-way
(home/push/away, or over/push/under) outcome from a continuous mean + sigma:

- :class:`nfl_hybrid.modern.joint_score.JointScoreModel.raw_probabilities`,
  which prices its own learned margin/total distribution directly with a
  hand-written continuity-corrected ``scipy.stats.norm.cdf`` formula -- THIS
  is the intended probability definition (the model JointScoreModel actually
  implements and the one Fix 4 must match exactly), and
- :class:`nfl_hybrid.pricing.margin_surface.MarginSurface`, the repository's
  existing interpretable margin-probability surface (moneyline / ATS
  push-cover / alternate spreads), which prices via an integer-margin
  discretized PMF instead.

These are NOT the same formula for an arbitrary real line. They coincide
exactly at integer lines (both continuity-correct at the same boundary) and
at half-point lines (both reduce to a plain, uncorrected ``norm.cdf`` there),
but diverge for other fractional lines (e.g. a quarter-point consensus line,
which a median-of-an-even-number-of-books consensus can genuinely produce --
see :func:`nfl_hybrid.odds_history.build_consensus`). Fix 4's real-data proof
must be correct for whatever real consensus line actually occurs, not only
for the common integer/half-point case, so this module implements
JointScoreModel's own closed-form formula DIRECTLY -- ``price_ats_distribution``
/ ``price_total_distribution`` no longer route through MarginSurface's PMF
machinery for the default (``method="normal"``) surface. This is not a third,
newly-invented formula: it is JointScoreModel's own formula, extracted
verbatim (same continuity-corrected boundary at an integer line; the same
uncorrected ``norm.cdf(edge/sd)`` at every other line -- including
quarter-points), so parity holds for ANY finite real line, not just the
integer/half-point subset -- see ``tests/test_distribution_pricing.py``.

An explicitly-passed non-"normal" ``surface`` (e.g. ``MarginSurface(method=
"empirical", ...)``, used elsewhere in the repository's pricing-calibration
selection path, never by Fix 4) still routes through
:class:`MarginSurface`'s own PMF-based pricing -- that capability is
preserved, just no longer the default path.

``price_ats_distribution`` is the one core closed-form function -- MONEYLINE
and TOTAL are documented special cases of it, not separate formulas:

- MONEYLINE = ``price_ats_distribution(mean_margin, margin_sd, 0.0)``
- TOTAL uses the identical closed form with the edge computed as
  ``mean_total - total_line`` instead of ``mean_margin + home_spread`` --
  see :func:`price_total_distribution`.

All functions are vectorized (accept scalars or equal-length array-likes)
and deterministic: same inputs, same outputs, no randomness.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from nfl_hybrid.pricing.margin_surface import MarginSurface


def _broadcast(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    return tuple(np.broadcast_to(a, shape).astype(float) for a in arrays)


def _closed_form_upper_push_lower(
    edge_mean: np.ndarray, sd: np.ndarray, threshold: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JointScoreModel's own formula, verbatim (see
    :meth:`nfl_hybrid.modern.joint_score.JointScoreModel.raw_probabilities`):
    continuity-corrected at an INTEGER ``threshold`` (the declared spread for
    ATS, the declared total for TOTAL); a plain, uncorrected ``norm.cdf`` at
    every other real ``threshold`` -- including quarter-points and any other
    fractional consensus line a median-of-books calculation can produce, not
    only half-points. Push probability is exactly 0 for any non-integer
    threshold: the underlying margin/total realization is always integer, so
    it can only exactly cancel an integer line."""
    is_integer = np.isclose(threshold, np.round(threshold), atol=1e-9)
    upper_boundary = norm.cdf((0.5 - edge_mean) / sd)
    lower_boundary = norm.cdf((-0.5 - edge_mean) / sd)
    push = np.where(is_integer, upper_boundary - lower_boundary, 0.0)
    upper = np.where(is_integer, 1.0 - upper_boundary, norm.cdf(edge_mean / sd))
    lower = 1.0 - upper - push
    return upper, push, lower


def price_ats_distribution(
    mean_margin: float | np.ndarray,
    margin_sd: float | np.ndarray,
    home_spread: float | np.ndarray,
    *,
    surface: MarginSurface | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The canonical ATS / moneyline three-way pricing function.

    ``home covers`` iff ``actual_margin + home_spread > 0`` (the repository's
    one spread convention: a negative ``home_spread`` means the home team is
    favored -- see :mod:`nfl_hybrid.pricing.margin_surface`'s module
    docstring and ``tests/test_distribution_pricing.py``'s sign tests). Push
    mass is the continuity-corrected probability the integer margin exactly
    cancels an INTEGER spread; exactly zero for ANY non-integer spread
    (half-point, quarter-point, or otherwise -- see
    :func:`_closed_form_upper_push_lower`).

    Returns ``(home_cover, push, away_cover)`` -- unconditional, three-way,
    each summing to 1 elementwise. Deterministic; no market-anchored mean is
    substituted for ``mean_margin`` here -- the caller supplies whichever
    mean it intends to price (a learned model's predicted margin, or a
    market-implied one), this function only prices the distribution around
    it.

    ``surface``: when omitted (or an explicit ``MarginSurface(method=
    "normal")``), this is JointScoreModel's own closed-form formula,
    extracted verbatim -- exact parity for any finite real ``home_spread``.
    Passing a non-"normal" surface (e.g. ``method="empirical"``) instead
    routes through that surface's own PMF-based pricing, preserving that
    capability for callers outside Fix 4 that need it.
    """
    surf = surface or MarginSurface(method="normal")
    mean_arr, sd_arr, spread_arr = _broadcast(
        np.atleast_1d(np.asarray(mean_margin, dtype=float)),
        np.atleast_1d(np.asarray(margin_sd, dtype=float)),
        np.atleast_1d(np.asarray(home_spread, dtype=float)),
    )
    scalar_input = np.isscalar(mean_margin) and np.isscalar(margin_sd) and np.isscalar(home_spread)

    home = np.full(mean_arr.shape, np.nan)
    push = np.full(mean_arr.shape, np.nan)
    away = np.full(mean_arr.shape, np.nan)
    ready = np.isfinite(mean_arr) & np.isfinite(sd_arr) & np.isfinite(spread_arr)

    if ready.any():
        if surf.method == "normal":
            edge_mean = mean_arr[ready] + spread_arr[ready]
            h, p, a = _closed_form_upper_push_lower(edge_mean, sd_arr[ready], spread_arr[ready])
            home[ready], push[ready], away[ready] = h, p, a
        else:
            for i in np.flatnonzero(ready):
                h, p, a = surf.cover_probabilities(
                    closing_spread=-mean_arr[i], cover_line=spread_arr[i], sigma=sd_arr[i]
                )
                home[i], push[i], away[i] = h, p, a

    if scalar_input:
        return float(home[0]), float(push[0]), float(away[0])
    return home, push, away


def price_total_distribution(
    mean_total: float | np.ndarray,
    total_sd: float | np.ndarray,
    total_line: float | np.ndarray,
    *,
    surface: MarginSurface | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The canonical TOTAL (over/push/under) three-way pricing function:
    the identical closed form as :func:`price_ats_distribution`
    (:func:`_closed_form_upper_push_lower`), with the edge computed as
    ``mean_total - total_line`` (over iff the edge is positive) instead of
    ``mean_margin + home_spread`` -- matching
    :meth:`JointScoreModel.raw_probabilities`'s own ``total_edge_mean``
    exactly, for any finite real ``total_line``, not just integer/half-point.

    Returns ``(over, push, under)``, unconditional three-way, summing to 1.
    """
    surf = surface or MarginSurface(method="normal")
    mean_arr, sd_arr, line_arr = _broadcast(
        np.atleast_1d(np.asarray(mean_total, dtype=float)),
        np.atleast_1d(np.asarray(total_sd, dtype=float)),
        np.atleast_1d(np.asarray(total_line, dtype=float)),
    )
    scalar_input = np.isscalar(mean_total) and np.isscalar(total_sd) and np.isscalar(total_line)

    over = np.full(mean_arr.shape, np.nan)
    push = np.full(mean_arr.shape, np.nan)
    under = np.full(mean_arr.shape, np.nan)
    ready = np.isfinite(mean_arr) & np.isfinite(sd_arr) & np.isfinite(line_arr)

    if ready.any():
        if surf.method == "normal":
            edge_mean = mean_arr[ready] - line_arr[ready]
            o, p, u = _closed_form_upper_push_lower(edge_mean, sd_arr[ready], line_arr[ready])
            over[ready], push[ready], under[ready] = o, p, u
        else:
            for i in np.flatnonzero(ready):
                o, p, u = surf.cover_probabilities(
                    closing_spread=-mean_arr[i], cover_line=-line_arr[i], sigma=sd_arr[i]
                )
                over[i], push[i], under[i] = o, p, u

    if scalar_input:
        return float(over[0]), float(push[0]), float(under[0])
    return over, push, under


def price_moneyline_distribution(
    mean_margin: float | np.ndarray,
    margin_sd: float | np.ndarray,
    *,
    surface: MarginSurface | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moneyline (win/tie/loss) as the fixed ``home_spread = 0`` special case
    of :func:`price_ats_distribution` -- never a separate formula."""
    return price_ats_distribution(mean_margin, margin_sd, 0.0, surface=surface)
