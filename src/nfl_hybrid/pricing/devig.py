"""Two-way de-vig methods: proportional, power, and Shin.

Each takes the two raw implied probabilities of a paired two-way market (they sum
to the overround Phi > 1) and returns fair probabilities summing to 1.

- **proportional:** q_i = p_i / (p1 + p2). Simplest; the repository default.
- **power:** find k with p1^k + p2^k = 1; q_i = p_i^k. Deflates favourites more.
- **shin:** Shin (1993) insider-trading model; solves for z in [0, 1) so the
  fair probabilities sum to one. Between proportional and power in favourite bias.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_EPS = 1e-9


def devig_proportional(p1: float, p2: float) -> tuple[float, float]:
    s = p1 + p2
    if s <= 0:
        return (np.nan, np.nan)
    return (p1 / s, p2 / s)


def devig_power(p1: float, p2: float) -> tuple[float, float]:
    p1 = float(np.clip(p1, _EPS, 1 - _EPS))
    p2 = float(np.clip(p2, _EPS, 1 - _EPS))

    def f(k: float) -> float:
        return p1 ** k + p2 ** k - 1.0

    # overround > 1 => need k > 1 to deflate; bracket generously
    lo, hi = 1.0, 1.0
    if f(1.0) <= 0:  # already <= 1 (no vig); proportional
        return devig_proportional(p1, p2)
    hi = 2.0
    for _ in range(60):
        if f(hi) < 0:
            break
        hi *= 1.5
    try:
        k = brentq(f, lo, hi, xtol=1e-10, maxiter=200)
    except ValueError:
        return devig_proportional(p1, p2)
    return (p1 ** k, p2 ** k)


def devig_shin(p1: float, p2: float) -> tuple[float, float]:
    p1 = float(np.clip(p1, _EPS, 1 - _EPS))
    p2 = float(np.clip(p2, _EPS, 1 - _EPS))
    phi = p1 + p2
    if phi <= 1.0:
        return devig_proportional(p1, p2)

    def q(pi: float, z: float) -> float:
        # Shin fair probability for outcome i
        val = z * z + 4.0 * (1.0 - z) * (pi * pi) / phi
        return (np.sqrt(max(val, 0.0)) - z) / (2.0 * (1.0 - z))

    def g(z: float) -> float:
        return q(p1, z) + q(p2, z) - 1.0

    try:
        z = brentq(g, 1e-9, 1 - 1e-9, xtol=1e-12, maxiter=200)
    except ValueError:
        return devig_proportional(p1, p2)
    return (q(p1, z), q(p2, z))


METHODS = {
    "proportional": devig_proportional,
    "power": devig_power,
    "shin": devig_shin,
}


def devig_pair(p1: float, p2: float, method: str = "proportional") -> tuple[float, float]:
    return METHODS[method](p1, p2)
