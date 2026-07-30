"""Interpretable margin probability surface.

One interface behind moneyline, ATS push/cover, and alternate spreads. Two
implementations, selectable from the calibration artifact:

- ``normal`` — discretized-normal CDF pricing (the repository's existing behaviour;
  the safe fallback). Mean home margin = ``-closing_spread``; sigma supplied per
  game (constant, or from the fitted sigma model).
- ``empirical`` — a conditional **integer** home-margin PMF built from Spreadspoke
  2002-2024, shrunk toward the discretized-normal baseline, with a normal tail
  beyond ``|margin| > tail_cutoff``. Key-number concentrations (3, 6, 7, 10, 14)
  are retained from data; **no key-number boost is hard-coded.**

Spread convention matches the repo: ``home covers line L`` iff
``home_margin + L > 0``; integer L pushes when ``home_margin + L == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_TAIL_CUTOFF = 21
_MARGIN_MIN, _MARGIN_MAX = -75, 75


def discretized_normal_pmf(mean: float, sigma: float, *, m_min: int = _MARGIN_MIN, m_max: int = _MARGIN_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Integer-margin PMF from a normal, via continuity correction. Sums to 1."""
    sigma = max(float(sigma), 1e-6)
    margins = np.arange(m_min, m_max + 1)
    upper = norm.cdf((margins + 0.5 - mean) / sigma)
    lower = norm.cdf((margins - 0.5 - mean) / sigma)
    p = upper - lower
    # fold the outer tails into the boundary integers so mass sums to exactly 1
    p[0] += norm.cdf((m_min - 0.5 - mean) / sigma)
    p[-1] += 1.0 - norm.cdf((m_max + 0.5 - mean) / sigma)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return margins, p


def _cover_from_pmf(margins: np.ndarray, probs: np.ndarray, cover_line: float) -> tuple[float, float, float]:
    """(home_cover, push, away_cover) for cover_line L: home covers if m + L > 0."""
    edge = margins + cover_line  # >0 home cover, ==0 push, <0 away cover
    home = float(probs[edge > 1e-9].sum())
    push = float(probs[np.isclose(edge, 0.0, atol=1e-9)].sum())
    away = float(probs[edge < -1e-9].sum())
    total = home + push + away
    if total > 0:
        home, push, away = home / total, push / total, away / total
    return home, push, away


@dataclass
class MarginSurface:
    method: str = "normal"
    # empirical config
    pmf_table: pd.DataFrame | None = None  # columns: bucket_center, margin, probability
    bucket_width: float = 2.0
    tail_cutoff: int = DEFAULT_TAIL_CUTOFF

    def __post_init__(self) -> None:
        self._lookup: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        if self.method == "empirical":
            if self.pmf_table is None or len(self.pmf_table) == 0:
                raise ValueError("empirical MarginSurface requires a pmf_table.")
            for center, g in self.pmf_table.groupby("bucket_center"):
                g = g.sort_values("margin")
                self._lookup[round(float(center), 3)] = (
                    g["margin"].to_numpy(int), g["probability"].to_numpy(float)
                )
            self._centers = np.array(sorted(self._lookup.keys()))

    # -- core --------------------------------------------------------------- #
    def pmf(self, closing_spread: float, sigma: float = 13.5) -> tuple[np.ndarray, np.ndarray]:
        if self.method == "normal":
            return discretized_normal_pmf(-float(closing_spread), sigma)
        # empirical: nearest registered bucket center
        center = float(self._centers[np.argmin(np.abs(self._centers - float(closing_spread)))])
        return self._lookup[round(center, 3)]

    def cover_probabilities(self, closing_spread: float, cover_line: float, sigma: float = 13.5) -> tuple[float, float, float]:
        margins, probs = self.pmf(closing_spread, sigma)
        return _cover_from_pmf(margins, probs, cover_line)

    def moneyline_probabilities(self, closing_spread: float, sigma: float = 13.5) -> tuple[float, float, float]:
        # home win = cover at line 0 (home_margin > 0); tie is the 0 push
        return self.cover_probabilities(closing_spread, 0.0, sigma)

    def to_config(self) -> dict:
        cfg = {"method": self.method, "bucket_width": self.bucket_width, "tail_cutoff": self.tail_cutoff}
        return cfg


# --------------------------------------------------------------------------- #
# Offline builder for the empirical shrinkage PMF
# --------------------------------------------------------------------------- #
def build_empirical_pmf_table(
    games: pd.DataFrame,
    *,
    bucket_width: float = 2.0,
    baseline_sigma: float = 13.5,
    shrinkage_k: float = 200.0,
    tail_cutoff: int = DEFAULT_TAIL_CUTOFF,
    spread_min: float = -20.0,
    spread_max: float = 20.0,
) -> pd.DataFrame:
    """Build a conditional integer-margin PMF, shrunk toward the discretized-normal
    baseline, with a normal tail beyond ``|margin| > tail_cutoff``.

    ``games`` needs ``home_margin`` (signed int) and ``closing_home_spread``.
    Returns a long, inspectable table: bucket_center, margin, empirical_count,
    baseline_probability, shrinkage_weight, probability.
    """
    g = games.dropna(subset=["home_margin", "closing_home_spread"]).copy()
    g["home_margin"] = g["home_margin"].round().astype(int)
    g["closing_home_spread"] = pd.to_numeric(g["closing_home_spread"], errors="coerce")

    centers = np.round(np.arange(spread_min, spread_max + bucket_width / 2, bucket_width), 3)
    core = np.arange(-tail_cutoff, tail_cutoff + 1)
    rows = []
    for center in centers:
        lo, hi = center - bucket_width / 2, center + bucket_width / 2
        sub = g[(g["closing_home_spread"] >= lo) & (g["closing_home_spread"] < hi)]
        n = len(sub)
        # baseline discretized-normal centered at -center (mean home margin)
        base_margins, base_probs = discretized_normal_pmf(-float(center), baseline_sigma)
        base_lookup = dict(zip(base_margins.tolist(), base_probs.tolist()))
        # empirical counts on the core integer range
        counts = sub["home_margin"].clip(-tail_cutoff, tail_cutoff).value_counts().to_dict() if n else {}
        core_emp = np.array([counts.get(int(m), 0) for m in core], dtype=float)
        core_base = np.array([base_lookup.get(int(m), 0.0) for m in core])
        core_base = core_base / core_base.sum() if core_base.sum() > 0 else core_base
        emp_frac = core_emp / core_emp.sum() if core_emp.sum() > 0 else core_base
        w = n / (n + shrinkage_k)
        core_prob = w * emp_frac + (1 - w) * core_base
        # tail mass beyond cutoff from the baseline normal, spread over tail integers
        tail_mass = float(1.0 - sum(base_lookup.get(int(m), 0.0) for m in core))
        tail_mass = max(tail_mass, 0.0)
        tail_margins = np.concatenate([np.arange(_MARGIN_MIN, -tail_cutoff), np.arange(tail_cutoff + 1, _MARGIN_MAX + 1)])
        tail_base = np.array([base_lookup.get(int(m), 1e-12) for m in tail_margins])
        tail_base = tail_base / tail_base.sum() if tail_base.sum() > 0 else np.full(len(tail_margins), 1.0 / len(tail_margins))
        tail_prob = tail_mass * tail_base
        # assemble & normalize exactly
        all_margins = np.concatenate([tail_margins[tail_margins < -tail_cutoff], core, tail_margins[tail_margins > tail_cutoff]])
        all_probs = np.concatenate([
            tail_prob[tail_margins < -tail_cutoff],
            core_prob * (1.0 - tail_mass),
            tail_prob[tail_margins > tail_cutoff],
        ])
        all_probs = np.clip(all_probs, 0.0, None)
        all_probs /= all_probs.sum()
        order = np.argsort(all_margins)
        all_margins, all_probs = all_margins[order], all_probs[order]
        for m, p in zip(all_margins, all_probs):
            in_core = -tail_cutoff <= m <= tail_cutoff
            rows.append({
                "bucket_center": float(center), "margin": int(m),
                "empirical_count": int(base_lookup and counts.get(int(m), 0)) if in_core else 0,
                "baseline_probability": float(base_lookup.get(int(m), 0.0)),
                "shrinkage_weight": float(w),
                "probability": float(p),
            })
    return pd.DataFrame(rows)
