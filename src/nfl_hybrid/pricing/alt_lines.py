"""Sharp-book pricing surface: quote ANY spread or total from the joint score
distribution.

Given a game's mean margin / total and their standard deviations (from the joint
score model, or market-anchored: mean margin = -home_spread, sd ~ 13.5), this
prices the full alternate ladder -- for every requested spread/total it returns
the fair probability, the fair decimal, and an offered decimal at a configurable
vig (hold) schedule -- plus the standard win/cover/over/push probabilities.

The three-region normal construction guarantees the put-call-style identity
P(home cover) + P(push) + P(away cover) = 1 exactly (and likewise over/push/under).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class AltLineConfig:
    margin_sd: float = 13.5
    total_sd: float = 13.5
    hold: float = 0.045  # two-way book hold applied to conditional (no-push) probs

    def __post_init__(self) -> None:
        if not (0.0 <= self.hold < 1.0):
            raise ValueError("hold must be in [0, 1).")


def _fair_decimal(win_prob: float, push_prob: float) -> float:
    win_prob = max(float(win_prob), 1e-9)
    return (1.0 - float(push_prob)) / win_prob


def _offered_decimal(win_prob: float, opp_prob: float, hold: float) -> float:
    """Offered decimal for a two-way side after applying a hold to the
    conditional (push-excluded) probabilities."""
    denom = win_prob + opp_prob
    if denom <= 0:
        return float("nan")
    conditional = win_prob / denom
    offered_implied = min(conditional * (1.0 + hold), 1.0 - 1e-9)
    return 1.0 / offered_implied


def _three_region(edge_mean: float, sd: float, line: float) -> tuple[float, float, float]:
    """(favourable, push, unfavourable) probabilities with integer-line push."""
    is_integer = np.isfinite(line) and np.isclose(line, round(line))
    if is_integer:
        push = float(norm.cdf((0.5 - edge_mean) / sd) - norm.cdf((-0.5 - edge_mean) / sd))
        fav = float(1.0 - norm.cdf((0.5 - edge_mean) / sd))
        unfav = float(norm.cdf((-0.5 - edge_mean) / sd))
    else:
        push = 0.0
        fav = float(norm.cdf(edge_mean / sd))
        unfav = 1.0 - fav
    return fav, push, unfav


def price_spread(mean_margin: float, spread: float, *, config: AltLineConfig | None = None) -> dict:
    """Price one spread. ``spread`` is the home team's line (home covers if
    home_margin + spread > 0)."""
    cfg = config or AltLineConfig()
    edge = mean_margin + spread
    home, push, away = _three_region(edge, cfg.margin_sd, spread)
    return {
        "line": spread,
        "home_cover_probability": home,
        "push_probability": push,
        "away_cover_probability": away,
        "home_fair_decimal": _fair_decimal(home, push),
        "away_fair_decimal": _fair_decimal(away, push),
        "home_offered_decimal": _offered_decimal(home, away, cfg.hold),
        "away_offered_decimal": _offered_decimal(away, home, cfg.hold),
    }


def price_total(mean_total: float, total: float, *, config: AltLineConfig | None = None) -> dict:
    cfg = config or AltLineConfig()
    edge = mean_total - total
    over, push, under = _three_region(edge, cfg.total_sd, total)
    return {
        "line": total,
        "over_probability": over,
        "push_probability": push,
        "under_probability": under,
        "over_fair_decimal": _fair_decimal(over, push),
        "under_fair_decimal": _fair_decimal(under, push),
        "over_offered_decimal": _offered_decimal(over, under, cfg.hold),
        "under_offered_decimal": _offered_decimal(under, over, cfg.hold),
    }


def price_spread_ladder(mean_margin: float, spreads, *, config: AltLineConfig | None = None) -> pd.DataFrame:
    return pd.DataFrame([price_spread(mean_margin, s, config=config) for s in spreads])


def price_total_ladder(mean_total: float, totals, *, config: AltLineConfig | None = None) -> pd.DataFrame:
    return pd.DataFrame([price_total(mean_total, t, config=config) for t in totals])


def price_moneyline(mean_margin: float, *, config: AltLineConfig | None = None) -> dict:
    """Win/tie/loss from the margin distribution (tie = integer-0 push)."""
    cfg = config or AltLineConfig()
    home, tie, away = _three_region(mean_margin, cfg.margin_sd, 0.0)
    return {
        "home_win_probability": home,
        "tie_probability": tie,
        "away_win_probability": away,
        "home_fair_decimal": _fair_decimal(home, tie),
        "away_fair_decimal": _fair_decimal(away, tie),
        "home_offered_decimal": _offered_decimal(home, away, cfg.hold),
        "away_offered_decimal": _offered_decimal(away, home, cfg.hold),
    }


def standard_ladder(spread: float, total: float, step_range: int = 14, *, step: float = 1.0):
    """A symmetric alternate ladder around the posted spread and total."""
    spreads = [round(spread + step * k, 1) for k in range(-step_range, step_range + 1)]
    totals = [round(total + step * k, 1) for k in range(-step_range, step_range + 1)]
    return spreads, totals


CALIBRATION_STATEMENT = (
    "Prices are market-anchored to the frozen production spec (model probability = "
    "de-vigged market). Any model overlay is PROVISIONAL and stakes ZERO until the live "
    "gate promotes it (>=8 live weeks, cumulative log-loss gain>0, pick-accuracy 95% CI "
    "lower>0.524). Alternate-ladder prices are fair probabilities from the joint score "
    "distribution with a configurable book hold; they are quotes, not edge claims."
)
