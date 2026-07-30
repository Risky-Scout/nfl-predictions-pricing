"""Uncertainty-adjusted fractional-Kelly staking.

The staking engine is deliberately conservative. Three properties are enforced
in code rather than left to judgement:

1. The *edge* used for sizing is penalized by the divergence between the model
   and the de-vigged market, so a bet is only large when the model both
   disagrees with the market *and* that disagreement is not merely a small
   probability wobble:

       penalized_edge = (model_prob - market_fair_prob)
                        - 0.5 * KL(model || market)

2. Sizing is fractional Kelly (default 1/8) with a hard 5% per-bet cap and a
   15% weekly exposure cap keyed by ``(season, week)`` -- never by team, so a
   slate of correlated same-team looks cannot each claim the full budget.

3. A governance abstention gate makes any market whose readiness status is
   ``RETAIN_BASELINE`` stake exactly zero. The row is still priced and printed;
   it simply carries the ``no-bet: edge not established`` flag. This mirrors
   :mod:`nfl_hybrid.governance.abstention`.

Cover/over/win probabilities are expected to come from the joint score
distribution (:mod:`nfl_hybrid.modern.joint_score`). This module never
reconstructs a probability from a linear edge heuristic; it consumes the
distributional probabilities it is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Markets whose readiness status does not qualify them for staking. Kept in one
# place so the gate matches governance/abstention.py.
NON_QUALIFYING_READINESS = frozenset({"RETAIN_BASELINE"})

NO_BET_FLAG = "no-bet: edge not established"


@dataclass(frozen=True)
class StakingPolicy:
    """Fractional-Kelly staking policy.

    All fractions are expressed as a share of current bankroll.
    """

    kelly_fraction: float = 0.125  # 1/8 Kelly
    per_bet_cap: float = 0.05  # 5% of bankroll on any single bet
    weekly_exposure_cap: float = 0.15  # 15% of bankroll per (season, week)
    minimum_penalized_edge: float = 0.02  # 2% minimum penalized edge to bet
    probability_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        for name in (
            "kelly_fraction",
            "per_bet_cap",
            "weekly_exposure_cap",
            "minimum_penalized_edge",
        ):
            value = getattr(self, name)
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1]; got {value}.")
        if self.probability_epsilon <= 0:
            raise ValueError("probability_epsilon must be positive.")


def binary_kl_divergence(
    model_prob: np.ndarray | float,
    market_prob: np.ndarray | float,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """KL(model || market) for a two-outcome (win / not-win) event, in nats.

    The push branch is folded into ``not-win`` because the divergence penalty is
    a regulariser on the win-side disagreement, not a full ternary distribution.
    """

    p = np.clip(np.asarray(model_prob, dtype=float), epsilon, 1.0 - epsilon)
    q = np.clip(np.asarray(market_prob, dtype=float), epsilon, 1.0 - epsilon)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def penalized_edge(
    model_prob: np.ndarray | float,
    market_fair_prob: np.ndarray | float,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """(model - market) - 0.5 * KL(model || market)."""

    p = np.asarray(model_prob, dtype=float)
    q = np.asarray(market_fair_prob, dtype=float)
    raw = p - q
    penalty = 0.5 * binary_kl_divergence(p, q, epsilon=epsilon)
    return raw - penalty


def full_kelly_fraction(
    win_prob: np.ndarray | float,
    offered_decimal: np.ndarray | float,
    push_prob: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Full-Kelly stake fraction for a bet that may push.

    With net decimal payout ``b = d - 1``, win probability ``p_w`` and push
    probability ``p_push`` (stake returned), the loss probability is
    ``p_l = 1 - p_w - p_push``. The growth-optimal fraction solving
    ``d/df E[log(1 + f * outcome)]`` for the push-adjusted bet reduces to::

        f* = (p_w * b - p_l) / b

    which is the standard ``(p_w * b - p_l) / b`` with the push mass simply
    removed from both win and loss sides. Negative values (no edge) clip to 0.
    """

    p_w = np.asarray(win_prob, dtype=float)
    push = np.asarray(push_prob, dtype=float)
    b = np.asarray(offered_decimal, dtype=float) - 1.0
    p_l = 1.0 - p_w - push
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(b > 0.0, (p_w * b - p_l) / b, 0.0)
    return np.clip(np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def stake_bets(
    frame: pd.DataFrame,
    *,
    policy: StakingPolicy | None = None,
) -> pd.DataFrame:
    """Compute recommended stakes for a betting card.

    Required columns:
      - ``model_probability``      win/cover/over probability from the joint model
      - ``market_fair_probability``de-vigged fair probability (reference proxy ok)
      - ``offered_decimal``        offered decimal odds (> 1)
      - ``season``, ``week``       used only to key the weekly exposure cap
      - ``readiness_status``       per-market governance status

    Optional columns:
      - ``push_probability``       tie/push mass (defaults to 0)

    Adds: ``penalized_edge``, ``full_kelly_fraction``, ``recommended_stake``,
    ``should_bet``, ``no_bet_reason``.
    """

    policy = policy or StakingPolicy()
    required = {
        "model_probability",
        "market_fair_probability",
        "offered_decimal",
        "season",
        "week",
        "readiness_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"stake_bets missing required columns: {missing}")

    out = frame.copy()
    eps = policy.probability_epsilon

    model_p = pd.to_numeric(out["model_probability"], errors="coerce").to_numpy(float)
    market_p = pd.to_numeric(
        out["market_fair_probability"], errors="coerce"
    ).to_numpy(float)
    offered = pd.to_numeric(out["offered_decimal"], errors="coerce").to_numpy(float)
    push = (
        pd.to_numeric(out.get("push_probability", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        if "push_probability" in out.columns
        else np.zeros(len(out))
    )
    readiness = out["readiness_status"].astype(str).to_numpy()

    pen_edge = penalized_edge(model_p, market_p, epsilon=eps)
    kelly = full_kelly_fraction(model_p, offered, push)
    sized = np.clip(kelly * policy.kelly_fraction, 0.0, policy.per_bet_cap)

    out["penalized_edge"] = pen_edge
    out["full_kelly_fraction"] = kelly

    # --- per-row eligibility and reasons -------------------------------------
    reasons: list[str] = []
    eligible = np.ones(len(out), dtype=bool)
    for i in range(len(out)):
        row_reasons: list[str] = []
        if readiness[i] in NON_QUALIFYING_READINESS:
            row_reasons.append(NO_BET_FLAG)
        if not np.isfinite(offered[i]) or offered[i] <= 1.0:
            row_reasons.append("no-bet: missing or invalid offered odds")
        if not np.isfinite(pen_edge[i]) or pen_edge[i] < policy.minimum_penalized_edge:
            row_reasons.append("no-bet: penalized edge below 2% minimum")
        if sized[i] <= 0.0:
            row_reasons.append("no-bet: nonpositive Kelly fraction")
        if row_reasons:
            eligible[i] = False
        reasons.append(";".join(dict.fromkeys(row_reasons)))

    stake = np.where(eligible, sized, 0.0)

    # --- 15% weekly exposure cap keyed by (season, week) ---------------------
    stake = _apply_weekly_exposure_cap(
        stake,
        out["season"].to_numpy(),
        out["week"].to_numpy(),
        cap=policy.weekly_exposure_cap,
    )

    out["recommended_stake"] = stake
    out["should_bet"] = stake > 0.0
    # If the weekly cap scaled an eligible bet all the way to zero, record it.
    reasons = [
        r if s > 0.0 or r else "no-bet: weekly exposure cap"
        for r, s in zip(reasons, stake)
    ]
    out["no_bet_reason"] = reasons
    return out


def _apply_weekly_exposure_cap(
    stake: np.ndarray,
    season: np.ndarray,
    week: np.ndarray,
    *,
    cap: float,
) -> np.ndarray:
    """Scale stakes down proportionally so each (season, week) totals <= cap."""

    stake = np.asarray(stake, dtype=float).copy()
    grouping = pd.DataFrame(
        {
            "season": np.asarray(season),
            "week": np.asarray(week).astype(str),
        }
    )
    for _, idx in grouping.groupby(["season", "week"], sort=False).indices.items():
        total = stake[idx].sum()
        if total > cap and total > 0:
            stake[idx] = stake[idx] * (cap / total)
    return stake
