"""Weekly betting-card construction.

Ties together the joint-score distribution, the de-vigged / reference market,
the penalized-Kelly staking engine and the governance abstention gate into a
single per-(game, market) card.

The frozen 2025 production decision retained the **market baseline** for all
three markets (see ``config/production_model_spec.json``): the production model
*is* the de-vigged market, so its edge is zero by construction and every market
carries readiness status ``RETAIN_BASELINE``. The card therefore prices each
game honestly at the market and recommends no bets, flagged
``no-bet: edge not established`` -- exactly the behaviour governance requires of
an unproven market. Supplying explicit model probabilities (from a fitted
:class:`~nfl_hybrid.modern.joint_score.JointScoreModel`) and a non-baseline
readiness status is all that is needed to let a genuinely-supported market stake.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_hybrid.staking.kelly import StakingPolicy, stake_bets

MARKET_ALIASES = {
    "pregame_moneyline": "moneyline",
    "moneyline": "moneyline",
    "pregame_ats": "ats",
    "ats": "ats",
    "pregame_total": "total",
    "total": "total",
}


@dataclass(frozen=True)
class CardConfig:
    margin_sd: float = 13.5
    total_sd: float = 13.5
    default_decimal_odds: float = 1.909  # -110


def readiness_from_production_spec(path: str | Path) -> dict[str, str]:
    """Map each production market to a staking readiness status.

    A market whose frozen production family is ``market_baseline`` did not beat
    the market in the final test, so it is ``RETAIN_BASELINE`` for staking.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    status: dict[str, str] = {}
    for row in payload["production_models"]:
        market = MARKET_ALIASES.get(row["market"], row["market"])
        family = row["production_model_family"]
        status[market] = (
            "RETAIN_BASELINE" if family == "market_baseline" else "PROVISIONAL_ONLY"
        )
    return status


def _reference_distribution(
    home_spread: np.ndarray,
    total_line: np.ndarray,
    *,
    config: CardConfig,
    surface=None,
) -> pd.DataFrame:
    """Closed-form market/reference probabilities from closing lines.

    Margin quantities (moneyline tie/win, ATS cover/push) come from the calibrated
    margin surface when ``surface`` is supplied (empirical PMF; correct key-number
    push mass); otherwise the normal continuity-corrected model. Totals always use
    the normal model with the fitted total sigma.
    """

    margin_mean = -home_spread
    total_mean = total_line
    m_sd, t_sd = config.margin_sd, config.total_sd

    if surface is not None:
        n = len(home_spread)
        tie = np.empty(n); home_win = np.empty(n); ats_push = np.empty(n); cover = np.empty(n)
        away_cover = np.empty(n)  # INDEPENDENT third component (not 1-cover-push)
        for i in range(n):
            hs = home_spread[i]
            hw, ti, _ = surface.moneyline_probabilities(hs, m_sd)
            home_win[i], tie[i] = hw, ti
            hc, pu, ac = surface.cover_probabilities(hs, hs, m_sd)  # cover the closing line
            cover[i], ats_push[i], away_cover[i] = hc, pu, ac
    else:
        away_cover = None
        tie = norm.cdf((0.5 - margin_mean) / m_sd) - norm.cdf((-0.5 - margin_mean) / m_sd)
        home_win = np.clip(1.0 - norm.cdf((0.5 - margin_mean) / m_sd), 0.0, 1.0)
        ats_edge = margin_mean + home_spread
        ats_integer = np.isfinite(home_spread) & np.isclose(home_spread, np.round(home_spread))
        ats_push = np.where(
            ats_integer,
            norm.cdf((0.5 - ats_edge) / m_sd) - norm.cdf((-0.5 - ats_edge) / m_sd),
            0.0,
        )
        cover = np.where(
            ats_integer,
            1.0 - norm.cdf((0.5 - ats_edge) / m_sd),
            norm.cdf(ats_edge / m_sd),
        )

    total_edge = total_mean - total_line  # == 0 for the reference line
    total_integer = np.isfinite(total_line) & np.isclose(total_line, np.round(total_line))
    total_push = np.where(
        total_integer,
        norm.cdf((0.5 - total_edge) / t_sd) - norm.cdf((-0.5 - total_edge) / t_sd),
        0.0,
    )
    over = np.where(
        total_integer,
        1.0 - norm.cdf((0.5 - total_edge) / t_sd),
        norm.cdf(total_edge / t_sd),
    )
    if surface is None:
        # independent away-cover component (not derived from cover/push)
        away_cover = np.where(
            ats_integer, norm.cdf((-0.5 - ats_edge) / m_sd), 1.0 - norm.cdf(ats_edge / m_sd)
        )
        under = np.where(
            total_integer, norm.cdf((-0.5 - total_edge) / t_sd), 1.0 - norm.cdf(total_edge / t_sd)
        )
    else:
        under = np.clip(1.0 - np.clip(over, 0, 1) - np.clip(total_push, 0, 1), 0.0, 1.0)

    return pd.DataFrame(
        {
            "reference_home_margin": margin_mean,
            "reference_home_win_probability": home_win,
            "tie_probability": np.clip(tie, 0.0, 1.0),
            "reference_cover_probability": np.clip(cover, 0.0, 1.0),
            "reference_away_cover_probability": np.clip(away_cover, 0.0, 1.0),
            "ats_push_probability": np.clip(ats_push, 0.0, 1.0),
            "reference_over_probability": np.clip(over, 0.0, 1.0),
            "reference_under_probability": np.clip(under, 0.0, 1.0),
            "total_push_probability": np.clip(total_push, 0.0, 1.0),
        }
    )


def build_betting_card(
    games: pd.DataFrame,
    *,
    readiness_by_market: dict[str, str],
    staking_policy: StakingPolicy | None = None,
    config: CardConfig | None = None,
    pricing_artifact=None,
) -> pd.DataFrame:
    """Build a per-(game, market) betting card.

    Required ``games`` columns: ``game_id``, ``season``, ``week``,
    ``home_team``, ``away_team``, ``home_spread`` (closing/reference),
    ``total_line``.

    Optional columns override the reference market with model / quote inputs:
    ``model_home_win_probability``, ``model_cover_probability``,
    ``model_over_probability``, ``market_ml_home_probability``,
    ``market_cover_probability``, ``market_over_probability``,
    ``offered_decimal_moneyline`` / ``_ats`` / ``_total``, ``market_source``.
    """

    config = config or CardConfig()
    policy = staking_policy or StakingPolicy()

    surface = None
    pricing_surface_version = "normal_builtin"
    if pricing_artifact is not None:
        # use the fitted margin surface + fitted sigmas from the frozen artifact
        surface = pricing_artifact.margin_surface
        config = CardConfig(
            margin_sd=pricing_artifact.margin_sigma,
            total_sd=pricing_artifact.total_sigma,
            default_decimal_odds=config.default_decimal_odds,
        )
        pricing_surface_version = f"{pricing_artifact.version}:{pricing_artifact.artifact_sha256[:12]}"

    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_spread",
        "total_line",
    }
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(f"build_betting_card missing columns: {missing}")

    g = games.copy()
    home_spread = pd.to_numeric(g["home_spread"], errors="coerce").to_numpy(float)
    total_line = pd.to_numeric(g["total_line"], errors="coerce").to_numpy(float)
    ref = _reference_distribution(home_spread, total_line, config=config, surface=surface)
    market_source = g.get("market_source", pd.Series(["REFERENCE-LINE PROXY"] * len(g)))

    def _col(name: str, fallback: np.ndarray) -> np.ndarray:
        if name in g.columns:
            return pd.to_numeric(g[name], errors="coerce").fillna(
                pd.Series(fallback, index=g.index)
            ).to_numpy(float)
        return fallback

    def _odds(name: str) -> np.ndarray:
        if name in g.columns:
            return pd.to_numeric(g[name], errors="coerce").fillna(
                config.default_decimal_odds
            ).to_numpy(float)
        return np.full(len(g), config.default_decimal_odds)

    # market fair probabilities (de-vigged quotes if supplied, else reference)
    market_ml = _col("market_ml_home_probability", ref["reference_home_win_probability"].to_numpy())
    market_cover = _col("market_cover_probability", ref["reference_cover_probability"].to_numpy())
    market_over = _col("market_over_probability", ref["reference_over_probability"].to_numpy())

    # model probabilities (fall back to market baseline == production behaviour)
    model_ml = _col("model_home_win_probability", market_ml)
    model_cover = _col("model_cover_probability", market_cover)
    model_over = _col("model_over_probability", market_over)

    base = {
        "game_id": g["game_id"].astype(str).to_numpy(),
        "season": g["season"].to_numpy(),
        "week": g["week"].to_numpy(),
        "matchup": (g["away_team"].astype(str) + " @ " + g["home_team"].astype(str)).to_numpy(),
        "model_spread": home_spread,  # home team's model/market spread
        "predicted_home_margin": ref["reference_home_margin"].to_numpy(),
        "closing_or_reference_line": home_spread,
        "total_line": total_line,
        "market_source": np.asarray(market_source),
    }

    per_market = []
    # each spec: market, win_prob, market_prob, push_prob, INDEPENDENT complement, odds
    specs = [
        (
            "moneyline", model_ml, market_ml, np.zeros(len(g)),
            np.clip(1.0 - ref["reference_home_win_probability"].to_numpy() - ref["tie_probability"].to_numpy(), 0, 1),
            _odds("offered_decimal_moneyline"),
        ),
        (
            "ats", model_cover, market_cover, ref["ats_push_probability"].to_numpy(),
            ref["reference_away_cover_probability"].to_numpy(),  # independent, not 1-cover-push
            _odds("offered_decimal_ats"),
        ),
        (
            "total", model_over, market_over, ref["total_push_probability"].to_numpy(),
            ref["reference_under_probability"].to_numpy(),  # independent
            _odds("offered_decimal_total"),
        ),
    ]

    for market, model_p, market_p, push_p, complement_p, odds in specs:
        block = pd.DataFrame(
            {
                **base,
                "market": market,
                "readiness_status": readiness_by_market.get(market, "RETAIN_BASELINE"),
                "calibrated_probability": model_p,
                "model_probability": model_p,
                "market_fair_probability": market_p,
                "push_probability": push_p,
                "independent_complement_probability": complement_p,
                "tie_probability": ref["tie_probability"].to_numpy(),
                "p_cover": model_cover,
                "p_over": model_over,
                "offered_decimal": odds,
            }
        )
        per_market.append(block)

    card = pd.concat(per_market, ignore_index=True)
    card = stake_bets(card, policy=policy)
    from nfl_hybrid.pricing.alt_lines import CALIBRATION_STATEMENT

    card["calibration_statement"] = CALIBRATION_STATEMENT
    card["pricing_surface_version"] = pricing_surface_version
    return card.sort_values(["week", "game_id", "market"], kind="stable").reset_index(
        drop=True
    )
