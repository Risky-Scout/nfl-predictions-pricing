"""Market-relative evaluation.

Coin-flip and legacy-Elo baselines can never establish *edge*: the only
benchmark that matters is the market itself. This module scores the model
against de-vigged closing implied probabilities, per market, per season and
pooled:

(a) Brier / log-loss vs de-vigged closing implied probabilities. When only
    Spreadspoke closing *reference* lines exist (no timestamped two-way quotes),
    the implied probabilities are derived from the score distribution and every
    such row is labelled ``REFERENCE-LINE PROXY`` via ``market_source``.
(b) ATS pick accuracy vs the 52.4% break-even line, with a 95% bootstrap CI.
(c) Margin MAE vs closing-spread MAE, plus the fraction of games where the model
    beats the market margin (a closing-line-value proxy).
(d) Class-wise expected calibration error (ECE).

Nothing here re-runs or mutates the frozen 2025 final test; it is a reusable
diagnostic that operates on any walk-forward prediction frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss

ATS_BREAKEVEN = 0.524  # -110 vig break-even win rate


@dataclass(frozen=True)
class MarketRelativeConfig:
    margin_sd: float = 13.5
    total_sd: float = 13.5
    ece_bins: int = 10
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 20260401
    probability_epsilon: float = 1e-6


def market_implied_probabilities_from_lines(
    frame: pd.DataFrame,
    *,
    config: MarketRelativeConfig | None = None,
) -> pd.DataFrame:
    """Derive de-vigged reference implied probabilities from closing lines.

    A closing spread/total is, by construction, the market's fair midpoint, so
    the reference cover and over probabilities are ~0.5 and the moneyline
    probability follows from the spread under the same normal score model the
    joint model uses. These are REFERENCE-LINE PROXIES, not timestamped quotes.
    """

    config = config or MarketRelativeConfig()
    home_spread = pd.to_numeric(frame["home_spread"], errors="coerce").to_numpy(float)
    # Market's implied home margin is the negative of the home spread.
    market_home_margin = -home_spread
    ml_home = 1.0 - norm.cdf((0.0 - market_home_margin) / config.margin_sd)
    n = len(frame)
    return pd.DataFrame(
        {
            "market_ml_home_probability": ml_home,
            "market_cover_home_probability": np.full(n, 0.5),
            "market_over_probability": np.full(n, 0.5),
            "market_source": np.where(
                np.isfinite(home_spread),
                "REFERENCE-LINE PROXY",
                "UNAVAILABLE",
            ),
        },
        index=frame.index,
    )


def classwise_ece(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error over equal-width probability bins."""

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0.0, 1.0])
    y, p = y[mask], p[mask]
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        ece += (sel.mean()) * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def _paired_accuracy_bootstrap(
    correct: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    correct = np.asarray(correct, dtype=float)
    correct = correct[np.isfinite(correct)]
    if correct.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = correct.size
    means = np.empty(repetitions)
    for i in range(repetitions):
        sample = rng.integers(0, n, size=n)
        means[i] = correct[sample].mean()
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _binary_scores(
    y: np.ndarray,
    p: np.ndarray,
    eps: float,
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    mask = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0.0, 1.0])
    if mask.sum() == 0:
        return {"n": 0, "brier": np.nan, "log_loss": np.nan}
    return {
        "n": int(mask.sum()),
        "brier": float(brier_score_loss(y[mask], p[mask])),
        "log_loss": float(log_loss(y[mask], p[mask], labels=[0, 1])),
    }


def _market_block(
    label: str,
    subset: pd.DataFrame,
    *,
    outcome_col: str,
    model_prob_col: str,
    market_prob_col: str,
    market_source: str,
    config: MarketRelativeConfig,
) -> dict[str, object]:
    y = subset[outcome_col].to_numpy(float)
    model_p = subset[model_prob_col].to_numpy(float)
    market_p = subset[market_prob_col].to_numpy(float)
    eps = config.probability_epsilon

    model_scores = _binary_scores(y, model_p, eps)
    market_scores = _binary_scores(y, market_p, eps)

    # ATS pick accuracy vs break-even, with a 95% bootstrap CI.
    pick_home = model_p > 0.5
    correct = np.where(pick_home, y == 1.0, y == 0.0).astype(float)
    valid = np.isfinite(y) & np.isfinite(model_p) & np.isin(y, [0.0, 1.0])
    correct = correct[valid]
    accuracy = float(correct.mean()) if correct.size else float("nan")
    ci_lower, ci_upper = _paired_accuracy_bootstrap(
        correct,
        repetitions=config.bootstrap_repetitions,
        seed=config.bootstrap_seed,
    )

    return {
        "segment": label,
        "market": outcome_col,
        "market_source": market_source,
        "n": model_scores["n"],
        "model_brier": model_scores["brier"],
        "market_brier": market_scores["brier"],
        "brier_gain_vs_market": market_scores["brier"] - model_scores["brier"],
        "model_log_loss": model_scores["log_loss"],
        "market_log_loss": market_scores["log_loss"],
        "log_loss_gain_vs_market": market_scores["log_loss"]
        - model_scores["log_loss"],
        "model_ece": classwise_ece(y, model_p, n_bins=config.ece_bins),
        "market_ece": classwise_ece(y, market_p, n_bins=config.ece_bins),
        "pick_accuracy": accuracy,
        "pick_accuracy_ci_lower": ci_lower,
        "pick_accuracy_ci_upper": ci_upper,
        "breakeven": ATS_BREAKEVEN,
        "beats_breakeven_ci": bool(
            np.isfinite(ci_lower) and ci_lower > ATS_BREAKEVEN
        ),
    }


def _margin_block(
    label: str,
    subset: pd.DataFrame,
    *,
    outcome_col: str,
    model_pred_col: str,
    market_pred: np.ndarray,
    market_source: str,
) -> dict[str, object]:
    y = subset[outcome_col].to_numpy(float)
    model_pred = subset[model_pred_col].to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(model_pred) & np.isfinite(market_pred)
    y, model_pred, market_pred = y[valid], model_pred[valid], market_pred[valid]
    if y.size == 0:
        return {
            "segment": label,
            "market": f"{outcome_col}_regression",
            "market_source": market_source,
            "n": 0,
            "model_mae": np.nan,
            "market_mae": np.nan,
            "mae_gain_vs_market": np.nan,
            "fraction_model_beats_market": np.nan,
        }
    model_err = np.abs(y - model_pred)
    market_err = np.abs(y - market_pred)
    return {
        "segment": label,
        "market": f"{outcome_col}_regression",
        "market_source": market_source,
        "n": int(y.size),
        "model_mae": float(model_err.mean()),
        "market_mae": float(market_err.mean()),
        "mae_gain_vs_market": float(market_err.mean() - model_err.mean()),
        "fraction_model_beats_market": float((model_err < market_err).mean()),
    }


def evaluate_market_relative(
    predictions: pd.DataFrame,
    *,
    config: MarketRelativeConfig | None = None,
    market_probabilities: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-season and pooled market-relative scorecards.

    ``predictions`` must contain the walk-forward schema columns: ``season``,
    ``home_win``, ``home_cover``, ``over``, ``home_margin``, ``total_points``,
    ``home_spread``, ``total_line``, ``predicted_margin``, and the calibrated
    ``*_probability_no_tie`` / ``*_probability_no_push`` columns.

    If ``market_probabilities`` (de-vigged timestamped quotes) is not supplied,
    reference-line proxies are derived from the closing lines and every row is
    labelled ``REFERENCE-LINE PROXY``.
    """

    config = config or MarketRelativeConfig()
    frame = predictions.copy()

    if market_probabilities is None:
        market_probabilities = market_implied_probabilities_from_lines(
            frame, config=config
        )
    frame = pd.concat([frame, market_probabilities], axis=1)
    market_source = str(
        pd.Series(frame["market_source"]).mode().iloc[0]
        if "market_source" in frame.columns and len(frame)
        else "REFERENCE-LINE PROXY"
    )

    probability_specs = [
        ("home_win", "home_win_probability_no_tie", "market_ml_home_probability"),
        (
            "home_cover",
            "home_cover_probability_no_push",
            "market_cover_home_probability",
        ),
        ("over", "over_probability_no_push", "market_over_probability"),
    ]

    rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []

    segments: list[tuple[str, pd.DataFrame]] = [("pooled", frame)]
    for season, sub in frame.groupby("season"):
        segments.append((f"season_{int(season)}", sub))

    for label, sub in segments:
        for outcome_col, model_col, market_col in probability_specs:
            if model_col not in sub.columns:
                continue
            rows.append(
                _market_block(
                    label,
                    sub,
                    outcome_col=outcome_col,
                    model_prob_col=model_col,
                    market_prob_col=market_col,
                    market_source=market_source,
                    config=config,
                )
            )
        if "predicted_margin" in sub.columns and "home_spread" in sub.columns:
            market_margin = -pd.to_numeric(
                sub["home_spread"], errors="coerce"
            ).to_numpy(float)
            margin_rows.append(
                _margin_block(
                    label,
                    sub,
                    outcome_col="home_margin",
                    model_pred_col="predicted_margin",
                    market_pred=market_margin,
                    market_source=market_source,
                )
            )

    return {
        "probability_scorecard": pd.DataFrame(rows),
        "margin_scorecard": pd.DataFrame(margin_rows),
    }
