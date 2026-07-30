"""2026 season operating loop.

The 2026 backtest promoted nothing, so the season starts with every market at
RETAIN_BASELINE and the card stakes zero. This module is the *live search
instrument*: it records immutable pre-kickoff predictions, scores them after
resolution against the market, maintains a cumulative live scorecard, and applies
the fixed live-promotion gate that alone can let a PROVISIONAL candidate stake.

Nothing here makes a selection decision from the 2025 holdout; it operates only on
live 2026 (or, in a rehearsal, replayed) resolved games.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

_EPS = 1e-6
BREAKEVEN = 0.524


@dataclass(frozen=True)
class LivePromotionConfig:
    minimum_live_weeks: int = 8
    breakeven: float = BREAKEVEN
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 20260901


def log_week_predictions(
    card: pd.DataFrame,
    *,
    output_dir: str | Path,
    season: int,
    week: int,
    logged_at_utc: str,
    overwrite: bool = False,
) -> Path:
    """Write an immutable pre-kickoff prediction record.

    ``logged_at_utc`` must be stamped before kickoff by the caller. Refuses to
    overwrite an existing file unless ``overwrite=True`` -- the record is the
    live validation ground truth and must not be edited after the fact.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"predictions_wk{int(week):02d}.csv"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; live prediction records are immutable."
        )
    record = card.copy()
    record["season"] = season
    record["week"] = int(week)
    record.insert(0, "logged_at_utc", logged_at_utc)
    # keep season/week adjacent to the timestamp for readability
    ordered = ["logged_at_utc", "season", "week"] + [
        c for c in record.columns if c not in ("logged_at_utc", "season", "week")
    ]
    record[ordered].to_csv(path, index=False)
    return path


def _binary(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    m = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0.0, 1.0])
    if m.sum() == 0:
        return dict(n=0, brier=np.nan, log_loss=np.nan, correct=np.nan)
    pick_home = p[m] > 0.5
    correct = np.where(pick_home, y[m] == 1, y[m] == 0).astype(float)
    return dict(
        n=int(m.sum()),
        brier=float(brier_score_loss(y[m], p[m])),
        log_loss=float(log_loss(y[m], p[m], labels=[0, 1])),
        correct_sum=float(correct.sum()),
    )


def score_resolved_week(
    logged: pd.DataFrame,
    resolved: pd.DataFrame,
) -> pd.DataFrame:
    """Score one logged week after games resolve.

    ``logged`` is a prediction record (one row per game x market). ``resolved``
    must supply per-game outcomes: ``game_id``, ``home_win``, ``home_cover``,
    ``over``. Returns one row per market with model and market log-loss/Brier and
    a pick-correct count (for cumulative pick-accuracy CIs).
    """

    outcome_map = {"moneyline": "home_win", "ats": "home_cover", "total": "over"}
    merged = logged.merge(resolved[["game_id", "home_win", "home_cover", "over"]], on="game_id", how="inner")
    rows = []
    for market, outcome_col in outcome_map.items():
        sub = merged[merged["market"] == market]
        if sub.empty:
            continue
        y = sub[outcome_col].to_numpy(float)
        model = _binary(y, sub["model_probability"].to_numpy(float))
        mkt = _binary(y, sub["market_fair_probability"].to_numpy(float))
        rows.append(
            {
                "season": int(sub["season"].iloc[0]),
                "week": int(sub["week"].iloc[0]),
                "market": market,
                "n": model["n"],
                "model_log_loss": model["log_loss"],
                "market_log_loss": mkt["log_loss"],
                "model_brier": model["brier"],
                "market_brier": mkt["brier"],
                "model_correct_sum": model.get("correct_sum", np.nan),
            }
        )
    return pd.DataFrame(rows)


def cumulative_live_scorecard(weekly_scores: pd.DataFrame, *, config: LivePromotionConfig | None = None) -> pd.DataFrame:
    """Aggregate weekly scores into a cumulative per-market live scorecard."""

    config = config or LivePromotionConfig()
    rows = []
    for market, sub in weekly_scores.groupby("market"):
        n = int(sub["n"].sum())
        weeks = int(sub["week"].nunique())
        # sample-size-weighted mean losses
        model_ll = float(np.average(sub["model_log_loss"], weights=sub["n"]))
        market_ll = float(np.average(sub["market_log_loss"], weights=sub["n"]))
        model_br = float(np.average(sub["model_brier"], weights=sub["n"]))
        market_br = float(np.average(sub["market_brier"], weights=sub["n"]))
        pick_acc = float(sub["model_correct_sum"].sum() / n) if n else np.nan
        # Wilson-ish bootstrap CI from aggregate correct/n
        lo, hi = _accuracy_ci(int(sub["model_correct_sum"].sum()), n, config)
        rows.append(
            {
                "market": market,
                "live_weeks": weeks,
                "n": n,
                "cumulative_log_loss_gain": market_ll - model_ll,
                "cumulative_brier_gain": market_br - model_br,
                "live_pick_accuracy": pick_acc,
                "pick_accuracy_ci_lower": lo,
                "pick_accuracy_ci_upper": hi,
            }
        )
    return pd.DataFrame(rows)


def _accuracy_ci(correct: int, n: int, config: LivePromotionConfig) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(config.bootstrap_seed)
    draws = rng.binomial(n, correct / n, size=config.bootstrap_repetitions) / n
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def live_promotion_decision(
    cumulative: pd.DataFrame,
    *,
    drift_alarm_markets: set[str] | None = None,
    config: LivePromotionConfig | None = None,
) -> pd.DataFrame:
    """Apply the fixed live gate. A market may stake only after >= min live weeks
    with cumulative log-loss gain > 0 AND live pick-accuracy CI lower > 0.524.
    Any market with a drift alarm is demoted immediately.
    """

    config = config or LivePromotionConfig()
    drift_alarm_markets = drift_alarm_markets or set()
    rows = []
    for _, r in cumulative.iterrows():
        market = r["market"]
        enough = r["live_weeks"] >= config.minimum_live_weeks
        ll = r["cumulative_log_loss_gain"] > 0
        acc = np.isfinite(r["pick_accuracy_ci_lower"]) and r["pick_accuracy_ci_lower"] > config.breakeven
        drift = market in drift_alarm_markets
        if drift:
            decision, reason = "DEMOTE_TO_RETAIN_BASELINE", "DRIFT_ALARM"
        elif enough and ll and acc:
            decision, reason = "PROMOTE_TO_LIVE_STAKE", "LIVE_GATE_PASSED"
        else:
            missing = []
            if not enough:
                missing.append(f"weeks<{config.minimum_live_weeks}")
            if not ll:
                missing.append("log_loss_gain<=0")
            if not acc:
                missing.append("pick_acc_ci_lower<=0.524")
            decision, reason = "HOLD_PROVISIONAL_STAKE_ZERO", ";".join(missing)
        rows.append({"market": market, "live_decision": decision, "reason": reason,
                     "live_weeks": int(r["live_weeks"]),
                     "cumulative_log_loss_gain": r["cumulative_log_loss_gain"],
                     "pick_accuracy_ci_lower": r["pick_accuracy_ci_lower"]})
    return pd.DataFrame(rows)
