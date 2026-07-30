import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.operations.season_loop import (
    LivePromotionConfig,
    cumulative_live_scorecard,
    live_promotion_decision,
    log_week_predictions,
    score_resolved_week,
)


def _card():
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2026, 2026],
            "week": [1, 1],
            "market": ["moneyline", "moneyline"],
            "model_probability": [0.6, 0.4],
            "market_fair_probability": [0.55, 0.45],
            "should_bet": [False, False],
        }
    )


def test_log_is_immutable(tmp_path):
    p = log_week_predictions(_card(), output_dir=tmp_path, season=2026, week=1, logged_at_utc="2026-09-05T12:00:00Z")
    assert p.exists()
    logged = pd.read_csv(p)
    assert "logged_at_utc" in logged.columns
    # second write must refuse to overwrite
    with pytest.raises(FileExistsError):
        log_week_predictions(_card(), output_dir=tmp_path, season=2026, week=1, logged_at_utc="x")


def test_score_resolved_week():
    logged = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2026, 2026],
            "week": [1, 1],
            "market": ["ats", "ats"],
            "model_probability": [0.7, 0.3],
            "market_fair_probability": [0.5, 0.5],
        }
    )
    resolved = pd.DataFrame({"game_id": ["g1", "g2"], "home_win": [1, 0], "home_cover": [1, 0], "over": [1, 0]})
    scored = score_resolved_week(logged, resolved)
    row = scored[scored["market"] == "ats"].iloc[0]
    assert row["n"] == 2
    # model picked both correctly (0.7>0.5 and 0.3<0.5 with outcomes 1,0)
    assert row["model_correct_sum"] == 2.0


def _weekly_with_edge(weeks, correct_per_week, n_per_week=16):
    rows = []
    for w in range(1, weeks + 1):
        rows.append(
            {
                "season": 2026, "week": w, "market": "ats", "n": n_per_week,
                "model_log_loss": 0.66, "market_log_loss": 0.69,  # model better
                "model_brier": 0.24, "market_brier": 0.25,
                "model_correct_sum": correct_per_week,
            }
        )
    return pd.DataFrame(rows)


def test_live_gate_holds_before_8_weeks():
    weekly = _weekly_with_edge(4, correct_per_week=11)
    cum = cumulative_live_scorecard(weekly)
    dec = live_promotion_decision(cum)
    assert dec.iloc[0]["live_decision"] == "HOLD_PROVISIONAL_STAKE_ZERO"
    assert "weeks<8" in dec.iloc[0]["reason"]


def test_live_gate_promotes_with_sustained_edge():
    # 10 weeks, ~69% pick accuracy (11/16), model beats market on log-loss
    weekly = _weekly_with_edge(10, correct_per_week=11)
    cum = cumulative_live_scorecard(weekly)
    dec = live_promotion_decision(cum)
    assert dec.iloc[0]["live_decision"] == "PROMOTE_TO_LIVE_STAKE"


def test_drift_alarm_demotes_immediately():
    weekly = _weekly_with_edge(10, correct_per_week=11)
    cum = cumulative_live_scorecard(weekly)
    dec = live_promotion_decision(cum, drift_alarm_markets={"ats"})
    assert dec.iloc[0]["live_decision"] == "DEMOTE_TO_RETAIN_BASELINE"
    assert dec.iloc[0]["reason"] == "DRIFT_ALARM"
