import time

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.pricing.betting_card import build_betting_card
from nfl_hybrid.pricing.weekly import (
    QuoteAgeConfig,
    WeeklyRunError,
    quote_is_fresh,
    validate_lines,
    validate_probability_identity,
    validate_starter_sums,
)


def test_quote_age_rejection_near_and_far_kickoff():
    cfg = QuoteAgeConfig()
    # within 2h of kickoff: 20-min-old quote is stale
    assert not quote_is_fresh(20.0, minutes_to_kickoff=60.0, cfg=cfg)
    assert quote_is_fresh(10.0, minutes_to_kickoff=60.0, cfg=cfg)
    # far from kickoff: 20-min-old quote is fine, 25h-old is not
    assert quote_is_fresh(20.0, minutes_to_kickoff=5000.0, cfg=cfg)
    assert not quote_is_fresh(25 * 60.0, minutes_to_kickoff=5000.0, cfg=cfg)
    # non-finite age is never fresh
    assert not quote_is_fresh(float("nan"), 60.0, cfg)


def _lines(status="PRICED", n=2):
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "home_team": ["KC", "BUF"][:n], "away_team": ["DEN", "NYJ"][:n],
        "status": [status] * n,
    })


def test_validate_lines_critical_failures():
    with pytest.raises(WeeklyRunError, match="duplicate"):
        bad = _lines(n=2); bad["game_id"] = ["g", "g"]
        validate_lines(bad)
    with pytest.raises(WeeklyRunError, match="team mapping"):
        bad = _lines(n=2); bad.loc[0, "home_team"] = np.nan
        validate_lines(bad)
    with pytest.raises(WeeklyRunError, match="no PRICED"):
        validate_lines(_lines(status="UNPRICED-AWAITING-LINES"), require_priced=True)
    # all-priced passes
    validate_lines(_lines())


def test_validate_starter_sums():
    good = pd.DataFrame({"team_id": ["KC", "KC"], "starter_probability": [0.9, 0.1]})
    validate_starter_sums(good)
    bad = pd.DataFrame({"team_id": ["KC", "KC"], "starter_probability": [0.9, 0.3]})
    with pytest.raises(WeeklyRunError, match="sum to 1"):
        validate_starter_sums(bad)


def _games():
    return pd.DataFrame({
        "game_id": ["g1", "g2"], "season": [2026, 2026], "week": ["1", "1"],
        "home_team": ["KC", "BUF"], "away_team": ["DEN", "NYJ"],
        "home_spread": [-3.0, -6.5], "total_line": [42.5, 48.5],
    })


def test_probability_identity_holds_on_real_card():
    readiness = {"moneyline": "RETAIN_BASELINE", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}
    card = build_betting_card(_games(), readiness_by_market=readiness)
    validate_probability_identity(card)  # must not raise


def test_deterministic_card_same_inputs():
    readiness = {"moneyline": "RETAIN_BASELINE", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}
    a = build_betting_card(_games(), readiness_by_market=readiness)
    b = build_betting_card(_games(), readiness_by_market=readiness)
    pd.testing.assert_frame_equal(a, b)


def test_pricing_runtime_regression():
    # pricing a small slate must be well under the 5s/week budget
    readiness = {"moneyline": "RETAIN_BASELINE", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}
    g = _games()
    build_betting_card(g, readiness_by_market=readiness)  # warm
    t = time.time()
    for _ in range(10):
        build_betting_card(g, readiness_by_market=readiness)
    assert (time.time() - t) / 10 < 1.0
