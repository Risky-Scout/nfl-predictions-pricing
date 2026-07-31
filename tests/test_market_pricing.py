import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.pricing.market_pricing import price_one_market, price_game_markets
from nfl_hybrid.pricing.weekly import QuoteAgeConfig


class _Artifact:
    devig_consensus = {
        "moneyline_devig": "proportional", "moneyline_consensus": "equal_mean",
        "ats_devig": "proportional", "ats_consensus": "equal_mean",
        "total_devig": "proportional", "total_consensus": "equal_mean",
    }


def _quote(book, side, line, p, market="spread", age_min=5.0, mtk_min=5000.0):
    now = pd.Timestamp("2026-09-10T00:00:00Z")
    return {
        "game_id": "g1", "provider_event_id": "e1", "bookmaker_id": book,
        "market_type": market, "outcome_side": side, "line_value": line,
        "raw_implied_probability": p,
        "bookmaker_last_update_utc": now - pd.Timedelta(minutes=age_min),
        "commence_time_utc": now + pd.Timedelta(minutes=mtk_min),
    }


AS_OF = "2026-09-10T00:00:00Z"


def _spread_book(book, line=-3.0, p_home=0.52, p_away=0.52, **kw):
    return [_quote(book, "home", line, p_home, **kw), _quote(book, "away", -line, p_away, **kw)]


def test_uses_configured_methods_and_prices():
    quotes = pd.DataFrame(sum((_spread_book(b) for b in ("b1", "b2", "b3")), []))
    mp = price_one_market(quotes, market="spread", as_of_utc=AS_OF,
                          devig_method="proportional", consensus_method="equal_mean")
    assert mp.status == "VALID_MARKET"
    assert mp.audit["devig_method"] == "proportional"
    assert mp.audit["consensus_method"] == "equal_mean"
    assert mp.fair_probability == pytest.approx(0.5, abs=1e-6)  # symmetric prices


def test_stale_book_excluded_and_counted():
    quotes = pd.DataFrame(
        _spread_book("fresh1") + _spread_book("fresh2") + _spread_book("fresh3")
        + _spread_book("stale", age_min=90.0, mtk_min=60.0)  # within 2h of kickoff, 90-min old
    )
    mp = price_one_market(quotes, market="spread", as_of_utc=AS_OF,
                          devig_method="proportional", consensus_method="equal_mean")
    assert mp.audit["books_retained"] == 3
    assert mp.audit["books_rejected_stale"] >= 1


def test_one_fresh_book_cannot_mask_stale_books():
    # 1 fresh + 2 stale near kickoff -> only 1 retained -> not actionable
    quotes = pd.DataFrame(
        _spread_book("fresh", age_min=2.0, mtk_min=30.0)
        + _spread_book("stale1", age_min=60.0, mtk_min=30.0)
        + _spread_book("stale2", age_min=45.0, mtk_min=30.0)
    )
    mp = price_one_market(quotes, market="spread", as_of_utc=AS_OF,
                          devig_method="proportional", consensus_method="equal_mean")
    assert mp.audit["books_retained"] == 1
    assert mp.status == "INDICATIVE_ONLY"  # below minimum coverage


def test_different_points_not_averaged():
    # b1,b2,b3 at -3 ; b4 at -2.5. Reference point = mode (-3) ; b4 excluded.
    quotes = pd.DataFrame(
        _spread_book("b1", line=-3.0) + _spread_book("b2", line=-3.0)
        + _spread_book("b3", line=-3.0) + _spread_book("b4", line=-2.5)
    )
    mp = price_one_market(quotes, market="spread", as_of_utc=AS_OF,
                          devig_method="proportional", consensus_method="equal_mean")
    assert mp.reference_point == 3.0
    assert mp.audit["books_retained"] == 3  # the -2.5 book is not mixed in


def test_incomplete_pair_rejected():
    # b1 has only the home side -> no valid pair
    quotes = pd.DataFrame([_quote("b1", "home", -3.0, 0.52)] + _spread_book("b2") + _spread_book("b3"))
    mp = price_one_market(quotes, market="spread", as_of_utc=AS_OF,
                          devig_method="proportional", consensus_method="equal_mean")
    assert mp.audit["books_retained"] == 2


def test_post_kickoff_and_future_quotes_rejected():
    # post-kickoff (mtk negative) and future-dated (negative age)
    quotes = pd.DataFrame(
        _spread_book("live", mtk_min=-10.0) + _spread_book("future", age_min=-30.0)
        + _spread_book("ok1") + _spread_book("ok2") + _spread_book("ok3")
    )
    mp = price_one_market(quotes, market="spread", as_of_utc=AS_OF,
                          devig_method="proportional", consensus_method="equal_mean")
    assert mp.audit["books_retained"] == 3


def test_calibration_serve_parity_on_fixture():
    # The canonical serve path and a calibration-style replay must compute the same
    # de-vigged, equal-mean consensus from the SAME quote fixture (shared primitives).
    from nfl_hybrid.pricing.devig import devig_pair
    quotes = pd.DataFrame(
        _spread_book("b1", p_home=0.53, p_away=0.51)
        + _spread_book("b2", p_home=0.52, p_away=0.52)
        + _spread_book("b3", p_home=0.54, p_away=0.50)
    )
    serve = price_one_market(quotes, market="spread", as_of_utc=AS_OF,
                             devig_method="proportional", consensus_method="equal_mean")
    # calibration-style: de-vig each book's pair, equal-mean the home fair probs
    fair = []
    for _, bg in quotes.groupby("bookmaker_id"):
        a = bg[bg["outcome_side"] == "home"]["raw_implied_probability"].iloc[0]
        b = bg[bg["outcome_side"] == "away"]["raw_implied_probability"].iloc[0]
        fair.append(devig_pair(float(a), float(b), "proportional")[0])
    calib = float(np.mean(fair))
    assert serve.fair_probability == pytest.approx(calib, abs=1e-12)
    assert serve.reference_point == 3.0
    assert serve.audit["devig_method_executed"] == "proportional"


def test_independent_status_per_market():
    # spread has 3 books; total has only 1 -> statuses/counts must be independent
    quotes = pd.DataFrame(
        sum((_spread_book(b) for b in ("b1", "b2", "b3")), [])
        + [_quote("b1", "over", 45.0, 0.52, market="total"),
           _quote("b1", "under", 45.0, 0.52, market="total")]
    )
    prices = price_game_markets(quotes, as_of_utc=AS_OF, artifact=_Artifact())
    assert prices["spread"].status == "VALID_MARKET"
    assert prices["spread"].audit["books_retained"] == 3
    assert prices["total"].status == "INDICATIVE_ONLY"  # only 1 book
    assert prices["total"].audit["books_retained"] == 1
    assert prices["moneyline"].status == "INCOMPLETE_MARKET"
    # spread's book count is NOT reused for total/moneyline
    assert prices["total"].audit["books_retained"] != prices["spread"].audit["books_retained"]


def test_price_game_markets_uses_artifact_methods():
    quotes = pd.DataFrame(
        sum((_spread_book(b) for b in ("b1", "b2", "b3")), [])
        + sum(([_quote(b, "over", 45.0, 0.52, market="total"), _quote(b, "under", 45.0, 0.52, market="total")]
               for b in ("b1", "b2", "b3")), [])
    )
    prices = price_game_markets(quotes, as_of_utc=AS_OF, artifact=_Artifact())
    assert prices["spread"].audit["consensus_method"] == "equal_mean"
    assert prices["total"].status == "VALID_MARKET"
    assert prices["moneyline"].status == "INCOMPLETE_MARKET"  # no moneyline quotes supplied
