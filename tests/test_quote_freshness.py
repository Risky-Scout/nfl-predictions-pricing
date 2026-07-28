from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.markets.quote_freshness import (
    QuoteFreshnessConfig,
    _normalize_outcome,
    _robust_outlier_mask,
    build_book_market_audit,
    build_freshness_consensus,
    prepare_quote_rows,
)


def _row(
    *,
    book: str,
    market: str,
    outcome: str,
    price: float,
    point: float | None,
    market_update: str = "2023-09-10T16:44:00Z",
) -> dict[str, object]:
    return {
        "game_id": "2023_01_A_B",
        "season": 2023,
        "week": 1,
        "kickoff_utc": "2023-09-10T17:00:00Z",
        "horizon": "closing_t10",
        "requested_snapshot_utc": "2023-09-10T16:50:00Z",
        "returned_snapshot_utc": "2023-09-10T16:45:00Z",
        "event_id": "event",
        "bookmaker_key": book,
        "bookmaker_title": book.title(),
        "bookmaker_last_update": market_update,
        "market": market,
        "market_last_update": market_update,
        "outcome_key": outcome,
        "price_decimal": price,
        "price_american": 100,
        "point": point,
    }


def _three_book_spreads() -> pd.DataFrame:
    rows = []
    for book, home_price, away_price in (
        ("a", 1.91, 1.91),
        ("b", 1.95, 1.87),
        ("c", 1.90, 1.92),
    ):
        rows.extend(
            [
                _row(
                    book=book,
                    market="spreads",
                    outcome="home",
                    price=home_price,
                    point=-3.0,
                ),
                _row(
                    book=book,
                    market="spreads",
                    outcome="away",
                    price=away_price,
                    point=3.0,
                ),
            ]
        )
    return pd.DataFrame(rows)


def test_outcome_normalization():
    assert _normalize_outcome("Home Team") == "home"
    assert _normalize_outcome("DRAW") == "tie"
    assert _normalize_outcome("O") == "over"


def test_fresh_complete_spreads_are_eligible():
    config = QuoteFreshnessConfig()
    prepared = prepare_quote_rows(_three_book_spreads(), config)
    books = build_book_market_audit(prepared, config)
    assert books["eligible"].all()

    consensus = build_freshness_consensus(books, config)
    assert len(consensus) == 1
    assert consensus.iloc[0]["eligible_books"] == 3
    assert consensus.iloc[0]["consensus_line"] == pytest.approx(-3.0)


def test_stale_book_is_excluded():
    rows = _three_book_spreads()
    rows.loc[
        rows["bookmaker_key"].eq("a"),
        ["market_last_update", "bookmaker_last_update"],
    ] = "2023-09-10T16:20:00Z"

    prepared = prepare_quote_rows(rows, QuoteFreshnessConfig())
    books = build_book_market_audit(
        prepared,
        QuoteFreshnessConfig(),
    )
    stale = books[books["bookmaker_key"].eq("a")].iloc[0]
    assert not stale["eligible"]
    assert "STALE_MARKET_QUOTE" in stale["exclusion_reason"]


def test_future_update_is_excluded():
    rows = _three_book_spreads()
    rows.loc[
        rows["bookmaker_key"].eq("a"),
        ["market_last_update", "bookmaker_last_update"],
    ] = "2023-09-10T16:46:00Z"

    prepared = prepare_quote_rows(rows, QuoteFreshnessConfig())
    books = build_book_market_audit(
        prepared,
        QuoteFreshnessConfig(),
    )
    future = books[books["bookmaker_key"].eq("a")].iloc[0]
    assert not future["eligible"]
    assert "FUTURE_MARKET_UPDATE" in future["exclusion_reason"]


def test_incomplete_market_is_excluded():
    rows = _three_book_spreads()
    rows = rows[
        ~(
            rows["bookmaker_key"].eq("a")
            & rows["outcome_key"].eq("away")
        )
    ]
    prepared = prepare_quote_rows(rows, QuoteFreshnessConfig())
    books = build_book_market_audit(
        prepared,
        QuoteFreshnessConfig(),
    )
    incomplete = books[books["bookmaker_key"].eq("a")].iloc[0]
    assert not incomplete["eligible"]
    assert "INCOMPLETE_MARKET" in incomplete["exclusion_reason"]


def test_inconsistent_spread_pair_is_excluded():
    rows = _three_book_spreads()
    rows.loc[
        rows["bookmaker_key"].eq("a")
        & rows["outcome_key"].eq("away"),
        "point",
    ] = 2.0
    prepared = prepare_quote_rows(rows, QuoteFreshnessConfig())
    books = build_book_market_audit(
        prepared,
        QuoteFreshnessConfig(),
    )
    bad = books[books["bookmaker_key"].eq("a")].iloc[0]
    assert not bad["eligible"]
    assert "INCONSISTENT_SPREAD_PAIR" in bad["exclusion_reason"]


def test_robust_outlier_detection():
    values = pd.Series([0.50, 0.50, 0.50, 0.50, 0.80])
    mask = _robust_outlier_mask(
        values,
        threshold=6.0,
        zero_mad_tolerance=0.10,
    )
    assert mask.tolist() == [False, False, False, False, True]


def test_consensus_requires_minimum_books():
    config = QuoteFreshnessConfig(minimum_books=3)
    rows = _three_book_spreads()
    rows = rows[~rows["bookmaker_key"].eq("c")]
    prepared = prepare_quote_rows(rows, config)
    books = build_book_market_audit(prepared, config)
    consensus = build_freshness_consensus(books, config)
    assert consensus.empty
