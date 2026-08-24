import pandas as pd
import pytest

from nfl_hybrid.evaluation import raw_market_reconstruction as rmr
from nfl_hybrid.odds_history import _no_vig


def _quote_row(**kwargs):
    base = {
        "game_id": "G1", "bookmaker_key": "bk1",
        "returned_snapshot_utc": pd.Timestamp("2024-09-01T00:00:00Z"),
        "market_last_update": pd.Timestamp("2024-09-01T00:00:00Z"),
        "market": "spreads", "outcome_key": "home",
        "price_decimal": 1.91, "point": -3.5,
    }
    base.update(kwargs)
    return base


def test_coherent_spread_quote_both_sides_required():
    quotes = pd.DataFrame(
        [
            _quote_row(outcome_key="home", point=-3.5, price_decimal=1.91),
            _quote_row(outcome_key="away", point=3.5, price_decimal=1.91),
        ]
    )
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    assert len(coherent) == 1
    assert coherent.iloc[0]["line"] == -3.5


def test_coherent_spread_quote_missing_side_excluded():
    quotes = pd.DataFrame([_quote_row(outcome_key="home", point=-3.5, price_decimal=1.91)])
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    assert coherent.empty


def test_coherent_spread_quote_mismatched_points_excluded():
    quotes = pd.DataFrame(
        [
            _quote_row(outcome_key="home", point=-3.5, price_decimal=1.91),
            _quote_row(outcome_key="away", point=4.0, price_decimal=1.91),
        ]
    )
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    assert coherent.empty


def test_no_cross_snapshot_side_pairing():
    quotes = pd.DataFrame(
        [
            _quote_row(outcome_key="home", point=-3.5, price_decimal=1.91, returned_snapshot_utc=pd.Timestamp("2024-09-01T00:00:00Z")),
            _quote_row(outcome_key="away", point=3.5, price_decimal=1.91, returned_snapshot_utc=pd.Timestamp("2024-09-01T06:00:00Z")),
        ]
    )
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    assert coherent.empty  # different snapshot events -- structurally never paired


def test_coherent_total_quote():
    quotes = pd.DataFrame(
        [
            _quote_row(market="totals", outcome_key="over", point=45.5, price_decimal=1.95),
            _quote_row(market="totals", outcome_key="under", point=45.5, price_decimal=1.87),
        ]
    )
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_TOTALS)
    assert len(coherent) == 1
    assert coherent.iloc[0]["line"] == 45.5


def test_coherent_total_quote_mismatched_points_excluded():
    quotes = pd.DataFrame(
        [
            _quote_row(market="totals", outcome_key="over", point=45.5, price_decimal=1.95),
            _quote_row(market="totals", outcome_key="under", point=46.0, price_decimal=1.87),
        ]
    )
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_TOTALS)
    assert coherent.empty


def test_price_at_or_below_one_excluded():
    quotes = pd.DataFrame(
        [
            _quote_row(outcome_key="home", point=-3.5, price_decimal=1.0),
            _quote_row(outcome_key="away", point=3.5, price_decimal=1.91),
        ]
    )
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    assert coherent.empty


def _coherent_row(game_id, bookmaker_key, returned, last_update, line, price_a, price_b):
    home_p, away_p, hold = _no_vig(price_a, price_b)
    return {
        "game_id": game_id, "bookmaker_key": bookmaker_key,
        "returned_snapshot_utc": pd.Timestamp(returned), "market_last_update": pd.Timestamp(last_update),
        "line": line, "home_or_over_price_decimal": price_a, "away_or_under_price_decimal": price_b,
        "home_or_over_novig_probability": home_p, "away_or_under_novig_probability": away_p, "hold": hold,
    }


def test_minimum_three_fresh_books_required():
    cutoff = pd.Timestamp("2024-09-10T12:00:00Z")
    coherent = pd.DataFrame(
        [
            _coherent_row("G1", "bk1", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.5, 1.91, 1.91),
            _coherent_row("G1", "bk2", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.0, 1.95, 1.87),
        ]
    )
    targets = pd.DataFrame([{"game_id": "G1", "target_cutoff_utc": cutoff}])
    result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market="ATS")
    assert result.coverage["market_ready"] == 0
    assert result.consensus.empty


def test_stale_book_excluded_from_final_count_but_counted_before_age_filter():
    cutoff = pd.Timestamp("2024-09-10T12:00:00Z")
    coherent = pd.DataFrame(
        [
            _coherent_row("G1", "bk1", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.5, 1.91, 1.91),
            _coherent_row("G1", "bk2", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.0, 1.95, 1.87),
            # bk3's only observation is 49 hours stale -- ordering-valid, but must not count toward
            # the final >=3 gate (freshness is applied BEFORE counting books).
            _coherent_row("G1", "bk3", "2024-09-08T11:00:00Z", "2024-09-08T11:00:00Z", -3.0, 1.90, 1.90),
        ]
    )
    targets = pd.DataFrame([{"game_id": "G1", "target_cutoff_utc": cutoff}])
    result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market="ATS")
    assert result.coverage["ge3_before_age_filter"] == 1
    assert result.coverage["market_ready"] == 0
    assert result.consensus.empty


def test_market_last_update_after_returned_snapshot_excluded():
    cutoff = pd.Timestamp("2024-09-10T12:00:00Z")
    coherent = pd.DataFrame(
        [
            _coherent_row("G1", "bk1", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.5, 1.91, 1.91),
            _coherent_row("G1", "bk2", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.0, 1.95, 1.87),
            # bk3's market_last_update is AFTER its own returned_snapshot_utc -- invalid ordering,
            # excluded even though it would otherwise be a fresh, coherent, on-time observation.
            _coherent_row("G1", "bk3", "2024-09-09T12:00:00Z", "2024-09-09T13:00:00Z", -3.0, 1.90, 1.90),
        ]
    )
    targets = pd.DataFrame([{"game_id": "G1", "target_cutoff_utc": cutoff}])
    result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market="ATS")
    assert result.coverage["ge3_before_age_filter"] == 0
    assert result.coverage["market_ready"] == 0


def test_returned_snapshot_after_cutoff_excluded():
    cutoff = pd.Timestamp("2024-09-10T12:00:00Z")
    coherent = pd.DataFrame(
        [
            _coherent_row("G1", "bk1", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.5, 1.91, 1.91),
            _coherent_row("G1", "bk2", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.0, 1.95, 1.87),
            # bk3's snapshot arrived AFTER the cutoff -- must never be used to price that cutoff.
            _coherent_row("G1", "bk3", "2024-09-10T13:00:00Z", "2024-09-10T13:00:00Z", -3.0, 1.90, 1.90),
        ]
    )
    targets = pd.DataFrame([{"game_id": "G1", "target_cutoff_utc": cutoff}])
    result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market="ATS")
    assert result.coverage["coherent_leq_cutoff"] == 1  # game has 2 books at/before cutoff (bk3 excluded)
    assert result.coverage["market_ready"] == 0


def test_freshest_per_book_selected_and_consensus_uses_actual_prices():
    cutoff = pd.Timestamp("2024-09-10T12:00:00Z")
    coherent = pd.DataFrame(
        [
            # bk1 has two valid observations -- the freshest one (closer to cutoff) must be used.
            _coherent_row("G1", "bk1", "2024-09-08T12:00:00Z", "2024-09-08T12:00:00Z", -2.5, 1.80, 2.05),
            _coherent_row("G1", "bk1", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.5, 1.91, 1.91),
            _coherent_row("G1", "bk2", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -3.0, 1.95, 1.87),
            _coherent_row("G1", "bk3", "2024-09-09T12:00:00Z", "2024-09-09T12:00:00Z", -4.0, 2.10, 1.75),
        ]
    )
    targets = pd.DataFrame([{"game_id": "G1", "target_cutoff_utc": cutoff}])
    result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market="ATS")
    assert result.coverage["market_ready"] == 1
    row = result.consensus.iloc[0]
    assert row["eligible_books"] == 3
    # Median of bk1's FRESHEST line (-3.5), bk2 (-3.0), bk3 (-4.0) -- never bk1's stale -2.5.
    assert row["consensus_line"] == -3.5
    # Real per-book median, not a fabricated synthetic constant.
    assert 0.0 < row["consensus_novig_probability"] < 1.0
    assert set(row["bookmaker_keys"]) == {"bk1", "bk2", "bk3"}


def test_age_boundary_48_hours_inclusive():
    cutoff = pd.Timestamp("2024-09-10T12:00:00Z")
    exactly_48h_before = cutoff - pd.Timedelta(hours=48)
    coherent_ok = pd.DataFrame(
        [
            _coherent_row("G1", "bk1", exactly_48h_before, exactly_48h_before, -3.5, 1.91, 1.91),
            _coherent_row("G1", "bk2", exactly_48h_before, exactly_48h_before, -3.0, 1.95, 1.87),
            _coherent_row("G1", "bk3", exactly_48h_before, exactly_48h_before, -4.0, 2.10, 1.75),
        ]
    )
    targets = pd.DataFrame([{"game_id": "G1", "target_cutoff_utc": cutoff}])
    result_ok = rmr.reconstruct_market_at_cutoffs(coherent_ok, targets, market="ATS")
    assert result_ok.coverage["market_ready"] == 1

    just_over_48h_before = cutoff - pd.Timedelta(hours=48, minutes=1)
    coherent_stale = pd.DataFrame(
        [
            _coherent_row("G1", "bk1", just_over_48h_before, just_over_48h_before, -3.5, 1.91, 1.91),
            _coherent_row("G1", "bk2", just_over_48h_before, just_over_48h_before, -3.0, 1.95, 1.87),
            _coherent_row("G1", "bk3", just_over_48h_before, just_over_48h_before, -4.0, 2.10, 1.75),
        ]
    )
    result_stale = rmr.reconstruct_market_at_cutoffs(coherent_stale, targets, market="ATS")
    assert result_stale.coverage["market_ready"] == 0


def test_eligible_count_matches_target_population_regardless_of_market_coverage():
    coherent = pd.DataFrame(columns=["game_id", "bookmaker_key", "returned_snapshot_utc", "market_last_update", "line", "home_or_over_novig_probability"])
    targets = pd.DataFrame(
        [
            {"game_id": "G1", "target_cutoff_utc": pd.Timestamp("2024-09-10T12:00:00Z")},
            {"game_id": "G2", "target_cutoff_utc": pd.Timestamp("2024-09-10T12:00:00Z")},
        ]
    )
    result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market="ATS")
    assert result.coverage == {
        "eligible": 2, "coherent_leq_cutoff": 0, "ge3_before_age_filter": 0, "market_ready": 0,
        "minimum_books": 3, "max_age_hours": 48.0,
    }
