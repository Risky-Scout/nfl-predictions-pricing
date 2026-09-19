"""Regression proofs for the live OPEN-discovery outage.

THE PROVEN DEFECT. In the 2026 Week-2 STAGE captures, BetMGM had posted a
total for ``2026_02_IND_KC`` but no spread:

    spread_home_value = None      total_value      = 46.5
    spread_away_value = None      total_over_odds  = -108
    spread_home_odds  = None      total_under_odds = -110
    spread_away_odds  = None

That single row out of 125 raised ``BdlMarketBridgeError: spread_home_value is
missing`` out of ``build_bookmaker_quotes``, which aborted the whole capture.
``evaluate_live_market_source`` then reported the source unregistered,
``load_stage_quotes`` returned ``None``, and OPEN discovery reported all 16
games unobserved -- 124 perfectly good rows discarded because one book had not
priced one market yet.

Hermetic: ``tmp_path`` estates and synthetic captures built by the real capture
hashing helpers. No network, no provider, no server.
"""
from __future__ import annotations

import json

import pytest

from nfl_hybrid.data import bdl_market_bridge as bridge
from nfl_hybrid.evaluation import raw_market_reconstruction as rmr

from test_bdl_market_bridge import _odds_row, write_capture  # noqa: E402

# The live row, reproduced field for field.
TOTALS_ONLY = dict(
    spread_home_value=None,
    spread_away_value=None,
    spread_home_odds=None,
    spread_away_odds=None,
    total_value="46.5",
    total_over_odds=-108,
    total_under_odds=-110,
)


def _betmgm_totals_only(game=1):
    return _odds_row(game, "betmgm", **TOTALS_ONLY)


def _capture(tmp_path, rows, **kw):
    return write_capture(tmp_path, odds_pages=[rows], **kw)


def _quotes(tmp_path, rows, **kw):
    return bridge.build_bookmaker_quotes(
        bridge.validate_capture_manifest(_capture(tmp_path, rows, **kw))
    )


# ===========================================================================
# A. Mixed capture -- the exact production shape.
# ===========================================================================
def test_a_totals_only_row_no_longer_destroys_the_whole_capture(tmp_path):
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    quotes = _quotes(tmp_path, good + [_betmgm_totals_only()])

    assert not quotes.empty
    assert sorted(set(quotes["bookmaker_key"])) == ["caesars", "draftkings", "fanduel"]


def test_the_offending_row_is_excluded_whole(tmp_path):
    """Not partially built: no spread rows AND no total rows from that book,
    so a half-market can never reach the reconstruction."""
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    quotes = _quotes(tmp_path, good + [_betmgm_totals_only()])
    assert quotes[quotes["bookmaker_key"] == "betmgm"].empty


def test_the_valid_rows_survive_intact(tmp_path):
    """Four certified rows per usable book: spread home/away, total over/under."""
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    quotes = _quotes(tmp_path, good + [_betmgm_totals_only()])

    assert len(quotes) == 3 * 4
    for book in ("draftkings", "fanduel", "caesars"):
        rows = quotes[quotes["bookmaker_key"] == book]
        assert sorted(set(rows["market"])) == sorted([rmr.MARKET_SPREADS, rmr.MARKET_TOTALS])
        assert sorted(set(rows["outcome_key"])) == ["away", "home", "over", "under"]


def test_no_spread_is_fabricated_for_the_excluded_book(tmp_path):
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    quotes = _quotes(tmp_path, good + [_betmgm_totals_only()])
    spreads = quotes[quotes["market"] == rmr.MARKET_SPREADS]
    assert "betmgm" not in set(spreads["bookmaker_key"])


def test_no_total_is_salvaged_from_the_excluded_book(tmp_path):
    """The book DID post a usable total. It is still dropped -- the row is the
    unit, and keeping half of it would be a new policy."""
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    quotes = _quotes(tmp_path, good + [_betmgm_totals_only()])
    totals = quotes[quotes["market"] == rmr.MARKET_TOTALS]
    assert "betmgm" not in set(totals["bookmaker_key"])
    assert 46.5 not in set(totals["point"])


def test_the_exclusion_is_recorded_not_silent(tmp_path):
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    quotes = _quotes(tmp_path, good + [_betmgm_totals_only()])

    record = quotes.attrs[bridge.EXCLUDED_ROWS_ATTR]
    assert record["source_row_count"] == 4
    assert record["excluded_row_count"] == 1
    assert record["excluded_reason_counts"] == {"spread_home_value is missing": 1}
    assert record["excluded_examples"][0]["bookmaker_key"] == "betmgm"
    assert record["excluded_examples"][0]["reason"] == "spread_home_value is missing"


def test_a_clean_capture_records_no_exclusions(tmp_path):
    quotes = _quotes(tmp_path, [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")])
    record = quotes.attrs[bridge.EXCLUDED_ROWS_ATTR]
    assert record["excluded_row_count"] == 0
    assert record["excluded_reason_counts"] == {}
    assert record["excluded_examples"] == []


def test_the_example_list_is_bounded_but_the_counts_are_complete(tmp_path):
    """One pathological capture must not turn provenance into a log dump."""
    bad = [
        _odds_row(1, f"book{i}", **TOTALS_ONLY)
        for i in range(bridge.EXCLUDED_ROW_EXAMPLE_LIMIT + 4)
    ]
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    quotes = _quotes(tmp_path, good + bad)

    record = quotes.attrs[bridge.EXCLUDED_ROWS_ATTR]
    assert record["excluded_row_count"] == len(bad)
    assert len(record["excluded_examples"]) == bridge.EXCLUDED_ROW_EXAMPLE_LIMIT
    assert record["excluded_reason_counts"] == {"spread_home_value is missing": len(bad)}


# ===========================================================================
# B. All-invalid capture still fails closed, with the reason intact.
# ===========================================================================
def test_a_capture_where_every_row_fails_still_fails_closed(tmp_path):
    with pytest.raises(bridge.BdlMarketBridgeError) as excinfo:
        _quotes(tmp_path, [_betmgm_totals_only(), _odds_row(1, "fanduel", **TOTALS_ONLY)])
    assert "no priceable market" in str(excinfo.value)


def test_the_all_invalid_error_surfaces_the_underlying_row_reason(tmp_path):
    with pytest.raises(bridge.BdlMarketBridgeError, match="spread_home_value is missing"):
        _quotes(tmp_path, [_betmgm_totals_only()])


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(total_value=None), "total_value"),
        (dict(updated_at=None), "updated_at"),
        (dict(spread_home_value="-3.5", spread_away_value="4.0"), "not opposites"),
        (dict(spread_home_odds=None), "missing"),
    ],
)
def test_every_row_level_reason_survives_into_the_all_invalid_error(tmp_path, kwargs, expected):
    """The reasons the existing single-row tests assert on must still be
    reachable now that they arrive via the aggregate error."""
    with pytest.raises(bridge.BdlMarketBridgeError, match=expected):
        _quotes(tmp_path, [_odds_row(1, "draftkings", **kwargs)])


def test_a_missing_vendor_reason_also_survives(tmp_path):
    row = _odds_row(1, "draftkings")
    row["vendor"] = None
    with pytest.raises(bridge.BdlMarketBridgeError, match="no vendor"):
        _quotes(tmp_path, [row])


def test_the_all_invalid_error_counts_the_rows_it_rejected(tmp_path):
    with pytest.raises(bridge.BdlMarketBridgeError, match=r"every one of the 3 odds row\(s\)"):
        _quotes(tmp_path, [_odds_row(1, f"book{i}", **TOTALS_ONLY) for i in range(3)])


def test_an_empty_odds_page_still_fails_closed(tmp_path):
    """Unchanged: no rows at all is a different failure from all-rows-bad."""
    with pytest.raises(bridge.BdlMarketBridgeError, match="contained no odds rows"):
        _quotes(tmp_path, [])


# ===========================================================================
# C. Structural failures stay OUTSIDE the row boundary.
# ===========================================================================
def test_an_unknown_game_id_is_still_a_capture_level_failure(tmp_path):
    """The crosswalk runs before row certification, so an odds row naming a
    game absent from the capture's /games response still fails the capture --
    it is not downgraded to an excluded row."""
    with pytest.raises(bridge.BdlMarketBridgeError, match="no matching row in the capture"):
        _quotes(tmp_path, [_odds_row(1, "fanduel"), _odds_row(9999, "draftkings")])


def test_capture_validation_failures_are_untouched(tmp_path):
    manifest = _capture(tmp_path, [_odds_row(1, "draftkings")], status="INCOMPLETE")
    with pytest.raises(bridge.BdlMarketBridgeError, match="only a COMPLETE capture"):
        bridge.validate_capture_manifest(manifest)


def test_a_corrupt_manifest_hash_is_untouched(tmp_path):
    manifest = _capture(tmp_path, [_odds_row(1, "draftkings")], corrupt_manifest_hash=True)
    with pytest.raises(bridge.BdlMarketBridgeError, match="manifest hash mismatch"):
        bridge.validate_capture_manifest(manifest)


# ===========================================================================
# D. The production entry point accepts an otherwise priceable capture.
# ===========================================================================
def _three_book_card(tmp_path, *, with_bad_row: bool):
    rows = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    if with_bad_row:
        rows.append(_betmgm_totals_only())
    return _capture(tmp_path / "cap", rows)


def test_evaluate_live_market_source_accepts_a_capture_with_one_incomplete_row(tmp_path):
    """The end of the live chain: this is what returned registered=False and
    emptied open_observations."""
    from nfl_hybrid.production import run_2026 as prod

    evidence = prod.evaluate_live_market_source(
        _three_book_card(tmp_path, with_bad_row=True), artifact_root_path=tmp_path / "artifacts"
    )
    assert evidence["registered"] is True
    assert evidence["status"] == prod.LIVE_MARKET_OK
    assert evidence["source"] is not None
    assert not evidence["source"].quotes.empty


def test_the_exclusion_reaches_the_recorded_production_provenance(tmp_path):
    from nfl_hybrid.production import run_2026 as prod

    evidence = prod.evaluate_live_market_source(
        _three_book_card(tmp_path, with_bad_row=True), artifact_root_path=tmp_path / "artifacts"
    )
    excluded = evidence["provenance"]["excluded_rows"]
    assert excluded["excluded_row_count"] == 1
    assert excluded["excluded_reason_counts"] == {"spread_home_value is missing": 1}
    # Survives the parquet round-trip, which drops DataFrame.attrs.
    assert json.dumps(excluded)


def test_a_capture_whose_every_row_is_incomplete_is_still_rejected(tmp_path):
    from nfl_hybrid.production import run_2026 as prod

    manifest = _capture(
        tmp_path / "cap", [_odds_row(1, f"book{i}", **TOTALS_ONLY) for i in range(3)]
    )
    evidence = prod.evaluate_live_market_source(
        manifest, artifact_root_path=tmp_path / "artifacts"
    )
    assert evidence["registered"] is False
    assert evidence["status"] == prod.LIVE_MARKET_INVALID


# ===========================================================================
# E. Downstream coherence and the 3-book floor are unchanged.
# ===========================================================================
def test_the_three_book_floor_is_unchanged():
    assert rmr.MINIMUM_FRESH_COHERENT_BOOKS == 3


def test_an_excluded_row_reduces_the_book_count_it_would_have_contributed(tmp_path):
    """Exclusion is not a free pass: losing a book genuinely costs a book, and
    the downstream floor is what then decides priceability."""
    good = [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")]
    with_bad = _quotes(tmp_path / "a", good + [_betmgm_totals_only()])
    all_good = _quotes(tmp_path / "b", good + [_odds_row(1, "betmgm")])

    for market in (rmr.MARKET_SPREADS, rmr.MARKET_TOTALS):
        assert rmr.build_coherent_book_observations(with_bad, market)["bookmaker_key"].nunique() == 3
        assert rmr.build_coherent_book_observations(all_good, market)["bookmaker_key"].nunique() == 4


def test_two_usable_books_still_do_not_clear_the_three_book_floor(tmp_path):
    """Quotes now build, but priceability is still decided by the floor --
    not by this change."""
    rows = [_odds_row(1, "draftkings"), _odds_row(1, "fanduel")]
    rows += [_odds_row(1, "betmgm", **TOTALS_ONLY), _odds_row(1, "caesars", **TOTALS_ONLY)]
    quotes = _quotes(tmp_path / "q", rows)

    targets = quotes[["game_id"]].drop_duplicates().copy()
    targets["target_cutoff_utc"] = quotes["returned_snapshot_utc"].max()
    for market in (rmr.MARKET_SPREADS, rmr.MARKET_TOTALS):
        coherent = rmr.build_coherent_book_observations(quotes, market)
        assert coherent["bookmaker_key"].nunique() == 2
        result = rmr.reconstruct_market_at_cutoffs(coherent, targets, market=market)
        assert result.consensus.empty, "two books must not clear the 3-book floor"


def test_coherent_pairing_rules_are_unchanged(tmp_path):
    """A surviving book still needs both sides, opposite spread points and
    real prices -- the exclusion path did not relax any of it."""
    quotes = _quotes(tmp_path, [_odds_row(1, v) for v in ("draftkings", "fanduel", "caesars")])
    coherent = rmr.build_coherent_book_observations(quotes, rmr.MARKET_SPREADS)
    assert len(coherent) == 3
    assert (coherent["home_or_over_price_decimal"] > 1.0).all()
    assert (coherent["away_or_under_price_decimal"] > 1.0).all()
