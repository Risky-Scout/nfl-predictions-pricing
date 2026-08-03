"""Synthetic, private-data-free tests for the exact market-contract utility.

These prove the R2 benchmark integrity guarantees: a selected point is always an
actual quote, its probability is paired to that exact point, ties break
deterministically, conflicting/invalid quotes fail closed, and contract IDs are
deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.markets.exact_contract import (
    ContractError,
    build_real_closing_benchmark,
    canonical_decimal_text,
    make_market_contract_id,
)

ML_AGG = "exact_point_consensus_v1"


def _q(game_id, market, side, book, line, prob, mtk=60.0):
    return dict(
        game_id=game_id, market_type=market, outcome_side=side, bookmaker_id=book,
        minutes_to_kickoff=mtk, line_value=line, devig_probability=prob,
    )


def _ml(game_id, prob=0.55, books=("b1", "b2")):
    return [_q(game_id, "moneyline", "home", b, np.nan, prob) for b in books]


def _games(ids, season=2022):
    return pd.DataFrame(
        [dict(game_id=g, season=season, week=1, home_team_id="H", away_team_id="A") for g in ids]
    )


def _build(rows, ids=("g1",), season=2022):
    return build_real_closing_benchmark(pd.DataFrame(rows), _games(ids, season)).benchmark


# --- Test A: spread probability is paired to the selected spread ------------- #
def test_spread_probability_paired_to_selected_spread():
    rows = _ml("g1") + [
        _q("g1", "spread", "home", "A", -2.5, 0.51),
        _q("g1", "spread", "home", "B", -2.5, 0.53),
        _q("g1", "spread", "home", "C", -3.0, 0.47),
        _q("g1", "total", "over", "b1", 44.0, 0.50),
        _q("g1", "total", "over", "b2", 44.0, 0.52),
    ]
    b = _build(rows).iloc[0]
    assert b["closing_home_spread"] == -2.5           # an actual quoted point
    assert b["spread_consensus_books"] == 2
    assert b["spread_candidate_point_count"] == 2
    # median of ONLY the -2.5 quotes; the 0.47 at -3.0 is excluded
    assert b["market_cover_home_probability"] == pytest.approx(0.52)


# --- Test B: total probability is paired to the selected total --------------- #
def test_total_probability_paired_to_selected_total():
    rows = _ml("g1") + [
        _q("g1", "spread", "home", "b1", -3.0, 0.52),
        _q("g1", "spread", "home", "b2", -3.0, 0.50),
        _q("g1", "total", "over", "A", 44.0, 0.50),
        _q("g1", "total", "over", "B", 44.0, 0.52),
        _q("g1", "total", "over", "C", 44.5, 0.60),   # different point, excluded
    ]
    b = _build(rows).iloc[0]
    assert b["closing_total_line"] == 44.0
    assert b["total_consensus_books"] == 2
    assert b["market_over_probability"] == pytest.approx(0.51)  # median(0.50, 0.52)


# --- Test C: no synthetic median point --------------------------------------- #
def test_no_synthetic_median_point_spread_and_total():
    rows = _ml("g1") + [
        # equal book counts at -2.5 and -3.0; arithmetic median -2.75 is NOT quoted
        _q("g1", "spread", "home", "A", -2.5, 0.51),
        _q("g1", "spread", "home", "B", -2.5, 0.53),
        _q("g1", "spread", "home", "C", -3.0, 0.47),
        _q("g1", "spread", "home", "D", -3.0, 0.49),
        # equal book counts at 44 and 44.5; arithmetic median 44.25 is NOT quoted
        _q("g1", "total", "over", "A", 44.0, 0.50),
        _q("g1", "total", "over", "B", 44.0, 0.52),
        _q("g1", "total", "over", "C", 44.5, 0.48),
        _q("g1", "total", "over", "D", 44.5, 0.46),
    ]
    b = _build(rows).iloc[0]
    assert b["closing_home_spread"] in (-2.5, -3.0)
    assert b["closing_home_spread"] != -2.75          # never emitted
    assert b["closing_total_line"] in (44.0, 44.5)
    assert b["closing_total_line"] != 44.25
    # deterministic tie-break: equidistant -> numerically smaller point
    assert b["closing_home_spread"] == -3.0
    assert b["closing_total_line"] == 44.0


# --- Test D: deterministic tie-break ----------------------------------------- #
def test_deterministic_tiebreak_repeatable():
    rows = _ml("g1") + [
        # -2.5 and -3.5 each 2 books; median of points = -3.0; both distance 0.5
        _q("g1", "spread", "home", "A", -2.5, 0.51),
        _q("g1", "spread", "home", "B", -2.5, 0.53),
        _q("g1", "spread", "home", "C", -3.5, 0.47),
        _q("g1", "spread", "home", "D", -3.5, 0.49),
        _q("g1", "total", "over", "b1", 44.0, 0.50),
        _q("g1", "total", "over", "b2", 44.0, 0.52),
    ]
    first = _build(rows).iloc[0]["closing_home_spread"]
    second = _build(rows).iloc[0]["closing_home_spread"]
    assert first == second == -3.5                    # numerically smaller, stable


# --- Test E: conflicting duplicate quote ------------------------------------- #
def test_conflicting_duplicate_quote_raises():
    rows = _ml("g1") + [
        _q("g1", "spread", "home", "A", -2.5, 0.51),
        _q("g1", "spread", "home", "A", -3.0, 0.55),   # same book, different line -> conflict
        _q("g1", "total", "over", "b1", 44.0, 0.50),
        _q("g1", "total", "over", "b2", 44.0, 0.52),
    ]
    with pytest.raises(ContractError):
        _build(rows)


def test_identical_duplicate_quote_is_deduped():
    rows = _ml("g1") + [
        _q("g1", "spread", "home", "A", -2.5, 0.51),
        _q("g1", "spread", "home", "A", -2.5, 0.51),   # exact duplicate -> collapse
        _q("g1", "spread", "home", "B", -2.5, 0.53),
        _q("g1", "total", "over", "b1", 44.0, 0.50),
        _q("g1", "total", "over", "b2", 44.0, 0.52),
    ]
    b = _build(rows).iloc[0]
    assert b["spread_consensus_books"] == 2           # A counted once


# --- Test F: invalid probabilities and points fail closed -------------------- #
@pytest.mark.parametrize(
    "bad",
    [
        _q("g1", "spread", "home", "X", -2.5, np.nan),
        _q("g1", "spread", "home", "X", -2.5, np.inf),
        _q("g1", "spread", "home", "X", -2.5, 0.0),
        _q("g1", "spread", "home", "X", -2.5, 1.0),
        _q("g1", "spread", "home", "X", -2.5, -0.1),
        _q("g1", "spread", "home", "X", -2.5, 1.1),
        _q("g1", "spread", "home", "X", np.nan, 0.5),   # missing spread point
        _q("g1", "total", "over", "X", np.inf, 0.5),    # nonfinite total point
    ],
)
def test_invalid_quote_values_fail_closed(bad):
    rows = _ml("g1") + [
        _q("g1", "spread", "home", "A", -2.5, 0.51),
        _q("g1", "spread", "home", "B", -2.5, 0.53),
        _q("g1", "total", "over", "b1", 44.0, 0.50),
        _q("g1", "total", "over", "b2", 44.0, 0.52),
        bad,
    ]
    with pytest.raises(ContractError):
        _build(rows)


# --- Test G: deterministic contract IDs -------------------------------------- #
def _cid(**kw):
    base = dict(
        game_id="g1", market_type="spread", outcome_side="home", line_value=-2.5,
        market_source="REAL-CLOSING", snapshot_minutes_to_kickoff=60.0,
        aggregation_method=ML_AGG,
    )
    base.update(kw)
    return make_market_contract_id(**base)


def test_contract_id_determinism():
    assert _cid() == _cid()                                   # identical payload
    assert _cid() != _cid(line_value=-3.0)                    # different point
    assert _cid() != _cid(snapshot_minutes_to_kickoff=30.0)   # different snapshot
    assert _cid() != _cid(market_type="total", outcome_side="over")  # different market
    assert _cid() != _cid(game_id="g2")                       # different game
    assert _cid() != _cid(market_source="REFERENCE-LINE PROXY")
    # moneyline null line is stable
    ml1 = _cid(market_type="moneyline", outcome_side="home", line_value=None)
    ml2 = _cid(market_type="moneyline", outcome_side="home", line_value=None)
    assert ml1 == ml2
    # -0.0 and 0.0 canonicalize identically
    assert _cid(line_value=-0.0) == _cid(line_value=0.0)


def test_canonical_decimal_text_cases():
    assert canonical_decimal_text(-0.0) == "0"
    assert canonical_decimal_text(0.0) == "0"
    assert canonical_decimal_text(-3.0) == "-3"
    assert canonical_decimal_text(44.5) == "44.5"
    assert canonical_decimal_text(None) is None
    assert canonical_decimal_text(float("nan")) is None
    assert canonical_decimal_text(float("inf")) is None
    # two genuinely distinct points must not collapse
    assert canonical_decimal_text(2.5) != canonical_decimal_text(2.75)
