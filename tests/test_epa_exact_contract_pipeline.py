"""Pre-matrix contract-binding tests (private-data-free).

Prove the closing contract is bound BEFORE target construction: labels and lines
are the closing contract, original references stay auditable, uncovered dev games
are audited (not silently proxied or inner-joined away), and pre-closing rows are
a clearly-labelled training-only proxy.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.augmented_matrix import _outcome_columns
import nfl_hybrid.markets.exact_contract as exact_contract
from nfl_hybrid.markets.exact_contract import (
    AGGREGATION_METHOD,
    MARKET_SOURCE_CLOSING,
    MARKET_SOURCE_PROXY,
    NO_BENCHMARK_ROW,
    bind_closing_contract,
    make_market_contract_id,
    validate_closing_era_contract_states,
)

# select_tournament_matrix + run_walk_forward strict source are exercised here too.
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_epa_tournament",
    Path(__file__).resolve().parents[1] / "scripts" / "run_epa_tournament.py",
)
run_epa_tournament = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_epa_tournament)


def _cid(game_id, market, side, line, snap=60.0):
    return make_market_contract_id(
        game_id=game_id, market_type=market, outcome_side=side, line_value=line,
        market_source=MARKET_SOURCE_CLOSING, snapshot_minutes_to_kickoff=snap,
        aggregation_method=AGGREGATION_METHOD,
    )


def _bench_row(game_id, spread, total, *, ml=0.55, cover=0.5, over=0.5, season=2022, snap=60.0):
    return dict(
        game_id=game_id, season=season, week=1, home_team_id="H", away_team_id="A",
        market_ml_home_probability=ml, market_cover_home_probability=cover,
        market_over_probability=over, closing_home_spread=spread,
        closing_total_line=total, closing_minutes_to_kickoff=snap,
        market_source=MARKET_SOURCE_CLOSING, aggregation_method=AGGREGATION_METHOD,
        moneyline_contract_id=_cid(game_id, "moneyline", "home", None, snap),
        spread_contract_id=_cid(game_id, "spread", "home", spread, snap),
        total_contract_id=_cid(game_id, "total", "over", total, snap),
        moneyline_consensus_books=3, spread_consensus_books=3, total_consensus_books=3,
        spread_candidate_point_count=1, total_candidate_point_count=1,
    )


def _games_row(game_id, season, ref_spread, ref_total, home_score, away_score):
    return dict(
        game_id=game_id, season=season, week=1, home_team_id="H", away_team_id="A",
        home_spread_reference=ref_spread, total_line_reference=ref_total,
        home_score=home_score, away_score=away_score,
    )


# Full closing-era contract-state rows for the authoritative two-state validator.
def _state_row(game_id, season, *, included, source, reason,
               ids=("m", "s", "t"), probs=(0.55, 0.5, 0.5)):
    return {
        "game_id": game_id,
        "season": season,
        "contract_included": included,
        "contract_exclusion_reason": reason,
        "market_contract_source": source,
        "moneyline_contract_id": ids[0],
        "spread_contract_id": ids[1],
        "total_contract_id": ids[2],
        "tournament_market_ml_home_probability": probs[0],
        "tournament_market_cover_home_probability": probs[1],
        "tournament_market_over_probability": probs[2],
    }


def _state_a(game_id, season=2022, **over):
    """A valid State A (included exact REAL-CLOSING contract) row."""
    row = _state_row(game_id, season, included=True,
                     source=MARKET_SOURCE_CLOSING, reason="")
    row.update(over)
    return row


def _state_b(game_id, season=2022, **over):
    """A valid State B (audited exclusion) row."""
    row = _state_row(game_id, season, included=False, source="",
                     reason=NO_BENCHMARK_ROW, ids=(pd.NA, pd.NA, pd.NA),
                     probs=(np.nan, np.nan, np.nan))
    row.update(over)
    return row


# --- Test A: closing spread bound before target construction ----------------- #
def test_closing_spread_bound_before_targets():
    # reference -2.5 (home cover); closing -3.0 (exact push); final home margin +3
    games = pd.DataFrame([_games_row("g1", 2022, -2.5, 44.5, 24, 21)])
    bench = pd.DataFrame([_bench_row("g1", -3.0, 44.5)])
    bound = bind_closing_contract(games, bench).games

    assert bound.loc[0, "home_spread_reference"] == -3.0        # closing bound in place
    oc = _outcome_columns(bound)
    assert oc.loc[0, "home_spread"] == -3.0                     # target sees closing
    assert pd.isna(oc.loc[0, "home_cover"])                     # ATS push -> nullable

    # under the reference contract it would have been a home cover (not a push)
    ref_oc = _outcome_columns(pd.DataFrame([_games_row("g1", 2022, -2.5, 44.5, 24, 21)]))
    assert ref_oc.loc[0, "home_cover"] == 1


# --- Test B: closing total bound before target construction ------------------ #
def test_closing_total_bound_before_targets():
    # reference 44.5 (under); closing 44.0 (exact push); total points 44
    games = pd.DataFrame([_games_row("g1", 2022, -3.0, 44.5, 24, 20)])
    bench = pd.DataFrame([_bench_row("g1", -3.0, 44.0)])
    bound = bind_closing_contract(games, bench).games
    assert bound.loc[0, "total_line_reference"] == 44.0
    oc = _outcome_columns(bound)
    assert oc.loc[0, "total_line"] == 44.0
    assert pd.isna(oc.loc[0, "over"])                           # total push -> nullable

    ref_oc = _outcome_columns(pd.DataFrame([_games_row("g1", 2022, -3.0, 44.5, 24, 20)]))
    assert ref_oc.loc[0, "over"] == 0                           # under at 44.5


# --- Test C: original references remain auditable ---------------------------- #
def test_original_references_retained():
    games = pd.DataFrame([_games_row("g1", 2022, -2.5, 44.5, 24, 21)])
    bench = pd.DataFrame([_bench_row("g1", -3.0, 44.0)])
    bound = bind_closing_contract(games, bench).games
    assert bound.loc[0, "original_home_spread_reference"] == -2.5
    assert bound.loc[0, "original_total_line_reference"] == 44.5
    assert bound.loc[0, "closing_home_spread"] == -3.0
    assert bound.loc[0, "closing_total_line"] == 44.0


# --- Test D: matched and unmatched dev games --------------------------------- #
def test_matched_and_unmatched_dev_games():
    games = pd.DataFrame([
        _games_row("cov", 2022, -2.5, 44.5, 24, 21),
        _games_row("unc", 2022, -1.0, 47.0, 20, 20),   # no benchmark row
    ])
    bench = pd.DataFrame([_bench_row("cov", -3.0, 44.0)])
    binding = bind_closing_contract(games, bench)
    bound = binding.games

    # both games participate in historical feature construction (both present)
    assert set(bound["game_id"]) == {"cov", "unc"}
    # the uncovered game is audited, not silently proxied or dropped
    unc = binding.audit[binding.audit["game_id"] == "unc"].iloc[0]
    assert not unc["contract_included"]
    assert unc["contract_exclusion_reason"] == "NO_BENCHMARK_ROW"
    assert unc["benchmark_merge_status"] == "left_only"

    # only the exact-contract game enters the tournament matrix
    tour = run_epa_tournament.select_tournament_matrix(bound)
    assert set(tour["game_id"]) == {"cov"}


# --- Test E: pre-2022 proxy is training-only --------------------------------- #
def test_pre_closing_proxy_is_training_only():
    games = pd.DataFrame([
        _games_row("p21", 2021, -3.0, 45.0, 21, 17),   # pre-closing -> proxy
        _games_row("r22", 2022, -3.0, 45.0, 21, 17),   # closing -> real
    ])
    bench = pd.DataFrame([_bench_row("r22", -3.0, 45.0)])
    bound = bind_closing_contract(games, bench).games
    src = dict(zip(bound["game_id"], bound["market_contract_source"]))
    assert src["p21"] == MARKET_SOURCE_PROXY
    assert src["r22"] == MARKET_SOURCE_CLOSING

    # proxy row is retained for training but is not a REAL-CLOSING scored row
    tour = run_epa_tournament.select_tournament_matrix(bound)
    assert set(tour["game_id"]) == {"p21", "r22"}       # both usable (proxy as history)

    # a 2022 test row carrying the proxy source must fail the strict test-contract guard
    from nfl_hybrid.selection.epa_tournament import _assert_test_contract
    proxy_test = tour[tour["game_id"] == "p21"].copy()
    proxy_test["home_spread"] = proxy_test["home_spread_reference"]
    proxy_test["total_line"] = proxy_test["total_line_reference"]
    proxy_test["closing_home_spread"] = proxy_test["home_spread"]
    proxy_test["closing_total_line"] = proxy_test["total_line"]
    with pytest.raises(ValueError):
        _assert_test_contract(proxy_test, MARKET_SOURCE_CLOSING)


# =============================================================================
# R2 Batch 2B: closing-era proxy contamination. One unambiguous season/source
# policy -- pre-2022 -> REFERENCE-LINE PROXY, 2022-2024 -> REAL-CLOSING -- is
# validated BEFORE tournament filtering; unmatched closing-era games are excluded
# and audited (never proxied). These are schema/value checks; no model is fit.
# =============================================================================
def test_select_tournament_matrix_accepts_valid_proxy_and_closing_mapping():
    # Under the strict two-state contract every row carries the full state schema:
    # pre-2022 proxy training rows plus valid State A closing-era contracts.
    matrix = pd.DataFrame(
        [
            _state_row("p2020", 2020, included=False, source=MARKET_SOURCE_PROXY,
                       reason="REFERENCE-LINE PROXY (pre-closing training only)"),
            _state_row("p2021", 2021, included=False, source=MARKET_SOURCE_PROXY,
                       reason="REFERENCE-LINE PROXY (pre-closing training only)"),
            _state_a("c2022", 2022),
            _state_a("c2023", 2023),
            _state_a("c2024", 2024),
        ]
    )
    tour = run_epa_tournament.select_tournament_matrix(matrix)
    # all five valid rows retained, in input order, with no source rewritten
    assert list(tour["game_id"]) == ["p2020", "p2021", "c2022", "c2023", "c2024"]
    assert list(tour["market_contract_source"]) == list(matrix["market_contract_source"])
    assert len(tour) == 5


def test_select_tournament_matrix_rejects_closing_era_proxy():
    # A 2022 row carrying a proxy source is now a State A violation (an included
    # closing-era row must be REAL-CLOSING) caught by the authoritative validator.
    matrix = pd.DataFrame(
        [
            _state_row("p2020", 2020, included=False, source=MARKET_SOURCE_PROXY,
                       reason="REFERENCE-LINE PROXY (pre-closing training only)"),
            _state_a("bad_2022_proxy", 2022, market_contract_source=MARKET_SOURCE_PROXY),
        ]
    )
    with pytest.raises(ValueError) as exc:
        run_epa_tournament.select_tournament_matrix(matrix)
    msg = str(exc.value)
    assert "bad_2022_proxy" in msg
    assert "2022" in msg
    assert "market_contract_source" in msg
    assert "REFERENCE-LINE PROXY" in msg   # actual source
    assert "REAL-CLOSING" in msg           # expected State A source


def test_select_tournament_matrix_rejects_preclosing_real_closing():
    # A pre-2022 row carrying REAL-CLOSING still fails the runner's proxy-only rule
    # (the closing-era State A row it accompanies is valid).
    matrix = pd.DataFrame(
        [
            _state_row("bad_2021_closing", 2021, included=False,
                       source=MARKET_SOURCE_CLOSING, reason=""),
            _state_a("c2022", 2022),
        ]
    )
    with pytest.raises(ValueError) as exc:
        run_epa_tournament.select_tournament_matrix(matrix)
    msg = str(exc.value)
    assert "invalid season/contract-source mapping" in msg
    assert "bad_2021_closing" in msg
    assert "2021" in msg
    assert "REAL-CLOSING" in msg           # actual source
    assert "REFERENCE-LINE PROXY" in msg   # expected source


def test_season_source_validator_rejects_missing_source():
    # A closing-era included row with a missing source is a State A violation.
    matrix = pd.DataFrame([_state_a("missing_src", 2022, market_contract_source=None)])
    with pytest.raises(ValueError) as exc:
        run_epa_tournament._validate_season_contract_sources(matrix)
    msg = str(exc.value)
    assert "missing_src" in msg
    assert "market_contract_source" in msg


def test_season_source_validator_rejects_nonnumeric_season():
    matrix = pd.DataFrame(
        {
            "game_id": ["bad_season"],
            "season": ["unknown"],
            "market_contract_source": [MARKET_SOURCE_CLOSING],
        }
    )
    with pytest.raises(ValueError) as exc:
        run_epa_tournament._validate_season_contract_sources(matrix)
    msg = str(exc.value)
    assert "non-numeric seasons" in msg
    assert "bad_season" in msg


def test_bind_closing_contract_unmatched_dev_game_is_excluded_not_proxied():
    # one 2022 game with NO benchmark row of its own (benchmark covers another game)
    games = pd.DataFrame([_games_row("unc2022", 2022, -2.5, 44.5, 20, 20)])
    bench = pd.DataFrame([_bench_row("other_game", -3.0, 44.0, season=2022)])
    bound = bind_closing_contract(games, bench).games
    row = bound[bound["game_id"] == "unc2022"].iloc[0]

    assert not bool(row["contract_included"])
    assert row["contract_exclusion_reason"] == "NO_BENCHMARK_ROW"
    assert row["market_contract_source"] != MARKET_SOURCE_PROXY
    # no proxy contract IDs were generated
    assert pd.isna(row["moneyline_contract_id"])
    assert pd.isna(row["spread_contract_id"])
    assert pd.isna(row["total_contract_id"])
    # no proxy tournament market probabilities were generated (never 0.5 fallback)
    assert pd.isna(row["tournament_market_ml_home_probability"])
    assert pd.isna(row["tournament_market_cover_home_probability"])
    assert pd.isna(row["tournament_market_over_probability"])


def test_bind_closing_contract_pre2022_game_uses_proxy():
    # one 2021 game with no exact closing benchmark of its own
    games = pd.DataFrame([_games_row("p2021", 2021, -3.0, 45.0, 21, 17)])
    bench = pd.DataFrame([_bench_row("other_game", -3.0, 45.0, season=2022)])
    bound = bind_closing_contract(games, bench).games
    row = bound[bound["game_id"] == "p2021"].iloc[0]

    assert row["market_contract_source"] == MARKET_SOURCE_PROXY
    ml = float(row["tournament_market_ml_home_probability"])
    assert np.isfinite(ml) and 0.0 < ml < 1.0
    assert np.isfinite(float(row["tournament_market_cover_home_probability"]))
    assert np.isfinite(float(row["tournament_market_over_probability"]))
    assert len(str(row["moneyline_contract_id"])) > 0
    assert len(str(row["spread_contract_id"])) > 0
    assert len(str(row["total_contract_id"])) > 0
    # eligible for training under the existing contract policy
    tour = run_epa_tournament.select_tournament_matrix(bound)
    assert "p2021" in set(tour["game_id"])


# =============================================================================
# R2 Batch 2B CORRECTION: the strict closing-era two-state contract validator.
#
# validate_closing_era_contract_states is the single implementation of the
# policy; every 2022-2024 game is either State A (included REAL-CLOSING contract)
# or State B (audited exclusion). The previously-exempt excluded rows are now
# validated, closing the silent-bypass defect. These are pure schema/value checks
# -- no model is fit and no evaluator runs.
# =============================================================================
def test_validator_rejects_excluded_closing_era_proxy():
    # The exact defect: an excluded closing-era row carrying a proxy source used to
    # bypass validation. It must now raise a State B violation.
    row = {
        "game_id": "silent_proxy",
        "season": 2022,
        "contract_included": False,
        "contract_exclusion_reason": NO_BENCHMARK_ROW,
        "market_contract_source": MARKET_SOURCE_PROXY,
        "moneyline_contract_id": pd.NA,
        "spread_contract_id": pd.NA,
        "total_contract_id": pd.NA,
        "tournament_market_ml_home_probability": np.nan,
        "tournament_market_cover_home_probability": np.nan,
        "tournament_market_over_probability": np.nan,
    }
    with pytest.raises(ValueError) as exc:
        validate_closing_era_contract_states(pd.DataFrame([row]), context="unit")
    msg = str(exc.value)
    assert "silent_proxy" in msg
    assert "2022" in msg
    assert "market_contract_source" in msg
    assert "REFERENCE-LINE PROXY" in msg
    assert "State B" in msg or "audited exclusion" in msg


def test_validator_rejects_missing_contract_included():
    row = _state_b("null_included")
    row["contract_included"] = pd.NA
    with pytest.raises(ValueError) as exc:
        validate_closing_era_contract_states(pd.DataFrame([row]), context="unit")
    msg = str(exc.value)
    assert "null_included" in msg
    assert "contract_included" in msg
    # a null contract_included is NOT interpreted as False
    assert "null" in msg or "missing" in msg


def test_validator_rejects_malformed_contract_included():
    # A single test node loops over the malformed cases; each must fail.
    for bad_value in ["False", 0, 1]:
        row = _state_b("bad_included")
        row["contract_included"] = bad_value
        with pytest.raises(ValueError) as exc:
            validate_closing_era_contract_states(pd.DataFrame([row]), context="unit")
        msg = str(exc.value)
        assert "bad_included" in msg
        assert "contract_included" in msg
        assert str(bad_value) in msg      # offending value
        assert "type=" in msg             # offending Python type name


def test_validator_accepts_well_formed_audited_exclusion():
    report = validate_closing_era_contract_states(
        pd.DataFrame([_state_b("clean_exclusion", 2022)]), context="unit"
    )
    assert report["closing_era_rows"] == 1
    assert report["included_exact_contract_rows"] == 0
    assert report["audited_excluded_rows"] == 1


def test_validator_rejects_exclusion_without_reason():
    row = _state_b("no_reason")
    row["contract_exclusion_reason"] = pd.NA
    with pytest.raises(ValueError) as exc:
        validate_closing_era_contract_states(pd.DataFrame([row]), context="unit")
    msg = str(exc.value)
    assert "no_reason" in msg
    assert "contract_exclusion_reason" in msg
    assert NO_BENCHMARK_ROW in msg


def test_validator_rejects_exclusion_with_contract_id():
    row = _state_b("stray_id")
    row["spread_contract_id"] = "unexpected_contract"
    with pytest.raises(ValueError) as exc:
        validate_closing_era_contract_states(pd.DataFrame([row]), context="unit")
    msg = str(exc.value)
    assert "stray_id" in msg
    assert "spread_contract_id" in msg
    assert "unexpected_contract" in msg
    assert "null" in msg or "blank" in msg


def test_validator_rejects_exclusion_with_market_probability():
    row = _state_b("stray_prob")
    row["tournament_market_cover_home_probability"] = 0.5
    with pytest.raises(ValueError) as exc:
        validate_closing_era_contract_states(pd.DataFrame([row]), context="unit")
    msg = str(exc.value)
    assert "stray_prob" in msg
    assert "tournament_market_cover_home_probability" in msg
    assert "0.5" in msg
    assert "null" in msg


def test_bind_closing_contract_asserts_complete_excluded_state(monkeypatch):
    # A matched (State A) game and an unmatched (State B) game in the closing era.
    games = pd.DataFrame([
        _games_row("cov2022", 2022, -2.5, 44.5, 24, 21),   # matched -> State A
        _games_row("unc2022", 2022, -1.0, 47.0, 20, 20),   # unmatched -> State B
    ])
    bench = pd.DataFrame([_bench_row("cov2022", -3.0, 44.0)])

    # 1) the normal binder result is a valid audited exclusion (State B)
    normal = bind_closing_contract(games, bench).games
    unc = normal[normal["game_id"] == "unc2022"].iloc[0]
    assert not bool(unc["contract_included"])
    assert unc["contract_exclusion_reason"] == NO_BENCHMARK_ROW
    assert (
        pd.isna(unc["market_contract_source"])
        or str(unc["market_contract_source"]).strip() == ""
    )

    # 2) corrupt exactly one excluded field immediately before authoritative
    # validation, through the binder's own validation seam (no production hook is
    # added). The binder must fail closed via the authoritative validator.
    real_validator = exact_contract.validate_closing_era_contract_states

    def corrupt_then_validate(frame, *, context, closing_seasons=(2022, 2023, 2024)):
        frame.loc[
            frame["game_id"] == "unc2022", "market_contract_source"
        ] = MARKET_SOURCE_PROXY
        return real_validator(frame, context=context, closing_seasons=closing_seasons)

    monkeypatch.setattr(
        exact_contract, "validate_closing_era_contract_states", corrupt_then_validate
    )
    with pytest.raises(ValueError) as exc:
        bind_closing_contract(games, bench)
    msg = str(exc.value)
    assert "unc2022" in msg
    assert "market_contract_source" in msg
    assert "REFERENCE-LINE PROXY" in msg
    assert "State B" in msg or "audited exclusion" in msg
