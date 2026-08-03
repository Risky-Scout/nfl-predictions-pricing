"""Runner regression tests: exact-contract binding order and no line overwrite.

Behavioural where possible; a narrow source-level check guards specifically
against the removed post-prediction line-overwrite defect.
"""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.labels import edge_to_nullable_binary
from nfl_hybrid.markets.exact_contract import (
    AGGREGATION_METHOD,
    MARKET_SOURCE_CLOSING,
    MARKET_SOURCE_PROXY,
    make_market_contract_id,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_epa_tournament.py"
FIXTURE = ROOT / "tests" / "fixtures" / "live_features"

_spec = importlib.util.spec_from_file_location("run_epa_tournament", RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #
def _cid(game_id, market, side, line, snap=180.0):
    return make_market_contract_id(
        game_id=game_id, market_type=market, outcome_side=side, line_value=line,
        market_source=MARKET_SOURCE_CLOSING, snapshot_minutes_to_kickoff=snap,
        aggregation_method=AGGREGATION_METHOD,
    )


def _bench_row(game_id, season, spread, total, *, ml=0.55, cover=0.5, over=0.5, snap=180.0):
    return dict(
        game_id=game_id, season=season, week=1, home_team_id="H", away_team_id="A",
        market_ml_home_probability=ml, market_cover_home_probability=cover,
        market_over_probability=over, closing_home_spread=spread,
        closing_total_line=total, closing_minutes_to_kickoff=snap,
        market_source=MARKET_SOURCE_CLOSING, aggregation_method=AGGREGATION_METHOD,
        moneyline_contract_id=_cid(game_id, "moneyline", "home", None, snap),
        spread_contract_id=_cid(game_id, "spread", "home", spread, snap),
        total_contract_id=_cid(game_id, "total", "over", total, snap),
        moneyline_consensus_books=5, spread_consensus_books=5, total_consensus_books=5,
        spread_candidate_point_count=1, total_candidate_point_count=1,
    )


def _fixture_games():
    g = pd.read_csv(FIXTURE / "games.csv")
    return g[g["home_score"].notna()].copy()


def _fixture_pbp():
    return pd.read_csv(FIXTURE / "pbp.csv")


def _benchmark_for_2022(games, spread_shift=0.0):
    """A benchmark covering every 2022 fixture game, with closing points optionally
    shifted away from the reference so binding is observable."""
    rows = []
    for _, r in games[games["season"] == 2022].iterrows():
        ref_spread = float(r["home_spread_reference"])
        ref_total = float(r["total_line_reference"])
        rows.append(_bench_row(str(r["game_id"]), 2022, ref_spread + spread_shift, ref_total + 0.5))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1-4: binding order, single build, closing overrides, coverage audit
# --------------------------------------------------------------------------- #
def test_matrix_lines_equal_closing_after_binding():
    games, pbp = _fixture_games(), _fixture_pbp()
    bench = _benchmark_for_2022(games, spread_shift=-1.5)  # closing != reference
    matrix, _, _ = runner.build_contract_bound_matrix(games, pbp, bench)
    real = matrix[matrix["market_contract_source"] == MARKET_SOURCE_CLOSING]
    assert len(real) > 0
    # every scored row's line equals the bound closing point (targets included)
    assert np.allclose(real["home_spread"].to_numpy(float), real["closing_home_spread"].to_numpy(float))
    assert np.allclose(real["total_line"].to_numpy(float), real["closing_total_line"].to_numpy(float))


def test_matrix_builder_called_exactly_once(monkeypatch):
    games, pbp = _fixture_games(), _fixture_pbp()
    bench = _benchmark_for_2022(games)
    calls = {"n": 0, "games_had_closing": None}
    orig = runner.build_augmented_feature_matrix

    def spy(g, p, **kw):
        calls["n"] += 1
        closing = dict(zip(bench["game_id"].astype(str), bench["closing_home_spread"].astype(float)))
        sub = g[g["game_id"].astype(str).isin(closing)]
        ref = pd.to_numeric(sub["home_spread_reference"], errors="coerce").to_numpy(float)
        exp = np.array([closing[str(x)] for x in sub["game_id"]], dtype=float)
        calls["games_had_closing"] = bool(np.allclose(ref, exp))
        return orig(g, p, **kw)

    monkeypatch.setattr(runner, "build_augmented_feature_matrix", spy)
    runner.build_contract_bound_matrix(games, pbp, bench)
    assert calls["n"] == 1                        # one and only one matrix build
    assert calls["games_had_closing"] is True     # closing already bound before build


def test_coverage_audit_includes_every_dev_matrix_game():
    games, pbp = _fixture_games(), _fixture_pbp()
    bench = _benchmark_for_2022(games)
    matrix, _, binding = runner.build_contract_bound_matrix(games, pbp, bench)
    coverage = runner.build_coverage_audit(matrix, binding)
    dev_matrix_ids = set(matrix[matrix["season"] == 2022]["game_id"].astype(str))
    assert set(coverage["game_id"].astype(str)) == dev_matrix_ids
    assert set(coverage.columns) >= {
        "game_id", "season", "week", "contract_included", "contract_exclusion_reason",
        "benchmark_merge_status",
    }


# --------------------------------------------------------------------------- #
# Synthetic full-scale scenario for run_tournament behaviour (needs >=100 train)
# --------------------------------------------------------------------------- #
FEATURES = ["home_spread", "total_line", "epa_net_edge_season_mean", "rest_diff"]


def _scenario(n_real=40, seed=1):
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    # proxy training history (pre-closing): 2020, 2021
    for season in (2020, 2021):
        for _ in range(70):
            gid += 1
            hs = float(np.round(rng.normal(0, 6)))
            tl = float(np.round(rng.normal(45, 5)))
            margin = -hs + rng.normal(0, 13)
            total = tl + rng.normal(0, 13)
            rows.append(_row(f"p{gid}", season, hs, tl, margin, total, MARKET_SOURCE_PROXY, rng))
    bench_rows = []
    for season in (2022, 2023, 2024):
        for _ in range(n_real):
            gid += 1
            gidn = f"r{gid}"
            hs = float(np.round(rng.normal(0, 6)))
            tl = float(np.round(rng.normal(45, 5)))
            margin = -hs + rng.normal(0, 13)
            total = tl + rng.normal(0, 13)
            rows.append(_row(gidn, season, hs, tl, margin, total, MARKET_SOURCE_CLOSING, rng, snap=180.0))
            bench_rows.append(_bench_row(gidn, season, hs, tl))
    return pd.DataFrame(rows), pd.DataFrame(bench_rows)


def _row(gid, season, hs, tl, margin, total, source, rng, snap=180.0):
    real = source == MARKET_SOURCE_CLOSING
    return dict(
        game_id=gid, season=season, week=int(rng.integers(1, 18)),
        home_spread=hs, total_line=tl, home_margin=margin, total_points=total,
        epa_net_edge_season_mean=float(rng.normal(0, 0.1)), rest_diff=float(rng.integers(-4, 5)),
        home_win=edge_to_nullable_binary(pd.Series([margin])).iloc[0],
        home_cover=edge_to_nullable_binary(pd.Series([margin + hs])).iloc[0],
        over=edge_to_nullable_binary(pd.Series([total - tl])).iloc[0],
        market_contract_source=source,
        # Full closing-era contract-state schema so the authoritative two-state
        # validator accepts closing-era REAL-CLOSING rows as State A. Pre-2022 proxy
        # rows are training history (not closing-era), so they are never State-checked.
        contract_included=bool(real),
        contract_exclusion_reason="" if real else "REFERENCE-LINE PROXY (pre-closing training only)",
        moneyline_contract_id=_cid(gid, "moneyline", "home", None, snap) if real else "proxy",
        spread_contract_id=_cid(gid, "spread", "home", hs, snap) if real else "proxy",
        total_contract_id=_cid(gid, "total", "over", tl, snap) if real else "proxy",
        tournament_market_ml_home_probability=0.55 if real else 0.5,
        tournament_market_cover_home_probability=0.5,
        tournament_market_over_probability=0.5,
        closing_home_spread=hs if real else np.nan,
        closing_total_line=tl if real else np.nan,
        closing_minutes_to_kickoff=snap if real else np.nan,
        original_home_spread_reference=hs,
        original_total_line_reference=tl,
    )


def test_run_tournament_validator_runs_before_eval(monkeypatch):
    matrix, bench = _scenario()
    called = {"eval": 0}
    orig_eval = runner.evaluate_market_relative
    monkeypatch.setattr(runner, "evaluate_market_relative",
                        lambda *a, **k: called.__setitem__("eval", called["eval"] + 1) or orig_eval(*a, **k))
    # tamper the benchmark contract so prediction != benchmark -> must raise, no eval
    bad = bench.copy()
    bad.loc[0, "closing_home_spread"] = bad.loc[0, "closing_home_spread"] + 1.0
    bad.loc[0, "spread_contract_id"] = _cid(bad.loc[0, "game_id"], "spread", "home", bad.loc[0, "closing_home_spread"])
    with pytest.raises(Exception):
        runner.run_tournament(matrix, FEATURES, bad, bootstrap_repetitions=50)
    assert called["eval"] == 0            # evaluation never reached after mismatch


def test_run_tournament_market_probs_from_contract_bound_rows(monkeypatch):
    matrix, bench = _scenario()
    captured = {}
    orig_eval = runner.evaluate_market_relative

    def spy(pred, **kw):
        mp = kw["market_probabilities"]
        captured.setdefault("checks", []).append(
            np.allclose(mp["market_cover_home_probability"].to_numpy(float),
                        pred["tournament_market_cover_home_probability"].to_numpy(float))
            and (mp["market_source"] == MARKET_SOURCE_CLOSING).all()
        )
        return orig_eval(pred, **kw)

    monkeypatch.setattr(runner, "evaluate_market_relative", spy)
    scorecard, _ = runner.run_tournament(matrix, FEATURES, bench, bootstrap_repetitions=50)
    assert captured["checks"] and all(captured["checks"])
    assert len(scorecard) > 0


def test_no_post_prediction_line_overwrite_in_source():
    src = RUNNER_PATH.read_text()
    # the removed defect assigned the closing line back onto a merged frame
    assert not re.search(r'merged\[\s*["\']home_spread["\']\s*\]\s*=', src)
    assert not re.search(r'merged\[\s*["\']total_line["\']\s*\]\s*=', src)
    assert not re.search(r'\[\s*["\']home_spread["\']\s*\]\s*=.*closing', src)
    assert not re.search(r'\[\s*["\']total_line["\']\s*\]\s*=.*closing', src)


def test_summary_reports_no_overwrite_and_validation_passed():
    matrix, bench = _scenario()
    scorecard, _ = runner.run_tournament(matrix, FEATURES, bench, bootstrap_repetitions=50)
    summary = runner.build_summary(matrix, _binding_stub(matrix), scorecard, bench, input_row_count=123)
    assert summary["post_prediction_line_overwrite"] is False
    assert summary["contract_validation_passed"] is True
    assert summary["contract_policy"] == "exact-closing-bound-before-matrix-v1"
    assert summary["benchmark_aggregation"] == "exact_point_consensus_v1"
    assert set(summary["exact_contract_games_by_season"]) == {2022, 2023, 2024}


def _binding_stub(matrix):
    class _B:
        audit = pd.DataFrame(
            {
                "game_id": matrix["game_id"].astype(str),
                "season": matrix["season"],
                "week": matrix["week"],
                "contract_included": matrix["market_contract_source"] == MARKET_SOURCE_CLOSING,
                "contract_exclusion_reason": np.where(
                    matrix["market_contract_source"] == MARKET_SOURCE_CLOSING, "INCLUDED", "NO_BENCHMARK_ROW"
                ),
                "benchmark_merge_status": "both",
            }
        )
    return _B()


# --------------------------------------------------------------------------- #
# R2 batch Issue 2: the runner must never silently omit a registered candidate.
# --------------------------------------------------------------------------- #
def test_run_tournament_fails_on_omitted_candidate(monkeypatch):
    matrix, bench = _scenario()
    orig = runner.run_walk_forward

    def drop_one(tour, features, *, test_seasons, market_prob_cols,
                 expected_test_contract_source=None, require_complete=False):
        # build a full result set (non-strict) then drop a registered candidate
        full = dict(orig(tour, features, test_seasons=test_seasons,
                         market_prob_cols=market_prob_cols))
        full.pop("C5_stacked", None)
        return full

    monkeypatch.setattr(runner, "run_walk_forward", drop_one)
    with pytest.raises(RuntimeError, match="C5_stacked"):
        runner.run_tournament(matrix, FEATURES, bench, bootstrap_repetitions=25)


def test_run_tournament_scorecard_covers_every_registered_candidate():
    matrix, bench = _scenario()
    scorecard, _ = runner.run_tournament(matrix, FEATURES, bench, bootstrap_repetitions=25)
    assert set(scorecard["candidate"]) == set(runner.CANDIDATES)


# =============================================================================
# R2 batch 1 correction: runner completeness + exact 15 candidate-market pairs.
# Every heavy dependency (walk-forward, contract validation, evaluation) is
# monkeypatched, so no real estimator or evaluator ever runs.
# =============================================================================
_RUNNER_MATRIX = pd.DataFrame(
    {
        "game_id": ["r1", "r2"],
        "season": [2022, 2022],
        "market_contract_source": [MARKET_SOURCE_CLOSING, MARKET_SOURCE_CLOSING],
        # Valid State A closing-era contracts so the authoritative validator in
        # select_tournament_matrix passes (heavy deps below stay monkeypatched).
        "contract_included": [True, True],
        "contract_exclusion_reason": ["", ""],
        "moneyline_contract_id": ["ml_r1", "ml_r2"],
        "spread_contract_id": ["sp_r1", "sp_r2"],
        "total_contract_id": ["to_r1", "to_r2"],
        "tournament_market_ml_home_probability": [0.55, 0.52],
        "tournament_market_cover_home_probability": [0.50, 0.51],
        "tournament_market_over_probability": [0.50, 0.49],
    }
)
_RUNNER_FEATURES = ["home_spread", "total_line"]
_RUNNER_BENCH = pd.DataFrame({"game_id": ["r1", "r2"]})


def _fake_pred(candidate):
    """Reusable fake prediction frame: contract, outcome and probability columns
    plus a candidate tag so the fake evaluator can vary its markets per call."""
    return pd.DataFrame(
        {
            "game_id": ["r1", "r2"],
            "season": [2022, 2022],
            "home_win": [1, 0],
            "home_cover": [1, 0],
            "over": [1, 0],
            "tournament_market_ml_home_probability": [0.55, 0.52],
            "tournament_market_cover_home_probability": [0.50, 0.51],
            "tournament_market_over_probability": [0.50, 0.49],
            "home_win_probability_no_tie": [0.60, 0.40],
            "home_cover_probability_no_push": [0.55, 0.45],
            "over_probability_no_push": [0.52, 0.48],
            "_candidate_tag": [candidate, candidate],
        }
    )


def _fake_walk_forward_factory(candidate_names):
    def fake(tour, features, *, test_seasons, market_prob_cols,
             expected_test_contract_source=None, require_complete=False):
        return {name: _fake_pred(name) for name in candidate_names}
    return fake


def _fake_evaluator_factory(markets_by_candidate):
    """Fake evaluator: a complete candidate yields three pooled market rows
    (home_win/home_cover/over -> moneyline/ats/total); overrides let a candidate
    omit or duplicate a market."""
    def fake(pred, **kwargs):
        cand = str(pred["_candidate_tag"].iloc[0])
        markets = markets_by_candidate.get(cand, ["home_win", "home_cover", "over"])
        n = len(markets)
        scorecard = pd.DataFrame(
            {
                "segment": ["pooled"] * n,
                "market": list(markets),
                "log_loss_gain_vs_market": [0.01] * n,
                "brier_gain_vs_market": [0.0] * n,
                "beats_breakeven_ci": [True] * n,
            }
        )
        return {"probability_scorecard": scorecard}
    return fake


def _patch_runner(monkeypatch, *, candidate_names, markets_by_candidate=None,
                  eval_counter=None):
    monkeypatch.setattr(runner, "run_walk_forward",
                        _fake_walk_forward_factory(candidate_names))
    monkeypatch.setattr(runner, "validate_prediction_contract", lambda *a, **k: None)
    base_eval = _fake_evaluator_factory(markets_by_candidate or {})

    def eval_wrapper(pred, **kwargs):
        if eval_counter is not None:
            eval_counter["n"] += 1
        return base_eval(pred, **kwargs)

    monkeypatch.setattr(runner, "evaluate_market_relative", eval_wrapper)


def test_runner_rejects_missing_candidate_result(monkeypatch):
    counter = {"n": 0}
    _patch_runner(
        monkeypatch,
        candidate_names=[c for c in runner.CANDIDATES if c != "C5_stacked"],
        eval_counter=counter,
    )
    with pytest.raises(RuntimeError, match="C5_stacked"):
        runner.run_tournament(_RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH)
    assert counter["n"] == 0  # raised before any evaluation


def test_runner_rejects_missing_candidate_market_pair(monkeypatch):
    _patch_runner(
        monkeypatch,
        candidate_names=list(runner.CANDIDATES),
        markets_by_candidate={"C3_logistic": ["home_win", "home_cover"]},  # omit total
    )
    with pytest.raises(RuntimeError, match="candidate-market coverage mismatch") as exc:
        runner.run_tournament(_RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH)
    assert "C3_logistic" in str(exc.value) and "total" in str(exc.value)


def test_runner_rejects_duplicate_candidate_market_pair(monkeypatch):
    _patch_runner(
        monkeypatch,
        candidate_names=list(runner.CANDIDATES),
        markets_by_candidate={"C3_logistic": ["home_win", "home_win", "home_cover", "over"]},
    )
    with pytest.raises(RuntimeError, match="duplicate candidate-market pairs"):
        runner.run_tournament(_RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH)


def test_runner_accepts_exact_fifteen_candidate_market_pairs(monkeypatch):
    _patch_runner(monkeypatch, candidate_names=list(runner.CANDIDATES))
    scorecard, results = runner.run_tournament(
        _RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH
    )
    assert scorecard["candidate"].nunique() == 5
    pairs = set(zip(scorecard["candidate"], scorecard["market_name"]))
    assert len(pairs) == 15


def _valid_scorecard():
    rows = []
    for c in runner.CANDIDATES:
        for mname in ("moneyline", "ats", "total"):
            rows.append(dict(candidate=c, market_name=mname, promotion="RETAIN_BASELINE"))
    return pd.DataFrame(rows)


def _summary_matrix():
    rows = []
    for season in (2022, 2023, 2024):
        for i in range(2):
            hs, tl = -3.0, 44.0
            rows.append(
                dict(
                    game_id=f"{season}_{i}", season=season, week=1,
                    market_contract_source=MARKET_SOURCE_CLOSING,
                    home_spread=hs, total_line=tl, home_margin=7.0, total_points=45.0,
                    closing_home_spread=hs, closing_total_line=tl,
                    original_home_spread_reference=hs, original_total_line_reference=tl,
                    home_cover=edge_to_nullable_binary(pd.Series([7.0 + hs])).iloc[0],
                    over=edge_to_nullable_binary(pd.Series([45.0 - tl])).iloc[0],
                )
            )
    return pd.DataFrame(rows)


def test_summary_counts_come_from_validated_scorecard():
    matrix = _summary_matrix()
    scorecard = _valid_scorecard()
    summary = runner.build_summary(matrix, None, scorecard, _RUNNER_BENCH, input_row_count=10)
    assert summary["n_candidates"] == 5
    assert summary["n_candidate_markets"] == 15
    # remove one pair: the summary must fail closed, never report the constant
    # registry size from an unvalidated scorecard.
    with pytest.raises(RuntimeError):
        runner.build_summary(matrix, None, scorecard.iloc[:-1], _RUNNER_BENCH, input_row_count=10)


# =============================================================================
# R2 Batch 2A: runner defense-in-depth candidate GAME COVERAGE. The runner must
# reject a candidate whose result frame is missing, extra, duplicated, or
# reordered relative to the expected requested-season grid BEFORE any contract
# validation or evaluation. Both are monkeypatched with counters; no real
# candidate, contract validator, or evaluator ever runs.
# =============================================================================
def _patch_runner_coverage(monkeypatch, results_by_candidate):
    """Reuse the Batch 1 fake evaluator; force run_walk_forward to return the given
    per-candidate results and count contract-validation and evaluation calls."""
    counters = {"contract": 0, "eval": 0}

    def fake_wf(tour, features, *, test_seasons, market_prob_cols,
                expected_test_contract_source=None, require_complete=False):
        return results_by_candidate

    monkeypatch.setattr(runner, "run_walk_forward", fake_wf)

    def fake_contract(*a, **k):
        counters["contract"] += 1

    monkeypatch.setattr(runner, "validate_prediction_contract", fake_contract)

    base_eval = _fake_evaluator_factory({})

    def eval_wrapper(pred, **kwargs):
        counters["eval"] += 1
        return base_eval(pred, **kwargs)

    monkeypatch.setattr(runner, "evaluate_market_relative", eval_wrapper)
    return counters


def _coverage_results(c3):
    """All five candidates return the expected two 2022 games except C3, which is
    replaced by the supplied malformed frame."""
    results = {name: _fake_pred(name) for name in runner.CANDIDATES}
    results["C3_logistic"] = c3
    return results


def test_runner_rejects_candidate_missing_expected_game(monkeypatch):
    c3 = _fake_pred("C3_logistic").iloc[:-1].copy()   # drops game r2
    counters = _patch_runner_coverage(monkeypatch, _coverage_results(c3))
    with pytest.raises(RuntimeError) as exc:
        runner.run_tournament(_RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH,
                              test_seasons=[2022])
    msg = str(exc.value)
    assert "C3_logistic" in msg
    assert "missing" in msg
    assert "(2022, 'r2')" in msg
    assert counters["contract"] == 0
    assert counters["eval"] == 0


def test_runner_rejects_candidate_extra_game(monkeypatch):
    base = _fake_pred("C3_logistic")
    extra = base.iloc[[0]].copy()
    extra["game_id"] = "unexpected_game"
    c3 = pd.concat([base, extra], ignore_index=True)   # r1, r2, unexpected_game
    counters = _patch_runner_coverage(monkeypatch, _coverage_results(c3))
    with pytest.raises(RuntimeError) as exc:
        runner.run_tournament(_RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH,
                              test_seasons=[2022])
    msg = str(exc.value)
    assert "C3_logistic" in msg
    assert "extra" in msg
    assert "unexpected_game" in msg
    assert counters["contract"] == 0
    assert counters["eval"] == 0


def test_runner_rejects_candidate_duplicate_game_key(monkeypatch):
    base = _fake_pred("C3_logistic")
    dup = base.iloc[[0]].copy()                        # a second (2022, r1)
    c3 = pd.concat([base, dup], ignore_index=True)     # r1, r2, r1
    counters = _patch_runner_coverage(monkeypatch, _coverage_results(c3))
    with pytest.raises(ValueError) as exc:
        runner.run_tournament(_RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH,
                              test_seasons=[2022])
    assert "duplicate season/game keys" in str(exc.value)
    assert counters["contract"] == 0
    assert counters["eval"] == 0


def test_runner_rejects_candidate_game_order_mismatch(monkeypatch):
    c3 = _fake_pred("C3_logistic").iloc[::-1].copy()   # r2, r1 (same set, reversed)
    counters = _patch_runner_coverage(monkeypatch, _coverage_results(c3))
    with pytest.raises(RuntimeError) as exc:
        runner.run_tournament(_RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH,
                              test_seasons=[2022])
    msg = str(exc.value)
    assert "C3_logistic" in msg
    assert "order_mismatch=True" in msg
    assert counters["contract"] == 0
    assert counters["eval"] == 0


def test_runner_accepts_exact_candidate_game_coverage(monkeypatch):
    results = {name: _fake_pred(name) for name in runner.CANDIDATES}
    counters = _patch_runner_coverage(monkeypatch, results)
    scorecard, out = runner.run_tournament(
        _RUNNER_MATRIX, _RUNNER_FEATURES, _RUNNER_BENCH, test_seasons=[2022]
    )
    # coverage passed -> every candidate was contract-validated and evaluated once
    assert counters["contract"] == 5
    assert counters["eval"] == 5
    assert set(scorecard["candidate"]) == set(runner.CANDIDATES)
    assert set(out) == set(runner.CANDIDATES)
    # identical ordered game keys for every candidate result
    from nfl_hybrid.selection.epa_tournament import _ordered_game_keys
    key_lists = [_ordered_game_keys(out[name], context=name) for name in runner.CANDIDATES]
    assert all(k == key_lists[0] for k in key_lists)
    assert key_lists[0] == [(2022, "r1"), (2022, "r2")]


# =============================================================================
# R2 Batch 2B: closing-era proxy contamination must be stopped BEFORE any model
# fitting or scoring. A 2022 row carrying REFERENCE-LINE PROXY fails in
# select_tournament_matrix, so run_walk_forward, contract validation, and the
# evaluator are never reached.
# =============================================================================
def test_runner_rejects_closing_era_proxy_before_walk_forward(monkeypatch):
    counters = {"wf": 0, "contract": 0, "eval": 0}

    def fake_wf(*a, **k):
        counters["wf"] += 1
        return {}

    def fake_contract(*a, **k):
        counters["contract"] += 1

    def fake_eval(*a, **k):
        counters["eval"] += 1
        return {"probability_scorecard": pd.DataFrame()}

    monkeypatch.setattr(runner, "run_walk_forward", fake_wf)
    monkeypatch.setattr(runner, "validate_prediction_contract", fake_contract)
    monkeypatch.setattr(runner, "evaluate_market_relative", fake_eval)

    def _proxy(gid, season):
        return dict(
            game_id=gid, season=season, contract_included=False,
            contract_exclusion_reason="REFERENCE-LINE PROXY (pre-closing training only)",
            market_contract_source=MARKET_SOURCE_PROXY,
            moneyline_contract_id="proxy", spread_contract_id="proxy",
            total_contract_id="proxy", tournament_market_ml_home_probability=0.5,
            tournament_market_cover_home_probability=0.5,
            tournament_market_over_probability=0.5,
        )

    # A closing-era row carrying a proxy source is a State A violation caught by the
    # authoritative validator in select_tournament_matrix, before any fit or score.
    bad_closing = dict(_proxy("bad_2022_proxy", 2022))
    bad_closing["contract_included"] = True
    matrix = pd.DataFrame([_proxy("p2020", 2020), bad_closing])

    with pytest.raises(ValueError) as exc:
        runner.run_tournament(matrix, _RUNNER_FEATURES, _RUNNER_BENCH,
                              test_seasons=[2022])
    msg = str(exc.value)
    assert "bad_2022_proxy" in msg
    assert "market_contract_source" in msg
    assert "REFERENCE-LINE PROXY" in msg   # actual source
    assert "REAL-CLOSING" in msg           # expected State A source
    # contamination stopped before fitting or scoring
    assert counters["wf"] == 0
    assert counters["contract"] == 0
    assert counters["eval"] == 0
