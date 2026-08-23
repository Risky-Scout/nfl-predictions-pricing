"""Focused tests for ``scripts/run_fix7_evidence_replay.py``.

CI-safe: loads the ACTUAL replay script file (never re-implements its
helpers) via ``importlib``, and exercises its pure/cross-check logic against
a small synthetic multi-season fixture -- same shape as Fix 6/Fix 7's own
fixtures. Real committed 2020-2024 evidence lives outside this repo
(``NFL_MODEL_ARTIFACT_ROOT``) and is not required here; the phases that
depend on it (Phase 1/2/9, which read real backfill data and the external
artifact root) are not exercised by these tests, matching the existing
convention in ``test_model_family_selection_2026.py``.
"""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.week1_prior import Week1PriorConfig
from nfl_hybrid.selection import feature_deduction_2026 as fd
from nfl_hybrid.selection import model_family_selection_2026 as mfs

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fix7_evidence_replay.py"


def _load_replay_script():
    """Load the exact production replay script file without copying any of
    its logic. Module-level code is only imports/constants/function defs --
    the actual replay run is behind ``if __name__ == "__main__"``, so
    importing it here never touches git, real data, or the artifact root."""
    spec = importlib.util.spec_from_file_location("run_fix7_evidence_replay_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


replay = _load_replay_script()

TEAMS = ("KC", "BUF", "SF", "DAL")


def _synthetic_games(extra_seasons: tuple[int, ...] = ()) -> pd.DataFrame:
    pairs = list(itertools.combinations(TEAMS, 2))
    rows = []
    game_num = 0
    seasons = (2020, 2021, 2022, 2023, 2024) + extra_seasons
    for season in seasons:
        for week, (a, b) in enumerate(pairs, start=1):
            game_num += 1
            home, away = (a, b) if (season + week) % 2 == 0 else (b, a)
            home_idx, away_idx = TEAMS.index(home), TEAMS.index(away)
            base = 20 + 2 * (home_idx - away_idx) + (season - 2020)
            home_score = max(base + (week % 3), 0)
            away_score = max(base - (week % 4), 0)
            rows.append({
                "game_id": f"G{game_num}", "season": season, "season_type": "REG", "week": week,
                "home_team_id": home, "away_team_id": away,
                "scheduled_kickoff_utc": pd.Timestamp(f"{min(season, 2026)}-09-{min(1 + week, 28):02d}T17:00:00Z"),
                "home_score": home_score, "away_score": away_score,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def week1_config() -> Week1PriorConfig:
    return Week1PriorConfig(k=8.0)


@pytest.fixture(scope="module")
def elo_matrix(week1_config):
    games = _synthetic_games()
    matrix, features = fd.build_candidate_matrix(games, ["ELO_STRENGTH"], week1_config=week1_config)
    return matrix, features


@pytest.fixture(scope="module")
def inner_results(elo_matrix):
    matrix, feature_columns = elo_matrix
    results = {}
    for candidate in mfs.CANDIDATE_REGISTRY:
        results[candidate.name] = {}
        for fold in fd.INNER_FOLDS:
            preds, _invalid = mfs.fit_predict_fold(matrix, feature_columns, fold, candidate)
            results[candidate.name][fold.name] = preds
    return results


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------
def test_tolerance_boundary_exact_match():
    mismatches: list = []
    original = 13.713259074953852
    replayed = original  # bit-identical replay
    record = replay._check_close("x", original, replayed, mismatches)
    assert record["match"] is True
    assert mismatches == []


def test_tolerance_boundary_epsilon_beyond_fails():
    mismatches: list = []
    original = 13.713259074953852
    replayed = original + 1.0  # far beyond max(1e-10, 1e-12*|original|)
    record = replay._check_close("x", original, replayed, mismatches)
    assert record["match"] is False
    assert mismatches and mismatches[0]["label"] == "x"


def test_check_equal_detects_mismatch():
    mismatches: list = []
    replay._check_equal("family", "RIDGE", "HUBER", mismatches)
    assert mismatches[0]["original_value"] == "RIDGE"
    assert mismatches[0]["replayed_value"] == "HUBER"


# ---------------------------------------------------------------------------
# Per-fold metric calculation reproduces independently computed values
# ---------------------------------------------------------------------------
def test_per_fold_metric_calculation_matches_independent_computation(inner_results):
    from nfl_hybrid.evaluation.metrics import regression_metrics

    preds = inner_results["RIDGE_ALPHA_1"]["A"]
    report = mfs._regression_report(preds)
    independent_margin = regression_metrics(preds["home_margin"].to_numpy(float), preds["pred_margin"].to_numpy(float))
    independent_total = regression_metrics(preds["total_points"].to_numpy(float), preds["pred_total"].to_numpy(float))
    assert report["margin"]["rmse"] == pytest.approx(independent_margin["rmse"])
    assert report["margin"]["mae"] == pytest.approx(independent_margin["mae"])
    assert report["total"]["rmse"] == pytest.approx(independent_total["rmse"])
    assert report["total"]["mae"] == pytest.approx(independent_total["mae"])
    assert report["primary_score"] == pytest.approx(0.5 * (independent_margin["rmse"] + independent_total["rmse"]))


def test_per_fold_detail_row_has_no_missing_metric_fields(inner_results):
    preds = inner_results["HUBER_FIXED"]["B"]
    report = mfs._regression_report(preds)
    for key in ("margin", "total"):
        assert np.isfinite(report[key]["rmse"])
        assert np.isfinite(report[key]["mae"])
    assert np.isfinite(report["primary_score"])


# ---------------------------------------------------------------------------
# Pooled reproduction: identical inner_results reproduce bit-for-bit
# ---------------------------------------------------------------------------
def test_pooled_reproduction_self_consistent(inner_results):
    for candidate in mfs.CANDIDATE_REGISTRY:
        pooled_a = mfs._pooled(inner_results, candidate.name)
        pooled_b = mfs._pooled(inner_results, candidate.name)
        assert mfs._primary_score(pooled_a) == mfs._primary_score(pooled_b)


def test_pooled_reproduction_gate_raises_on_genuine_mismatch(inner_results):
    original = {
        "ridge_selection": {"per_alpha": {
            c.name: {"pooled_primary_score": 999.0}  # deliberately wrong
            for c in mfs.CANDIDATE_REGISTRY if c.family == "RIDGE"
        }},
        "family_selection": {"pooled_primary": {"HUBER": 999.0, "HGBR": 999.0}},
    }
    with pytest.raises(replay.ReplayMismatch):
        replay.phase6_pooled_reproduction_gate({"inner_results": inner_results}, original)


# ---------------------------------------------------------------------------
# Ridge paired reproduction: deterministic, no RNG
# ---------------------------------------------------------------------------
def test_ridge_paired_reproduction_deterministic(inner_results):
    result_a = mfs.select_ridge_alpha(inner_results)
    result_b = mfs.select_ridge_alpha(inner_results)
    assert result_a == result_b


def test_ridge_alpha_cross_check_matches_self(inner_results):
    replayed = mfs.select_ridge_alpha(inner_results)
    original = {"ridge_selection": replayed}
    result = replay.phase7_ridge_alpha_cross_check({"inner_results": inner_results}, original)
    assert result["replay_result"]["selected_alpha_name"] == replayed["selected_alpha_name"]


def test_ridge_alpha_cross_check_raises_on_mismatch(inner_results):
    replayed = mfs.select_ridge_alpha(inner_results)
    wrong_name = next(n for n in mfs.RIDGE_ALPHA_NAMES if n != replayed["best_alpha_name"])
    corrupted = {**replayed, "best_alpha_name": wrong_name}
    original = {"ridge_selection": corrupted}
    with pytest.raises(replay.ReplayMismatch):
        replay.phase7_ridge_alpha_cross_check({"inner_results": inner_results}, original)


# ---------------------------------------------------------------------------
# Family paired reproduction
# ---------------------------------------------------------------------------
def test_family_selection_cross_check_matches_self(inner_results):
    ridge_result = mfs.select_ridge_alpha(inner_results)
    family_result = mfs.select_family(inner_results, ridge_selected_name=ridge_result["selected_alpha_name"], huber_valid=True)
    original = {"family_selection": family_result}
    inner = {"inner_results": inner_results, "huber_valid": True}
    ridge_cross_check = {"replay_result": ridge_result}
    result = replay.phase8_family_selection_cross_check(inner, ridge_cross_check, original)
    assert result["replay_result"]["selected_family"] == family_result["selected_family"]


def test_replay_cannot_silently_accept_a_different_winner(inner_results, monkeypatch):
    """If a replayed cross-check would select a DIFFERENT family than the
    frozen winner, the replay must HARD STOP -- never silently swap in the
    new winner."""
    ridge_result = mfs.select_ridge_alpha(inner_results)
    family_result = mfs.select_family(inner_results, ridge_selected_name=ridge_result["selected_alpha_name"], huber_valid=True)
    # Force EXPECTED_SELECTED_FAMILY to something the synthetic data will
    # NOT reproduce (real selected_family here is whatever the synthetic
    # fixture happens to pick -- assert the guard fires whenever the two
    # disagree, regardless of which family that is).
    other_family = next(f for f in ("RIDGE", "HUBER", "HGBR") if f != family_result["selected_family"])
    monkeypatch.setattr(replay, "EXPECTED_SELECTED_FAMILY", other_family)
    original = {"family_selection": family_result}
    inner = {"inner_results": inner_results, "huber_valid": True}
    ridge_cross_check = {"replay_result": ridge_result}
    with pytest.raises(replay.ReplayMismatch):
        replay.phase8_family_selection_cross_check(inner, ridge_cross_check, original)


# ---------------------------------------------------------------------------
# 2024 metric reconstruction (MAE) alongside originally-persisted RMSE
# ---------------------------------------------------------------------------
def test_2024_style_finalist_report_includes_reconstructed_mae(elo_matrix):
    matrix, feature_columns = elo_matrix
    spec = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "RIDGE_ALPHA_1")
    preds, invalid = mfs.fit_predict_fold(matrix, feature_columns, fd.OUTER_FOLD, spec)
    assert not invalid
    report = mfs._regression_report(preds)
    # RMSE (originally persisted quantity in the real Fix 7 run) and MAE
    # (never persisted) both come out of the SAME report -- proving the gap
    # was a reporting omission, not a missing computation.
    assert np.isfinite(report["margin"]["rmse"]) and np.isfinite(report["margin"]["mae"])
    assert np.isfinite(report["total"]["rmse"]) and np.isfinite(report["total"]["mae"])


# ---------------------------------------------------------------------------
# Audit-rule reproduction (PASS/FAIL detection)
# ---------------------------------------------------------------------------
def test_audit_rule_replay_detects_pass():
    per_finalist = {
        "RIDGE": {"primary_score": 10.0, "margin_RMSE": 10.0, "total_RMSE": 10.0, "segment_primary_score": {"week1": 10.0, "weeks2_4": 10.0, "weeks5_plus": 10.0}},
        "HUBER": {"primary_score": 10.05, "margin_RMSE": 10.05, "total_RMSE": 10.05, "segment_primary_score": {"week1": 10.05, "weeks2_4": 10.05, "weeks5_plus": 10.05}},
        "HGBR": {"primary_score": 10.1, "margin_RMSE": 10.1, "total_RMSE": 10.1, "segment_primary_score": {"week1": 10.1, "weeks2_4": 10.1, "weeks5_plus": 10.1}},
    }
    finalist_replay = {"per_finalist": per_finalist}
    original = {"locked_audit": {"rules": {
        "overall": {"selected": 10.0, "best": 10.0, "pass": True}, "margin": {"selected": 10.0, "best": 10.0, "pass": True}, "total": {"selected": 10.0, "best": 10.0, "pass": True},
        "week1": {"selected": 10.0, "best": 10.0, "pass": True}, "weeks2_4": {"selected": 10.0, "best": 10.0, "pass": True}, "weeks5_plus": {"selected": 10.0, "best": 10.0, "pass": True},
    }}}
    saved = replay.EXPECTED_SELECTED_FAMILY
    try:
        replay.EXPECTED_SELECTED_FAMILY = "RIDGE"
        result = replay.phase10_recheck_audit_rules(finalist_replay, original)
    finally:
        replay.EXPECTED_SELECTED_FAMILY = saved
    assert result["overall_pass"] is True


def test_audit_rule_replay_raises_on_failure():
    per_finalist = {
        "RIDGE": {"primary_score": 50.0, "margin_RMSE": 50.0, "total_RMSE": 50.0, "segment_primary_score": {"week1": 50.0, "weeks2_4": 50.0, "weeks5_plus": 50.0}},
        "HUBER": {"primary_score": 10.0, "margin_RMSE": 10.0, "total_RMSE": 10.0, "segment_primary_score": {"week1": 10.0, "weeks2_4": 10.0, "weeks5_plus": 10.0}},
        "HGBR": {"primary_score": 10.1, "margin_RMSE": 10.1, "total_RMSE": 10.1, "segment_primary_score": {"week1": 10.1, "weeks2_4": 10.1, "weeks5_plus": 10.1}},
    }
    finalist_replay = {"per_finalist": per_finalist}
    original = {"locked_audit": {"rules": {
        "overall": {"selected": 50.0, "best": 10.0, "pass": False}, "margin": {"selected": 50.0, "best": 10.0, "pass": False}, "total": {"selected": 50.0, "best": 10.0, "pass": False},
        "week1": {"selected": 50.0, "best": 10.0, "pass": False}, "weeks2_4": {"selected": 50.0, "best": 10.0, "pass": False}, "weeks5_plus": {"selected": 50.0, "best": 10.0, "pass": False},
    }}}
    saved = replay.EXPECTED_SELECTED_FAMILY
    try:
        replay.EXPECTED_SELECTED_FAMILY = "RIDGE"
        with pytest.raises(replay.ReplayMismatch):
            replay.phase10_recheck_audit_rules(finalist_replay, original)
    finally:
        replay.EXPECTED_SELECTED_FAMILY = saved


# ---------------------------------------------------------------------------
# Fit counters kept separate from the original preregistered budget
# ---------------------------------------------------------------------------
def test_replay_fit_counters_are_separate_from_original_budget():
    assert mfs.EXPECTED_TOTAL_PAIRED_FITS == 21
    assert mfs.EXPECTED_TOTAL_INDIVIDUAL_FITS == 42
    # The replay script must never mutate these module-level constants.
    assert mfs.EXPECTED_INNER_PAIRED_FITS == 18
    assert mfs.EXPECTED_OUTER_PAIRED_FITS == 3


# ---------------------------------------------------------------------------
# Firewall: no season >= 2025 survives, in a synthetic fixture that
# deliberately contains 2025/2026 rows
# ---------------------------------------------------------------------------
def test_no_season_ge_2025_after_firewall():
    games_with_future = _synthetic_games(extra_seasons=(2025, 2026))
    assert (games_with_future["season"] >= 2025).sum() > 0  # sanity: future rows really are present pre-firewall
    firewalled, counts = fd.enforce_2025_firewall(games_with_future)
    assert (firewalled["season"] >= 2025).sum() == 0
    assert counts["rows_season_ge_2025_passed_beyond_firewall"] == 0


# ---------------------------------------------------------------------------
# Replay artifact canonical hash determinism
# ---------------------------------------------------------------------------
def test_replay_evidence_hash_deterministic():
    payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    h1 = mfs._sha256_hex(payload)
    h2 = mfs._sha256_hex(payload)
    assert h1 == h2


def test_replay_evidence_hash_sensitive_to_change():
    payload_a = {"a": 1}
    payload_b = {"a": 2}
    assert mfs._sha256_hex(payload_a) != mfs._sha256_hex(payload_b)


# ---------------------------------------------------------------------------
# Reporting-only hardening leaves final_model_spec_hash unchanged: this
# module was NOT edited by the replay work (Section 18's narrow-proof bar
# was not attempted -- the replay stays in a separate script instead), so
# the frozen hash function must still reproduce the exact frozen value for
# the exact frozen spec.
# ---------------------------------------------------------------------------
def test_final_model_spec_hash_function_unchanged_for_frozen_spec():
    selected_spec = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "RIDGE_ALPHA_100")
    frozen_feature_columns = (
        "home_elo_pregame_rating", "home_elo_pregame_win_probability", "home_elo_pregame_expected_margin",
        "away_elo_pregame_rating", "away_elo_pregame_win_probability", "away_elo_pregame_expected_margin",
    )
    recomputed = mfs.compute_final_model_spec_hash(
        fix6_feature_manifest_hash="d523a007f7babfd0e8913f7ad6204f3670120d5d0f36183ecead9e153a8b24bf",
        frozen_feature_columns=frozen_feature_columns, selected_family="RIDGE", selected_spec=selected_spec,
    )
    assert recomputed == "418724cb32fcb9a6a58f44fc77a7d579dbbc64d8078991c13d363751892ddede"


def test_candidate_registry_hash_function_unchanged():
    assert mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY) == "360e40e8a541c3a4480a576813c67ec13bf5dafc8d198dc8b8ca05e675cc9b58"
