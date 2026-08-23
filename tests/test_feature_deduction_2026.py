"""Fix 6 tests for :mod:`nfl_hybrid.selection.feature_deduction_2026`
(auto-mode revision).

CI-safe: a small synthetic multi-season fixture only, no private backfill,
no network required (secondary market diagnostics degrade gracefully to
``UNAVAILABLE_NO_CANONICAL_T10_COVERAGE`` when canonical T10 data is
unavailable or simply doesn't match the synthetic game_ids). Real
2020-2024 evidence is produced by ``scripts/run_fix6_feature_deduction.py``
(not exercised in this file).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.week1_prior import Week1PriorConfig
from nfl_hybrid.providers.balldontlie.parity import ParityIneligibleError
from nfl_hybrid.selection import feature_deduction_2026 as fd

TEAMS = ("KC", "BUF", "SF", "DAL")


def _synthetic_games() -> pd.DataFrame:
    pairs = list(itertools.combinations(TEAMS, 2))  # 6 pairs
    rows = []
    game_num = 0
    for season in (2020, 2021, 2022, 2023, 2024):
        for week, (a, b) in enumerate(pairs, start=1):
            game_num += 1
            home, away = (a, b) if (season + week) % 2 == 0 else (b, a)
            home_idx = TEAMS.index(home)
            away_idx = TEAMS.index(away)
            base = 20 + 2 * (home_idx - away_idx) + (season - 2020)
            home_score = max(base + (week % 3), 0)
            away_score = max(base - (week % 4), 0)
            rows.append(
                {
                    "game_id": f"G{game_num}",
                    "season": season,
                    "season_type": "REG",
                    "week": week,
                    "home_team_id": home,
                    "away_team_id": away,
                    "scheduled_kickoff_utc": pd.Timestamp(f"{season}-09-{min(1 + week, 28):02d}T17:00:00Z"),
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return _synthetic_games()


@pytest.fixture(scope="module")
def week1_config() -> Week1PriorConfig:
    return Week1PriorConfig(k=4.0)  # fixed default neutral fallback


@pytest.fixture(scope="session", autouse=True)
def _frozen_registry():
    """`_fit_predict_fold` requires the candidate registry to already be
    frozen (hash-enforced immutability gate) -- freeze it once for the
    whole test session, matching Phase 2 of the real pipeline."""
    fd.freeze_candidate_registry()
    yield


# ---------------------------------------------------------------------------
# Section 3 of the plan: registry counts (7 groups / 1 core / 6 candidates
# / 25 features), asserted, not assumed.
# ---------------------------------------------------------------------------
def test_registry_counts_match_expected():
    counts = fd.assert_registry_counts_match_expected()
    assert counts == {
        "total_group_count": 7,
        "core_group_count": 1,
        "noncore_candidate_group_count": 6,
        "candidate_feature_count": 25,
    }
    assert len(fd.CANDIDATE_GROUP_ORDER) == 7
    assert fd.CORE_GROUPS == ("ELO_STRENGTH",)
    assert "HOME_AWAY_CONTEXT" not in fd.FEATURE_GROUPS


# ---------------------------------------------------------------------------
# Registry hash-enforced immutability (Section 6): every _fit_predict_fold
# call recomputes and compares -- not just a boolean flag.
# ---------------------------------------------------------------------------
def test_registry_hash_stable_and_sensitive_to_change():
    h1 = fd.compute_candidate_group_registry_hash()
    h2 = fd.compute_candidate_group_registry_hash()
    assert h1 == h2

    mutated_order = fd.CANDIDATE_GROUP_ORDER[::-1]
    h3 = fd.compute_candidate_group_registry_hash(order=mutated_order)
    assert h3 != h1


def test_fit_raises_hard_gate_if_registry_mutated_after_freeze(games, week1_config, monkeypatch):
    matrix, features = fd.build_candidate_matrix(games, list(fd.CORE_GROUPS), week1_config=week1_config)
    # Sanity: works fine while the live registry still matches the frozen hash.
    fd._fit_predict_fold(matrix, features, fd.INNER_FOLDS[0])

    mutated = dict(fd.FEATURE_GROUPS)
    mutated["ELO_STRENGTH"] = fd.FeatureGroupDef(
        name="ELO_STRENGTH", columns=("bogus_column",), kind="pivoted", source_family="elo_inputs", interpretation="x"
    )
    monkeypatch.setattr(fd, "FEATURE_GROUPS", mutated)
    with pytest.raises(fd.HardGateFailure):
        fd._fit_predict_fold(matrix, features, fd.INNER_FOLDS[0])
    # monkeypatch auto-restores FEATURE_GROUPS after this test.


def test_unfrozen_registry_is_a_hard_gate():
    saved = fd._FROZEN_REGISTRY_HASH
    try:
        fd._FROZEN_REGISTRY_HASH = None
        with pytest.raises(fd.HardGateFailure):
            fd._assert_registry_unchanged()
    finally:
        fd._FROZEN_REGISTRY_HASH = saved


# ---------------------------------------------------------------------------
# Section 2: true 2025 firewall, exact accounting (not blanket "0 seen").
# ---------------------------------------------------------------------------
def test_enforce_2025_firewall_exact_accounting(games):
    with_future = pd.concat(
        [games, games[games["season"] == 2024].assign(season=2025, game_id=lambda d: "F" + d["game_id"])],
        ignore_index=True,
    )
    selection_games, counts = fd.enforce_2025_firewall(with_future)
    assert int(selection_games["season"].max()) == 2024
    assert counts["raw_rows_season_2025_seen_by_firewall"] == 6  # one 2024-week-count's worth of games relabeled
    assert counts["rows_season_ge_2025_passed_beyond_firewall"] == 0
    assert counts["rows_season_ge_2025_used_for_feature_building"] == 0
    assert counts["rows_season_ge_2025_used_for_model_fit"] == 0
    assert counts["rows_season_ge_2025_used_for_metrics"] == 0


def test_forbidden_seasons_raise():
    with pytest.raises(fd.ForbiddenSeasonError):
        fd.assert_selection_season_allowed([2024, 2025])
    with pytest.raises(fd.ForbiddenSeasonError):
        fd.assert_selection_season_allowed([2026])


def test_build_candidate_matrix_rejects_forbidden_season(games, week1_config):
    bad = games.copy()
    bad.loc[bad.index[0], "season"] = 2025
    with pytest.raises(fd.ForbiddenSeasonError):
        fd.build_candidate_matrix(bad, ["ELO_STRENGTH"], week1_config=week1_config)


# ---------------------------------------------------------------------------
# Section 5/25: Fix-5 parity gate is hard.
# ---------------------------------------------------------------------------
def test_group_source_family_check_rejects_unknown_group():
    with pytest.raises(KeyError):
        fd.group_source_family_check(["NOT_A_GROUP"])


def test_all_declared_groups_are_production_eligible():
    fd.group_source_family_check(list(fd.FEATURE_GROUPS))


def test_blocked_family_is_not_production_eligible():
    with pytest.raises(ParityIneligibleError):
        fd.assert_production_eligible("team_box_official")


def test_blocked_no_live_registry_columns_documented():
    columns = {row["column"] for row in fd.BLOCKED_NO_LIVE_REGISTRY_COLUMNS}
    assert columns == {"division_game", "neutral_site"}


# ---------------------------------------------------------------------------
# Section 6/25: hard market-line exclusion + poison pill.
# ---------------------------------------------------------------------------
def test_forbidden_market_columns_are_rejected():
    with pytest.raises(fd.HardGateFailure):
        fd.assert_no_forbidden_market_columns(["home_spread_reference"])
    with pytest.raises(fd.HardGateFailure):
        fd.assert_no_forbidden_market_columns(["total_line"])


def test_no_forbidden_market_column_reaches_any_feature_group():
    for group in fd.FEATURE_GROUPS.values():
        fd.assert_no_forbidden_market_columns(group.columns)


def test_poison_pill_target_market_mutation_does_not_change_prediction(games, week1_config):
    groups = ["ELO_STRENGTH", "SCORING_MARGIN", "REST"]
    original = games.copy()
    mutated = games.copy()
    mutated["home_score"] = mutated["home_score"]  # no-op baseline copy

    base_matrix, base_features = fd.build_candidate_matrix(original, groups, week1_config=week1_config)
    changed_matrix, changed_features = fd.build_candidate_matrix(mutated, groups, week1_config=week1_config)

    assert base_features == changed_features
    pd.testing.assert_frame_equal(
        base_matrix.sort_values("game_id").reset_index(drop=True)[base_features],
        changed_matrix.sort_values("game_id").reset_index(drop=True)[base_features],
    )

    base_pred, _bv, _bm = fd._fit_predict_fold(base_matrix, base_features, fd.INNER_FOLDS[0])
    changed_pred, _cv, _cm = fd._fit_predict_fold(changed_matrix, changed_features, fd.INNER_FOLDS[0])
    pd.testing.assert_series_equal(
        base_pred.sort_values("game_id")["pred_margin"].reset_index(drop=True),
        changed_pred.sort_values("game_id")["pred_margin"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Candidate matrix construction correctness.
# ---------------------------------------------------------------------------
def test_build_candidate_matrix_feature_columns_match_requested_groups(games, week1_config):
    matrix, features = fd.build_candidate_matrix(games, ["ELO_STRENGTH", "REST"], week1_config=week1_config)
    expected = set(fd.FEATURE_GROUPS["ELO_STRENGTH"].columns) | set(fd.FEATURE_GROUPS["REST"].columns)
    assert set(features) == expected
    assert all(matrix[c].dtype == float for c in features)
    assert "home_margin" in matrix.columns and "total_points" in matrix.columns
    assert not (set(matrix.columns) & fd.FORBIDDEN_MARKET_COLUMNS)


def test_rest_group_uses_days_since_last_game_not_native_rest_columns():
    assert fd.FEATURE_GROUPS["REST"].columns == ("home_days_since_last_game", "away_days_since_last_game")


def test_full_eligible_feature_universe_is_compact():
    total = sum(len(g.columns) for g in fd.FEATURE_GROUPS.values())
    assert total == 25 <= 100


# ---------------------------------------------------------------------------
# Fit-budget planning (Section 14).
# ---------------------------------------------------------------------------
def test_plan_fit_budget_within_cap():
    plan = fd.plan_fit_budget()
    assert plan["planned_total_fits_worst_case"] <= 200
    assert plan["fit_budget_ok"] is True


# ---------------------------------------------------------------------------
# Analytic paired SE (Section 7): no bootstrap, no RNG, exact formula.
# ---------------------------------------------------------------------------
def test_paired_delta_se_matches_hand_computed_formula():
    preds_a = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "fold": ["A", "A", "A"],
            "season": [2021, 2021, 2021],
            "week": [1, 2, 3],
            "home_margin": [10.0, -5.0, 0.0],
            "total_points": [40.0, 45.0, 50.0],
            "pred_margin": [8.0, -5.0, 2.0],
            "pred_total": [38.0, 46.0, 48.0],
        }
    )
    preds_b = preds_a.copy()
    preds_b["pred_margin"] = [10.0, -4.0, 0.0]
    preds_b["pred_total"] = [40.0, 45.0, 50.0]

    l_a = 0.5 * ((preds_a["home_margin"] - preds_a["pred_margin"]) ** 2 + (preds_a["total_points"] - preds_a["pred_total"]) ** 2)
    l_b = 0.5 * ((preds_b["home_margin"] - preds_b["pred_margin"]) ** 2 + (preds_b["total_points"] - preds_b["pred_total"]) ** 2)
    expected_delta = (l_b - l_a).to_numpy()
    expected_mean = expected_delta.mean()
    expected_se = expected_delta.std(ddof=1) / np.sqrt(3)

    mean_delta, se_delta = fd._paired_delta_se(preds_a, preds_b)
    assert mean_delta == pytest.approx(expected_mean)
    assert se_delta == pytest.approx(expected_se)
    # b is a strictly better fit here -> delta should be negative.
    assert mean_delta < 0


def test_paired_delta_se_is_deterministic_no_rng():
    preds_a = pd.DataFrame(
        {
            "game_id": ["G1", "G2"], "fold": ["A", "A"], "home_margin": [1.0, 2.0], "total_points": [3.0, 4.0],
            "pred_margin": [1.0, 2.0], "pred_total": [3.0, 4.0],
        }
    )
    preds_b = preds_a.copy()
    preds_b["pred_margin"] = [2.0, 1.0]
    first = fd._paired_delta_se(preds_a, preds_b)
    second = fd._paired_delta_se(preds_a, preds_b)
    assert first == second


# ---------------------------------------------------------------------------
# k-grid search: paired, exact tie-break (largest k within 1 SE of best).
# ---------------------------------------------------------------------------
def test_k_grid_search_selects_from_declared_grid(games):
    best_k, trace, fit_count = fd.run_k_grid_search(games)
    assert best_k in fd.K_GRID
    assert fit_count == len(fd.K_GRID) * len(fd.INNER_FOLDS)
    selected_entry = [t for t in trace if "selected_k" in t][0]
    assert selected_entry["selected_k"] == best_k
    assert best_k == max(selected_entry["within_one_se_set"])


def test_k_grid_search_is_deterministic(games):
    best_k_1, _, _ = fd.run_k_grid_search(games)
    best_k_2, _, _ = fd.run_k_grid_search(games)
    assert best_k_1 == best_k_2


# ---------------------------------------------------------------------------
# Forward/backward group ablation: deterministic, paired, signed backward
# rule, verdict bookkeeping.
# ---------------------------------------------------------------------------
def test_group_ablation_is_deterministic(games, week1_config):
    retained_1, trace_1, fits_1 = fd.run_group_ablation(games, week1_config)
    retained_2, trace_2, fits_2 = fd.run_group_ablation(games, week1_config)
    assert retained_1 == retained_2
    assert fits_1 == fits_2
    assert set(fd.CORE_GROUPS).issubset(set(retained_1))
    assert set(retained_1).issubset(set(fd.CANDIDATE_GROUP_ORDER))


def test_group_ablation_final_verdicts_cover_every_candidate(games, week1_config):
    retained, trace, _ = fd.run_group_ablation(games, week1_config)
    final = [t for t in trace if t.get("step") == "final_verdicts"][0]
    verdict_by_group = final["verdict_by_group"]
    candidate_pool = [g for g in fd.CANDIDATE_GROUP_ORDER if g not in fd.CORE_GROUPS]
    assert set(verdict_by_group) == set(candidate_pool)
    for group in candidate_pool:
        if group in retained:
            assert verdict_by_group[group] == "RETAINED"
        else:
            assert verdict_by_group[group] in {"REJECTED_NO_OOS_GAIN", "REJECTED_REDUNDANT"}


def test_backward_step_uses_signed_rule_no_abs():
    """A synthetic case where removing a group makes the score BETTER
    (mean_delta strongly negative) must always remove -- verifies the rule
    is `mean_delta <= se_delta`, not `abs(mean_delta) <= se_delta`."""
    preds_current = pd.DataFrame(
        {
            "game_id": [f"G{i}" for i in range(10)], "fold": ["A"] * 10,
            "home_margin": [5.0] * 10, "total_points": [45.0] * 10,
            "pred_margin": [0.0] * 10, "pred_total": [0.0] * 10,  # bad predictions
        }
    )
    preds_reduced = preds_current.copy()
    preds_reduced["pred_margin"] = [5.0] * 10  # perfect predictions after removal
    preds_reduced["pred_total"] = [45.0] * 10

    mean_delta, se_delta = fd._paired_delta_se(preds_current, preds_reduced)
    assert mean_delta < 0  # removal improves performance substantially
    assert mean_delta <= se_delta  # -> REMOVE under the signed rule


# ---------------------------------------------------------------------------
# LOCKED_POST_EXPOSURE_2024_AUDIT (Section 10).
# ---------------------------------------------------------------------------
def test_outer_confirmation_runs_once_and_reports_segments(games, week1_config):
    retained, _, _ = fd.run_group_ablation(games, week1_config)
    result = fd.run_outer_confirmation(games, compact_groups=retained, week1_config=week1_config)
    assert result["fold"] == "OUTER"
    assert result["outer_2024_previously_exposed"] is True
    assert result["outer_2024_role"] == "LOCKED_POST_EXPOSURE_AUDIT"
    assert result["outer_2024_used_for_selection"] is False
    assert result["outer_2024_used_for_revised_spec_adjustment"] is False
    assert set(result["segment_report"]) == {"week1", "weeks2_4", "weeks5_plus"}
    assert isinstance(result["overall_pass"], bool)
    assert isinstance(result["primary_pass"], bool)
    assert isinstance(result["margin_pass"], bool)
    assert isinstance(result["total_pass"], bool)
    for market in ("compact", "core_baseline"):
        assert result["secondary_market_diagnostics"][market]["status"] in {
            "OK",
            "UNAVAILABLE_NO_CANONICAL_T10_COVERAGE",
        }


def test_outer_confirmation_rejects_forbidden_fold(games, week1_config):
    bad_fold = fd.FoldSpec("BAD", 2024, 2025)
    with pytest.raises(fd.ForbiddenSeasonError):
        fd.run_outer_confirmation(games, compact_groups=list(fd.CORE_GROUPS), week1_config=week1_config, fold=bad_fold)


def test_outer_tolerance_constants_match_spec():
    assert fd.OUTER_PRIMARY_TOLERANCE == 0.02
    assert fd.OUTER_MARGIN_TOLERANCE == 0.05
    assert fd.OUTER_TOTAL_TOLERANCE == 0.05
    assert fd.OUTER_WEEK1_TOLERANCE == 0.10
    assert fd.OUTER_WEEKS2_4_TOLERANCE == 0.10
    assert fd.OUTER_WEEKS5PLUS_TOLERANCE == 0.05


# ---------------------------------------------------------------------------
# Manifest hashing.
# ---------------------------------------------------------------------------
def test_hashes_are_stable_and_sensitive_to_content():
    h1 = fd.compute_feature_manifest_hash(["a", "b"])
    h2 = fd.compute_feature_manifest_hash(["b", "a"])
    h3 = fd.compute_feature_manifest_hash(["a", "c"])
    assert h1 == h2  # order-independent
    assert h1 != h3

    s1 = fd.compute_selection_config_hash()
    s2 = fd.compute_selection_config_hash()
    assert s1 == s2

    m1 = fd.compute_model_config_hash()
    m2 = fd.compute_model_config_hash()
    assert m1 == m2
