"""Fix 7 tests for :mod:`nfl_hybrid.selection.model_family_selection_2026`.

CI-safe: a small synthetic multi-season fixture only (same shape as Fix 6's
own fixture in ``test_feature_deduction_2026.py``), no private backfill, no
network. Real 2020-2024 evidence is produced by
``scripts/run_fix7_model_family_selection.py`` (not exercised here).
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features.week1_prior import Week1PriorConfig
from nfl_hybrid.selection import feature_deduction_2026 as fd
from nfl_hybrid.selection import model_family_selection_2026 as mfs

TEAMS = ("KC", "BUF", "SF", "DAL")


def _synthetic_games() -> pd.DataFrame:
    pairs = list(itertools.combinations(TEAMS, 2))  # 6 pairs -> 6 games/season, weeks 1-6
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
    return Week1PriorConfig(k=8.0)  # matches the committed Fix 6 k; irrelevant to ELO_STRENGTH itself


@pytest.fixture(scope="module")
def elo_matrix(games, week1_config):
    matrix, features = fd.build_candidate_matrix(games, ["ELO_STRENGTH"], week1_config=week1_config)
    return matrix, features


HGBR_CANDIDATE = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HGBR_INCUMBENT")
RIDGE_CANDIDATE = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "RIDGE_ALPHA_1")
HUBER_CANDIDATE = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HUBER_FIXED")


# ---------------------------------------------------------------------------
# Correction #1: Fix-6 feature contract -- committed evidence authoritative,
# live registry an independent cross-check.
# ---------------------------------------------------------------------------
def test_fix6_contract_passes_against_real_committed_evidence():
    import json
    from nfl_hybrid.data.external_data import REPO_ROOT

    summary = json.loads((REPO_ROOT / "outputs" / "fix6_feature_selection_summary.json").read_text())
    columns, feature_hash = mfs.load_and_verify_fix6_feature_contract(summary)
    assert columns == mfs.REQUIRED_ORDER
    assert feature_hash == summary["feature_manifest_hash"]


def test_fix6_contract_rejects_reordered_committed_evidence():
    bad_summary = {
        "final_features": list(reversed(mfs.REQUIRED_ORDER)),
        "feature_manifest_hash": "irrelevant",
    }
    with pytest.raises(fd.HardGateFailure, match="COMMITTED_FIX6_EVIDENCE_UNEXPECTED_ORDER"):
        mfs.load_and_verify_fix6_feature_contract(bad_summary)


def test_fix6_contract_rejects_hash_mismatch():
    bad_summary = {
        "final_features": list(mfs.REQUIRED_ORDER),
        "feature_manifest_hash": "0" * 64,
    }
    with pytest.raises(fd.HardGateFailure, match="COMMITTED_FIX6_HASH_MISMATCH"):
        mfs.load_and_verify_fix6_feature_contract(bad_summary)


def test_fix6_contract_rejects_live_registry_divergence(monkeypatch):
    good_summary = {
        "final_features": list(mfs.REQUIRED_ORDER),
        "feature_manifest_hash": fd.compute_feature_manifest_hash(mfs.REQUIRED_ORDER),
    }
    mutated = dict(fd.FEATURE_GROUPS)
    mutated["ELO_STRENGTH"] = fd.FeatureGroupDef(
        name="ELO_STRENGTH", columns=("bogus",), kind="pivoted", source_family="elo_inputs", interpretation="x"
    )
    monkeypatch.setattr(fd, "FEATURE_GROUPS", mutated)
    with pytest.raises(fd.HardGateFailure, match="LIVE_FIX6_REGISTRY_DIVERGED_FROM_COMMITTED_EVIDENCE"):
        mfs.load_and_verify_fix6_feature_contract(good_summary)


# ---------------------------------------------------------------------------
# Correction #2: HGBR random_state semantics.
# ---------------------------------------------------------------------------
def test_hgbr_random_state_matches_committed_config():
    hgbr = next(c for c in mfs.CANDIDATE_REGISTRY if c.name == "HGBR_INCUMBENT")
    assert hgbr.random_state == fd.MODEL_CONFIG.random_state == 42
    mfs.assert_hgbr_random_state_matches_committed_config()  # must not raise


def test_hgbr_random_state_mismatch_is_a_hard_gate():
    mutated_hgbr = replace(HGBR_CANDIDATE, random_state=None)
    mutated_registry = tuple(mutated_hgbr if c.name == "HGBR_INCUMBENT" else c for c in mfs.CANDIDATE_REGISTRY)
    with pytest.raises(fd.HardGateFailure, match="HGBR_RANDOM_STATE_MISMATCH"):
        mfs.assert_hgbr_random_state_matches_committed_config(mutated_registry)


def test_final_model_spec_hash_reports_42_not_none_for_hgbr():
    h = mfs.compute_final_model_spec_hash(
        fix6_feature_manifest_hash="abc", frozen_feature_columns=mfs.REQUIRED_ORDER,
        selected_family="HGBR", selected_spec=HGBR_CANDIDATE,
    )
    h_wrong = mfs.compute_final_model_spec_hash(
        fix6_feature_manifest_hash="abc", frozen_feature_columns=mfs.REQUIRED_ORDER,
        selected_family="HGBR", selected_spec=replace(HGBR_CANDIDATE, random_state=None),
    )
    assert h != h_wrong  # the hash is sensitive to random_state, so 42 vs None cannot be silently equivalent


# ---------------------------------------------------------------------------
# Candidate spec / registry hashing: stable, sensitive to real content
# changes (family/complexity/preprocessing/hyperparameters/random_state).
# ---------------------------------------------------------------------------
def test_candidate_spec_hash_stable_and_sensitive():
    h1 = mfs.compute_candidate_spec_hash(RIDGE_CANDIDATE)
    h2 = mfs.compute_candidate_spec_hash(RIDGE_CANDIDATE)
    assert h1 == h2
    h3 = mfs.compute_candidate_spec_hash(replace(RIDGE_CANDIDATE, hyperparameters={**RIDGE_CANDIDATE.hyperparameters, "alpha": 999.0}))
    assert h3 != h1


def test_candidate_registry_hash_stable_and_sensitive():
    h1 = mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY)
    h2 = mfs.compute_candidate_registry_hash(mfs.CANDIDATE_REGISTRY)
    assert h1 == h2
    mutated = mfs.CANDIDATE_REGISTRY[:-1] + (replace(HGBR_CANDIDATE, random_state=1),)
    assert mfs.compute_candidate_registry_hash(mutated) != h1


# ---------------------------------------------------------------------------
# Preprocessing contract: scaler fit on TRAIN rows only, deterministic
# Ridge/HGBR predictions.
# ---------------------------------------------------------------------------
def test_ridge_predictions_are_deterministic(elo_matrix):
    matrix, features = elo_matrix
    preds1, invalid1 = mfs.fit_predict_fold(matrix, features, fd.INNER_FOLDS[0], RIDGE_CANDIDATE)
    preds2, invalid2 = mfs.fit_predict_fold(matrix, features, fd.INNER_FOLDS[0], RIDGE_CANDIDATE)
    assert invalid1 == invalid2 == []
    pd.testing.assert_frame_equal(
        preds1.sort_values("game_id").reset_index(drop=True), preds2.sort_values("game_id").reset_index(drop=True)
    )


def test_hgbr_predictions_are_deterministic(elo_matrix):
    matrix, features = elo_matrix
    preds1, invalid1 = mfs.fit_predict_fold(matrix, features, fd.INNER_FOLDS[0], HGBR_CANDIDATE)
    preds2, invalid2 = mfs.fit_predict_fold(matrix, features, fd.INNER_FOLDS[0], HGBR_CANDIDATE)
    assert invalid1 == invalid2 == []
    pd.testing.assert_frame_equal(
        preds1.sort_values("game_id").reset_index(drop=True), preds2.sort_values("game_id").reset_index(drop=True)
    )


def test_hgbr_incumbent_config_unchanged_from_committed_default():
    from dataclasses import asdict

    assert HGBR_CANDIDATE.hyperparameters == asdict(fd.MODEL_CONFIG)


def test_scaler_fit_on_train_rows_only(elo_matrix):
    matrix, features = elo_matrix
    fold = fd.INNER_FOLDS[0]
    train = matrix[matrix["season"] <= fold.train_max_season]
    _preds, _invalid, bundle = mfs.fit_predict_fold(matrix, features, fold, RIDGE_CANDIDATE, return_fitted=True)
    scaler = bundle["margin"].named_steps["scaler"]
    manual_mean = train[features].to_numpy(float).mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, manual_mean)


# ---------------------------------------------------------------------------
# Huber INVALID_NUMERICAL: limited to exactly the three predeclared
# triggers; anything else is a real exception.
# ---------------------------------------------------------------------------
def test_huber_convergence_warning_is_classified_invalid_numerical(elo_matrix):
    matrix, features = elo_matrix
    fold = fd.INNER_FOLDS[0]
    train = matrix[matrix["season"] <= fold.train_max_season]
    validate = matrix[matrix["season"] == fold.validate_season]
    starved = replace(HUBER_CANDIDATE, hyperparameters={**HUBER_CANDIDATE.hyperparameters, "max_iter": 1})
    pipe, pred, ok, reason = mfs._fit_one_ridge_or_huber_target(
        "HUBER", starved.hyperparameters, train[features], train["home_margin"].to_numpy(float), validate[features]
    )
    assert ok is False
    assert reason == "CONVERGENCE_WARNING"


class _StubEstimator:
    def __init__(self, coef, intercept, pred):
        self.coef_ = coef
        self.intercept_ = intercept
        self._pred = pred

    def predict(self, X):
        return self._pred


class _StubPipeline:
    def __init__(self, estimator):
        self.named_steps = {"estimator": estimator}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.named_steps["estimator"].predict(X)


def test_huber_nonfinite_fitted_parameters_is_classified_invalid_numerical(monkeypatch):
    stub = _StubPipeline(_StubEstimator(coef=np.array([np.nan, 1.0]), intercept=0.0, pred=np.array([1.0])))
    monkeypatch.setattr(mfs, "_build_pipeline", lambda family, hp: stub)
    x_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    x_val = pd.DataFrame({"a": [1.0], "b": [2.0]})
    _pipe, _pred, ok, reason = mfs._fit_one_ridge_or_huber_target("HUBER", {}, x_train, np.array([1.0, 2.0]), x_val)
    assert ok is False
    assert reason == "NONFINITE_FITTED_PARAMETERS"


def test_huber_nonfinite_validation_prediction_is_classified_invalid_numerical(monkeypatch):
    stub = _StubPipeline(_StubEstimator(coef=np.array([1.0, 1.0]), intercept=0.0, pred=np.array([np.nan])))
    monkeypatch.setattr(mfs, "_build_pipeline", lambda family, hp: stub)
    x_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    x_val = pd.DataFrame({"a": [1.0], "b": [2.0]})
    _pipe, _pred, ok, reason = mfs._fit_one_ridge_or_huber_target("HUBER", {}, x_train, np.array([1.0, 2.0]), x_val)
    assert ok is False
    assert reason == "NONFINITE_VALIDATION_PREDICTION"


def test_ridge_nonfinite_prediction_is_a_hard_gate_not_invalid_numerical(monkeypatch):
    stub = _StubPipeline(_StubEstimator(coef=np.array([1.0]), intercept=0.0, pred=np.array([np.nan])))
    monkeypatch.setattr(mfs, "_build_pipeline", lambda family, hp: stub)
    x_train = pd.DataFrame({"a": [1.0, 2.0]})
    x_val = pd.DataFrame({"a": [1.0]})
    with pytest.raises(fd.HardGateFailure):
        mfs._fit_one_ridge_or_huber_target("RIDGE", {}, x_train, np.array([1.0, 2.0]), x_val)


# ---------------------------------------------------------------------------
# Paired analytic SE: hand-computed formula, unpairable ledgers hard-STOP.
# ---------------------------------------------------------------------------
def test_paired_delta_se_matches_hand_computed_formula():
    preds_a = pd.DataFrame(
        {"game_id": ["G1", "G2", "G3"], "fold": ["A", "A", "A"], "home_margin": [10.0, -5.0, 0.0],
         "total_points": [40.0, 45.0, 50.0], "pred_margin": [8.0, -5.0, 2.0], "pred_total": [38.0, 46.0, 48.0]}
    )
    preds_b = preds_a.copy()
    preds_b["pred_margin"] = [10.0, -4.0, 0.0]
    preds_b["pred_total"] = [40.0, 45.0, 50.0]

    l_a = 0.5 * ((preds_a["home_margin"] - preds_a["pred_margin"]) ** 2 + (preds_a["total_points"] - preds_a["pred_total"]) ** 2)
    l_b = 0.5 * ((preds_b["home_margin"] - preds_b["pred_margin"]) ** 2 + (preds_b["total_points"] - preds_b["pred_total"]) ** 2)
    expected_delta = (l_a - l_b).to_numpy()
    mean_delta, se_delta, n = mfs.paired_delta_se(preds_a, preds_b)
    assert n == 3
    assert mean_delta == pytest.approx(expected_delta.mean())
    assert se_delta == pytest.approx(expected_delta.std(ddof=1) / np.sqrt(3))
    assert mean_delta > 0  # b is strictly better here -> a's delta is positive


def test_paired_delta_se_unpairable_ledgers_hard_gate():
    preds_a = pd.DataFrame(
        {"game_id": ["G1", "G2"], "fold": ["A", "A"], "home_margin": [1.0, 2.0], "total_points": [3.0, 4.0],
         "pred_margin": [1.0, 2.0], "pred_total": [3.0, 4.0]}
    )
    preds_b = pd.DataFrame(
        {"game_id": ["G1", "G3"], "fold": ["A", "A"], "home_margin": [1.0, 2.0], "total_points": [3.0, 4.0],
         "pred_margin": [1.0, 2.0], "pred_total": [3.0, 4.0]}
    )
    with pytest.raises(fd.HardGateFailure, match="UNPAIRABLE_PREDICTION_LEDGERS"):
        mfs.paired_delta_se(preds_a, preds_b)


def test_paired_delta_se_is_deterministic_no_rng():
    preds_a = pd.DataFrame(
        {"game_id": ["G1", "G2"], "fold": ["A", "A"], "home_margin": [1.0, 2.0], "total_points": [3.0, 4.0],
         "pred_margin": [1.0, 2.0], "pred_total": [3.0, 4.0]}
    )
    preds_b = preds_a.copy()
    preds_b["pred_margin"] = [2.0, 1.0]
    assert mfs.paired_delta_se(preds_a, preds_b) == mfs.paired_delta_se(preds_a, preds_b)


# ---------------------------------------------------------------------------
# Ridge alpha selection: largest-alpha-within-one-SE tie rule.
# ---------------------------------------------------------------------------
def _fake_predictions(game_ids, fold_name, actual, pred):
    return pd.DataFrame(
        {
            "game_id": game_ids, "fold": [fold_name] * len(game_ids),
            "home_margin": actual, "total_points": [40.0] * len(game_ids),
            "pred_margin": pred, "pred_total": [40.0] * len(game_ids),
        }
    )


def test_ridge_alpha_selects_largest_within_one_se():
    game_ids = [f"G{i}" for i in range(20)]
    actual = [5.0] * 20
    inner_results = {}
    # 0.1/1.0/10.0 are IDENTICAL predictions (delta=0, SE=0 vs best -> trivially
    # within one SE); 100.0 has a constant large offset (SE=0, mean_delta>0 ->
    # strictly excluded). best_alpha ties among 0.1/1/10 and resolves to the
    # first in registry order (0.1). Expect selected = largest qualifying = 10.0.
    preds_by_alpha = {
        "RIDGE_ALPHA_0_1": [5.0] * 20,
        "RIDGE_ALPHA_1": [5.0] * 20,
        "RIDGE_ALPHA_10": [5.0] * 20,
        "RIDGE_ALPHA_100": [0.0] * 20,
    }
    for name, pred in preds_by_alpha.items():
        inner_results[name] = {"A": _fake_predictions(game_ids, "A", actual, pred)}
    # collapse the 3-inner-fold expectation to a single pooled fold for this synthetic check
    orig_inner_folds = fd.INNER_FOLDS
    import unittest.mock as mock
    with mock.patch.object(fd, "INNER_FOLDS", (fd.FoldSpec("A", 2020, 2021),)):
        result = mfs.select_ridge_alpha(inner_results)
    assert result["best_alpha_name"] == "RIDGE_ALPHA_0_1"
    assert result["selected_alpha_name"] == "RIDGE_ALPHA_10"
    assert result["selected_alpha_value"] == 10.0
    assert result["per_alpha"]["RIDGE_ALPHA_100"]["within_one_se_of_best"] is False


# ---------------------------------------------------------------------------
# Family selection: simplest-within-one-SE default, 2-of-3-fold promotion.
# ---------------------------------------------------------------------------
def test_family_selection_simple_wins_by_default():
    folds = [fd.FoldSpec("A", 2020, 2021), fd.FoldSpec("B", 2021, 2022), fd.FoldSpec("C", 2022, 2023)]
    inner_results = {"RIDGE_ALPHA_1": {}, "HGBR_INCUMBENT": {}}
    for i, fold in enumerate(folds):
        game_ids = [f"G{i}_{j}" for j in range(10)]
        actual = [5.0] * 10
        # Identical (perfect) predictions -> delta=0, SE=0 exactly -> trivially
        # within one SE of each other. RIDGE (lower complexity) must win by default.
        inner_results["RIDGE_ALPHA_1"][fold.name] = _fake_predictions(game_ids, fold.name, actual, [5.0] * 10)
        inner_results["HGBR_INCUMBENT"][fold.name] = _fake_predictions(game_ids, fold.name, actual, [5.0] * 10)
    import unittest.mock as mock
    with mock.patch.object(fd, "INNER_FOLDS", tuple(folds)):
        result = mfs.select_family(inner_results, ridge_selected_name="RIDGE_ALPHA_1", huber_valid=False)
    assert result["selected_family"] == "RIDGE"
    assert result["selection_reason"] == "TENTATIVE_IS_LOWEST_COMPLEXITY"


def test_family_selection_promotion_requires_2_of_3_folds():
    folds = [fd.FoldSpec("A", 2020, 2021), fd.FoldSpec("B", 2021, 2022), fd.FoldSpec("C", 2022, 2023)]
    inner_results = {"RIDGE_ALPHA_1": {}, "HGBR_INCUMBENT": {}}
    # HGBR wins fold A by a huge margin (dominating the POOLED comparison and
    # making it "best" overall) but loses folds B and C by a small margin ->
    # only 1/3 individual-fold wins against RIDGE. Promotion must FAIL and
    # fall back to RIDGE despite HGBR being the pooled-best/tentative family.
    ridge_pred_by_fold = [[0.0] * 10, [5.0] * 10, [5.0] * 10]
    hgbr_pred_by_fold = [[5.0] * 10, [4.9] * 10, [4.9] * 10]
    for i, fold in enumerate(folds):
        game_ids = [f"G{i}_{j}" for j in range(10)]
        actual = [5.0] * 10
        inner_results["RIDGE_ALPHA_1"][fold.name] = _fake_predictions(game_ids, fold.name, actual, ridge_pred_by_fold[i])
        inner_results["HGBR_INCUMBENT"][fold.name] = _fake_predictions(game_ids, fold.name, actual, hgbr_pred_by_fold[i])
    import unittest.mock as mock
    with mock.patch.object(fd, "INNER_FOLDS", tuple(folds)):
        result = mfs.select_family(inner_results, ridge_selected_name="RIDGE_ALPHA_1", huber_valid=False)
    # HGBR is pooled-better (tentative), but only wins 1/3 folds against RIDGE -> falls back to RIDGE.
    assert result["tentative"] == "HGBR"
    assert result["lower_family_checks"]["RIDGE"]["wins"] == 1
    assert result["selected_family"] == "RIDGE"
    assert result["selection_reason"] == "TENTATIVE_FAILED_2OF3_AGAINST_LOWEST_UNBEATEN_LOWER_FAMILY"


# ---------------------------------------------------------------------------
# Correction #3: target/future poison-pill tests, made non-vacuous.
# ---------------------------------------------------------------------------
def test_target_score_poison_pill_does_not_change_own_prediction(games, week1_config, elo_matrix):
    fold = fd.INNER_FOLDS[0]  # FoldSpec("A", train_max_season=2020, validate_season=2021)
    base_matrix, feature_columns = elo_matrix

    validation_games = base_matrix[base_matrix["season"] == fold.validate_season].sort_values("week")
    eligible_targets = validation_games.loc[validation_games["week"] < validation_games["week"].max()]
    assert not eligible_targets.empty, "FIXTURE_INADEQUATE: no game with a strictly later game in the same validation season"
    target_game_id = eligible_targets.iloc[0]["game_id"]
    assert target_game_id in validation_games["game_id"].values  # target really belongs to the tested fold

    mutated = games.copy()
    mutated.loc[mutated["game_id"] == target_game_id, ["home_score", "away_score"]] = [63, 0]
    mut_matrix, _ = fd.build_candidate_matrix(mutated, ["ELO_STRENGTH"], week1_config=week1_config)

    base_row = base_matrix.loc[base_matrix["game_id"] == target_game_id, feature_columns]
    mut_row = mut_matrix.loc[mut_matrix["game_id"] == target_game_id, feature_columns]
    pd.testing.assert_frame_equal(base_row.reset_index(drop=True), mut_row.reset_index(drop=True))

    # Fit ONCE on the ORIGINAL training rows; reuse the SAME fitted bundle for both predictions.
    _fold_preds, _invalid, fitted = mfs.fit_predict_fold(base_matrix, feature_columns, fold, HGBR_CANDIDATE, return_fitted=True)
    pred_base = mfs.predict_with_fitted_bundle(fitted, base_row, HGBR_CANDIDATE.family)
    pred_mut = mfs.predict_with_fitted_bundle(fitted, mut_row, HGBR_CANDIDATE.family)
    pd.testing.assert_frame_equal(pred_base, pred_mut)


def test_future_game_poison_pill_does_not_change_earlier_target(games, week1_config, elo_matrix):
    fold = fd.INNER_FOLDS[0]
    base_matrix, feature_columns = elo_matrix

    validation_games = base_matrix[base_matrix["season"] == fold.validate_season].sort_values("week")
    eligible_targets = validation_games.loc[validation_games["week"] < validation_games["week"].max()]
    assert not eligible_targets.empty
    target_row = eligible_targets.iloc[0]
    target_game_id = target_row["game_id"]

    later = validation_games.loc[validation_games["week"] > target_row["week"], "game_id"]
    assert not later.empty, "FIXTURE_INADEQUATE: no strictly-later game exists for the chosen target"
    future_game_id = later.iloc[0]

    mutated = games.copy()
    mutated.loc[mutated["game_id"] == future_game_id, ["home_score", "away_score"]] = [1, 59]
    mut_matrix, _ = fd.build_candidate_matrix(mutated, ["ELO_STRENGTH"], week1_config=week1_config)

    base_row = base_matrix.loc[base_matrix["game_id"] == target_game_id, feature_columns]
    mut_row = mut_matrix.loc[mut_matrix["game_id"] == target_game_id, feature_columns]
    pd.testing.assert_frame_equal(base_row.reset_index(drop=True), mut_row.reset_index(drop=True))

    _fold_preds, _invalid, fitted = mfs.fit_predict_fold(base_matrix, feature_columns, fold, HGBR_CANDIDATE, return_fitted=True)
    pred_base = mfs.predict_with_fitted_bundle(fitted, base_row, HGBR_CANDIDATE.family)
    pred_mut = mfs.predict_with_fitted_bundle(fitted, mut_row, HGBR_CANDIDATE.family)
    pd.testing.assert_frame_equal(pred_base, pred_mut)


def test_current_market_poison_pill(games, week1_config):
    original = games.copy()
    mutated = games.copy()
    mutated["home_spread"] = -3.5
    mutated["total_line"] = 47.5
    mutated.loc[mutated.index[0], "home_spread"] = 12.0  # drastic mutation on a fabricated market column
    mutated.loc[mutated.index[0], "total_line"] = 3.0

    base_matrix, base_features = fd.build_candidate_matrix(original, ["ELO_STRENGTH"], week1_config=week1_config)
    changed_matrix, changed_features = fd.build_candidate_matrix(mutated, ["ELO_STRENGTH"], week1_config=week1_config)
    assert base_features == changed_features
    pd.testing.assert_frame_equal(
        base_matrix.sort_values("game_id").reset_index(drop=True)[base_features],
        changed_matrix.sort_values("game_id").reset_index(drop=True)[base_features],
    )

    fold = fd.INNER_FOLDS[0]
    base_pred, _invalid = mfs.fit_predict_fold(base_matrix, base_features, fold, HGBR_CANDIDATE)
    changed_pred, _invalid2 = mfs.fit_predict_fold(changed_matrix, changed_features, fold, HGBR_CANDIDATE)
    pd.testing.assert_series_equal(
        base_pred.sort_values("game_id")["pred_margin"].reset_index(drop=True),
        changed_pred.sort_values("game_id")["pred_margin"].reset_index(drop=True),
    )


def test_poison_pill_fixture_inadequacy_is_detected():
    """If a fold's validation season has only one game, the target-selection
    guard must fail loudly rather than silently pass."""
    single_game = pd.DataFrame(
        {"game_id": ["ONLY"], "season": [2021], "week": [1]}
    )
    eligible = single_game.loc[single_game["week"] < single_game["week"].max()]
    assert eligible.empty  # matches the guard condition the real test raises on


# ---------------------------------------------------------------------------
# Fit-budget accounting.
# ---------------------------------------------------------------------------
def test_fit_budget_matches_expected_and_is_within_cap():
    plan = mfs.plan_fit_budget()
    assert plan["expected_paired"] == 21
    assert plan["expected_individual"] == 42
    assert plan["fit_budget_ok"] is True


def test_inner_fits_use_expected_paired_count(elo_matrix):
    matrix, features = elo_matrix
    inner_results, huber_invalid, paired, individual = mfs.run_inner_fits(matrix, features)
    assert paired == 18  # 6 candidates x 3 inner folds
    assert individual == 36
    assert set(inner_results) == {c.name for c in mfs.CANDIDATE_REGISTRY}


# ---------------------------------------------------------------------------
# Selection-matrix hash / cache key.
# ---------------------------------------------------------------------------
def test_selection_matrix_hash_stable_and_sensitive(elo_matrix):
    matrix, features = elo_matrix
    h1, n1, cols1 = mfs.compute_selection_matrix_hash(matrix, tuple(features))
    h2, n2, cols2 = mfs.compute_selection_matrix_hash(matrix, tuple(features))
    assert h1 == h2 and n1 == n2 and cols1 == cols2
    mutated = matrix.copy()
    mutated.loc[mutated.index[0], features[0]] = mutated.loc[mutated.index[0], features[0]] + 1.0
    h3, _n3, _c3 = mfs.compute_selection_matrix_hash(mutated, tuple(features))
    assert h3 != h1


def test_prediction_cache_hits_and_detects_mismatch(elo_matrix):
    matrix, features = elo_matrix
    fold = fd.INNER_FOLDS[0]
    cache = mfs.PredictionCache()
    key = mfs.make_cache_key(RIDGE_CANDIDATE, fold.name, "fh", "sh", "ph")
    expected_ids = frozenset(matrix.loc[matrix["season"] == fold.validate_season, "game_id"])
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return mfs.fit_predict_fold(matrix, features, fold, RIDGE_CANDIDATE)

    preds1, _inv1, cached1 = cache.get_or_fit(key, expected_ids, compute)
    preds2, _inv2, cached2 = cache.get_or_fit(key, expected_ids, compute)
    assert cached1 is False and cached2 is True
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# LOCKED_POST_EXPOSURE_2024_AUDIT thresholds.
# ---------------------------------------------------------------------------
def test_audit_tolerance_constants_match_spec():
    assert mfs.AUDIT_PRIMARY_TOLERANCE == 0.02
    assert mfs.AUDIT_MARGIN_TOLERANCE == 0.05
    assert mfs.AUDIT_TOTAL_TOLERANCE == 0.05
    assert mfs.AUDIT_WEEK1_TOLERANCE == 0.10
    assert mfs.AUDIT_WEEKS2_4_TOLERANCE == 0.10
    assert mfs.AUDIT_WEEKS5PLUS_TOLERANCE == 0.05


def test_locked_2024_audit_runs_and_reports_all_six_rules(elo_matrix):
    matrix, features = elo_matrix
    finalists = {"RIDGE": RIDGE_CANDIDATE, "HGBR": HGBR_CANDIDATE}
    result = mfs.run_locked_2024_audit(matrix, features, finalists=finalists, selected_family="HGBR")
    assert result["outer_2024_previously_exposed"] is True
    assert result["outer_2024_role"] == "LOCKED_POST_EXPOSURE_AUDIT"
    assert result["outer_2024_used_for_model_selection"] is False
    assert set(result["rules"]) == {"overall", "margin", "total", "week1", "weeks2_4", "weeks5_plus"}
    assert isinstance(result["overall_pass"], bool)
    assert "HGBR" in result["fitted_bundles"] and "RIDGE" in result["fitted_bundles"]


# ---------------------------------------------------------------------------
# Correction #5: HGBR interpretability adapter -- no fit, uses real sklearn
# permutation_importance.
# ---------------------------------------------------------------------------
def test_hgbr_permutation_importance_adapter_never_fits(elo_matrix):
    matrix, features = elo_matrix
    fold = fd.OUTER_FOLD
    _preds, _invalid, bundle = mfs.fit_predict_fold(matrix, features, fold, HGBR_CANDIDATE, return_fitted=True)
    audit_frame = matrix[matrix["season"] == fold.validate_season]

    adapter = mfs._MarginOnlyPredictor(bundle["model"])
    with pytest.raises(RuntimeError, match="must never be called"):
        adapter.fit(audit_frame[features])

    result = mfs.run_hgbr_permutation_importance(bundle["model"], audit_frame, features)
    assert set(result["margin"]["importances_mean"]) == set(features)
    assert set(result["total"]["importances_mean"]) == set(features)
    assert result["random_state"] == fd.MODEL_CONFIG.random_state


def test_ridge_interpretability_reports_elo_disclosure(elo_matrix):
    matrix, features = elo_matrix
    fold = fd.OUTER_FOLD
    _preds, _invalid, bundle = mfs.fit_predict_fold(matrix, features, fold, RIDGE_CANDIDATE, return_fitted=True)
    result = mfs.ridge_or_huber_interpretability(bundle, features)
    assert set(result["margin"]["standardized_coefficients"]) == set(features)
    assert result["elo_core_feature_relationship"] == mfs.ELO_CORE_FEATURE_RELATIONSHIP_TEXT


def test_local_sensitivity_reference_values_from_training_only(elo_matrix):
    matrix, features = elo_matrix
    fold = fd.OUTER_FOLD
    train_matrix = matrix[matrix["season"] <= fold.train_max_season]
    audit_matrix = matrix[matrix["season"] == fold.validate_season]
    _preds, _invalid, bundle = mfs.fit_predict_fold(matrix, features, fold, HGBR_CANDIDATE, return_fitted=True)
    result = mfs.local_sensitivity_example(bundle, HGBR_CANDIDATE.family, audit_matrix, train_matrix, features)
    assert result["label"] == "LOCAL_SENSITIVITY"
    assert result["not_shapley_attribution"] is True
    assert result["not_additive_contribution"] is True
    for feat in features:
        assert result["perturbation_deltas"][feat]["reference_value"] == pytest.approx(float(train_matrix[feat].median()))


# ---------------------------------------------------------------------------
# Deterministic replay of the final model spec hash.
# ---------------------------------------------------------------------------
def test_final_model_spec_hash_deterministic_replay():
    h1 = mfs.compute_final_model_spec_hash(
        fix6_feature_manifest_hash="abc", frozen_feature_columns=mfs.REQUIRED_ORDER,
        selected_family="RIDGE", selected_spec=RIDGE_CANDIDATE,
    )
    h2 = mfs.compute_final_model_spec_hash(
        fix6_feature_manifest_hash="abc", frozen_feature_columns=mfs.REQUIRED_ORDER,
        selected_family="RIDGE", selected_spec=RIDGE_CANDIDATE,
    )
    assert h1 == h2


def test_preregistration_hash_deterministic_and_sensitive():
    body = {"a": 1, "b": [1, 2, 3]}
    h1 = mfs.compute_preregistration_hash(body)
    h2 = mfs.compute_preregistration_hash(body)
    assert h1 == h2
    assert mfs.compute_preregistration_hash({**body, "b": [1, 2, 4]}) != h1


# ---------------------------------------------------------------------------
# No ATS/TOTAL probability-selection path exists anywhere in this module.
# ---------------------------------------------------------------------------
def test_no_ats_total_probability_selection_path():
    import inspect

    source = inspect.getsource(mfs)
    banned = ("ats_probability", "total_probability", "roi", "closing_line_value", " clv")
    lowered = source.lower()
    for token in banned:
        assert token not in lowered


# ---------------------------------------------------------------------------
# Feature-cache ID-reuse remediation.
#
# ``fd._MATRIX_CACHE`` and ``fd._TEAM_STATE_CACHE`` must be keyed by a
# deterministic CONTENT fingerprint of the games frame, never by
# ``id(games)``. ``id()`` is a CPython memory address that is recycled once a
# frame is garbage-collected, so an ``id(games)`` key allowed a brand-new,
# different frame to be handed a stale cached matrix belonging to a freed
# object (process-lifetime cache corruption; a forced id-reuse reproduction
# returned a stale Elo matrix with a 73.333% ``home_elo_pregame_rating``
# mismatch, which is the shape of the Python 3.14 CI failure).
# ---------------------------------------------------------------------------
CACHE_CFG = Week1PriorConfig(k=8.0)


def _reset_feature_caches() -> None:
    fd._MATRIX_CACHE.clear()
    fd._TEAM_STATE_CACHE.clear()


def _variant_games(*, score_delta: int = 0, drop_last: bool = False, extra_col: bool = False) -> pd.DataFrame:
    """A games frame with the exact shape/schema of ``_synthetic_games()``
    unless asked otherwise -- ``score_delta`` keeps the row count and schema
    identical while changing scores/results (and therefore downstream Elo)."""
    g = _synthetic_games()
    if score_delta:
        g["home_score"] = g["home_score"] + int(score_delta)
    if drop_last:
        g = g.iloc[:-1].reset_index(drop=True)
    if extra_col:
        g["home_spread"] = -3.5
    return g


def _elo_ratings(matrix: pd.DataFrame) -> np.ndarray:
    return matrix.sort_values("game_id")["home_elo_pregame_rating"].to_numpy()


# --- A: identity-independent fingerprint + identical feature output --------
def test_content_fingerprint_is_object_identity_independent(week1_config):
    g1 = _synthetic_games()
    g2 = _synthetic_games()  # distinct object, byte-identical content
    assert g1 is not g2 and id(g1) != id(g2)

    fp1 = fd._games_content_fingerprint(g1)
    fp2 = fd._games_content_fingerprint(g2)
    assert fp1 == fp2
    assert isinstance(fp1, str) and len(fp1) == 64  # sha-256 hexdigest

    _reset_feature_caches()
    m1, f1 = fd.build_candidate_matrix(g1, ["ELO_STRENGTH"], week1_config=week1_config)
    m2, f2 = fd.build_candidate_matrix(g2, ["ELO_STRENGTH"], week1_config=week1_config)  # content cache hit
    assert f1 == f2
    pd.testing.assert_frame_equal(
        m1.sort_values("game_id").reset_index(drop=True),
        m2.sort_values("game_id").reset_index(drop=True),
    )
    assert len(fd._MATRIX_CACHE) == 1  # no duplicate entry for identical content


# --- B: same length, different content => different fingerprint -----------
def test_content_fingerprint_changes_for_same_length_different_content():
    g1 = _synthetic_games()
    g2 = _variant_games(score_delta=17)
    assert g1.shape == g2.shape and list(g1.columns) == list(g2.columns)
    assert fd._games_content_fingerprint(g1) != fd._games_content_fingerprint(g2)

    # schema change (extra column) is also detected
    g3 = _variant_games(extra_col=True)
    assert len(g3) == len(g1)
    assert fd._games_content_fingerprint(g3) != fd._games_content_fingerprint(g1)


# --- C: in-place mutation changes fingerprint, no stale state ------------
def test_inplace_mutation_changes_fingerprint_and_gets_no_stale_state(week1_config):
    g = _synthetic_games()
    fp_before = fd._games_content_fingerprint(g)

    _reset_feature_caches()
    m_before, feats = fd.build_candidate_matrix(g, ["ELO_STRENGTH"], week1_config=week1_config)

    g.loc[0, "home_score"] = int(g.loc[0, "home_score"]) + 40  # same object, same length
    fp_after = fd._games_content_fingerprint(g)
    assert fp_after != fp_before

    m_after, _ = fd.build_candidate_matrix(g, ["ELO_STRENGTH"], week1_config=week1_config)

    _reset_feature_caches()  # pristine recompute of the post-mutation frame
    m_ref, _ = fd.build_candidate_matrix(g, ["ELO_STRENGTH"], week1_config=week1_config)
    pd.testing.assert_frame_equal(
        m_after.sort_values("game_id").reset_index(drop=True),
        m_ref.sort_values("game_id").reset_index(drop=True),
    )
    # the mutation genuinely moved the Elo matrix -- so a stale hit would have been observable
    assert not np.array_equal(_elo_ratings(m_before), _elo_ratings(m_after))


# --- D: _MATRIX_CACHE cannot return a stale matrix for different content --
def test_matrix_cache_no_stale_after_gc_and_realloc(week1_config):
    import gc

    _reset_feature_caches()
    g1 = _synthetic_games()
    m1, _ = fd.build_candidate_matrix(g1, ["ELO_STRENGTH"], week1_config=week1_config)
    elo1 = _elo_ratings(m1)

    del g1, m1
    gc.collect()  # free the frame; its id() may now be handed to a new object

    g2 = _variant_games(score_delta=25)  # different content
    m2, _ = fd.build_candidate_matrix(g2, ["ELO_STRENGTH"], week1_config=week1_config)
    elo2 = _elo_ratings(m2)

    _reset_feature_caches()
    m2_ref, _ = fd.build_candidate_matrix(_variant_games(score_delta=25), ["ELO_STRENGTH"], week1_config=week1_config)
    np.testing.assert_array_equal(elo2, _elo_ratings(m2_ref))
    assert not np.array_equal(elo1, elo2)


def test_matrix_cache_key_is_content_addressed_not_identity(week1_config):
    _reset_feature_caches()
    g = _synthetic_games()
    fd.build_candidate_matrix(g, ["ELO_STRENGTH"], week1_config=week1_config)
    (key,) = list(fd._MATRIX_CACHE)
    fingerprint, groups, cfg_hash = key
    assert fingerprint == fd._games_content_fingerprint(g)
    assert groups == ("ELO_STRENGTH",)
    assert isinstance(fingerprint, str) and isinstance(cfg_hash, str)
    assert id(g) not in key and len(g) not in key  # no object-identity / length component


# --- E: _TEAM_STATE_CACHE cannot return stale state for different content -
def test_team_state_cache_no_stale_for_different_content(week1_config):
    import gc

    _reset_feature_caches()
    g1 = _synthetic_games()
    ts1 = fd.build_production_team_state(g1, week1_config=week1_config)
    r1 = ts1.sort_values(["game_id", "team_id"])["elo_pregame_rating"].to_numpy()

    (key1,) = list(fd._TEAM_STATE_CACHE)
    assert key1[0] == fd._games_content_fingerprint(g1)
    assert id(g1) not in key1 and len(g1) not in key1

    del g1
    gc.collect()

    g2 = _variant_games(score_delta=19)
    ts2 = fd.build_production_team_state(g2, week1_config=week1_config)
    r2 = ts2.sort_values(["game_id", "team_id"])["elo_pregame_rating"].to_numpy()

    _reset_feature_caches()
    ts2_ref = fd.build_production_team_state(_variant_games(score_delta=19), week1_config=week1_config)
    r2_ref = ts2_ref.sort_values(["game_id", "team_id"])["elo_pregame_rating"].to_numpy()
    np.testing.assert_array_equal(r2, r2_ref)
    assert not np.array_equal(r1, r2)


def test_team_state_cache_key_separates_rolling_config(week1_config):
    _reset_feature_caches()
    g = _synthetic_games()
    fd.build_production_team_state(g, week1_config=week1_config)  # None -> normalises to windows=(8,)
    fd.build_production_team_state(g, week1_config=week1_config, rolling_config=fd.PregameRollingConfig(windows=(8,)))
    assert len(fd._TEAM_STATE_CACHE) == 1  # None and explicit (8,) collapse to one entry
    # a genuinely different config (keeps window 8 so the last8 columns still
    # exist) is a distinct cache entry rather than a stale hit
    fd.build_production_team_state(
        g, week1_config=week1_config, rolling_config=fd.PregameRollingConfig(windows=(8, 4))
    )
    assert len(fd._TEAM_STATE_CACHE) == 2


# --- H: high-churn transient copies stay correct without manual clearing --
def test_high_churn_transient_copies_stay_correct_without_clearing(week1_config):
    _reset_feature_caches()
    ref_matrix, _ = fd.build_candidate_matrix(_synthetic_games(), ["ELO_STRENGTH"], week1_config=week1_config)
    ref_elo = _elo_ratings(ref_matrix)

    seen: dict[int, np.ndarray] = {}
    for delta in (0, 5, 0, 11, 5, 0, 23, 11, 0):
        g = _variant_games(score_delta=delta)  # fresh transient object every iteration
        m, _ = fd.build_candidate_matrix(g, ["ELO_STRENGTH"], week1_config=week1_config)
        elo = _elo_ratings(m)
        if delta in seen:
            np.testing.assert_array_equal(elo, seen[delta])
        else:
            seen[delta] = elo
        if delta == 0:
            np.testing.assert_array_equal(elo, ref_elo)
        del g, m

    # one entry per DISTINCT content, not one per call -> no unbounded growth
    assert len(fd._MATRIX_CACHE) == len({0, 5, 11, 23})
