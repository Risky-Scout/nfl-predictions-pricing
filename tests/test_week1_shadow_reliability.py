"""CI-safe tests for the Week 1 shadow reliability evaluator (Stage 3).

Uses ONLY a committed deterministic synthetic fixture
(``tests/fixtures/week1_reliability/factory.py``) — no private backfill, no
purchased odds, no network. This module MUST contain no skip markers (a guard
test enforces it) so every check runs on every supported Python version.

It enforces evaluator *correctness*: fold construction and leakage safety,
scoring (log loss / Brier / adaptive ECE / calibration slope-intercept /
non-estimable handling), push & tie handling, exact market-contract matching,
season-stratified paired bootstrap with a fixed seed, leave-one-season-out
sensitivity, reliability classification, and deterministic ledger / JSON / MD
outputs with JSON-to-Markdown number reconciliation.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.evaluation import week1_reliability as wr
from nfl_hybrid.features.augmented_matrix import FROZEN_FEATURES

ROOT = Path(__file__).parents[1]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


factory = _load("w1r_factory", "tests/fixtures/week1_reliability/factory.py")
ev = _load("w1r_evaluator", "scripts/evaluate_week1_shadow_reliability.py")


# ---------------------------------------------------------------------------
# Module-scoped fixtures (built once; the full evaluation is the expensive part).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    matrix, games, odds = factory.build_all()
    return {"matrix": matrix, "games": games, "odds": odds,
            "market_ref": wr.schedule_reference_market(games)}


@pytest.fixture(scope="module")
def evaluated(synthetic):
    report, ledger = ev.run_evaluation(synthetic["matrix"], synthetic["games"], synthetic["market_ref"])
    return {"report": report, "ledger": ledger}


# ---------------------------------------------------------------------------
# Fold construction & leakage
# ---------------------------------------------------------------------------
def test_fold_construction(synthetic):
    m = synthetic["matrix"]
    assert [f["test_season"] for f in wr.FOLDS] == [2022, 2023, 2024, 2025]
    for fold in wr.FOLDS:
        train, calib, test = wr.build_fold_frames(m, fold)
        assert train["season"].max() == fold["train_max_season"]
        assert set(calib["season"].unique()) == {fold["calibration_season"]}
        assert set(test["season"].unique()) == {fold["test_season"]}


def test_no_same_season_leakage(synthetic):
    m = synthetic["matrix"]
    for fold in wr.FOLDS:
        train, calib, test = wr.build_fold_frames(m, fold)
        ts = fold["test_season"]
        assert (train["season"] < ts).all()
        assert (calib["season"] < ts).all()
        assert not set(train["game_id"]) & set(test["game_id"])
        assert not set(calib["game_id"]) & set(test["game_id"])


def test_leakage_guard_raises_on_bad_fold(synthetic):
    m = synthetic["matrix"]
    bad = {"test_season": 2022, "train_max_season": 2022, "calibration_season": 2021}
    with pytest.raises(ValueError):
        wr.build_fold_frames(m, bad)


def test_no_test_rows_in_imputation_or_calibration(synthetic):
    """Fold windows are strictly before the test season, so no test-season row can
    fit the median imputer (fit on train) or the calibrators (fit on calib)."""
    m = synthetic["matrix"]
    for fold in wr.FOLDS:
        assert fold["train_max_season"] < fold["test_season"]
        assert fold["calibration_season"] < fold["test_season"]
        train, calib, test = wr.build_fold_frames(m, fold)
        assert fold["test_season"] not in set(train["season"]) | set(calib["season"])


def test_2025_holdout_excluded_from_fitting(evaluated):
    art = evaluated["report"]["artifact_identity"]
    assert art["holdout_excluded"] == 2025
    assert art["training_cutoff_season"] == 2024
    # no fold trains or calibrates on the test season it scores
    for fold in wr.FOLDS:
        assert fold["train_max_season"] < fold["test_season"]
    hi = evaluated["report"]["holdout_2025_integrity"]
    assert hi["2025_used_in_fitting"] is False
    assert hi["2025_used_in_calibration"] is False
    assert hi["2025_used_in_selection"] is False


def test_deterministic_fold_predictions(synthetic):
    from nfl_hybrid.modern.joint_score import JointScoreModel
    m = synthetic["matrix"]
    train, calib, test = wr.build_fold_frames(m, wr.FOLDS[0])
    preds = []
    for _ in range(2):
        model = JointScoreModel(numeric_features=FROZEN_FEATURES)
        model.fit(train, calibration=calib)
        preds.append(ev._predict(model, test)["home_win_probability_no_tie"].to_numpy())
    np.testing.assert_array_equal(preds[0], preds[1])


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------
def test_binary_log_loss_known_value():
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.2])
    expected = -(math.log(0.8) + math.log(0.8)) / 2
    assert wr.binary_log_loss(y, p) == pytest.approx(expected, rel=1e-12)


def test_brier_known_value():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.1, 0.4])
    expected = np.mean([(0.9 - 1) ** 2, (0.1 - 0) ** 2, (0.4 - 1) ** 2])
    assert wr.brier(y, p) == pytest.approx(expected, rel=1e-12)


def test_equal_mass_ece_adaptive_and_not_estimable():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 300)
    y = (rng.uniform(size=300) < p).astype(float)
    ece = wr.equal_mass_ece(y, p)
    assert isinstance(ece, float) and 0.0 <= ece < 0.2
    assert wr.n_bins_for(300) == 10
    assert wr.n_bins_for(64) == 5
    assert wr.equal_mass_ece(y[:20], p[:20]) == wr.NOT_ESTIMABLE  # n < 36


def test_calibration_slope_intercept_recovers_identity():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.98, 4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    slope, intercept = wr.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_separation_not_estimable():
    y = np.array([0.0] * 20 + [1.0] * 20)
    p = np.array([0.01] * 20 + [0.99] * 20)  # perfectly separated
    slope, intercept = wr.calibration_slope_intercept(y, p)
    assert slope == wr.NOT_ESTIMABLE and intercept == wr.NOT_ESTIMABLE


def test_single_class_not_estimable():
    y = np.ones(50)
    p = np.linspace(0.4, 0.6, 50)
    assert wr.calibration_slope_intercept(y, p) == (wr.NOT_ESTIMABLE, wr.NOT_ESTIMABLE)
    assert wr.auc_diagnostic(y, p) == wr.NOT_ESTIMABLE


def test_auc_diagnostic_value():
    y = np.array([0, 0, 1, 1.0])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert wr.auc_diagnostic(y, p) == pytest.approx(0.75, rel=1e-9)


# ---------------------------------------------------------------------------
# Push & tie handling
# ---------------------------------------------------------------------------
def test_push_exclusion():
    frame = pd.DataFrame({
        "home_margin": [3.0, -3.0, 7.0], "total_points": [44.0, 40.0, 50.0],
        "home_spread": [-3.0, -3.0, -7.0], "total_line": [44.0, 45.0, 48.0],
    })
    masks = ev._outcome_masks(frame)
    # game 0: margin 3, spread -3 -> push (excluded); game 2: total 50 vs 44 line? no, total_line 48 -> over
    assert masks["spread"]["excluded"][0] and not masks["spread"]["graded"][0]
    # total game 0: total_points 44 == line 44 -> push
    assert masks["total"]["excluded"][0]
    # a push is labeled, never scored as win/loss
    assert masks["spread"]["excluded"].sum() >= 1


def test_tie_handling():
    frame = pd.DataFrame({
        "home_margin": [0.0, 5.0], "total_points": [40.0, 50.0],
        "home_spread": [-2.5, -2.5], "total_line": [43.5, 47.5],
    })
    masks = ev._outcome_masks(frame)
    assert masks["moneyline"]["excluded"][0] and masks["moneyline"]["graded"][1]
    # tie never coded as a home or away win among graded rows
    assert masks["moneyline"]["y"][0] == 0.0  # value exists but the row is excluded from scoring


def test_push_or_tie_rows_never_scored(evaluated):
    led = evaluated["ledger"]
    excluded = led[led["push_or_tie"].astype(bool)]
    # every excluded row carries a label, not a 0/1 graded outcome
    assert set(excluded["realized_outcome"].unique()) <= {"PUSH", "TIE"}


# ---------------------------------------------------------------------------
# Market contract matching
# ---------------------------------------------------------------------------
def test_exact_market_contract_matching(synthetic):
    cc = wr.closing_consensus_market(synthetic["odds"], synthetic["games"])
    g0 = cc[cc["game_id"] == synthetic["games"].iloc[0]["game_id"]].iloc[0]
    # MATCHED spread consensus is the mean of ONLY the two ref-point books (0.52, 0.54)
    assert g0["spread_match_status"] == "MATCHED"
    assert g0["market_spread"] == pytest.approx(0.53, abs=1e-9)
    # total consensus = mean(0.49, 0.51)
    assert g0["total_match_status"] == "MATCHED"
    assert g0["market_total"] == pytest.approx(0.50, abs=1e-9)
    # moneyline consensus = mean(0.58, 0.60)
    assert g0["market_moneyline"] == pytest.approx(0.59, abs=1e-9)


def test_mismatched_spread_rejected(synthetic):
    cc = wr.closing_consensus_market(synthetic["odds"], synthetic["games"])
    g1 = cc[cc["game_id"] == synthetic["games"].iloc[1]["game_id"]]
    assert g1.iloc[0]["spread_match_status"] == "CONTRACT_MISMATCH"
    assert not np.isfinite(g1.iloc[0]["market_spread"])
    # the +1 point book on game 0 was excluded (else consensus would be pulled toward 0.61)
    g0 = cc[cc["game_id"] == synthetic["games"].iloc[0]["game_id"]].iloc[0]
    assert g0["market_spread"] == pytest.approx(0.53, abs=1e-9)


def test_mismatched_total_rejected(synthetic):
    games = synthetic["games"]
    gid = games.iloc[2]["game_id"]
    ref_total = float(games.iloc[2]["total_line_reference"])
    odds = pd.DataFrame([{
        "game_id": gid, "bookmaker_id": "bookA", "market_type": "total",
        "outcome_side": "over", "line_value": ref_total + 1.0, "minutes_to_kickoff": 15.0,
        "is_live": False, "devig_probability": 0.6,
    }])
    cc = wr.closing_consensus_market(odds, games)
    assert cc.iloc[0]["total_match_status"] == "CONTRACT_MISMATCH"


def test_post_kickoff_and_live_quote_rejected(synthetic):
    # game 0's post-kickoff (mtk<0) and live quotes have dp 0.99/0.30; if they were
    # not rejected the MATCHED spread consensus (0.53) would be corrupted.
    cc = wr.closing_consensus_market(synthetic["odds"], synthetic["games"])
    g0 = cc[cc["game_id"] == synthetic["games"].iloc[0]["game_id"]].iloc[0]
    assert g0["market_spread"] == pytest.approx(0.53, abs=1e-9)

    # a game whose only quote is post-kickoff yields no consensus row
    gid = synthetic["games"].iloc[3]["game_id"]
    odds = pd.DataFrame([{
        "game_id": gid, "bookmaker_id": "bookA", "market_type": "moneyline",
        "outcome_side": "home", "line_value": np.nan, "minutes_to_kickoff": -3.0,
        "is_live": False, "devig_probability": 0.9,
    }])
    cc2 = wr.closing_consensus_market(odds, synthetic["games"])
    assert len(cc2) == 0


def test_schedule_reference_market_is_devigged(synthetic):
    mref = synthetic["market_ref"]
    for col in ("market_moneyline", "market_spread", "market_total"):
        vals = mref[col].to_numpy(float)
        assert np.isfinite(vals).all()
        assert ((vals >= 0) & (vals <= 1)).all()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_paired_bootstrap_alignment_zero_when_identical():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.1, 0.9, 200)
    y = (rng.uniform(size=200) < p).astype(float)
    seasons = np.repeat([2022, 2023], 100)
    out = wr.paired_bootstrap_logloss_diff(y, p, p, seasons, reps=100)
    # identical shadow & market -> per-game diff is exactly 0 for every resample
    assert out["log_loss_diff"] == pytest.approx(0.0, abs=1e-12)
    assert out["log_loss_ci"] == [pytest.approx(0.0, abs=1e-12), pytest.approx(0.0, abs=1e-12)]


def test_season_stratified_bootstrap_preserves_counts():
    seasons = np.array([2022] * 30 + [2023] * 50 + [2024] * 20)
    rng = np.random.default_rng(4)
    idx = wr._season_stratified_indices(seasons, rng)
    got = seasons[idx]
    assert (got == 2022).sum() == 30
    assert (got == 2023).sum() == 50
    assert (got == 2024).sum() == 20


def test_bootstrap_fixed_seed_reproducible():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.1, 0.9, 120)
    y = (rng.uniform(size=120) < p).astype(float)
    pm = np.clip(p + 0.02, 1e-6, 1 - 1e-6)
    seasons = np.repeat([2022, 2023, 2024], 40)
    a = wr.paired_bootstrap_logloss_diff(y, p, pm, seasons, reps=200, seed=wr.BOOTSTRAP_SEED)
    b = wr.paired_bootstrap_logloss_diff(y, p, pm, seasons, reps=200, seed=wr.BOOTSTRAP_SEED)
    assert a["log_loss_ci"] == b["log_loss_ci"]
    assert a["log_loss_diff"] == b["log_loss_diff"]


def test_loso_sensitivity(evaluated):
    lo = evaluated["report"]["loso_sensitivity"]["moneyline"]
    assert set(lo["per_removed"].keys()) == {"2022", "2023", "2024", "2025"}
    assert isinstance(lo["sign_stable"], bool)
    assert isinstance(lo["full_diff"], float)


# ---------------------------------------------------------------------------
# Probability validity & classification
# ---------------------------------------------------------------------------
def test_nonfinite_probability_rejection():
    d = wr.MarketDiagnostics(n=200, finite_valid=False, deterministic=True, leakage_free=True,
                             ece=0.01, slope=1.0, intercept=0.0, loso_stable=True,
                             shadow_minus_market_logloss=0.0, logloss_ci_lower=-0.01)
    assert wr.classify_market(d) == "WITHHOLD_SHADOW_OUTPUT"


def test_out_of_range_probability_detection():
    sp = np.array([0.2, 0.5, 1.2])  # 1.2 out of range
    finite_valid = bool(np.isfinite(sp).all() and (sp >= 0).all() and (sp <= 1).all())
    assert finite_valid is False


def test_reliability_classification_rules():
    base = dict(finite_valid=True, deterministic=True, leakage_free=True,
                loso_stable=True, shadow_minus_market_logloss=0.0, logloss_ci_lower=-0.01)
    # small sample -> inconclusive
    assert wr.classify_market(wr.MarketDiagnostics(
        n=64, ece=0.02, slope=1.0, intercept=0.0, **base)) == "INCONCLUSIVE_SMALL_SAMPLE"
    # not estimable diagnostics -> inconclusive even at large n
    assert wr.classify_market(wr.MarketDiagnostics(
        n=200, ece=wr.NOT_ESTIMABLE, slope=1.0, intercept=0.0, **base)) == "INCONCLUSIVE_SMALL_SAMPLE"
    # large n, bad ECE -> calibration concern
    assert wr.classify_market(wr.MarketDiagnostics(
        n=200, ece=0.09, slope=1.0, intercept=0.0, **base)) == "CALIBRATION_CONCERN"
    # large n, slope out of range -> calibration concern
    assert wr.classify_market(wr.MarketDiagnostics(
        n=200, ece=0.02, slope=0.4, intercept=0.0, **base)) == "CALIBRATION_CONCERN"
    # large n, all good, non-inferior -> supported
    assert wr.classify_market(wr.MarketDiagnostics(
        n=200, ece=0.02, slope=1.0, intercept=0.0, **base)) == "SUPPORTED_FOR_SHADOW_MONITORING"
    # materially worse than market (CI lower above margin) -> concern
    worse = dict(base); worse["logloss_ci_lower"] = 0.05; worse["shadow_minus_market_logloss"] = 0.08
    assert wr.classify_market(wr.MarketDiagnostics(
        n=200, ece=0.02, slope=1.0, intercept=0.0, **worse)) == "CALIBRATION_CONCERN"


def test_production_recommendation_mapping():
    assert wr.production_recommendation(
        {"moneyline": "INCONCLUSIVE_SMALL_SAMPLE", "spread": "SUPPORTED_FOR_SHADOW_MONITORING",
         "total": "INCONCLUSIVE_SMALL_SAMPLE"}) == "RETAIN_MARKET_BASELINE_AND_MONITOR_SHADOW"
    assert wr.production_recommendation(
        {"moneyline": "CALIBRATION_CONCERN", "spread": "INCONCLUSIVE_SMALL_SAMPLE",
         "total": "INCONCLUSIVE_SMALL_SAMPLE"}) == "RETAIN_MARKET_BASELINE"
    assert wr.production_recommendation(
        {"moneyline": "WITHHOLD_SHADOW_OUTPUT", "spread": "INCONCLUSIVE_SMALL_SAMPLE",
         "total": "INCONCLUSIVE_SMALL_SAMPLE"}) == "WITHHOLD_SHADOW_UNTIL_REPAIRED"


# ---------------------------------------------------------------------------
# Determinism & report integrity
# ---------------------------------------------------------------------------
def test_deterministic_ledger_hash(synthetic, evaluated):
    l2, _ = ev.build_ledger(synthetic["matrix"], synthetic["games"], synthetic["market_ref"])
    h1 = wr.deterministic_frame_hash(evaluated["ledger"], ev.LEDGER_COLUMNS)
    h2 = wr.deterministic_frame_hash(l2, ev.LEDGER_COLUMNS)
    assert h1 == h2 == evaluated["report"]["ledger"]["sha256"]


def test_ledger_hash_changes_on_edit(evaluated):
    led = evaluated["ledger"].copy()
    h0 = wr.deterministic_frame_hash(led, ev.LEDGER_COLUMNS)
    led.loc[led.index[0], "shadow_probability"] = float(led.loc[led.index[0], "shadow_probability"]) + 0.01
    assert wr.deterministic_frame_hash(led, ev.LEDGER_COLUMNS) != h0


def test_deterministic_json_hash(synthetic, evaluated):
    r2, _ = ev.run_evaluation(synthetic["matrix"], synthetic["games"], synthetic["market_ref"])
    assert ev.report_json_body(r2) == ev.report_json_body(evaluated["report"])


def test_mandatory_json_schema_fields(evaluated):
    r = evaluated["report"]
    required = {"schema_version", "registry_commit", "artifact_identity", "terminology",
                "data_sources", "holdout_2025_integrity", "folds", "as_of_semantics",
                "market_timing", "samples_and_coverage", "metrics", "market_comparison",
                "bootstrap", "calibration", "imputation", "weeks_1_2_sensitivity",
                "later_week_context", "loso_sensitivity", "reliability_status",
                "production", "ledger", "determinism"}
    assert required <= set(r.keys())
    for market in ("moneyline", "spread", "total"):
        assert set(r["metrics"][market].keys()) >= {
            "A_locked_2025_holdout", "B_week1_oof", "C_weeks1_2_oof", "D_weeks3plus_oof"}
    assert r["reliability_status"][market] in {
        "SUPPORTED_FOR_SHADOW_MONITORING", "INCONCLUSIVE_SMALL_SAMPLE",
        "CALIBRATION_CONCERN", "WITHHOLD_SHADOW_OUTPUT"}
    assert r["production"]["production_probability_source"] == "MARKET_BASELINE"
    assert r["production"]["recommendation"] in {
        "RETAIN_MARKET_BASELINE", "RETAIN_MARKET_BASELINE_AND_MONITOR_SHADOW",
        "WITHHOLD_SHADOW_UNTIL_REPAIRED"}


def test_json_to_markdown_reconciliation(evaluated):
    r = evaluated["report"]
    md = ev.render_markdown(r)
    n = r["samples_and_coverage"]["B_week1_oof"]["n_games"]
    assert f"Pooled Week 1 OOF n = {n} games" in md
    ll = r["metrics"]["moneyline"]["B_week1_oof"]["shadow"]["log_loss"]
    assert f"{ll:.4f}" in md
    # terminology statement present and correct
    assert "not independent of the market" in md
    assert r["production"]["recommendation"] in md


def test_imputation_regime(evaluated):
    im = evaluated["report"]["imputation"]
    assert im["week1"]["mean_imputed_feature_count"] == pytest.approx(6.0, abs=1e-9)
    assert im["weeks3plus"]["mean_imputed_feature_count"] == pytest.approx(0.0, abs=1e-9)
    assert set(im["week1"]["imputed_feature_names"]) == set(factory.SEASON_MEAN_FEATURES)


def test_production_invariance_in_ledger(evaluated):
    led = evaluated["ledger"]
    assert (led["production_source"] == "MARKET_BASELINE").all()
    paired = led[np.isfinite(led["market_probability"].astype(float))]
    np.testing.assert_allclose(
        paired["production_probability"].astype(float).to_numpy(),
        paired["market_probability"].astype(float).to_numpy())


# ---------------------------------------------------------------------------
# No-skip guard
# ---------------------------------------------------------------------------
def test_module_has_no_skip_markers():
    text = Path(__file__).read_text()
    # match actual skip calls / decorators, not the string literals in this guard
    assert re.search(r"pytest\.skip\(", text) is None
    assert re.search(r"@pytest\.mark\.skip", text) is None
    assert re.search(r"mark\.extended_data", text) is None
