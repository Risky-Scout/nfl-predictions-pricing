"""Fix 7: limited model-family/hyperparameter selection + estimator spec
freeze -- consumes the already-frozen Fix 6 six-Elo-feature input
(``outputs/fix6_feature_selection_summary.json``) and selects/freezes a
predictive estimator family (Ridge / Huber / the incumbent
:class:`~nfl_hybrid.modern.joint_score.JointScoreModel` HGBR) for that fixed
input. Fix 7 does NOT reopen feature selection, does NOT calibrate
ATS/TOTAL, and does NOT promote a production model -- see
``docs/MODEL_FAMILY_SELECTION_2026.md``.

Read-only dependency on Fix 6 (:mod:`nfl_hybrid.selection.feature_deduction_2026`,
aliased ``fd`` below): reuses ``fd.HardGateFailure``, ``fd.enforce_2025_firewall``,
``fd.build_candidate_matrix``, ``fd.FEATURE_GROUPS["ELO_STRENGTH"]``,
``fd.compute_feature_manifest_hash``, ``fd.INNER_FOLDS``/``fd.OUTER_FOLD``,
``fd.assert_fold_seasons_allowed``, ``fd.JointScoreModel``/``fd.JointScoreConfig``/
``fd.MODEL_CONFIG``, ``fd._segment_labels``. Fix 7 deliberately does NOT call
``fd._fit_predict_fold`` (it requires Fix 6's own registry-freeze side effect,
a Fix-6-specific selection-lifecycle coupling Fix 7 must not depend on) --
HGBR is fit here through a narrow, self-contained adapter around the same
public ``JointScoreModel``/``JointScoreConfig`` API instead, so HGBR behavior
stays byte-identical to the committed incumbent with zero coupling to Fix
6's internal state.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_hybrid.evaluation.metrics import regression_metrics
from nfl_hybrid.selection import feature_deduction_2026 as fd

HardGateFailure = fd.HardGateFailure

SCHEMA_VERSION = "fix7-model-family-selection-v1"

# The committed Fix 6 feature order (verified programmatically against
# outputs/fix6_feature_selection_summary.json in Phase 1 -- see
# ``load_and_verify_fix6_feature_contract`` below; this literal is only the
# EXPECTED value used for that verification, never trusted on its own).
REQUIRED_ORDER: tuple[str, ...] = (
    "home_elo_pregame_rating",
    "home_elo_pregame_win_probability",
    "home_elo_pregame_expected_margin",
    "away_elo_pregame_rating",
    "away_elo_pregame_win_probability",
    "away_elo_pregame_expected_margin",
)

# Copied verbatim from the committed outputs/fix6_feature_selection_summary.json
# ("elo_core_feature_relationship" field) -- disclosed again, not reworded,
# alongside Fix 7's own Ridge/Huber coefficient interpretability output.
ELO_CORE_FEATURE_RELATIONSHIP_TEXT = (
    "Of the six preregistered core ELO_STRENGTH features, elo_pregame_win_probability "
    "and elo_pregame_expected_margin (each side, home/away) are a DETERMINISTIC "
    "BIJECTIVE PAIR: both are computed in LegacyElo.predict "
    "(src/nfl_hybrid/legacy/elo.py:99-100) as monotonic, invertible transforms of the "
    "identical underlying scalar `adjusted_difference` -- p_home = "
    "1/(1+10^(-adjusted_difference/scale)) (logistic) and margin = "
    "adjusted_difference/elo_to_margin (linear) -- so, for a fixed game row, either "
    "quantity is exactly recoverable from the other; retaining both provides no "
    "additional predictive information beyond either alone. The away-side "
    "win_probability/expected_margin are likewise exact deterministic transforms of "
    "the home-side values (away_win_probability = 1 - home_win_probability, "
    "away_expected_margin = -home_expected_margin; "
    "src/nfl_hybrid/features/elo_state.py). elo_pregame_rating (home/away separately) "
    "is a DIFFERENT, non-redundant quantity: it is the raw pre-context Elo rating that "
    "combines with match-context adjustments not separately exposed as their own "
    "ELO_STRENGTH columns (home_field_elo, travel, rest/bye, QB adjustment, playoff "
    "multiplier -- see LegacyElo.predict) to form adjusted_difference, so "
    "home_rating/away_rating are NOT deterministically recoverable from "
    "win_probability/expected_margin alone, and vice versa. Net: of the 6 retained "
    "columns, there are 3 non-redundant underlying quantities per game (home_rating, "
    "away_rating, adjusted_difference), with win_probability/expected_margin on both "
    "sides being redundant deterministic encodings of the third."
)


def _canonical_json(payload: dict) -> bytes:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_hex(payload: dict) -> str:
    return sha256(_canonical_json(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Phase 1: Fix-6 feature contract -- committed evidence is authoritative,
# the live registry is an independent cross-check (both must agree).
# ---------------------------------------------------------------------------
def load_and_verify_fix6_feature_contract(summary: dict) -> tuple[tuple[str, ...], str]:
    """``summary`` is the parsed contents of the committed
    ``outputs/fix6_feature_selection_summary.json``. Returns
    ``(frozen_feature_columns, fix6_feature_manifest_hash)``. Hard STOPs on
    any divergence between the committed evidence, the live
    ``fd.FEATURE_GROUPS`` registry, and a fresh hash recomputation -- never a
    silent repair in either direction."""
    committed_feature_list = tuple(summary["final_features"])
    committed_feature_hash = summary["feature_manifest_hash"]
    if committed_feature_list != REQUIRED_ORDER:
        raise HardGateFailure(
            f"COMMITTED_FIX6_EVIDENCE_UNEXPECTED_ORDER: {committed_feature_list} != {REQUIRED_ORDER}"
        )

    live_registry_list = tuple(fd.FEATURE_GROUPS["ELO_STRENGTH"].columns)
    if live_registry_list != committed_feature_list:
        raise HardGateFailure(
            f"LIVE_FIX6_REGISTRY_DIVERGED_FROM_COMMITTED_EVIDENCE: {live_registry_list} != {committed_feature_list}"
        )

    recomputed_hash = fd.compute_feature_manifest_hash(committed_feature_list)
    if recomputed_hash != committed_feature_hash:
        raise HardGateFailure(
            f"COMMITTED_FIX6_HASH_MISMATCH: recomputed={recomputed_hash} committed={committed_feature_hash}"
        )

    return committed_feature_list, committed_feature_hash


# ---------------------------------------------------------------------------
# Candidate registry -- frozen before the first real fit. Semantic hash
# includes family/complexity/preprocessing/hyperparameters/random_state, not
# just a name.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str  # "RIDGE" | "HUBER" | "HGBR"
    complexity_rank: int  # 0=RIDGE, 1=HUBER, 2=HGBR
    preprocessing: dict
    hyperparameters: dict
    target_sharing_rule: str = "SAME_FAMILY_AND_HYPERPARAMETERS_FOR_MARGIN_AND_TOTAL"
    random_state: int | None = None


def compute_candidate_spec_hash(spec: CandidateSpec) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "family": spec.family,
        "complexity_rank": spec.complexity_rank,
        "preprocessing": spec.preprocessing,
        "hyperparameters": spec.hyperparameters,
        "target_sharing_rule": spec.target_sharing_rule,
        "random_state": spec.random_state,
    }
    return _sha256_hex(payload)


def compute_candidate_registry_hash(registry: tuple[CandidateSpec, ...]) -> str:
    return _sha256_hex(
        {
            "schema_version": SCHEMA_VERSION,
            "candidates": [{"name": c.name, "spec_hash": compute_candidate_spec_hash(c)} for c in registry],
        }
    )


def _hgbr_preprocessing_spec() -> dict:
    return {"type": "JointScoreModel.preprocessor (ColumnTransformer: numeric SimpleImputer(median))"}


def _hgbr_hyperparameters() -> dict:
    from dataclasses import asdict

    return asdict(fd.MODEL_CONFIG)


_STANDARD_SCALER_SPEC = {"type": "StandardScaler", "with_mean": True, "with_std": True}

CANDIDATE_REGISTRY: tuple[CandidateSpec, ...] = (
    CandidateSpec("RIDGE_ALPHA_0_1", "RIDGE", 0, _STANDARD_SCALER_SPEC, {"alpha": 0.1, "fit_intercept": True, "solver": "svd"}),
    CandidateSpec("RIDGE_ALPHA_1", "RIDGE", 0, _STANDARD_SCALER_SPEC, {"alpha": 1.0, "fit_intercept": True, "solver": "svd"}),
    CandidateSpec("RIDGE_ALPHA_10", "RIDGE", 0, _STANDARD_SCALER_SPEC, {"alpha": 10.0, "fit_intercept": True, "solver": "svd"}),
    CandidateSpec("RIDGE_ALPHA_100", "RIDGE", 0, _STANDARD_SCALER_SPEC, {"alpha": 100.0, "fit_intercept": True, "solver": "svd"}),
    CandidateSpec(
        "HUBER_FIXED", "HUBER", 1, _STANDARD_SCALER_SPEC,
        {"epsilon": 1.35, "alpha": 0.0001, "fit_intercept": True, "max_iter": 1000, "tol": 1e-5},
    ),
    CandidateSpec(
        "HGBR_INCUMBENT", "HGBR", 2, _hgbr_preprocessing_spec(), _hgbr_hyperparameters(),
        # Corrected (execution-safety review #2): must be the ACTUAL committed
        # JointScoreConfig.random_state (42 today), never left at the
        # CandidateSpec default of None even though the full config is
        # already carried in `hyperparameters`.
        random_state=fd.MODEL_CONFIG.random_state,
    ),
)

RIDGE_ALPHA_NAMES: tuple[str, ...] = tuple(c.name for c in CANDIDATE_REGISTRY if c.family == "RIDGE")


def assert_hgbr_random_state_matches_committed_config(registry: tuple[CandidateSpec, ...] = CANDIDATE_REGISTRY) -> None:
    """Execution-safety correction #2: hard gate, not a convention. The HGBR
    candidate's semantic ``random_state`` must equal the actually-committed
    ``fd.MODEL_CONFIG.random_state`` at call time -- never a hand-copied
    literal that could silently drift from the real config."""
    hgbr = next(c for c in registry if c.name == "HGBR_INCUMBENT")
    if hgbr.random_state != fd.MODEL_CONFIG.random_state:
        raise HardGateFailure(
            f"HGBR_RANDOM_STATE_MISMATCH: CandidateSpec.random_state={hgbr.random_state!r} "
            f"!= fd.MODEL_CONFIG.random_state={fd.MODEL_CONFIG.random_state!r}"
        )


# ---------------------------------------------------------------------------
# Preprocessing + fit/predict per fold, dispatched by family.
# ---------------------------------------------------------------------------
def _build_pipeline(family: str, hyperparameters: dict) -> Pipeline:
    est_cls = Ridge if family == "RIDGE" else HuberRegressor
    return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("estimator", est_cls(**hyperparameters))])


def _fit_one_ridge_or_huber_target(family: str, hyperparameters: dict, x_train, y_train, x_val):
    """Returns ``(pipeline, prediction_or_None, ok, reason)``. ``ok=False``
    only for HUBER's three predeclared numerical-invalidity triggers
    (ConvergenceWarning, non-finite fitted coefficient/intercept, non-finite
    validation prediction) -- any other exception propagates uncaught, and a
    non-finite RIDGE prediction is unconditionally a hard STOP (mandatory
    reference candidate)."""
    if family == "HUBER":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe = _build_pipeline(family, hyperparameters).fit(x_train, y_train)
        conv_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        estimator = pipe.named_steps["estimator"]
        finite_fit = bool(np.all(np.isfinite(estimator.coef_)) and np.isfinite(estimator.intercept_))
        if conv_warned or not finite_fit:
            reason = "CONVERGENCE_WARNING" if conv_warned else "NONFINITE_FITTED_PARAMETERS"
            return pipe, None, False, reason
        pred = pipe.predict(x_val)
        if not np.all(np.isfinite(pred)):
            return pipe, pred, False, "NONFINITE_VALIDATION_PREDICTION"
        return pipe, pred, True, None

    pipe = _build_pipeline(family, hyperparameters).fit(x_train, y_train)
    pred = pipe.predict(x_val)
    if not np.all(np.isfinite(pred)):
        raise HardGateFailure(f"{family}: non-finite prediction (mandatory reference candidate)")
    return pipe, pred, True, None


def fit_predict_fold(
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    fold: "fd.FoldSpec",
    candidate: CandidateSpec,
    *,
    return_fitted: bool = False,
):
    """Fits ONE model on ``season <= fold.train_max_season`` and predicts
    ONCE on ``season == fold.validate_season`` -- a single fixed fit per
    fold, per candidate. Returns ``(predictions, huber_invalid_reasons)``,
    or ``(predictions, huber_invalid_reasons, fitted_bundle)`` when
    ``return_fitted=True`` -- ``fitted_bundle`` is ``{"model": JointScoreModel}``
    for HGBR or ``{"margin": Pipeline, "total": Pipeline}`` for Ridge/Huber,
    reusable for prediction on OTHER feature rows with no additional fit
    (used by Phase 7's audit, Phase 8's interpretability, and the poison-pill
    tests)."""
    fd.assert_fold_seasons_allowed(fold)
    train = matrix[matrix["season"] <= fold.train_max_season]
    validate = matrix[matrix["season"] == fold.validate_season]
    if train.empty or validate.empty:
        raise ValueError(f"Fold {fold.name}: empty train ({len(train)}) or validate ({len(validate)}) set.")

    feature_columns = list(feature_columns)

    if candidate.family == "HGBR":
        model = fd.JointScoreModel(numeric_features=feature_columns, categorical_features=(), config=fd.MODEL_CONFIG)
        model.fit(train)
        pred_margin, pred_total = model.predict_means(validate)  # public accessor only
        out = validate[["game_id", "season", "week", "home_margin", "total_points"]].copy()
        out["pred_margin"], out["pred_total"] = pred_margin, pred_total
        out["fold"], out["candidate"] = fold.name, candidate.name
        invalid_reasons: list[dict] = []
        bundle = {"model": model}
    else:
        x_train = train[feature_columns]
        x_val = validate[feature_columns]
        margin_pipe, margin_pred, margin_ok, margin_reason = _fit_one_ridge_or_huber_target(
            candidate.family, candidate.hyperparameters, x_train, train["home_margin"].to_numpy(float), x_val
        )
        total_pipe, total_pred, total_ok, total_reason = _fit_one_ridge_or_huber_target(
            candidate.family, candidate.hyperparameters, x_train, train["total_points"].to_numpy(float), x_val
        )
        invalid_reasons = []
        if not margin_ok:
            invalid_reasons.append({"fold": fold.name, "target": "margin", "reason": margin_reason})
        if not total_ok:
            invalid_reasons.append({"fold": fold.name, "target": "total", "reason": total_reason})

        out = validate[["game_id", "season", "week", "home_margin", "total_points"]].copy()
        out["pred_margin"] = margin_pred if margin_ok else np.nan
        out["pred_total"] = total_pred if total_ok else np.nan
        out["fold"], out["candidate"] = fold.name, candidate.name
        bundle = {"margin": margin_pipe, "total": total_pipe}

    if return_fitted:
        return out, invalid_reasons, bundle
    return out, invalid_reasons


def predict_with_fitted_bundle(bundle: dict, feature_frame: pd.DataFrame, family: str) -> pd.DataFrame:
    """Applies an ALREADY-FITTED bundle (from ``fit_predict_fold(...,
    return_fitted=True)``) to a NEW feature frame -- no fit call anywhere in
    this function. Used by the poison-pill tests and Phase 8 interpretability
    to guarantee zero additional real fits."""
    if family == "HGBR":
        margin, total = bundle["model"].predict_means(feature_frame)
    else:
        margin = bundle["margin"].predict(feature_frame)
        total = bundle["total"].predict(feature_frame)
    return pd.DataFrame(
        {"pred_margin": np.asarray(margin, dtype=float), "pred_total": np.asarray(total, dtype=float)}
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Primary metric + analytic paired SE (no bootstrap, no RNG, no seed).
# ---------------------------------------------------------------------------
def _primary_score(preds: pd.DataFrame) -> float:
    margin = regression_metrics(preds["home_margin"].to_numpy(float), preds["pred_margin"].to_numpy(float))
    total = regression_metrics(preds["total_points"].to_numpy(float), preds["pred_total"].to_numpy(float))
    return 0.5 * (margin["rmse"] + total["rmse"])


def _regression_report(preds: pd.DataFrame) -> dict:
    margin = regression_metrics(preds["home_margin"].to_numpy(float), preds["pred_margin"].to_numpy(float))
    total = regression_metrics(preds["total_points"].to_numpy(float), preds["pred_total"].to_numpy(float))
    return {"margin": margin, "total": total, "primary_score": 0.5 * (margin["rmse"] + total["rmse"])}


def paired_delta_se(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> tuple[float, float, int]:
    """``delta_i = L_a_i - L_b_i`` (negative => a is better than b).
    ``SE(delta) = sample_std(delta, ddof=1) / sqrt(n)``. Hard STOP if the two
    ledgers cannot be paired one-to-one by ``(game_id, fold)``."""
    merged = preds_a.merge(
        preds_b[["game_id", "fold", "pred_margin", "pred_total"]],
        on=["game_id", "fold"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(merged) != len(preds_a) or len(merged) != len(preds_b):
        raise HardGateFailure("UNPAIRABLE_PREDICTION_LEDGERS: candidate game_id sets differ")
    n = len(merged)
    margin_actual = merged["home_margin"].to_numpy(float)
    total_actual = merged["total_points"].to_numpy(float)
    l_a = 0.5 * ((margin_actual - merged["pred_margin_a"].to_numpy(float)) ** 2 + (total_actual - merged["pred_total_a"].to_numpy(float)) ** 2)
    l_b = 0.5 * ((margin_actual - merged["pred_margin_b"].to_numpy(float)) ** 2 + (total_actual - merged["pred_total_b"].to_numpy(float)) ** 2)
    delta = l_a - l_b
    mean_delta = float(delta.mean())
    se_delta = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean_delta, se_delta, n


# ---------------------------------------------------------------------------
# Frozen selection-matrix fingerprint (correction #1: computed BEFORE the
# preregistration hash, not "alongside" it).
# ---------------------------------------------------------------------------
def compute_selection_matrix_hash(matrix: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[str, int, list[str]]:
    cols = ["game_id", "season", "week"] + list(feature_columns) + ["home_margin", "total_points"]
    ordered = matrix[cols].sort_values("game_id").reset_index(drop=True)
    records = ordered.astype(object).where(pd.notnull(ordered), None).to_dict(orient="records")
    payload = {"schema_version": SCHEMA_VERSION, "columns": cols, "row_count": len(ordered), "rows": records}
    digest = _sha256_hex(payload)
    return digest, len(ordered), cols


# ---------------------------------------------------------------------------
# Prediction cache -- keyed by candidate_spec_hash/fold_id/Fix6 feature hash/
# selection_matrix_hash/preregistration_hash (Section 7). On a fold set with
# a fixed candidate registry (no forward/backward search) every key is only
# ever requested once by construction, so this is a correctness/defense-in-
# depth layer, not a performance optimization -- but it is wired exactly as
# specified, including the game_id-set verification on a hit.
# ---------------------------------------------------------------------------
class PredictionCache:
    def __init__(self) -> None:
        self._store: dict[tuple, tuple[pd.DataFrame, list[dict]]] = {}

    def get_or_fit(self, key: tuple, expected_game_ids: frozenset, compute_fn):
        cached = self._store.get(key)
        if cached is not None:
            preds, invalid_reasons = cached
            if frozenset(preds["game_id"]) != expected_game_ids:
                cached = None  # mismatch -> discard and recompute (counted against the real fit budget)
        if cached is None:
            preds, invalid_reasons = compute_fn()
            self._store[key] = (preds, invalid_reasons)
            return preds, invalid_reasons, False
        return cached[0], cached[1], True


def make_cache_key(
    candidate: CandidateSpec, fold_name: str, fix6_feature_manifest_hash: str, selection_matrix_hash: str, preregistration_hash: str
) -> tuple:
    return (compute_candidate_spec_hash(candidate), fold_name, fix6_feature_manifest_hash, selection_matrix_hash, preregistration_hash)


def compute_preregistration_hash(preregistration_without_hash: dict) -> str:
    return _sha256_hex({**preregistration_without_hash, "schema_version": SCHEMA_VERSION})


# ---------------------------------------------------------------------------
# Inner fold execution (Phase 4).
# ---------------------------------------------------------------------------
def run_inner_fits(
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    registry: tuple[CandidateSpec, ...] = CANDIDATE_REGISTRY,
    *,
    cache: "PredictionCache | None" = None,
    fix6_feature_manifest_hash: str = "",
    selection_matrix_hash: str = "",
    preregistration_hash: str = "",
):
    """Fits every candidate on every inner fold once. Returns
    ``(inner_results, huber_invalid_reasons, real_data_paired_fit_count,
    real_data_individual_fit_count)``. ``inner_results[candidate_name][fold_name]``
    is that candidate/fold's validation-prediction ledger."""
    cache = cache if cache is not None else PredictionCache()
    inner_results: dict[str, dict[str, pd.DataFrame]] = {}
    huber_invalid_reasons: list[dict] = []
    paired_fit_count = 0
    individual_fit_count = 0
    for candidate in registry:
        inner_results[candidate.name] = {}
        for fold in fd.INNER_FOLDS:
            key = make_cache_key(candidate, fold.name, fix6_feature_manifest_hash, selection_matrix_hash, preregistration_hash)
            expected_game_ids = frozenset(matrix.loc[matrix["season"] == fold.validate_season, "game_id"])
            preds, invalid_reasons, was_cached = cache.get_or_fit(
                key, expected_game_ids, lambda c=candidate, f=fold: fit_predict_fold(matrix, feature_columns, f, c)
            )
            inner_results[candidate.name][fold.name] = preds
            huber_invalid_reasons.extend(invalid_reasons)
            if not was_cached:
                paired_fit_count += 1
                individual_fit_count += 2  # one margin estimator + one total estimator, every family
    return inner_results, huber_invalid_reasons, paired_fit_count, individual_fit_count


def _pooled(inner_results: dict[str, dict[str, pd.DataFrame]], candidate_name: str) -> pd.DataFrame:
    return pd.concat([inner_results[candidate_name][f.name] for f in fd.INNER_FOLDS], ignore_index=True)


# ---------------------------------------------------------------------------
# Ridge alpha selection (Phase 5): largest alpha within one SE of best.
# ---------------------------------------------------------------------------
def select_ridge_alpha(inner_results: dict[str, dict[str, pd.DataFrame]], registry: tuple[CandidateSpec, ...] = CANDIDATE_REGISTRY) -> dict:
    pooled = {name: _pooled(inner_results, name) for name in RIDGE_ALPHA_NAMES}
    pooled_primary = {name: _primary_score(pooled[name]) for name in RIDGE_ALPHA_NAMES}
    best_alpha_name = min(RIDGE_ALPHA_NAMES, key=lambda n: pooled_primary[n])

    alpha_value = {c.name: c.hyperparameters["alpha"] for c in registry if c.family == "RIDGE"}

    per_alpha: dict[str, dict] = {}
    qualifying = {best_alpha_name}
    for name in RIDGE_ALPHA_NAMES:
        if name == best_alpha_name:
            per_alpha[name] = {
                "alpha": alpha_value[name], "pooled_primary_score": pooled_primary[name],
                "mean_delta_vs_best": 0.0, "se_delta_vs_best": 0.0, "within_one_se_of_best": True, "is_best": True,
            }
            continue
        mean_delta, se_delta, n = paired_delta_se(pooled[name], pooled[best_alpha_name])
        within = bool(mean_delta <= se_delta)
        if within:
            qualifying.add(name)
        per_alpha[name] = {
            "alpha": alpha_value[name], "pooled_primary_score": pooled_primary[name],
            "mean_delta_vs_best": mean_delta, "se_delta_vs_best": se_delta,
            "within_one_se_of_best": within, "is_best": False, "n": n,
        }

    selected_name = max(qualifying, key=lambda n: alpha_value[n])
    for name in RIDGE_ALPHA_NAMES:
        per_alpha[name]["selected"] = name == selected_name

    return {
        "best_alpha_name": best_alpha_name,
        "selected_alpha_name": selected_name,
        "selected_alpha_value": alpha_value[selected_name],
        "per_alpha": per_alpha,
    }


# ---------------------------------------------------------------------------
# Family selection (Phase 6): fully deterministic Steps A-D.
# ---------------------------------------------------------------------------
_COMPLEXITY_RANK = {"RIDGE": 0, "HUBER": 1, "HGBR": 2}


def select_family(inner_results: dict[str, dict[str, pd.DataFrame]], *, ridge_selected_name: str, huber_valid: bool) -> dict:
    family_representative = {"RIDGE": ridge_selected_name, "HGBR": "HGBR_INCUMBENT"}
    if huber_valid:
        family_representative["HUBER"] = "HUBER_FIXED"
    valid_families_in_order = [f for f in ("RIDGE", "HUBER", "HGBR") if f in family_representative]

    pooled = {fam: _pooled(inner_results, family_representative[fam]) for fam in valid_families_in_order}
    pooled_primary = {fam: _primary_score(pooled[fam]) for fam in valid_families_in_order}

    # Step A
    best = min(valid_families_in_order, key=lambda f: pooled_primary[f])

    # Step B
    one_se_set = set()
    comparisons_vs_best: dict[str, dict] = {}
    for fam in valid_families_in_order:
        if fam == best:
            one_se_set.add(fam)
            continue
        mean_delta, se_delta, n = paired_delta_se(pooled[fam], pooled[best])
        comparisons_vs_best[fam] = {"mean_delta": mean_delta, "se_delta": se_delta, "n": n}
        if mean_delta <= se_delta:
            one_se_set.add(fam)

    # Step C
    tentative = min(one_se_set, key=lambda f: _COMPLEXITY_RANK[f])

    # Step D
    lowest_complexity = min(valid_families_in_order, key=lambda f: _COMPLEXITY_RANK[f])
    lower_family_checks: dict[str, dict] = {}
    if tentative == lowest_complexity:
        selected, reason = tentative, "TENTATIVE_IS_LOWEST_COMPLEXITY"
    else:
        lower_families = sorted(
            (f for f in valid_families_in_order if _COMPLEXITY_RANK[f] < _COMPLEXITY_RANK[tentative]),
            key=lambda f: _COMPLEXITY_RANK[f],
        )
        all_passed = True
        first_failed = None
        for s in lower_families:
            wins = sum(
                1
                for fold in fd.INNER_FOLDS
                if _primary_score(inner_results[family_representative[tentative]][fold.name])
                < _primary_score(inner_results[family_representative[s]][fold.name])
            )
            passed = wins >= 2
            lower_family_checks[s] = {"wins": wins, "of": len(fd.INNER_FOLDS), "passed": passed}
            if not passed:
                all_passed = False
                if first_failed is None:
                    first_failed = s
        if all_passed:
            selected, reason = tentative, "TENTATIVE_REPRODUCIBLY_BEAT_ALL_LOWER_FAMILIES"
        else:
            selected, reason = first_failed, "TENTATIVE_FAILED_2OF3_AGAINST_LOWEST_UNBEATEN_LOWER_FAMILY"

    return {
        "family_representative": family_representative,
        "valid_families_in_order": valid_families_in_order,
        "best": best,
        "pooled_primary": pooled_primary,
        "one_se_set": sorted(one_se_set, key=lambda f: _COMPLEXITY_RANK[f]),
        "comparisons_vs_best": comparisons_vs_best,
        "tentative": tentative,
        "lower_family_checks": lower_family_checks,
        "selected_family": selected,
        "selection_reason": reason,
    }


FAMILY_SELECTION_ALGORITHM_SOURCE = (
    "Step A: best = argmin_family(pooled_inner_primary_score). "
    "Step B: one_se_set = {best} u {family : mean_delta(family) <= se_delta(family)}, "
    "delta_i = L_family_i - L_best_i pooled across inner folds. "
    "Step C: tentative = lowest-complexity family in one_se_set (RIDGE<HUBER<HGBR). "
    "Step D: if tentative is already the lowest-complexity valid family, select tentative "
    "(TENTATIVE_IS_LOWEST_COMPLEXITY). Otherwise, for every lower-complexity valid family s "
    "in ascending complexity order, require tentative's per-fold primary_score to beat s's "
    "in >=2 of 3 inner folds; if tentative beats every such s, select tentative "
    "(TENTATIVE_REPRODUCIBLY_BEAT_ALL_LOWER_FAMILIES); otherwise select the first (lowest-"
    "complexity) s that tentative failed to reproducibly beat "
    "(TENTATIVE_FAILED_2OF3_AGAINST_LOWEST_UNBEATEN_LOWER_FAMILY) -- never a jump to a "
    "different, more-complex family."
)


# ---------------------------------------------------------------------------
# Final model spec hash (Phase 6 freeze) -- specification only, no fitted
# parameters, no 2024 results.
# ---------------------------------------------------------------------------
def compute_final_model_spec_hash(
    *, fix6_feature_manifest_hash: str, frozen_feature_columns: Sequence[str], selected_family: str, selected_spec: CandidateSpec
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fix6_feature_manifest_hash": fix6_feature_manifest_hash,
        "ordered_feature_list": list(frozen_feature_columns),
        "selected_family": selected_family,
        "selected_candidate_spec": {
            "name": selected_spec.name,
            "family": selected_spec.family,
            "complexity_rank": selected_spec.complexity_rank,
            "preprocessing": selected_spec.preprocessing,
            "hyperparameters": selected_spec.hyperparameters,
            "target_sharing_rule": selected_spec.target_sharing_rule,
            "random_state": selected_spec.random_state,
        },
        "margin_target_definition": "home_score - away_score",
        "total_target_definition": "home_score + away_score",
        "preprocessing_specification": selected_spec.preprocessing,
        "determinism_specification": {"random_state": selected_spec.random_state, "single_threaded": True},
    }
    return _sha256_hex(payload)


# ---------------------------------------------------------------------------
# LOCKED_POST_EXPOSURE_2024_AUDIT (Phase 7) -- exact predeclared thresholds,
# frozen before any inner fit (persisted in preregistration).
# ---------------------------------------------------------------------------
AUDIT_PRIMARY_TOLERANCE = 0.02
AUDIT_MARGIN_TOLERANCE = 0.05
AUDIT_TOTAL_TOLERANCE = 0.05
AUDIT_WEEK1_TOLERANCE = 0.10
AUDIT_WEEKS2_4_TOLERANCE = 0.10
AUDIT_WEEKS5PLUS_TOLERANCE = 0.05

# Per the operator's contract (V0 Section 4): 2024 is treated as
# previously-exposed even for THIS (Fix 7) audit, because Fix 6's own outer
# confirmation already fit/evaluated a model on this identical six-feature
# input over this identical 2024 season -- so it is never described as an
# untouched/independent/first holdout here either.
AUDIT_2024_PREVIOUSLY_EXPOSED = True
AUDIT_2024_ROLE = "LOCKED_POST_EXPOSURE_AUDIT"
AUDIT_2024_USED_FOR_MODEL_SELECTION = False
AUDIT_2024_USED_FOR_ESTIMATOR_ADJUSTMENT = False


def run_locked_2024_audit(
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    finalists: dict[str, CandidateSpec],
    selected_family: str,
) -> dict:
    """Fits each finalist ONCE on ``season <= fd.OUTER_FOLD.train_max_season``
    (2023) and evaluates ONCE on ``season == fd.OUTER_FOLD.validate_season``
    (2024). Returns fitted bundles too (``return_fitted=True``) so Phase 8
    can reuse them with zero additional fits."""
    fd.assert_fold_seasons_allowed(fd.OUTER_FOLD)

    per_finalist: dict[str, dict] = {}
    fitted_bundles: dict[str, dict] = {}
    for family, spec in finalists.items():
        preds, invalid_reasons, bundle = fit_predict_fold(matrix, feature_columns, fd.OUTER_FOLD, spec, return_fitted=True)
        if invalid_reasons:
            raise HardGateFailure(f"LOCKED_AUDIT_HUBER_INVALID_NUMERICAL: {invalid_reasons}")
        fitted_bundles[family] = bundle
        preds = preds.assign(segment=fd._segment_labels(preds["week"]))
        report = _regression_report(preds)
        segment_report = {}
        for segment in ("week1", "weeks2_4", "weeks5_plus"):
            seg = preds[preds["segment"] == segment]
            segment_report[segment] = {"primary_score": _primary_score(seg)} if len(seg) else {"primary_score": None}
        per_finalist[family] = {
            "candidate_name": spec.name,
            "primary_score": report["primary_score"],
            "margin_rmse": report["margin"]["rmse"],
            "total_rmse": report["total"]["rmse"],
            "segment_primary_score": {k: v["primary_score"] for k, v in segment_report.items()},
        }

    def _best(metric_path):
        values = [_dig(per_finalist[f], metric_path) for f in per_finalist if _dig(per_finalist[f], metric_path) is not None]
        return min(values)

    def _dig(d, path):
        for p in path:
            if d is None:
                return None
            d = d[p]
        return d

    best_2024_primary = _best(["primary_score"])
    best_2024_margin_rmse = _best(["margin_rmse"])
    best_2024_total_rmse = _best(["total_rmse"])
    best_week1_primary = _best(["segment_primary_score", "week1"])
    best_weeks2_4_primary = _best(["segment_primary_score", "weeks2_4"])
    best_weeks5plus_primary = _best(["segment_primary_score", "weeks5_plus"])

    selected = per_finalist[selected_family]
    rules = {
        "overall": {
            "selected": selected["primary_score"], "best": best_2024_primary, "tolerance": AUDIT_PRIMARY_TOLERANCE,
            "pass": bool(selected["primary_score"] <= best_2024_primary * (1 + AUDIT_PRIMARY_TOLERANCE)),
        },
        "margin": {
            "selected": selected["margin_rmse"], "best": best_2024_margin_rmse, "tolerance": AUDIT_MARGIN_TOLERANCE,
            "pass": bool(selected["margin_rmse"] <= best_2024_margin_rmse * (1 + AUDIT_MARGIN_TOLERANCE)),
        },
        "total": {
            "selected": selected["total_rmse"], "best": best_2024_total_rmse, "tolerance": AUDIT_TOTAL_TOLERANCE,
            "pass": bool(selected["total_rmse"] <= best_2024_total_rmse * (1 + AUDIT_TOTAL_TOLERANCE)),
        },
        "week1": {
            "selected": selected["segment_primary_score"]["week1"], "best": best_week1_primary, "tolerance": AUDIT_WEEK1_TOLERANCE,
            "pass": bool(selected["segment_primary_score"]["week1"] <= best_week1_primary * (1 + AUDIT_WEEK1_TOLERANCE)),
        },
        "weeks2_4": {
            "selected": selected["segment_primary_score"]["weeks2_4"], "best": best_weeks2_4_primary, "tolerance": AUDIT_WEEKS2_4_TOLERANCE,
            "pass": bool(selected["segment_primary_score"]["weeks2_4"] <= best_weeks2_4_primary * (1 + AUDIT_WEEKS2_4_TOLERANCE)),
        },
        "weeks5_plus": {
            "selected": selected["segment_primary_score"]["weeks5_plus"], "best": best_weeks5plus_primary, "tolerance": AUDIT_WEEKS5PLUS_TOLERANCE,
            "pass": bool(selected["segment_primary_score"]["weeks5_plus"] <= best_weeks5plus_primary * (1 + AUDIT_WEEKS5PLUS_TOLERANCE)),
        },
    }
    overall_pass = all(r["pass"] for r in rules.values())

    return {
        "fold": fd.OUTER_FOLD.name,
        "outer_2024_previously_exposed": AUDIT_2024_PREVIOUSLY_EXPOSED,
        "outer_2024_role": AUDIT_2024_ROLE,
        "outer_2024_used_for_model_selection": AUDIT_2024_USED_FOR_MODEL_SELECTION,
        "outer_2024_used_for_estimator_adjustment": AUDIT_2024_USED_FOR_ESTIMATOR_ADJUSTMENT,
        "selected_family": selected_family,
        "per_finalist": per_finalist,
        "rules": rules,
        "overall_pass": overall_pass,
        "fitted_bundles": fitted_bundles,
    }


# ---------------------------------------------------------------------------
# Phase 8: post-freeze interpretability -- zero additional real fits.
# ---------------------------------------------------------------------------
class _MarginOnlyPredictor(RegressorMixin, BaseEstimator):
    """No-fit sklearn-compatible view around an already-fitted
    JointScoreModel. Inherits ``BaseEstimator``/``RegressorMixin`` only for
    sklearn's tag machinery (``permutation_importance``'s scorer path calls
    ``is_regressor(estimator)`` -> ``get_tags(estimator)``, which requires
    ``__sklearn_tags__`` -- confirmed by running it against a plain object
    during implementation). Neither mixin defines ``fit``, so this does not
    reintroduce one: ``fit`` below exists ONLY to satisfy
    ``permutation_importance``'s ``HasMethods(["fit"])`` attribute-existence
    validation (confirmed by reading
    ``sklearn/inspection/_permutation_importance.py``: the function never
    calls ``.fit()`` and never calls ``check_is_fitted()``) -- it must never
    actually be invoked."""

    def __init__(self, fitted_model: "fd.JointScoreModel" = None) -> None:
        self.fitted_model = fitted_model

    def fit(self, X, y=None):  # pragma: no cover -- must never be invoked
        raise RuntimeError("adapter is prediction-only; fit must never be called")

    def predict(self, X):
        margin, _total = self.fitted_model.predict_means(X)  # public accessor only
        return margin


class _TotalOnlyPredictor(RegressorMixin, BaseEstimator):
    def __init__(self, fitted_model: "fd.JointScoreModel" = None) -> None:
        self.fitted_model = fitted_model

    def fit(self, X, y=None):  # pragma: no cover -- must never be invoked
        raise RuntimeError("adapter is prediction-only; fit must never be called")

    def predict(self, X):
        _margin, total = self.fitted_model.predict_means(X)
        return total


def run_hgbr_permutation_importance(fitted_model: "fd.JointScoreModel", audit_frame: pd.DataFrame, feature_columns: Sequence[str]) -> dict:
    feature_columns = list(feature_columns)
    margin_result = permutation_importance(
        _MarginOnlyPredictor(fitted_model), audit_frame[feature_columns], audit_frame["home_margin"],
        scoring="neg_root_mean_squared_error", n_repeats=25, random_state=fd.MODEL_CONFIG.random_state,
    )
    total_result = permutation_importance(
        _TotalOnlyPredictor(fitted_model), audit_frame[feature_columns], audit_frame["total_points"],
        scoring="neg_root_mean_squared_error", n_repeats=25, random_state=fd.MODEL_CONFIG.random_state,
    )
    return {
        "margin": {"importances_mean": dict(zip(feature_columns, margin_result.importances_mean.tolist())), "importances_std": dict(zip(feature_columns, margin_result.importances_std.tolist()))},
        "total": {"importances_mean": dict(zip(feature_columns, total_result.importances_mean.tolist())), "importances_std": dict(zip(feature_columns, total_result.importances_std.tolist()))},
        "scoring": "neg_root_mean_squared_error",
        "n_repeats": 25,
        "random_state": fd.MODEL_CONFIG.random_state,
    }


def ridge_or_huber_interpretability(bundle: dict, feature_columns: Sequence[str]) -> dict:
    feature_columns = list(feature_columns)
    result: dict = {}
    for target in ("margin", "total"):
        pipe = bundle[target]
        scaler: StandardScaler = pipe.named_steps["scaler"]
        estimator = pipe.named_steps["estimator"]
        coef = np.asarray(estimator.coef_, dtype=float)
        result[target] = {
            "intercept": float(estimator.intercept_),
            "standardized_coefficients": dict(zip(feature_columns, coef.tolist())),
            "original_scale_coefficients": dict(zip(feature_columns, (coef / scaler.scale_).tolist())),
        }
    result["elo_core_feature_relationship"] = ELO_CORE_FEATURE_RELATIONSHIP_TEXT
    return result


def local_sensitivity_example(
    bundle: dict, family: str, audit_matrix: pd.DataFrame, train_matrix: pd.DataFrame, feature_columns: Sequence[str]
) -> dict:
    """One 2024 example row; reference values are the MEDIAN of each
    feature computed from ``train_matrix`` (season<=2023) ONLY, never from
    any 2024 row. Uses the already-fitted ``bundle`` -- no refit."""
    feature_columns = list(feature_columns)
    example_row = audit_matrix.iloc[[0]]
    base = predict_with_fitted_bundle(bundle, example_row[feature_columns], family)
    reference_values = {feat: float(train_matrix[feat].median()) for feat in feature_columns}

    perturbation_deltas = {}
    for feat in feature_columns:
        perturbed = example_row.copy()
        perturbed[feat] = reference_values[feat]
        pred = predict_with_fitted_bundle(bundle, perturbed[feature_columns], family)
        perturbation_deltas[feat] = {
            "reference_value": reference_values[feat],
            "original_value": float(example_row[feat].iloc[0]),
            "margin_delta": float(pred["pred_margin"].iloc[0] - base["pred_margin"].iloc[0]),
            "total_delta": float(pred["pred_total"].iloc[0] - base["pred_total"].iloc[0]),
        }

    return {
        "label": "LOCAL_SENSITIVITY",
        "not_shapley_attribution": True,
        "not_additive_contribution": True,
        "example_game_id": str(example_row["game_id"].iloc[0]),
        "example_week": int(example_row["week"].iloc[0]),
        "predicted_margin": float(base["pred_margin"].iloc[0]),
        "predicted_total": float(base["pred_total"].iloc[0]),
        "reference_source": "outer model training data only (season<=2023); never a 2024 validation row",
        "perturbation_deltas": perturbation_deltas,
    }


# ---------------------------------------------------------------------------
# Fit budget (Section 11): pre-registered, checked before/while fits accrue.
# ---------------------------------------------------------------------------
EXPECTED_INNER_PAIRED_FITS = len(CANDIDATE_REGISTRY) * 3  # 6 candidates x 3 inner folds = 18
EXPECTED_INNER_INDIVIDUAL_FITS = EXPECTED_INNER_PAIRED_FITS * 2  # 36
EXPECTED_OUTER_PAIRED_FITS = 3  # up to 3 finalists, one fit each
EXPECTED_OUTER_INDIVIDUAL_FITS = EXPECTED_OUTER_PAIRED_FITS * 2  # 6
EXPECTED_TOTAL_PAIRED_FITS = EXPECTED_INNER_PAIRED_FITS + EXPECTED_OUTER_PAIRED_FITS  # 21
EXPECTED_TOTAL_INDIVIDUAL_FITS = EXPECTED_INNER_INDIVIDUAL_FITS + EXPECTED_OUTER_INDIVIDUAL_FITS  # 42
MAX_PAIRED_FITS = 30
MAX_INDIVIDUAL_FITS = 60


def plan_fit_budget() -> dict:
    return {
        "expected_paired": EXPECTED_TOTAL_PAIRED_FITS,
        "expected_individual": EXPECTED_TOTAL_INDIVIDUAL_FITS,
        "max_paired": MAX_PAIRED_FITS,
        "max_individual": MAX_INDIVIDUAL_FITS,
        "fit_budget_ok": EXPECTED_TOTAL_PAIRED_FITS <= MAX_PAIRED_FITS and EXPECTED_TOTAL_INDIVIDUAL_FITS <= MAX_INDIVIDUAL_FITS,
    }
