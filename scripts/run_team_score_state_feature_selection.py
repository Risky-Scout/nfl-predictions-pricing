"""TEAM_SCORE_STATE_V1.1 feature selection -- corrected operator-contract rerun.

Branch: research/team-score-state-v1-2026
Certified baseline (untouched): v2026.1-fix8-certified / d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7

Corrected rerun of the invalidated TEAM_SCORE_STATE_V1 (V0) attempt -- see
``$NFL_MODEL_ARTIFACT_ROOT/team-score-state-v1-2026/invalidated-as-built-v0/
invalidation_manifest.json``. V0's feature semantics, candidate columns,
adoption rules, and bootstrap seed did not match the operator contract; none
of its candidate predictions, selection freeze, or winner arithmetic are
reused here as scientific evidence. This run starts from the certified
ELO_ONLY baseline and tests only:
  A. ELO_ONLY
  B. ELO_PLUS_SCORE_SIGNALS
  C. ELO_PLUS_SCORE_COMPONENTS
against :mod:`nfl_hybrid.features.team_score_state`'s corrected V1.1 semantics
(league-relative offense/defense deviations, matchup expected-score-deviation
margin/total signals).

Mirrors the QB_LAGGED_DEPTH_STATE_V1 task's process byte-for-byte
(preregistration/freeze/bootstrap/adoption-gate machinery, Fix-7 primary
score, RIDGE_ALPHA_100, original <=2023 folds) with the operator contract's
corrected bootstrap seed (20260826) and adoption-gate RMSE slack (+0.05) --
only the feature contract under test and these two values differ from V0. No
QB research is imported; this branch never touches
research/qb-lagged-depth-feature-selection-2026 or commit 2959904.

Writes evidence under $NFL_MODEL_ARTIFACT_ROOT/team-score-state-v1-2026/
(alongside the preserved invalidated-as-built-v0/ V0 evidence). No commits.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_hybrid.data.external_data import artifact_root, resolve
from nfl_hybrid.evaluation import official_horizon_oof as ooof
from nfl_hybrid.evaluation.metrics import regression_metrics
from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.features import team_score_state as tss

TASK_SLUG = "team-score-state-v1-2026"
SCHEMA_VERSION = "team-score-state-feature-selection-v1.1"
SEED = 20260826
N_BOOTSTRAP = 10000
HORIZONS = ("TUE", "FRI")
RMSE_ADOPTION_SLACK = 0.05

ELO_FEATURE_COLUMNS: tuple[str, ...] = ooof.ELO_FEATURE_COLUMNS
CANDIDATE_GROUPS: dict[str, tuple[str, ...]] = {
    "ELO_ONLY": ELO_FEATURE_COLUMNS,
    "ELO_PLUS_SCORE_SIGNALS": ELO_FEATURE_COLUMNS + tss.CANDIDATE_SIGNAL_COLUMNS,
    "ELO_PLUS_SCORE_COMPONENTS": ELO_FEATURE_COLUMNS + tss.CANDIDATE_COMPONENT_COLUMNS,
}
SCORE_CANDIDATES = ("ELO_PLUS_SCORE_SIGNALS", "ELO_PLUS_SCORE_COMPONENTS")

RIDGE_HYPERPARAMETERS = {"alpha": 100.0, "fit_intercept": True, "solver": "svd"}
RIDGE_PREPROCESSING = {"type": "StandardScaler", "with_mean": True, "with_std": True}


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_max_season: int
    validate_season: int


SELECTION_FOLDS: tuple[FoldSpec, ...] = (
    FoldSpec("A", 2020, 2021),
    FoldSpec("B", 2021, 2022),
    FoldSpec("C", 2022, 2023),
)


class HardGateFailure(RuntimeError):
    pass


def _artifact_dir() -> Path:
    d = artifact_root() / TASK_SLUG
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256_hex(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_games_all() -> pd.DataFrame:
    return pd.read_parquet(resolve("backfill.games"))


def build_feature_matrix(games: pd.DataFrame, horizon: str, ledger: pd.DataFrame) -> pd.DataFrame:
    elo_matrix = ooof.build_official_horizon_matrix(games, horizon, ledger)
    score_matrix = tss.build_team_score_feature_matrix(games, horizon, membership_ledger=ledger)
    merged = elo_matrix.merge(score_matrix, on="game_id", how="inner", validate="one_to_one")
    if len(merged) != len(elo_matrix):
        raise HardGateFailure("Team score-state matrix did not cover every ELO-eligible target (population mismatch).")
    return merged


# ---------------------------------------------------------------------------
# Section: semantics
# ---------------------------------------------------------------------------
def write_semantics() -> str:
    semantics = tss.team_score_state_v1_1_semantics()
    semantics_hash = tss.compute_semantics_hash(semantics)
    _write_json(_artifact_dir() / "team_score_state_v1_1_semantics.json", semantics)
    return semantics_hash


# ---------------------------------------------------------------------------
# Preregistration
# ---------------------------------------------------------------------------
def write_preregistration(semantics_hash: str) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "TEAM_SCORE_STATE_V1_1_FEATURE_SELECTION_2026_CORRECTED_RERUN",
        "supersedes": "team_score_state_feature_selection_preregistration.json (V0, invalidated -- non-authoritative)",
        "semantics_hash": semantics_hash,
        "baseline_feature_set": {"name": "CERTIFIED_ELO_BASELINE", "columns": list(ELO_FEATURE_COLUMNS)},
        "candidate_groups": {name: list(cols) for name, cols in CANDIDATE_GROUPS.items()},
        "candidate_b_is_superset_of_candidate_c": False,
        "candidate_b_is_subset_of_candidate_c": False,
        "estimator": {
            "preprocessing": RIDGE_PREPROCESSING, "hyperparameters": RIDGE_HYPERPARAMETERS,
            "family": "RIDGE", "targets_fit_separately": True,
        },
        "folds": [{"name": f.name, "train_max_season": f.train_max_season, "validate_season": f.validate_season} for f in SELECTION_FOLDS],
        "horizons": list(HORIZONS),
        "target_scope": "REG+POST",
        "primary_metric_formula": "primary_score = 0.5 * (margin_RMSE + total_RMSE) (Fix-7 definition, reused verbatim)",
        "bootstrap": {
            "resamples": N_BOOTSTRAP, "seed": SEED, "unit": "game_id_cluster",
            "quantities": ["primary_loss_delta", "margin_squared_error_delta", "total_squared_error_delta"],
            "ci": "95% percentile", "negative_delta_means": "score-state candidate better",
        },
        "adoption_rule": {
            "conditions": [
                "pooled primary score < ELO_ONLY",
                f"pooled margin RMSE <= ELO_ONLY margin RMSE + {RMSE_ADOPTION_SLACK}",
                f"pooled total RMSE <= ELO_ONLY total RMSE + {RMSE_ADOPTION_SLACK}",
                "95% cluster-bootstrap CI for pooled primary-loss delta vs ELO_ONLY lies entirely below zero",
                "neither TUE nor FRI pooled primary score worse than ELO_ONLY by more than 0.10 points",
                "no single validation fold primary score worse than ELO_ONLY by more than 0.15 points",
            ],
            "tie_break": "if both score-state candidates pass, choose the simpler group (SIGNALS) if within one SE of the better passing candidate",
            "if_neither_passes": "retain ELO_ONLY",
        },
        "fit_budget_plan": {
            "expected_paired": len(CANDIDATE_GROUPS) * len(SELECTION_FOLDS) * len(HORIZONS),
            "expected_individual": len(CANDIDATE_GROUPS) * len(SELECTION_FOLDS) * len(HORIZONS) * 2,
            "planned_paired": 18, "planned_individual": 36,
            "hard_ceiling_paired": 24, "hard_ceiling_individual": 48,
        },
        "season_2024_role": "LOCKED_POST_EXPOSURE_2024_DIAGNOSTIC",
        "season_2025_role": "POST_EXPOSURE_2025_DIAGNOSTIC_ONLY",
        "no_market_features": True,
        "no_qb_epa_injury_weather_features": True,
        "qb_research_excluded": {"branch": "research/qb-lagged-depth-feature-selection-2026", "commit": "2959904", "imported": False},
        "selection_forbidden_seasons": [2024, 2025, 2026],
        "no_hidden_selection_reruns": True,
    }
    prereg_hash = _sha256_hex(payload)
    payload["team_score_state_feature_selection_preregistration_v1_1_hash"] = prereg_hash
    _write_json(_artifact_dir() / "team_score_state_feature_selection_preregistration_v1_1.json", payload)
    return prereg_hash


def verify_preregistration_hash(prereg_hash: str) -> None:
    path = _artifact_dir() / "team_score_state_feature_selection_preregistration_v1_1.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = on_disk.pop("team_score_state_feature_selection_preregistration_v1_1_hash")
    recomputed = _sha256_hex(on_disk)
    if recomputed != stored_hash or recomputed != prereg_hash:
        raise HardGateFailure("Preregistration hash verification failed at fit entry point.")


# ---------------------------------------------------------------------------
# Ridge fitting
# ---------------------------------------------------------------------------
def _build_ridge_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("estimator", Ridge(**RIDGE_HYPERPARAMETERS))])


FIT_COUNTER = {"paired": 0, "individual": 0}


def fit_predict_fold(matrix: pd.DataFrame, feature_columns: tuple[str, ...], fold: FoldSpec) -> pd.DataFrame:
    train = matrix[matrix["season"] <= fold.train_max_season]
    validate = matrix[matrix["season"] == fold.validate_season]
    if train.empty or validate.empty:
        raise HardGateFailure(f"Empty train/validate split for fold {fold.name}.")

    margin_pipe = _build_ridge_pipeline()
    margin_pipe.fit(train[list(feature_columns)], train["home_margin"].to_numpy(float))
    FIT_COUNTER["individual"] += 1
    total_pipe = _build_ridge_pipeline()
    total_pipe.fit(train[list(feature_columns)], train["total_points"].to_numpy(float))
    FIT_COUNTER["individual"] += 1
    FIT_COUNTER["paired"] += 1

    out = validate[["game_id", "season", "home_margin", "total_points"]].copy()
    out["pred_margin"] = margin_pipe.predict(validate[list(feature_columns)])
    out["pred_total"] = total_pipe.predict(validate[list(feature_columns)])
    out["fold"] = fold.name
    return out


def regression_report(preds: pd.DataFrame) -> dict:
    margin = regression_metrics(preds["home_margin"].to_numpy(float), preds["pred_margin"].to_numpy(float))
    total = regression_metrics(preds["total_points"].to_numpy(float), preds["pred_total"].to_numpy(float))
    return {"margin": margin, "total": total, "primary_score": 0.5 * (margin["rmse"] + total["rmse"])}


def run_selection_fits(games_2023: pd.DataFrame, ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    matrices = {h: build_feature_matrix(games_2023, h, ledger) for h in HORIZONS}

    metric_rows: list[dict] = []
    pooled_preds: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate, feature_columns in CANDIDATE_GROUPS.items():
        for horizon in HORIZONS:
            per_fold = []
            for fold in SELECTION_FOLDS:
                preds = fit_predict_fold(matrices[horizon], feature_columns, fold)
                preds["horizon"] = horizon
                per_fold.append(preds)
                report = regression_report(preds)
                metric_rows.append(
                    {
                        "candidate": candidate, "fold": fold.name, "horizon": horizon,
                        "margin_rmse": report["margin"]["rmse"], "margin_mae": report["margin"]["mae"],
                        "total_rmse": report["total"]["rmse"], "total_mae": report["total"]["mae"],
                        "primary_score": report["primary_score"], "n": len(preds),
                    }
                )
            pooled_preds[(candidate, horizon)] = pd.concat(per_fold, ignore_index=True)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(_artifact_dir() / "team_score_state_v1_1_selection_metrics.csv", index=False)

    if FIT_COUNTER["paired"] != 18 or FIT_COUNTER["individual"] != 36:
        raise HardGateFailure(f"Fit count deviated from the planned 18 paired / 36 individual: {FIT_COUNTER}")
    return metrics_df, pooled_preds


# ---------------------------------------------------------------------------
# Pooled comparison helpers
# ---------------------------------------------------------------------------
def pooled_frame(pooled_preds: dict[tuple[str, str], pd.DataFrame], candidate: str) -> pd.DataFrame:
    return pd.concat([pooled_preds[(candidate, h)] for h in HORIZONS], ignore_index=True)


def paired_losses(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> pd.DataFrame:
    merged = preds_a.merge(
        preds_b[["game_id", "fold", "horizon", "pred_margin", "pred_total"]],
        on=["game_id", "fold", "horizon"], suffixes=("_a", "_b"), validate="one_to_one",
    )
    if len(merged) != len(preds_a) or len(merged) != len(preds_b):
        raise HardGateFailure("UNPAIRABLE_PREDICTION_LEDGERS: candidate game_id/fold/horizon sets differ")
    margin_err_a = merged["home_margin"].to_numpy(float) - merged["pred_margin_a"].to_numpy(float)
    total_err_a = merged["total_points"].to_numpy(float) - merged["pred_total_a"].to_numpy(float)
    margin_err_b = merged["home_margin"].to_numpy(float) - merged["pred_margin_b"].to_numpy(float)
    total_err_b = merged["total_points"].to_numpy(float) - merged["pred_total_b"].to_numpy(float)
    merged["l_a"] = 0.5 * (margin_err_a ** 2 + total_err_a ** 2)
    merged["l_b"] = 0.5 * (margin_err_b ** 2 + total_err_b ** 2)
    merged["margin_se_a"] = margin_err_a ** 2
    merged["margin_se_b"] = margin_err_b ** 2
    merged["total_se_a"] = total_err_a ** 2
    merged["total_se_b"] = total_err_b ** 2
    return merged


def bootstrap_cluster_delta(paired: pd.DataFrame, *, value_col_a: str, value_col_b: str, rng: np.random.Generator) -> dict:
    game_ids = paired["game_id"].unique()
    n_games = len(game_ids)
    game_index = {g: i for i, g in enumerate(game_ids)}
    row_game_idx = paired["game_id"].map(game_index).to_numpy()

    delta = (paired[value_col_a] - paired[value_col_b]).to_numpy(float)
    sum_delta = np.zeros(n_games)
    count = np.zeros(n_games)
    np.add.at(sum_delta, row_game_idx, delta)
    np.add.at(count, row_game_idx, 1.0)

    observed_mean = float(delta.mean())
    draws = rng.integers(0, n_games, size=(N_BOOTSTRAP, n_games))
    boot_sum = sum_delta[draws].sum(axis=1)
    boot_count = count[draws].sum(axis=1)
    boot_means = boot_sum / boot_count

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "observed_mean_delta": observed_mean, "ci_low": float(lo), "ci_high": float(hi),
        "ci_entirely_below_zero": bool(hi < 0.0), "n_games": int(n_games), "n_rows": int(len(paired)),
    }


# ---------------------------------------------------------------------------
# Bootstrap + adoption gate
# ---------------------------------------------------------------------------
def run_bootstrap_and_adoption(metrics_df: pd.DataFrame, pooled_preds: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    elo_pooled = pooled_frame(pooled_preds, "ELO_ONLY")

    bootstrap_rows = []
    adoption: dict[str, dict] = {}
    for candidate in SCORE_CANDIDATES:
        cand_pooled = pooled_frame(pooled_preds, candidate)
        paired = paired_losses(cand_pooled, elo_pooled)

        primary_loss = bootstrap_cluster_delta(paired, value_col_a="l_a", value_col_b="l_b", rng=rng)
        margin_se = bootstrap_cluster_delta(paired, value_col_a="margin_se_a", value_col_b="margin_se_b", rng=rng)
        total_se = bootstrap_cluster_delta(paired, value_col_a="total_se_a", value_col_b="total_se_b", rng=rng)
        for quantity, result in (("primary_loss_delta", primary_loss), ("margin_squared_error_delta", margin_se), ("total_squared_error_delta", total_se)):
            bootstrap_rows.append({"candidate": candidate, "quantity": quantity, **result})

        pooled_report_cand = regression_report(cand_pooled)
        pooled_report_elo = regression_report(elo_pooled)

        cond1 = pooled_report_cand["primary_score"] < pooled_report_elo["primary_score"]
        cond2 = pooled_report_cand["margin"]["rmse"] <= pooled_report_elo["margin"]["rmse"] + RMSE_ADOPTION_SLACK
        cond3 = pooled_report_cand["total"]["rmse"] <= pooled_report_elo["total"]["rmse"] + RMSE_ADOPTION_SLACK
        cond4 = primary_loss["ci_entirely_below_zero"]

        horizon_diffs = {}
        cond5 = True
        for horizon in HORIZONS:
            cand_h = regression_report(pooled_preds[(candidate, horizon)])
            elo_h = regression_report(pooled_preds[("ELO_ONLY", horizon)])
            diff = cand_h["primary_score"] - elo_h["primary_score"]
            horizon_diffs[horizon] = diff
            if diff > 0.10:
                cond5 = False

        fold_diffs = {}
        cond6 = True
        for fold in SELECTION_FOLDS:
            cand_fold_rows = pd.concat(
                [pooled_preds[(candidate, h)][pooled_preds[(candidate, h)]["fold"] == fold.name] for h in HORIZONS], ignore_index=True
            )
            elo_fold_rows = pd.concat(
                [pooled_preds[("ELO_ONLY", h)][pooled_preds[("ELO_ONLY", h)]["fold"] == fold.name] for h in HORIZONS], ignore_index=True
            )
            diff = regression_report(cand_fold_rows)["primary_score"] - regression_report(elo_fold_rows)["primary_score"]
            fold_diffs[fold.name] = diff
            if diff > 0.15:
                cond6 = False

        conditions = {
            "pooled_primary_score_lt_elo": cond1, "pooled_margin_rmse_le_elo_plus_slack": cond2,
            "pooled_total_rmse_le_elo_plus_slack": cond3, "bootstrap_ci_entirely_below_zero": cond4,
            "no_horizon_worse_by_gt_0_10": cond5, "no_fold_worse_by_gt_0_15": cond6,
        }
        adoption[candidate] = {
            "pooled_candidate_primary_score": pooled_report_cand["primary_score"],
            "pooled_elo_primary_score": pooled_report_elo["primary_score"],
            "pooled_candidate_margin_rmse": pooled_report_cand["margin"]["rmse"],
            "pooled_elo_margin_rmse": pooled_report_elo["margin"]["rmse"],
            "pooled_candidate_total_rmse": pooled_report_cand["total"]["rmse"],
            "pooled_elo_total_rmse": pooled_report_elo["total"]["rmse"],
            "horizon_primary_score_diff": horizon_diffs, "fold_primary_score_diff": fold_diffs,
            "conditions": conditions, "adoptable": all(conditions.values()),
        }

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(_artifact_dir() / "team_score_state_v1_1_bootstrap.csv", index=False)
    return bootstrap_df, adoption


def select_winner(adoption: dict, pooled_preds: dict[tuple[str, str], pd.DataFrame]) -> dict:
    passing = [c for c in SCORE_CANDIDATES if adoption[c]["adoptable"]]
    if not passing:
        return {"selected_candidate": "ELO_ONLY", "reason": "NEITHER_SCORE_CANDIDATE_PASSED_ADOPTION_GATE"}
    if len(passing) == 1:
        return {"selected_candidate": passing[0], "reason": "SOLE_PASSING_CANDIDATE"}

    simpler, richer = "ELO_PLUS_SCORE_SIGNALS", "ELO_PLUS_SCORE_COMPONENTS"
    scores = {c: adoption[c]["pooled_candidate_primary_score"] for c in passing}
    better = simpler if scores[simpler] <= scores[richer] else richer
    if better == simpler:
        return {"selected_candidate": simpler, "reason": "SIMPLER_CANDIDATE_ALREADY_BETTER_OR_EQUAL"}

    simpler_pooled = pooled_frame(pooled_preds, simpler)
    richer_pooled = pooled_frame(pooled_preds, richer)
    paired = paired_losses(simpler_pooled, richer_pooled)
    delta = (paired["l_a"] - paired["l_b"]).to_numpy(float)
    n = len(delta)
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    within_one_se = abs(scores[simpler] - scores[richer]) <= se
    if within_one_se:
        return {"selected_candidate": simpler, "reason": "BOTH_PASSED_SIMPLER_WITHIN_ONE_SE_OF_BETTER", "se": se}
    return {"selected_candidate": richer, "reason": "BOTH_PASSED_RICHER_BEATS_SIMPLER_BY_MORE_THAN_ONE_SE", "se": se}


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
def write_freeze(semantics_hash: str, prereg_hash: str, metrics_df: pd.DataFrame, bootstrap_df: pd.DataFrame, adoption: dict, winner: dict) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "semantics_hash": semantics_hash,
        "preregistration_hash": prereg_hash,
        "selected_candidate": winner["selected_candidate"],
        "selection_reason": winner["reason"],
        "selected_feature_list": list(CANDIDATE_GROUPS[winner["selected_candidate"]]),
        "metrics": metrics_df.to_dict(orient="records"),
        "bootstrap": bootstrap_df.to_dict(orient="records"),
        "adoption_gate_results": adoption,
        "fit_counts": dict(FIT_COUNTER),
    }
    freeze_hash = _sha256_hex(payload)
    payload["team_score_state_feature_selection_freeze_v1_1_hash"] = freeze_hash
    _write_json(_artifact_dir() / "team_score_state_feature_selection_freeze_v1_1.json", payload)
    return freeze_hash


# ---------------------------------------------------------------------------
# 2024 (locked, post-exposure) + 2025 (diagnostic-only) evaluations
# ---------------------------------------------------------------------------
def _run_diagnostic(
    games_all: pd.DataFrame, max_season: int, fold: FoldSpec, winner_candidate: str, label: str, out_name: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    games = games_all[games_all["season"] <= max_season].copy()
    ledger = he.build_horizon_membership_ledger(games)

    rows = []
    preds_by_candidate: dict[str, pd.DataFrame] = {}
    candidates = ("ELO_ONLY",) if winner_candidate == "ELO_ONLY" else ("ELO_ONLY", winner_candidate)
    for candidate_name in candidates:
        feature_columns = CANDIDATE_GROUPS[candidate_name]
        per_horizon = []
        for horizon in HORIZONS:
            matrix = build_feature_matrix(games, horizon, ledger)
            preds = fit_predict_fold(matrix, feature_columns, fold)
            preds["horizon"] = horizon
            per_horizon.append(preds)
            report = regression_report(preds)
            rows.append(
                {
                    "label": label, "candidate": candidate_name, "horizon": horizon,
                    "margin_rmse": report["margin"]["rmse"], "margin_mae": report["margin"]["mae"],
                    "total_rmse": report["total"]["rmse"], "total_mae": report["total"]["mae"],
                    "primary_score": report["primary_score"], "n": len(preds),
                }
            )
        preds_by_candidate[candidate_name] = pd.concat(per_horizon, ignore_index=True)
    out = pd.DataFrame(rows)
    out.to_csv(_artifact_dir() / out_name, index=False)
    return out, preds_by_candidate


def run_2024_diagnostic(games_all: pd.DataFrame, winner_candidate: str) -> pd.DataFrame:
    out, _ = _run_diagnostic(
        games_all, 2024, FoldSpec("OUTER_2024", 2023, 2024), winner_candidate,
        "LOCKED_POST_EXPOSURE_2024_DIAGNOSTIC", "team_score_state_v1_1_2024_diagnostic.csv",
    )
    return out


def run_2025_diagnostic(games_all: pd.DataFrame, winner_candidate: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return _run_diagnostic(
        games_all, 2025, FoldSpec("EXPOSED_2025", 2024, 2025), winner_candidate,
        "POST_EXPOSURE_2025_DIAGNOSTIC_ONLY", "team_score_state_v1_1_2025_diagnostic.csv",
    )


# ---------------------------------------------------------------------------
# Section 15: 2025 market-relative diagnostic -- diagnostic only, no
# selection authority. Compares the frozen winner and ELO_ONLY to sportsbook
# consensus (the games table's own closing-line reference columns) on the
# identical 2025 rows produced by run_2025_diagnostic.
# ---------------------------------------------------------------------------
def run_2025_market_relative_diagnostic(
    games_all: pd.DataFrame, winner_candidate: str, preds_2025: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    market_cols = games_all[["game_id", "home_spread_reference", "total_line_reference"]].copy()
    market_cols["game_id"] = market_cols["game_id"].astype(str)

    rows: list[dict] = []
    candidates = ("ELO_ONLY",) if winner_candidate == "ELO_ONLY" else ("ELO_ONLY", winner_candidate)
    for candidate_name in candidates:
        preds = preds_2025[candidate_name].merge(market_cols, on="game_id", how="left", validate="many_to_one")
        preds = preds[preds["home_spread_reference"].notna() & preds["total_line_reference"].notna()].copy()
        preds["market_pred_margin"] = -pd.to_numeric(preds["home_spread_reference"], errors="coerce")
        preds["market_pred_total"] = pd.to_numeric(preds["total_line_reference"], errors="coerce")

        model_margin = regression_metrics(preds["home_margin"].to_numpy(float), preds["pred_margin"].to_numpy(float))
        model_total = regression_metrics(preds["total_points"].to_numpy(float), preds["pred_total"].to_numpy(float))
        market_margin = regression_metrics(preds["home_margin"].to_numpy(float), preds["market_pred_margin"].to_numpy(float))
        market_total = regression_metrics(preds["total_points"].to_numpy(float), preds["market_pred_total"].to_numpy(float))

        rows.append(
            {
                "candidate": candidate_name, "n": len(preds),
                "model_margin_mae": model_margin["mae"], "model_margin_rmse": model_margin["rmse"],
                "model_total_mae": model_total["mae"], "model_total_rmse": model_total["rmse"],
                "sportsbook_margin_mae": market_margin["mae"], "sportsbook_margin_rmse": market_margin["rmse"],
                "sportsbook_total_mae": market_total["mae"], "sportsbook_total_rmse": market_total["rmse"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(_artifact_dir() / "team_score_state_v1_1_2025_market_relative_diagnostic.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------
def write_summary_and_answers(
    *, adoption: dict, winner: dict, diag_2024: pd.DataFrame, diag_2025: pd.DataFrame,
    market_diag_2025: pd.DataFrame, semantics_hash: str, prereg_hash: str, freeze_hash: str,
) -> dict:
    q1 = f"{'YES' if adoption['ELO_PLUS_SCORE_SIGNALS']['adoptable'] else 'NO'} -- see adoption_gate_results.ELO_PLUS_SCORE_SIGNALS in the freeze file for the per-condition breakdown."
    q2 = f"{'YES' if adoption['ELO_PLUS_SCORE_COMPONENTS']['adoptable'] else 'NO'} -- see adoption_gate_results.ELO_PLUS_SCORE_COMPONENTS in the freeze file."
    q3 = f"{winner['selected_candidate']} was frozen on <=2023 evidence ({winner['reason']})."

    diag_pivot = diag_2024.pivot(index="horizon", columns="candidate", values="primary_score")
    q4_lines = [f"{h}: ELO_ONLY={diag_pivot.loc[h, 'ELO_ONLY']:.4f}" for h in HORIZONS]
    if winner["selected_candidate"] != "ELO_ONLY":
        for h in HORIZONS:
            q4_lines[HORIZONS.index(h)] += f", {winner['selected_candidate']}={diag_pivot.loc[h, winner['selected_candidate']]:.4f}"
    q4 = "LOCKED_POST_EXPOSURE_2024_DIAGNOSTIC (no selection authority): " + "; ".join(q4_lines)

    diag25_pivot = diag_2025.pivot(index="horizon", columns="candidate", values="primary_score")
    q5_lines = [f"{h}: ELO_ONLY={diag25_pivot.loc[h, 'ELO_ONLY']:.4f}" for h in HORIZONS]
    if winner["selected_candidate"] != "ELO_ONLY":
        for h in HORIZONS:
            q5_lines[HORIZONS.index(h)] += f", {winner['selected_candidate']}={diag25_pivot.loc[h, winner['selected_candidate']]:.4f}"
    q5 = "POST_EXPOSURE_2025_DIAGNOSTIC_ONLY (no selection authority): " + "; ".join(q5_lines)

    mkt_pivot = market_diag_2025.set_index("candidate")
    q5b_lines = [
        f"ELO_ONLY: model_margin_rmse={mkt_pivot.loc['ELO_ONLY', 'model_margin_rmse']:.4f} vs sportsbook={mkt_pivot.loc['ELO_ONLY', 'sportsbook_margin_rmse']:.4f}; "
        f"model_total_rmse={mkt_pivot.loc['ELO_ONLY', 'model_total_rmse']:.4f} vs sportsbook={mkt_pivot.loc['ELO_ONLY', 'sportsbook_total_rmse']:.4f}"
    ]
    if winner["selected_candidate"] != "ELO_ONLY":
        w = winner["selected_candidate"]
        q5b_lines.append(
            f"{w}: model_margin_rmse={mkt_pivot.loc[w, 'model_margin_rmse']:.4f} vs sportsbook={mkt_pivot.loc[w, 'sportsbook_margin_rmse']:.4f}; "
            f"model_total_rmse={mkt_pivot.loc[w, 'model_total_rmse']:.4f} vs sportsbook={mkt_pivot.loc[w, 'sportsbook_total_rmse']:.4f}"
        )
    q5b = "2025 market-relative diagnostic (no selection authority): " + "; ".join(q5b_lines)

    recertification_justified = winner["selected_candidate"] != "ELO_ONLY"
    q6 = f"{'YES' if recertification_justified else 'NO'} -- {'a score-state candidate passed every frozen adoption gate on <=2023 evidence' if recertification_justified else 'no score-state candidate passed every frozen adoption gate; the certified Elo baseline is retained'}."

    final_line = (
        "TEAM SCORE STATE V1.1 COMPLETE — FORMAL RECERTIFICATION JUSTIFIED"
        if recertification_justified
        else "TEAM SCORE STATE V1.1 COMPLETE — KEEP CERTIFIED ELO BASELINE"
    )
    summary = {
        "schema_version": SCHEMA_VERSION, "semantics_hash": semantics_hash, "preregistration_hash": prereg_hash,
        "freeze_hash": freeze_hash, "selected_candidate": winner["selected_candidate"], "selection_reason": winner["reason"],
        "answers": {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q5b": q5b, "Q6": q6}, "final_line": final_line,
    }
    _write_json(_artifact_dir() / "team_score_state_v1_1_feature_selection_summary.json", summary)
    return summary


def write_report_md(
    *, metrics_df: pd.DataFrame, bootstrap_df: pd.DataFrame, adoption: dict, winner: dict,
    diag_2024: pd.DataFrame, diag_2025: pd.DataFrame, market_diag_2025: pd.DataFrame, summary: dict,
) -> None:
    lines = [
        "# Team Score State V1.1 -- Corrected Operator-Contract Rerun Report",
        "",
        "**Branch:** `research/team-score-state-v1-2026`",
        "**Certified baseline (untouched):** `v2026.1-fix8-certified` / `d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7`",
        "**QB research excluded:** `research/qb-lagged-depth-feature-selection-2026` / commit `2959904` -- not imported.",
        "**Prior run:** TEAM_SCORE_STATE_V1 (V0) invalidated -- see `invalidated-as-built-v0/invalidation_manifest.json`. "
        "V0's feature semantics, candidate columns, adoption rules, and bootstrap seed did not match the operator "
        "contract; none of its predictions/freeze/winner arithmetic are reused as evidence here.",
        "",
        "## Corrected semantics",
        "",
        "League-relative offense/defense deviations (`team_points_for_mean`/`team_points_against_mean` minus a "
        "shared same-season, same-cutoff `league_points_per_team_game_mean`), combined into home/away "
        "expected-score-deviation matchup signals (`score_state_margin_signal`, `score_state_total_signal`). "
        "Frozen and disclosed before any fit -- see `team_score_state_v1_1_semantics.json`.",
        "",
        "## Selection metrics (folds A/B/C, <=2023)",
        "",
        metrics_df.to_string(index=False),
        "",
        f"## Bootstrap ({10000:,} resamples, seed {SEED}, game_id cluster)",
        "",
        bootstrap_df.to_string(index=False),
        "",
        "## Adoption gate results",
        "",
        json.dumps(adoption, indent=2, default=str),
        "",
        f"## Selected candidate: {winner['selected_candidate']} ({winner['reason']})",
        "",
        "## 2024 diagnostic (LOCKED_POST_EXPOSURE_2024_DIAGNOSTIC, no selection authority)",
        "",
        diag_2024.to_string(index=False),
        "",
        "## 2025 diagnostic (POST_EXPOSURE_2025_DIAGNOSTIC_ONLY, no selection authority)",
        "",
        diag_2025.to_string(index=False),
        "",
        "## 2025 market-relative diagnostic (sportsbook consensus, no selection authority)",
        "",
        market_diag_2025.to_string(index=False),
        "",
        "## Final answers",
        "",
    ]
    for key, value in summary["answers"].items():
        lines.append(f"**{key}.** {value}")
        lines.append("")
    lines.append("```")
    lines.append(summary["final_line"])
    lines.append("```")
    (_artifact_dir() / "TEAM_SCORE_STATE_V1_1_FEATURE_SELECTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    games_all = load_games_all()
    games_2023 = games_all[games_all["season"] <= 2023].copy()
    ledger = he.build_horizon_membership_ledger(games_2023)

    semantics_hash = write_semantics()
    prereg_hash = write_preregistration(semantics_hash)
    verify_preregistration_hash(prereg_hash)

    metrics_df, pooled_preds = run_selection_fits(games_2023, ledger)
    bootstrap_df, adoption = run_bootstrap_and_adoption(metrics_df, pooled_preds)
    winner = select_winner(adoption, pooled_preds)
    freeze_hash = write_freeze(semantics_hash, prereg_hash, metrics_df, bootstrap_df, adoption, winner)

    print(f"Fit counts after selection freeze: {FIT_COUNTER}")
    print(f"Selected candidate: {winner['selected_candidate']} ({winner['reason']})")

    # ONLY after the freeze file above is written may 2024/2025 be accessed.
    diag_2024 = run_2024_diagnostic(games_all, winner["selected_candidate"])
    diag_2025, preds_2025 = run_2025_diagnostic(games_all, winner["selected_candidate"])
    market_diag_2025 = run_2025_market_relative_diagnostic(games_all, winner["selected_candidate"], preds_2025)

    summary = write_summary_and_answers(
        adoption=adoption, winner=winner, diag_2024=diag_2024, diag_2025=diag_2025,
        market_diag_2025=market_diag_2025, semantics_hash=semantics_hash, prereg_hash=prereg_hash, freeze_hash=freeze_hash,
    )
    write_report_md(
        metrics_df=metrics_df, bootstrap_df=bootstrap_df, adoption=adoption, winner=winner,
        diag_2024=diag_2024, diag_2025=diag_2025, market_diag_2025=market_diag_2025, summary=summary,
    )
    print(summary["final_line"])


if __name__ == "__main__":
    main()
