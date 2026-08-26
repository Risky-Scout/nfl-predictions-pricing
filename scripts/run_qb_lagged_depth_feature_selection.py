"""QB_LAGGED_DEPTH_STATE_V1 feature selection (research task, sections 1-17).

Branch: research/qb-lagged-depth-feature-selection-2026
Certified baseline (untouched): v2026.1-fix8-certified / d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7

Writes evidence under $NFL_MODEL_ARTIFACT_ROOT/qb-lagged-depth-feature-selection-2026/.
No commits. No market features. No 2024/2025 outcome access before the
selection freeze file is written (enforced by construction: this script
never reads season>2023 outcome columns until after `write_freeze` runs).
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
from nfl_hybrid.features import qb_lagged_depth_state as qls

TASK_SLUG = "qb-lagged-depth-feature-selection-2026"
SCHEMA_VERSION = "qb-lagged-depth-feature-selection-v1"
SEED = 20260825
N_BOOTSTRAP = 10000
HORIZONS = ("TUE", "FRI")

ELO_FEATURE_COLUMNS: tuple[str, ...] = ooof.ELO_FEATURE_COLUMNS
QB_CHANGE_COLUMNS: tuple[str, ...] = (
    "home_qb_depth_changed", "away_qb_depth_changed",
    "home_qb_depth_state_missing", "away_qb_depth_state_missing",
)
QB_CONTINUITY_COLUMNS: tuple[str, ...] = ("home_qb_depth_continuity_cards", "away_qb_depth_continuity_cards")

CANDIDATE_GROUPS: dict[str, tuple[str, ...]] = {
    "ELO_ONLY": ELO_FEATURE_COLUMNS,
    "ELO_PLUS_QB_CHANGE": ELO_FEATURE_COLUMNS + QB_CHANGE_COLUMNS,
    "ELO_PLUS_QB_CHANGE_CONTINUITY": ELO_FEATURE_COLUMNS + QB_CHANGE_COLUMNS + QB_CONTINUITY_COLUMNS,
}
QB_CANDIDATES = ("ELO_PLUS_QB_CHANGE", "ELO_PLUS_QB_CHANGE_CONTINUITY")

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
COVERAGE_HARD_STOP_FLOOR = 0.80


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


def load_depth_charts_raw() -> pd.DataFrame:
    return pd.read_parquet(resolve("backfill.depth_charts"))


def build_states(games: pd.DataFrame, depth_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (card_table, historical_states, live_states, chain_source)
    where chain_source is whichever source the CALLER wants used for
    feature construction (historical for 2020-2024, live for 2025+) --
    built with continuity already computed."""
    card_table = qls.build_card_table(games)
    historical = qls.normalize_historical_card_states(depth_raw, card_table)
    live = qls.normalize_live_card_states(depth_raw, card_table)
    return card_table, historical, live, None


def build_feature_matrix(
    games: pd.DataFrame, horizon: str, ledger: pd.DataFrame, card_table: pd.DataFrame, chain: pd.DataFrame
) -> pd.DataFrame:
    elo_matrix = ooof.build_official_horizon_matrix(games, horizon, ledger)
    qb_matrix = qls.build_lagged_depth_feature_matrix(games, horizon, card_table, chain, membership_ledger=ledger)
    merged = elo_matrix.merge(qb_matrix, on="game_id", how="inner", validate="one_to_one")
    if len(merged) != len(elo_matrix):
        raise HardGateFailure("QB lagged-depth matrix did not cover every ELO-eligible target (population mismatch).")
    return merged


# ---------------------------------------------------------------------------
# Section 1 + 4: semantics + source-bridge proof
# ---------------------------------------------------------------------------
def write_semantics_and_bridge(card_table: pd.DataFrame, historical: pd.DataFrame, live: pd.DataFrame) -> dict:
    semantics = qls.qb_lagged_depth_state_v1_semantics()
    semantics_hash = qls.compute_semantics_hash(semantics)
    _write_json(_artifact_dir() / "qb_lagged_depth_state_v1_semantics.json", semantics)

    proof = qls.assert_card_schema_bridge(historical, live)
    proof["schema_version"] = SCHEMA_VERSION
    proof["required_columns"] = list(qls.NORMALIZED_CARD_STATE_COLUMNS)
    _write_json(_artifact_dir() / "qb_lagged_depth_source_bridge.json", proof)
    return {"semantics_hash": semantics_hash, "bridge": proof}


# ---------------------------------------------------------------------------
# Section 6: structural coverage gate
# ---------------------------------------------------------------------------
def run_coverage_gate(games_2023: pd.DataFrame, card_table: pd.DataFrame, chain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        cov = qls.compute_structural_coverage(games_2023, horizon, card_table, chain)
        rows.append(cov[cov["season"].between(2020, 2023)])
    coverage = pd.concat(rows, ignore_index=True).sort_values(["horizon", "season"]).reset_index(drop=True)
    coverage.to_csv(_artifact_dir() / "qb_lagged_depth_structural_coverage.csv", index=False)

    validation_seasons = {f.validate_season for f in SELECTION_FOLDS}
    failures = coverage[(coverage["season"].isin(validation_seasons)) & (coverage["coverage"] < COVERAGE_HARD_STOP_FLOOR)]
    if len(failures):
        raise HardGateFailure(
            f"Structural coverage hard-stop: validation season(s) below {COVERAGE_HARD_STOP_FLOOR:.0%} floor:\n"
            f"{failures.to_string(index=False)}"
        )
    return coverage


# ---------------------------------------------------------------------------
# Section 7: preregistration
# ---------------------------------------------------------------------------
def write_preregistration(semantics_hash: str, bridge: dict, coverage: pd.DataFrame) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "QB_LAGGED_DEPTH_FEATURE_SELECTION_2026",
        "semantics_hash": semantics_hash,
        "source_bridge_schema_equal": bridge["schema_equal"],
        "structural_coverage_floor": COVERAGE_HARD_STOP_FLOOR,
        "structural_coverage_summary": coverage.to_dict(orient="records"),
        "baseline_feature_set": {"name": "CERTIFIED_ELO_BASELINE", "columns": list(ELO_FEATURE_COLUMNS)},
        "candidate_groups": {name: list(cols) for name, cols in CANDIDATE_GROUPS.items()},
        "estimator": {
            "preprocessing": RIDGE_PREPROCESSING,
            "hyperparameters": RIDGE_HYPERPARAMETERS,
            "family": "RIDGE",
            "targets_fit_separately": True,
        },
        "folds": [{"name": f.name, "train_max_season": f.train_max_season, "validate_season": f.validate_season} for f in SELECTION_FOLDS],
        "horizons": list(HORIZONS),
        "primary_metric_formula": "primary_score = 0.5 * (margin_RMSE + total_RMSE) (Fix-7 definition, reused verbatim)",
        "bootstrap": {
            "resamples": N_BOOTSTRAP, "seed": SEED, "unit": "game_id_cluster",
            "quantities": ["primary_loss_delta", "margin_squared_error_delta", "total_squared_error_delta"],
            "ci": "95% percentile", "negative_delta_means": "QB candidate better",
        },
        "adoption_rule": {
            "conditions": [
                "pooled primary score < ELO_ONLY",
                "pooled margin RMSE <= ELO_ONLY",
                "pooled total RMSE <= ELO_ONLY",
                "95% cluster-bootstrap CI for pooled primary-loss delta vs ELO_ONLY lies entirely below zero",
                "neither TUE nor FRI pooled primary score worse than ELO_ONLY by more than 0.10 points",
                "no single validation fold primary score worse than ELO_ONLY by more than 0.15 points",
            ],
            "tie_break": "if both QB candidates pass, choose the simpler group if within one SE of the better passing candidate",
            "if_neither_passes": "retain ELO_ONLY",
        },
        "fit_budget_plan": {
            "expected_paired": len(CANDIDATE_GROUPS) * len(SELECTION_FOLDS) * len(HORIZONS),
            "expected_individual": len(CANDIDATE_GROUPS) * len(SELECTION_FOLDS) * len(HORIZONS) * 2,
            "max_paired": 24, "max_individual": 48,
        },
        "season_2024_role": "LOCKED_POST_EXPOSURE_2024_DIAGNOSTIC",
        "season_2025_role": "POST_EXPOSURE_2025_SOURCE_BRIDGE_DIAGNOSTIC",
        "no_market_features": True,
        "no_player_id_features": True,
        "selection_forbidden_seasons": [2024, 2025, 2026],
    }
    prereg_hash = _sha256_hex(payload)
    payload["qb_lagged_depth_feature_selection_preregistration_hash"] = prereg_hash
    _write_json(_artifact_dir() / "qb_lagged_depth_feature_selection_preregistration.json", payload)
    return prereg_hash


def verify_preregistration_hash(prereg_hash: str) -> None:
    path = _artifact_dir() / "qb_lagged_depth_feature_selection_preregistration.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = on_disk.pop("qb_lagged_depth_feature_selection_preregistration_hash")
    recomputed = _sha256_hex(on_disk)
    if recomputed != stored_hash or recomputed != prereg_hash:
        raise HardGateFailure("Preregistration hash verification failed at fit entry point.")


# ---------------------------------------------------------------------------
# Ridge fitting (Section 8/14)
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


def run_selection_fits(
    games_2023: pd.DataFrame, ledger_by_horizon: dict[str, pd.DataFrame], card_table: pd.DataFrame, chain: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    matrices = {h: build_feature_matrix(games_2023, h, ledger_by_horizon[h], card_table, chain) for h in HORIZONS}

    metric_rows: list[dict] = []
    pooled_preds: dict[tuple[str, str], pd.DataFrame] = {}  # (candidate, horizon) -> concat over folds
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
    metrics_df.to_csv(_artifact_dir() / "qb_lagged_depth_selection_metrics.csv", index=False)

    if FIT_COUNTER["paired"] > 24 or FIT_COUNTER["individual"] > 48:
        raise HardGateFailure(f"Fit budget ceiling exceeded: {FIT_COUNTER}")
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
    """Game_id-cluster bootstrap of mean(delta) = mean(a - b) over ROWS,
    resampling unique game_id clusters with replacement (Section 12)."""
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
        "observed_mean_delta": observed_mean,
        "ci_low": float(lo), "ci_high": float(hi),
        "ci_entirely_below_zero": bool(hi < 0.0),
        "n_games": int(n_games), "n_rows": int(len(paired)),
    }


# ---------------------------------------------------------------------------
# Section 12/13: bootstrap + adoption gate
# ---------------------------------------------------------------------------
def run_bootstrap_and_adoption(metrics_df: pd.DataFrame, pooled_preds: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    elo_pooled = pooled_frame(pooled_preds, "ELO_ONLY")

    bootstrap_rows = []
    adoption: dict[str, dict] = {}
    for candidate in QB_CANDIDATES:
        cand_pooled = pooled_frame(pooled_preds, candidate)
        paired = paired_losses(cand_pooled, elo_pooled)  # a=candidate, b=ELO_ONLY

        primary_loss = bootstrap_cluster_delta(paired, value_col_a="l_a", value_col_b="l_b", rng=rng)
        margin_se = bootstrap_cluster_delta(paired, value_col_a="margin_se_a", value_col_b="margin_se_b", rng=rng)
        total_se = bootstrap_cluster_delta(paired, value_col_a="total_se_a", value_col_b="total_se_b", rng=rng)
        for quantity, result in (("primary_loss_delta", primary_loss), ("margin_squared_error_delta", margin_se), ("total_squared_error_delta", total_se)):
            bootstrap_rows.append({"candidate": candidate, "quantity": quantity, **result})

        # Pooled regression report (all folds x horizons).
        pooled_report_cand = regression_report(cand_pooled)
        pooled_report_elo = regression_report(elo_pooled)

        cond1 = pooled_report_cand["primary_score"] < pooled_report_elo["primary_score"]
        cond2 = pooled_report_cand["margin"]["rmse"] <= pooled_report_elo["margin"]["rmse"]
        cond3 = pooled_report_cand["total"]["rmse"] <= pooled_report_elo["total"]["rmse"]
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
            cand_fold = metrics_df[(metrics_df["candidate"] == candidate) & (metrics_df["fold"] == fold.name)]
            elo_fold = metrics_df[(metrics_df["candidate"] == "ELO_ONLY") & (metrics_df["fold"] == fold.name)]
            cand_fold_pooled_rows = pd.concat(
                [pooled_preds[(candidate, h)][pooled_preds[(candidate, h)]["fold"] == fold.name] for h in HORIZONS], ignore_index=True
            )
            elo_fold_pooled_rows = pd.concat(
                [pooled_preds[("ELO_ONLY", h)][pooled_preds[("ELO_ONLY", h)]["fold"] == fold.name] for h in HORIZONS], ignore_index=True
            )
            diff = regression_report(cand_fold_pooled_rows)["primary_score"] - regression_report(elo_fold_pooled_rows)["primary_score"]
            fold_diffs[fold.name] = diff
            if diff > 0.15:
                cond6 = False

        conditions = {
            "pooled_primary_score_lt_elo": cond1, "pooled_margin_rmse_le_elo": cond2,
            "pooled_total_rmse_le_elo": cond3, "bootstrap_ci_entirely_below_zero": cond4,
            "no_horizon_worse_by_gt_0_10": cond5, "no_fold_worse_by_gt_0_15": cond6,
        }
        adoption[candidate] = {
            "pooled_candidate_primary_score": pooled_report_cand["primary_score"],
            "pooled_elo_primary_score": pooled_report_elo["primary_score"],
            "pooled_candidate_margin_rmse": pooled_report_cand["margin"]["rmse"],
            "pooled_elo_margin_rmse": pooled_report_elo["margin"]["rmse"],
            "pooled_candidate_total_rmse": pooled_report_cand["total"]["rmse"],
            "pooled_elo_total_rmse": pooled_report_elo["total"]["rmse"],
            "horizon_primary_score_diff": horizon_diffs,
            "fold_primary_score_diff": fold_diffs,
            "conditions": conditions,
            "adoptable": all(conditions.values()),
        }

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(_artifact_dir() / "qb_lagged_depth_bootstrap.csv", index=False)
    return bootstrap_df, adoption


def select_winner(adoption: dict, pooled_preds: dict[tuple[str, str], pd.DataFrame]) -> dict:
    passing = [c for c in QB_CANDIDATES if adoption[c]["adoptable"]]
    if not passing:
        return {"selected_candidate": "ELO_ONLY", "reason": "NEITHER_QB_CANDIDATE_PASSED_ADOPTION_GATE"}
    if len(passing) == 1:
        return {"selected_candidate": passing[0], "reason": "SOLE_PASSING_CANDIDATE"}

    # Both passed -- pick the simpler (fewer features) if within one SE of the better.
    simpler, richer = "ELO_PLUS_QB_CHANGE", "ELO_PLUS_QB_CHANGE_CONTINUITY"
    scores = {c: adoption[c]["pooled_candidate_primary_score"] for c in passing}
    better, worse = (simpler, richer) if scores[simpler] <= scores[richer] else (richer, simpler)
    if better == simpler:
        return {"selected_candidate": simpler, "reason": "SIMPLER_CANDIDATE_ALREADY_BETTER_OR_EQUAL"}

    # richer is the better one; check whether simpler is within one SE of richer via paired bootstrap-free analytic SE
    simpler_pooled = pooled_frame(pooled_preds, simpler)
    richer_pooled = pooled_frame(pooled_preds, richer)
    paired = paired_losses(simpler_pooled, richer_pooled)  # a=simpler, b=richer
    delta = (paired["l_a"] - paired["l_b"]).to_numpy(float)
    n = len(delta)
    se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    within_one_se = abs(scores[simpler] - scores[richer]) <= se
    if within_one_se:
        return {"selected_candidate": simpler, "reason": "BOTH_PASSED_SIMPLER_WITHIN_ONE_SE_OF_BETTER", "se": se}
    return {"selected_candidate": richer, "reason": "BOTH_PASSED_RICHER_BEATS_SIMPLER_BY_MORE_THAN_ONE_SE", "se": se}


# ---------------------------------------------------------------------------
# Section 15: freeze
# ---------------------------------------------------------------------------
def write_freeze(
    semantics_hash: str, prereg_hash: str, metrics_df: pd.DataFrame, bootstrap_df: pd.DataFrame,
    adoption: dict, winner: dict,
) -> str:
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
    payload["qb_lagged_depth_feature_selection_freeze_hash"] = freeze_hash
    _write_json(_artifact_dir() / "qb_lagged_depth_feature_selection_freeze.json", payload)
    return freeze_hash


# ---------------------------------------------------------------------------
# Section 16: 2024 diagnostic (LOCKED_POST_EXPOSURE, no selection authority)
# ---------------------------------------------------------------------------
def run_2024_diagnostic(games_all: pd.DataFrame, depth_raw: pd.DataFrame, winner_candidate: str) -> pd.DataFrame:
    games_2024 = games_all[games_all["season"] <= 2024].copy()
    card_table = qls.build_card_table(games_2024)
    historical = qls.normalize_historical_card_states(depth_raw, card_table)
    chain = qls.compute_card_chain_continuity(historical)
    ledger = he.build_horizon_membership_ledger(games_2024)

    rows = []
    for candidate_name in ("ELO_ONLY", winner_candidate) if winner_candidate != "ELO_ONLY" else ("ELO_ONLY",):
        feature_columns = CANDIDATE_GROUPS[candidate_name]
        for horizon in HORIZONS:
            matrix = build_feature_matrix(games_2024, horizon, ledger, card_table, chain)
            fold = FoldSpec("OUTER_2024", 2023, 2024)
            preds = fit_predict_fold(matrix, feature_columns, fold)
            report = regression_report(preds)
            rows.append(
                {
                    "label": "LOCKED_POST_EXPOSURE_2024_DIAGNOSTIC", "candidate": candidate_name, "horizon": horizon,
                    "margin_rmse": report["margin"]["rmse"], "margin_mae": report["margin"]["mae"],
                    "total_rmse": report["total"]["rmse"], "total_mae": report["total"]["mae"],
                    "primary_score": report["primary_score"], "n": len(preds),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(_artifact_dir() / "qb_lagged_depth_2024_diagnostic.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Section 17: 2025 source-bridge diagnostic (live source, no selection authority)
# ---------------------------------------------------------------------------
def run_2025_bridge_diagnostic(games_all: pd.DataFrame, depth_raw: pd.DataFrame, winner_candidate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    games_2025 = games_all[games_all["season"] <= 2025].copy()
    card_table = qls.build_card_table(games_2025)
    live = qls.normalize_live_card_states(depth_raw, card_table)
    chain = qls.compute_card_chain_continuity(live)
    ledger = he.build_horizon_membership_ledger(games_2025)

    rows = []
    for candidate_name in ("ELO_ONLY", winner_candidate) if winner_candidate != "ELO_ONLY" else ("ELO_ONLY",):
        feature_columns = CANDIDATE_GROUPS[candidate_name]
        for horizon in HORIZONS:
            matrix = build_feature_matrix(games_2025, horizon, ledger, card_table, chain)
            fold = FoldSpec("BRIDGE_2025", 2024, 2025)
            preds = fit_predict_fold(matrix, feature_columns, fold)
            report = regression_report(preds)
            rows.append(
                {
                    "label": "POST_EXPOSURE_2025_SOURCE_BRIDGE_DIAGNOSTIC", "candidate": candidate_name, "horizon": horizon,
                    "margin_rmse": report["margin"]["rmse"], "margin_mae": report["margin"]["mae"],
                    "total_rmse": report["total"]["rmse"], "total_mae": report["total"]["mae"],
                    "primary_score": report["primary_score"], "n": len(preds),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(_artifact_dir() / "qb_lagged_depth_2025_bridge_diagnostic.csv", index=False)

    # Structural distribution shift: historical (2020-2024) vs live-normalized (2025).
    games_2024 = games_all[games_all["season"] <= 2024].copy()
    hist_card_table = qls.build_card_table(games_2024)
    hist_states = qls.normalize_historical_card_states(depth_raw, hist_card_table)
    hist_chain = qls.compute_card_chain_continuity(hist_states)
    hist_ledger = he.build_horizon_membership_ledger(games_2024)

    shift_rows = []
    for horizon in HORIZONS:
        hist_resolved = qls.resolve_lagged_target_state(games_2024, horizon, hist_card_table, hist_chain, membership_ledger=hist_ledger)
        live_resolved = qls.resolve_lagged_target_state(games_2025, horizon, card_table, chain, membership_ledger=ledger)
        live_resolved = live_resolved[live_resolved["season"] == 2025]
        for feature in ("qb_depth_changed", "qb_depth_continuity_cards", "qb_depth_state_missing"):
            h_vals = hist_resolved[feature].to_numpy(float)
            l_vals = live_resolved[feature].to_numpy(float)
            hist_std = float(np.std(h_vals, ddof=1)) if len(h_vals) > 1 else np.nan
            live_std = float(np.std(l_vals, ddof=1)) if len(l_vals) > 1 else np.nan
            pooled_std = float(np.sqrt(0.5 * (hist_std ** 2 + live_std ** 2))) if hist_std and live_std else np.nan
            mean_diff = (float(np.mean(l_vals)) if len(l_vals) else np.nan) - (float(np.mean(h_vals)) if len(h_vals) else np.nan)
            effect_size = mean_diff / pooled_std if pooled_std else np.nan
            shift_rows.append(
                {
                    "horizon": horizon, "feature": feature,
                    "historical_n": len(h_vals), "historical_mean": float(np.mean(h_vals)) if len(h_vals) else np.nan,
                    "historical_std": hist_std,
                    "live_2025_n": len(l_vals), "live_2025_mean": float(np.mean(l_vals)) if len(l_vals) else np.nan,
                    "live_2025_std": live_std,
                    "standardized_mean_difference": effect_size,
                }
            )
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(_artifact_dir() / "qb_lagged_depth_feature_distribution_shift.csv", index=False)
    return out, shift_df


# ---------------------------------------------------------------------------
# Section 20/21: summary + report + final questions
# ---------------------------------------------------------------------------
MATERIAL_SHIFT_EFFECT_SIZE_THRESHOLD = 0.2  # conventional Cohen's-d "small effect" boundary


def _material_shift(shift_df: pd.DataFrame) -> bool:
    """Descriptive materiality flag: a standardized mean difference (raw
    mean difference / pooled std) larger than the conventional 'small
    effect' boundary on any of the three structural features. Standardized
    rather than a raw-mean threshold because qb_depth_continuity_cards is a
    count (0-17ish) on a completely different scale from the two [0,1] rate
    features -- a single raw-difference cutoff would trivially flag the
    count feature as 'material' for even a tiny relative shift. Reported
    descriptively; never used to change the frozen candidate."""
    return bool((shift_df["standardized_mean_difference"].abs() > MATERIAL_SHIFT_EFFECT_SIZE_THRESHOLD).any())


def write_summary_and_answers(
    *, coverage: pd.DataFrame, adoption: dict, winner: dict, diag_2024: pd.DataFrame,
    diag_2025: pd.DataFrame, shift_df: pd.DataFrame, semantics_hash: str, prereg_hash: str, freeze_hash: str,
) -> dict:
    q1_min_validation_coverage = coverage[coverage["season"].isin({f.validate_season for f in SELECTION_FOLDS})]["coverage"].min()
    q1 = (
        f"YES -- every validation season (2021/2022/2023) cleared the 80% structural-coverage floor on both "
        f"TUE and FRI (minimum observed: {q1_min_validation_coverage:.1%}). See qb_lagged_depth_structural_coverage.csv "
        "for the full 2020-2023 by-season/horizon table; 2020 itself (train-only, never a validation season) "
        "came in lower on TUE because of real 2020 COVID-rescheduled games (BUF@TEN and DAL@BAL moved to "
        "Tuesday, BAL@PIT to Wednesday) pushing that week's league-wide card_end_utc past the following week's "
        "TUE cutoff for every team -- a genuine structural effect of the frozen, conservative "
        "max-kickoff-in-card availability rule, not a defect."
    )

    q2 = f"{'YES' if adoption['ELO_PLUS_QB_CHANGE']['adoptable'] else 'NO'} -- see adoption_gate_results.ELO_PLUS_QB_CHANGE in the freeze file for the per-condition breakdown."
    q3 = f"{'YES' if adoption['ELO_PLUS_QB_CHANGE_CONTINUITY']['adoptable'] else 'NO'} -- see adoption_gate_results.ELO_PLUS_QB_CHANGE_CONTINUITY in the freeze file."

    q4 = f"{winner['selected_candidate']} was frozen on <=2023 evidence ({winner['reason']})."

    diag_pivot = diag_2024.pivot(index="horizon", columns="candidate", values="primary_score")
    q5_lines = [f"{h}: ELO_ONLY={diag_pivot.loc[h, 'ELO_ONLY']:.4f}" for h in HORIZONS]
    if winner["selected_candidate"] != "ELO_ONLY":
        for h in HORIZONS:
            q5_lines[HORIZONS.index(h)] += f", {winner['selected_candidate']}={diag_pivot.loc[h, winner['selected_candidate']]:.4f}"
    q5 = "LOCKED_POST_EXPOSURE_2024_DIAGNOSTIC (no selection authority): " + "; ".join(q5_lines)

    diag25_pivot = diag_2025.pivot(index="horizon", columns="candidate", values="primary_score")
    q6_lines = [f"{h}: ELO_ONLY={diag25_pivot.loc[h, 'ELO_ONLY']:.4f}" for h in HORIZONS]
    if winner["selected_candidate"] != "ELO_ONLY":
        for h in HORIZONS:
            q6_lines[HORIZONS.index(h)] += f", {winner['selected_candidate']}={diag25_pivot.loc[h, winner['selected_candidate']]:.4f}"
    q6 = "POST_EXPOSURE_2025_SOURCE_BRIDGE_DIAGNOSTIC (no selection authority): " + "; ".join(q6_lines)

    material = _material_shift(shift_df)
    max_effect = shift_df["standardized_mean_difference"].abs().max()
    q7 = (
        f"{'YES' if material else 'NO'} material shift (standardized mean difference, |Cohen's d|, exceeding "
        f"{MATERIAL_SHIFT_EFFECT_SIZE_THRESHOLD} on any structural feature; observed max |d|={max_effect:.3f}) -- "
        "see qb_lagged_depth_feature_distribution_shift.csv."
    )

    recertification_justified = winner["selected_candidate"] != "ELO_ONLY"
    q8 = f"{'YES' if recertification_justified else 'NO'} -- {'a QB candidate passed every frozen adoption gate on <=2023 evidence' if recertification_justified else 'no QB candidate passed every frozen adoption gate; the certified Elo baseline is retained'}."

    final_line = (
        "QB LAGGED DEPTH FEATURE SELECTION COMPLETE — FORMAL RECERTIFICATION JUSTIFIED"
        if recertification_justified
        else "QB LAGGED DEPTH FEATURE SELECTION COMPLETE — KEEP CERTIFIED ELO BASELINE"
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "semantics_hash": semantics_hash,
        "preregistration_hash": prereg_hash,
        "freeze_hash": freeze_hash,
        "selected_candidate": winner["selected_candidate"],
        "selection_reason": winner["reason"],
        "answers": {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6, "Q7": q7, "Q8": q8},
        "final_line": final_line,
    }
    _write_json(_artifact_dir() / "qb_lagged_depth_feature_selection_summary.json", summary)
    return summary


def write_report_md(
    *, coverage: pd.DataFrame, metrics_df: pd.DataFrame, bootstrap_df: pd.DataFrame, adoption: dict,
    winner: dict, diag_2024: pd.DataFrame, diag_2025: pd.DataFrame, shift_df: pd.DataFrame, summary: dict,
) -> None:
    lines = [
        "# QB Lagged Depth State V1 -- Feature Selection Report",
        "",
        "**Branch:** `research/qb-lagged-depth-feature-selection-2026`",
        "**Certified baseline (untouched):** `v2026.1-fix8-certified` / `d24e4ce57294dc6572cbb17a4ee6c24e08c80ad7`",
        "",
        "## Structural coverage (2020-2023, no outcomes)",
        "",
        coverage.to_string(index=False),
        "",
        "## Selection metrics (folds A/B/C, <=2023)",
        "",
        metrics_df.to_string(index=False),
        "",
        "## Bootstrap (10,000 resamples, seed 20260825, game_id cluster)",
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
        "## 2025 source-bridge diagnostic (POST_EXPOSURE_2025_SOURCE_BRIDGE_DIAGNOSTIC, no selection authority)",
        "",
        diag_2025.to_string(index=False),
        "",
        "## Historical vs live-normalized feature distribution shift",
        "",
        shift_df.to_string(index=False),
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
    (_artifact_dir() / "QB_LAGGED_DEPTH_FEATURE_SELECTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    games_all = load_games_all()
    depth_raw = load_depth_charts_raw()

    games_2023 = games_all[games_all["season"] <= 2023].copy()
    card_table, historical, live, _ = build_states(games_2023, depth_raw)
    chain = qls.compute_card_chain_continuity(historical)

    bridge_info = write_semantics_and_bridge(card_table, historical, live)
    coverage = run_coverage_gate(games_2023, card_table, chain)

    ledger = he.build_horizon_membership_ledger(games_2023)
    ledger_by_horizon = {h: ledger for h in HORIZONS}

    prereg_hash = write_preregistration(bridge_info["semantics_hash"], bridge_info["bridge"], coverage)
    verify_preregistration_hash(prereg_hash)

    metrics_df, pooled_preds = run_selection_fits(games_2023, ledger_by_horizon, card_table, chain)
    bootstrap_df, adoption = run_bootstrap_and_adoption(metrics_df, pooled_preds)
    winner = select_winner(adoption, pooled_preds)
    freeze_hash = write_freeze(bridge_info["semantics_hash"], prereg_hash, metrics_df, bootstrap_df, adoption, winner)

    print(f"Fit counts after selection freeze: {FIT_COUNTER}")
    print(f"Selected candidate: {winner['selected_candidate']} ({winner['reason']})")

    # ONLY after the freeze file above is written may 2024/2025 be accessed.
    diag_2024 = run_2024_diagnostic(games_all, depth_raw, winner["selected_candidate"])
    diag_2025, shift_df = run_2025_bridge_diagnostic(games_all, depth_raw, winner["selected_candidate"])

    summary = write_summary_and_answers(
        coverage=coverage, adoption=adoption, winner=winner, diag_2024=diag_2024, diag_2025=diag_2025,
        shift_df=shift_df, semantics_hash=bridge_info["semantics_hash"], prereg_hash=prereg_hash, freeze_hash=freeze_hash,
    )
    write_report_md(
        coverage=coverage, metrics_df=metrics_df, bootstrap_df=bootstrap_df, adoption=adoption, winner=winner,
        diag_2024=diag_2024, diag_2025=diag_2025, shift_df=shift_df, summary=summary,
    )
    print(summary["final_line"])


if __name__ == "__main__":
    main()
