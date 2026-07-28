from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression


_CONFIG_COLUMNS = [
    "market",
    "variant",
    "architecture",
    "feature_set",
    "model_name",
]


@dataclass(frozen=True)
class CalibrationConfig:
    epsilon: float = 1e-6
    push_prior_strength: float = 40.0
    conditional_c: float = 0.20
    bootstrap_repetitions: int = 2000
    random_seed: int = 42


def _push_bucket(market: str, line: float) -> str:
    if market == "pregame_moneyline":
        return "GLOBAL"
    if not np.isfinite(line):
        return "MISSING"
    if not np.isclose(line, round(line), atol=1e-9):
        return "NO_PUSH"
    if market == "pregame_ats":
        return f"INT_{int(round(abs(line)))}"
    return f"INT_{int(round(line))}"


def _fit_conditional_calibrator(
    probability: np.ndarray,
    target: np.ndarray,
    config: CalibrationConfig,
) -> LogisticRegression | None:
    p = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=int)
    valid = np.isfinite(p) & np.isin(y, [0, 1])
    p = p[valid]
    y = y[valid]
    if len(p) < 100 or len(np.unique(y)) < 2:
        return None
    x = logit(np.clip(p, config.epsilon, 1.0 - config.epsilon)).reshape(-1, 1)
    model = LogisticRegression(
        C=config.conditional_c,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=3000,
        random_state=config.random_seed,
    )
    model.fit(x, y)
    return model


def _predict_conditional(
    model: LogisticRegression | None,
    probability: np.ndarray,
    config: CalibrationConfig,
) -> np.ndarray:
    p = np.clip(
        np.asarray(probability, dtype=float),
        config.epsilon,
        1.0 - config.epsilon,
    )
    if model is None:
        return p
    x = logit(p).reshape(-1, 1)
    return np.clip(
        model.predict_proba(x)[:, 1],
        config.epsilon,
        1.0 - config.epsilon,
    )


def _fit_push_scales(
    train: pd.DataFrame,
    market: str,
    config: CalibrationConfig,
) -> tuple[float, dict[str, float]]:
    raw = np.clip(
        pd.to_numeric(
            train["model_push_probability"], errors="coerce"
        ).fillna(0.0).to_numpy(float),
        0.0,
        1.0,
    )
    actual = pd.to_numeric(
        train["actual_push"], errors="coerce"
    ).fillna(0).to_numpy(float)
    lines = pd.to_numeric(
        train["market_line"], errors="coerce"
    ).to_numpy(float)

    global_rate = (float(actual.sum()) + 0.5) / (len(actual) + 1.0)
    global_mean_raw = max(float(raw.mean()), config.epsilon)
    global_scale = global_rate / global_mean_raw

    buckets = np.array(
        [_push_bucket(market, value) for value in lines],
        dtype=object,
    )
    scales: dict[str, float] = {}
    for bucket in sorted(set(buckets)):
        if bucket == "NO_PUSH":
            scales[bucket] = 0.0
            continue
        mask = buckets == bucket
        n = int(mask.sum())
        events = float(actual[mask].sum())
        shrunk_rate = (
            events + config.push_prior_strength * global_rate
        ) / (n + config.push_prior_strength)
        mean_raw = max(float(raw[mask].mean()), config.epsilon)
        scales[bucket] = shrunk_rate / mean_raw
    return global_scale, scales


def _predict_push(
    frame: pd.DataFrame,
    market: str,
    global_scale: float,
    bucket_scales: dict[str, float],
    config: CalibrationConfig,
) -> np.ndarray:
    raw = np.clip(
        pd.to_numeric(
            frame["model_push_probability"], errors="coerce"
        ).fillna(0.0).to_numpy(float),
        0.0,
        1.0,
    )
    lines = pd.to_numeric(
        frame["market_line"], errors="coerce"
    ).to_numpy(float)
    output = np.zeros(len(frame), dtype=float)
    for index, (probability, line) in enumerate(zip(raw, lines)):
        bucket = _push_bucket(market, line)
        if bucket == "NO_PUSH":
            output[index] = 0.0
            continue
        scale = bucket_scales.get(bucket, global_scale)
        output[index] = probability * scale
    return np.clip(output, 0.0, 0.25)


def _recombine(
    conditional_upper: np.ndarray,
    push_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conditional = np.clip(np.asarray(conditional_upper, dtype=float), 0.0, 1.0)
    push = np.clip(np.asarray(push_probability, dtype=float), 0.0, 1.0)
    lower = (1.0 - push) * (1.0 - conditional)
    upper = (1.0 - push) * conditional
    matrix = np.column_stack([lower, push, upper])
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix[:, 0], matrix[:, 1], matrix[:, 2]


def apply_expanding_three_way_calibration(
    selected_oof: pd.DataFrame,
    config: CalibrationConfig | None = None,
) -> pd.DataFrame:
    cfg = config or CalibrationConfig()
    required = {
        "market",
        "variant",
        "season",
        "actual_category",
        "binary_target",
        "scored_binary",
        "market_line",
        "model_push_probability",
        "model_conditional_upper_probability",
    }
    missing = sorted(required - set(selected_oof.columns))
    if missing:
        raise ValueError(f"OOF predictions missing columns: {missing}")

    frame = selected_oof.sort_values(
        ["market", "variant", "season", "game_id"],
        kind="stable",
    ).copy()
    outputs: list[pd.DataFrame] = []

    for (market, variant), group in frame.groupby(
        ["market", "variant"],
        sort=True,
    ):
        selection_status = str(
            group["selection_status"].iloc[0]
            if "selection_status" in group.columns
            else ""
        )
        retain_baseline = selection_status.startswith("RETAIN_BASELINE")
        seasons = sorted(int(value) for value in group["season"].unique())
        for season in seasons:
            test = group[group["season"] == season].copy()
            train = group[group["season"] < season].copy()

            if retain_baseline:
                conditional = test[
                    "model_conditional_upper_probability"
                ].to_numpy(float)
                push = test["model_push_probability"].to_numpy(float)
                method = "baseline_retained_no_calibration"
            elif train.empty:
                conditional = test[
                    "model_conditional_upper_probability"
                ].to_numpy(float)
                push = test["model_push_probability"].to_numpy(float)
                method = "identity_no_prior_oof"
            else:
                binary_train = train[train["scored_binary"].astype(bool)]
                calibrator = _fit_conditional_calibrator(
                    binary_train[
                        "model_conditional_upper_probability"
                    ].to_numpy(float),
                    binary_train["binary_target"].to_numpy(int),
                    cfg,
                )
                conditional = _predict_conditional(
                    calibrator,
                    test[
                        "model_conditional_upper_probability"
                    ].to_numpy(float),
                    cfg,
                )
                global_scale, bucket_scales = _fit_push_scales(
                    train,
                    market,
                    cfg,
                )
                push = _predict_push(
                    test,
                    market,
                    global_scale,
                    bucket_scales,
                    cfg,
                )
                method = "expanding_prior_seasons_only"

            lower, push, upper = _recombine(conditional, push)
            test["calibrated_lower_probability"] = lower
            test["calibrated_push_probability"] = push
            test["calibrated_upper_probability"] = upper
            test["calibrated_conditional_upper_probability"] = np.divide(
                upper,
                lower + upper,
                out=np.full(len(test), 0.5),
                where=(lower + upper) > 0,
            )
            test["calibration_method"] = method
            outputs.append(test)

    return pd.concat(outputs, ignore_index=True)


def _score_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    epsilon = 1e-6
    actual = output["actual_category"].to_numpy(int)
    before = output[
        [
            "model_lower_probability",
            "model_push_probability",
            "model_upper_probability",
        ]
    ].to_numpy(float)
    after = output[
        [
            "calibrated_lower_probability",
            "calibrated_push_probability",
            "calibrated_upper_probability",
        ]
    ].to_numpy(float)
    before_true = np.clip(before[np.arange(len(output)), actual], epsilon, 1.0)
    after_true = np.clip(after[np.arange(len(output)), actual], epsilon, 1.0)
    one_hot = np.eye(3)[actual]

    output["before_three_way_log_loss_row"] = -np.log(before_true)
    output["after_three_way_log_loss_row"] = -np.log(after_true)
    output["before_three_way_brier_row"] = np.sum(
        (before - one_hot) ** 2,
        axis=1,
    )
    output["after_three_way_brier_row"] = np.sum(
        (after - one_hot) ** 2,
        axis=1,
    )

    scored = output["scored_binary"].astype(bool).to_numpy()
    y = output["binary_target"].to_numpy(float)
    p_before = np.clip(
        output["model_conditional_upper_probability"].to_numpy(float),
        epsilon,
        1.0 - epsilon,
    )
    p_after = np.clip(
        output["calibrated_conditional_upper_probability"].to_numpy(float),
        epsilon,
        1.0 - epsilon,
    )
    output["before_binary_log_loss_row"] = np.nan
    output["after_binary_log_loss_row"] = np.nan
    output["before_binary_brier_row"] = np.nan
    output["after_binary_brier_row"] = np.nan
    output.loc[scored, "before_binary_log_loss_row"] = -(
        y[scored] * np.log(p_before[scored])
        + (1.0 - y[scored]) * np.log(1.0 - p_before[scored])
    )
    output.loc[scored, "after_binary_log_loss_row"] = -(
        y[scored] * np.log(p_after[scored])
        + (1.0 - y[scored]) * np.log(1.0 - p_after[scored])
    )
    output.loc[scored, "before_binary_brier_row"] = (
        p_before[scored] - y[scored]
    ) ** 2
    output.loc[scored, "after_binary_brier_row"] = (
        p_after[scored] - y[scored]
    ) ** 2
    return output


def _metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in scored.groupby(["market", "variant"], sort=True):
        binary = frame[frame["scored_binary"].astype(bool)]
        rows.append(
            {
                "market": keys[0],
                "variant": keys[1],
                "rows": len(frame),
                "binary_rows": len(binary),
                "before_binary_log_loss": binary[
                    "before_binary_log_loss_row"
                ].mean(),
                "after_binary_log_loss": binary[
                    "after_binary_log_loss_row"
                ].mean(),
                "binary_log_loss_gain": binary[
                    "before_binary_log_loss_row"
                ].mean()
                - binary["after_binary_log_loss_row"].mean(),
                "before_binary_brier": binary[
                    "before_binary_brier_row"
                ].mean(),
                "after_binary_brier": binary[
                    "after_binary_brier_row"
                ].mean(),
                "binary_brier_gain": binary[
                    "before_binary_brier_row"
                ].mean()
                - binary["after_binary_brier_row"].mean(),
                "before_three_way_log_loss": frame[
                    "before_three_way_log_loss_row"
                ].mean(),
                "after_three_way_log_loss": frame[
                    "after_three_way_log_loss_row"
                ].mean(),
                "three_way_log_loss_gain": frame[
                    "before_three_way_log_loss_row"
                ].mean()
                - frame["after_three_way_log_loss_row"].mean(),
                "before_three_way_brier": frame[
                    "before_three_way_brier_row"
                ].mean(),
                "after_three_way_brier": frame[
                    "after_three_way_brier_row"
                ].mean(),
                "three_way_brier_gain": frame[
                    "before_three_way_brier_row"
                ].mean()
                - frame["after_three_way_brier_row"].mean(),
                "before_mean_push": frame[
                    "model_push_probability"
                ].mean(),
                "after_mean_push": frame[
                    "calibrated_push_probability"
                ].mean(),
                "actual_push_rate": frame["actual_push"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    scored: pd.DataFrame,
    config: CalibrationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, object]] = []
    for keys, frame in scored.groupby(["market", "variant"], sort=True):
        clusters = sorted(frame["cluster_id"].astype(str).unique())
        if not clusters:
            continue
        gains = []
        for _ in range(config.bootstrap_repetitions):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            parts = [frame[frame["cluster_id"].astype(str) == value] for value in sampled]
            sample = pd.concat(parts, ignore_index=True)
            gains.append(
                (
                    sample["before_three_way_log_loss_row"].mean()
                    - sample["after_three_way_log_loss_row"].mean(),
                    sample["before_three_way_brier_row"].mean()
                    - sample["after_three_way_brier_row"].mean(),
                )
            )
        values = np.asarray(gains, dtype=float)
        rows.append(
            {
                "market": keys[0],
                "variant": keys[1],
                "bootstrap_method": "season_week_cluster",
                "repetitions": config.bootstrap_repetitions,
                "three_way_log_loss_gain": values[:, 0].mean(),
                "three_way_log_loss_gain_ci_lower": np.quantile(values[:, 0], 0.025),
                "three_way_log_loss_gain_ci_upper": np.quantile(values[:, 0], 0.975),
                "three_way_brier_gain": values[:, 1].mean(),
                "three_way_brier_gain_ci_lower": np.quantile(values[:, 1], 0.025),
                "three_way_brier_gain_ci_upper": np.quantile(values[:, 1], 0.975),
            }
        )
    return pd.DataFrame(rows)


def _selected_oof(
    oof: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        mask = (
            oof["market"].eq(row["market"])
            & oof["variant"].eq(row["variant"])
            & oof["architecture"].eq(row["architecture"])
            & oof["feature_set"].eq(row["feature_set"])
            & oof["model_name"].eq(row["model_name"])
        )
        subset = oof[mask].copy()
        if subset.empty:
            raise ValueError(
                "No OOF rows found for selected configuration: "
                f"{row['market']} {row['variant']} "
                f"{row['architecture']} {row['feature_set']} "
                f"{row['model_name']}"
            )
        subset["selection_status"] = row["selection_status"]
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def calibrate_selected_distributional_models(
    distributional_root: str | Path,
    output_root: str | Path,
    *,
    config: CalibrationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or CalibrationConfig()
    source = Path(distributional_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(source / "distributional_oof_predictions.parquet")
    selected = pd.read_csv(source / "distributional_selected.csv")
    selected_rows = _selected_oof(oof, selected)
    calibrated = apply_expanding_three_way_calibration(selected_rows, cfg)
    scored = _score_rows(calibrated)
    metrics = _metrics(scored)
    bootstrap = _bootstrap(scored, cfg)

    scored.to_parquet(
        output / "calibrated_distributional_oof_predictions.parquet",
        index=False,
    )
    metrics.to_csv(output / "three_way_calibration_metrics.csv", index=False)
    bootstrap.to_csv(
        output / "three_way_calibration_bootstrap.csv",
        index=False,
    )
    return scored, metrics, bootstrap
