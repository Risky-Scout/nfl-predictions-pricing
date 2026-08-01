"""Deterministic, leakage-safe scoring primitives for the Week 1 shadow
reliability evaluation (Stage 3).

Pure functions only: metrics, adaptive equal-mass ECE, IRLS calibration
slope/intercept, season-stratified paired bootstrap, market de-vigging, fold
construction, reliability classification, and deterministic ledger hashing.

Nothing here fits, tunes, or mutates a production model or artifact. The
functions are used both by ``scripts/evaluate_week1_shadow_reliability.py`` (full
local run) and by ``tests/test_week1_shadow_reliability.py`` (CI, synthetic
fixture). See the pre-registration:
``docs/model-selection/week1-shadow-reliability-2026/registry.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_hybrid.pricing.devig import devig_proportional

# ---------------------------------------------------------------------------
# Frozen constants (mirror the pre-registration; do not edit after results).
# ---------------------------------------------------------------------------
NOT_ESTIMABLE = "NOT_ESTIMABLE"
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_REPS = 5000
MIN_CONCLUSIVE_N = 100
NON_INFERIORITY_LOGLOSS_MARGIN = 0.02
ECE_THRESHOLD = 0.05
SLOPE_RANGE = (0.75, 1.25)
INTERCEPT_RANGE = (-0.10, 0.10)
_EPS = 1e-6

# Rolling-origin folds (test season -> train cutoff / calibration season).
FOLDS = (
    {"test_season": 2022, "train_max_season": 2020, "calibration_season": 2021},
    {"test_season": 2023, "train_max_season": 2021, "calibration_season": 2022},
    {"test_season": 2024, "train_max_season": 2022, "calibration_season": 2023},
    {"test_season": 2025, "train_max_season": 2023, "calibration_season": 2024},
)

# shadow output column -> (outcome column, push/tie meaning) per market
MARKET_SHADOW_COL = {
    "moneyline": "home_win_probability_no_tie",
    "spread": "home_cover_probability_no_push",
    "total": "over_probability_no_push",
}


# ---------------------------------------------------------------------------
# Odds / market helpers
# ---------------------------------------------------------------------------
def american_to_implied(american: float) -> float:
    """Implied (vigged) probability from an American price."""
    a = float(american)
    if not np.isfinite(a) or a == 0:
        return np.nan
    return (-a) / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)


def devig_two_way(price_a: float, price_b: float) -> tuple[float, float]:
    """Proportional de-vig (artifact-selected) of a two-way American pair.

    Returns fair (p_a, p_b) summing to 1, or (nan, nan) if not computable.
    """
    pa = american_to_implied(price_a)
    pb = american_to_implied(price_b)
    if not (np.isfinite(pa) and np.isfinite(pb)):
        return (np.nan, np.nan)
    return devig_proportional(pa, pb)


def schedule_reference_market(games: pd.DataFrame) -> pd.DataFrame:
    """Per-game de-vigged schedule-reference market probabilities (all seasons).

    Exact-contract by construction: probabilities are quoted at the same
    ``home_spread_reference`` / ``total_line_reference`` the shadow consumes.
    """
    rows = []
    for _, g in games.iterrows():
        ml_home, _ = devig_two_way(g.get("home_moneyline_reference"), g.get("away_moneyline_reference"))
        cover_home, _ = devig_two_way(g.get("home_spread_price_reference"), g.get("away_spread_price_reference"))
        over, _ = devig_two_way(g.get("over_price_reference"), g.get("under_price_reference"))
        rows.append({
            "game_id": str(g["game_id"]),
            "market_moneyline": ml_home,
            "market_spread": cover_home,
            "market_total": over,
        })
    return pd.DataFrame(rows)


def closing_consensus_market(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Real multi-book pre-kickoff CLOSING consensus (2022-2024), matched to the
    shadow's reference contract point. Rows whose closing consensus line differs
    from the reference contract point are flagged CONTRACT_MISMATCH per market.
    """
    o = odds[odds["market_type"].isin(["moneyline", "spread", "total"])].copy()
    o = o[o["minutes_to_kickoff"].astype(float) > 0]  # exclude in-play / post-kickoff
    if "is_live" in o.columns:
        o = o[~o["is_live"].astype(bool)]
    o["mtk"] = o["minutes_to_kickoff"].astype(float)
    closing_snap = o.groupby("game_id")["mtk"].transform("min")
    closing = o[o["mtk"] == closing_snap].copy()

    ref = games.set_index(games["game_id"].astype(str))
    rows = []
    for gid, sub in closing.groupby("game_id"):
        gid = str(gid)
        rec = {"game_id": gid}
        # moneyline (no contract point)
        ml = sub[(sub["market_type"] == "moneyline") & (sub["outcome_side"] == "home")]
        rec["market_moneyline"] = float(ml["devig_probability"].mean()) if len(ml) else np.nan
        rec["moneyline_match_status"] = "MATCHED" if len(ml) else "NO_QUOTE"
        # spread
        sp = sub[(sub["market_type"] == "spread") & (sub["outcome_side"] == "home")]
        rec["market_spread"], rec["spread_match_status"] = _matched_consensus(
            sp, float(ref.loc[gid, "home_spread_reference"]) if gid in ref.index else np.nan)
        # total
        tot = sub[(sub["market_type"] == "total") & (sub["outcome_side"] == "over")]
        rec["market_total"], rec["total_match_status"] = _matched_consensus(
            tot, float(ref.loc[gid, "total_line_reference"]) if gid in ref.index else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def _matched_consensus(sub: pd.DataFrame, ref_point: float) -> tuple[float, str]:
    if not len(sub):
        return (np.nan, "NO_QUOTE")
    if not np.isfinite(ref_point):
        return (np.nan, "NO_REFERENCE_POINT")
    at_point = sub[np.isclose(sub["line_value"].astype(float), ref_point)]
    if not len(at_point):
        return (np.nan, "CONTRACT_MISMATCH")
    return (float(at_point["devig_probability"].mean()), "MATCHED")


# ---------------------------------------------------------------------------
# Fold construction & leakage guards
# ---------------------------------------------------------------------------
def build_fold_frames(matrix: pd.DataFrame, fold: dict, *, weeks: tuple[int, ...] | None = None):
    """Return (train, calibration, test) frames for a fold with hard leakage guards.

    - train: seasons <= train_max_season
    - calibration: season == calibration_season
    - test: season == test_season, restricted to ``weeks`` if given
    """
    ts = fold["test_season"]
    train = matrix[matrix["season"] <= fold["train_max_season"]].copy()
    calib = matrix[matrix["season"] == fold["calibration_season"]].copy()
    test = matrix[matrix["season"] == ts].copy()
    if weeks is not None:
        test = test[test["week"].isin(weeks)].copy()
    # leakage guards: no test-season rows may appear in train/calibration
    if (train["season"] == ts).any():
        raise ValueError(f"fold {ts}: test season leaked into training")
    if (calib["season"] == ts).any():
        raise ValueError(f"fold {ts}: test season leaked into calibration")
    if fold["train_max_season"] >= ts or fold["calibration_season"] >= ts:
        raise ValueError(f"fold {ts}: train/calibration window not strictly before test season")
    return train, calib, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def n_bins_for(n: int) -> int:
    return int(max(3, min(10, n // 12)))


def equal_mass_ece(y: np.ndarray, p: np.ndarray):
    """Adaptive equal-mass ECE. Returns float, or NOT_ESTIMABLE when n < 36."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n < 36:
        return NOT_ESTIMABLE
    nb = n_bins_for(n)
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for idx in np.array_split(order, nb):
        if len(idx) == 0:
            continue
        ece += (len(idx) / n) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def _irls_logistic(x: np.ndarray, y: np.ndarray, max_iter: int = 100):
    """Unpenalized logistic regression (intercept + single slope) via IRLS.

    Returns (intercept, slope) or None if non-estimable / separated / diverged.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) < 30 or np.unique(y).size < 2:
        return None
    # complete separation on the single predictor -> MLE non-finite -> not estimable
    x0, x1 = x[y == 0], x[y == 1]
    if x0.max() < x1.min() or x1.max() < x0.min():
        return None
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        WX = X * w[:, None]
        try:
            hessian = X.T @ WX
            new_beta = np.linalg.solve(hessian, X.T @ (w * z))
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(new_beta)):
            return None
        if np.max(np.abs(new_beta - beta)) < 1e-8:
            beta = new_beta
            break
        beta = new_beta
    if not np.all(np.isfinite(beta)) or np.abs(beta[1]) > 50:
        return None
    return float(beta[0]), float(beta[1])


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray):
    """Calibration slope & intercept from logistic fit of y on logit(p).

    Returns (slope, intercept) or (NOT_ESTIMABLE, NOT_ESTIMABLE).
    """
    p = _clip(p)
    logit = np.log(p / (1.0 - p))
    fit = _irls_logistic(logit, y)
    if fit is None:
        return NOT_ESTIMABLE, NOT_ESTIMABLE
    intercept, slope = fit
    return slope, intercept


def auc_diagnostic(y: np.ndarray, p: np.ndarray):
    """Rank-based AUC (Mann-Whitney). NOT_ESTIMABLE if a single class."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return NOT_ESTIMABLE
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _assign_tie_ranks(p, ranks)
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(p: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(p, kind="stable")
    sp = p[order]
    i = 0
    n = len(sp)
    while i < n:
        j = i
        while j + 1 < n and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            avg = np.mean(ranks[order[i:j + 1]])
            ranks[order[i:j + 1]] = avg
        i = j + 1


def score_market(y: np.ndarray, p: np.ndarray) -> dict:
    """Full metric bundle on graded (non-push/non-tie) rows."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "n": int(n),
        "log_loss": binary_log_loss(y, p) if n else NOT_ESTIMABLE,
        "brier": brier(y, p) if n else NOT_ESTIMABLE,
        "ece": equal_mass_ece(y, p),
        "ece_n_bins": n_bins_for(n) if n >= 36 else NOT_ESTIMABLE,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "auc": auc_diagnostic(y, p),
        "prediction_mean": float(p.mean()) if n else NOT_ESTIMABLE,
        "prediction_sd": float(p.std(ddof=0)) if n else NOT_ESTIMABLE,
        "outcome_mean": float(y.mean()) if n else NOT_ESTIMABLE,
    }


# ---------------------------------------------------------------------------
# Season-stratified paired bootstrap
# ---------------------------------------------------------------------------
def _season_stratified_indices(seasons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seasons = np.asarray(seasons)
    idx_parts = []
    for s in np.unique(seasons):
        pool = np.where(seasons == s)[0]
        idx_parts.append(rng.choice(pool, size=len(pool), replace=True))
    return np.concatenate(idx_parts)


def paired_bootstrap_logloss_diff(
    y: np.ndarray, p_shadow: np.ndarray, p_market: np.ndarray, seasons: np.ndarray,
    *, reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired season-stratified bootstrap of (shadow - market) log loss & Brier.

    Negative favors shadow. Returns point estimates and 95% percentile CIs.
    """
    y = np.asarray(y, dtype=float)
    ps = _clip(p_shadow)
    pm = _clip(p_market)
    seasons = np.asarray(seasons)
    n = len(y)
    if n == 0:
        return {"n": 0, "log_loss_diff": NOT_ESTIMABLE, "log_loss_ci": [NOT_ESTIMABLE, NOT_ESTIMABLE],
                "brier_diff": NOT_ESTIMABLE, "brier_ci": [NOT_ESTIMABLE, NOT_ESTIMABLE]}

    ll_shadow_i = -(y * np.log(ps) + (1 - y) * np.log(1 - ps))
    ll_market_i = -(y * np.log(pm) + (1 - y) * np.log(1 - pm))
    ll_diff_i = ll_shadow_i - ll_market_i
    br_shadow_i = (ps - y) ** 2
    br_market_i = (pm - y) ** 2
    br_diff_i = br_shadow_i - br_market_i

    rng = np.random.default_rng(seed)
    ll_boot = np.empty(reps)
    br_boot = np.empty(reps)
    for b in range(reps):
        idx = _season_stratified_indices(seasons, rng)
        ll_boot[b] = ll_diff_i[idx].mean()
        br_boot[b] = br_diff_i[idx].mean()
    return {
        "n": int(n),
        "shadow_log_loss": float(ll_shadow_i.mean()),
        "market_log_loss": float(ll_market_i.mean()),
        "log_loss_diff": float(ll_diff_i.mean()),
        "log_loss_ci": [float(np.percentile(ll_boot, 2.5)), float(np.percentile(ll_boot, 97.5))],
        "shadow_brier": float(br_shadow_i.mean()),
        "market_brier": float(br_market_i.mean()),
        "brier_diff": float(br_diff_i.mean()),
        "brier_ci": [float(np.percentile(br_boot, 2.5)), float(np.percentile(br_boot, 97.5))],
    }


# ---------------------------------------------------------------------------
# Reliability classification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketDiagnostics:
    n: int
    finite_valid: bool
    deterministic: bool
    leakage_free: bool
    ece: object
    slope: object
    intercept: object
    loso_stable: bool
    shadow_minus_market_logloss: object
    logloss_ci_lower: object  # lower bound of paired 95% CI of (shadow - market) log loss


def classify_market(d: MarketDiagnostics) -> str:
    """Apply the pre-registered reliability status rules for one market."""
    if not (d.finite_valid and d.deterministic and d.leakage_free):
        return "WITHHOLD_SHADOW_OUTPUT"
    diags_estimable = d.ece != NOT_ESTIMABLE and d.slope != NOT_ESTIMABLE and d.intercept != NOT_ESTIMABLE
    if d.n < MIN_CONCLUSIVE_N or not diags_estimable:
        return "INCONCLUSIVE_SMALL_SAMPLE"
    concerns = []
    if d.ece > ECE_THRESHOLD:
        concerns.append("ece")
    if not (SLOPE_RANGE[0] <= d.slope <= SLOPE_RANGE[1]):
        concerns.append("slope")
    if not (INTERCEPT_RANGE[0] <= d.intercept <= INTERCEPT_RANGE[1]):
        concerns.append("intercept")
    # materially worse than market: the paired 95% CI of (shadow - market) log loss
    # lies entirely above the +0.02 non-inferiority margin.
    if isinstance(d.logloss_ci_lower, (int, float)) and d.logloss_ci_lower > NON_INFERIORITY_LOGLOSS_MARGIN:
        concerns.append("non_inferiority")
    if not d.loso_stable:
        concerns.append("instability")
    if concerns:
        return "CALIBRATION_CONCERN"
    # SUPPORTED additionally requires shadow non-inferior to market within +0.02 log loss
    if isinstance(d.shadow_minus_market_logloss, (int, float)) and d.shadow_minus_market_logloss > NON_INFERIORITY_LOGLOSS_MARGIN:
        return "CALIBRATION_CONCERN"
    return "SUPPORTED_FOR_SHADOW_MONITORING"


def production_recommendation(statuses: dict[str, str]) -> str:
    vals = set(statuses.values())
    if "WITHHOLD_SHADOW_OUTPUT" in vals:
        return "WITHHOLD_SHADOW_UNTIL_REPAIRED"
    if "CALIBRATION_CONCERN" in vals:
        return "RETAIN_MARKET_BASELINE"
    return "RETAIN_MARKET_BASELINE_AND_MONITOR_SHADOW"


# ---------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------
def deterministic_frame_hash(df: pd.DataFrame, columns: list[str]) -> str:
    """Stable sha256 of a frame: fixed column order, sorted rows, rounded floats."""
    out = df[columns].copy()
    sort_cols = [c for c in ("evaluation_population", "evaluation_fold", "game_id", "market") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    records = []
    for _, row in out.iterrows():
        rec = []
        for c in columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                rec.append(None if pd.isna(v) else round(float(v), 10))
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                rec.append(int(v))
            elif isinstance(v, (bool, np.bool_)):
                rec.append(bool(v))
            elif v is None or (np.isscalar(v) and pd.isna(v)):
                rec.append(None)
            else:
                rec.append(str(v))
        records.append(rec)
    payload = {"columns": columns, "rows": records}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def canonical_json_hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
