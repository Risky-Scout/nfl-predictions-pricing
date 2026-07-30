"""Pre-registered 2026 EPA candidate tournament (dev seasons 2022-2024).

Five candidates (see docs/model-selection/candidates-2026/registry.md), each
producing per-game no-tie/no-push probabilities for moneyline, ATS and totals,
scored against the REAL-CLOSING de-vigged market. Walk-forward: each test season
trains only on prior seasons. 2025 is never passed in.

Kept intentionally small and explicit so a skeptic can audit every model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from nfl_hybrid.modern.joint_score import JointScoreModel

PROB_COLS = [
    "home_win_probability_no_tie",
    "home_cover_probability_no_push",
    "over_probability_no_push",
]
_EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _margin_to_probs(pred_margin, pred_total, home_spread, total_line, margin_sd, total_sd):
    home_win = 1.0 - norm.cdf((0.0 - pred_margin) / margin_sd)
    ats_edge = pred_margin + home_spread
    cover = norm.cdf(ats_edge / margin_sd)
    over = norm.cdf((pred_total - total_line) / total_sd)
    return _clip(home_win), _clip(cover), _clip(over)


def _prep(frame, features):
    x = frame[features].to_numpy(float)
    return np.nan_to_num(x, nan=0.0)


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def candidate_market_residual(train, test, features):
    """C1: ridge on EPA/rest predicting margin & total residuals vs market."""
    xtr, xte = _prep(train, features), _prep(test, features)
    margin_resid = train["home_margin"].to_numpy(float) - (-train["home_spread"].to_numpy(float))
    total_resid = train["total_points"].to_numpy(float) - train["total_line"].to_numpy(float)

    m = Ridge(alpha=10.0).fit(xtr, margin_resid)
    t = Ridge(alpha=10.0).fit(xtr, total_resid)
    pred_margin = -test["home_spread"].to_numpy(float) + m.predict(xte)
    pred_total = test["total_line"].to_numpy(float) + t.predict(xte)

    margin_sd = max(float(np.std(margin_resid - m.predict(xtr), ddof=1)), 1.0)
    total_sd = max(float(np.std(total_resid - t.predict(xtr), ddof=1)), 1.0)
    hw, cov, ov = _margin_to_probs(
        pred_margin, pred_total, test["home_spread"].to_numpy(float),
        test["total_line"].to_numpy(float), margin_sd, total_sd,
    )
    return pd.DataFrame(
        {PROB_COLS[0]: hw, PROB_COLS[1]: cov, PROB_COLS[2]: ov, "predicted_margin": pred_margin},
        index=test.index,
    )


def _isotonic_gbm(xtr, ytr, xcal, ycal, xte):
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=1.0, random_state=42,
    )
    clf.fit(xtr, ytr)
    raw_cal = clf.predict_proba(xcal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    if np.unique(ycal).size > 1:
        iso.fit(raw_cal, ycal)
        return _clip(iso.predict(clf.predict_proba(xte)[:, 1]))
    return _clip(clf.predict_proba(xte)[:, 1])


def candidate_gbm(train, test, features, calibration_season):
    """C2: GBM per market, isotonic-calibrated on the prior season only."""
    cal = train[train["season"] == calibration_season]
    fit = train[train["season"] != calibration_season]
    if len(cal) < 30 or len(fit) < 50:
        cal, fit = train, train
    xfit, xcal, xte = _prep(fit, features), _prep(cal, features), _prep(test, features)
    out = {}
    for col, target in zip(PROB_COLS, ["home_win", "home_cover", "over"]):
        out[col] = _isotonic_gbm(xfit, fit[target].to_numpy(int), xcal, cal[target].to_numpy(int), xte)
    return pd.DataFrame(out, index=test.index)


def candidate_logistic(train, test, features):
    """C3: standardized per-market logistic regression (interpretable floor)."""
    scaler = StandardScaler().fit(_prep(train, features))
    xtr, xte = scaler.transform(_prep(train, features)), scaler.transform(_prep(test, features))
    out = {}
    for col, target in zip(PROB_COLS, ["home_win", "home_cover", "over"]):
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(xtr, train[target].to_numpy(int))
        out[col] = _clip(clf.predict_proba(xte)[:, 1])
    return pd.DataFrame(out, index=test.index)


def candidate_jointscore_epa(train, test, features, calibration_season):
    """C4: JointScoreModel with market + EPA numeric features."""
    model = JointScoreModel(numeric_features=list(features))
    cal = train[train["season"] == calibration_season]
    fit = train[train["season"] != calibration_season]
    if len(cal) < 30 or len(fit) < 50:
        fit, cal = train, train
    model.fit(fit, calibration=cal)
    pred = model.predict_markets(test)
    return pd.DataFrame(
        {
            PROB_COLS[0]: _clip(pred["home_win_probability_no_tie"].to_numpy()),
            PROB_COLS[1]: _clip(pred["home_cover_probability_no_push"].to_numpy()),
            PROB_COLS[2]: _clip(pred["over_probability_no_push"].to_numpy()),
            "predicted_margin": pred["predicted_margin"].to_numpy(),
        },
        index=test.index,
    )


def candidate_stacked(train, test, features, calibration_season, market_prob_col_map):
    """C5: logistic meta over C1-C4 OOF probs + reference market prob meta-feature.

    Base models are fit on all-but-last training season and predict the held-out
    last training season -> meta-training features (leakage-safe blend holdout).
    Base models are then refit on all train to predict the test season.
    """
    seasons = sorted(train["season"].unique())
    holdout = seasons[-1]
    base_fit = train[train["season"] != holdout]
    meta_train = train[train["season"] == holdout]
    if len(base_fit) < 50 or len(meta_train) < 30:
        return None  # not enough seasons to stack cleanly

    def base_probs(tr, te, cal_season):
        c1 = candidate_market_residual(tr, te, features)
        c2 = candidate_gbm(tr, te, features, cal_season)
        c3 = candidate_logistic(tr, te, features)
        c4 = candidate_jointscore_epa(tr, te, features, cal_season)
        return c1, c2, c3, c4

    out = {}
    for col, target, mkt_col in zip(PROB_COLS, ["home_win", "home_cover", "over"], market_prob_col_map):
        # meta-train features
        cal_inner = sorted(base_fit["season"].unique())[-1]
        mc1, mc2, mc3, mc4 = base_probs(base_fit, meta_train, cal_inner)
        meta_x = np.column_stack([
            mc1[col].to_numpy(), mc2[col].to_numpy(), mc3[col].to_numpy(), mc4[col].to_numpy(),
            meta_train[mkt_col].to_numpy(float),
        ])
        meta_y = meta_train[target].to_numpy(int)
        if np.unique(meta_y).size < 2:
            out[col] = meta_train[mkt_col].to_numpy(float)[:0]
            continue
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(np.nan_to_num(meta_x), meta_y)
        # test features from base models refit on all train
        tc1, tc2, tc3, tc4 = base_probs(train, test, calibration_season)
        test_x = np.column_stack([
            tc1[col].to_numpy(), tc2[col].to_numpy(), tc3[col].to_numpy(), tc4[col].to_numpy(),
            test[mkt_col].to_numpy(float),
        ])
        out[col] = _clip(meta.predict_proba(np.nan_to_num(test_x))[:, 1])
    return pd.DataFrame(out, index=test.index)


CANDIDATES = ("C1_market_residual", "C2_gbm", "C3_logistic", "C4_jointscore_epa", "C5_stacked")


def run_walk_forward(matrix, features, *, test_seasons, market_prob_cols):
    """Return {candidate: predictions_df} pooled over test seasons.

    ``market_prob_cols`` maps [ml, ats, total] reference-market probability
    columns present in ``matrix`` (used only as the C5 meta-feature).
    """
    results = {name: [] for name in CANDIDATES}
    for test_season in test_seasons:
        train = matrix[matrix["season"] < test_season].copy()
        test = matrix[matrix["season"] == test_season].copy()
        if len(train) < 100 or len(test) == 0:
            continue
        cal_season = sorted(train["season"].unique())[-1]

        preds = {
            "C1_market_residual": candidate_market_residual(train, test, features),
            "C2_gbm": candidate_gbm(train, test, features, cal_season),
            "C3_logistic": candidate_logistic(train, test, features),
            "C4_jointscore_epa": candidate_jointscore_epa(train, test, features, cal_season),
            "C5_stacked": candidate_stacked(train, test, features, cal_season, market_prob_cols),
        }
        keep = ["game_id", "season", "week", "home_win", "home_cover", "over",
                "home_margin", "total_points", "home_spread", "total_line"]
        for name, pred in preds.items():
            if pred is None:
                continue
            merged = pd.concat([test[keep].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
            results[name].append(merged)

    return {name: pd.concat(frames, ignore_index=True) for name, frames in results.items() if frames}
