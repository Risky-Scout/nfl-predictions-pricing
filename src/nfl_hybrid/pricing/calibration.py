"""Offline pricing-calibration path (Section 3-7 of the completion spec).

Fits and selects, by leakage-safe expanding walk-forward, the pricing mathematics:
margin surface, margin/total sigma, de-vig method, and consensus method. Produces
one versioned, human-readable production artifact. This module does no runtime
pricing; the weekly path only *loads* the artifact it emits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from nfl_hybrid.data.team_ids import try_canonical_team_id
from nfl_hybrid.pricing.devig import devig_pair
from nfl_hybrid.pricing.margin_surface import (
    MarginSurface,
    build_empirical_pmf_table,
    discretized_normal_pmf,
)

_EPS = 1e-6
TEST_SEASONS = [2022, 2023, 2024]
BOOT_SEED = 20260815
BOOT_N = 2000


# --------------------------------------------------------------------------- #
# Data extraction
# --------------------------------------------------------------------------- #
def load_spreadspoke_margins(path: str, *, season_min: int = 2002, season_max: int = 2024) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.copy()
    df["season"] = pd.to_numeric(df["schedule_season"], errors="coerce")
    df = df[(df["season"] >= season_min) & (df["season"] <= season_max)]
    hs = pd.to_numeric(df["score_home"], errors="coerce")
    as_ = pd.to_numeric(df["score_away"], errors="coerce")
    sf = pd.to_numeric(df["spread_favorite"], errors="coerce")
    tot = pd.to_numeric(df["over_under_line"], errors="coerce")

    def home_spread(row_fav, row_home, s):
        if pd.isna(s):
            return np.nan
        fav = try_canonical_team_id(str(row_fav), allow_pick=True)
        home = try_canonical_team_id(str(row_home))
        if fav == "PICK" or s == 0:
            return 0.0
        return s if fav == home else -s

    out = pd.DataFrame({
        "season": df["season"].astype("Int64"),
        "home_margin": (hs - as_),
        "total_points": (hs + as_),
        "closing_home_spread": [home_spread(f, h, s) for f, h, s in zip(df["team_favorite_id"], df["team_home"], sf)],
        "closing_total": tot,
    })
    return out.dropna(subset=["home_margin", "closing_home_spread"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Bootstrap on paired per-game log-loss differences
# --------------------------------------------------------------------------- #
def _paired_bootstrap_ci(loss_a: np.ndarray, loss_b: np.ndarray, *, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    diff = np.asarray(loss_a, float) - np.asarray(loss_b, float)  # a - b; <0 means a better
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(n)])
    return (float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _pointwise_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _equal_mass_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p)
    y, p = y[order], p[order]
    ece = 0.0
    for chunk_y, chunk_p in zip(np.array_split(y, bins), np.array_split(p, bins)):
        if len(chunk_p):
            ece += (len(chunk_p) / len(y)) * abs(chunk_p.mean() - chunk_y.mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# Margin surface selection (moneyline log-loss, walk-forward on Spreadspoke)
# --------------------------------------------------------------------------- #
def select_margin_surface(margins: pd.DataFrame, *, baseline_sigma: float = 13.5, buckets=(1.0, 2.0, 3.0)) -> dict:
    def moneyline_probs(surface: MarginSurface, test: pd.DataFrame) -> np.ndarray:
        return np.array([surface.moneyline_probabilities(s, baseline_sigma)[0] for s in test["closing_home_spread"]])

    results = {}
    # normal baseline
    normal_losses, normal_y = [], []
    for ts in TEST_SEASONS:
        test = margins[margins["season"] == ts]
        surf = MarginSurface(method="normal")
        p = moneyline_probs(surf, test)
        y = (test["home_margin"] > 0).astype(int).to_numpy()
        normal_losses.append(_pointwise_logloss(y, p)); normal_y.append((y, p))
    normal_loss = np.concatenate(normal_losses)
    yv = np.concatenate([a for a, _ in normal_y]); pv = np.concatenate([b for _, b in normal_y])
    results["normal_baseline"] = {
        "log_loss": float(normal_loss.mean()), "brier": float(brier_score_loss(yv, np.clip(pv, _EPS, 1 - _EPS))),
        "ece": _equal_mass_ece(yv, pv), "pointwise": normal_loss,
    }
    # empirical, per bucket width
    for bw in buckets:
        losses, ys = [], []
        for ts in TEST_SEASONS:
            train = margins[margins["season"] < ts]
            test = margins[margins["season"] == ts]
            tbl = build_empirical_pmf_table(train, bucket_width=bw, baseline_sigma=baseline_sigma)
            surf = MarginSurface(method="empirical", pmf_table=tbl, bucket_width=bw)
            p = moneyline_probs(surf, test)
            y = (test["home_margin"] > 0).astype(int).to_numpy()
            losses.append(_pointwise_logloss(y, p)); ys.append((y, p))
        loss = np.concatenate(losses)
        yv = np.concatenate([a for a, _ in ys]); pv = np.concatenate([b for _, b in ys])
        results[f"empirical_bw{bw}"] = {
            "log_loss": float(loss.mean()), "brier": float(brier_score_loss(yv, np.clip(pv, _EPS, 1 - _EPS))),
            "ece": _equal_mass_ece(yv, pv), "pointwise": loss, "bucket_width": bw,
        }

    # selection: best empirical vs normal, with complexity tie-break toward normal
    best_emp = min((k for k in results if k.startswith("empirical")), key=lambda k: results[k]["log_loss"])
    mean_d, lo, hi = _paired_bootstrap_ci(results["normal_baseline"]["pointwise"], results[best_emp]["pointwise"])
    empirical_wins = results[best_emp]["log_loss"] < results["normal_baseline"]["log_loss"] and lo > 0
    selected = best_emp if empirical_wins else "normal_baseline"
    return {
        "selected": selected,
        "candidates": {k: {m: v[m] for m in ("log_loss", "brier", "ece")} for k, v in results.items()},
        "normal_minus_best_empirical_logloss": {"mean": mean_d, "ci_lower": lo, "ci_upper": hi},
        "best_empirical_bucket_width": results[best_emp].get("bucket_width"),
    }


# --------------------------------------------------------------------------- #
# Sigma selection (margin & total) via OOS probability score
# --------------------------------------------------------------------------- #
def _fit_linear_sigma(resid: np.ndarray, total: np.ndarray) -> tuple[float, float]:
    # sigma(total) = a + b*total, fit so that |resid| ~ sigma (MAE-consistent scale)
    x = np.c_[np.ones_like(total), total]
    absr = np.abs(resid) * np.sqrt(np.pi / 2.0)  # E|N|=sigma*sqrt(2/pi)
    coef, *_ = np.linalg.lstsq(x, absr, rcond=None)
    return float(coef[0]), float(coef[1])


def select_sigma(residuals: pd.DataFrame, resid_col: str, *, kind: str = "margin") -> dict:
    """residuals: columns season, <resid_col>, closing_total. Select constant vs
    linear_in_total by out-of-sample Gaussian log-density (probability score)."""
    def gauss_nll(resid, sigma):
        sigma = np.maximum(sigma, 1e-6)
        return 0.5 * np.log(2 * np.pi * sigma ** 2) + (resid ** 2) / (2 * sigma ** 2)

    const_losses, lin_losses = [], []
    lin_coefs = []
    for ts in TEST_SEASONS:
        tr = residuals[residuals["season"] < ts]
        te = residuals[residuals["season"] == ts]
        if len(tr) < 100 or len(te) == 0:
            continue
        r_tr = tr[resid_col].to_numpy(float); r_te = te[resid_col].to_numpy(float)
        t_te = te["closing_total"].to_numpy(float)
        sd = max(float(np.std(r_tr, ddof=1)), 1.0)
        const_losses.append(gauss_nll(r_te, sd))
        a, b = _fit_linear_sigma(r_tr, tr["closing_total"].to_numpy(float))
        sig_te = np.clip(a + b * t_te, max(0.5 * sd, 1.0), 3 * sd)
        lin_losses.append(gauss_nll(r_te, sig_te)); lin_coefs.append((a, b))
    const_loss = np.concatenate(const_losses); lin_loss = np.concatenate(lin_losses)
    mean_d, lo, hi = _paired_bootstrap_ci(const_loss, lin_loss)  # const - linear
    linear_wins = lin_loss.mean() < const_loss.mean() and lo > 0
    overall_sd = max(float(np.std(residuals[resid_col], ddof=1)), 1.0)
    a, b = _fit_linear_sigma(residuals[resid_col].to_numpy(float), residuals["closing_total"].to_numpy(float))
    return {
        "kind": kind,
        "selected": "linear_in_total" if linear_wins else "constant",
        "constant_sigma": overall_sd,
        "linear_coef": {"a": a, "b": b},
        "oos_gauss_nll": {"constant": float(const_loss.mean()), "linear_in_total": float(lin_loss.mean())},
        "constant_minus_linear": {"mean": mean_d, "ci_lower": lo, "ci_upper": hi},
        "total_range": [float(residuals["closing_total"].min()), float(residuals["closing_total"].max())],
    }


# --------------------------------------------------------------------------- #
# De-vig + consensus selection from purchased closing odds
# --------------------------------------------------------------------------- #
def _closing_snapshot(odds: pd.DataFrame) -> pd.DataFrame:
    o = odds[odds["market_type"].isin(["moneyline", "spread", "total"])].copy()
    o = o[pd.to_numeric(o["minutes_to_kickoff"], errors="coerce") > 0]
    o["mtk"] = pd.to_numeric(o["minutes_to_kickoff"], errors="coerce")
    o = o[o["mtk"] == o.groupby("game_id")["mtk"].transform("min")]
    return o


def select_devig_consensus(odds: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    """outcomes: game_id, season, home_win, home_cover(no-push), over(no-push).
    Grid over de-vig x consensus per market, walk-forward log-loss.
    """
    closing = _closing_snapshot(odds)
    market_outcome = {"moneyline": ("home_win", "home", "away"),
                      "spread": ("home_cover", "home", "away"),
                      "total": ("over", "over", "under")}
    devig_methods = ["proportional", "power", "shin"]
    consensus_methods = ["equal_mean", "median", "sharpest_book_anchor"]

    def per_book_fair(sub_market: pd.DataFrame, side_a: str, side_b: str, method: str) -> pd.DataFrame:
        # group by (game, book, point) -> pair the two sides -> de-vig -> home/over prob
        rows = []
        keys = ["game_id", "bookmaker_id", "market_pair_line"] if sub_market["market_type"].iloc[0] != "moneyline" else ["game_id", "bookmaker_id"]
        for _, grp in sub_market.groupby(keys):
            a = grp[grp["outcome_side"] == side_a]["raw_implied_probability"]
            b = grp[grp["outcome_side"] == side_b]["raw_implied_probability"]
            if len(a) == 1 and len(b) == 1:
                qa, _ = devig_pair(float(a.iloc[0]), float(b.iloc[0]), method)
                rows.append({"game_id": grp["game_id"].iloc[0], "book": grp["bookmaker_id"].iloc[0], "fair": qa})
        return pd.DataFrame(rows)

    results = {}
    for market, (ycol, side_a, side_b) in market_outcome.items():
        sub = closing[closing["market_type"] == market]
        grid = {}
        for dv in devig_methods:
            fair = per_book_fair(sub, side_a, side_b, dv)
            if fair.empty:
                continue
            # coverage: require >=3 books
            counts = fair.groupby("game_id")["book"].nunique()
            eligible = counts[counts >= 3].index
            fair = fair[fair["game_id"].isin(eligible)]
            merged_base = fair.merge(outcomes[["game_id", "season", ycol]], on="game_id", how="inner")
            for cons in consensus_methods:
                if cons == "equal_mean":
                    cons_p = merged_base.groupby(["game_id", "season", ycol])["fair"].mean()
                elif cons == "median":
                    cons_p = merged_base.groupby(["game_id", "season", ycol])["fair"].median()
                else:  # sharpest_book_anchor: leakage-safe per test season
                    cons_p = _anchor_consensus(merged_base, ycol)
                    if cons_p is None:
                        continue
                cp = cons_p.reset_index()
                cp.columns = list(cp.columns[:-1]) + ["p"]
                y = cp[ycol].to_numpy(float); p = np.clip(cp["p"].to_numpy(float), _EPS, 1 - _EPS)
                mask = np.isin(y, [0.0, 1.0]) & np.isfinite(p)
                if mask.sum() < 50:
                    continue
                pw = _pointwise_logloss(y[mask], p[mask])
                grid[f"{dv}+{cons}"] = {"log_loss": float(pw.mean()), "brier": float(brier_score_loss(y[mask], p[mask])),
                                        "ece": _equal_mass_ece(y[mask], p[mask]), "n": int(mask.sum()),
                                        "game_ids": cp["game_id"].to_numpy()[mask], "pointwise": pw}
        if not grid:
            results[market] = {"selected_devig": "proportional", "selected_consensus": "equal_mean", "grid": {}}
            continue
        best = min(grid, key=lambda k: grid[k]["log_loss"])
        default = "proportional+equal_mean"
        tie_break = None
        # principled complexity tie-break: keep the simple default unless the best is
        # better on a paired game-clustered bootstrap (CI of default-best excludes 0).
        if default in grid and best != default:
            mean_d, lo, hi = _paired_bootstrap_ci_aligned(
                grid[default]["game_ids"], grid[default]["pointwise"],
                grid[best]["game_ids"], grid[best]["pointwise"],
            )
            tie_break = {"default_minus_best_mean": mean_d, "ci_lower": lo, "ci_upper": hi}
            if not (mean_d > 0 and lo > 0):
                best = default
        dv, cons = best.split("+")
        clean_grid = {k: {m: v[m] for m in ("log_loss", "brier", "ece", "n")} for k, v in grid.items()}
        results[market] = {"selected_devig": dv, "selected_consensus": cons, "grid": clean_grid,
                           "tie_break_vs_default": tie_break}
    return results


def _paired_bootstrap_ci_aligned(ids_a, loss_a, ids_b, loss_b, *, n: int = BOOT_N, seed: int = BOOT_SEED):
    """Align two loss arrays by game id, then paired bootstrap of (a - b)."""
    da = pd.Series(loss_a, index=ids_a)
    db = pd.Series(loss_b, index=ids_b)
    common = da.index.intersection(db.index)
    if len(common) == 0:
        return (np.nan, np.nan, np.nan)
    return _paired_bootstrap_ci(da.loc[common].to_numpy(), db.loc[common].to_numpy(), n=n, seed=seed)


def _anchor_consensus(merged: pd.DataFrame, ycol: str):
    """Sharpest-book anchor chosen using only prior-season outcomes (leakage-safe)."""
    parts = []
    for ts in TEST_SEASONS:
        prior = merged[merged["season"] < ts]
        test = merged[merged["season"] == ts]
        if prior.empty or test.empty:
            continue
        # book eligibility: >=200 quotes across >=2 prior seasons
        elig = []
        for book, bg in prior.groupby("book"):
            if len(bg) >= 200 and bg["season"].nunique() >= 2:
                y = bg[ycol].to_numpy(float); p = np.clip(bg["fair"].to_numpy(float), _EPS, 1 - _EPS)
                m = np.isin(y, [0, 1])
                if m.sum() >= 100:
                    elig.append((book, _pointwise_logloss(y[m], p[m]).mean()))
        if not elig:
            return None
        anchor = min(elig, key=lambda t: t[1])[0]
        anchored = test[test["book"] == anchor]
        if anchored.empty:  # fallback to mean on that season
            anchored = test.groupby(["game_id", "season", ycol])["fair"].mean().reset_index()
            parts.append(anchored.set_index(["game_id", "season", ycol])["fair"])
        else:
            parts.append(anchored.groupby(["game_id", "season", ycol])["fair"].mean())
    if not parts:
        return None
    return pd.concat(parts)
