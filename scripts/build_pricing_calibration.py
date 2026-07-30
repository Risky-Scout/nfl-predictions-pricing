"""Offline calibration build. Fits/selects the pricing mathematics via leakage-safe
walk-forward and writes ONE versioned production artifact + a readable PMF export.

Outputs:
  config/pricing_calibration_2026.json         (human-readable artifact)
  config/pricing_calibration_2026_pmf.csv      (empirical margin PMF export)
Run: PYTHONPATH=src .venv/bin/python scripts/build_pricing_calibration.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.pricing.calibration import (
    load_spreadspoke_margins,
    select_devig_consensus,
    select_margin_surface,
    select_sigma,
    TEST_SEASONS,
    BOOT_SEED,
)
from nfl_hybrid.pricing.margin_surface import MarginSurface, build_empirical_pmf_table, discretized_normal_pmf

SPREADSPOKE = "data/spreadspoke_enhanced.csv"
PURCHASED = "data/purchased/odds_closing_dev_2022_2024.parquet"
GAMES = "data/backfill_2020_2025/canonical/games.parquet"
ART = Path("config/pricing_calibration_2026.json")
PMF_CSV = Path("config/pricing_calibration_2026_pmf.csv")
KEY_NUMBERS = [3, 6, 7, 10, 14]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def push_calibration(margins: pd.DataFrame, surface: MarginSurface, sigma: float) -> dict:
    """Predicted vs actual push mass at integer key numbers (walk-forward test seasons)."""
    test = margins[margins["season"].isin(TEST_SEASONS)]
    out = {}
    for kn in KEY_NUMBERS:
        for sign, label in [(-1, f"home_-{kn}"), (1, f"away_+{kn}")]:
            spread = sign * kn  # home spread; push when home_margin == -spread
            games = test[np.isclose(test["closing_home_spread"], spread)]
            if len(games) < 30:
                continue
            actual = float((games["home_margin"] == -spread).mean())
            # predicted push at that integer line for a game with that closing spread
            _, push, _ = surface.cover_probabilities(spread, spread, sigma)
            out[label] = {"n": int(len(games)), "predicted_push": round(push, 4), "actual_push": round(actual, 4)}
    mae = np.mean([abs(v["predicted_push"] - v["actual_push"]) for v in out.values()]) if out else np.nan
    return {"by_key_number": out, "push_mae": float(mae)}


def build_outcomes() -> pd.DataFrame:
    games = pd.read_parquet(GAMES)
    dev = games[games["season"].isin(TEST_SEASONS)].copy()
    hm = pd.to_numeric(dev["home_score"], errors="coerce") - pd.to_numeric(dev["away_score"], errors="coerce")
    tp = pd.to_numeric(dev["home_score"], errors="coerce") + pd.to_numeric(dev["away_score"], errors="coerce")
    hs = pd.to_numeric(dev["home_spread_reference"], errors="coerce")
    tl = pd.to_numeric(dev["total_line_reference"], errors="coerce")
    cov, ov = hm + hs, tp - tl
    out = pd.DataFrame({
        "game_id": dev["game_id"].astype(str), "season": dev["season"],
        "home_win": (hm > 0).astype(float),
        "home_cover": np.where(cov > 0, 1.0, np.where(cov < 0, 0.0, np.nan)),
        "over": np.where(ov > 0, 1.0, np.where(ov < 0, 0.0, np.nan)),
    })
    return out


def main() -> None:
    margins = load_spreadspoke_margins(SPREADSPOKE)
    resid = margins.dropna(subset=["closing_total", "total_points"]).copy()
    resid["margin_resid"] = resid["home_margin"] - (-resid["closing_home_spread"])
    resid["total_resid"] = resid["total_points"] - resid["closing_total"]
    resid = resid[resid["season"] >= 2020]

    print("selecting margin surface ...")
    ms = select_margin_surface(margins)
    print("selecting sigma ...")
    margin_sigma = select_sigma(resid, "margin_resid", kind="margin")
    total_sigma = select_sigma(resid, "total_resid", kind="total")
    print("selecting de-vig + consensus ...")
    odds = pd.read_parquet(PURCHASED); odds["game_id"] = odds["game_id"].astype(str)
    devig = select_devig_consensus(odds, build_outcomes())

    # --- margin surface decision ------------------------------------------- #
    # Pre-registration primary metric (moneyline log-loss) is a statistical tie
    # (CI includes 0). Per registry Amendment 1, the empirical surface is selected
    # because push/key-number calibration -- the quantity the surface must get right
    # for ATS-push and alternate-line pricing -- is decisively better and correct.
    best_bw = ms["best_empirical_bucket_width"]
    sigma_const = margin_sigma["constant_sigma"]
    final_pmf = build_empirical_pmf_table(margins[margins["season"] <= 2024], bucket_width=best_bw, baseline_sigma=sigma_const)
    emp_surface = MarginSurface(method="empirical", pmf_table=final_pmf, bucket_width=best_bw)
    norm_surface = MarginSurface(method="normal")
    push_emp = push_calibration(margins, emp_surface, sigma_const)
    push_norm = push_calibration(margins, norm_surface, sigma_const)
    margin_method = "empirical"  # Amendment 1 (documented in registry)

    PMF_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_pmf.to_csv(PMF_CSV, index=False)

    artifact = {
        "artifact_version": "pricing_calibration_2026.v1",
        "created_at_utc": "2026-07-30T00:00:00Z",
        "training_cutoff_season": 2024,
        "holdout_untouched": 2025,
        "code_commit": _git_commit(),
        "random_seed": BOOT_SEED,
        "source_hashes": {
            "spreadspoke": _sha256(SPREADSPOKE),
            "purchased_closing_odds": _sha256(PURCHASED),
        },
        "margin_surface": {
            "selected": margin_method,
            "bucket_width": best_bw,
            "tail_cutoff": 21,
            "baseline_sigma": sigma_const,
            "pmf_export_csv": str(PMF_CSV),
            "fallback": "normal",
            "primary_metric_moneyline_logloss_tie": ms["normal_minus_best_empirical_logloss"],
            "candidates": ms["candidates"],
            "selection_note": (
                "Moneyline log-loss statistically indistinguishable from normal "
                "(bootstrap CI includes 0). Empirical selected per registry Amendment 1 "
                "for correct push/key-number mass; normal retained as configurable fallback."
            ),
        },
        "push_calibration": {"empirical": push_emp, "normal": push_norm},
        "margin_sigma": margin_sigma,
        "total_sigma": total_sigma,
        "devig_consensus": {m: {k: v[k] for k in ("selected_devig", "selected_consensus", "tie_break_vs_default")}
                            for m, v in devig.items()},
        "devig_consensus_full_grid": {m: v["grid"] for m, v in devig.items()},
        "runtime_defaults": {
            "moneyline_devig": devig["moneyline"]["selected_devig"],
            "moneyline_consensus": devig["moneyline"]["selected_consensus"],
            "ats_devig": devig["spread"]["selected_devig"],
            "ats_consensus": devig["spread"]["selected_consensus"],
            "total_devig": devig["total"]["selected_devig"],
            "total_consensus": devig["total"]["selected_consensus"],
            "margin_sigma_value": sigma_const,
            "total_sigma_value": total_sigma["constant_sigma"],
        },
    }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    artifact["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    ART.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print("\n=== SELECTIONS ===")
    print(f"margin surface : {margin_method} (bucket_width={best_bw})  [push MAE emp={push_emp['push_mae']:.4f} vs normal={push_norm['push_mae']:.4f}]")
    print(f"margin sigma   : {margin_sigma['selected']} ({sigma_const:.2f})")
    print(f"total sigma    : {total_sigma['selected']} ({total_sigma['constant_sigma']:.2f})")
    for m in ("moneyline", "spread", "total"):
        print(f"{m:10s}: devig={devig[m]['selected_devig']} consensus={devig[m]['selected_consensus']}")
    print(f"\nartifact_sha256={artifact['artifact_sha256']}")
    print(f"wrote {ART} and {PMF_CSV}")


if __name__ == "__main__":
    main()
