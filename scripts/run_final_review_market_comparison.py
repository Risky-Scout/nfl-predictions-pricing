"""FINAL REVIEW CERTIFICATION -- Sections 9-10: absolute probability
quality and sportsbook comparison. Reuses the certified Fix-8 2025
residual ledgers and market-consensus files verbatim (no refit, no
recalibration, no new feature research).

Sign convention (verified numerically against actual outcomes, not
assumed): ATS ``consensus_line`` is the home team's spread
(implied_margin = -consensus_line, confirmed via
corr(-consensus_line, actual_margin) > 0 and corr(predicted_margin,
-consensus_line) ~ 0.90 on TUE 2025 data). TOTAL ``consensus_line`` is the
total line directly (implied_total = consensus_line, confirmed via
corr(consensus_line, actual_total) > 0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.certification import final_review_2026 as cert  # noqa: E402
from nfl_hybrid.data.external_data import artifact_root  # noqa: E402

ROOT = artifact_root() / "fix8-official-oof-calibration-2026"
OUTPUT_DIR = artifact_root() / "final-review-certification-2026"


def main() -> int:
    per_stream_arrays = {s: cert.load_stream_paired_arrays(s, ROOT) for s in cert.STREAM_NAMES}

    absolute_quality = {s: cert.absolute_quality_metrics(a) for s, a in per_stream_arrays.items()}
    for s, m in absolute_quality.items():
        print(f"[{s}] n={m['n']} auc={m['auc']} slope={m['calibration_slope']} intercept={m['calibration_intercept']} "
              f"sharpness={m['sharpness_mean_abs_dev_from_half']:.4f} label={m['discrimination_label']}")

    point_forecast = {}
    probability_vs_market = {}
    for horizon in ("tue", "fri"):
        residual_ledger = pd.read_parquet(ROOT / f"{horizon}_official_oof_residual_ledger.parquet")
        for market_key, market_label in (("ats", "ATS"), ("total", "TOTAL")):
            consensus = pd.read_parquet(ROOT / f"{horizon}_{market_key}_market_consensus.parquet")
            key = f"{market_label}_{horizon.upper()}"
            pf = cert.point_forecast_vs_market(residual_ledger, consensus, market=market_label)
            point_forecast[key] = pf
            print(f"[{key}] point-forecast: model_rmse={pf['model_rmse']:.3f} market_rmse={pf['market_rmse']:.3f} "
                  f"delta={pf['model_minus_market_rmse']:+.3f} n={pf['n']}")

            stream = f"{market_label}_{horizon.upper()}"
            ledger = pd.read_parquet(ROOT / f"{market_key}_{horizon}_calibration_ledger.parquet")
            raw = pd.read_parquet(ROOT / f"{market_key}_{horizon}_raw_probabilities.parquet")
            season_num = pd.to_numeric(ledger["season"], errors="coerce")
            ledger_2025 = ledger[season_num == 2025]
            joined = ledger_2025.merge(raw[["game_id", "binary_target"]], on="game_id", how="left")
            scored = joined[(joined["calibration_status"] == "CALIBRATED") & joined["binary_target"].notna()]
            pvm = cert.probability_vs_market_paired(scored, consensus)
            probability_vs_market[stream] = pvm
            print(f"[{stream}] probability: model_minus_market log_loss_delta={pvm['log_loss_delta']:+.4f} "
                  f"ci95={pvm['log_loss_delta_ci95']} n={pvm['n']}")

    point_forecast_beats_market = all(v["model_minus_market_rmse"] < 0 for v in point_forecast.values())
    probability_edge_demonstrated = all(v["model_beats_market_ci95_entirely_below_zero"] for v in probability_vs_market.values())

    point_forecasting_status = "MET" if point_forecast_beats_market else "NOT_MET"
    probability_edge_status = "DEMONSTRATED" if probability_edge_demonstrated else "NOT_DEMONSTRATED"
    print(f"\nPOINT_FORECASTING_VS_SPORTSBOOK = {point_forecasting_status}")
    print(f"SPORTSBOOK_PROBABILITY_EDGE = {probability_edge_status}")

    result = {
        "schema_version": cert.SCHEMA_VERSION,
        "absolute_quality": absolute_quality,
        "point_forecast_vs_market": point_forecast,
        "probability_vs_market": probability_vs_market,
        "point_forecasting_vs_sportsbook_status": point_forecasting_status,
        "sportsbook_probability_edge_status": probability_edge_status,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "absolute_quality_and_market_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str)
    )
    print(f"\nPersisted: {OUTPUT_DIR}/absolute_quality_and_market_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
