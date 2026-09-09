"""DAILY: generate a 2026 recalibration CANDIDATE and report promotion state.

Runs the repository's EXISTING chronological calibration machinery over the
current composite games population and records the result as an immutable
CANDIDATE. It never promotes anything: the certified production calibrator
(``$NFL_MODEL_ARTIFACT_ROOT/fix8-official-oof-calibration-2026/
production_calibration_seed.json``) is never opened for writing by this
script, and promotion is decided only by the frozen, machine-readable
prospective preregistration (see
:func:`nfl_hybrid.production.recalibration_2026.promotion_authorization`).

The pipeline is exactly the certified one, phase for phase, with no new
scientific step:

  1. the composite games population
     (:func:`nfl_hybrid.production.run_2026.load_games_population_with_provenance`)
     -- certified historical 2020-2025 + canonical BDL 2026 REG/POST;
  2. the card-scoped TUE/FRI horizon membership ledger
     (:func:`nfl_hybrid.features.horizon_elo.build_horizon_membership_ledger`);
  3. the official eligible-target matrix and the certified chronological
     Ridge-alpha-100 OOF on the frozen six Elo features
     (:mod:`nfl_hybrid.evaluation.official_horizon_oof`);
  4. raw timestamped bookmaker-history market reconstruction
     (:mod:`nfl_hybrid.evaluation.raw_market_reconstruction`);
  5. ATS/TOTAL raw pricing
     (:func:`nfl_hybrid.evaluation.chronological_calibration.build_raw_probabilities`);
  6. the frozen conditional + push calibrator fits
     (:mod:`nfl_hybrid.calibration.three_way`), assembled into a
     candidate-shaped seed by
     :func:`nfl_hybrid.production.recalibration_2026.build_candidate_seed`.

Because step 1 is the composite population, a completed 2026 game becomes
part of a candidate's evidence automatically as soon as the existing
chronology rules permit it -- no separate retraining engine, no manual step.

Usage:
  python scripts/generate_2026_recalibration_candidate.py
  python scripts/generate_2026_recalibration_candidate.py --report-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.data.external_data import artifact_root  # noqa: E402
from nfl_hybrid.evaluation import official_horizon_oof as ohf  # noqa: E402
from nfl_hybrid.evaluation import raw_market_reconstruction as rmr  # noqa: E402
from nfl_hybrid.evaluation.chronological_calibration import (  # noqa: E402
    MARKET_ATS,
    MARKET_TOTAL,
    build_raw_probabilities,
)
from nfl_hybrid.features import horizon_elo as he  # noqa: E402
from nfl_hybrid.production import recalibration_2026 as rc  # noqa: E402
from nfl_hybrid.production import run_2026 as prod  # noqa: E402

_MARKET_BY_STREAM = {
    "ATS_TUE": MARKET_ATS,
    "ATS_FRI": MARKET_ATS,
    "TOTAL_TUE": MARKET_TOTAL,
    "TOTAL_FRI": MARKET_TOTAL,
}
_HORIZON_BY_STREAM = {"ATS_TUE": "TUE", "ATS_FRI": "FRI", "TOTAL_TUE": "TUE", "TOTAL_FRI": "FRI"}
_RAW_KEY_BY_MARKET = {MARKET_ATS: rmr.MARKET_SPREADS, MARKET_TOTAL: rmr.MARKET_TOTALS}


def _price_stream(residual_ledger: pd.DataFrame, consensus: pd.DataFrame, *, market: str, horizon: str):
    """Identical in shape to ``scripts/run_fix8_official_oof_calibration.py``'s
    own ``_price_market``: map each target's consensus line onto its residual
    row and let the certified :func:`build_raw_probabilities` do the pricing."""
    base = residual_ledger.reset_index(drop=True).copy()
    base["game_id"] = base["game_id"].astype(str)
    consensus_line = (
        consensus.set_index("game_id")["consensus_line"] if not consensus.empty else pd.Series(dtype=float)
    )
    threshold = base["game_id"].map(consensus_line).to_numpy(dtype=float)
    raw = build_raw_probabilities(
        base,
        market=market,
        forecast_horizon=prod._FORECAST_HORIZON_LABEL[horizon],
        threshold=threshold,
    )
    return raw, threshold


def generate_candidate(
    *,
    artifact_root_path: Path,
    operational_root: Path,
    repo_root: Path,
    games_population_root: Path | None = None,
) -> dict:
    games, games_provenance = prod.load_games_population_with_provenance(
        games_population_root=games_population_root
    )
    membership_ledger = he.build_horizon_membership_ledger(games)

    fix71_summary = json.loads(
        (repo_root / "outputs" / "fix7_1_horizon_asof_elo_recertification_summary.json").read_text()
    )
    fix8_prereg = json.loads(
        (repo_root / "outputs" / "fix8_official_oof_calibration_preregistration.json").read_text()
    )
    hash_checks = prod.cert.verify_certified_hashes(fix71_summary, fix8_prereg)
    feature_state_hash = hash_checks["horizon_feature_semantics_hash"]

    quotes = rmr.load_raw_bookmaker_quotes()
    coherent = {
        market: rmr.build_coherent_book_observations(quotes, raw_key)
        for market, raw_key in _RAW_KEY_BY_MARKET.items()
    }

    residual_by_horizon: dict[str, pd.DataFrame] = {}
    for horizon in he.HORIZONS:
        matrix = ohf.build_official_horizon_matrix(games, horizon, membership_ledger)
        _, residual_ledger, _ = ohf.build_official_horizon_oof(
            matrix, horizon=horizon, feature_state_hash=feature_state_hash
        )
        residual_by_horizon[horizon] = residual_ledger

    raw_by_stream: dict[str, pd.DataFrame] = {}
    threshold_by_stream: dict[str, np.ndarray] = {}
    for stream in rc.STREAM_NAMES:
        market = _MARKET_BY_STREAM[stream]
        horizon = _HORIZON_BY_STREAM[stream]
        residual_ledger = residual_by_horizon[horizon]
        targets = residual_ledger[["game_id", "target_cutoff_utc"]].copy()
        reconstruction = rmr.reconstruct_market_at_cutoffs(
            coherent[market], targets, market=_RAW_KEY_BY_MARKET[market]
        )
        raw, threshold = _price_stream(
            residual_ledger, reconstruction.consensus, market=market, horizon=horizon
        )
        raw_by_stream[stream] = raw
        threshold_by_stream[stream] = threshold

    seed = rc.build_candidate_seed(raw_by_stream=raw_by_stream, threshold_by_stream=threshold_by_stream)

    maturity = rc.prospective_maturity_state(operational_root)
    authorization = rc.promotion_authorization(repo_root)
    result = rc.write_candidate(
        artifact_root_path,
        seed=seed,
        generated_for_cutoff_utc=None,
        generated_at_utc=prod.utc_now().isoformat(),
        git_commit=prod._git_commit(repo_root),
        games_population_provenance=games_provenance,
        prospective_maturity=maturity,
        authorization=authorization,
    )
    return {
        "status": result.status,
        "candidate_id": result.candidate_id,
        "candidate_dir": str(result.directory),
        "manifest": result.manifest,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report candidate/promotion state without generating a new candidate.",
    )
    parser.add_argument("--artifact-root", default=None, help="Override NFL_MODEL_ARTIFACT_ROOT.")
    parser.add_argument(
        "--operational-root",
        default=None,
        help="Root holding production-2026/ ledgers (defaults to the artifact root).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        artifact_root_path = Path(args.artifact_root) if args.artifact_root else artifact_root()
    except Exception as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2
    operational_root = Path(args.operational_root) if args.operational_root else artifact_root_path

    generation: dict | None = None
    if not args.report_only:
        try:
            generation = generate_candidate(
                artifact_root_path=artifact_root_path,
                operational_root=operational_root,
                repo_root=REPO_ROOT,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "CANDIDATE_GENERATION_FAILED", "detail": f"{type(exc).__name__}: {exc}"},
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1

    report = rc.candidate_state_report(
        artifact_root_path=artifact_root_path,
        operational_root=operational_root,
        repo_root=REPO_ROOT,
    )
    print(
        json.dumps(
            {"generation": generation, "candidate_state": report}, indent=2, sort_keys=True, default=str
        )
    )
    # A missing preregistered promotion authorization is the EXPECTED steady
    # state, not a failure: candidate generation succeeded, the certified
    # calibrator stays active, and the state is reported. Exit 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
