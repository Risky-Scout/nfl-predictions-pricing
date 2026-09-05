"""Canonical 2026 operational production entrypoint (Section 12).

Modes:
  --preflight                           Verify live infrastructure, print no secrets.
  --horizon TUE|FRI [--as-of ISO8601]   Force-run one horizon (manual/testing).
  --run-due [--as-of ISO8601]           Decide from local NY time whether TUE or FRI
                                         is due right now; NOT_DUE exits 0 with no writes.
  --attach-results --result-file PATH   Attach real results to the evaluation ledger
                                         (never mutates a forecast).

The live 2026 market source is EXPLICIT and optional:
``--market-capture-manifest PATH`` names the exact frozen BallDontLie
capture manifest to price from. No capture is ever auto-selected (not by
mtime, directory name, capture_started_at, UUID, or "first COMPLETE"), and
omitting the flag leaves every historical code path untouched.

``--as-of`` is for deterministic historical/integration testing only;
production default is the current UTC time. A manual ``--horizon`` run never
fabricates a future market snapshot -- it only runs the current-or-most-
recent-past cutoff for that horizon relative to ``--as-of`` (or now).

Reuses the certified pipeline unchanged
(``src/nfl_hybrid/production/run_2026.py``, which itself reuses
``nfl_hybrid.features.horizon_elo`` /
``nfl_hybrid.evaluation.official_horizon_oof`` /
``nfl_hybrid.evaluation.chronological_calibration`` /
``nfl_hybrid.evaluation.raw_market_reconstruction`` / the certified hash
gate / the frozen production calibration seed). No duplicate
reimplementation of any scientific rule. Never commits, never opens a PR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_hybrid.production import run_2026 as prod  # noqa: E402


def _resolve_as_of(args: argparse.Namespace) -> pd.Timestamp:
    if args.as_of:
        return prod._as_utc(args.as_of)
    return prod.utc_now()


def cmd_preflight(args: argparse.Namespace) -> int:
    # When the operator also names a horizon, the capture is cross-checked
    # against that horizon's real DST-aware certified cutoff for the current
    # card -- the same cutoff a run right now would price at.
    expected_horizon = args.horizon
    expected_cutoff = (
        prod.current_or_recent_cutoff(_resolve_as_of(args), expected_horizon) if expected_horizon else None
    )
    result = prod.run_preflight(
        market_capture_manifest=args.market_capture_manifest,
        expected_horizon=expected_horizon,
        expected_target_cutoff_utc=expected_cutoff,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    # Exit 0 ONLY when a real live production run could actually be
    # performed -- i.e. infrastructure is ready AND every required live 2026
    # input (schedule, registered market source) is genuinely available.
    # Infra-ready-but-no-live-inputs is BLOCKED_ON_LIVE_INPUTS and exits 1:
    # it must not be mistaken for readiness by an operator or a scheduler.
    return 0 if result.get("production_run_ready") else 1


def cmd_horizon(args: argparse.Namespace) -> int:
    as_of_utc = _resolve_as_of(args)
    manifest = prod.run_horizon_batch(
        horizon=args.horizon, as_of_utc=as_of_utc, force=True,
        market_capture_manifest=args.market_capture_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if manifest["status"] in ("SUCCESS", "NOT_DUE") else 1


def cmd_run_due(args: argparse.Namespace) -> int:
    as_of_utc = _resolve_as_of(args)
    exit_code = 0
    any_due = False
    for horizon in prod.he.HORIZONS:
        due = prod.is_within_due_window(as_of_utc, horizon)
        if not due["due"]:
            continue
        any_due = True
        manifest = prod.run_horizon_batch(
            horizon=horizon, as_of_utc=as_of_utc, force=True,
            market_capture_manifest=args.market_capture_manifest,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        if manifest["status"] not in ("SUCCESS", "NOT_DUE"):
            exit_code = 1
    if not any_due:
        print(json.dumps({"status": "NOT_DUE", "as_of_utc": str(as_of_utc)}, indent=2, default=str))
    return exit_code


def cmd_attach_results(args: argparse.Namespace) -> int:
    evaluation_root = prod.artifact_root() / "production-2026" / "evaluation-ledger"
    records = json.loads(Path(args.result_file).read_text())
    attachment_run_time = prod.utc_now()
    exit_code = 0
    for rec in records:
        try:
            prod.attach_result(
                evaluation_root, game_id=rec["game_id"], horizon=rec["horizon"],
                target_cutoff_utc=rec["target_cutoff_utc"], result=rec["result"],
                result_available_at_utc=rec["result_available_at_utc"], attachment_run_time=attachment_run_time,
                result_source_hash=rec["result_source_hash"],
            )
            print(f"attached: {rec['game_id']}/{rec['horizon']}")
        except prod.ProductionHardStop as exc:
            print(f"FAILED {rec['game_id']}: {exc.status} {exc.detail}")
            exit_code = 1
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--horizon", choices=["TUE", "FRI"])
    parser.add_argument("--run-due", action="store_true")
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--attach-results", action="store_true")
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument(
        "--market-capture-manifest", type=str, default=None,
        help=(
            "Path to the EXACT official frozen BallDontLie capture manifest.json for this card's "
            "live market. Never auto-selected: with no value there is no live 2026 market source "
            "and production stays BLOCKED_ON_LIVE_INPUTS."
        ),
    )
    args = parser.parse_args()

    if args.preflight:
        return cmd_preflight(args)
    if args.attach_results:
        if not args.result_file:
            parser.error("--attach-results requires --result-file")
        return cmd_attach_results(args)
    if args.horizon:
        return cmd_horizon(args)
    if args.run_due:
        return cmd_run_due(args)
    parser.error("one of --preflight, --horizon, --run-due, --attach-results is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
