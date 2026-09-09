"""Resolve, for one instant, what the authoritative 2026 production workflow
is allowed to do.

Emits a machine-readable decision (and GitHub Actions step outputs when
``GITHUB_OUTPUT`` is set) answering two independent questions:

  daily_due      -- may the DAILY maintenance pass run? Always true: evidence
                    refresh, population update, prospective evaluation,
                    recalibration-candidate generation and public verification
                    are safe, idempotent, every-day work.

  certified_due  -- is a CERTIFIED public prediction horizon due right now?
                    Answered ONLY by the existing certified semantics --
                    :func:`nfl_hybrid.production.run_2026.is_within_due_window`
                    (America/New_York weekday + the 12:00-12:20 local window)
                    and :func:`nfl_hybrid.production.run_2026.current_or_recent_cutoff`
                    (the card-scoped noon cutoff built on
                    ``nfl_hybrid.features.horizon_elo``'s own
                    ``_card_monday_date``/``_card_noon_cutoff_utc``).

NO NEW HORIZON, NO NEW CLOCK
  There is no DAILY forecast horizon. ``certified_horizon`` is only ever
  ``TUE`` or ``FRI`` (:data:`nfl_hybrid.features.horizon_elo.HORIZONS`), and
  it is decided by the certified window functions above -- never by a
  hard-coded EDT/EST offset, never by comparing UTC hours, and never by a
  second timezone implementation. A scheduled invocation that is not inside a
  certified window resolves ``certified_due=false``, which the workflow treats
  as a clean no-op rather than a failure.

  Two DST-redundant cron firings therefore cost nothing: whichever one is not
  actually 12:00-12:20 America/New_York that week resolves ``false`` here, and
  the certified production job is skipped.

Usage:
  python scripts/resolve_production_schedule.py
  python scripts/resolve_production_schedule.py --as-of 2026-09-08T16:05:00Z
  python scripts/resolve_production_schedule.py --force-horizon TUE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_hybrid.production import run_2026 as prod  # noqa: E402

MODE_DAILY = "daily"
MODE_CERTIFIED = "certified"


def resolve(
    *, as_of: str | None = None, force_horizon: str | None = None, daily: bool = True
) -> dict:
    as_of_utc = prod._as_utc(as_of) if as_of else prod.utc_now()

    windows = {horizon: prod.is_within_due_window(as_of_utc, horizon) for horizon in prod.he.HORIZONS}
    due_horizons = [horizon for horizon, window in windows.items() if window["due"]]

    if force_horizon:
        if force_horizon not in prod.he.HORIZONS:
            raise SystemExit(
                f"--force-horizon must be one of {list(prod.he.HORIZONS)}; there is no DAILY forecast horizon"
            )
        certified_horizon: str | None = force_horizon
        basis = "OPERATOR_FORCED_HORIZON"
    elif len(due_horizons) == 1:
        certified_horizon = due_horizons[0]
        basis = "CERTIFIED_DUE_WINDOW"
    elif len(due_horizons) > 1:
        # Structurally impossible (TUE and FRI are different weekdays), but
        # never silently pick one.
        raise SystemExit(f"ambiguous certified horizon: {due_horizons} are both due at {as_of_utc.isoformat()}")
    else:
        certified_horizon = None
        basis = "NOT_IN_ANY_CERTIFIED_DUE_WINDOW"

    target_cutoff_utc = (
        prod.current_or_recent_cutoff(as_of_utc, certified_horizon).isoformat() if certified_horizon else None
    )

    return {
        "as_of_utc": as_of_utc.isoformat(),
        "as_of_local_ny": as_of_utc.tz_convert(prod.NY_ZONE).isoformat(),
        "timezone": str(prod.NY_ZONE),
        "daily_due": bool(daily),
        "certified_due": certified_horizon is not None,
        "certified_horizon": certified_horizon or "",
        "certified_target_cutoff_utc": target_cutoff_utc or "",
        "certified_basis": basis,
        "certified_horizons_available": list(prod.he.HORIZONS),
        "due_windows": windows,
    }


def _write_github_output(decision: dict) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key in (
            "daily_due",
            "certified_due",
            "certified_horizon",
            "certified_target_cutoff_utc",
            "certified_basis",
            "as_of_utc",
            "as_of_local_ny",
        ):
            value = decision[key]
            if isinstance(value, bool):
                value = "true" if value else "false"
            handle.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", default=None, help="ISO8601 instant to resolve for (deterministic testing).")
    parser.add_argument(
        "--force-horizon",
        default=None,
        help="TUE or FRI. Operator override for a workflow_dispatch replay of an existing capture.",
    )
    parser.add_argument(
        "--no-daily", action="store_true", help="Report daily_due=false (dispatch of the certified pass alone)."
    )
    args = parser.parse_args(argv)

    decision = resolve(as_of=args.as_of, force_horizon=args.force_horizon, daily=not args.no_daily)
    _write_github_output(decision)
    print(json.dumps(decision, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
