"""Resolve the ``(season, week, season_type)`` identity of the card a
certified horizon would price right now, so an unattended capture never has to
guess a week number.

Runs on the production host (it reads the composite games population). It
captures nothing, fits nothing and writes nothing.

HOW THE CARD IS IDENTIFIED
  Entirely by existing certified primitives, in this order:

    1. ``run_2026.current_or_recent_cutoff(as_of, horizon)`` gives the
       card-scoped noon America/New_York cutoff for this horizon
       (:func:`nfl_hybrid.features.horizon_elo._card_monday_date` /
       ``_card_noon_cutoff_utc``);
    2. ``horizon_elo.build_horizon_membership_ledger`` over the SAME composite
       games population production reads assigns every game to its canonical
       card and both horizons' cutoffs;
    3. the card whose ``<horizon>_cutoff_utc`` equals the cutoff from step 1
       AND which still has at least one horizon-eligible game IS the card.

  No week arithmetic, no "current week = weeks since kickoff", no calendar
  heuristic. ``CARD_KEY`` (season, week, season_type) is taken verbatim from
  the ledger.

FAIL CLOSED
  Zero matching cards, or more than one, is an error. A capture would
  otherwise be filed under a week that does not correspond to the cutoff it
  was frozen for, and the certified market path cross-checks exactly that.

Usage:
  python scripts/resolve_current_card_identity.py --horizon TUE
  python scripts/resolve_current_card_identity.py --horizon FRI --as-of 2026-09-11T16:05:00Z
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from nfl_hybrid.features import horizon_elo as he  # noqa: E402
from nfl_hybrid.production import run_2026 as prod  # noqa: E402


class CardIdentityError(RuntimeError):
    """The card for this horizon and instant is not unambiguously determined."""


def resolve_card(*, horizon: str, as_of: str | None = None) -> dict:
    if horizon not in he.HORIZONS:
        raise CardIdentityError(f"horizon must be one of {list(he.HORIZONS)}; got {horizon!r}")

    as_of_utc = prod._as_utc(as_of) if as_of else prod.utc_now()
    target_cutoff_utc = prod.current_or_recent_cutoff(as_of_utc, horizon)

    games, provenance = prod.load_games_population_with_provenance()
    ledger = he.build_horizon_membership_ledger(games)

    cutoff_column = f"{horizon.lower()}_cutoff_utc"
    eligible_column = f"{horizon.lower()}_eligible"
    match = ledger.loc[
        (pd.to_datetime(ledger[cutoff_column], utc=True) == target_cutoff_utc) & ledger[eligible_column]
    ]
    if match.empty:
        raise CardIdentityError(
            f"no {horizon} card with cutoff {target_cutoff_utc.isoformat()} has an eligible game in the "
            f"games population (rows={provenance['reg_post_row_count']}, "
            f"schedule_2026_available={provenance['schedule_2026_available']})"
        )

    cards = match[list(he.CARD_KEY)].drop_duplicates().to_dict(orient="records")
    if len(cards) != 1:
        raise CardIdentityError(
            f"FAIL CLOSED: {len(cards)} cards share the {horizon} cutoff {target_cutoff_utc.isoformat()}: {cards}"
        )

    card = cards[0]
    return {
        "horizon": horizon,
        "as_of_utc": as_of_utc.isoformat(),
        "target_cutoff_utc": target_cutoff_utc.isoformat(),
        "season": int(card["season"]),
        # BDL captures are filed by integer week; the population carries it as
        # a string so the historical estate's own labels survive round-tripping.
        "week": int(str(card["week"])),
        "season_type": str(card["season_type"]),
        "eligible_game_count": int(len(match)),
        "games_population_content_sha256": provenance["reg_post_content_sha256"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--horizon", required=True, choices=list(he.HORIZONS))
    parser.add_argument("--as-of", default=None)
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also append season/week/season_type to $GITHUB_OUTPUT when it is set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        card = resolve_card(horizon=args.horizon, as_of=args.as_of)
    except Exception as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "detail": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2

    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            for key in ("season", "week", "season_type", "target_cutoff_utc"):
                handle.write(f"{key}={card[key]}\n")

    print(json.dumps(card, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
