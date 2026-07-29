"""``predict_week`` CLI: season/week -> betting-card CSV.

Sources the target week's games either from a canonical backfill directory
(``--backfill-dir``, reads ``canonical/games.parquet``) or an explicit games
file (``--games``), attaches the frozen production readiness status per market,
runs the joint reference distribution + penalized-Kelly staking, and writes a
betting card. With the frozen 2025 production spec (market baseline for every
market) the card recommends no bets and flags ``no-bet: edge not established``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.io import read_tabular
from nfl_hybrid.pricing.betting_card import (
    CardConfig,
    build_betting_card,
    readiness_from_production_spec,
)
from nfl_hybrid.staking.kelly import StakingPolicy

_WEEK_ORDER = [str(w) for w in range(1, 19)] + [
    "Wildcard",
    "Division",
    "Conference",
    "Superbowl",
]


def _week_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    try:
        return (_WEEK_ORDER.index(text), text)
    except ValueError:
        return (len(_WEEK_ORDER), text)


def _load_games(args: argparse.Namespace) -> pd.DataFrame:
    if args.games:
        frame = read_tabular(args.games)
    elif args.backfill_dir:
        path = Path(args.backfill_dir) / "canonical" / "games.parquet"
        frame = pd.read_parquet(path)
    else:  # pragma: no cover - argparse enforces one of the two
        raise ValueError("Provide either --games or --backfill-dir.")

    frame = frame.copy()
    rename = {
        "home_team_id": "home_team",
        "away_team_id": "away_team",
        "home_spread_reference": "home_spread",
        "total_line_reference": "total_line",
    }
    for src, dst in rename.items():
        if src in frame.columns and dst not in frame.columns:
            frame[dst] = frame[src]

    if "season" in frame.columns:
        frame = frame[pd.to_numeric(frame["season"], errors="coerce") == args.season]
    if "week" in frame.columns:
        frame = frame[frame["week"].astype(str) == str(args.week)]
    return frame.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a weekly NFL betting card (season/week -> CSV)."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", required=True, help='e.g. "1"-"18" or "Wildcard".')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--games", help="CSV/parquet of games for the week.")
    source.add_argument("--backfill-dir", help="Backfill dir with canonical/games.parquet.")
    parser.add_argument(
        "--production-spec",
        default="config/production_model_spec.json",
        help="Frozen production spec for per-market readiness status.",
    )
    parser.add_argument("--output", required=True, help="Output betting-card CSV path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    games = _load_games(args)
    if len(games) == 0:
        raise SystemExit(
            f"No games found for season {args.season} week {args.week}."
        )

    spec_path = Path(args.production_spec)
    if spec_path.exists():
        readiness = readiness_from_production_spec(spec_path)
    else:
        readiness = {"moneyline": "RETAIN_BASELINE", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}

    card = build_betting_card(
        games,
        readiness_by_market=readiness,
        staking_policy=StakingPolicy(),
        config=CardConfig(),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    card.to_csv(out, index=False)

    bets = int(card["should_bet"].sum())
    print(f"card_rows={len(card)} games={games['game_id'].nunique()} recommended_bets={bets}")
    print(f"readiness={readiness}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
