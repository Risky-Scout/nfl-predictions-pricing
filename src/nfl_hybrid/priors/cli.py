from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_hybrid.priors.quarterback import QuarterbackPriorBuilder, starter_mixture
from nfl_hybrid.priors.team import EmpiricalBayesTeamPrior, TeamPriorConfig


def _write(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe NFL season priors from canonical inputs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    team = sub.add_parser("team", help="Build empirical-Bayes team/unit priors.")
    team.add_argument("--history", type=Path, required=True)
    team.add_argument("--output", type=Path, required=True)
    team.add_argument("--target-season", type=int, default=2026)
    team.add_argument("--as-of-utc")
    team.add_argument("--season-weights", default="0.60,0.28,0.12")
    team.add_argument("--prior-strength", type=float, default=500.0)

    qb = sub.add_parser("qb-mixture", help="Build player and team starter-mixture QB priors.")
    qb.add_argument("--history", type=Path, required=True)
    qb.add_argument("--starter-probabilities", type=Path, required=True)
    qb.add_argument("--player-output", type=Path, required=True)
    qb.add_argument("--team-output", type=Path, required=True)
    qb.add_argument("--as-of-utc")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "team":
        history = pd.read_csv(args.history)
        weights = tuple(float(x) for x in args.season_weights.split(","))
        config = TeamPriorConfig(
            season_weights=weights,
            prior_strength=args.prior_strength,
            target_season=args.target_season,
        )
        output = EmpiricalBayesTeamPrior(config).build(
            history,
            target_season=args.target_season,
            as_of_utc=args.as_of_utc,
        )
        _write(output, args.output)
        return

    history = pd.read_csv(args.history)
    probabilities = pd.read_csv(args.starter_probabilities)
    players = QuarterbackPriorBuilder().build(
        history,
        as_of_utc=args.as_of_utc,
    )
    teams = starter_mixture(players, probabilities)
    _write(players, args.player_output)
    _write(teams, args.team_output)


if __name__ == "__main__":
    main()
