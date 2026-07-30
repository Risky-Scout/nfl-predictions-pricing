"""Refit the full 2026 prior stack (team + QB + roster + coach) from populated
templates, and report quality vs the EPA-only fundamental priors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.data.availability import assert_available_before
from nfl_hybrid.priors.coach import HierarchicalCoachPrior
from nfl_hybrid.priors.quarterback import QuarterbackPriorBuilder, starter_mixture
from nfl_hybrid.priors.roster import RosterAdjustmentModel
from nfl_hybrid.priors.team import EmpiricalBayesTeamPrior, TeamPriorConfig

T = Path("data/templates")
OUT = Path("outputs/priors_2026")
ASOF = "2026-08-01T00:00:00Z"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}

    # ---- leakage check: no 2026 rows feed the prior; all available before as-of
    team_hist = pd.read_csv(T / "team_metric_history.csv")
    assert (team_hist["season"] <= 2025).all(), "team history contains target-season rows"
    lk = team_hist.copy()
    lk["asof"] = ASOF
    assert_available_before(lk, available_at_column="available_at_utc", prediction_time_column="asof")
    report["leakage_team_history_pass"] = True

    # ---- team priors (full 15-metric set)
    team = EmpiricalBayesTeamPrior(TeamPriorConfig(target_season=2026)).build(
        team_hist, target_season=2026, as_of_utc=ASOF
    )
    team["prior_family"] = "fundamental_full"
    team.to_parquet(OUT / "team_priors_2026_full.parquet", index=False)
    report["team_priors_rows"] = int(len(team))
    report["team_priors_metrics"] = int(team["component"].nunique())
    report["team_priors_teams"] = int(team["entity_id"].nunique())

    # ---- QB priors + starter mixture
    qb_hist = pd.read_csv(T / "quarterback_history.csv")
    starters = pd.read_csv(T / "starter_probabilities.csv")
    players = QuarterbackPriorBuilder().build(qb_hist, as_of_utc=ASOF)
    players.to_parquet(OUT / "qb_player_priors_2026_full.parquet", index=False)
    # collapse to one row per player (most-representative = most effective dropbacks)
    players_unique = (
        players.sort_values("effective_dropbacks", ascending=False)
        .drop_duplicates("player_id")
        .reset_index(drop=True)
    )
    # 2026 team mixture uses the projected 2026 starter rows
    proj = starters[starters["season"] == 2026][["team_id", "player_id", "starter_probability"]]
    proj = proj[proj["player_id"].isin(players_unique["player_id"])]
    if len(proj):
        team_qb = starter_mixture(players_unique, proj)
        team_qb.to_parquet(OUT / "qb_team_mixtures_2026_full.parquet", index=False)
        report["qb_team_mixtures_rows"] = int(len(team_qb))
    report["qb_player_priors_rows"] = int(len(players))
    report["qb_2026_starter_candidates"] = int(len(proj))

    # ---- roster adjustment model
    roster_hist = pd.read_csv(T / "offseason_roster_history.csv")
    feat = [c for c in roster_hist.columns if c.startswith("returning_") and roster_hist[c].notna().any()]
    fit_df = roster_hist.dropna(subset=["early_season_performance_residual"] + feat)
    if len(fit_df) >= 20 and feat:
        model = RosterAdjustmentModel(feature_columns=feat).fit(fit_df)
        report["roster_model_fit"] = True
        report["roster_features_used"] = feat
        report["roster_residual_sd"] = round(float(model.residual_sd_), 4)
        report["roster_training_rows"] = int(len(fit_df))
    else:
        report["roster_model_fit"] = False

    # ---- coach priors
    coach_hist = pd.read_csv(T / "coach_history.csv")
    coach = HierarchicalCoachPrior().build(coach_hist, as_of_utc=ASOF)
    coach.to_parquet(OUT / "coach_priors_2026.parquet", index=False)
    report["coach_priors_rows"] = int(len(coach))
    report["coach_priors_coaches"] = int(coach_hist["coach_id"].nunique())

    # ---- quality vs EPA-only fundamental priors
    epa_only = OUT / "team_priors_2026_fundamental.parquet"
    if epa_only.exists():
        prev = pd.read_parquet(epa_only)
        report["comparison_vs_epa_only"] = {
            "epa_only_metrics": int(prev["component"].nunique()),
            "full_metrics": int(team["component"].nunique()),
            "epa_only_rows": int(len(prev)),
            "full_rows": int(len(team)),
            "added_metrics": sorted(set(team["component"]) - set(prev["component"])),
        }

    (OUT / "refit_2026_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    # show the widened priors: teams with widest QB-driven uncertainty etc.
    print("\nTop-5 team offense_epa_per_play priors (full refit):")
    off = team[team["component"] == "offense_epa_per_play"].nlargest(5, "prior_mean")
    print(off[["entity_id", "prior_mean", "prior_standard_deviation", "regression_weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
