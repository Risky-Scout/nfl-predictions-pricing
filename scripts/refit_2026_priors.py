"""Refit the full 2026 prior stack (team + QB + roster + coach) from populated
templates, and report quality vs the EPA-only fundamental priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_hybrid.data.availability import assert_available_before
from nfl_hybrid.priors.coach import HierarchicalCoachPrior
from nfl_hybrid.priors.quarterback import QuarterbackPriorBuilder, starter_mixture
from nfl_hybrid.priors.roster import RosterAdjustmentModel
from nfl_hybrid.priors.starter_overrides import (
    apply_overrides,
    flag_review,
    league_centered_fallback,
    load_overrides,
)
from nfl_hybrid.priors.team import EmpiricalBayesTeamPrior, TeamPriorConfig

T = Path("data/templates")
OUT = Path("outputs/priors_2026")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refit the full 2026-family prior stack.")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--as-of-utc", default="2026-08-01T00:00:00Z")
    ap.add_argument("--overrides", default="data/templates/starter_overrides.csv")
    args = ap.parse_args()
    SEASON, ASOF = args.season, args.as_of_utc

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"season": SEASON, "as_of_utc": ASOF}

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
    players_unique = (
        players.sort_values("effective_dropbacks", ascending=False)
        .drop_duplicates("player_id")
        .reset_index(drop=True)
    )
    players_unique["prior_source"] = "historical"

    # target-season starter candidates + human overrides (kept separate from generated)
    proj_all = starters[starters["season"] == SEASON].copy()
    overrides = load_overrides(args.overrides)
    proj_applied = apply_overrides(proj_all, overrides, as_of_utc=ASOF)
    proj_applied = flag_review(proj_applied, availability=None)

    # never drop a candidate for lacking a prior: add a league-centered fallback
    missing = sorted(set(proj_applied["player_id"]) - set(players_unique["player_id"]))
    if missing:
        fb = league_centered_fallback(missing, players_unique)
        combined = pd.concat([players_unique, fb], ignore_index=True)
    else:
        combined = players_unique
    report["qb_fallback_prior_candidates"] = int(len(missing))
    report["qb_override_rows_applied"] = int((proj_applied["origin"] == "override").sum())

    mix_in = proj_applied[["team_id", "player_id", "starter_probability"]]
    team_qb = starter_mixture(combined, mix_in)
    team_qb.to_parquet(OUT / "qb_team_mixtures_2026_full.parquet", index=False)
    proj_applied.to_csv(OUT / "starter_probabilities_applied.csv", index=False)

    # validation: every team's probabilities sum to 1.0
    sums = proj_applied.groupby("team_id")["starter_probability"].sum()
    report["qb_teams"] = int(proj_applied["team_id"].nunique())
    report["qb_all_teams_sum_to_one"] = bool(np.allclose(sums.to_numpy(), 1.0, atol=1e-6))
    report["qb_needs_review_count"] = int(
        proj_applied.groupby("team_id")["qb_review_status"].first().eq("NEEDS_QB_REVIEW").sum()
    )
    report["qb_team_mixtures_rows"] = int(len(team_qb))
    report["qb_player_priors_rows"] = int(len(players))
    report["qb_valid_team_mixtures"] = int(team_qb["team_id"].nunique() if "team_id" in team_qb.columns else 0)

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
