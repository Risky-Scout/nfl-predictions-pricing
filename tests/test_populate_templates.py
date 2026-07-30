"""Schema + leakage tests for template population on small synthetic inputs."""

import numpy as np
import pandas as pd

from nfl_hybrid.priors.populate_templates import (
    build_quarterback_history,
    build_team_metric_history,
)
from nfl_hybrid.priors.quarterback import QuarterbackPriorBuilder
from nfl_hybrid.priors.team import EmpiricalBayesTeamPrior, TeamPriorConfig


def _synthetic_pbp(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for season in range(2020, 2026):
        for team in ("KC", "BUF"):
            opp = "BUF" if team == "KC" else "KC"
            for g in range(4):
                for _ in range(40):
                    pid += 1
                    dbk = int(rng.random() < 0.6)
                    rows.append(
                        dict(
                            game_id=f"{season}_{team}_{g}", season=season,
                            posteam=team, defteam=opp,
                            play_type="pass" if dbk else "run",
                            epa=float(rng.normal(0.05, 1.0)), success=int(rng.random() < 0.45),
                            down=int(rng.integers(1, 5)), yardline_100=int(rng.integers(1, 100)),
                            qb_dropback=dbk, rush_attempt=1 - dbk, pass_attempt=dbk,
                            sack=0, interception=0, fumble_lost=0, special_teams_play=0,
                            passer_player_id=f"QB_{team}" if dbk else None,
                        )
                    )
    return pd.DataFrame(rows)


def test_team_metric_history_schema_and_engine_compat():
    pbp = _synthetic_pbp()
    hist = build_team_metric_history(pbp, "hash123")
    # required by EmpiricalBayesTeamPrior
    for col in ("entity_id", "season", "metric", "value", "sample_size"):
        assert col in hist.columns
    assert (hist["season"] <= 2025).all()  # no target-season leakage
    assert "offense_epa_per_play" in set(hist["metric"])
    assert "defense_allowed_epa_per_play" in set(hist["metric"])
    # engine accepts it and produces 2026 priors
    prior = EmpiricalBayesTeamPrior(TeamPriorConfig(target_season=2026)).build(
        hist, target_season=2026, as_of_utc="2026-08-01T00:00:00Z"
    )
    assert len(prior) > 0
    assert prior["entity_id"].nunique() == 2


def test_team_history_available_before_asof():
    pbp = _synthetic_pbp()
    hist = build_team_metric_history(pbp, "h")
    avail = pd.to_datetime(hist["available_at_utc"], utc=True)
    asof = pd.Timestamp("2026-08-01T00:00:00Z")
    assert (avail <= asof).all()
    # each season's stats become available only after that season ends
    for season in hist["season"].unique():
        s_avail = pd.to_datetime(hist[hist["season"] == season]["available_at_utc"].iloc[0], utc=True)
        assert s_avail.year == season + 1


def test_quarterback_history_schema_and_engine_compat():
    pbp = _synthetic_pbp()
    qb = build_quarterback_history(pbp, pd.DataFrame(), "h")
    for col in ("player_id", "value", "dropbacks", "recency_dropbacks", "years_experience"):
        assert col in qb.columns
    # recency_dropbacks is 0 for the latest season and positive for earlier ones
    latest = qb.sort_values("season").groupby("player_id").tail(1)
    assert (latest["recency_dropbacks"] == 0).all()
    players = QuarterbackPriorBuilder().build(qb, as_of_utc="2026-08-01T00:00:00Z")
    assert len(players) > 0
