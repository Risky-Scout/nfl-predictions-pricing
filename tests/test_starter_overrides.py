import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.priors.populate_templates import build_projected_starters, _rank_from_depth
from nfl_hybrid.priors.starter_overrides import (
    apply_overrides,
    flag_review,
    league_centered_fallback,
    load_overrides,
    empty_override_frame,
)


def _depth():
    # depth_team holds the rank as a string (pos_rank is empty), reproducing the defect
    return pd.DataFrame({
        "season": [2024, 2024, 2024, 2024],
        "week": [1, 1, 1, 1],
        "position": ["QB", "QB", "QB", "QB"],
        "club_code": ["KC", "KC", "BUF", "BUF"],
        "gsis_id": ["mahomes", "wentz", "allen", "trubisky"],
        "full_name": ["P Mahomes", "C Wentz", "J Allen", "M Trubisky"],
        "pos_rank": [np.nan, np.nan, np.nan, np.nan],
        "depth_team": ["1", "2", "1", "2"],
    })


def test_rank_from_depth_uses_depth_team_when_pos_rank_empty():
    qb = _depth()
    rank = _rank_from_depth(qb)
    assert rank.tolist() == [1.0, 2.0, 1.0, 2.0]


def test_projected_starters_sum_to_one_per_team():
    proj = build_projected_starters(_depth(), target_season=2026, as_of_utc="2026-08-01T00:00:00Z")
    sums = proj.groupby("team_id")["starter_probability"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-9)
    # starter has the highest probability
    kc = proj[proj["team_id"] == "KC"].sort_values("starter_probability", ascending=False)
    assert kc.iloc[0]["player_id"] == "mahomes"
    # no mechanical 0.02 constant
    assert proj["starter_probability"].nunique() > 1


def test_override_applied_and_renormalized():
    gen = pd.DataFrame({
        "team_id": ["KC", "KC"], "player_id": ["mahomes", "wentz"],
        "starter_probability": [0.88, 0.12], "season": [2026, 2026],
    })
    ov = empty_override_frame()
    ov.loc[0] = ["KC", "wentz", 0.9, "beat_writer", "2026-08-20T00:00:00Z",
                 "2026-08-20T00:00:00Z", "2026-12-01T00:00:00Z", "Mahomes injured (test)"]
    out = apply_overrides(gen, ov, as_of_utc="2026-08-21T00:00:00Z")
    assert np.isclose(out.groupby("team_id")["starter_probability"].sum().iloc[0], 1.0)
    # override row present and marked
    assert (out["origin"] == "override").any()
    wentz = out[out["player_id"] == "wentz"].iloc[0]
    assert wentz["starter_probability"] > 0.5  # override dominates


def test_expired_override_ignored():
    gen = pd.DataFrame({"team_id": ["KC"], "player_id": ["mahomes"], "starter_probability": [1.0], "season": [2026]})
    ov = empty_override_frame()
    ov.loc[0] = ["KC", "backup", 0.9, "x", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", "old"]
    out = apply_overrides(gen, ov, as_of_utc="2026-08-01T00:00:00Z")
    assert (out["origin"] == "generated").all()  # expired override dropped


def test_missing_prior_qb_retained_via_fallback():
    priors = pd.DataFrame({"player_id": ["vet"], "prior_mean": [0.1],
                           "prior_standard_deviation": [0.15], "effective_dropbacks": [900.0]})
    fb = league_centered_fallback(["rookie1", "rookie2"], priors)
    assert set(fb["player_id"]) == {"rookie1", "rookie2"}
    # widened uncertainty vs the historical prior
    assert (fb["prior_standard_deviation"] > 0.15).all()
    assert (fb["prior_source"] == "league_centered_fallback").all()


def test_injury_triggers_needs_review():
    starters = pd.DataFrame({
        "team_id": ["KC", "KC", "BUF"],
        "player_id": ["mahomes", "wentz", "allen"],
        "starter_probability": [0.88, 0.12, 1.0],
    })
    availability = pd.DataFrame({"gsis_id": ["mahomes"], "report_status": ["Out"]})
    flagged = flag_review(starters, availability)
    kc_status = flagged[flagged["team_id"] == "KC"]["qb_review_status"].iloc[0]
    buf_status = flagged[flagged["team_id"] == "BUF"]["qb_review_status"].iloc[0]
    assert kc_status == "NEEDS_QB_REVIEW"  # top candidate is Out
    assert buf_status == "OK"


def test_load_overrides_missing_file_returns_empty():
    df = load_overrides("does/not/exist.csv")
    assert len(df) == 0
    assert list(df.columns)  # schema present
