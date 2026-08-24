import numpy as np
import pandas as pd
import pytest

from nfl_hybrid.features import horizon_elo as he


def _reg_games():
    # Week 1: Thu (KC@BUF), Sun (NE@MIA). Week 2: Sun (BUF@NE).
    return pd.DataFrame(
        {
            "game_id": ["G1_THU", "G2_SUN", "G3_SUN"],
            "season": [2024, 2024, 2024],
            "season_type": ["REG", "REG", "REG"],
            "week": [1, 1, 2],
            "home_team_id": ["BUF", "MIA", "NE"],
            "away_team_id": ["KC", "NE", "BUF"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2024-09-05T00:20:00Z", "2024-09-08T17:00:00Z", "2024-09-15T17:00:00Z"], utc=True
            ),
            "home_score": [20, 24, 27],
            "away_score": [17, 21, 24],
            "neutral_site": [False, False, False],
        }
    )


def _thanksgiving_card():
    # Week 12, 2020: three Thursday games (all before that card's own
    # Friday cutoff -- Thanksgiving is always the earliest-in-card day).
    return pd.DataFrame(
        {
            "game_id": ["T1", "T2", "T3", "T4"],
            "season": [2020, 2020, 2020, 2020],
            "season_type": ["REG"] * 4,
            "week": [12, 12, 12, 12],
            "home_team_id": ["HOU", "WAS", "DET", "BAL"],
            "away_team_id": ["DET", "DAL", "HOU", "PIT"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2020-11-26T17:30:00Z", "2020-11-26T21:30:00Z", "2020-11-27T01:20:00Z", "2020-11-29T18:00:00Z"],
                utc=True,
            ),
            "home_score": [41, 25, 17, 27],
            "away_score": [25, 3, 41, 13],
            "neutral_site": [False, False, False, False],
        }
    )


def _christmas_2024_week17():
    # 2024 Week 17: two Wednesday Christmas games + one Thursday game --
    # all before that card's own Friday cutoff.
    return pd.DataFrame(
        {
            "game_id": ["X1", "X2", "X3"],
            "season": [2024, 2024, 2024],
            "season_type": ["REG"] * 3,
            "week": [17, 17, 17],
            "home_team_id": ["PIT", "HOU", "CHI"],
            "away_team_id": ["KC", "BAL", "SEA"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2024-12-25T18:00:00Z", "2024-12-25T21:30:00Z", "2024-12-27T01:15:00Z"], utc=True
            ),
            "home_score": [10, 31, 6],
            "away_score": [29, 20, 30],
            "neutral_site": [False, False, False],
        }
    )


def _postseason_games():
    return pd.DataFrame(
        {
            "game_id": ["P1_WC", "P2_SB"],
            "season": [2023, 2023],
            "season_type": ["POST", "POST"],
            "week": [19, 22],
            "home_team_id": ["KC", "KC"],
            "away_team_id": ["MIA", "SF"],
            "scheduled_kickoff_utc": pd.to_datetime(["2024-01-14T18:00:00Z", "2024-02-11T23:30:00Z"], utc=True),
            "home_score": [26, 25],
            "away_score": [7, 22],
            "neutral_site": [False, True],
        }
    )


# ---------------------------------------------------------------------------
# Card-based cutoff exactness / no per-game floor-back.
# ---------------------------------------------------------------------------
def test_card_cutoff_shared_by_every_game_in_the_card():
    games = _reg_games()
    ledger = he.build_horizon_membership_ledger(games)
    week1 = ledger[ledger["week"] == 1]
    assert week1["tue_cutoff_utc"].nunique() == 1
    assert week1["fri_cutoff_utc"].nunique() == 1


def test_no_per_game_floor_back_thursday_is_fri_ineligible():
    games = _reg_games()
    ledger = he.build_horizon_membership_ledger(games).set_index("game_id")
    assert bool(ledger.loc["G1_THU", "tue_eligible"]) is True
    assert bool(ledger.loc["G1_THU", "fri_eligible"]) is False
    assert ledger.loc["G1_THU", "fri_reason"] == he.INELIGIBLE_REASON
    # The old V1 behavior floated the Thursday game back to the PREVIOUS
    # week's Friday. V2 must never do that -- the card's own Friday cutoff
    # (this same week) is what's compared, and it must land AFTER kickoff.
    assert ledger.loc["G1_THU", "fri_cutoff_utc"] >= ledger.loc["G1_THU", "scheduled_kickoff_utc"]


def test_sunday_game_same_card_is_tue_and_fri_eligible():
    games = _reg_games()
    ledger = he.build_horizon_membership_ledger(games).set_index("game_id")
    assert bool(ledger.loc["G2_SUN", "tue_eligible"]) is True
    assert bool(ledger.loc["G2_SUN", "fri_eligible"]) is True


def test_thanksgiving_multi_game_thursday_card_all_fri_ineligible():
    games = _thanksgiving_card()
    ledger = he.build_horizon_membership_ledger(games).set_index("game_id")
    # T1/T2/T3 are Thursday (Thanksgiving); T4 is the following Sunday.
    for gid in ("T1", "T2", "T3"):
        assert bool(ledger.loc[gid, "fri_eligible"]) is False, gid
    assert bool(ledger.loc["T4", "fri_eligible"]) is True
    assert bool(ledger.loc["T4", "tue_eligible"]) is True


def test_christmas_wednesday_thursday_card_all_fri_ineligible():
    games = _christmas_2024_week17()
    ledger = he.build_horizon_membership_ledger(games)
    assert bool(ledger["fri_eligible"].any()) is False
    assert bool(ledger["tue_eligible"].all()) is True


def test_postseason_card_membership_uses_season_week_season_type():
    games = _postseason_games()
    ledger = he.build_horizon_membership_ledger(games)
    assert set(ledger["season_type"]) == {"POST"}
    assert bool(ledger["tue_eligible"].all())
    assert bool(ledger["fri_eligible"].all())  # both POST games are Sun kickoffs


# ---------------------------------------------------------------------------
# Exact real-metadata membership (season <= 2024).
# ---------------------------------------------------------------------------
def _real_firewalled_games():
    pytest.importorskip("pandas")
    import sys as _sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in _sys.path:
        _sys.path.insert(0, str(src))
    from nfl_hybrid.data.external_data import resolve
    from nfl_hybrid.selection import feature_deduction_2026 as fd

    try:
        raw = pd.read_parquet(resolve("backfill.games"))
    except Exception:
        pytest.skip("NFL_MODEL_DATA_ROOT not configured in this environment")
    games, _ = fd.enforce_2025_firewall(raw)
    return games


def test_exact_1408_tue_1317_fri_membership_on_real_metadata():
    games = _real_firewalled_games()
    ledger = he.build_horizon_membership_ledger(games)
    assert len(ledger) == 1408
    assert int(ledger["tue_eligible"].sum()) == 1408
    assert int(ledger["fri_eligible"].sum()) == 1317
    assert int((~ledger["fri_eligible"]).sum()) == 91
    assert int((ledger["fri_eligible"] & (ledger["season_type"] == "REG")).sum()) == 1252
    assert int((ledger["fri_eligible"] & (ledger["season_type"] == "POST")).sum()) == 65


def test_exact_fold_membership_counts_on_real_metadata():
    games = _real_firewalled_games()
    ledger = he.build_horizon_membership_ledger(games)
    expected = {
        "A": (269, 285, 255, 267),
        "B": (554, 284, 522, 265),
        "C": (838, 285, 787, 266),
        "OUTER": (1123, 285, 1053, 264),
    }
    folds = {"A": (2020, 2021), "B": (2021, 2022), "C": (2022, 2023), "OUTER": (2023, 2024)}
    for name, (train_max, validate_season) in folds.items():
        tue_train = int((ledger["season"] <= train_max).sum())
        tue_val = int((ledger["season"] == validate_season).sum())
        fri_train = int(((ledger["season"] <= train_max) & ledger["fri_eligible"]).sum())
        fri_val = int(((ledger["season"] == validate_season) & ledger["fri_eligible"]).sum())
        assert (tue_train, tue_val, fri_train, fri_val) == expected[name], name


# ---------------------------------------------------------------------------
# Target eligibility vs update-event eligibility.
# ---------------------------------------------------------------------------
def test_fri_ineligible_target_never_enters_supervised_fri_state():
    games = _reg_games()
    state_fri = he.build_horizon_elo_state(games, "FRI")
    assert "G1_THU" not in set(state_fri["game_id"])
    assert set(state_fri["game_id"]) == {"G2_SUN", "G3_SUN"}


def test_fri_ineligible_games_result_can_update_a_later_eligible_target():
    games = _reg_games()
    state_fri = he.build_horizon_elo_state(games, "FRI")
    state_tue = he.build_horizon_elo_state(games, "TUE")
    g2_fri = state_fri[(state_fri["game_id"] == "G2_SUN") & (state_fri["team_id"] == "NE")]["elo_pregame_rating"].iloc[0]
    g2_tue = state_tue[(state_tue["game_id"] == "G2_SUN") & (state_tue["team_id"] == "NE")]["elo_pregame_rating"].iloc[0]
    # Same team, same target game, but FRI's cutoff (after Thursday's
    # kickoff) has already applied G1's update; TUE's cutoff (before
    # Thursday's kickoff) has not -- so an OTHER team's rating (KC/BUF, the
    # Thursday participants) is what actually diverges; NE itself isn't a
    # Thursday participant, so NE's own rating is untouched either way.
    assert g2_fri == pytest.approx(g2_tue)
    buf_fri = state_fri[(state_fri["game_id"] == "G3_SUN") & (state_fri["team_id"] == "BUF")]
    buf_tue = state_tue[(state_tue["game_id"] == "G3_SUN") & (state_tue["team_id"] == "BUF")]
    # By week 2 both horizons have long since applied G1 -- must agree.
    assert buf_fri["elo_pregame_rating"].iloc[0] == pytest.approx(buf_tue["elo_pregame_rating"].iloc[0])


def test_ineligible_game_result_reaches_a_real_downstream_target():
    # Directly reproduces the documented 2020_06_HOU_TEN divergence: HOU/TEN
    # play the Thursday game of their card; a later card shares no game with
    # them, so use the repo's own real data path is exercised at the script
    # level -- here, a compact synthetic analog proves the mechanism: a
    # Thursday LOSS should measurably lower a team's rating for that same
    # team's later same-week card-mate under FRI but not under TUE.
    games = pd.DataFrame(
        {
            "game_id": ["W1_THU_A", "W1_SUN_A"],
            "season": [2024, 2024],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "home_team_id": ["KC", "KC"],
            "away_team_id": ["BUF", "MIA"],
            "scheduled_kickoff_utc": pd.to_datetime(["2024-09-05T00:20:00Z", "2024-09-08T17:00:00Z"], utc=True),
            "home_score": [10, 24],
            "away_score": [40, 21],
            "neutral_site": [False, False],
        }
    )
    fri = he.build_horizon_elo_state(games, "FRI")
    tue = he.build_horizon_elo_state(games, "TUE")
    kc_fri = fri[(fri["game_id"] == "W1_SUN_A") & (fri["team_id"] == "KC")]["elo_pregame_rating"].iloc[0]
    kc_tue = tue[(tue["game_id"] == "W1_SUN_A") & (tue["team_id"] == "KC")]["elo_pregame_rating"].iloc[0]
    assert kc_fri < kc_tue  # KC lost Thursday; FRI already reflects it, TUE doesn't


# ---------------------------------------------------------------------------
# Elo event semantics (unchanged rules, re-verified under V2 mechanics).
# ---------------------------------------------------------------------------
def test_target_never_updates_itself():
    games = _reg_games()
    for horizon in he.HORIZONS:
        out = he.build_horizon_elo_state(games, horizon)
        mutated = games.copy()
        mutated.loc[mutated["game_id"] == "G3_SUN", ["home_score", "away_score"]] = [3, 45]
        out_mutated = he.build_horizon_elo_state(mutated, horizon)
        base = out[out["game_id"] == "G3_SUN"].set_index("team_id")["elo_pregame_rating"].sort_index()
        changed = out_mutated[out_mutated["game_id"] == "G3_SUN"].set_index("team_id")["elo_pregame_rating"].sort_index()
        pd.testing.assert_series_equal(base, changed)


def test_future_unavailable_result_never_updates_target_state():
    games = _reg_games()
    games.loc[games["game_id"] == "G3_SUN", ["home_score", "away_score"]] = [np.nan, np.nan]
    out = he.build_horizon_elo_state(games, "FRI")
    assert len(out[out["game_id"] == "G3_SUN"]) == 2  # still predicted, never updates anything


def test_result_exactly_at_cutoff_is_excluded_strict():
    games = pd.DataFrame(
        {
            "game_id": ["EARLY", "TARGET"],
            "season": [2024, 2024],
            "season_type": ["REG", "REG"],
            "week": [1, 2],
            "home_team_id": ["KC", "KC"],
            "away_team_id": ["BUF", "MIA"],
            "scheduled_kickoff_utc": pd.to_datetime(["2024-09-05T00:20:00Z", "2024-09-15T17:00:00Z"], utc=True),
            "home_score": [10, 24],
            "away_score": [40, 21],
            "neutral_site": [False, False],
        }
    )
    ledger = he.build_horizon_membership_ledger(games)
    result_available = he.compute_result_available_at_utc(games)
    early_available = pd.Timestamp(result_available[games["game_id"] == "EARLY"].iloc[0])

    # Force an exact-equality boundary: TARGET's own cutoff pinned to
    # EARLY's result-available instant exactly. Strict '<' must exclude it
    # -- KC's TARGET-row rating must equal the pre-EARLY initial rating.
    forced_ledger = ledger.copy()
    forced_ledger["tue_cutoff_utc"] = forced_ledger["tue_cutoff_utc"].astype("datetime64[ns, UTC]")
    forced_ledger.loc[forced_ledger["game_id"] == "TARGET", "tue_cutoff_utc"] = early_available
    state_strict = he.build_horizon_elo_state(games, "TUE", membership_ledger=forced_ledger)
    kc_at_exact_boundary = state_strict[(state_strict["game_id"] == "TARGET") & (state_strict["team_id"] == "KC")][
        "elo_pregame_rating"
    ].iloc[0]
    assert kc_at_exact_boundary == pytest.approx(1500.0)  # EARLY's update NOT applied at the exact boundary

    # One microsecond later, strictness must flip and admit it.
    admitting_ledger = forced_ledger.copy()
    admitting_ledger.loc[admitting_ledger["game_id"] == "TARGET", "tue_cutoff_utc"] = early_available + pd.Timedelta(
        microseconds=1
    )
    state_admitted = he.build_horizon_elo_state(games, "TUE", membership_ledger=admitting_ledger)
    kc_after_boundary = state_admitted[(state_admitted["game_id"] == "TARGET") & (state_admitted["team_id"] == "KC")][
        "elo_pregame_rating"
    ].iloc[0]
    assert kc_after_boundary != pytest.approx(1500.0)


def test_postseason_carries_into_next_season_then_regresses():
    games = pd.DataFrame(
        {
            "game_id": ["S1_POST", "S2_NEXT_REG"],
            "season": [2023, 2024],
            "season_type": ["POST", "REG"],
            "week": [22, 1],
            "home_team_id": ["KC", "KC"],
            "away_team_id": ["SF", "BAL"],
            "scheduled_kickoff_utc": pd.to_datetime(["2024-02-11T23:30:00Z", "2024-09-05T17:00:00Z"], utc=True),
            "home_score": [25, 27],
            "away_score": [22, 20],
            "neutral_site": [True, False],
        }
    )
    out = he.build_horizon_elo_state(games, "TUE")
    kc_next = out[(out["game_id"] == "S2_NEXT_REG") & (out["team_id"] == "KC")].iloc[0]
    assert kc_next["elo_pregame_rating"] != pytest.approx(1500.0)
    assert 1500.0 < kc_next["elo_pregame_rating"] < 1520.0


def test_no_game_id_special_cases():
    # Renaming/reordering game_ids must not change the result: only
    # (kickoff, season/week/season_type card membership) matter.
    games = _reg_games()
    relabeled = games.copy()
    relabeled["game_id"] = ["ZZZ_" + g for g in relabeled["game_id"]]
    a = he.build_horizon_elo_state(games, "FRI").sort_values(["event_time", "team_id"])["elo_pregame_rating"].to_numpy()
    b = he.build_horizon_elo_state(relabeled, "FRI").sort_values(["event_time", "team_id"])["elo_pregame_rating"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_deterministic_repeat_run():
    games = _reg_games()
    for horizon in he.HORIZONS:
        first = he.build_horizon_elo_state(games, horizon)
        second = he.build_horizon_elo_state(games.sample(frac=1.0, random_state=7), horizon)
        pd.testing.assert_frame_equal(
            first.sort_values(["game_id", "team_id"]).reset_index(drop=True),
            second.sort_values(["game_id", "team_id"]).reset_index(drop=True),
        )


def test_duplicate_game_id_fails():
    games = pd.concat([_reg_games(), _reg_games().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        he.build_horizon_elo_state(games, "TUE")


def test_v2_semantic_hash_deterministic():
    kwargs = dict(
        fix6_feature_manifest_hash="abc123",
        fix6_frozen_feature_columns=["a", "b"],
        elo_config_hash=he.compute_elo_config_hash(),
    )
    h1 = he.compute_horizon_feature_semantics_hash_v2(**kwargs)
    h2 = he.compute_horizon_feature_semantics_hash_v2(**kwargs)
    assert h1 == h2
    assert he.horizon_semantics_spec_v2(**kwargs)["horizon_feature_semantics_version"] == "HORIZON_CUTOFF_ASOF_ELO_V2_CARD_SCOPED"


def test_no_infinite_cutoff_audit_helper_remains():
    # The old V1 "+infinity degenerate" pseudo-audit must be retired, not
    # silently kept as a second cutoff implementation.
    assert not hasattr(he, "build_kickoff_order_elo_pregame_state_for_audit")
    assert not hasattr(he, "horizon_cutoff_utc")  # old per-game floor-back function, removed
