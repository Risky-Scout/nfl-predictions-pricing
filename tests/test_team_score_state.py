import pandas as pd
import pytest

from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.features import team_score_state as tss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _games(rows):
    """rows: list of dicts with game_id, season, season_type, week,
    home_team_id, away_team_id, scheduled_kickoff_utc (str), home_score,
    away_score (None if not yet played)."""
    df = pd.DataFrame(rows)
    df["scheduled_kickoff_utc"] = pd.to_datetime(df["scheduled_kickoff_utc"], utc=True)
    if "neutral_site" not in df:
        df["neutral_site"] = False
    return df


def _week_apart(base: str, n: int) -> str:
    return (pd.Timestamp(base) + pd.Timedelta(weeks=n)).isoformat()


def _state_row(out: pd.DataFrame, game_id: str, team_id: str) -> pd.Series:
    return out[(out["game_id"] == game_id) & (out["team_id"] == team_id)].iloc[0]


# ---------------------------------------------------------------------------
# Synthetic multi-team scenario used by several formula-verification tests
# below.
#
# Week 1 (season 2024): KC beats BUF 24-17 (G1); SF beats LAR 20-10 (G2).
# By week 2's TUE cutoff both results are long available, so the shared
# league_points_per_team_game_mean = (24+17+20+10) / 4 = 17.75.
#   KC: 1 prior game, pf_mean=24, pa_mean=17
#     offense_score_deviation = 24 - 17.75 = 6.25
#     defense_allow_deviation = 17 - 17.75 = -0.75
#   DEN: 0 prior games -> both deviations 0, missing=1
# Week 2: DEN (home) vs KC (away) (G3), not yet played.
# ---------------------------------------------------------------------------
def _synthetic_games():
    rows = [
        {"game_id": "G1", "season": 2024, "season_type": "REG", "week": 1, "home_team_id": "KC", "away_team_id": "BUF",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 0), "home_score": 24, "away_score": 17},
        {"game_id": "G2", "season": 2024, "season_type": "REG", "week": 1, "home_team_id": "SF", "away_team_id": "LAR",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 0), "home_score": 20, "away_score": 10},
        {"game_id": "G3", "season": 2024, "season_type": "REG", "week": 2, "home_team_id": "DEN", "away_team_id": "KC",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 1), "home_score": None, "away_score": None},
    ]
    return _games(rows)


# ---------------------------------------------------------------------------
# Expanding mean accumulation + league mean calculation
# ---------------------------------------------------------------------------
def test_expanding_points_mean_accumulates_correctly():
    games = _synthetic_games()
    out = tss.build_team_score_state(games, "TUE")
    kc_g3 = _state_row(out, "G3", "KC")
    assert kc_g3["score_state_games_played"] == 1
    assert kc_g3["team_points_for_mean"] == pytest.approx(24.0)
    assert kc_g3["team_points_against_mean"] == pytest.approx(17.0)
    assert kc_g3["score_state_missing"] == 0


def test_league_mean_calculation():
    games = _synthetic_games()
    out = tss.build_team_score_state(games, "TUE")
    kc_g3 = _state_row(out, "G3", "KC")
    den_g3 = _state_row(out, "G3", "DEN")
    expected_league_mean = (24 + 17 + 20 + 10) / 4
    assert kc_g3["league_points_per_team_game_mean"] == pytest.approx(expected_league_mean)
    # Shared identically by both sides of the same matchup at the same cutoff.
    assert den_g3["league_points_per_team_game_mean"] == pytest.approx(expected_league_mean)


def test_offense_sign():
    games = _synthetic_games()
    out = tss.build_team_score_state(games, "TUE")
    kc_g3 = _state_row(out, "G3", "KC")
    league_mean = kc_g3["league_points_per_team_game_mean"]
    assert kc_g3["offense_score_deviation"] == pytest.approx(24.0 - league_mean)
    assert kc_g3["offense_score_deviation"] > 0  # KC's pf_mean (24) exceeds the league mean (17.75)


def test_defense_allow_sign():
    games = _synthetic_games()
    out = tss.build_team_score_state(games, "TUE")
    kc_g3 = _state_row(out, "G3", "KC")
    league_mean = kc_g3["league_points_per_team_game_mean"]
    assert kc_g3["defense_allow_deviation"] == pytest.approx(17.0 - league_mean)
    assert kc_g3["defense_allow_deviation"] < 0  # KC's pa_mean (17) is below the league mean (17.75) -- good defense


# ---------------------------------------------------------------------------
# Neutral missing state
# ---------------------------------------------------------------------------
def test_neutral_missing_state_zero_prior_games():
    games = _synthetic_games()
    out = tss.build_team_score_state(games, "TUE")
    den_g3 = _state_row(out, "G3", "DEN")
    assert den_g3["score_state_missing"] == 1
    assert den_g3["score_state_games_played"] == 0
    assert den_g3["offense_score_deviation"] == 0.0
    assert den_g3["defense_allow_deviation"] == 0.0
    assert den_g3["team_points_for_mean"] == 0.0
    assert den_g3["team_points_against_mean"] == 0.0


# ---------------------------------------------------------------------------
# Matchup formulas: home/away expected deviation, margin signal, total signal
# ---------------------------------------------------------------------------
def test_home_expected_score_deviation_formula():
    games = _synthetic_games()
    matrix = tss.build_team_score_feature_matrix(games, "TUE")
    row = matrix[matrix["game_id"] == "G3"].iloc[0]
    # home=DEN (offense_dev=0, missing), away=KC (defense_dev=-0.75)
    expected = 0.5 * (row["home_offense_score_deviation"] + row["away_defense_allow_deviation"])
    assert row["home_expected_score_deviation"] == pytest.approx(expected)
    assert row["home_expected_score_deviation"] == pytest.approx(0.5 * (0.0 + (17.0 - 17.75)))


def test_away_expected_score_deviation_formula():
    games = _synthetic_games()
    matrix = tss.build_team_score_feature_matrix(games, "TUE")
    row = matrix[matrix["game_id"] == "G3"].iloc[0]
    expected = 0.5 * (row["away_offense_score_deviation"] + row["home_defense_allow_deviation"])
    assert row["away_expected_score_deviation"] == pytest.approx(expected)
    assert row["away_expected_score_deviation"] == pytest.approx(0.5 * ((24.0 - 17.75) + 0.0))


def test_margin_signal_formula():
    games = _synthetic_games()
    matrix = tss.build_team_score_feature_matrix(games, "TUE")
    row = matrix[matrix["game_id"] == "G3"].iloc[0]
    expected = row["home_expected_score_deviation"] - row["away_expected_score_deviation"]
    assert row["score_state_margin_signal"] == pytest.approx(expected)
    assert row["score_state_margin_signal"] == pytest.approx(-3.5)


def test_total_signal_formula():
    games = _synthetic_games()
    matrix = tss.build_team_score_feature_matrix(games, "TUE")
    row = matrix[matrix["game_id"] == "G3"].iloc[0]
    expected = row["home_expected_score_deviation"] + row["away_expected_score_deviation"]
    assert row["score_state_total_signal"] == pytest.approx(expected)
    assert row["score_state_total_signal"] == pytest.approx(2.75)


def test_matrix_missing_flags_propagate_per_side():
    games = _synthetic_games()
    matrix = tss.build_team_score_feature_matrix(games, "TUE")
    row = matrix[matrix["game_id"] == "G3"].iloc[0]
    assert row["home_score_state_missing"] == 1  # DEN
    assert row["away_score_state_missing"] == 0  # KC


# ---------------------------------------------------------------------------
# Week 1 has no prior state
# ---------------------------------------------------------------------------
def test_week1_has_no_prior_state():
    rows = [
        {"game_id": "G1", "season": 2024, "season_type": "REG", "week": 1, "home_team_id": "KC", "away_team_id": "BUF",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 0), "home_score": None, "away_score": None},
    ]
    games = _games(rows)
    out = tss.build_team_score_state(games, "TUE")
    kc = out[out["team_id"] == "KC"].iloc[0]
    assert kc["score_state_missing"] == 1
    assert kc["score_state_games_played"] == 0
    assert kc["league_points_per_team_game_mean"] == 0.0  # no eligible league observations exist yet either


# ---------------------------------------------------------------------------
# STRICT availability boundary -- an event's own game may never update its
# own state ("own result excluded"), and a not-yet-available prior result
# must not leak in.
# ---------------------------------------------------------------------------
def test_own_result_excluded_and_target_never_updates_itself():
    rows = [
        {"game_id": "G1", "season": 2024, "season_type": "REG", "week": 1, "home_team_id": "KC", "away_team_id": "BUF",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 0), "home_score": 24, "away_score": 17},
        {"game_id": "G2", "season": 2024, "season_type": "REG", "week": 2, "home_team_id": "KC", "away_team_id": "DEN",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 1), "home_score": 30, "away_score": 10},
    ]
    games = _games(rows)
    out = tss.build_team_score_state(games, "TUE")
    kc_g1 = _state_row(out, "G1", "KC")
    assert kc_g1["score_state_missing"] == 1  # G1's own result never feeds G1's own state
    kc_g2 = _state_row(out, "G2", "KC")
    assert kc_g2["score_state_games_played"] == 1  # only G1, not G2 itself
    assert kc_g2["team_points_for_mean"] == pytest.approx(24.0)


def test_strict_boundary_excludes_result_available_exactly_at_cutoff_equality_excluded(monkeypatch):
    # Force result_available_at_utc to land EXACTLY on week 2's TUE cutoff --
    # STRICT '<' must exclude it (equality excluded).
    rows = [
        {"game_id": "G1", "season": 2024, "season_type": "REG", "week": 1, "home_team_id": "KC", "away_team_id": "BUF",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 0), "home_score": 24, "away_score": 17},
        {"game_id": "G2", "season": 2024, "season_type": "REG", "week": 2, "home_team_id": "KC", "away_team_id": "DEN",
         "scheduled_kickoff_utc": _week_apart("2024-09-08T17:00:00Z", 1), "home_score": 30, "away_score": 10},
    ]
    games = _games(rows)
    ledger = he.build_horizon_membership_ledger(games)
    exact_cutoff = ledger.set_index("game_id").loc["G2", "tue_cutoff_utc"]

    def fake_result_available_at(work):
        out = work["scheduled_kickoff_utc"].copy()
        out.iloc[0] = exact_cutoff  # G1's availability == G2's TUE cutoff exactly
        return out

    monkeypatch.setattr(tss, "compute_result_available_at_utc", fake_result_available_at)
    out = tss.build_team_score_state(games, "TUE", membership_ledger=ledger)
    kc_g2 = _state_row(out, "G2", "KC")
    assert kc_g2["score_state_missing"] == 1
    assert kc_g2["score_state_games_played"] == 0


# ---------------------------------------------------------------------------
# Season boundary reset -- no cross-season carryover.
# ---------------------------------------------------------------------------
def test_season_boundary_resets_state():
    rows = [
        {"game_id": "G1", "season": 2023, "season_type": "REG", "week": 1, "home_team_id": "KC", "away_team_id": "BUF",
         "scheduled_kickoff_utc": "2023-09-10T17:00:00Z", "home_score": 40, "away_score": 3},
        {"game_id": "G2", "season": 2024, "season_type": "REG", "week": 1, "home_team_id": "KC", "away_team_id": "DEN",
         "scheduled_kickoff_utc": "2024-09-08T17:00:00Z", "home_score": None, "away_score": None},
    ]
    games = _games(rows)
    out = tss.build_team_score_state(games, "TUE")
    kc_g2 = _state_row(out, "G2", "KC")
    assert kc_g2["score_state_missing"] == 1
    assert kc_g2["score_state_games_played"] == 0
    assert kc_g2["league_points_per_team_game_mean"] == 0.0  # 2023's league totals do not carry into 2024


# ---------------------------------------------------------------------------
# REG+POST target scope, REG->POST same-season carry
# ---------------------------------------------------------------------------
def test_reg_to_post_same_season_carry():
    rows = [
        {"game_id": "G1", "season": 2024, "season_type": "REG", "week": 18, "home_team_id": "KC", "away_team_id": "BUF",
         "scheduled_kickoff_utc": "2025-01-05T17:00:00Z", "home_score": 20, "away_score": 13},
        {"game_id": "G2", "season": 2024, "season_type": "POST", "week": 19, "home_team_id": "KC", "away_team_id": "MIA",
         "scheduled_kickoff_utc": "2025-01-19T17:00:00Z", "home_score": None, "away_score": None},
    ]
    games = _games(rows)
    out = tss.build_team_score_state(games, "TUE")
    assert "G2" in set(out["game_id"])
    kc_post = _state_row(out, "G2", "KC")
    assert kc_post["score_state_games_played"] == 1  # the REG-season game carried into the POST target
    assert kc_post["team_points_for_mean"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# No forbidden data sources required or usable as features
# ---------------------------------------------------------------------------
def test_no_market_qb_epa_injury_weather_columns_required_or_produced():
    from nfl_hybrid.selection import feature_deduction_2026 as fd

    rows = [
        {"game_id": "G1", "season": 2024, "season_type": "REG", "week": 1, "home_team_id": "KC", "away_team_id": "BUF",
         "scheduled_kickoff_utc": "2024-09-08T17:00:00Z", "home_score": None, "away_score": None},
    ]
    games = _games(rows)  # deliberately has no market/QB/EPA/injury/weather columns at all
    out = tss.build_team_score_state(games, "TUE")
    assert len(out) == 2  # module works fine without those columns ever existing

    matrix_cols = tuple(tss.build_team_score_feature_matrix(games, "TUE").columns)
    assert set(matrix_cols) & fd.FORBIDDEN_MARKET_COLUMNS == set()
    fd.assert_no_forbidden_market_columns(matrix_cols)  # must not raise
    assert not any("qb" in c.lower() or "epa" in c.lower() for c in matrix_cols)


# ---------------------------------------------------------------------------
# Candidate columns: exact contract, no extras, B is NOT a superset of C
# ---------------------------------------------------------------------------
def test_candidate_signal_columns_exact():
    assert tss.CANDIDATE_SIGNAL_COLUMNS == (
        "score_state_margin_signal", "score_state_total_signal",
        "home_score_state_missing", "away_score_state_missing",
    )


def test_candidate_component_columns_exact():
    assert tss.CANDIDATE_COMPONENT_COLUMNS == (
        "home_offense_score_deviation", "home_defense_allow_deviation",
        "away_offense_score_deviation", "away_defense_allow_deviation",
        "home_score_state_missing", "away_score_state_missing",
    )


def test_candidate_b_is_not_a_superset_relationship():
    # Deliberately different this time from V0: B is not a subset of C.
    signal_only = set(tss.CANDIDATE_SIGNAL_COLUMNS) - set(tss.CANDIDATE_COMPONENT_COLUMNS)
    assert signal_only == {"score_state_margin_signal", "score_state_total_signal"}


# ---------------------------------------------------------------------------
# Pivoted matrix structure
# ---------------------------------------------------------------------------
def test_feature_matrix_has_exactly_the_contracted_columns():
    games = _synthetic_games()
    matrix = tss.build_team_score_feature_matrix(games, "TUE")
    expected = {
        "game_id",
        "home_offense_score_deviation", "home_defense_allow_deviation", "home_score_state_missing",
        "away_offense_score_deviation", "away_defense_allow_deviation", "away_score_state_missing",
        "home_expected_score_deviation", "away_expected_score_deviation",
        "score_state_margin_signal", "score_state_total_signal",
    }
    assert set(matrix.columns) == expected


# ---------------------------------------------------------------------------
# Structural existence checks (contract section 4)
# ---------------------------------------------------------------------------
def test_structural_fields_exist():
    games = _synthetic_games()
    state = tss.build_team_score_state(games, "TUE")
    for col in ("league_points_per_team_game_mean", "offense_score_deviation", "defense_allow_deviation"):
        assert col in state.columns

    matrix = tss.build_team_score_feature_matrix(games, "TUE")
    for col in ("score_state_margin_signal", "score_state_total_signal"):
        assert col in matrix.columns


# ---------------------------------------------------------------------------
# Semantics hash determinism
# ---------------------------------------------------------------------------
def test_semantics_hash_deterministic():
    a = tss.compute_semantics_hash()
    b = tss.compute_semantics_hash(tss.team_score_state_v1_1_semantics())
    assert a == b
    assert len(a) == 64
