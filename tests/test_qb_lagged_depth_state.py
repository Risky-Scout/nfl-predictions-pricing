import pandas as pd
import pytest

from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.features import qb_lagged_depth_state as qls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _games(season, weeks, *, team_a="KC", team_b="BUF", monday_night_week=None):
    """One (team_a vs team_b) game per week, one calendar week apart,
    Sunday 17:00 UTC kickoffs -- except `monday_night_week`, which kicks off
    the following day (Monday 01:00 UTC) to exercise card_end_utc = MAX
    kickoff in the card."""
    base = pd.Timestamp("2024-09-08T17:00:00Z")  # a Sunday
    rows = []
    for w in weeks:
        kickoff = base + pd.Timedelta(weeks=(season - 2024) * 22 + (w - 1))
        if w == monday_night_week:
            kickoff = kickoff + pd.Timedelta(hours=32)  # lands on Monday ~01:00 UTC
        rows.append(
            {
                "game_id": f"G_{season}_{w}",
                "season": season,
                "season_type": "REG",
                "week": w,
                "home_team_id": team_a,
                "away_team_id": team_b,
                "scheduled_kickoff_utc": kickoff,
                "home_score": None,
                "away_score": None,
                "neutral_site": False,
            }
        )
    return pd.DataFrame(rows)


def _historical_row(season, week, club_code, gsis_id, depth_team, *, game_type="REG", position="QB"):
    return {
        "season": season, "week": week, "game_type": game_type, "club_code": club_code,
        "position": position, "depth_team": depth_team, "gsis_id": gsis_id, "dt": None,
    }


def _historical_frame(rows):
    return pd.DataFrame(rows)


def _live_row(dt, team, gsis_id, pos_rank, *, pos_abb="QB"):
    return {"dt": dt, "team": team, "gsis_id": gsis_id, "pos_abb": pos_abb, "pos_rank": pos_rank}


def _live_frame(rows):
    return pd.DataFrame(rows)


def _card_table_and_chain(historical_rows, games):
    card_table = qls.build_card_table(games)
    normalized = qls.normalize_historical_card_states(_historical_frame(historical_rows), card_table)
    chain = qls.compute_card_chain_continuity(normalized)
    return card_table, chain


# ---------------------------------------------------------------------------
# 1. Historical weekly normalization
# ---------------------------------------------------------------------------
def test_historical_normalization_resolves_clean_qb1():
    games = _games(2024, [1, 2])
    rows = [
        _historical_row(2024, 1, "KC", "00-KC-A", "1"),
        _historical_row(2024, 1, "KC", "00-KC-B", "2"),
        _historical_row(2024, 1, "BUF", "00-BUF-A", "1"),
    ]
    card_table = qls.build_card_table(games)
    out = qls.normalize_historical_card_states(_historical_frame(rows), card_table)
    kc = out[(out["team_id"] == "KC") & (out["source_card_week"] == 1)].iloc[0]
    assert kc["qb1_player_id"] == "00-KC-A"
    assert kc["qb_state_resolution_status"] == qls.CARD_STATUS_RESOLVED
    assert kc["source_semantic_version"] == qls.HISTORICAL_SOURCE_SEMANTIC_VERSION


def test_historical_normalization_drops_spurious_week_game_type_combo():
    # A week/game_type combo with no matching real card (e.g. a stray REG
    # row at a week number that games.parquet only has as POST) must be
    # dropped by the inner join against card_table, not guessed into a card.
    games = _games(2024, [1])
    rows = [
        _historical_row(2024, 1, "KC", "00-KC-A", "1"),
        _historical_row(2024, 2, "KC", "00-KC-B", "1", game_type="REG"),  # week 2 has no card here
    ]
    card_table = qls.build_card_table(games)
    out = qls.normalize_historical_card_states(_historical_frame(rows), card_table)
    assert set(out["source_card_week"]) == {1}


# ---------------------------------------------------------------------------
# 2 & 5. Live rolling normalization + same-card state excluded
# ---------------------------------------------------------------------------
def test_live_normalization_uses_latest_snapshot_at_or_before_card_end_not_after():
    games = _games(2024, [1, 2])
    card_table = qls.build_card_table(games)
    week1_end = card_table.loc[card_table["source_card_week"] == 1, "card_end_utc"].iloc[0]

    rows = [
        _live_row((week1_end - pd.Timedelta(days=1)).isoformat(), "KC", "00-KC-A", 1),
        # Posted AFTER week 1's card ends but before week 2's -- belongs to
        # week 2's canonical state, must NOT leak into week 1's.
        _live_row((week1_end + pd.Timedelta(hours=2)).isoformat(), "KC", "00-KC-B", 1),
    ]
    out = qls.normalize_live_card_states(_live_frame(rows), card_table)
    wk1 = out[(out["team_id"] == "KC") & (out["source_card_week"] == 1)].iloc[0]
    wk2 = out[(out["team_id"] == "KC") & (out["source_card_week"] == 2)].iloc[0]
    assert wk1["qb1_player_id"] == "00-KC-A"
    assert wk2["qb1_player_id"] == "00-KC-B"


def test_lagged_target_never_uses_same_card_live_snapshot():
    # Contrast with qb_depthchart_asof.resolve_depthchart_qb1_asof: a
    # snapshot posted Tuesday morning of week 2 (before week 2's OWN TUE
    # cutoff) must never be used as week 2's own state -- only week 1's
    # (prior card) state may feed week 2's target.
    games = _games(2024, [1, 2])
    ledger = he.build_horizon_membership_ledger(games)
    card_table = qls.build_card_table(games)
    week2_cutoff = ledger.set_index("game_id").loc["G_2024_2", "tue_cutoff_utc"]

    rows = [
        # Posted before week 1's card ends -- this is week 1's own canonical
        # rolling state, and the only thing allowed to feed week 2's target.
        _live_row((card_table.loc[0, "card_end_utc"] - pd.Timedelta(hours=1)).isoformat(), "KC", "00-KC-WK1", 1),
        # Posted after week 1 ends but before week 2's OWN TUE cutoff --
        # eligible under QB_DEPTH_CHART_QB1_ASOF_V1's own-cutoff rule, but
        # must be excluded here (it is week 2's OWN card's rolling data, not
        # week 1's prior-card state).
        _live_row((week2_cutoff - pd.Timedelta(hours=1)).isoformat(), "KC", "00-KC-WK2", 1),
    ]
    normalized = qls.normalize_live_card_states(_live_frame(rows), card_table)
    chain = qls.compute_card_chain_continuity(normalized)
    out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain, membership_ledger=ledger)
    row = out[(out["game_id"] == "G_2024_2") & (out["team_id"] == "KC")].iloc[0]
    assert row["qb1_player_id"] == "00-KC-WK1"
    assert row["qb1_player_id"] != "00-KC-WK2"


# ---------------------------------------------------------------------------
# 3. Same normalized schema (structural bridge proof)
# ---------------------------------------------------------------------------
def test_schema_bridge_passes_for_both_adapters():
    games = _games(2024, [1, 2])
    card_table = qls.build_card_table(games)
    hist = qls.normalize_historical_card_states(
        _historical_frame([_historical_row(2024, 1, "KC", "00-A", "1")]), card_table
    )
    live = qls.normalize_live_card_states(
        _live_frame([_live_row(str(card_table.loc[0, "card_end_utc"]), "KC", "00-B", 1)]), card_table
    )
    proof = qls.assert_card_schema_bridge(hist, live)
    assert proof["schema_equal"] is True
    assert list(hist.columns) == list(qls.NORMALIZED_CARD_STATE_COLUMNS)
    assert list(live.columns) == list(qls.NORMALIZED_CARD_STATE_COLUMNS)


def test_schema_bridge_rejects_column_drift():
    bad = pd.DataFrame(columns=[*qls.NORMALIZED_CARD_STATE_COLUMNS, "extra_col"])
    good = pd.DataFrame(columns=qls.NORMALIZED_CARD_STATE_COLUMNS)
    with pytest.raises(qls.SchemaBridgeError):
        qls.assert_card_schema_bridge(bad, good)


# ---------------------------------------------------------------------------
# 4. Prior-card-only rule
# ---------------------------------------------------------------------------
def test_target_uses_prior_card_state_not_own_week():
    games = _games(2024, [1, 2, 3])
    rows = [
        _historical_row(2024, 1, "KC", "00-KC-A", "1"),
        _historical_row(2024, 2, "KC", "00-KC-B", "1"),  # week 2's OWN state -- must not feed week 2's target
    ]
    card_table, chain = _card_table_and_chain(rows, games)
    out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain)
    week2_kc = out[(out["game_id"] == "G_2024_2") & (out["team_id"] == "KC")].iloc[0]
    assert week2_kc["qb1_player_id"] == "00-KC-A"
    assert week2_kc["prior_card_week"] == 1


# ---------------------------------------------------------------------------
# 6. Strict availability boundary
# ---------------------------------------------------------------------------
def test_strict_availability_boundary_not_yet_available_at_equality():
    games = _games(2024, [1, 2])
    card_table = qls.build_card_table(games)
    ledger = he.build_horizon_membership_ledger(games)
    week2_cutoff = ledger.set_index("game_id").loc["G_2024_2", "tue_cutoff_utc"]

    chain = pd.DataFrame(
        [
            {
                "season": 2024, "source_card_week": 1, "team_id": "KC", "qb1_player_id": "00-KC-A",
                "qb_state_available_at_utc": week2_cutoff,  # exactly equal -- STRICT < must fail
                "qb_state_resolution_status": qls.CARD_STATUS_RESOLVED,
                "source_semantic_version": qls.HISTORICAL_SOURCE_SEMANTIC_VERSION,
                "qb_depth_changed_at_card": 0, "qb_depth_continuity_cards_at_card": 1,
            }
        ]
    )
    out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain, membership_ledger=ledger)
    row = out[(out["game_id"] == "G_2024_2") & (out["team_id"] == "KC")].iloc[0]
    assert row["lagged_resolution_status"] == qls.TARGET_STATUS_STATE_NOT_YET_AVAILABLE
    assert row["qb_depth_state_missing"] == 1

    chain_ok = chain.copy()
    chain_ok["qb_state_available_at_utc"] = week2_cutoff - pd.Timedelta(seconds=1)
    out_ok = qls.resolve_lagged_target_state(games, "TUE", card_table, chain_ok, membership_ledger=ledger)
    row_ok = out_ok[(out_ok["game_id"] == "G_2024_2") & (out_ok["team_id"] == "KC")].iloc[0]
    assert row_ok["lagged_resolution_status"] == qls.TARGET_STATUS_RESOLVED
    assert row_ok["qb_depth_state_missing"] == 0


# ---------------------------------------------------------------------------
# 7. Monday-night / card-end handling
# ---------------------------------------------------------------------------
def test_card_end_utc_is_max_kickoff_including_monday_night():
    games = _games(2024, [1], monday_night_week=1)
    # Add a second, earlier Sunday game to the same card.
    extra = _games(2024, [1]).iloc[[0]].copy()
    extra["game_id"] = "G_2024_1_EARLY"
    combined = pd.concat([games, extra], ignore_index=True)
    card_table = qls.build_card_table(combined)
    row = card_table[card_table["source_card_week"] == 1].iloc[0]
    assert row["card_end_utc"] == combined["scheduled_kickoff_utc"].max()


# ---------------------------------------------------------------------------
# 8. TUE/FRI target membership (reused certified ledger)
# ---------------------------------------------------------------------------
def test_fri_excludes_thursday_opener_tue_includes_it():
    games = _games(2024, [1, 2])
    games.loc[games["game_id"] == "G_2024_1", "scheduled_kickoff_utc"] -= pd.Timedelta(days=3)  # Thursday opener
    card_table, chain = _card_table_and_chain(
        [_historical_row(2024, 1, "KC", "00-A", "1")], games
    )
    fri_out = qls.resolve_lagged_target_state(games, "FRI", card_table, chain)
    tue_out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain)
    assert "G_2024_1" not in set(fri_out["game_id"])
    assert "G_2024_1" in set(tue_out["game_id"])


# ---------------------------------------------------------------------------
# 9. Season reset (no cross-season carryover)
# ---------------------------------------------------------------------------
def test_season_boundary_resets_prior_card_lookup():
    games = pd.concat([_games(2024, [1, 2]), _games(2025, [1, 2])], ignore_index=True)
    rows = [
        _historical_row(2024, 1, "KC", "00-KC-2024A", "1"),
        _historical_row(2024, 2, "KC", "00-KC-2024A", "1"),
        _historical_row(2025, 1, "KC", "00-KC-2025A", "1"),
    ]
    card_table, chain = _card_table_and_chain(rows, games)
    out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain)
    wk1_2025 = out[(out["game_id"] == "G_2025_1") & (out["team_id"] == "KC")].iloc[0]
    assert wk1_2025["lagged_resolution_status"] == qls.TARGET_STATUS_NO_PRIOR_CARD
    assert wk1_2025["qb_depth_state_missing"] == 1


# ---------------------------------------------------------------------------
# 10 & 11. Change + continuity features (with skip-gap semantics)
# ---------------------------------------------------------------------------
def test_change_and_continuity_across_consecutive_resolved_cards():
    games = _games(2024, [1, 2, 3, 4])
    rows = [
        _historical_row(2024, 1, "KC", "00-A", "1"),
        _historical_row(2024, 2, "KC", "00-A", "1"),
        _historical_row(2024, 3, "KC", "00-B", "1"),  # QB change at week 3
    ]
    card_table, chain = _card_table_and_chain(rows, games)
    out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain)
    out = out[out["team_id"] == "KC"].set_index("game_id")

    wk2 = out.loc["G_2024_2"]  # prior card = week 1 (first-ever resolved card)
    assert wk2["qb_depth_changed"] == 0
    assert wk2["qb_depth_continuity_cards"] == 1

    wk3 = out.loc["G_2024_3"]  # prior card = week 2, same QB as week 1
    assert wk3["qb_depth_changed"] == 0
    assert wk3["qb_depth_continuity_cards"] == 2

    wk4 = out.loc["G_2024_4"]  # prior card = week 3, QB changed from week 2's
    assert wk4["qb_depth_changed"] == 1
    assert wk4["qb_depth_continuity_cards"] == 1


def test_continuity_skips_over_an_unresolved_gap():
    games = _games(2024, [1, 2, 3, 4])
    rows = [
        _historical_row(2024, 1, "KC", "00-A", "1"),
        # week 2: ambiguous tie -> unresolved gap
        _historical_row(2024, 2, "KC", "00-X", "1"),
        _historical_row(2024, 2, "KC", "00-Y", "1"),
        _historical_row(2024, 3, "KC", "00-A", "1"),  # same QB as the last AVAILABLE (week 1) card
    ]
    card_table, chain = _card_table_and_chain(rows, games)
    out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain)
    out = out[out["team_id"] == "KC"].set_index("game_id")

    wk4 = out.loc["G_2024_4"]  # prior card = week 3, whose "previous available" is week 1 (same QB)
    assert wk4["qb_depth_changed"] == 0
    assert wk4["qb_depth_continuity_cards"] == 2


# ---------------------------------------------------------------------------
# 12. Missingness forces changed=0 / continuity=0 regardless of chain state
# ---------------------------------------------------------------------------
def test_missing_required_state_zeroes_changed_and_continuity():
    games = _games(2024, [1, 2, 3])
    rows = [
        _historical_row(2024, 1, "KC", "00-A", "1"),
        _historical_row(2024, 2, "KC", "00-X", "1"),
        _historical_row(2024, 2, "KC", "00-Y", "1"),  # ambiguous -> week 3's required (week 2) state is missing
    ]
    card_table, chain = _card_table_and_chain(rows, games)
    out = qls.resolve_lagged_target_state(games, "TUE", card_table, chain)
    wk3 = out[(out["game_id"] == "G_2024_3") & (out["team_id"] == "KC")].iloc[0]
    assert wk3["qb_depth_state_missing"] == 1
    assert wk3["qb_depth_changed"] == 0
    assert wk3["qb_depth_continuity_cards"] == 0
    assert wk3["lagged_resolution_status"] == qls.TARGET_STATUS_PRIOR_CARD_UNRESOLVED


# ---------------------------------------------------------------------------
# 13. Ambiguous source abstention (never guessed)
# ---------------------------------------------------------------------------
def test_ambiguous_top_rank_abstains():
    games = _games(2024, [1, 2])
    rows = [
        _historical_row(2024, 1, "KC", "00-X", "1"),
        _historical_row(2024, 1, "KC", "00-Y", "1"),
    ]
    card_table = qls.build_card_table(games)
    out = qls.normalize_historical_card_states(_historical_frame(rows), card_table)
    row = out[out["team_id"] == "KC"].iloc[0]
    assert row["qb_state_resolution_status"] == qls.CARD_STATUS_AMBIGUOUS
    assert row["qb1_player_id"] is None


# ---------------------------------------------------------------------------
# 14. No actual-starter field ever read
# ---------------------------------------------------------------------------
def test_no_actual_starter_field_required_or_used():
    assert "home_qb_id" not in qls.REQUIRED_HISTORICAL_COLUMNS
    assert "away_qb_id" not in qls.REQUIRED_HISTORICAL_COLUMNS
    assert "home_score" not in qls.REQUIRED_HISTORICAL_COLUMNS
    games = _games(2024, [1, 2])
    games = games.drop(columns=["home_score", "away_score"])  # not even present
    card_table = qls.build_card_table(games)  # must not require score/starter columns
    assert len(card_table) == 2


# ---------------------------------------------------------------------------
# 15 & 16. No player-ID / no market feature in the candidate columns
# ---------------------------------------------------------------------------
def test_candidate_feature_columns_exclude_player_id_and_market_data():
    from nfl_hybrid.selection import feature_deduction_2026 as fd

    assert not any("player" in c.lower() or "qb1" in c.lower() for c in qls.CANDIDATE_FEATURE_COLUMNS)
    assert set(qls.CANDIDATE_FEATURE_COLUMNS) & fd.FORBIDDEN_MARKET_COLUMNS == set()
    fd.assert_no_forbidden_market_columns(qls.CANDIDATE_FEATURE_COLUMNS)  # must not raise


# ---------------------------------------------------------------------------
# Feature matrix pivot
# ---------------------------------------------------------------------------
def test_feature_matrix_pivots_home_away_correctly():
    games = _games(2024, [1, 2])
    rows = [
        _historical_row(2024, 1, "KC", "00-KC-A", "1"),
        _historical_row(2024, 1, "BUF", "00-BUF-A", "1"),
    ]
    card_table, chain = _card_table_and_chain(rows, games)
    matrix = qls.build_lagged_depth_feature_matrix(games, "TUE", card_table, chain)
    row = matrix[matrix["game_id"] == "G_2024_2"].iloc[0]
    assert row["home_qb_depth_state_missing"] == 0
    assert row["away_qb_depth_state_missing"] == 0
    assert set(matrix.columns) == {"game_id", *qls.CANDIDATE_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Semantics hash determinism
# ---------------------------------------------------------------------------
def test_semantics_hash_deterministic():
    a = qls.compute_semantics_hash()
    b = qls.compute_semantics_hash(qls.qb_lagged_depth_state_v1_semantics())
    assert a == b
    assert len(a) == 64
