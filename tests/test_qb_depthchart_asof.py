import json

import pandas as pd
import pytest

from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.features import qb_depthchart_asof as qd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _week1_card():
    # Week 1, 2025: Thursday opener (BUF@KC) + Sunday game (MIA@NE).
    # Mirrors tests/test_horizon_elo.py's own fixture convention.
    return pd.DataFrame(
        {
            "game_id": ["G_THU", "G_SUN"],
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "home_team_id": ["KC", "NE"],
            "away_team_id": ["BUF", "MIA"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2025-09-05T00:20:00Z", "2025-09-07T17:00:00Z"], utc=True
            ),
            "home_score": [None, None],
            "away_score": [None, None],
            "neutral_site": [False, False],
        }
    )


def _snapshots(rows):
    """rows: list of (dt, team, gsis_id, pos_abb, pos_rank)."""
    return pd.DataFrame(rows, columns=["dt", "team", "gsis_id", "pos_abb", "pos_rank"])


# ---------------------------------------------------------------------------
# Certified cutoff reuse
# ---------------------------------------------------------------------------
def test_reuses_certified_tue_and_fri_cutoffs_exactly():
    games = _week1_card()
    ledger = he.build_horizon_membership_ledger(games)
    snaps = _snapshots(
        [
            ("2025-09-02T07:00:00Z", "KC", "00-KC1", "QB", 1),
            ("2025-09-02T07:00:00Z", "BUF", "00-BUF1", "QB", 1),
            ("2025-09-02T07:00:00Z", "NE", "00-NE1", "QB", 1),
            ("2025-09-02T07:00:00Z", "MIA", "00-MIA1", "QB", 1),
        ]
    )
    for horizon in ("TUE", "FRI"):
        out = qd.resolve_depthchart_qb1_asof(games, snaps, horizon, membership_ledger=ledger)
        h = horizon.lower()
        expected = ledger.set_index("game_id")[f"{h}_cutoff_utc"]
        for game_id, cutoff in out.set_index("game_id")["target_cutoff_utc"].items():
            assert cutoff == expected.loc[game_id]


def test_thursday_game_absent_as_fri_target_after_kickoff():
    games = _week1_card()
    snaps = _snapshots([("2025-09-02T07:00:00Z", "KC", "00-KC1", "QB", 1)])
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "FRI")
    assert "G_THU" not in set(out["game_id"])
    tue_out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE")
    assert "G_THU" in set(tue_out["game_id"])


# ---------------------------------------------------------------------------
# Latest-snapshot-before-cutoff selection
# ---------------------------------------------------------------------------
def test_latest_snapshot_before_cutoff_used_and_later_one_excluded():
    games = _week1_card()
    ledger = he.build_horizon_membership_ledger(games)
    tue_cutoff = ledger.set_index("game_id").loc["G_SUN", "tue_cutoff_utc"]
    fri_cutoff = ledger.set_index("game_id").loc["G_SUN", "fri_cutoff_utc"]
    assert tue_cutoff < fri_cutoff

    before_tue = tue_cutoff - pd.Timedelta(hours=6)
    between = tue_cutoff + pd.Timedelta(hours=6)
    assert between < fri_cutoff

    snaps = _snapshots(
        [
            (before_tue.isoformat(), "NE", "00-OLD", "QB", 1),
            (between.isoformat(), "NE", "00-NEW", "QB", 1),
            ("2099-01-01T00:00:00Z", "NE", "00-FUTURE", "QB", 1),  # strictly after FRI cutoff
            (before_tue.isoformat(), "MIA", "00-MIA1", "QB", 1),
        ]
    )
    tue_out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE").set_index("team_id")
    fri_out = qd.resolve_depthchart_qb1_asof(games, snaps, "FRI").set_index("team_id")

    assert tue_out.loc["NE", "qb1_player_id"] == "00-OLD"
    assert fri_out.loc["NE", "qb1_player_id"] == "00-NEW"  # sees the update TUE didn't
    assert fri_out.loc["NE", "qb1_source_snapshot"] != "2099-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Resolution status coverage
# ---------------------------------------------------------------------------
def test_unique_qb1_resolved():
    games = _week1_card()
    snaps = _snapshots(
        [
            ("2025-09-01T00:00:00Z", "KC", "00-KC1", "QB", 1),
            ("2025-09-01T00:00:00Z", "BUF", "00-BUF1", "QB", 1),
        ]
    )
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE").set_index("team_id")
    assert out.loc["KC", "qb1_resolution_status"] == qd.STATUS_RESOLVED
    assert out.loc["KC", "qb1_player_id"] == "00-KC1"


def test_duplicate_top_rank_is_ambiguous():
    games = _week1_card()
    snaps = _snapshots(
        [
            ("2025-09-01T00:00:00Z", "KC", "00-KC1", "QB", 1),
            ("2025-09-01T00:00:00Z", "KC", "00-KC2", "QB", 1),
        ]
    )
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE").set_index("team_id")
    assert out.loc["KC", "qb1_resolution_status"] == qd.STATUS_AMBIGUOUS
    assert pd.isna(out.loc["KC", "qb1_player_id"])


def test_no_qb_rows_abstains():
    games = _week1_card()
    snaps = _snapshots([("2025-09-01T00:00:00Z", "KC", "00-K1", "K", 1)])
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE").set_index("team_id")
    assert out.loc["KC", "qb1_resolution_status"] == qd.STATUS_NO_QB_ROWS


def test_no_valid_snapshot_before_cutoff_abstains():
    games = _week1_card()
    snaps = _snapshots([("2099-01-01T00:00:00Z", "KC", "00-KC1", "QB", 1)])
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE").set_index("team_id")
    assert out.loc["KC", "qb1_resolution_status"] == qd.STATUS_NO_VALID_SNAPSHOT
    assert out.loc["KC", "qb1_source_snapshot"] is None


def test_identifier_failure_abstains():
    games = _week1_card()
    snaps = _snapshots([("2025-09-01T00:00:00Z", "KC", None, "QB", 1)])
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE").set_index("team_id")
    assert out.loc["KC", "qb1_resolution_status"] == qd.STATUS_IDENTIFIER_UNRESOLVED


def test_schema_drift_when_required_columns_missing():
    games = _week1_card()
    old_schema = pd.DataFrame({"season": [2020], "week": [1], "depth_team": ["1"], "position": ["QB"]})
    out = qd.resolve_depthchart_qb1_asof(games, old_schema, "TUE")
    assert (out["qb1_resolution_status"] == qd.STATUS_SCHEMA_DRIFT).all()


def test_schema_drift_when_dt_unparseable():
    games = _week1_card()
    snaps = _snapshots([("not-a-timestamp", "KC", "00-KC1", "QB", 1)])
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE")
    assert (out["qb1_resolution_status"] == qd.STATUS_SCHEMA_DRIFT).all()


def test_source_unavailable_when_no_data():
    games = _week1_card()
    out_none = qd.resolve_depthchart_qb1_asof(games, None, "TUE")
    assert (out_none["qb1_resolution_status"] == qd.STATUS_SOURCE_UNAVAILABLE).all()
    out_empty = qd.resolve_depthchart_qb1_asof(games, pd.DataFrame(), "TUE")
    assert (out_empty["qb1_resolution_status"] == qd.STATUS_SOURCE_UNAVAILABLE).all()


# ---------------------------------------------------------------------------
# No hindsight / no leakage guarantees
# ---------------------------------------------------------------------------
def test_no_actual_starter_or_target_game_columns_in_output():
    games = _week1_card()
    snaps = _snapshots([("2025-09-01T00:00:00Z", "KC", "00-KC1", "QB", 1)])
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE")
    forbidden = {"home_qb_id", "away_qb_id", "actual_starter", "home_score", "away_score", "market", "odds"}
    assert forbidden.isdisjoint(out.columns)


def test_games_with_actual_qb_columns_are_ignored():
    games = _week1_card()
    games["home_qb_id"] = "SHOULD_NEVER_BE_READ"
    games["away_qb_id"] = "SHOULD_NEVER_BE_READ"
    snaps = _snapshots([("2025-09-01T00:00:00Z", "KC", "00-KC1", "QB", 1), ("2025-09-01T00:00:00Z", "BUF", "00-BUF1", "QB", 1)])
    out = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE").set_index("team_id")
    assert out.loc["KC", "qb1_player_id"] == "00-KC1"


def test_no_market_or_human_override_inputs_accepted():
    import inspect

    sig = inspect.signature(qd.resolve_depthchart_qb1_asof)
    params = set(sig.parameters)
    assert "market" not in params
    assert "override" not in params
    assert "injuries" not in params


# ---------------------------------------------------------------------------
# Date-only vs exact-timestamp normalization parity
# ---------------------------------------------------------------------------
def test_exact_timestamp_branch_used_directly():
    result = qd.normalize_source_available_at("2026-08-25T07:39:23Z", precision=qd.PRECISION_EXACT_TIMESTAMP)
    assert result == pd.Timestamp("2026-08-25T07:39:23Z", tz="UTC")


def test_date_only_branch_is_conservative_next_day():
    # A Wednesday date-only record must not be usable for Wednesday-noon
    # knowledge -- available_at is the FOLLOWING day at 00:00 America/New_York.
    result = qd.normalize_source_available_at("2026-03-04", precision=qd.PRECISION_DATE_ONLY)  # a Wednesday
    expected_local = pd.Timestamp("2026-03-05T00:00:00").tz_localize(qd.NY_ZONE)
    assert result == expected_local.tz_convert("UTC")


def test_date_only_branch_is_dst_aware():
    # Nov 1, 2025 -> Nov 2 00:00 America/New_York is still EDT (DST ends Nov 2).
    result = qd.normalize_source_available_at("2025-11-01", precision=qd.PRECISION_DATE_ONLY)
    assert result == pd.Timestamp("2025-11-02T00:00:00-04:00").tz_convert("UTC")


def test_unknown_precision_rejected():
    with pytest.raises(ValueError):
        qd.normalize_source_available_at("2026-01-01", precision="GUESS")


# ---------------------------------------------------------------------------
# Live observation log: append-only, no credentials, deterministic content.
# ---------------------------------------------------------------------------
def test_observation_log_is_append_only_and_never_overwrites(tmp_path):
    frame1 = pd.DataFrame({"dt": ["2026-08-24T07:53:13Z"] * 2, "team": ["KC", "BUF"]})
    frame2 = pd.DataFrame({"dt": ["2026-08-25T07:39:23Z"] * 3, "team": ["KC", "BUF", "NE"]})

    r1 = qd.record_live_observation(data_root=tmp_path, snapshot_frame=frame1, observed_at_utc="2026-08-24T08:00:00Z")
    r2 = qd.record_live_observation(data_root=tmp_path, snapshot_frame=frame2, observed_at_utc="2026-08-25T08:00:00Z")

    log = qd.read_live_observation_log(tmp_path)
    assert len(log) == 2
    assert log.iloc[0]["row_count"] == 2
    assert log.iloc[1]["row_count"] == 3
    assert r1["source_content_hash"] != r2["source_content_hash"]

    log_path = tmp_path / qd.LIVE_OBSERVATION_LOG_SUBDIR / qd.LIVE_OBSERVATION_LOG_FILENAME
    original_text = log_path.read_text(encoding="utf-8")
    assert original_text.count("\n") == 2  # two appended lines, nothing rewritten
    first_line = original_text.splitlines()[0]
    assert json.loads(first_line)["row_count"] == 2  # first record untouched by second append


def test_observation_log_has_no_credential_fields(tmp_path):
    frame = pd.DataFrame({"dt": ["2026-08-25T07:39:23Z"], "team": ["KC"]})
    record = qd.record_live_observation(data_root=tmp_path, snapshot_frame=frame, observed_at_utc="2026-08-25T08:00:00Z")
    blob = json.dumps(record).lower()
    for banned in ("key", "token", "secret", "password", "authorization"):
        assert banned not in blob


def test_live_snapshot_availability_requires_both_observed_and_source_time():
    log = pd.DataFrame(
        [
            {
                "observed_at_utc": "2026-08-25T08:00:00Z",
                "source_snapshot_timestamp": "2026-08-25T07:39:23Z",
                "row_count": 469064,
            }
        ]
    )
    # Cutoff before our own observation -> not usable even though source dt qualifies.
    early_cutoff = pd.Timestamp("2026-08-25T07:45:00Z")
    assert qd.live_snapshot_available_at_cutoff(log, early_cutoff) is None
    # Cutoff after both -> usable.
    late_cutoff = pd.Timestamp("2026-08-25T09:00:00Z")
    assert qd.live_snapshot_available_at_cutoff(log, late_cutoff) is not None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_deterministic_rerun_produces_identical_output():
    games = _week1_card()
    snaps = _snapshots(
        [
            ("2025-09-01T00:00:00Z", "KC", "00-KC1", "QB", 1),
            ("2025-09-01T00:00:00Z", "BUF", "00-BUF1", "QB", 1),
            ("2025-09-01T00:00:00Z", "NE", "00-NE1", "QB", 1),
            ("2025-09-01T00:00:00Z", "MIA", "00-MIA1", "QB", 1),
        ]
    )
    out1 = qd.resolve_depthchart_qb1_asof(games, snaps, "TUE")
    out2 = qd.resolve_depthchart_qb1_asof(games, snaps.sample(frac=1, random_state=7), "TUE")
    pd.testing.assert_frame_equal(
        out1.sort_values(["game_id", "team_id"]).reset_index(drop=True),
        out2.sort_values(["game_id", "team_id"]).reset_index(drop=True),
    )


def test_semantics_hash_is_stable_and_no_outcome_fields():
    semantics = qd.qb_depthchart_asof_v1_semantics()
    assert semantics["source"] == qd.SOURCE_NAME
    assert semantics["human_override"] == "DISABLED"
    assert semantics["injury_override"] == "DISABLED"
    assert semantics["actual_starter_fallback"] == "PROHIBITED"
    assert semantics["fallback_rule"].startswith("NONE")
    h1 = qd.compute_semantics_hash(semantics)
    h2 = qd.compute_semantics_hash(qd.qb_depthchart_asof_v1_semantics())
    assert h1 == h2


# ---------------------------------------------------------------------------
# Continuity/change candidate (Section 14) -- never compared to a starter.
# ---------------------------------------------------------------------------
def test_continuity_change_indicator_uses_only_prior_resolved_states():
    games = pd.DataFrame(
        {
            "game_id": ["W1_KC", "W2_KC"],
            "season": [2025, 2025],
        }
    )
    resolved = pd.DataFrame(
        {
            "game_id": ["W1_KC", "W2_KC"],
            "team_id": ["KC", "KC"],
            "horizon": ["TUE", "TUE"],
            "target_cutoff_utc": pd.to_datetime(["2025-09-02T17:00:00Z", "2025-09-09T17:00:00Z"], utc=True),
            "qb1_player_id": ["00-QB1", "00-QB2"],
            "qb1_resolution_status": [qd.STATUS_RESOLVED, qd.STATUS_RESOLVED],
        }
    )
    out = qd.compute_qb1_continuity(resolved, games).set_index("game_id")
    assert pd.isna(out.loc["W1_KC", "qb1_changed_from_previous_resolved_same_horizon"])  # no prior card
    assert bool(out.loc["W2_KC", "qb1_changed_from_previous_resolved_same_horizon"]) is True


def test_continuity_cards_streak_resets_on_change_and_missing_when_unresolved():
    games = pd.DataFrame({"game_id": ["A1", "A2", "A3", "A4"], "season": [2025] * 4})
    resolved = pd.DataFrame(
        {
            "game_id": ["A1", "A2", "A3", "A4"],
            "team_id": ["KC"] * 4,
            "horizon": ["TUE"] * 4,
            "target_cutoff_utc": pd.to_datetime(
                ["2025-09-02T17:00:00Z", "2025-09-09T17:00:00Z", "2025-09-16T17:00:00Z", "2025-09-23T17:00:00Z"],
                utc=True,
            ),
            "qb1_player_id": ["00-A", "00-A", "00-B", None],
            "qb1_resolution_status": [
                qd.STATUS_RESOLVED, qd.STATUS_RESOLVED, qd.STATUS_RESOLVED, qd.STATUS_AMBIGUOUS,
            ],
        }
    )
    out = qd.compute_qb1_continuity(resolved, games).set_index("game_id")
    assert out.loc["A1", "qb1_continuity_cards"] == 1  # no prior resolved card
    assert out.loc["A2", "qb1_continuity_cards"] == 2  # unchanged from A1
    assert out.loc["A3", "qb1_continuity_cards"] == 1  # changed -> streak resets
    assert pd.isna(out.loc["A4", "qb1_continuity_cards"])  # current row not resolved


# ---------------------------------------------------------------------------
# Canonical hash serialization hardening (Section 6).
# ---------------------------------------------------------------------------
def test_canonical_hash_rejects_non_primitive_payload():
    with pytest.raises(TypeError):
        qd._canonical_json_hash({"bad": pd.Timestamp("2026-01-01", tz="UTC")})


def test_canonical_hash_uses_no_default_str_fallback():
    import inspect

    source = inspect.getsource(qd._canonical_json_hash)
    dumps_lines = [line for line in source.splitlines() if "json.dumps(" in line or "sort_keys" in line]
    joined = "\n".join(dumps_lines)
    assert "default=str" not in joined
    assert "ensure_ascii=True" in joined


def test_semantics_hash_matches_certified_pre_hardening_value():
    # The certified V1 semantics payload contains no datetimes and no
    # non-primitive values, so removing the default=str fallback must not
    # change its hash -- verified against the certified, already-persisted
    # value (qb-depthchart-asof-v1-2026/qb_depthchart_asof_v1_semantics.json).
    certified_hash = "041195c3ae45b6edf9ccf4a3cad9d50c52fc572433747bc91e2c868fa4d1f788"
    assert qd.compute_semantics_hash() == certified_hash


def test_canonical_hash_is_ascii_and_compact_separators():
    payload = {"b": 1, "a": "x"}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert qd._canonical_json_hash(payload) == __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
    assert ", " not in text and ": " not in text


# ---------------------------------------------------------------------------
# Semantics / implementation consistency (Section 7): V1.1 additive version.
# ---------------------------------------------------------------------------
def test_v1_1_semantics_is_additive_and_preserves_v1_hash():
    v1 = qd.qb_depthchart_asof_v1_semantics()
    v1_hash = qd.compute_semantics_hash(v1)
    v1_1 = qd.qb_depthchart_asof_v1_1_semantics()

    assert v1_1["schema_version"] == qd.SCHEMA_VERSION_V1_1
    assert v1_1["schema_version"] != v1["schema_version"]
    assert v1_1["supersedes"]["semantics_hash"] == v1_hash
    assert v1_1["supersedes"]["schema_version"] == v1["schema_version"]

    # V1 itself is untouched -- its hash must not move because V1.1 exists.
    assert qd.compute_semantics_hash(qd.qb_depthchart_asof_v1_semantics()) == v1_hash

    # V1.1 documents the continuity/change behavior that V1 never mentioned.
    assert "continuity_change_semantics" not in v1
    assert "continuity_change_semantics" in v1_1
    assert "qb1_continuity_cards" in v1_1["continuity_change_semantics"]
    assert "qb1_changed_from_previous_resolved_same_horizon" in v1_1["continuity_change_semantics"]


def test_v1_1_semantics_hash_is_stable_and_differs_from_v1():
    h1 = qd.compute_semantics_v1_1_hash()
    h2 = qd.compute_semantics_v1_1_hash(qd.qb_depthchart_asof_v1_1_semantics())
    assert h1 == h2
    assert h1 != qd.compute_semantics_hash()


# ---------------------------------------------------------------------------
# Historical as-of coverage fail-closed behavior (Section 1/3): a source that
# only carries snapshots far after a historical target cutoff must abstain
# for every target, never fabricate a resolution from later information.
# ---------------------------------------------------------------------------
def test_no_rolling_snapshot_coverage_for_a_historical_season_is_fully_unresolved():
    # A real 2020 Week 1 card (actual 2020 kickoff dates) -- the rolling
    # snapshot schema (per the source-contract audit) begins 2025-08-03, so
    # zero rows exist before that regardless of how much snapshot data the
    # source otherwise holds.
    games = _week1_card()
    games["season"] = 2020
    games["scheduled_kickoff_utc"] = pd.to_datetime(
        ["2020-09-10T00:20:00Z", "2020-09-13T17:00:00Z"], utc=True
    )
    # Only 2025-era rolling snapshots exist (mirrors the real source: the
    # rolling-snapshot schema begins 2025-08-03, zero rows before that).
    snaps = _snapshots(
        [
            ("2025-08-03T10:09:07Z", "KC", "00-KC1", "QB", 1),
            ("2025-08-03T10:09:07Z", "BUF", "00-BUF1", "QB", 1),
            ("2025-08-03T10:09:07Z", "NE", "00-NE1", "QB", 1),
            ("2025-08-03T10:09:07Z", "MIA", "00-MIA1", "QB", 1),
        ]
    )
    for horizon in ("TUE", "FRI"):
        out = qd.resolve_depthchart_qb1_asof(games, snaps, horizon)
        assert (out["qb1_resolution_status"] == qd.STATUS_NO_VALID_SNAPSHOT).all()
        assert out["qb1_player_id"].isna().all()


# ---------------------------------------------------------------------------
# Selection-period no-hindsight guarantee (Section 2/3): a snapshot dated
# after a historical target's own cutoff must never be used to resolve it,
# even when it is the ONLY snapshot the source has for that team.
# ---------------------------------------------------------------------------
def test_only_future_snapshot_available_is_never_used_for_historical_target():
    games = _week1_card()
    ledger = he.build_horizon_membership_ledger(games)
    tue_cutoff = ledger.set_index("game_id").loc["G_SUN", "tue_cutoff_utc"]
    future_only = _snapshots(
        [
            ((tue_cutoff + pd.Timedelta(days=365)).isoformat(), "NE", "00-FUTURE", "QB", 1),
            ((tue_cutoff + pd.Timedelta(days=365)).isoformat(), "MIA", "00-FUTURE-MIA", "QB", 1),
        ]
    )
    out = qd.resolve_depthchart_qb1_asof(games, future_only, "TUE", membership_ledger=ledger).set_index("team_id")
    assert out.loc["NE", "qb1_resolution_status"] == qd.STATUS_NO_VALID_SNAPSHOT
    assert pd.isna(out.loc["NE", "qb1_player_id"])


# ---------------------------------------------------------------------------
# Observation log: a later call may never backdate/precede an already
# appended observation (Section 8).
# ---------------------------------------------------------------------------
def test_observation_log_rejects_backdated_append(tmp_path):
    frame = pd.DataFrame({"dt": ["2026-08-25T07:39:23Z"], "team": ["KC"]})
    qd.record_live_observation(data_root=tmp_path, snapshot_frame=frame, observed_at_utc="2026-08-25T08:00:00Z")
    with pytest.raises(ValueError):
        qd.record_live_observation(
            data_root=tmp_path, snapshot_frame=frame, observed_at_utc="2026-08-25T07:00:00Z"
        )
    # The rejected append must not have been written -- log still has one line.
    log = qd.read_live_observation_log(tmp_path)
    assert len(log) == 1


def test_continuity_resets_at_season_boundary():
    games = pd.DataFrame({"game_id": ["S1", "S2"], "season": [2025, 2026]})
    resolved = pd.DataFrame(
        {
            "game_id": ["S1", "S2"],
            "team_id": ["KC", "KC"],
            "horizon": ["TUE", "TUE"],
            "target_cutoff_utc": pd.to_datetime(["2025-12-01T17:00:00Z", "2026-09-01T17:00:00Z"], utc=True),
            "qb1_player_id": ["00-QB1", "00-QB1"],
            "qb1_resolution_status": [qd.STATUS_RESOLVED, qd.STATUS_RESOLVED],
        }
    )
    out = qd.compute_qb1_continuity(resolved, games).set_index("game_id")
    assert pd.isna(out.loc["S2", "qb1_changed_from_previous_resolved_same_horizon"])  # season reset, no prior
