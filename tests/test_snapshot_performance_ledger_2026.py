"""Acceptance proofs for the append-only snapshot performance ledger.

Hermetic: every estate is a ``tmp_path`` directory and every snapshot is
synthetic. No real forecast, no provider, no network, no real artifact root.

What these prove, in the directive's terms: all three snapshots are
immutable; final results attach WITHOUT mutating the snapshots; season
reporting compares the model against the sportsbook OPEN and CLOSE; and
nothing is invented for a game that has not finished.
"""
from __future__ import annotations

import json

import pytest

from nfl_hybrid.production import snapshot_performance_ledger_2026 as pl
from nfl_hybrid.production import snapshot_stages_2026 as st

KICKOFF = "2026-09-20T17:00:00Z"


def _quotes(line: float, total: float) -> list[dict]:
    return [
        {"bookmaker_key": book, "market": "spreads", "point": line, "price_decimal": 1.91}
        for book in ("draftkings", "fanduel", "betmgm")
    ] + [
        {"bookmaker_key": book, "market": "totals", "point": total, "price_decimal": 1.91}
        for book in ("draftkings", "fanduel", "betmgm")
    ]


def _snapshot(stage: str, *, spread: float, total: float, margin: float, model_total: float, at: str, **over):
    payload = {
        "season": 2026,
        "week": 2,
        "game_id": "2026_02_BUF_KC",
        "snapshot_stage": stage,
        "snapshot_at_utc": at,
        "scheduled_kickoff_utc": KICKOFF,
        "model_home_margin": margin,
        "model_total": model_total,
        "market_consensus_home_spread": spread,
        "market_consensus_total": total,
        "market_book_quotes": _quotes(spread, total),
        "market_observed_at_utc": at,
        "model_tag": "certified-tag",
        "model_sha": "cert-sha",
        "production_git_commit": "a" * 40,
        "active_calibrator_source": "BASELINE",
        "active_calibrator_candidate_id": None,
        "active_calibrator_seed_sha256": "b" * 64,
        "capture_manifest_sha256": "c" * 64,
        "forecast_run_id": "20260920T160000Z__TUE__abcd1234",
        "forecast_prediction_hash": "d" * 64,
    }
    payload.update(over)
    return payload


def _full_game(root):
    """OPEN -3.0/44.5, MID -3.5/45.0, CLOSE -4.5/45.5 -- the line moved toward
    the home team, which the model liked at OPEN."""
    pl.record_snapshot(root, _snapshot(st.STAGE_OPEN, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-14T14:00:00Z"))
    pl.record_snapshot(root, _snapshot(st.STAGE_MID, spread=-3.5, total=45.0, margin=6.4, model_total=47.2, at="2026-09-18T16:00:00Z"))
    pl.record_snapshot(root, _snapshot(st.STAGE_CLOSE, spread=-4.5, total=45.5, margin=6.8, model_total=47.5, at="2026-09-20T16:00:00Z"))


# ===========================================================================
# Immutability of all three snapshots.
# ===========================================================================
@pytest.mark.parametrize("stage", st.SNAPSHOT_STAGES)
def test_a_snapshot_is_written_once(tmp_path, stage):
    snap = _snapshot(stage, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-14T14:00:00Z")
    assert pl.record_snapshot(tmp_path, snap).status == "WRITTEN"
    assert pl.record_snapshot(tmp_path, snap).status == "IDEMPOTENT_NOOP"


@pytest.mark.parametrize("stage", st.SNAPSHOT_STAGES)
def test_a_snapshot_can_never_be_revised(tmp_path, stage):
    pl.record_snapshot(tmp_path, _snapshot(stage, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-14T14:00:00Z"))
    with pytest.raises(pl.SnapshotImmutabilityViolation, match="immutable"):
        pl.record_snapshot(tmp_path, _snapshot(stage, spread=-7.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-14T14:00:00Z"))


def test_an_open_snapshot_cannot_be_overwritten_by_a_later_observation(tmp_path):
    """The OPEN is the FIRST valid market. A later, different market is not a
    correction to it."""
    pl.record_snapshot(tmp_path, _snapshot(st.STAGE_OPEN, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-14T14:00:00Z"))
    with pytest.raises(pl.SnapshotImmutabilityViolation):
        pl.record_snapshot(tmp_path, _snapshot(st.STAGE_OPEN, spread=-4.0, total=45.0, margin=6.5, model_total=47.5, at="2026-09-17T14:00:00Z"))
    stored = pl.read_snapshot(tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC", stage=st.STAGE_OPEN)
    assert stored["market_consensus_home_spread"] == -3.0
    assert stored["snapshot_at_utc"] == "2026-09-14T14:00:00Z"


def test_the_three_stages_are_stored_separately(tmp_path):
    _full_game(tmp_path)
    for stage in st.SNAPSHOT_STAGES:
        assert pl.read_snapshot(tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC", stage=stage) is not None


def test_a_snapshot_missing_any_provenance_field_is_refused(tmp_path):
    for dropped in ("production_git_commit", "capture_manifest_sha256", "active_calibrator_source", "forecast_prediction_hash"):
        snap = _snapshot(st.STAGE_CLOSE, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-20T16:00:00Z")
        del snap[dropped]
        with pytest.raises(pl.PerformanceLedgerError, match="missing required field"):
            pl.record_snapshot(tmp_path, snap)


def test_a_consensus_with_no_per_book_evidence_is_refused(tmp_path):
    snap = _snapshot(st.STAGE_CLOSE, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-20T16:00:00Z")
    snap["market_book_quotes"] = None
    with pytest.raises(pl.PerformanceLedgerError, match="per-book quotes"):
        pl.record_snapshot(tmp_path, snap)


def test_a_consensus_line_with_an_empty_book_list_is_refused(tmp_path):
    """A published number whose books nobody can check is not reviewable."""
    snap = _snapshot(st.STAGE_CLOSE, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-20T16:00:00Z")
    snap["market_book_quotes"] = []
    with pytest.raises(pl.PerformanceLedgerError, match="not optional"):
        pl.record_snapshot(tmp_path, snap)


def test_a_snapshot_with_no_market_at_all_may_record_no_books(tmp_path):
    """The legitimate empty case: no consensus was established, so there are
    no books to retain and none are demanded."""
    snap = _snapshot(st.STAGE_OPEN, spread=None, total=None, margin=6.0, model_total=47.0, at="2026-09-14T14:00:00Z")
    snap["market_book_quotes"] = []
    assert pl.record_snapshot(tmp_path, snap).status == "WRITTEN"


def test_an_unknown_stage_is_refused(tmp_path):
    snap = _snapshot(st.STAGE_CLOSE, spread=-3.0, total=44.5, margin=6.0, model_total=47.0, at="2026-09-20T16:00:00Z")
    snap["snapshot_stage"] = "TUE"
    with pytest.raises(st.SnapshotStageError, match="unknown snapshot stage"):
        pl.record_snapshot(tmp_path, snap)


# ===========================================================================
# Results attach without mutating anything.
# ===========================================================================
def test_a_result_attaches_without_touching_any_snapshot(tmp_path):
    _full_game(tmp_path)
    before = {
        stage: pl.snapshot_path(tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC", stage=stage).read_bytes()
        for stage in st.SNAPSHOT_STAGES
    }

    pl.attach_result(
        tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC",
        final_home_score=27, final_away_score=20, result_attached_at_utc="2026-09-21T04:00:00Z",
    )

    for stage, payload in before.items():
        path = pl.snapshot_path(tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC", stage=stage)
        assert path.read_bytes() == payload


def test_the_result_is_its_own_record(tmp_path):
    _full_game(tmp_path)
    result = pl.attach_result(
        tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC",
        final_home_score=27, final_away_score=20, result_attached_at_utc="2026-09-21T04:00:00Z",
    )
    assert result.status == "WRITTEN"
    outcome = pl.read_outcome(tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC")
    assert outcome["realized_margin"] == 7.0
    assert outcome["realized_total"] == 47.0


def test_a_repeated_identical_result_is_a_noop_and_a_different_one_fails_closed(tmp_path):
    _full_game(tmp_path)
    kwargs = dict(
        season=2026, week=2, game_id="2026_02_BUF_KC",
        final_home_score=27, final_away_score=20, result_attached_at_utc="2026-09-21T04:00:00Z",
    )
    assert pl.attach_result(tmp_path, **kwargs).status == "WRITTEN"
    assert pl.attach_result(tmp_path, **kwargs).status == "IDEMPOTENT_NOOP"
    with pytest.raises(pl.SnapshotImmutabilityViolation, match="immutable"):
        pl.attach_result(tmp_path, **{**kwargs, "final_home_score": 30})


def test_a_half_known_result_is_not_a_result(tmp_path):
    with pytest.raises(pl.PerformanceLedgerError, match="both final scores"):
        pl.attach_result(
            tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC",
            final_home_score=27, final_away_score=None, result_attached_at_utc="2026-09-21T04:00:00Z",
        )


# ===========================================================================
# ATS / total grading.
# ===========================================================================
@pytest.mark.parametrize(
    "margin,spread,expected",
    [
        (7.0, -3.5, pl.RESULT_HOME_COVER),   # home favoured by 3.5, won by 7
        (2.0, -3.5, pl.RESULT_AWAY_COVER),   # home favoured by 3.5, won by 2
        (3.0, -3.0, pl.RESULT_PUSH),         # exact
        (-1.0, 3.0, pl.RESULT_HOME_COVER),   # home +3 underdog, lost by 1
        (-7.0, 3.0, pl.RESULT_AWAY_COVER),
        (0.0, 0.0, pl.RESULT_PUSH),          # pick'em, impossible tie, still defined
    ],
)
def test_ats_result_uses_captured_sportsbook_notation(margin, spread, expected):
    assert pl.ats_result(margin, spread) == expected


@pytest.mark.parametrize(
    "realized,market,expected",
    [(47.0, 45.5, pl.RESULT_OVER), (44.0, 45.5, pl.RESULT_UNDER), (45.0, 45.0, pl.RESULT_PUSH)],
)
def test_total_result_grades_over_under_and_push(realized, market, expected):
    assert pl.total_result(realized, market) == expected


@pytest.mark.parametrize("missing", [(None, -3.5), (7.0, None), (None, None)])
def test_a_result_is_undefined_rather_than_guessed_when_a_value_is_missing(missing):
    assert pl.ats_result(*missing) is None
    assert pl.total_result(*missing) is None


# ===========================================================================
# OPEN-to-CLOSE movement and closing line value.
# ===========================================================================
def test_open_to_close_movement_is_reported(tmp_path):
    _full_game(tmp_path)
    rows = pl.build_report_rows(tmp_path)
    assert {row["open_close_spread_movement"] for row in rows} == {-1.5}  # -3.0 -> -4.5
    assert {row["open_close_total_movement"] for row in rows} == {1.0}  # 44.5 -> 45.5


def test_backing_the_home_team_before_the_line_moved_toward_it_is_positive_clv():
    assert pl.closing_line_value_points(
        model_home_margin_at_open=6.0, open_spread=-3.0, close_spread=-4.5
    ) == 1.5


def test_backing_the_away_team_before_the_line_moved_toward_it_is_positive_clv():
    """A home spread moving +3.0 -> +4.5 moves TOWARD the away side: the away
    team went from -3.0 to -4.5, so laying 3 early beat the close by 1.5."""
    assert pl.closing_line_value_points(
        model_home_margin_at_open=-6.0, open_spread=3.0, close_spread=4.5
    ) == 1.5


def test_pricing_early_against_an_adverse_move_is_negative_clv():
    """Away side at -3.0 that closes -1.5 -- the market moved toward the home
    team, so the early away price was worse than the close."""
    assert pl.closing_line_value_points(
        model_home_margin_at_open=-6.0, open_spread=3.0, close_spread=1.5
    ) == -1.5
    # Symmetrically for the home side: -3.0 closing -1.5 is an adverse move.
    assert pl.closing_line_value_points(
        model_home_margin_at_open=6.0, open_spread=-3.0, close_spread=-1.5
    ) == -1.5


def test_clv_is_undefined_when_the_model_abstained_or_had_no_side():
    assert pl.closing_line_value_points(model_home_margin_at_open=None, open_spread=-3.0, close_spread=-4.0) is None
    assert pl.closing_line_value_points(model_home_margin_at_open=3.0, open_spread=-3.0, close_spread=-4.0) is None
    assert pl.closing_line_value_points(model_home_margin_at_open=6.0, open_spread=None, close_spread=-4.0) is None


# ===========================================================================
# Season reporting: model versus OPEN and CLOSE.
# ===========================================================================
def test_the_report_compares_the_model_against_each_stage_market(tmp_path):
    _full_game(tmp_path)
    pl.attach_result(
        tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC",
        final_home_score=27, final_away_score=20, result_attached_at_utc="2026-09-21T04:00:00Z",
    )
    rows = {row["snapshot_stage"]: row for row in pl.build_report_rows(tmp_path)}

    # Realized: margin 7, total 47.
    assert rows[st.STAGE_OPEN]["model_margin_abs_error"] == 1.0  # model 6.0
    assert rows[st.STAGE_OPEN]["market_margin_abs_error"] == 4.0  # open spread -3.0 => +3.0
    assert rows[st.STAGE_CLOSE]["market_margin_abs_error"] == 2.5  # close spread -4.5 => +4.5
    assert rows[st.STAGE_OPEN]["model_total_abs_error"] == 0.0  # model 47.0
    assert rows[st.STAGE_OPEN]["market_total_abs_error"] == 2.5  # open total 44.5
    for stage in st.SNAPSHOT_STAGES:
        assert rows[stage]["ats_result"] == pl.RESULT_HOME_COVER
        assert rows[stage]["outcome_status"] == pl.OUTCOME_FINAL


def test_the_market_error_uses_the_same_sign_convention_as_the_model(tmp_path):
    """A home spread of -4.5 is a predicted home margin of +4.5. Without the
    flip the market would look absurdly wrong and the comparison meaningless."""
    _full_game(tmp_path)
    pl.attach_result(
        tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC",
        final_home_score=27, final_away_score=20, result_attached_at_utc="2026-09-21T04:00:00Z",
    )
    close = [r for r in pl.build_report_rows(tmp_path) if r["snapshot_stage"] == st.STAGE_CLOSE][0]
    assert close["market_margin_abs_error"] == abs(4.5 - 7.0)


def test_a_pending_game_is_reported_as_pending_not_omitted(tmp_path):
    _full_game(tmp_path)
    rows = pl.build_report_rows(tmp_path)
    assert len(rows) == 3
    for row in rows:
        assert row["outcome_status"] == pl.OUTCOME_PENDING
        assert row["realized_margin"] is None
        assert row["model_margin_abs_error"] is None
        assert row["ats_result"] is None


def test_the_report_summary_separates_the_stages(tmp_path):
    _full_game(tmp_path)
    pl.attach_result(
        tmp_path, season=2026, week=2, game_id="2026_02_BUF_KC",
        final_home_score=27, final_away_score=20, result_attached_at_utc="2026-09-21T04:00:00Z",
    )
    report = pl.report_json(tmp_path)
    assert report["schema_version"] == "nfl-snapshot-performance-report-v1"
    assert report["snapshot_rows"] == 3
    assert report["graded_rows"] == 3
    assert report["pending_rows"] == 0
    assert report["by_stage"][st.STAGE_OPEN]["model_margin_mae"] == 1.0
    assert report["by_stage"][st.STAGE_OPEN]["market_margin_mae"] == 4.0
    assert report["by_stage"][st.STAGE_CLOSE]["market_margin_mae"] == 2.5


def test_the_csv_report_has_a_stable_header_and_one_row_per_snapshot(tmp_path):
    _full_game(tmp_path)
    csv_text = pl.report_csv(tmp_path)
    lines = csv_text.strip().split("\n")
    assert lines[0] == ",".join(pl.REPORT_COLUMNS)
    assert len(lines) == 4  # header + three stages


def test_the_csv_leaves_missing_values_empty_rather_than_zero(tmp_path):
    _full_game(tmp_path)
    for line in pl.report_csv(tmp_path).strip().split("\n")[1:]:
        cells = line.split(",")
        realized = cells[list(pl.REPORT_COLUMNS).index("realized_margin")]
        assert realized == ""


def test_an_empty_ledger_reports_nothing_rather_than_failing(tmp_path):
    assert pl.build_report_rows(tmp_path) == []
    assert pl.report_json(tmp_path)["snapshot_rows"] == 0


def test_reporting_never_mutates_the_ledger(tmp_path):
    _full_game(tmp_path)
    before = {p: p.read_bytes() for p in sorted(pl.ledger_root(tmp_path).rglob("*.json"))}
    pl.build_report_rows(tmp_path)
    pl.report_json(tmp_path)
    pl.report_csv(tmp_path)
    assert {p: p.read_bytes() for p in sorted(pl.ledger_root(tmp_path).rglob("*.json"))} == before


# ===========================================================================
# Storage discipline.
# ===========================================================================
def test_the_ledger_lives_inside_the_existing_production_tree(tmp_path):
    """No second production estate: it is a sibling of the forecast and
    evaluation ledgers under production-2026."""
    root = pl.ledger_root(tmp_path)
    assert root.parent.name == "production-2026"
    assert root.relative_to(tmp_path).parts == ("production-2026", "snapshot-performance-ledger")


def test_one_small_file_per_snapshot_keeps_a_season_bounded(tmp_path):
    for week in (1, 2, 3):
        for game in range(16):
            for stage in st.SNAPSHOT_STAGES:
                pl.record_snapshot(
                    tmp_path,
                    _snapshot(
                        stage, spread=-3.0, total=44.5, margin=6.0, model_total=47.0,
                        at="2026-09-14T14:00:00Z", week=week, game_id=f"2026_{week:02d}_G{game:02d}",
                    ),
                )
    files = list(pl.ledger_root(tmp_path).rglob("*.json"))
    assert len(files) == 3 * 16 * 3
    assert sum(p.stat().st_size for p in files) < 2_000_000  # comfortably under 2 MB for 3 weeks


def test_no_temporary_file_is_left_behind(tmp_path):
    _full_game(tmp_path)
    assert not list(pl.ledger_root(tmp_path).rglob(".*.tmp"))


def test_the_stored_snapshot_is_exactly_what_was_recorded(tmp_path):
    snap = _snapshot(st.STAGE_CLOSE, spread=-4.5, total=45.5, margin=6.8, model_total=47.5, at="2026-09-20T16:00:00Z")
    path = pl.record_snapshot(tmp_path, snap).path
    assert json.loads(path.read_text()) == snap
