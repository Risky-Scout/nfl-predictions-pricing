"""Proofs for the current-week public pregame board.

THE PRODUCT DEFECT THIS FIXES. Publication used to be CLOSE-only, so a game
reached the public page roughly an hour before its own kickoff and the Sunday
slate was invisible all week. The public product is the model's prediction for
the current week BEFORE the games, so a game is now public from its OPEN
snapshot onwards and is REPRESENTED by the latest pregame snapshot it has
reached:

    PUBLICATION_PREFERENCE = (CLOSE, MID, OPEN)

WHAT MUST NOT MOVE. That order is public presentation only. The immutable OPEN,
MID and CLOSE forecasts of record keep their own bytes, keep being graded
separately and chronologically, and are never rewritten, merged or back-filled
by publication -- which is asserted here directly rather than assumed, because
"display preference" and "evaluation provenance" are only separate concerns for
as long as something keeps checking.

THE EARLIER DEFECT THIS SUITE ALREADY COVERED. The publisher used to refuse a
week with nothing closed yet:

    FAIL_CLOSED: no game in the current week has a publishable snapshot yet
    -- refusing to publish an empty feed              (exit 2 -> workflow 10)

so every attempt exited non-zero, latest.json was never replaced, and the page
served the Week-1 card generated ``2026-09-15T05:14:43Z`` for eight days. Those
proofs are kept and extended: an empty week is still a valid publication, and a
PARTIALLY opened week now publishes exactly the games that have opened.

Hermetic: ``tmp_path`` estates and synthetic forecast records. No network, no
provider, no server, nothing published.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.production import snapshot_stages_2026 as st

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


feed = _load(REPO_ROOT / "scripts" / "export_current_week_nfl_feed.py", "_feed_pregame_board")
entry = _load(REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py", "_entry_pregame_board")

# One card: an afternoon Sunday game, a Sunday-nighter and a Monday-nighter --
# the rollover shape, with three games so one card can hold one game at each of
# the three stages at the same time.
SUNDAY = pd.Timestamp("2026-09-20T17:00:00Z")
SUNDAY_NIGHT = pd.Timestamp("2026-09-21T00:20:00Z")
MONDAY = pd.Timestamp("2026-09-22T00:15:00Z")
NEXT_THURSDAY = pd.Timestamp("2026-09-25T00:15:00Z")
OPEN_AT = pd.Timestamp("2026-09-14T14:00:00Z")

KICKOFFS = {"SUN": SUNDAY, "SNF": SUNDAY_NIGHT, "MON": MONDAY}

MODEL_RUN_AT = "2026-09-14T15:30:00Z"


def _record(game_id, *, cutoff, kickoff, margin, spread, run_created_at_utc=MODEL_RUN_AT):
    prediction = {
        "game_id": game_id, "horizon": "TUE", "target_cutoff_utc": str(cutoff),
        "season": 2026, "week": "2", "season_type": "REG",
        "home_team_id": "KC", "away_team_id": "BUF",
        "scheduled_kickoff_utc": str(kickoff).replace("+00:00", "Z").replace(" ", "T"),
        "prediction": {
            "model_status": "OOF", "predicted_margin": margin,
            "predicted_total": 45.0, "model_config_hash": "cfg",
        },
        "markets": {
            name: {
                "status": "OK",
                "market": {
                    "consensus_line": line, "eligible_books": 4,
                    # A stage's market is observed at its own cutoff and never
                    # later: this is the no-lookahead property the published
                    # row has to preserve.
                    "selected_returned_snapshot_timestamps": [
                        str(pd.Timestamp(cutoff) - pd.Timedelta(minutes=5)).replace("+00:00", "Z").replace(" ", "T")
                    ],
                },
            }
            for name, line in (("ATS", spread), ("TOTAL", 44.5))
        },
    }
    record = {
        "game_id": game_id, "horizon": "TUE", "target_cutoff_utc": str(cutoff),
        "run_id": "r1", "run_created_at_utc": run_created_at_utc,
        "created_at_utc": run_created_at_utc, "prediction": prediction,
    }
    record["prediction_hash"] = feed._v2._sha256_hex(prediction)
    return record


@pytest.fixture
def estate(tmp_path):
    card = pd.DataFrame(
        [{"game_id": game_id, "scheduled_kickoff_utc": kickoff} for game_id, kickoff in KICKOFFS.items()]
    )
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    forecast_dir.mkdir(parents=True)
    return {
        "root": tmp_path, "card": card, "forecast_dir": forecast_dir,
        "observations": {game_id: OPEN_AT for game_id in KICKOFFS},
        "output": tmp_path / "public" / "latest.json",
    }


def _cutoff(estate, game_id, stage):
    schedule = st.resolve_game_snapshots(
        game_id=game_id, scheduled_kickoff_utc=KICKOFFS[game_id],
        card_earliest_kickoff_utc=SUNDAY,
        open_observed_at_utc=estate["observations"].get(game_id),
    )
    return schedule.cutoff_for(stage)


def _snapshot_path(estate, game_id, stage):
    cutoff = _cutoff(estate, game_id, stage)
    safe = str(cutoff).replace(":", "").replace("+", "_")
    return estate["forecast_dir"] / f"{game_id}__{safe}.json"


def _write(estate, game_id, stage, *, margin=6.5, spread=-4.5, run_created_at_utc=MODEL_RUN_AT):
    record = _record(
        game_id, cutoff=_cutoff(estate, game_id, stage), kickoff=KICKOFFS[game_id],
        margin=margin, spread=spread, run_created_at_utc=run_created_at_utc,
    )
    _snapshot_path(estate, game_id, stage).write_text(json.dumps(record, indent=2, sort_keys=True))


def _publish(estate, *, week=2, generated_at="2026-09-22T06:00:00Z"):
    return feed.publish_current_week_feed(
        card=estate["card"], forecast_dir=estate["forecast_dir"],
        artifact_root=estate["root"], season=2026, week=week,
        generated_at_utc=generated_at, horizon="TUE",
        output_path=estate["output"], open_observations=estate["observations"],
    )


def _served(estate) -> dict:
    return json.loads(estate["output"].read_text())


def _rows(estate) -> dict[str, dict]:
    return {game["game_id"]: game for game in _served(estate)["games"]}


def _ledger_snapshot(estate) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(estate["forecast_dir"].glob("*.json"))}


# ===========================================================================
# 1. A current-week OPEN is publicly visible before MID or CLOSE exists.
# ===========================================================================
def test_the_publication_preference_is_close_then_mid_then_open():
    assert feed.PUBLICATION_PREFERENCE == (st.STAGE_CLOSE, st.STAGE_MID, st.STAGE_OPEN)


def test_a_game_with_only_an_open_snapshot_is_published(estate):
    _write(estate, "SUN", st.STAGE_OPEN)
    result = _publish(estate)

    assert result["status"] == "OK"
    assert result["game_count"] == 1
    assert result["stages"] == {"SUN": st.STAGE_OPEN}
    assert list(_rows(estate)) == ["SUN"]
    assert _rows(estate)["SUN"][feed.STAGE_KEY] == st.STAGE_OPEN


def test_an_open_only_game_is_not_frozen_and_can_still_advance(estate):
    """OPEN carries no one-way door: only CLOSE is frozen."""
    _write(estate, "SUN", st.STAGE_OPEN)
    result = _publish(estate)
    assert result["frozen_close_games"] == []


# ===========================================================================
# 2-3. MID replaces OPEN, then CLOSE replaces MID.
# ===========================================================================
def test_mid_replaces_open_for_the_same_game(estate):
    _write(estate, "SUN", st.STAGE_OPEN, spread=-1.5)
    _publish(estate)
    assert _rows(estate)["SUN"]["market_home_spread"] == -1.5

    _write(estate, "SUN", st.STAGE_MID, spread=-3.5)
    result = _publish(estate, generated_at="2026-09-19T06:00:00Z")

    assert result["stages"] == {"SUN": st.STAGE_MID}
    row = _rows(estate)["SUN"]
    assert row[feed.STAGE_KEY] == st.STAGE_MID
    assert row["market_home_spread"] == -3.5
    # Exactly one row for the game: MID replaces its public representation
    # rather than being added beside it.
    assert len(_served(estate)["games"]) == 1


def test_close_replaces_mid_for_the_same_game(estate):
    _write(estate, "SUN", st.STAGE_OPEN, spread=-1.5)
    _write(estate, "SUN", st.STAGE_MID, spread=-3.5)
    _publish(estate)
    assert _rows(estate)["SUN"][feed.STAGE_KEY] == st.STAGE_MID

    _write(estate, "SUN", st.STAGE_CLOSE, spread=-4.5)
    result = _publish(estate, generated_at="2026-09-20T16:30:00Z")

    row = _rows(estate)["SUN"]
    assert row[feed.STAGE_KEY] == st.STAGE_CLOSE
    assert row["market_home_spread"] == -4.5
    assert result["frozen_close_games"] == ["SUN"]
    assert len(_served(estate)["games"]) == 1


def test_a_published_close_is_never_reopened_by_an_earlier_stage(estate):
    """The one-way door still holds now that OPEN and MID are publishable."""
    _write(estate, "SUN", st.STAGE_CLOSE, spread=-4.5)
    _publish(estate)
    closed = _rows(estate)["SUN"]

    _write(estate, "SUN", st.STAGE_OPEN, spread=99.5)
    _write(estate, "SUN", st.STAGE_MID, spread=-99.5)
    result = _publish(estate, generated_at="2026-09-20T16:45:00Z")

    assert result["stages"] == {"SUN": st.STAGE_CLOSE}
    assert _rows(estate)["SUN"] == closed


# ===========================================================================
# 4. One card, three games, three different stages at the same time.
# ===========================================================================
def test_one_game_can_be_close_while_another_is_mid_and_another_is_open(estate):
    _write(estate, "SUN", st.STAGE_OPEN, spread=-1.5)
    _write(estate, "SUN", st.STAGE_MID, spread=-2.5)
    _write(estate, "SUN", st.STAGE_CLOSE, spread=-3.5)
    _write(estate, "SNF", st.STAGE_OPEN, spread=-5.5)
    _write(estate, "SNF", st.STAGE_MID, spread=-6.5)
    _write(estate, "MON", st.STAGE_OPEN, spread=-7.5)

    result = _publish(estate, generated_at="2026-09-20T16:30:00Z")

    assert result["stages"] == {
        "SUN": st.STAGE_CLOSE, "SNF": st.STAGE_MID, "MON": st.STAGE_OPEN,
    }
    rows = _rows(estate)
    assert rows["SUN"]["market_home_spread"] == -3.5
    assert rows["SNF"]["market_home_spread"] == -6.5
    assert rows["MON"]["market_home_spread"] == -7.5
    # Only the closed game is frozen; the other two are expected to advance.
    assert result["frozen_close_games"] == ["SUN"]


def test_every_row_names_its_own_stage_and_its_own_model_instant(estate):
    _write(estate, "SUN", st.STAGE_CLOSE, run_created_at_utc="2026-09-20T15:00:00Z")
    _write(estate, "SNF", st.STAGE_MID, run_created_at_utc="2026-09-18T15:00:00Z")
    _write(estate, "MON", st.STAGE_OPEN, run_created_at_utc="2026-09-14T15:00:00Z")
    _publish(estate, generated_at="2026-09-20T16:30:00Z")

    rows = _rows(estate)
    assert rows["SUN"][feed.MODEL_GENERATED_KEY] == "2026-09-20T15:00:00Z"
    assert rows["SNF"][feed.MODEL_GENERATED_KEY] == "2026-09-18T15:00:00Z"
    assert rows["MON"][feed.MODEL_GENERATED_KEY] == "2026-09-14T15:00:00Z"


# ===========================================================================
# 5. Partial availability publishes what exists.
# ===========================================================================
def test_one_unopened_market_does_not_hold_the_slate_back(estate):
    _write(estate, "SUN", st.STAGE_OPEN)
    _write(estate, "SNF", st.STAGE_OPEN)

    result = _publish(estate)

    assert sorted(_rows(estate)) == ["SNF", "SUN"]
    assert result["unpublishable"] == {"MON": "NO_SNAPSHOT_YET"}


def test_a_late_opening_game_appears_on_the_next_publication(estate):
    _write(estate, "SUN", st.STAGE_OPEN)
    _publish(estate)
    assert sorted(_rows(estate)) == ["SUN"]

    _write(estate, "MON", st.STAGE_OPEN)
    _publish(estate, generated_at="2026-09-15T06:00:00Z")
    assert sorted(_rows(estate)) == ["MON", "SUN"]


def test_a_game_with_no_snapshot_is_absent_rather_than_placeholdered(estate):
    _write(estate, "SUN", st.STAGE_OPEN)
    _publish(estate)
    assert "MON" not in _rows(estate)
    assert "MON" not in estate["output"].read_text()


# ===========================================================================
# 6. A week with nothing opened yet is still a valid publication.
# ===========================================================================
def test_a_week_with_no_snapshots_publishes_successfully(estate):
    result = _publish(estate)
    assert result["status"] == "OK_AWAITING_FIRST_SNAPSHOT"
    assert result["game_count"] == 0


def test_the_empty_envelope_names_the_correct_week(estate):
    _publish(estate)
    served = _served(estate)
    assert served["season"] == 2026
    assert served["week"] == 2
    assert served["games"] == []


def test_the_empty_envelope_is_schema_valid(estate):
    _publish(estate)
    served = _served(estate)
    assert served["schema_version"] == "wizard-nfl-pricing-v3"
    assert tuple(served.keys()) == feed.TOP_LEVEL_KEY_ORDER
    assert served["horizon"] == "TUE"
    assert served["generated_at_utc"].endswith("Z")


def test_the_empty_publication_is_not_a_failure(estate):
    """It must not raise, so the workflow cannot exit 10."""
    result = _publish(estate)
    assert result["status"].startswith("OK")
    assert estate["output"].is_file()


def test_a_stale_older_week_is_replaced_by_the_empty_current_week(estate):
    """Requirement D: publication never falls back to an older week."""
    estate["output"].parent.mkdir(parents=True, exist_ok=True)
    estate["output"].write_text(
        json.dumps({"schema_version": "wizard-nfl-pricing-v2", "season": 2026,
                    "week": 1, "horizon": "TUE",
                    "generated_at_utc": "2026-09-15T05:14:43Z", "games": [{"stale": True}]})
    )
    _publish(estate)
    served = _served(estate)
    assert served["week"] == 2
    assert served["games"] == []


# ===========================================================================
# 7. The immutable stage records are untouched by publication.
# ===========================================================================
def test_publication_writes_nothing_into_the_forecast_ledger(estate):
    _write(estate, "SUN", st.STAGE_OPEN, spread=-1.5)
    _write(estate, "SUN", st.STAGE_MID, spread=-2.5)
    _write(estate, "SUN", st.STAGE_CLOSE, spread=-3.5)
    _write(estate, "SNF", st.STAGE_OPEN, spread=-5.5)
    before = _ledger_snapshot(estate)

    _publish(estate, generated_at="2026-09-20T16:30:00Z")
    _publish(estate, generated_at="2026-09-20T16:40:00Z")

    assert _ledger_snapshot(estate) == before


def test_advancing_a_game_leaves_its_earlier_stage_records_byte_identical(estate):
    _write(estate, "SUN", st.STAGE_OPEN, spread=-1.5)
    open_bytes = _snapshot_path(estate, "SUN", st.STAGE_OPEN).read_bytes()
    _publish(estate)

    _write(estate, "SUN", st.STAGE_MID, spread=-2.5)
    mid_bytes = _snapshot_path(estate, "SUN", st.STAGE_MID).read_bytes()
    _publish(estate, generated_at="2026-09-19T06:00:00Z")

    _write(estate, "SUN", st.STAGE_CLOSE, spread=-3.5)
    _publish(estate, generated_at="2026-09-20T16:30:00Z")

    assert _snapshot_path(estate, "SUN", st.STAGE_OPEN).read_bytes() == open_bytes
    assert _snapshot_path(estate, "SUN", st.STAGE_MID).read_bytes() == mid_bytes


def test_only_close_is_ever_recorded_in_the_published_state_sidecar(estate):
    _write(estate, "SUN", st.STAGE_OPEN)
    _write(estate, "SNF", st.STAGE_MID)
    _publish(estate)
    sidecar = feed.sidecar_path(estate["root"], season=2026, week=2)
    assert json.loads(sidecar.read_text()) == {}

    _write(estate, "MON", st.STAGE_CLOSE)
    _publish(estate, generated_at="2026-09-21T23:30:00Z")
    assert sorted(json.loads(sidecar.read_text())) == ["MON"]


def test_the_sidecar_still_stores_the_certified_per_game_shape(estate):
    """The board's provenance lives BESIDE the frozen game, not inside it, so a
    sidecar written before the board carried provenance still validates."""
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    frozen = json.loads(feed.sidecar_path(estate["root"], season=2026, week=2).read_text())["SUN"]
    assert sorted(frozen["published_game"]) == sorted(feed.BASE_GAME_KEY_ORDER)
    assert feed.STAGE_KEY not in frozen["published_game"]


def test_a_sidecar_without_provenance_still_publishes(estate):
    """Exactly the state the live server is in before this change ships."""
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    sidecar = feed.sidecar_path(estate["root"], season=2026, week=2)
    state = json.loads(sidecar.read_text())
    del state["SUN"][feed.MODEL_GENERATED_KEY]
    sidecar.write_text(json.dumps(state, indent=2, sort_keys=True))

    _publish(estate, generated_at="2026-09-20T16:30:00Z")
    row = _rows(estate)["SUN"]
    assert row[feed.STAGE_KEY] == st.STAGE_CLOSE
    # The freeze instant it does carry, never a re-derivation from a later
    # snapshot.
    assert row[feed.MODEL_GENERATED_KEY] == state["SUN"]["frozen_at_utc"]


# ===========================================================================
# 8. No lookahead: a row is assembled from its OWN stage's record only.
# ===========================================================================
def test_a_published_row_carries_only_its_own_stage_market(estate):
    _write(estate, "SUN", st.STAGE_OPEN, spread=-1.5)
    _write(estate, "SUN", st.STAGE_CLOSE, spread=-9.5)
    # Published from CLOSE, because CLOSE exists...
    _publish(estate, generated_at="2026-09-20T16:30:00Z")
    assert _rows(estate)["SUN"]["market_home_spread"] == -9.5

    # ...and a fresh estate holding ONLY the OPEN publishes the OPEN market,
    # never a later one.
    estate["output"].unlink()
    feed.sidecar_path(estate["root"], season=2026, week=2).unlink()
    _snapshot_path(estate, "SUN", st.STAGE_CLOSE).unlink()
    _publish(estate, generated_at="2026-09-15T06:00:00Z")
    assert _rows(estate)["SUN"]["market_home_spread"] == -1.5


def test_every_published_market_instant_precedes_its_own_stage_cutoff(estate):
    for game_id, stage in (("SUN", st.STAGE_CLOSE), ("SNF", st.STAGE_MID), ("MON", st.STAGE_OPEN)):
        _write(estate, game_id, stage)
    _publish(estate, generated_at="2026-09-20T16:30:00Z")

    stages = _publish(estate, generated_at="2026-09-20T16:31:00Z")["stages"]
    for game_id, row in _rows(estate).items():
        cutoff = _cutoff(estate, game_id, stages[game_id])
        assert pd.Timestamp(row["market_as_of_utc"]) <= pd.Timestamp(cutoff)
        assert pd.Timestamp(row["market_as_of_utc"]) < pd.Timestamp(KICKOFFS[game_id])


def test_the_selector_reads_one_file_per_stage_named_by_that_stage_cutoff(estate):
    """The cutoff IS the filename, so a stage can only ever load its own row."""
    _write(estate, "SUN", st.STAGE_OPEN)
    chosen, _ = feed.select_publishable_snapshots(
        card=estate["card"], forecast_dir=estate["forecast_dir"],
        open_observations=estate["observations"],
    )
    stage, record = chosen["SUN"]
    assert stage == st.STAGE_OPEN
    assert pd.Timestamp(record["target_cutoff_utc"]) == _cutoff(estate, "SUN", st.STAGE_OPEN)


def test_the_evaluation_side_never_reads_the_public_preference():
    """Structural proof that display preference cannot reach grading.

    The graders, the results attachment and the recalibration candidate all read
    the immutable stage ledgers. None of them imports the board exporter, reads
    the published board, or knows the preference order exists -- so there is no
    path by which "CLOSE represents this game publicly" could become "CLOSE is
    the record this game is graded on".
    """
    for script in (
        "report_2026_snapshot_performance.py",
        "report_2026_prospective_performance.py",
        "attach_2026_results_from_population.py",
        "generate_2026_recalibration_candidate.py",
    ):
        source = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
        for forbidden in (
            "PUBLICATION_PREFERENCE",
            "export_current_week_nfl_feed",
            "publish_2026_current_week",
            "published_close_state",
            "latest.json",
            feed.STAGE_KEY + '"',
        ):
            assert forbidden not in source, f"{script} reached into publication: {forbidden}"


def test_publication_leaves_the_evaluation_and_performance_trees_alone(estate):
    """Publishing writes exactly two things: latest.json and its own sidecar."""
    for game_id, stage in (("SUN", st.STAGE_CLOSE), ("SNF", st.STAGE_MID), ("MON", st.STAGE_OPEN)):
        _write(estate, game_id, stage)
    _publish(estate, generated_at="2026-09-20T16:30:00Z")

    written = sorted(
        str(path.relative_to(estate["root"]))
        for path in estate["root"].rglob("*")
        if path.is_file() and "forecast-ledger" not in path.parts
    )
    assert written == [
        "production-2026/published-close-state/season=2026/week=02/published_close_state.json",
        "public/latest.json",
    ]


# ===========================================================================
# 9. Invalidated OPEN records cannot reappear.
# ===========================================================================
def test_an_invalidated_artifact_is_not_reachable_by_the_board(estate):
    """The board reads only the explicit forecast dir, so a quarantined row
    under invalidated/ is structurally out of reach -- including now that OPEN
    is publishable."""
    invalidated = estate["root"] / "invalidated" / "stage-open-current-capture-resolver-2026-09-19"
    invalidated.mkdir(parents=True)
    (invalidated / "SUN__OPEN.json").write_text(json.dumps({"snapshot_stage": "OPEN"}))

    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    assert list(_rows(estate)) == ["SUN"]
    assert "invalidated" not in estate["output"].read_text()


def test_an_open_with_no_recorded_observation_has_no_publishable_cutoff(estate):
    """Invalidating a defective OPEN removes the OBSERVATION, and OPEN's cutoff
    IS that observation, so the stage becomes unreachable rather than falling
    back to a guessed instant."""
    _write(estate, "SUN", st.STAGE_OPEN)
    orphan = _snapshot_path(estate, "SUN", st.STAGE_OPEN)
    assert orphan.is_file()

    estate["observations"] = {}
    result = _publish(estate)

    assert result["game_count"] == 0
    assert result["unpublishable"]["SUN"] == "NO_SNAPSHOT_YET"
    # The quarantined bytes still exist on disk and are simply never read.
    assert orphan.is_file()


# ===========================================================================
# 10. Historical repair can never publish.
# ===========================================================================
def test_the_stage_entrypoint_has_no_publication_path():
    import ast

    tree = ast.parse((REPO_ROOT / "scripts" / "run_2026_stage_snapshots.py").read_text(encoding="utf-8"))
    names = (
        {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    )
    for forbidden in (
        "publish_wizard_nfl_local", "publish_2026_current_week",
        "write_latest", "publish_current_week_feed",
    ):
        assert not any(forbidden in n for n in names), forbidden


def test_the_publisher_accepts_no_historical_week_override():
    """--season/--week exist only on the stage entrypoint, never here."""
    import contextlib
    import io

    parser_src = (REPO_ROOT / "scripts" / "publish_2026_current_week.py").read_text(encoding="utf-8")
    assert '"--season"' not in parser_src
    assert '"--week"' not in parser_src

    buffer = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buffer):
        _load(
            REPO_ROOT / "scripts" / "publish_2026_current_week.py", "_pub_help"
        ).main(["--help"])
    assert "--season" not in buffer.getvalue()


# ===========================================================================
# 11. The PR #57 publication-card rollover rule is unchanged.
# ===========================================================================
def _rollover_games() -> pd.DataFrame:
    rows = []
    for week, pairs in (
        (2, [(("HOU", "KC"), SUNDAY), (("BUF", "NYJ"), MONDAY)]),
        (3, [(("HOU", "KC"), NEXT_THURSDAY), (("BUF", "NYJ"), pd.Timestamp("2026-09-27T17:00:00Z"))]),
    ):
        for (away, home), kickoff in pairs:
            rows.append(
                {
                    "game_id": f"2026_{week:02d}_{away}_{home}",
                    "season": 2026, "week": week, "season_type": "REG",
                    "home_team_id": home, "away_team_id": away,
                    "scheduled_kickoff_utc": kickoff,
                    "home_score": None, "away_score": None, "neutral_site": False,
                }
            )
    return pd.DataFrame(rows)


def test_the_previous_week_stays_public_before_its_final_kickoff():
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-21T05:50:00Z")
    )
    assert info["week"] == "2"


def test_publication_hands_over_after_the_final_kickoff():
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-22T01:00:00Z")
    )
    assert info["week"] == "3"


def test_the_handover_still_happens_even_with_no_snapshot_on_the_new_week():
    """Requirement D again, at the card layer: a new week with nothing
    snapshotted must still become the publication card."""
    _, info = entry.resolve_publication_card(
        _rollover_games(), pd.Timestamp("2026-09-23T01:38:00Z")
    )
    assert info["week"] == "3"


# ===========================================================================
# The board's own public contract.
# ===========================================================================
def test_the_board_contract_is_the_certified_shape_plus_provenance():
    assert feed.BASE_GAME_KEY_ORDER == feed._v2.GAME_KEY_ORDER
    assert feed.GAME_KEY_ORDER == feed.BASE_GAME_KEY_ORDER + (
        feed.STAGE_KEY, feed.MODEL_GENERATED_KEY
    )
    assert feed.SCHEMA_VERSION == "wizard-nfl-pricing-v3"
    # The certified exporter keeps its own contract untouched, so its immutable
    # archives stay byte-valid.
    assert feed._v2.SCHEMA_VERSION == "wizard-nfl-pricing-v2"
    assert len(feed._v2.GAME_KEY_ORDER) == 11


def test_every_row_is_in_the_frozen_board_key_order(estate):
    for game_id, stage in (("SUN", st.STAGE_CLOSE), ("SNF", st.STAGE_MID), ("MON", st.STAGE_OPEN)):
        _write(estate, game_id, stage)
    _publish(estate, generated_at="2026-09-20T16:30:00Z")

    for row in _served(estate)["games"]:
        assert tuple(row.keys()) == feed.GAME_KEY_ORDER
        assert row[feed.STAGE_KEY] in st.SNAPSHOT_STAGES
        assert row[feed.MODEL_GENERATED_KEY].endswith("Z")


def test_an_unknown_stage_can_never_be_published():
    with pytest.raises(st.SnapshotStageError):
        feed._board_game(
            {key: None for key in feed.BASE_GAME_KEY_ORDER},
            stage="PREGAME", model_generated_at_utc="2026-09-20T15:00:00Z",
        )


def test_a_record_without_a_model_instant_fails_closed(estate):
    _write(estate, "SUN", st.STAGE_OPEN)
    path = _snapshot_path(estate, "SUN", st.STAGE_OPEN)
    record = json.loads(path.read_text())
    del record["run_created_at_utc"]
    path.write_text(json.dumps(record, indent=2, sort_keys=True))

    with pytest.raises(feed.WizardExportError, match="run_created_at_utc"):
        _publish(estate)


# ===========================================================================
# Output discipline: byte stability, atomicity, real corruption.
# ===========================================================================
def test_a_populated_board_is_byte_stable_across_republication(estate):
    _write(estate, "SUN", st.STAGE_OPEN)
    _publish(estate)
    first = estate["output"].read_bytes()
    _publish(estate)
    assert estate["output"].read_bytes() == first


def test_the_empty_envelope_is_the_same_contract_a_consumer_already_parses(estate):
    """A page that reads the populated board can read this one: identical
    top-level keys and types, only the games list is empty."""
    _publish(estate)
    empty = _served(estate)

    _write(estate, "SUN", st.STAGE_OPEN)
    _publish(estate, generated_at="2026-09-15T06:00:00Z")
    populated = _served(estate)

    assert tuple(empty.keys()) == tuple(populated.keys()) == feed.TOP_LEVEL_KEY_ORDER
    for key in ("schema_version", "season", "week", "horizon"):
        assert type(empty[key]) is type(populated[key])
    assert isinstance(empty["games"], list)


def test_publication_stays_atomic_with_no_leftover_temp_file(estate):
    _publish(estate)
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate, generated_at="2026-09-20T16:30:00Z")
    assert not list(estate["output"].parent.glob("*.tmp"))
    assert not list(estate["output"].parent.glob(".*.tmp"))


def test_genuine_corruption_still_fails_closed(estate):
    """Fail-closed is reserved for real inconsistency, not an empty week."""
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    _write(estate, "SUN", st.STAGE_CLOSE, margin=99.0)
    with pytest.raises(feed.WizardExportError, match="already-closed game is never mutated"):
        _publish(estate, generated_at="2026-09-20T16:30:00Z")


def test_a_frozen_close_survives_its_snapshot_file_vanishing(estate):
    _write(estate, "SUN", st.STAGE_CLOSE)
    _publish(estate)
    frozen = _rows(estate)["SUN"]

    _snapshot_path(estate, "SUN", st.STAGE_CLOSE).unlink()
    result = _publish(estate, generated_at="2026-09-20T16:30:00Z")

    assert result["stages"] == {"SUN": st.STAGE_CLOSE}
    assert _rows(estate)["SUN"] == frozen
