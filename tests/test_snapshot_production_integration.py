"""Integration proofs for the OPEN / MID / CLOSE production call chain.

These do NOT test the stage modules in isolation -- that is
``test_snapshot_stages_2026.py`` and ``test_snapshot_performance_ledger_2026.py``.
Here the REAL chain runs:

    scheduler  ->  stage execution
    execution  ->  certified Ridge fit at the stage instant
    forecast   ->  immutable forecast-of-record ledger
    forecast   ->  performance ledger
    CLOSE      ->  current-week publication, frozen per game
    results    ->  later training eligibility
    ledger     ->  season CSV/JSON reporting

Hermetic: synthetic games injected into the real ``run_horizon_batch``, a
synthetic calibration seed, and ``tmp_path`` roots. No network, no real
artifact root, no real capture, nothing published.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_hybrid.features import horizon_elo as he
from nfl_hybrid.production import run_2026 as prod
from nfl_hybrid.production import snapshot_execution_2026 as ex
from nfl_hybrid.production import snapshot_performance_ledger_2026 as pl
from nfl_hybrid.production import snapshot_stages_2026 as st

from test_production_card_2026 import (  # noqa: E402
    _synthetic_games,
    _write_synthetic_calibration_seed,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_feed_exporter():
    path = _REPO_ROOT / "scripts" / "export_current_week_nfl_feed.py"
    spec = importlib.util.spec_from_file_location("export_current_week_nfl_feed", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_current_week_nfl_feed"] = module
    spec.loader.exec_module(module)
    return module


feed = _load_feed_exporter()


# ---------------------------------------------------------------------------
# a staggered final card: Thursday, two Sunday slots, Monday night
# ---------------------------------------------------------------------------
def _staggered_games() -> pd.DataFrame:
    games = _synthetic_games().copy()
    last_week = int(games["week"].max())
    rows = games.index[games["week"] == last_week].tolist()
    assert len(rows) == 4
    monday = games.loc[rows[0], "scheduled_kickoff_utc"].normalize() - pd.Timedelta(days=6)
    offsets = [
        pd.Timedelta(days=3, hours=0, minutes=15),  # Thu 8:15pm ET -> Fri 00:15Z
        pd.Timedelta(days=6, hours=17),  # Sun 1:00pm ET
        pd.Timedelta(days=6, hours=17),  # Sun 1:00pm ET -- shares a CLOSE
        pd.Timedelta(days=7, hours=0, minutes=15),  # Mon 8:15pm ET -> Tue 00:15Z
    ]
    for row, offset in zip(rows, offsets, strict=True):
        games.loc[row, "scheduled_kickoff_utc"] = monday + offset
    return games


def _card_of(games: pd.DataFrame) -> pd.DataFrame:
    last_week = int(games["week"].max())
    return games[games["week"] == last_week][["game_id", "scheduled_kickoff_utc"]].reset_index(drop=True)


@pytest.fixture
def estate(tmp_path):
    games = _staggered_games()
    _write_synthetic_calibration_seed(tmp_path)
    return {"root": tmp_path, "games": games, "card": _card_of(games)}


def _kickoffs(card: pd.DataFrame) -> dict[str, pd.Timestamp]:
    return {str(r.game_id): pd.Timestamp(r.scheduled_kickoff_utc) for r in card.itertuples(index=False)}


def _open_observations(card: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """A plausible board-opening instant well before every kickoff."""
    earliest = st.card_earliest_kickoff_utc(card["scheduled_kickoff_utc"])
    return {gid: earliest - pd.Timedelta(days=5) for gid in _kickoffs(card)}


def _run_stage(estate, stage, as_of, **kw):
    return ex.run_due_stage_snapshots(
        stage=stage,
        horizon="TUE",
        card=estate["card"],
        as_of_utc=as_of,
        operational_root=estate["root"],
        games=estate["games"],
        calibrator_root=estate["root"],
        open_observations=_open_observations(estate["card"]),
        **kw,
    )


# ===========================================================================
# 1. scheduler -> stage execution
# ===========================================================================
def test_the_scheduler_groups_games_sharing_a_kickoff_into_one_close_batch(estate):
    card = estate["card"]
    latest = max(_kickoffs(card).values())
    batches, skipped = ex.plan_stage_batches(
        stage=st.STAGE_CLOSE, card=card, as_of_utc=latest, open_observations=_open_observations(card)
    )
    # Thu, the shared Sunday pair, and Mon -> three instants for four games.
    assert len(batches) == 3
    assert sorted(len(b.game_ids) for b in batches) == [1, 1, 2]
    assert not skipped
    for batch in batches:
        assert len({_kickoffs(card)[g] for g in batch.game_ids}) == 1


def test_a_future_stage_instant_is_not_executed(estate):
    card = estate["card"]
    earliest_close = min(st.close_cutoff_utc(k) for k in _kickoffs(card).values())
    batches, skipped = ex.plan_stage_batches(
        stage=st.STAGE_CLOSE,
        card=card,
        as_of_utc=earliest_close - pd.Timedelta(minutes=1),
        open_observations=_open_observations(card),
    )
    assert batches == []
    assert set(skipped.values()) == {ex.SKIP_NOT_DUE}


def test_close_batches_become_due_one_kickoff_at_a_time(estate):
    card = estate["card"]
    instants = sorted({st.close_cutoff_utc(k) for k in _kickoffs(card).values()})
    seen = []
    for instant in instants:
        batches, _ = ex.plan_stage_batches(
            stage=st.STAGE_CLOSE, card=card, as_of_utc=instant, open_observations=_open_observations(card)
        )
        seen.append(len(batches))
    assert seen == [1, 2, 3]


def test_mid_is_one_batch_for_normal_games_and_its_own_for_the_thursday_game(estate):
    card = estate["card"]
    latest = max(_kickoffs(card).values())
    batches, _ = ex.plan_stage_batches(
        stage=st.STAGE_MID, card=card, as_of_utc=latest, open_observations=_open_observations(card)
    )
    sizes = sorted(len(b.game_ids) for b in batches)
    assert sizes == [1, 3]  # Thursday midpoint alone; Sun/Sun/Mon share Friday noon


def test_every_planned_stage_instant_is_strictly_pregame(estate):
    card = estate["card"]
    kickoffs = _kickoffs(card)
    latest = max(kickoffs.values())
    for stage in st.SNAPSHOT_STAGES:
        batches, _ = ex.plan_stage_batches(
            stage=stage, card=card, as_of_utc=latest, open_observations=_open_observations(card)
        )
        for batch in batches:
            for game_id in batch.game_ids:
                assert batch.cutoff_utc < kickoffs[game_id]


# ===========================================================================
# 2-3. execution -> forecast -> immutable ledger
# ===========================================================================
def test_a_close_execution_writes_the_forecast_of_record_at_the_stage_instant(estate):
    card = estate["card"]
    result = _run_stage(estate, st.STAGE_CLOSE, max(_kickoffs(card).values()))
    assert result["status"] == "OK"
    assert result["due_batches"] == 3

    ledger = estate["root"] / "production-2026" / "forecast-ledger" / "TUE"
    records = [json.loads(p.read_text()) for p in sorted(ledger.glob("*.json"))]
    assert len(records) == 4
    for record in records:
        assert record["prediction"]["snapshot_stage"] == st.STAGE_CLOSE
        cutoff = pd.Timestamp(record["target_cutoff_utc"])
        assert cutoff == st.close_cutoff_utc(_kickoffs(card)[record["game_id"]])


def test_the_stage_forecast_uses_the_certified_model_unchanged(estate):
    """Same feature semantics hash and model config hash as a card run -- a
    stage is an as-of instant, not a new model family."""
    card = estate["card"]
    _run_stage(estate, st.STAGE_CLOSE, max(_kickoffs(card).values()))
    ledger = estate["root"] / "production-2026" / "forecast-ledger" / "TUE"
    record = json.loads(sorted(ledger.glob("*.json"))[0].read_text())

    card_run = prod.run_horizon_batch(
        horizon="TUE", as_of_utc=max(_kickoffs(card).values()), force=True,
        operational_root=estate["root"], games=estate["games"], calibrator_root=estate["root"],
    )
    assert record["prediction"]["horizon_feature_semantics_hash"] == card_run["input_hashes"]["horizon_feature_semantics_hash"]
    assert record["prediction"]["operational_model_spec_hash"] == card_run["input_hashes"]["operational_model_spec_hash"]
    assert record["prediction"]["certified_baseline_sha"] == prod.CERTIFIED_SHA


def test_re_executing_the_same_stage_is_idempotent(estate):
    card = estate["card"]
    as_of = max(_kickoffs(card).values())
    _run_stage(estate, st.STAGE_CLOSE, as_of)
    ledger = estate["root"] / "production-2026" / "forecast-ledger" / "TUE"
    before = {p: p.read_bytes() for p in sorted(ledger.glob("*.json"))}

    second = _run_stage(estate, st.STAGE_CLOSE, as_of)
    assert second["status"] == "OK"
    assert {p: p.read_bytes() for p in sorted(ledger.glob("*.json"))} == before
    for execution in second["executions"]:
        assert all(w.endswith("IDEMPOTENT_NOOP") for w in execution["ledger_writes"])


def test_a_stage_can_never_price_a_game_outside_the_certified_card(estate):
    card = estate["card"]
    batches, _ = ex.plan_stage_batches(
        stage=st.STAGE_CLOSE, card=card, as_of_utc=max(_kickoffs(card).values()),
        open_observations=_open_observations(card),
    )
    bogus = ex.StageBatch(stage=st.STAGE_CLOSE, cutoff_utc=batches[0].cutoff_utc, game_ids=("not-a-real-game",))
    execution = ex.execute_stage_batch(
        bogus, horizon="TUE", operational_root=estate["root"],
        as_of_utc=max(_kickoffs(card).values()), games=estate["games"], calibrator_root=estate["root"],
    )
    assert execution.run_status == "SCHEDULE_UNAVAILABLE"
    assert execution.ledger_writes == ()


def test_the_three_stages_produce_three_distinct_forecasts_of_record(estate):
    card = estate["card"]
    as_of = max(_kickoffs(card).values())
    for stage in st.SNAPSHOT_STAGES:
        assert _run_stage(estate, stage, as_of)["status"] == "OK"

    ledger = estate["root"] / "production-2026" / "forecast-ledger" / "TUE"
    stages = [json.loads(p.read_text())["prediction"]["snapshot_stage"] for p in ledger.glob("*.json")]
    assert sorted(set(stages)) == ["CLOSE", "MID", "OPEN"]
    assert len(stages) == 12  # four games x three stages


# ===========================================================================
# 4. performance ledger is fed by live execution
# ===========================================================================
def test_the_performance_ledger_is_appended_automatically(estate):
    card = estate["card"]
    as_of = max(_kickoffs(card).values())
    for stage in st.SNAPSHOT_STAGES:
        _run_stage(estate, stage, as_of)

    rows = pl.build_report_rows(estate["root"])
    assert len(rows) == 12
    assert {row["snapshot_stage"] for row in rows} == set(st.SNAPSHOT_STAGES)
    for row in rows:
        assert row["outcome_status"] == pl.OUTCOME_PENDING
        assert row["model_home_margin"] is not None


def test_each_performance_row_carries_the_forecasts_own_provenance(estate):
    card = estate["card"]
    _run_stage(estate, st.STAGE_CLOSE, max(_kickoffs(card).values()))

    ledger = estate["root"] / "production-2026" / "forecast-ledger" / "TUE"
    record = json.loads(sorted(ledger.glob("*.json"))[0].read_text())
    snapshot = pl.read_snapshot(
        estate["root"], season=record["prediction"]["season"], week=record["prediction"]["week"],
        game_id=record["game_id"], stage=st.STAGE_CLOSE,
    )
    assert snapshot["forecast_prediction_hash"] == record["prediction_hash"]
    assert snapshot["model_home_margin"] == record["prediction"]["prediction"]["predicted_margin"]
    assert snapshot["active_calibrator_source"] == record["prediction"]["active_calibrator_source"]
    assert snapshot["model_sha"] == prod.CERTIFIED_SHA
    # Run identity is the run that WROTE the forecast, not the pass that
    # happened to record the performance row.
    assert snapshot["forecast_run_id"] == record["run_id"]
    assert snapshot["production_git_commit"] == record["git_commit"]


def test_a_replayed_performance_row_is_byte_identical(estate):
    """A performance row must not embed the replaying pass's run_id, or the
    ledger's own immutability check would reject an honest retry."""
    card = estate["card"]
    as_of = max(_kickoffs(card).values())
    _run_stage(estate, st.STAGE_CLOSE, as_of)
    root = pl.ledger_root(estate["root"])
    before = {p: p.read_bytes() for p in sorted(root.rglob("*.json"))}

    _run_stage(estate, st.STAGE_CLOSE, as_of)

    assert {p: p.read_bytes() for p in sorted(root.rglob("*.json"))} == before


def test_a_failed_batch_records_no_performance_row(estate):
    card = estate["card"]
    earliest_close = min(st.close_cutoff_utc(k) for k in _kickoffs(card).values())
    bad = ex.StageBatch(stage=st.STAGE_CLOSE, cutoff_utc=earliest_close, game_ids=("nope",))
    ex.execute_stage_batch(
        bad, horizon="TUE", operational_root=estate["root"], as_of_utc=earliest_close,
        games=estate["games"], calibrator_root=estate["root"],
    )
    assert pl.build_report_rows(estate["root"]) == []


# ===========================================================================
# 5. results attach, and feed later training eligibility
# ===========================================================================
def test_results_attach_without_changing_any_forecast(estate):
    card = estate["card"]
    _run_stage(estate, st.STAGE_CLOSE, max(_kickoffs(card).values()))
    ledger = estate["root"] / "production-2026" / "forecast-ledger" / "TUE"
    before = {p: p.read_bytes() for p in sorted(ledger.glob("*.json"))}

    last_week = int(estate["games"]["week"].max())
    finals = estate["games"][estate["games"]["week"] == last_week]
    report = ex.attach_final_results(
        operational_root=estate["root"], games=finals, result_attached_at_utc="2023-12-19T04:00:00Z"
    )
    assert len(report["attached"]) == 4
    assert report["pending"] == []
    assert {p: p.read_bytes() for p in sorted(ledger.glob("*.json"))} == before


def test_an_unresolved_game_is_left_pending_not_defaulted(estate):
    last_week = int(estate["games"]["week"].max())
    unplayed = estate["games"][estate["games"]["week"] == last_week].copy()
    unplayed["home_score"] = None
    unplayed["away_score"] = None
    report = ex.attach_final_results(
        operational_root=estate["root"], games=unplayed, result_attached_at_utc="2023-12-19T04:00:00Z"
    )
    assert report["attached"] == []
    assert len(report["pending"]) == 4


def test_a_provider_final_game_becomes_eligible_for_a_later_stage_fit(estate):
    """The Thursday game resolves before the Sunday CLOSE, so it is in the
    Sunday fit's training membership and not in the Thursday fit's."""
    card = estate["card"]
    kickoffs = _kickoffs(card)
    thursday_id = min(kickoffs, key=lambda g: kickoffs[g])
    thursday_close = st.close_cutoff_utc(kickoffs[thursday_id])
    sunday_close = sorted({st.close_cutoff_utc(k) for k in kickoffs.values()})[1]

    ledger = he.build_horizon_membership_ledger(estate["games"])
    from nfl_hybrid.evaluation import official_horizon_oof as ohf

    matrix = ohf.build_official_horizon_matrix(estate["games"], "TUE", ledger)
    results = {}
    for label, instant in (("thu", thursday_close), ("sun", sunday_close)):
        staged = matrix.copy()
        staged.loc[staged["game_id"].astype(str) == thursday_id, "target_cutoff_utc"] = instant
        predictions, _ = ohf.generate_official_horizon_oof_predictions(staged, horizon="TUE")
        row = predictions[predictions["game_id"] == thursday_id].iloc[0]
        results[label] = set(row["training_game_ids"])

    # The Sunday-instant fit trains on strictly more history than the
    # Thursday-instant one, and the extra rows are games that resolved in
    # between.
    assert results["sun"] > results["thu"]


def test_an_unresolved_game_never_enters_a_stage_fit(estate):
    """Blank the last card's scores and prove no fit trains on them, at any
    stage instant."""
    from nfl_hybrid.evaluation import official_horizon_oof as ohf

    games = estate["games"].copy()
    last_week = int(games["week"].max())
    unplayed_ids = set(games.loc[games["week"] == last_week, "game_id"].astype(str))
    games.loc[games["week"] == last_week, ["home_score", "away_score"]] = None

    ledger = he.build_horizon_membership_ledger(games)
    matrix = ohf.build_official_horizon_matrix(games, "TUE", ledger)
    kickoffs = _kickoffs(_card_of(games))
    latest_close = max(st.close_cutoff_utc(k) for k in kickoffs.values())
    matrix.loc[matrix["game_id"].astype(str).isin(unplayed_ids), "target_cutoff_utc"] = latest_close

    predictions, _ = ohf.generate_official_horizon_oof_predictions(matrix, horizon="TUE")
    for row in predictions.itertuples(index=False):
        assert not (set(row.training_game_ids) & unplayed_ids)


# ===========================================================================
# 6. season reporting from real ledger rows
# ===========================================================================
def test_season_reporting_compares_the_model_against_open_and_close(estate, tmp_path):
    card = estate["card"]
    as_of = max(_kickoffs(card).values())
    for stage in st.SNAPSHOT_STAGES:
        _run_stage(estate, stage, as_of)
    last_week = int(estate["games"]["week"].max())
    ex.attach_final_results(
        operational_root=estate["root"],
        games=estate["games"][estate["games"]["week"] == last_week],
        result_attached_at_utc="2023-12-19T04:00:00Z",
    )

    written = ex.write_season_reports(operational_root=estate["root"], output_dir=tmp_path / "reports")
    assert written["snapshot_rows"] == 12
    assert written["graded_rows"] == 12

    report = json.loads(Path(written["json_path"]).read_text())
    for stage in (st.STAGE_OPEN, st.STAGE_CLOSE):
        assert report["by_stage"][stage]["graded_snapshots"] == 4
        assert report["by_stage"][stage]["model_margin_mae"] is not None

    csv_text = Path(written["csv_path"]).read_text()
    assert csv_text.split("\n")[0] == ",".join(pl.REPORT_COLUMNS)
    assert len(csv_text.strip().split("\n")) == 13


# ===========================================================================
# 7. CLOSE -> current-week publication, frozen per game
# ===========================================================================
def _public_record(game_id, *, cutoff, kickoff, margin, total, spread, market_total, run_created="2026-09-14T15:30:00Z"):
    prediction = {
        "game_id": game_id, "horizon": "TUE", "target_cutoff_utc": str(cutoff),
        "season": 2026, "week": "2", "season_type": "REG",
        "home_team_id": "KC", "away_team_id": "BUF",
        "scheduled_kickoff_utc": str(kickoff).replace("+00:00", "Z").replace(" ", "T"),
        "prediction": {
            "model_status": "OOF", "predicted_margin": margin,
            "predicted_total": total, "model_config_hash": "cfg",
        },
        "markets": {
            name: {
                "status": "OK",
                "market": {
                    "consensus_line": line, "eligible_books": 4,
                    "selected_returned_snapshot_timestamps": ["2026-09-20T15:55:00Z"],
                },
            }
            for name, line in (("ATS", spread), ("TOTAL", market_total))
        },
    }
    record = {
        "game_id": game_id, "horizon": "TUE", "target_cutoff_utc": str(cutoff),
        "run_id": "r1", "run_created_at_utc": run_created, "created_at_utc": run_created,
        "prediction": prediction,
    }
    record["prediction_hash"] = feed._v2._sha256_hex(prediction)
    return record


@pytest.fixture
def feed_estate(tmp_path):
    """Two games: one that will reach CLOSE first, one kicking off later."""
    early_kick = pd.Timestamp("2026-09-20T17:00:00Z")
    late_kick = pd.Timestamp("2026-09-22T00:15:00Z")
    card = pd.DataFrame(
        [
            {"game_id": "EARLY", "scheduled_kickoff_utc": early_kick},
            {"game_id": "LATE", "scheduled_kickoff_utc": late_kick},
        ]
    )
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    forecast_dir.mkdir(parents=True)
    observations = {"EARLY": pd.Timestamp("2026-09-14T14:00:00Z"), "LATE": pd.Timestamp("2026-09-14T14:00:00Z")}
    return {
        "root": tmp_path, "card": card, "forecast_dir": forecast_dir,
        "observations": observations, "early_kick": early_kick, "late_kick": late_kick,
        "output": tmp_path / "public" / "latest.json",
    }


def _write(feed_estate, game_id, stage, **kw):
    kick = feed_estate["early_kick"] if game_id == "EARLY" else feed_estate["late_kick"]
    schedule = st.resolve_game_snapshots(
        game_id=game_id, scheduled_kickoff_utc=kick,
        card_earliest_kickoff_utc=feed_estate["early_kick"],
        open_observed_at_utc=feed_estate["observations"][game_id],
    )
    cutoff = schedule.cutoff_for(stage)
    record = _public_record(game_id, cutoff=cutoff, kickoff=kick, **kw)
    safe = str(cutoff).replace(":", "").replace("+", "_")
    (feed_estate["forecast_dir"] / f"{game_id}__{safe}.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def _publish(feed_estate, generated_at="2026-09-20T16:05:00Z"):
    return feed.publish_current_week_feed(
        card=feed_estate["card"], forecast_dir=feed_estate["forecast_dir"],
        artifact_root=feed_estate["root"], season=2026, week=2,
        generated_at_utc=generated_at, horizon="TUE",
        output_path=feed_estate["output"], open_observations=feed_estate["observations"],
    )


def test_the_feed_publishes_the_current_week_with_a_mixture_of_stages(feed_estate):
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    _write(feed_estate, "LATE", st.STAGE_MID, margin=2.0, total=43.0, spread=-1.5, market_total=42.5)

    result = _publish(feed_estate)
    assert result["status"] == "OK"
    assert result["game_count"] == 2
    assert result["stages"] == {"EARLY": st.STAGE_CLOSE, "LATE": st.STAGE_MID}
    assert result["frozen_close_games"] == ["EARLY"]

    published = json.loads(feed_estate["output"].read_text())
    assert published["schema_version"] == "wizard-nfl-pricing-v2"
    assert tuple(published.keys()) == feed.TOP_LEVEL_KEY_ORDER
    for game in published["games"]:
        assert tuple(game.keys()) == feed.GAME_KEY_ORDER


def test_the_public_schema_gains_no_stage_field(feed_estate):
    """Stage is operational bookkeeping and stays out of the public payload."""
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    _publish(feed_estate)
    published = json.loads(feed_estate["output"].read_text())
    assert "snapshot_stage" not in published
    for game in published["games"]:
        assert "snapshot_stage" not in game


def test_a_future_game_keeps_updating_until_its_own_close(feed_estate):
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    _write(feed_estate, "LATE", st.STAGE_MID, margin=2.0, total=43.0, spread=-1.5, market_total=42.5)
    _publish(feed_estate)
    before = json.loads(feed_estate["output"].read_text())
    late_before = [g for g in before["games"] if g["game_id"] == "LATE"][0]

    # LATE now reaches its own CLOSE with a different projection.
    _write(feed_estate, "LATE", st.STAGE_CLOSE, margin=3.5, total=44.0, spread=-2.5, market_total=43.5)
    result = _publish(feed_estate, generated_at="2026-09-21T23:20:00Z")

    after = json.loads(feed_estate["output"].read_text())
    late_after = [g for g in after["games"] if g["game_id"] == "LATE"][0]
    assert late_after != late_before
    assert late_after["predicted_home_margin"] == 3.5
    assert result["stages"]["LATE"] == st.STAGE_CLOSE
    assert result["frozen_close_games"] == ["EARLY", "LATE"]


def test_an_already_closed_game_is_frozen_across_later_publications(feed_estate):
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    _publish(feed_estate)
    frozen = [g for g in json.loads(feed_estate["output"].read_text())["games"] if g["game_id"] == "EARLY"][0]

    _write(feed_estate, "LATE", st.STAGE_MID, margin=2.0, total=43.0, spread=-1.5, market_total=42.5)
    _publish(feed_estate, generated_at="2026-09-21T12:00:00Z")

    still = [g for g in json.loads(feed_estate["output"].read_text())["games"] if g["game_id"] == "EARLY"][0]
    assert still == frozen


def test_a_later_pass_that_would_change_a_closed_game_fails_the_whole_publication(feed_estate):
    """The one-way door. Not a silent overwrite, not a partial publish."""
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    _publish(feed_estate)
    published = feed_estate["output"].read_bytes()

    # Re-derive EARLY's CLOSE forecast with a different projection.
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=99.0, total=47.0, spread=-4.5, market_total=45.5)
    with pytest.raises(feed.WizardExportError, match="already-closed game is never mutated"):
        _publish(feed_estate, generated_at="2026-09-21T12:00:00Z")

    assert feed_estate["output"].read_bytes() == published


def test_a_frozen_close_survives_its_snapshot_file_disappearing(feed_estate):
    """The published close outlives the ledger lookup that produced it."""
    record = _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    _publish(feed_estate)
    for path in feed_estate["forecast_dir"].glob("EARLY__*.json"):
        path.unlink()

    result = _publish(feed_estate, generated_at="2026-09-21T12:00:00Z")
    assert result["stages"]["EARLY"] == st.STAGE_CLOSE
    published = json.loads(feed_estate["output"].read_text())
    assert [g["game_id"] for g in published["games"]] == ["EARLY"]
    assert record["prediction"]["prediction"]["predicted_margin"] == 6.5


def test_publication_is_atomic_and_leaves_no_temporary_file(feed_estate):
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    _publish(feed_estate)
    assert not list(feed_estate["output"].parent.glob("*.tmp"))
    assert not list(feed_estate["output"].parent.glob(".*.tmp"))
    assert json.loads(feed_estate["output"].read_text())["games"]


def test_a_game_with_no_snapshot_is_absent_rather_than_placeheld(feed_estate):
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    result = _publish(feed_estate)
    assert result["unpublishable"] == {"LATE": "NO_SNAPSHOT_YET"}
    assert [g["game_id"] for g in json.loads(feed_estate["output"].read_text())["games"]] == ["EARLY"]


def test_an_empty_week_refuses_to_publish(feed_estate):
    with pytest.raises(feed.WizardExportError, match="refusing to publish an empty feed"):
        _publish(feed_estate)
    assert not feed_estate["output"].exists()


def test_the_published_bytes_are_reported_for_the_public_verifier(feed_estate):
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    result = _publish(feed_estate)
    from hashlib import sha256

    assert result["published_sha256"] == sha256(feed_estate["output"].read_bytes()).hexdigest()


def test_the_close_state_sidecar_lives_in_the_existing_production_tree(feed_estate):
    _write(feed_estate, "EARLY", st.STAGE_CLOSE, margin=6.5, total=47.0, spread=-4.5, market_total=45.5)
    result = _publish(feed_estate)
    relative = Path(result["close_state_path"]).relative_to(feed_estate["root"])
    assert relative.parts[0] == "production-2026"
    assert relative.parts[1] == "published-close-state"
