"""Proofs that publishing an immutable card is replayable.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
A certified Week-1 TUE run succeeded (``run_status=SUCCESS``,
``forecast_count=16``, ``game_count=16``), but the publish replay failed:

    selected 0 forecast record(s) for run '20260908T160500Z__TUE__30b2bf7b'
    but the run manifest declares game_count=16

The forecast ledger is immutable and first-write-wins, so an identical replay
is an IDEMPOTENT_NOOP: the rows keep the ``run_id`` of the run that first
wrote them while every later invocation mints a fresh one. Selecting forecasts
by ``record["run_id"] == manifest["run_id"]`` therefore matched nothing on any
replay, and the exporter called a complete card partial.

Two things had to change together. Selection is now keyed to the CARD --
``(horizon, target_cutoff_utc)`` -- with membership proved by the declared
game count and forecast batch hash. And ``generated_at_utc`` is read from the
immutable rows rather than the replay's manifest, without which a replay would
serialize different bytes and collide with its own immutable archive.

Everything here is HERMETIC and SYNTHETIC: ``tmp_path`` estates and the
existing synthetic forecast/manifest builders. No real Week-1 forecast, no
live provider, no publication.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_export_wizard_nfl_pricing import (  # noqa: E402
    DEFAULT_CUTOFF,
    DEFAULT_RUN_ID,
    _forecast_record,
    _manifest,
    _write_manifest,
    _write_record,
    exp,
)

# The replay run_id from the real failure, and the instant it resolved.
REPLAY_RUN_ID = "20260908T160500Z__TUE__30b2bf7b"
REPLAY_RUN_CREATED = "2026-09-08T16:05:00Z"
# What the rows actually carry, from the run that first wrote them.
ORIGINAL_RUN_CREATED = "2026-09-08T15:30:00Z"


def _card_estate(tmp_path: Path, *, game_count: int = 16) -> tuple[Path, Path, list[dict]]:
    """One certified TUE card of ``game_count`` games, written once by an
    original run -- the Week-1 shape, at Week-1 scale by default."""
    forecast_dir = tmp_path / "forecast-ledger" / "TUE"
    records = []
    for index in range(game_count):
        record = _forecast_record(
            game_id=f"G{index + 1:02d}",
            run_id=DEFAULT_RUN_ID,
            run_created_at_utc=ORIGINAL_RUN_CREATED,
            prediction_kwargs=dict(
                scheduled_kickoff_utc=f"2026-09-08T{17 + (index % 6):02d}:00:00Z",
                predicted_margin=1.5 + index,
                predicted_total=44.0 + index,
            ),
        )
        _write_record(forecast_dir, record)
        records.append(record)
    manifest_path = _write_manifest(
        tmp_path / "run-manifests", _manifest(records=records, run_created_at_utc=ORIGINAL_RUN_CREATED)
    )
    return manifest_path, forecast_dir, records


def _replay_manifest(tmp_path: Path, records: list[dict], **overrides) -> Path:
    """The manifest a REPLAY writes: a brand-new run_id and its own creation
    instant, over forecast rows it did not write."""
    manifest = _manifest(records=records, run_id=REPLAY_RUN_ID, run_created_at_utc=REPLAY_RUN_CREATED)
    manifest.update(overrides)
    return _write_manifest(tmp_path / "run-manifests", manifest)


def _export(manifest_path: Path, forecast_dir: Path, output: Path) -> dict:
    return exp.run_export(
        run_manifest_path=manifest_path, forecast_dir=forecast_dir, output_path=output
    )


def _archive_bytes(output: Path) -> bytes:
    archives = sorted((output.parent / "archive").rglob("*.json"))
    assert len(archives) == 1, archives
    return archives[0].read_bytes()


# ===========================================================================
# PART 1 -- the replay path that was broken.
# ===========================================================================
def test_first_execution_publishes_the_whole_card(tmp_path):
    manifest_path, forecast_dir, records = _card_estate(tmp_path)
    result = _export(manifest_path, forecast_dir, tmp_path / "public" / "latest.json")
    assert result["status"] == "OK"
    assert result["game_count"] == len(records) == 16


def test_an_exact_idempotent_replay_republishes_identical_bytes(tmp_path):
    """Re-exporting from the ORIGINAL manifest is unchanged behaviour."""
    manifest_path, forecast_dir, _ = _card_estate(tmp_path)
    output = tmp_path / "public" / "latest.json"
    _export(manifest_path, forecast_dir, output)
    first = output.read_bytes()

    assert _export(manifest_path, forecast_dir, output)["status"] == "OK"
    assert output.read_bytes() == first


def test_a_replay_with_a_new_run_id_publishes_the_same_card(tmp_path):
    """THE regression. The replay's run_id appears in no immutable row, and
    that must no longer matter."""
    manifest_path, forecast_dir, records = _card_estate(tmp_path)
    output = tmp_path / "public" / "latest.json"
    _export(manifest_path, forecast_dir, output)
    first = output.read_bytes()

    replay_path = _replay_manifest(tmp_path, records)
    result = _export(replay_path, forecast_dir, output)

    assert result["status"] == "OK"
    assert result["game_count"] == 16
    assert output.read_bytes() == first


def test_a_replay_succeeds_even_when_the_original_never_exported(tmp_path):
    """The failed production run published nothing, so the replay is the
    FIRST export of that card. It must still work."""
    _, forecast_dir, records = _card_estate(tmp_path)
    replay_path = _replay_manifest(tmp_path, records)
    result = _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")
    assert result["status"] == "OK"
    assert result["game_count"] == 16


def test_no_forecast_row_is_rewritten_to_carry_the_replays_run_id(tmp_path):
    """The rows are evidence. Making the replay work must never edit them."""
    manifest_path, forecast_dir, records = _card_estate(tmp_path)
    output = tmp_path / "public" / "latest.json"
    _export(manifest_path, forecast_dir, output)
    before = {path: path.read_bytes() for path in sorted(forecast_dir.glob("*.json"))}

    _export(_replay_manifest(tmp_path, records), forecast_dir, output)

    assert {path: path.read_bytes() for path in sorted(forecast_dir.glob("*.json"))} == before
    for payload in before.values():
        assert json.loads(payload)["run_id"] == DEFAULT_RUN_ID


def test_the_replay_leaves_the_immutable_archive_untouched(tmp_path):
    """Byte-identical output is what makes the archive idempotent instead of
    a refusing-to-overwrite failure."""
    manifest_path, forecast_dir, records = _card_estate(tmp_path)
    output = tmp_path / "public" / "latest.json"
    _export(manifest_path, forecast_dir, output)
    archived = _archive_bytes(output)

    _export(_replay_manifest(tmp_path, records), forecast_dir, output)

    assert _archive_bytes(output) == archived


def test_generated_at_comes_from_the_rows_not_the_replay_manifest(tmp_path):
    """The card was generated when its forecasts were, not when someone
    re-exported them."""
    _, forecast_dir, records = _card_estate(tmp_path)
    output = tmp_path / "public" / "latest.json"
    _export(_replay_manifest(tmp_path, records), forecast_dir, output)

    card = json.loads(output.read_text())
    assert card["generated_at_utc"] == ORIGINAL_RUN_CREATED
    assert card["generated_at_utc"] != REPLAY_RUN_CREATED


def test_a_replay_does_not_leave_a_temporary_file_behind(tmp_path):
    manifest_path, forecast_dir, records = _card_estate(tmp_path)
    output = tmp_path / "public" / "latest.json"
    _export(manifest_path, forecast_dir, output)
    _export(_replay_manifest(tmp_path, records), forecast_dir, output)
    assert not list(output.parent.glob("*.tmp"))
    assert not list(output.parent.glob(".*.tmp"))


# ===========================================================================
# PART 2 -- everything that must STILL fail closed.
#
# The fix removed a membership check, so each of these proves the replacement
# is not weaker: membership is now proved by the declared game count and the
# cryptographic batch hash rather than by a run_id label a file can claim.
# ===========================================================================
def test_a_partial_ledger_remains_blocked(tmp_path):
    """A missing row is the original symptom's legitimate cause, and it must
    still be refused."""
    _, forecast_dir, records = _card_estate(tmp_path)
    next(iter(sorted(forecast_dir.glob("*.json")))).unlink()

    replay_path = _replay_manifest(tmp_path, records)
    with pytest.raises(exp.WizardExportError, match="game_count"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_an_extra_row_in_the_same_card_remains_blocked(tmp_path):
    _, forecast_dir, records = _card_estate(tmp_path)
    _write_record(forecast_dir, _forecast_record(game_id="G99", run_created_at_utc=ORIGINAL_RUN_CREATED))

    replay_path = _replay_manifest(tmp_path, records)
    with pytest.raises(exp.WizardExportError, match="game_count"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_a_full_card_at_the_wrong_cutoff_remains_blocked(tmp_path):
    """A manifest naming another week's cutoff selects that week's rows, not
    this one's -- so it can never publish this card by accident."""
    _, forecast_dir, records = _card_estate(tmp_path)
    replay_path = _replay_manifest(tmp_path, records, target_cutoff_utc="2026-09-15T16:00:00+00:00")
    with pytest.raises(exp.WizardExportError, match="game_count"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


@pytest.mark.parametrize("missing", [None, "", "not-a-timestamp"])
def test_a_manifest_without_a_usable_cutoff_remains_blocked(tmp_path, missing):
    """The cutoff IS the forecast identity, so it is never inferred."""
    _, forecast_dir, records = _card_estate(tmp_path)
    replay_path = _replay_manifest(tmp_path, records, target_cutoff_utc=missing)
    with pytest.raises(exp.WizardExportError, match="target_cutoff_utc"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_a_wrong_forecast_batch_hash_remains_blocked(tmp_path):
    """The batch hash is what replaced run_id as the membership proof, so a
    manifest that declares the wrong one must be refused."""
    _, forecast_dir, records = _card_estate(tmp_path)
    replay_path = _replay_manifest(tmp_path, records, output_hashes={"forecast_batch_hash": "0" * 64})
    with pytest.raises(exp.WizardExportError, match="forecast_batch_hash"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_a_substituted_row_breaks_the_batch_hash_even_at_the_right_count(tmp_path):
    """Count alone is not membership. Swapping one game for a different one
    keeps the count correct and must still fail on the hash."""
    _, forecast_dir, records = _card_estate(tmp_path)
    replay_path = _replay_manifest(tmp_path, records)

    sorted(forecast_dir.glob("*.json"))[0].unlink()
    _write_record(forecast_dir, _forecast_record(game_id="G99", run_created_at_utc=ORIGINAL_RUN_CREATED))

    with pytest.raises(exp.WizardExportError, match="forecast_batch_hash"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_a_tampered_prediction_remains_blocked(tmp_path):
    _, forecast_dir, records = _card_estate(tmp_path)
    path = sorted(forecast_dir.glob("*.json"))[0]
    tampered = json.loads(path.read_text())
    tampered["prediction"]["prediction"]["predicted_margin"] = 99.0
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True))

    replay_path = _replay_manifest(tmp_path, records)
    with pytest.raises(exp.WizardExportError, match="prediction_hash"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_an_unidentifiable_row_remains_blocked(tmp_path):
    """A row that cannot state its cutoff cannot be proved to belong to
    another card, so it is never silently skipped."""
    _, forecast_dir, records = _card_estate(tmp_path)
    path = sorted(forecast_dir.glob("*.json"))[0]
    broken = json.loads(path.read_text())
    del broken["target_cutoff_utc"]
    path.write_text(json.dumps(broken, indent=2, sort_keys=True))

    replay_path = _replay_manifest(tmp_path, records)
    with pytest.raises(exp.WizardExportError, match="target_cutoff_utc"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


@pytest.mark.parametrize("field", ["horizon", "target_cutoff_utc"])
def test_a_row_contradicting_its_own_signed_payload_remains_blocked(tmp_path, field):
    """Selection reads the top-level identity; the batch hash covers the
    nested payload. If the two disagree, the row is ambiguous -- the gap
    between what was matched and what was signed."""
    _, forecast_dir, records = _card_estate(tmp_path)
    path = sorted(forecast_dir.glob("*.json"))[0]
    contradictory = json.loads(path.read_text())
    contradictory["prediction"][field] = "FRI" if field == "horizon" else "2026-09-11T16:00:00+00:00"
    path.write_text(json.dumps(contradictory, indent=2, sort_keys=True))

    replay_path = _replay_manifest(tmp_path, records)
    with pytest.raises(exp.WizardExportError, match="ambiguous forecast identity"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_rows_from_two_different_runs_remain_blocked(tmp_path):
    """A card half-written by one run and half by another has no single
    creation instant, so it is refused rather than given an arbitrary one."""
    _, forecast_dir, records = _card_estate(tmp_path, game_count=2)
    path = sorted(forecast_dir.glob("*.json"))[0]
    other_run = json.loads(path.read_text())
    other_run["run_created_at_utc"] = "2026-09-08T15:31:00Z"
    path.write_text(json.dumps(other_run, indent=2, sort_keys=True))

    replay_path = _replay_manifest(tmp_path, records)
    with pytest.raises(exp.WizardExportError, match="run_created_at_utc"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_a_non_success_replay_manifest_remains_blocked(tmp_path):
    _, forecast_dir, records = _card_estate(tmp_path)
    replay_path = _replay_manifest(tmp_path, records, status="MODEL_NOT_READY")
    with pytest.raises(exp.WizardExportError, match="SUCCESS"):
        _export(replay_path, forecast_dir, tmp_path / "public" / "latest.json")


def test_the_replay_publishes_nothing_when_it_fails_closed(tmp_path):
    """No partial card, and an already-published card is left alone."""
    manifest_path, forecast_dir, records = _card_estate(tmp_path)
    output = tmp_path / "public" / "latest.json"
    _export(manifest_path, forecast_dir, output)
    published = output.read_bytes()

    sorted(forecast_dir.glob("*.json"))[0].unlink()
    with pytest.raises(exp.WizardExportError):
        _export(_replay_manifest(tmp_path, records), forecast_dir, output)

    assert output.read_bytes() == published
    assert not list(output.parent.glob("*.tmp"))


def test_selection_never_scans_outside_the_explicit_forecast_dir(tmp_path):
    """No "find latest" behaviour was introduced: a complete card sitting in a
    sibling horizon directory is not reachable."""
    _, forecast_dir, records = _card_estate(tmp_path, game_count=2)
    sibling = tmp_path / "forecast-ledger" / "FRI"
    for record in records:
        _write_record(sibling, record)

    for path in sorted(forecast_dir.glob("*.json")):
        path.unlink()

    with pytest.raises(exp.WizardExportError, match="game_count"):
        _export(_replay_manifest(tmp_path, records), forecast_dir, tmp_path / "public" / "latest.json")


def test_the_public_schema_is_unchanged_by_the_replay_fix(tmp_path):
    _, forecast_dir, records = _card_estate(tmp_path, game_count=2)
    output = tmp_path / "public" / "latest.json"
    _export(_replay_manifest(tmp_path, records), forecast_dir, output)

    card = json.loads(output.read_text())
    assert tuple(card.keys()) == exp.TOP_LEVEL_KEY_ORDER
    assert card["schema_version"] == "wizard-nfl-pricing-v2"
    for game in card["games"]:
        assert tuple(game.keys()) == exp.GAME_KEY_ORDER


def test_run_id_is_no_longer_a_membership_criterion(tmp_path):
    """Stated directly: rows whose run_id matches nothing still publish, and
    the run_id is not consulted during selection."""
    _, forecast_dir, records = _card_estate(tmp_path, game_count=3)
    for path in sorted(forecast_dir.glob("*.json")):
        record = json.loads(path.read_text())
        record["run_id"] = f"unrelated-{record['game_id']}"
        path.write_text(json.dumps(record, indent=2, sort_keys=True))

    result = _export(
        _replay_manifest(tmp_path, records), forecast_dir, tmp_path / "public" / "latest.json"
    )
    assert result["status"] == "OK"
    assert result["game_count"] == 3
    assert DEFAULT_CUTOFF  # the cutoff, not the run, is what identified the card
