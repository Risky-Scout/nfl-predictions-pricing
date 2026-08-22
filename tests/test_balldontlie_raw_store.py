"""Raw snapshot storage: NFL_LIVE_DATA_ROOT resolution, idempotence,
secret redaction (Sections 21-22)."""
from __future__ import annotations

import json

import pytest

from nfl_hybrid.providers.balldontlie.raw_store import (
    LiveDataRootUnavailableError,
    live_data_root,
    write_raw_snapshot,
)


def test_live_data_root_requires_env_or_override(monkeypatch):
    monkeypatch.delenv("NFL_LIVE_DATA_ROOT", raising=False)
    with pytest.raises(LiveDataRootUnavailableError):
        live_data_root()


def test_live_data_root_uses_override(tmp_path):
    root = live_data_root(root_override=tmp_path)
    assert root == tmp_path


def test_write_snapshot_idempotent_same_payload(tmp_path):
    payload = [{"id": 1, "status_state": "final"}]
    first = write_raw_snapshot(
        provider="balldontlie",
        endpoint="/games",
        query_parameters={"seasons": [2025]},
        raw_data=payload,
        requested_at_utc="2026-08-21T00:00:00Z",
        root_override=tmp_path,
    )
    second = write_raw_snapshot(
        provider="balldontlie",
        endpoint="/games",
        query_parameters={"seasons": [2025]},
        raw_data=payload,
        requested_at_utc="2026-08-21T01:00:00Z",
        root_override=tmp_path,
    )
    assert first.snapshot_path == second.snapshot_path
    assert first.already_existed is False
    assert second.already_existed is True
    assert first.manifest.response_sha256 == second.manifest.response_sha256

    # No duplicate raw snapshot files were created.
    snapshots = list((tmp_path / "balldontlie" / "games").glob("*.json"))
    assert len([p for p in snapshots if not p.name.endswith(".manifest.json")]) == 1


def test_write_snapshot_changed_payload_creates_new_immutable_snapshot(tmp_path):
    first = write_raw_snapshot(
        provider="balldontlie",
        endpoint="/games",
        query_parameters={},
        raw_data=[{"id": 1, "status_state": "in_progress"}],
        requested_at_utc="2026-08-21T00:00:00Z",
        root_override=tmp_path,
    )
    second = write_raw_snapshot(
        provider="balldontlie",
        endpoint="/games",
        query_parameters={},
        raw_data=[{"id": 1, "status_state": "final"}],  # genuinely changed
        requested_at_utc="2026-08-21T03:00:00Z",
        root_override=tmp_path,
    )
    assert first.snapshot_path != second.snapshot_path
    # Both immutable snapshots still exist -- the first was never overwritten.
    assert json.loads(first.snapshot_path.read_text())[0]["status_state"] == "in_progress"
    assert json.loads(second.snapshot_path.read_text())[0]["status_state"] == "final"


def test_query_parameters_with_key_like_names_are_redacted(tmp_path):
    result = write_raw_snapshot(
        provider="balldontlie",
        endpoint="/games",
        query_parameters={"seasons": [2025], "api_key": "should-never-appear", "authToken": "also-secret"},
        raw_data=[{"id": 1}],
        requested_at_utc="2026-08-21T00:00:00Z",
        root_override=tmp_path,
    )
    stored = json.loads(result.manifest_path.read_text())
    assert "api_key" not in stored["query_parameters"]
    assert "authToken" not in stored["query_parameters"]
    assert stored["query_parameters"] == {"seasons": [2025]}
