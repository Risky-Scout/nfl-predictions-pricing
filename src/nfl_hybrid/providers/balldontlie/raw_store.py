"""Raw BDL snapshot / provenance storage (Sections 21-22).

Deliberately NOT ``NFL_MODEL_DATA_ROOT`` (:mod:`nfl_hybrid.data.external_data`)
-- that env var is reserved for the historical, licensed nflverse/odds
estate. Live 2026 provider snapshots are generated pipeline output, not
part of that estate, so they get their own root,
``NFL_LIVE_DATA_ROOT``, mirroring the same
external-vs-generated separation Fix 1.5 already established for
``NFL_MODEL_ARTIFACT_ROOT``. No hardcoded machine path anywhere in this
module.

A snapshot is content-addressed: its filename is the sha256 of its own
canonical JSON body, so writing the exact same raw payload twice is a
no-op write to the same path (idempotence, Section 22) rather than an
ever-growing pile of identical files. If the provider's data genuinely
changed between polls, the new payload hashes differently and is stored as
a new, additional immutable snapshot -- prior snapshots are never
overwritten or mutated in place.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from nfl_hybrid.data.provenance import utc_now_iso

ENV_VAR = "NFL_LIVE_DATA_ROOT"

_SECRET_KEY_MARKERS = ("key", "token", "authorization", "secret", "password")


class LiveDataRootUnavailableError(RuntimeError):
    pass


def live_data_root(root_override: str | os.PathLike | None = None) -> Path:
    raw = root_override if root_override is not None else os.environ.get(ENV_VAR)
    if not raw:
        raise LiveDataRootUnavailableError(
            f"{ENV_VAR} is not set and no root_override was given. Point it at "
            f"where live provider snapshots should be written, e.g. "
            f"{ENV_VAR}=/path/to/your-live-data-cache"
        )
    root = Path(raw).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _redact_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if not any(marker in key.lower() for marker in _SECRET_KEY_MARKERS)
    }


def _content_hash(body: Any) -> str:
    encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SnapshotManifest:
    provider: str
    endpoint: str
    query_parameters: dict[str, Any]
    requested_at_utc: str
    received_at_utc: str
    response_sha256: str
    row_count: int
    pages_fetched: int
    schema_version: str


@dataclass(frozen=True)
class WrittenSnapshot:
    snapshot_path: Path
    manifest_path: Path
    manifest: SnapshotManifest
    already_existed: bool


def write_raw_snapshot(
    *,
    provider: str,
    endpoint: str,
    query_parameters: dict[str, Any],
    raw_data: list[dict[str, Any]],
    requested_at_utc: str,
    received_at_utc: str | None = None,
    pages_fetched: int = 1,
    schema_version: str = "1",
    root_override: str | os.PathLike | None = None,
) -> WrittenSnapshot:
    """Write one immutable raw snapshot plus its manifest. Idempotent: an
    identical ``raw_data`` payload for the same endpoint hashes to the same
    filename and is not rewritten (only its manifest's ``received_at_utc``
    reflects the call that happens to run this function, never mutating the
    stored raw body itself)."""
    root = live_data_root(root_override)
    endpoint_slug = endpoint.strip("/").replace("/", "_") or "root"
    endpoint_dir = root / provider / endpoint_slug
    endpoint_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _content_hash(raw_data)
    snapshot_path = endpoint_dir / f"{content_hash}.json"
    manifest_path = endpoint_dir / f"{content_hash}.manifest.json"

    already_existed = snapshot_path.exists()
    if not already_existed:
        snapshot_path.write_text(json.dumps(raw_data, sort_keys=True, default=str, indent=2), encoding="utf-8")

    manifest = SnapshotManifest(
        provider=provider,
        endpoint=endpoint,
        query_parameters=_redact_params(query_parameters),
        requested_at_utc=requested_at_utc,
        received_at_utc=received_at_utc or utc_now_iso(),
        response_sha256=content_hash,
        row_count=len(raw_data),
        pages_fetched=pages_fetched,
        schema_version=schema_version,
    )
    # The manifest itself is small and purely descriptive -- unlike the raw
    # body, re-writing it on every ingestion (to record the latest
    # confirming poll's received_at_utc) is safe and expected; it never
    # changes response_sha256, which is what idempotence is checked against.
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")

    return WrittenSnapshot(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        manifest=manifest,
        already_existed=already_existed,
    )
