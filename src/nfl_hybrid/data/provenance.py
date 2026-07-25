from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    digest = sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True).values
    digest.update(hashed.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceManifest:
    dataset_name: str
    source_name: str
    source_locator: str
    retrieved_at_utc: str
    rows: int
    columns: int
    dataframe_sha256: str
    source_version: str | None = None
    source_license_note: str | None = None
    request_parameters: dict[str, Any] | None = None

    @classmethod
    def from_frame(
        cls,
        *,
        dataset_name: str,
        source_name: str,
        source_locator: str,
        frame: pd.DataFrame,
        source_version: str | None = None,
        source_license_note: str | None = None,
        request_parameters: dict[str, Any] | None = None,
    ) -> "SourceManifest":
        return cls(
            dataset_name=dataset_name,
            source_name=source_name,
            source_locator=source_locator,
            retrieved_at_utc=utc_now_iso(),
            rows=len(frame),
            columns=len(frame.columns),
            dataframe_sha256=dataframe_fingerprint(frame),
            source_version=source_version,
            source_license_note=source_license_note,
            request_parameters=request_parameters,
        )

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
