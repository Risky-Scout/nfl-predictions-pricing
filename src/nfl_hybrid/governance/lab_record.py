from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import hashlib
import json
import subprocess


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_lab_record(
    repo_root: str | Path,
    output_path: str | Path,
    *,
    input_files: Iterable[str | Path],
    model_spec_path: str | Path,
    validation_protocol_path: str | Path,
    notes: str = "",
) -> dict[str, object]:
    repo = Path(repo_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    inputs = []
    for value in input_files:
        path = Path(value).expanduser().resolve()
        inputs.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    payload: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "model_spec": {
            "path": str(Path(model_spec_path).expanduser().resolve()),
            "sha256": sha256_file(model_spec_path),
        },
        "validation_protocol": {
            "path": str(Path(validation_protocol_path).expanduser().resolve()),
            "sha256": sha256_file(validation_protocol_path),
        },
        "inputs": inputs,
        "notes": notes,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
