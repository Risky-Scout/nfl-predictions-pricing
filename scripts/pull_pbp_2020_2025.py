"""Targeted nflverse play-by-play pull for 2020-2025 with a SHA-256 manifest.

Writes into the "backfill" namespace of NFL_MODEL_DATA_ROOT (the same root
consumers resolve reads from -- see nfl_hybrid.data.external_data), alongside
the other raw tables, so the provenance layout is unchanged. Equivalent to
re-running the backfill without ``--skip-large-tables`` but without re-pulling
the tables already cached.
"""

from __future__ import annotations

from nfl_hybrid.data.external_data import namespace_root
from nfl_hybrid.data.io import write_frame
from nfl_hybrid.data.providers.nflverse import NflverseAdapter
from nfl_hybrid.data.provenance import SourceManifest

SEASONS = tuple(range(2020, 2026))


def main() -> None:
    base = namespace_root("backfill")
    raw = base / "raw"
    manifests = base / "manifests"
    raw.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    adapter = NflverseAdapter()
    print(f"pulling PBP for seasons {SEASONS} ...", flush=True)
    result = adapter.load_pbp(SEASONS)
    print(f"pbp rows={len(result.data)} cols={result.data.shape[1]}", flush=True)

    path = write_frame(result.data, raw / "pbp")
    SourceManifest.from_frame(
        dataset_name="pbp",
        source_name=str(result.metadata.get("source_name", "nflverse")),
        source_locator=str(
            result.metadata.get("source_locator")
            or result.metadata.get("loader")
            or "nflverse.load_pbp"
        ),
        frame=result.data,
        request_parameters={
            key: value
            for key, value in result.metadata.items()
            if key not in {"source_name", "source_retrieved_at_utc"}
        },
    ).write_json(manifests / "pbp.json")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
