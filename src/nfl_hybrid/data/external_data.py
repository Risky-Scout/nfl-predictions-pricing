"""Central resolver for the private, git-ignored external historical-data estate.

All access to the local NFL historical data estate (the nflverse backfill
parquets, the historical odds-history snapshot stores) must go through this
module instead of hard-coded relative-path literals such as
``"data/backfill_2020_2025/..."``. The physical location is supplied entirely
at runtime via the ``NFL_MODEL_DATA_ROOT`` environment variable (or an
explicit ``root_override``) -- nothing here defaults to a machine-specific
path. The same code and the same registry keys work unchanged on any machine
that points the env var at its own copy of the data estate, laid out with the
same directory names used in this registry (``backfill-2020-2025/...``,
``odds-api-history-2020-2023/...``, etc -- the exact names already used by
the historical backfill/odds-history pipelines, so no renaming is required to
adopt this resolver).

A second, small set of registry entries are repo-relative (``scope="repo"``):
data that is already vendored inside this repository's own ``data/``
directory (e.g. the purchased closing-odds parquet), independent of
``NFL_MODEL_DATA_ROOT``.

A third scope (``scope="generated"``) is for GENERATED pipeline artifacts --
derived outputs a producer computes from the historical estate (e.g. the Fix
3 chronological OOF ledgers), never raw/canonical historical data itself.
These resolve under a separate ``NFL_MODEL_ARTIFACT_ROOT`` environment
variable, deliberately distinct from ``NFL_MODEL_DATA_ROOT``: mixing derived,
regenerable pipeline output into the same tree as the raw historical estate
that produced it would make the two impossible to distinguish by location
alone, and would risk a careless bulk-copy of one silently dragging the other
along.

Two lookup functions are provided, deliberately kept separate:

- :func:`resolve` -- strict. Raises :class:`ExternalDataUnavailableError` if
  the key is unknown, the root is unavailable, or the path is missing. Use
  this at the point a dataset is actually read.
- :func:`describe` -- pure, non-raising. Returns the documented (unvalidated)
  location as a string, for use in labels/report metadata. Never touches
  disk and never requires ``NFL_MODEL_DATA_ROOT`` to be set, so it is safe to
  call at module import time (e.g. for module-level constants that some
  modules load via ``importlib`` during test collection, without private
  data configured).

No key in this registry ever points at ``backfill-2020-2025-smoke`` (the
lighter smoke-test run of the backfill pipeline) or any other fallback --
a missing asset always fails clearly via :func:`resolve` rather than silently
substituting a different, incomplete dataset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "NFL_MODEL_DATA_ROOT"
ARTIFACT_ENV_VAR = "NFL_MODEL_ARTIFACT_ROOT"

# src/nfl_hybrid/data/external_data.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]


class ExternalDataUnavailableError(RuntimeError):
    """A registered dataset could not be resolved to an existing path."""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    relative_path: str
    scope: str  # "external" (NFL_MODEL_DATA_ROOT), "generated" (NFL_MODEL_ARTIFACT_ROOT), or "repo" (this checkout)
    kind: str  # "file" or "dir"
    description: str


_REGISTRY: dict[str, DatasetSpec] = {}


def _register(key: str, relative_path: str, *, scope: str = "external",
              kind: str = "file", description: str = "") -> None:
    _REGISTRY[key] = DatasetSpec(key, relative_path, scope, kind, description)


# --- backfill-2020-2025: nflverse historical backfill, seasons 2020-2025 ------
_register("backfill.games", "backfill-2020-2025/canonical/games.parquet",
           description="Canonical game schedule/results table")
_register("backfill.game_crosswalk", "backfill-2020-2025/canonical/game_crosswalk.parquet",
           description="Cross-provider game id crosswalk")
_register("backfill.pbp", "backfill-2020-2025/raw/pbp.parquet",
           description="nflverse play-by-play, EPA-ready, 2020-2025")
_register("backfill.team_stats", "backfill-2020-2025/raw/team_stats.parquet",
           description="nflverse team-game box score stats")
_register("backfill.player_stats", "backfill-2020-2025/raw/player_stats.parquet",
           description="nflverse player-game box score stats")
_register("backfill.weekly_rosters", "backfill-2020-2025/raw/weekly_rosters.parquet",
           description="nflverse weekly roster snapshots")
_register("backfill.rosters", "backfill-2020-2025/raw/rosters.parquet",
           description="nflverse season rosters")
_register("backfill.injuries", "backfill-2020-2025/raw/nflverse_injuries.parquet",
           description="nflverse injury reports")
_register("backfill.snap_counts", "backfill-2020-2025/raw/snap_counts.parquet",
           description="nflverse snap counts")
_register("backfill.depth_charts", "backfill-2020-2025/raw/depth_charts.parquet",
           description="nflverse depth charts")
_register("backfill.spreadspoke_games", "backfill-2020-2025/raw/spreadspoke_games.parquet",
           description="SpreadSpoke historical games/lines pull")
_register("backfill.nflverse_games", "backfill-2020-2025/raw/nflverse_games.parquet",
           description="nflverse schedule/results pull")
_register("backfill.manifest_pbp", "backfill-2020-2025/manifests/pbp.json",
           description="Source manifest for backfill.pbp")
_register("backfill.manifest_team_stats", "backfill-2020-2025/manifests/team_stats.json",
           description="Source manifest for backfill.team_stats")
_register("backfill.manifest_snap_counts", "backfill-2020-2025/manifests/snap_counts.json",
           description="Source manifest for backfill.snap_counts")
_register("backfill.manifest_weekly_rosters", "backfill-2020-2025/manifests/weekly_rosters.json",
           description="Source manifest for backfill.weekly_rosters")
_register("backfill.manifest_depth_charts", "backfill-2020-2025/manifests/depth_charts.json",
           description="Source manifest for backfill.depth_charts")

# --- historical odds-history snapshot stores: three distinct acquisition runs -
_register("odds_history.2020_2023", "odds-api-history-2020-2023", kind="dir",
           description="The Odds API historical snapshots -- 2020-2023 backfill run")
_register("odds_history.2024_confirmation", "odds-api-history-2024-confirmation", kind="dir",
           description="The Odds API historical snapshots -- 2024 confirmation run")
_register("odds_history.2025_final_test", "odds-api-history-2025-final-test", kind="dir",
           description="The Odds API historical snapshots -- 2025 final-test run")

# --- already-vendored, repo-relative data (independent of NFL_MODEL_DATA_ROOT) -
# This IS the authoritative closing-odds dev-seasons dataset. It is a one-time,
# credit-limited purchase (see scripts/buy_closing_odds_dev.py) whose result is
# committed to the repo rather than kept in the external estate, so both its
# producer and its consumers resolve it at this single repo-relative location --
# there is no separate "backfill.odds_closing_dev_2022_2024" duplicate.
_register("purchased.odds_closing_dev_2022_2024",
           "data/purchased/odds_closing_dev_2022_2024.parquet", scope="repo",
           description="Committed purchased closing-odds dataset, dev seasons 2022-2024")
_register("purchased.odds_closing_dev_2022_2024_manifest",
           "data/purchased/odds_closing_dev_2022_2024.manifest.json", scope="repo",
           description="Source manifest for purchased.odds_closing_dev_2022_2024")
# Deliberately NOT registered: a purchase-run summary.json. Nothing reads it as
# a required input (it's an informational-only producer side-effect that was
# never committed), so registering it here would recreate exactly the
# "required key permanently missing" problem this registry exists to avoid.
# scripts/buy_closing_odds_dev.py writes it as a sibling of the manifest path
# instead of registering a phantom required key for it.

# --- chronological OOF ledger (Fix 3): one authoritative expanding-window
# out-of-fold prediction/residual/uncertainty record for the 2020-2025 estate.
# scope="generated": this is a DERIVED pipeline artifact, computed FROM the
# historical estate, not part of it -- it resolves under NFL_MODEL_ARTIFACT_ROOT,
# never NFL_MODEL_DATA_ROOT, and is never committed to the repo.
_register("oof_chronological.predictions",
           "chronological-oof-2020-2025/oof_predictions.parquet",
           scope="generated",
           description="Chronological OOF predictions, pre-outcome (Fix 3, phase 1)")
_register("oof_chronological.residual_ledger",
           "chronological-oof-2020-2025/oof_residual_ledger.parquet",
           scope="generated",
           description="Chronological OOF residual ledger with expanding uncertainty (Fix 3, phase 2)")
_register("oof_chronological.manifest",
           "chronological-oof-2020-2025/oof_manifest.json",
           scope="generated",
           description="Provenance/config manifest for the chronological OOF run (Fix 3)")

# --- namespace roots: directory-level write targets for producer scripts, so a
# producer and the per-file registry above always resolve under the same root.
# External-scope namespaces only; generated-scope keys above are resolved
# directly per-key by resolve_for_write, not through a namespace root.
_NAMESPACE_ROOTS: dict[str, str] = {
    "backfill": "backfill-2020-2025",
}


def registry() -> dict[str, DatasetSpec]:
    """The full dataset registry, keyed by logical dataset key."""
    return dict(_REGISTRY)


def namespace_root(namespace: str, *, root_override: str | os.PathLike | None = None) -> Path:
    """Resolve the writable root directory for a producer namespace (e.g. the
    ``backfill-2020-2025`` directory a backfill/pull script writes into).

    Requires ``NFL_MODEL_DATA_ROOT`` (or ``root_override``) to point at an
    existing directory, but does NOT require the namespace subdirectory itself
    to exist yet -- callers create it. This is the single source of truth for
    where persistent producer output goes, so a producer and the per-file
    registry entries above always agree on the same canonical data estate.
    """
    if namespace not in _NAMESPACE_ROOTS:
        raise KeyError(f"Unknown namespace: {namespace!r}. Known namespaces: {sorted(_NAMESPACE_ROOTS)}")
    return external_root(root_override) / _NAMESPACE_ROOTS[namespace]


def resolve_for_write(key: str, *, root_override: str | os.PathLike | None = None) -> Path:
    """Resolve a registered dataset key to its target path for WRITING.

    Unlike :func:`resolve`, this does not require the path to already exist --
    it's for producers about to create it. For ``scope="external"`` keys, the
    external root itself must still exist (``NFL_MODEL_DATA_ROOT`` configured);
    for ``scope="generated"`` keys, the artifact root must exist
    (``NFL_MODEL_ARTIFACT_ROOT`` configured); for ``scope="repo"`` keys,
    resolution never depends on either env var. Callers are responsible for
    creating parent directories.
    """
    if key not in _REGISTRY:
        raise KeyError(f"Unknown external dataset key: {key!r}. Known keys: {sorted(_REGISTRY)}")
    spec = _REGISTRY[key]
    if spec.scope == "repo":
        return REPO_ROOT / spec.relative_path
    if spec.scope == "generated":
        return artifact_root(root_override) / spec.relative_path
    return external_root(root_override) / spec.relative_path


def external_root(root_override: str | os.PathLike | None = None) -> Path:
    """Resolve the external RAW/CANONICAL historical-data root directory, or
    raise a clear, actionable error. Never defaults to any machine-specific
    path -- ``root_override`` or the ``NFL_MODEL_DATA_ROOT`` environment
    variable is the only source. Never used for generated pipeline artifacts
    -- see :func:`artifact_root`."""
    raw = root_override if root_override is not None else os.environ.get(ENV_VAR)
    if not raw:
        raise ExternalDataUnavailableError(
            f"{ENV_VAR} is not set and no root_override was given. Point it at "
            f"your local copy of the historical data estate, e.g. "
            f"{ENV_VAR}=/path/to/your-nfl-data-estate"
        )
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise ExternalDataUnavailableError(
            f"{ENV_VAR}={root} does not exist or is not a directory."
        )
    return root


def artifact_root(root_override: str | os.PathLike | None = None) -> Path:
    """Resolve the GENERATED-artifact root directory (derived pipeline
    outputs -- e.g. the Fix 3 chronological OOF ledgers -- never raw/canonical
    historical data), or raise a clear, actionable error. Never defaults to
    any machine-specific path -- ``root_override`` or the
    ``NFL_MODEL_ARTIFACT_ROOT`` environment variable is the only source.
    Deliberately a separate root from :func:`external_root`: generated
    pipeline output must never be written into, or resolved from, the same
    tree as the historical estate that produced it."""
    raw = root_override if root_override is not None else os.environ.get(ARTIFACT_ENV_VAR)
    if not raw:
        raise ExternalDataUnavailableError(
            f"{ARTIFACT_ENV_VAR} is not set and no root_override was given. Point it at "
            f"where generated pipeline artifacts should be written, e.g. "
            f"{ARTIFACT_ENV_VAR}=/path/to/your-artifact-root"
        )
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise ExternalDataUnavailableError(
            f"{ARTIFACT_ENV_VAR}={root} does not exist or is not a directory."
        )
    return root


def describe(key: str) -> str:
    """Pure, non-raising lookup: the documented (unvalidated) location of a
    registered dataset, for labels/report metadata. Never touches disk and
    never requires either root env var to be set -- safe to call at import
    time."""
    spec = _REGISTRY[key]
    if spec.scope == "repo":
        return str(REPO_ROOT / spec.relative_path)
    if spec.scope == "generated":
        return f"${{{ARTIFACT_ENV_VAR}}}/{spec.relative_path}"
    return f"${{{ENV_VAR}}}/{spec.relative_path}"


def resolve(key: str, *, root_override: str | os.PathLike | None = None) -> Path:
    """Resolve a registered dataset key to a real, existing path.

    Raises :class:`ExternalDataUnavailableError` (or ``KeyError`` for an
    unknown key) if the asset cannot be resolved. Never falls back to any
    other dataset -- in particular, never substitutes
    ``backfill-2020-2025-smoke`` for the full backfill. A missing required
    asset always fails clearly rather than silently degrading.
    """
    if key not in _REGISTRY:
        raise KeyError(f"Unknown external dataset key: {key!r}. Known keys: {sorted(_REGISTRY)}")
    spec = _REGISTRY[key]
    if spec.scope == "repo":
        path = REPO_ROOT / spec.relative_path
        env_hint = "the repo checkout"
    elif spec.scope == "generated":
        path = artifact_root(root_override) / spec.relative_path
        env_hint = ARTIFACT_ENV_VAR
    else:
        path = external_root(root_override) / spec.relative_path
        env_hint = ENV_VAR
    exists = path.is_dir() if spec.kind == "dir" else path.is_file()
    if not exists:
        raise ExternalDataUnavailableError(
            f"Registered dataset '{key}' ({spec.description}) not found at {path}. "
            f"Check {env_hint}."
        )
    return path


def validate(root_override: str | os.PathLike | None = None) -> dict[str, dict]:
    """Attempt to resolve every registered key; never raises.

    Returns ``{key: {"resolved": bool, "path": str, "error": str | None}}``.
    """
    out: dict[str, dict] = {}
    for key in sorted(_REGISTRY):
        try:
            path = resolve(key, root_override=root_override)
            out[key] = {"resolved": True, "path": str(path), "error": None}
        except ExternalDataUnavailableError as exc:
            out[key] = {"resolved": False, "path": describe(key), "error": str(exc)}
    return out


def main() -> int:
    """CLI: report whether every registered asset resolves correctly.

    Run: PYTHONPATH=src python scripts/validate_data_registry.py
    """
    results = validate()
    n_ok = sum(1 for v in results.values() if v["resolved"])
    print(f"{ENV_VAR}={os.environ.get(ENV_VAR)!r}")
    print(f"{ARTIFACT_ENV_VAR}={os.environ.get(ARTIFACT_ENV_VAR)!r}")
    for key, info in results.items():
        status = "OK" if info["resolved"] else "MISSING"
        print(f"  [{status:7s}] {key:40s} {info['path']}")
        if not info["resolved"]:
            print(f"             -> {info['error']}")
    print(f"\n{n_ok}/{len(results)} registered assets resolved.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
