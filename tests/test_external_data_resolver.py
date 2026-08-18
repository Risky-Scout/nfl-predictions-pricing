"""Unit tests for the central external-data resolver (Fix 1.5).

These use only synthetic temp directories -- never the real private data
estate -- so they run unconditionally in CI with zero skips.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_hybrid.data.external_data import (
    ENV_VAR,
    ExternalDataUnavailableError,
    describe,
    external_root,
    namespace_root,
    registry,
    resolve,
    resolve_for_write,
    validate,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _build_full_layout(root):
    """A minimal stand-in for a real NFL_MODEL_DATA_ROOT: every registered
    external-scope key resolves, using empty stub files/dirs (content is
    irrelevant to the resolver -- only path existence matters)."""
    for spec in registry().values():
        if spec.scope != "external":
            continue
        target = root / spec.relative_path
        if spec.kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
        else:
            _touch(target)


def test_registry_covers_required_backfill_assets():
    keys = set(registry())
    required = {
        "backfill.games", "backfill.game_crosswalk", "backfill.pbp",
        "backfill.team_stats", "backfill.player_stats", "backfill.weekly_rosters",
        "backfill.injuries", "backfill.snap_counts", "backfill.depth_charts",
        "backfill.spreadspoke_games",
    }
    assert required <= keys


def test_registry_covers_three_odds_history_roots():
    keys = set(registry())
    assert {
        "odds_history.2020_2023",
        "odds_history.2024_confirmation",
        "odds_history.2025_final_test",
    } <= keys
    # each preserves a distinct relative path (their distinct historical roles)
    paths = {registry()[k].relative_path for k in
              ("odds_history.2020_2023", "odds_history.2024_confirmation",
               "odds_history.2025_final_test")}
    assert len(paths) == 3


def test_registry_never_references_smoke_variant():
    for spec in registry().values():
        assert "smoke" not in spec.relative_path


def test_no_root_configured_fails_clearly(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(ExternalDataUnavailableError, match=ENV_VAR):
        external_root()
    with pytest.raises(ExternalDataUnavailableError):
        resolve("backfill.games")


def test_unknown_key_raises_key_error():
    with pytest.raises(KeyError):
        resolve("no_such_dataset")


def test_resolve_succeeds_against_a_full_synthetic_root(tmp_path):
    _build_full_layout(tmp_path)
    for key, spec in registry().items():
        if spec.scope != "external":
            continue
        path = resolve(key, root_override=tmp_path)
        assert path.exists()
        assert path == tmp_path / spec.relative_path


def test_resolve_via_env_var(tmp_path, monkeypatch):
    _build_full_layout(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    p = resolve("backfill.pbp")
    assert p == tmp_path / "backfill-2020-2025" / "raw" / "pbp.parquet"


def test_repo_scoped_key_ignores_env_var(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    # the purchased odds parquet is vendored in this checkout; it must resolve
    # even with NFL_MODEL_DATA_ROOT completely unset.
    path = resolve("purchased.odds_closing_dev_2022_2024")
    assert path.is_file()


def test_missing_asset_fails_clearly_and_does_not_substitute_smoke(tmp_path):
    """The core anti-substitution proof: a root that only has the *-smoke*
    layout (missing pbp/rosters, per the real NFL-Model-Data smoke run) must
    NOT silently satisfy resolve("backfill.pbp") -- it must fail clearly."""
    smoke_root = tmp_path / "backfill-2020-2025-smoke"
    for name in ("games", "game_crosswalk"):
        _touch(smoke_root / "canonical" / f"{name}.parquet")
    for name in ("team_stats", "player_stats", "weekly_rosters", "depth_charts",
                 "snap_counts", "nflverse_injuries", "spreadspoke_games", "nflverse_games"):
        _touch(smoke_root / "raw" / f"{name}.parquet")
    # deliberately no backfill-2020-2025/ (non-smoke) directory at all, and no
    # pbp.parquet anywhere -- this mirrors the real backfill-2020-2025-smoke/
    # which is missing exactly pbp and rosters.

    with pytest.raises(ExternalDataUnavailableError, match="backfill.pbp"):
        resolve("backfill.pbp", root_override=tmp_path)
    with pytest.raises(ExternalDataUnavailableError, match="backfill.games"):
        resolve("backfill.games", root_override=tmp_path)


def test_describe_never_raises_and_needs_no_env_var(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    for key in registry():
        s = describe(key)
        assert isinstance(s, str) and s


def test_validate_reports_per_key_status_without_raising(tmp_path):
    _build_full_layout(tmp_path)
    # deliberately omit one file to prove partial-failure reporting
    (tmp_path / "backfill-2020-2025" / "raw" / "pbp.parquet").unlink()
    results = validate(root_override=tmp_path)
    assert results["backfill.pbp"]["resolved"] is False
    assert results["backfill.games"]["resolved"] is True
    assert results["purchased.odds_closing_dev_2022_2024"]["resolved"] is True


def test_resolved_backfill_games_is_actually_readable_parquet(tmp_path):
    """End-to-end: resolve() returns a path pandas can actually read, not just
    an existing-but-arbitrary file."""
    games = pd.DataFrame({"game_id": ["a", "b"], "season": [2024, 2024]})
    target = tmp_path / "backfill-2020-2025" / "canonical" / "games.parquet"
    target.parent.mkdir(parents=True)
    games.to_parquet(target, index=False)
    loaded = pd.read_parquet(resolve("backfill.games", root_override=tmp_path))
    pd.testing.assert_frame_equal(loaded, games)


# ================================================================================
# Producer/consumer split-brain regression coverage.
# ================================================================================
def test_no_leftover_phantom_backfill_odds_closing_key():
    """odds_closing_dev_2022_2024 is a repo-committed purchased asset, not a
    separate external-estate artifact -- there must be exactly one registered
    key for it, not a permanently-unresolvable duplicate under 'backfill.*'."""
    keys = set(registry())
    assert "backfill.odds_closing_dev_2022_2024" not in keys
    assert "purchased.odds_closing_dev_2022_2024" in keys


def test_purchased_odds_closing_dev_has_manifest_key():
    keys = set(registry())
    assert "purchased.odds_closing_dev_2022_2024_manifest" in keys
    # the manifest key matches the ACTUAL committed sidecar naming convention
    assert registry()["purchased.odds_closing_dev_2022_2024_manifest"].relative_path == (
        "data/purchased/odds_closing_dev_2022_2024.manifest.json"
    )


def test_no_registered_key_is_permanently_unresolvable_in_the_repo_checkout():
    """Every repo-scoped key (independent of NFL_MODEL_DATA_ROOT) must resolve
    right now in this checkout -- a registered key that can never resolve
    would recreate the exact 'permanently missing required key' problem this
    registry exists to eliminate."""
    for key, spec in registry().items():
        if spec.scope == "repo":
            resolve(key)  # must not raise


def test_namespace_root_matches_backfill_registry_prefix(tmp_path):
    """The producer write-target helper and the per-file consumer registry
    entries must agree on the same root -- this is the core split-brain fix."""
    base = namespace_root("backfill", root_override=tmp_path)
    assert base == tmp_path / "backfill-2020-2025"
    for key in ("backfill.games", "backfill.pbp", "backfill.team_stats"):
        spec = registry()[key]
        assert str(spec.relative_path).startswith("backfill-2020-2025/")
        # constructing the path via namespace_root + the trailing part of the
        # registry's own relative_path reproduces the exact consumer path
        trailing = spec.relative_path.split("backfill-2020-2025/", 1)[1]
        assert base / trailing == tmp_path / spec.relative_path


def test_producer_write_then_consumer_read_agree(tmp_path):
    """End-to-end split-brain proof: write a stub pbp file the way a producer
    (namespace_root-based) would, then confirm resolve() (the consumer path)
    finds that exact file -- proving both sides target the same estate."""
    base = namespace_root("backfill", root_override=tmp_path)
    pbp_out = base / "raw" / "pbp.parquet"
    pbp_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": ["x"], "epa": [0.1]}).to_parquet(pbp_out, index=False)

    consumer_path = resolve("backfill.pbp", root_override=tmp_path)
    assert consumer_path == pbp_out
    assert consumer_path.samefile(pbp_out)


def test_resolve_for_write_does_not_require_existence(tmp_path):
    # external-scope: root must exist, but the target file need not.
    target = resolve_for_write("backfill.games", root_override=tmp_path)
    assert target == tmp_path / "backfill-2020-2025" / "canonical" / "games.parquet"
    assert not target.exists()

    # repo-scope: never depends on NFL_MODEL_DATA_ROOT at all, and resolves
    # regardless of whether the file currently exists.
    target2 = resolve_for_write("purchased.odds_closing_dev_2022_2024_manifest")
    assert target2.name == "odds_closing_dev_2022_2024.manifest.json"


def test_resolve_for_write_still_requires_external_root_to_exist():
    with pytest.raises(ExternalDataUnavailableError):
        resolve_for_write("backfill.games", root_override="/nonexistent/definitely/not/a/real/path")
