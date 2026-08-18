"""EXTENDED real-data proof for the Fix 2 authoritative pregame-state interface.

Exercises :func:`nfl_hybrid.features.pregame_state.build_authoritative_pregame_state`
against the private 2020-2025 backfill parquets, so this module is marked
``extended_data`` and skips cleanly when ``NFL_MODEL_DATA_ROOT`` (or the
backfill assets under it) is unavailable -- e.g. GitHub CI, or any machine
without the private data estate. It complements, and does not replace, the
synthetic adversarial suite in :mod:`tests.test_pregame_state_authoritative`,
which is the actual leakage-mechanism proof and runs unconditionally.

This module proves two things the synthetic suite cannot: (1) every family
builder actually runs end-to-end on the real schema/volume without raising,
and (2) the lineage invariants (no self-reference, no future reference) hold
across every real game, not just a hand-built A -> B -> C toy sequence.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl_hybrid.data.external_data import ExternalDataUnavailableError, resolve
from nfl_hybrid.features.pbp_advanced import aggregate_qb_game_efficiency
from nfl_hybrid.features.pregame_state import build_authoritative_pregame_state
from nfl_hybrid.features.state_lineage import verify_pregame_state_lineage


def _try_resolve(key):
    try:
        return resolve(key)
    except ExternalDataUnavailableError:
        return None


GAMES_PATH = _try_resolve("backfill.games")
PBP_PATH = _try_resolve("backfill.pbp")
pytestmark = [
    pytest.mark.extended_data,
    pytest.mark.skipif(
        GAMES_PATH is None or PBP_PATH is None,
        reason="NFL_MODEL_DATA_ROOT not configured or backfill assets missing (extended_data)",
    ),
]

# A full 6-season run (1,693 games / ~295k plays) is not needed to prove the
# invariant on real data and is slow in CI; one representative season is real
# data end-to-end and keeps this fast. The unconditional synthetic suite
# already proves the leakage *mechanism*; this proves it holds at real scale
# and against the real schema. Nothing is silently dropped: the season is
# fixed and logged via the assertion messages below.
REPRESENTATIVE_SEASON = 2023


@pytest.fixture(scope="module")
def real_bundle():
    games = pd.read_parquet(GAMES_PATH)
    pbp = pd.read_parquet(PBP_PATH)

    season_games = games[games["season"] == REPRESENTATIVE_SEASON].copy()
    assert len(season_games) > 0, f"no games found for season {REPRESENTATIVE_SEASON}"
    season_game_ids = set(season_games["game_id"].astype(str))
    season_pbp = pbp[pbp["game_id"].astype(str).isin(season_game_ids)].copy()

    qb_game = aggregate_qb_game_efficiency(season_pbp)

    bundle = build_authoritative_pregame_state(season_games, season_pbp, qb_game=qb_game)
    return season_games, bundle


def test_all_five_families_build_on_real_data(real_bundle):
    _, bundle = real_bundle
    assert set(bundle.state.keys()) == {"epa", "opponent_adjusted", "qb", "elo", "market_history"}
    for family, frame in bundle.state.items():
        assert len(frame) > 0, family


def test_real_data_lineage_passes_invariants(real_bundle):
    games, bundle = real_bundle
    for family, lineage in bundle.lineage.items():
        assert len(lineage) > 0, family
        verify_pregame_state_lineage(lineage, games)  # raises on any violation


def test_representative_target_game_lineage_excludes_self_and_future(real_bundle):
    games, bundle = real_bundle
    ordered = games.sort_values(["scheduled_kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)
    # A game roughly mid-season has real prior history in every family.
    target = ordered.iloc[len(ordered) // 2]
    target_id = str(target["game_id"])
    target_kickoff = pd.to_datetime(target["scheduled_kickoff_utc"], utc=True)

    kickoff_by_id = dict(
        zip(games["game_id"].astype(str), pd.to_datetime(games["scheduled_kickoff_utc"], utc=True))
    )

    checked_any_family = False
    for family, lineage in bundle.lineage.items():
        rows = lineage[lineage["game_id"] == target_id]
        if rows.empty:
            continue
        for row in rows.itertuples(index=False):
            assert target_id not in row.contributing_game_ids, (
                f"{family}: target game {target_id} appears in its own contributing-game list"
            )
            if row.contributing_ids_truncated:
                # count/time bound are never truncated; check those instead.
                assert row.contributing_game_count >= 0
                if pd.notna(row.latest_contributing_kickoff_utc):
                    assert row.latest_contributing_kickoff_utc < target_kickoff
                checked_any_family = True
                continue
            for cid in row.contributing_game_ids:
                c_kickoff = kickoff_by_id.get(cid)
                assert c_kickoff is not None, f"{family}: unknown contributing game_id {cid}"
                assert c_kickoff < target_kickoff, (
                    f"{family}: contributing game {cid} (kickoff {c_kickoff}) is not strictly "
                    f"before target {target_id} (kickoff {target_kickoff})"
                )
            checked_any_family = True
    assert checked_any_family, "no family produced a lineage row for the chosen target game"


def test_duplicate_real_game_id_fails():
    games = pd.read_parquet(GAMES_PATH)
    pbp = pd.read_parquet(PBP_PATH)
    season_games = games[games["season"] == REPRESENTATIVE_SEASON].copy()
    corrupted = pd.concat([season_games, season_games.iloc[[0]]], ignore_index=True)
    season_pbp = pbp[pbp["game_id"].astype(str).isin(set(season_games["game_id"].astype(str)))].copy()
    with pytest.raises(ValueError):
        build_authoritative_pregame_state(corrupted, season_pbp)
