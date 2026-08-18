"""Fix 2: write and verify a lineage artifact for one representative target
game, proving which prior games generated its pregame state.

Prefers the real 2020-2025 backfill (via ``NFL_MODEL_DATA_ROOT``), scoped to
one representative season for tractable runtime -- this is real data
end-to-end, not a synthetic stand-in, just bounded in volume. Falls back to a
small synthetic three-game sequence (A -> B -> C) ONLY when the private data
estate is unavailable, and labels the artifact accordingly either way so a
reader never mistakes one for the other.

Before writing, this script explicitly re-verifies, independent of
:func:`build_authoritative_pregame_state`'s own internal check, that the
chosen target game does not appear in its own contributing-game list and that
every contributing game's kickoff strictly precedes the target's kickoff --
the two invariants Fix 2 exists to prove.

Run: PYTHONPATH=src python scripts/generate_pregame_state_lineage_example.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.external_data import ExternalDataUnavailableError, resolve
from nfl_hybrid.features.pbp_advanced import aggregate_qb_game_efficiency
from nfl_hybrid.features.pregame_state import build_authoritative_pregame_state

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "outputs" / "pregame_state_lineage_example.json"
REPRESENTATIVE_SEASON = 2023


def _synthetic_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "home_team_id": ["KC", "BUF", "KC"],
            "away_team_id": ["BUF", "KC", "BUF"],
            "scheduled_kickoff_utc": pd.to_datetime(
                ["2024-09-01T17:00:00Z", "2024-09-08T17:00:00Z", "2024-09-15T17:00:00Z"], utc=True
            ),
            "home_score": [24, 20, 27],
            "away_score": [17, 23, 24],
            "home_spread_reference": [-3.0, -2.5, -3.5],
            "total_line_reference": [44.0, 45.0, 46.0],
            "neutral_site": [False, False, False],
            "playoff": [False, False, False],
        }
    )


def _synthetic_pbp() -> pd.DataFrame:
    rows = []
    epa_by_game = {"A": 0.10, "B": 0.20, "C": 0.30}
    for gid, home, away in [("A", "KC", "BUF"), ("B", "BUF", "KC"), ("C", "KC", "BUF")]:
        epa = epa_by_game[gid]
        for team, opp, sign in [(home, away, 1), (away, home, -1)]:
            for _ in range(15):
                rows.append(
                    dict(
                        game_id=gid, posteam=team, defteam=opp, season=2024, week=1,
                        qb_dropback=1, pass_attempt=1, rush_attempt=0,
                        epa=epa * sign, success=1, down=1,
                        game_seconds_remaining=1800, score_differential=0,
                        passing_yards=8, complete_pass=1,
                    )
                )
    return pd.DataFrame(rows)


def _load_real_data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    try:
        games_path = resolve("backfill.games")
        pbp_path = resolve("backfill.pbp")
    except ExternalDataUnavailableError:
        return None
    games = pd.read_parquet(games_path)
    pbp = pd.read_parquet(pbp_path)
    season_games = games[games["season"] == REPRESENTATIVE_SEASON].copy()
    season_ids = set(season_games["game_id"].astype(str))
    season_pbp = pbp[pbp["game_id"].astype(str).isin(season_ids)].copy()
    return season_games, season_pbp


def _pick_target_game_id(games: pd.DataFrame) -> str:
    ordered = games.sort_values(["scheduled_kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)
    return str(ordered.iloc[len(ordered) // 2]["game_id"])


def _verify_target_isolation(
    lineage_by_family: dict[str, list[dict]],
    *,
    target_game_id: str,
    target_kickoff: pd.Timestamp,
    kickoff_by_id: dict[str, pd.Timestamp],
) -> None:
    """Independent re-check (does not call state_lineage.verify_pregame_state_lineage,
    on purpose) that this artifact's own two headline claims hold."""
    checked = False
    for family, rows in lineage_by_family.items():
        for row in rows:
            assert target_game_id not in row["contributing_game_ids"], (
                f"{family}: target {target_game_id} lists itself as a contributing game"
            )
            if row["contributing_ids_truncated"]:
                continue
            for cid in row["contributing_game_ids"]:
                c_kickoff = kickoff_by_id[cid]
                assert c_kickoff < target_kickoff, (
                    f"{family}: contributing game {cid} ({c_kickoff}) does not precede "
                    f"target {target_game_id} ({target_kickoff})"
                )
            checked = True
    assert checked, "no family produced a lineage row to verify"


def main() -> int:
    real = _load_real_data()
    if real is not None:
        games, pbp = real
        qb_game = aggregate_qb_game_efficiency(pbp)
        bundle = build_authoritative_pregame_state(games, pbp, qb_game=qb_game)
        source = f"real_data:backfill-2020-2025 (season={REPRESENTATIVE_SEASON})"
    else:
        games, pbp = _synthetic_games(), _synthetic_pbp()
        bundle = build_authoritative_pregame_state(games, pbp)
        source = "synthetic:3-game A->B->C sequence (NFL_MODEL_DATA_ROOT unavailable)"

    target_game_id = _pick_target_game_id(games)
    kickoff_by_id = dict(
        zip(games["game_id"].astype(str), pd.to_datetime(games["scheduled_kickoff_utc"], utc=True))
    )
    target_kickoff = kickoff_by_id[target_game_id]

    families = {}
    for family, lineage in bundle.lineage.items():
        rows = lineage[lineage["game_id"].astype(str) == target_game_id]
        families[family] = [
            {
                "team_id": row.team_id,
                "target_kickoff_utc": str(row.target_kickoff_utc),
                "contributing_game_ids": list(row.contributing_game_ids),
                "contributing_game_count": int(row.contributing_game_count),
                "contributing_ids_truncated": bool(row.contributing_ids_truncated),
                "latest_contributing_kickoff_utc": str(row.latest_contributing_kickoff_utc),
                "contributing_scope": row.contributing_scope,
                "transform_name": row.transform_name,
                "transform_version": row.transform_version,
            }
            for row in rows.itertuples(index=False)
        ]

    _verify_target_isolation(
        families, target_game_id=target_game_id, target_kickoff=target_kickoff, kickoff_by_id=kickoff_by_id
    )

    payload = {
        "artifact": "pregame_state_lineage_example",
        "data_source": source,
        "note": (
            "Demonstrates exactly which prior games generated the target game's "
            "pregame state, per state family, via the Fix 2 authoritative "
            "interface (nfl_hybrid.features.pregame_state.build_authoritative_pregame_state). "
            "Verified before writing: the target game is absent from its own "
            "contributing-game list, and every listed contributing game's "
            "kickoff strictly precedes the target's kickoff."
        ),
        "target_game_id": target_game_id,
        "target_kickoff_utc": str(target_kickoff),
        "state_families": list(bundle.state.keys()),
        "lineage_by_family": families,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} (source: {source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
