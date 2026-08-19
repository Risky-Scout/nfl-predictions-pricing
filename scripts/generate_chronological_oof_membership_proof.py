"""Fix 3: write and verify a chronological-OOF membership-proof artifact for
one representative target game.

Prefers the real 2020-2025 backfill (via ``NFL_MODEL_DATA_ROOT``), scoped to
one representative season for tractable runtime -- real data end-to-end, just
bounded in volume, mirroring
``scripts/generate_pregame_state_lineage_example.py`` (Fix 2). Falls back to a
small synthetic sequence ONLY when the private data estate is unavailable,
and labels the artifact accordingly either way.

Before writing, this script explicitly re-verifies, independent of
:func:`generate_oof_predictions`'s own bookkeeping, the exact real-data proof
the Fix 3 specification asks for: ``max_training_result_available_at_utc <
target_cutoff_utc`` and ``target_game_id not in training_game_ids``.
Eligibility is decided by each training game's ``result_available_at_utc``
(:func:`nfl_hybrid.evaluation.chronological_oof.compute_result_available_at_utc`),
never by kickoff order alone -- see the correction note in
``chronological_oof.py``'s own module docstring.

This writes a small, git-committed JSON proof only -- never the full OOF
ledger. The full ledger is a GENERATED pipeline artifact, persisted
separately via
:func:`nfl_hybrid.evaluation.chronological_oof.persist_oof_ledger` under
``NFL_MODEL_ARTIFACT_ROOT`` (never ``NFL_MODEL_DATA_ROOT``, which this script
only reads the raw historical estate from -- the two roots are deliberately
distinct; see ``nfl_hybrid.data.external_data``'s module docstring).

``result_available_at_utc`` / ``max_training_result_available_at_utc`` below
are DERIVED, conservative eligibility timestamps
(``result_availability_basis = "HISTORICAL_CONSERVATIVE_BATCH"``) -- kickoff
plus a documented floor, snapped to the next Tue/Fri production-batch
boundary -- never an observed final-whistle time.

Run: PYTHONPATH=src python scripts/generate_chronological_oof_membership_proof.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.external_data import ExternalDataUnavailableError, resolve
from nfl_hybrid.evaluation.chronological_oof import (
    ChronologicalOOFConfig,
    build_oof_feature_matrix,
    compute_feature_state_hash,
    generate_oof_predictions,
)
from nfl_hybrid.features.pbp_advanced import aggregate_qb_game_efficiency

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "outputs" / "chronological_oof_membership_proof.json"
REPRESENTATIVE_SEASON = 2023
MIN_TRAINING_GAMES = 60


def _synthetic_games(n: int = 20) -> pd.DataFrame:
    teams = ["KC", "BUF", "SF", "DAL"]
    return pd.DataFrame(
        {
            "game_id": [f"S{i:03d}" for i in range(n)],
            "season": 2024,
            "week": list(range(1, n + 1)),
            "home_team_id": [teams[i % 4] for i in range(n)],
            "away_team_id": [teams[(i + 1) % 4] for i in range(n)],
            "scheduled_kickoff_utc": pd.date_range("2024-09-01", periods=n, freq="7D", tz="UTC"),
            "home_score": [20 + (i % 10) for i in range(n)],
            "away_score": [17 + ((i + 3) % 10) for i in range(n)],
            "home_spread_reference": -3.0,
            "total_line_reference": 44.0,
            "neutral_site": False,
            "playoff": False,
        }
    )


def _synthetic_pbp(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game in games.itertuples(index=False):
        for team, opp in [(game.home_team_id, game.away_team_id), (game.away_team_id, game.home_team_id)]:
            for _ in range(15):
                rows.append(
                    dict(
                        game_id=game.game_id, posteam=team, defteam=opp, season=2024, week=1,
                        qb_dropback=1, pass_attempt=1, rush_attempt=0, epa=0.1, success=1, down=1,
                        game_seconds_remaining=1800, score_differential=0, passing_yards=8, complete_pass=1,
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


def main() -> int:
    real = _load_real_data()
    if real is not None:
        games, pbp = real
        qb_game = aggregate_qb_game_efficiency(pbp)
        matrix, feature_columns, bundle = build_oof_feature_matrix(games, pbp, qb_game=qb_game)
        source = f"real_data:backfill-2020-2025 (season={REPRESENTATIVE_SEASON})"
        min_training_games = MIN_TRAINING_GAMES
    else:
        games = _synthetic_games()
        pbp = _synthetic_pbp(games)
        matrix, feature_columns, bundle = build_oof_feature_matrix(games, pbp)
        source = "synthetic:20-game round-robin sequence (NFL_MODEL_DATA_ROOT unavailable)"
        min_training_games = 5

    feature_state_hash = compute_feature_state_hash(feature_columns, bundle)

    target_position = len(matrix) // 2
    preds = generate_oof_predictions(
        matrix,
        feature_columns=feature_columns,
        config=ChronologicalOOFConfig(min_training_games=min_training_games),
        feature_state_hash=feature_state_hash,
        target_row_positions=[target_position],
    )
    row = preds.iloc[0]
    if row["status"] != "OOF":
        raise RuntimeError("representative target lacked sufficient warmup history; pick a later position")

    target_game_id = str(row["game_id"])
    target_cutoff = pd.Timestamp(row["target_cutoff_utc"])
    max_training_result_available = pd.Timestamp(row["max_training_result_available_at_utc"])
    training_ids = list(row["training_game_ids"])

    # Independent re-check, not reusing generate_oof_predictions's own bookkeeping.
    # Keyed on result_available_at_utc, not scheduled_kickoff_utc: a game's
    # kickoff order is not proof its RESULT was available before the cutoff.
    available_by_id = dict(
        zip(matrix["game_id"], pd.to_datetime(matrix["result_available_at_utc"], utc=True))
    )
    assert target_game_id not in training_ids, "target_game_id must not be in training_game_ids"
    if training_ids:
        real_max_training_available = max(available_by_id[gid] for gid in training_ids)
        assert real_max_training_available == max_training_result_available
        assert real_max_training_available < target_cutoff, (
            "max_training_result_available_at_utc must be < target_cutoff_utc"
        )

    payload = {
        "artifact": "chronological_oof_membership_proof",
        "data_source": source,
        "note": (
            "Demonstrates, for one representative target game, that its "
            "chronological OOF training set satisfies "
            "max_training_result_available_at_utc < target_cutoff_utc and "
            "target_game_id not in training_game_ids, via "
            "nfl_hybrid.evaluation.chronological_oof.generate_oof_predictions. "
            "Eligibility is decided by each candidate's result_available_at_utc "
            "(kickoff plus a conservative floor, snapped to the next Tue/Fri "
            "production batch boundary), never by kickoff order alone. "
            "Independently re-verified above before being written, not just "
            "trusted from the ledger's own bookkeeping."
        ),
        "target_game_id": target_game_id,
        "target_cutoff_utc": str(target_cutoff),
        "training_as_of_utc": str(pd.Timestamp(row["training_as_of_utc"])),
        "max_training_result_available_at_utc": str(max_training_result_available),
        "result_availability_basis": row["result_availability_basis"],
        "training_game_count": int(row["training_game_count"]),
        "training_game_ids_truncated": bool(row["training_game_ids_truncated"]),
        "training_membership_hash": row["training_membership_hash"],
        "model_config_hash": row["model_config_hash"],
        "feature_state_hash": feature_state_hash,
        "predicted_margin": float(row["predicted_margin"]),
        "predicted_total": float(row["predicted_total"]),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} (source: {source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
