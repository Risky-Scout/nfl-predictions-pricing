"""``predict_week`` CLI: season/week -> betting-card CSV.

Sources the target week's games either from a canonical backfill directory
(``--backfill-dir``, reads ``canonical/games.parquet``) or an explicit games
file (``--games``), attaches the frozen production readiness status per market,
runs the joint reference distribution + penalized-Kelly staking, and writes a
betting card. With the frozen 2025 production spec (market baseline for every
market) the card recommends no bets and flags ``no-bet: edge not established``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_hybrid.data.io import read_tabular
from nfl_hybrid.pricing.betting_card import (
    CardConfig,
    build_betting_card,
    readiness_from_production_spec,
)
from nfl_hybrid.staking.kelly import StakingPolicy

_WEEK_ORDER = [str(w) for w in range(1, 19)] + [
    "Wildcard",
    "Division",
    "Conference",
    "Superbowl",
]


def _week_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    try:
        return (_WEEK_ORDER.index(text), text)
    except ValueError:
        return (len(_WEEK_ORDER), text)


def _load_games(args: argparse.Namespace) -> pd.DataFrame:
    if args.games:
        frame = read_tabular(args.games)
    elif args.backfill_dir:
        path = Path(args.backfill_dir) / "canonical" / "games.parquet"
        frame = pd.read_parquet(path)
    else:  # pragma: no cover - argparse enforces one of the two
        raise ValueError("Provide either --games or --backfill-dir.")

    frame = frame.copy()
    rename = {
        "home_team_id": "home_team",
        "away_team_id": "away_team",
        "home_spread_reference": "home_spread",
        "total_line_reference": "total_line",
    }
    for src, dst in rename.items():
        if src in frame.columns and dst not in frame.columns:
            frame[dst] = frame[src]

    if "season" in frame.columns:
        frame = frame[pd.to_numeric(frame["season"], errors="coerce") == args.season]
    if "week" in frame.columns:
        frame = frame[frame["week"].astype(str) == str(args.week)]
    return frame.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a weekly NFL betting card (season/week -> CSV)."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", required=True, help='e.g. "1"-"18" or "Wildcard".')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--games", help="CSV/parquet of games for the week.")
    source.add_argument("--backfill-dir", help="Backfill dir with canonical/games.parquet.")
    parser.add_argument(
        "--production-spec",
        default="config/production_model_spec.json",
        help="Frozen production spec for per-market readiness status.",
    )
    parser.add_argument(
        "--pricing-artifact",
        default="config/pricing_calibration_2026.json",
        help="Frozen pricing-calibration artifact (empirical margin surface + sigmas).",
    )
    parser.add_argument("--output", required=True, help="Output betting-card CSV path.")
    parser.add_argument("--as-of-utc", default=None, help="Deterministic replay timestamp (UTC).")
    parser.add_argument("--preliminary", action="store_true", help="Label the card PRELIMINARY.")
    parser.add_argument("--injury-data-utc", default=None, help="Timestamp of the injury/roster refresh.")
    parser.add_argument("--require-priced", action="store_true", help="Fail if no game has posted lines.")
    parser.add_argument("--canonical-games", default=None, help="Canonical games parquet (schedule + history) for live fundamental features.")
    parser.add_argument("--play-by-play", default=None, help="Play-by-play parquet for live fundamental features.")
    parser.add_argument("--require-full-card", action="store_true", help="Fail if a fundamental probability cannot be produced.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    games = _load_games(args)
    if len(games) == 0:
        raise SystemExit(
            f"No games found for season {args.season} week {args.week}."
        )
    as_of = args.as_of_utc or "2026-07-30T00:00:00Z"

    spec_path = Path(args.production_spec)
    if spec_path.exists():
        readiness = readiness_from_production_spec(spec_path)
    else:
        readiness = {"moneyline": "RETAIN_BASELINE", "ats": "RETAIN_BASELINE", "total": "RETAIN_BASELINE"}

    # load the frozen pricing-calibration artifact (empirical margin surface + fitted
    # sigmas); fall back to the built-in normal model if it is not present.
    artifact = None
    try:
        from nfl_hybrid.pricing.artifact import load_pricing_artifact

        artifact = load_pricing_artifact(args.pricing_artifact)
    except FileNotFoundError:
        print("WARNING: pricing artifact not found; using built-in normal surface.")

    # critical pre-pricing validation (raises WeeklyRunError -> job stops)
    from nfl_hybrid.pricing.weekly import (
        assemble_audit,
        validate_lines,
        validate_probability_identity,
    )

    if "status" in games.columns:
        validate_lines(games, require_priced=args.require_priced)
        priceable = games[games["status"] == "PRICED"].copy()
    else:
        priceable = games

    card = build_betting_card(
        priceable,
        readiness_by_market=readiness,
        staking_policy=StakingPolicy(),
        config=CardConfig(),
        pricing_artifact=artifact,
    )
    validate_probability_identity(card)
    card = assemble_audit(
        card, as_of_utc=as_of, artifact=artifact, lines=games,
        injury_data_utc=args.injury_data_utc, preliminary=args.preliminary,
    )

    # fundamental SHADOW columns (NULL when its EPA inputs are not in the live path;
    # never the market probability). Production stays the market baseline.
    from nfl_hybrid.pricing.fundamental_shadow import attach_shadow_columns, load_shadow, score_shadow

    try:
        _, _, shadow_meta = load_shadow()
    except Exception:
        shadow_meta = None
    fundamental = None
    fstatus = "NO_PREGAME_EPA_STATE_IN_LIVE_PATH"
    feature_snapshot_hash = None
    live_features = None
    finality_summary = {}
    if args.canonical_games and args.play_by_play and shadow_meta is not None:
        from nfl_hybrid.features.augmented_matrix import build_live_augmented_features

        canonical = pd.read_parquet(args.canonical_games)
        pbp = pd.read_parquet(args.play_by_play)
        # build target games from the schedule for (season, week), attaching current lines
        sched = canonical[
            (pd.to_numeric(canonical["season"], errors="coerce") == args.season)
            & (canonical["week"].astype(str) == str(args.week))
        ].copy()
        line_map = priceable.set_index(priceable["game_id"].astype(str))
        sched["game_id"] = sched["game_id"].astype(str)
        sched = sched[sched["game_id"].isin(line_map.index)]
        sched["home_spread"] = sched["game_id"].map(pd.to_numeric(line_map["home_spread"], errors="coerce"))
        sched["total_line"] = sched["game_id"].map(pd.to_numeric(line_map["total_line"], errors="coerce"))
        feats, fstat = build_live_augmented_features(canonical, sched, pbp, as_of_utc=as_of)
        fundamental = score_shadow(feats)
        fstatus = "AVAILABLE" if fundamental is not None else "SCORING_FAILED"
        feature_snapshot_hash = fstat.get("feature_snapshot_hash")
        live_features = feats
        finality_summary = {
            k: fstat.get(k) for k in (
                "eligible_final_game_count", "excluded_in_progress_game_count",
                "excluded_unknown_finality_game_count", "excluded_post_asof_game_count",
                "excluded_post_asof_play_count", "maximum_eligible_completion_timestamp",
                "finality_source_counts", "finality_exclusion_reason_counts",
            )
        }
    if args.require_full_card and fundamental is None:
        from nfl_hybrid.pricing.weekly import WeeklyRunError
        raise WeeklyRunError("full-card mode requires a fundamental probability, which is unavailable")
    card = attach_shadow_columns(
        card, fundamental, status=fstatus,
        artifact_version=(shadow_meta or {}).get("artifact"),
        features=live_features, feature_snapshot_hash=feature_snapshot_hash,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    card.to_csv(out, index=False)

    # immutable run manifest for replay/audit
    import hashlib as _hashlib
    import json as _json
    import subprocess as _sp

    try:
        runtime_head = _sp.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        runtime_head = "unknown"
    manifest = {
        "season": args.season, "week": str(args.week), "as_of_utc": as_of,
        "horizon": "preliminary" if args.preliminary else "live",
        # the CURRENT repo HEAD, distinct from the artifacts' training commits
        "runtime_code_commit": runtime_head,
        "calibration_artifact_training_commit": getattr(artifact, "code_commit", "unknown"),
        "calibration_artifact_hash": getattr(artifact, "artifact_sha256", "none"),
        "fundamental_artifact": (shadow_meta or {}).get("artifact", "NO_FROZEN_FUNDAMENTAL_ARTIFACT"),
        "fundamental_artifact_training_commit": (shadow_meta or {}).get("code_commit", "none"),
        "fundamental_artifact_hash": (shadow_meta or {}).get("model_sha256", "none"),
        "fundamental_input_status": fstatus,
        "feature_snapshot_hash": feature_snapshot_hash or "NONE",
        "finality_summary": finality_summary,
        "finality_decision_hash": _hashlib.sha256(
            _json.dumps(finality_summary, sort_keys=True, default=str).encode()
        ).hexdigest() if finality_summary else "NONE",
        "fundamental_non_null_rows": int(card["fundamental_probability"].notna().sum()) if "fundamental_probability" in card else 0,
        "injury_data_utc": args.injury_data_utc or "MISSING",
        "games": int(priceable["game_id"].nunique()),
        "card_rows": int(len(card)),
        "card_status_counts": card["card_status"].value_counts().to_dict() if "card_status" in card else {},
        "market_status_counts": (games["market_status"].value_counts().to_dict() if "market_status" in games else {}),
        "fundamental_input_status_counts": card["fundamental_input_status"].value_counts().to_dict() if "fundamental_input_status" in card else {},
        "production_source_counts": card["production_source"].value_counts().to_dict() if "production_source" in card else {},
        "production_source": "MARKET_BASELINE",
        "card_sha256": _hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (out.parent / f"run_manifest_wk{args.week}.json").write_text(_json.dumps(manifest, indent=2, sort_keys=True, default=str))

    bets = int(card["should_bet"].sum())
    n_no_price = int((card["card_status"] == "NO_PRICE").sum()) if "card_status" in card else 0
    print(f"card_rows={len(card)} games={priceable['game_id'].nunique()} recommended_bets={bets} no_price_rows={n_no_price}")
    print(f"status={'PRELIMINARY' if args.preliminary else 'VALID_LIVE_CARD'} artifact={getattr(artifact,'artifact_sha256','none')[:12]}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
