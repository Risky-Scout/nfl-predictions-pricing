from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd


MARKET_MAP = {
    "pregame_moneyline": "h2h",
    "pregame_ats": "spreads",
    "pregame_total": "totals",
}

HORIZON_MINUTES = {
    "opening_7d": 10080.0,
    "opening_72h": 4320.0,
    "opening_24h": 1440.0,
    "opening_6h": 360.0,
    "opening_60m": 60.0,
}

COMMON_FEATURES = [
    "market_t10_novig_probability",
    "market_t10_hold",
    "market_t10_eligible_books",
    "market_t10_probability_sd",
    "market_opening_horizon_minutes",
    "market_opening_novig_probability",
    "market_probability_movement",
    "market_t10_snapshot_lag_minutes",
]

LINE_FEATURES = [
    "market_t10_consensus_line",
    "market_t10_line_sd",
    "market_opening_line",
    "market_line_movement",
]


@dataclass(frozen=True)
class OddsAttachmentConfig:
    development_seasons: tuple[int, ...] = (2020, 2021, 2022, 2023)
    minimum_books: int = 3
    maximum_snapshot_lag_minutes: float = 5.0


def model_features(market: str) -> list[str]:
    if market not in MARKET_MAP:
        raise ValueError(f"Unsupported market: {market}")
    return COMMON_FEATURES if market == "pregame_moneyline" else COMMON_FEATURES + LINE_FEATURES


def _assert_unique(frame: pd.DataFrame, key: str, label: str) -> None:
    if frame[key].duplicated().any():
        raise ValueError(f"{label} contains duplicate {key} values.")


def build_market_odds_features(
    consensus: pd.DataFrame,
    movement: pd.DataFrame,
    market: str,
    config: OddsAttachmentConfig | None = None,
) -> pd.DataFrame:
    cfg = config or OddsAttachmentConfig()
    source_market = MARKET_MAP[market]

    closing = consensus[
        consensus["market"].eq(source_market)
        & consensus["horizon"].eq("closing_t10")
        & consensus["season"].isin(cfg.development_seasons)
    ].copy()

    move = movement[
        movement["market"].eq(source_market)
        & movement["season"].isin(cfg.development_seasons)
    ].copy()

    _assert_unique(closing, "game_id", f"{market} closing consensus")
    _assert_unique(move, "game_id", f"{market} movement")

    closing["requested_snapshot_utc"] = pd.to_datetime(
        closing["requested_snapshot_utc"], utc=True, errors="raise"
    )
    closing["returned_snapshot_utc"] = pd.to_datetime(
        closing["returned_snapshot_utc"], utc=True, errors="raise"
    )
    closing["market_t10_snapshot_lag_minutes"] = (
        closing["requested_snapshot_utc"]
        - closing["returned_snapshot_utc"]
    ).dt.total_seconds() / 60.0

    if (closing["market_t10_snapshot_lag_minutes"] < 0).any():
        raise ValueError("Future historical odds snapshot detected.")
    if (
        closing["market_t10_snapshot_lag_minutes"]
        > cfg.maximum_snapshot_lag_minutes
    ).any():
        raise ValueError("Historical odds snapshot exceeds lag contract.")
    if (closing["eligible_books"] < cfg.minimum_books).any():
        raise ValueError("Historical odds row has too few books.")

    closing = closing[
        [
            "game_id",
            "season",
            "week",
            "requested_snapshot_utc",
            "returned_snapshot_utc",
            "eligible_books",
            "consensus_line",
            "line_sd",
            "consensus_home_or_over_novig_probability",
            "probability_sd",
            "median_hold",
            "market_t10_snapshot_lag_minutes",
        ]
    ].rename(
        columns={
            "requested_snapshot_utc": "market_t10_requested_snapshot_utc",
            "returned_snapshot_utc": "market_t10_returned_snapshot_utc",
            "eligible_books": "market_t10_eligible_books",
            "consensus_line": "market_t10_consensus_line",
            "line_sd": "market_t10_line_sd",
            "consensus_home_or_over_novig_probability":
                "market_t10_novig_probability",
            "probability_sd": "market_t10_probability_sd",
            "median_hold": "market_t10_hold",
        }
    )

    move = move[
        [
            "game_id",
            "opening_available",
            "opening_horizon",
            "opening_line",
            "opening_home_or_over_novig_probability",
            "line_movement",
            "probability_movement",
        ]
    ].rename(
        columns={
            "opening_available": "market_opening_available",
            "opening_line": "market_opening_line",
            "opening_home_or_over_novig_probability":
                "market_opening_novig_probability",
            "line_movement": "market_line_movement",
            "probability_movement": "market_probability_movement",
        }
    )
    move["market_opening_horizon_minutes"] = (
        move["opening_horizon"].map(HORIZON_MINUTES).astype(float)
    )
    move["market_opening_missing"] = (
        ~move["market_opening_available"].astype(bool)
    ).astype(int)
    move = move.drop(columns=["opening_horizon"])

    output = closing.merge(move, on="game_id", how="left", validate="one_to_one")
    if output["market_opening_available"].isna().any():
        raise ValueError("Missing opening/movement row.")
    if output[model_features(market)].isna().any().any():
        raise ValueError("Missing approved odds model feature.")
    return output.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def attach_to_compact(
    compact: pd.DataFrame,
    odds: pd.DataFrame,
    market: str,
    config: OddsAttachmentConfig | None = None,
) -> pd.DataFrame:
    cfg = config or OddsAttachmentConfig()
    compact = compact[compact["season"].isin(cfg.development_seasons)].copy()
    _assert_unique(compact, "game_id", f"{market} compact")
    _assert_unique(odds, "game_id", f"{market} odds")

    compact_ids = set(compact["game_id"].astype(str))
    odds_ids = set(odds["game_id"].astype(str))
    if compact_ids != odds_ids:
        raise ValueError(
            f"{market} game coverage mismatch: "
            f"missing={len(compact_ids - odds_ids)}, "
            f"extra={len(odds_ids - compact_ids)}"
        )

    odds_for_merge = odds.copy()

    # Avoid duplicate metadata columns, but preserve fields that are absent
    # from the compact matrix, such as week.
    duplicate_metadata = [
        column
        for column in ("season", "week")
        if column in compact.columns and column in odds_for_merge.columns
    ]
    odds_for_merge = odds_for_merge.drop(
        columns=duplicate_metadata,
        errors="ignore",
    )

    attached = compact.merge(
        odds_for_merge,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    sort_columns = [
        column
        for column in ("season", "week", "game_id")
        if column in attached.columns
    ]

    return attached.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)


def build_all(
    compact_root: str | Path,
    odds_root: str | Path,
    output_root: str | Path,
) -> pd.DataFrame:
    compact_root = Path(compact_root).expanduser().resolve()
    odds_root = Path(odds_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    consensus = pd.read_parquet(odds_root / "consensus_by_horizon.parquet")
    movement = pd.read_parquet(
        odds_root / "opening_closing_market_features.parquet"
    )

    rows = []
    for market in MARKET_MAP:
        compact_path = compact_root / f"{market}_market_augmented.parquet"
        manifest_path = (
            compact_root / f"{market}_market_augmented.manifest.json"
        )
        compact = pd.read_parquet(compact_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        odds = build_market_odds_features(consensus, movement, market)
        enriched = attach_to_compact(compact, odds, market)

        added = model_features(market)
        features = list(dict.fromkeys(manifest["features"] + added))
        out_name = f"{market}_market_augmented_odds_enriched"

        enriched.to_parquet(
            output_root / f"{out_name}.parquet", index=False
        )
        manifest.update(
            {
                "variant": "market_augmented_odds_enriched",
                "development_seasons": [2020, 2021, 2022, 2023],
                "rows": len(enriched),
                "features": features,
                "feature_count": len(features),
                "new_odds_features": added,
                "audit_only_columns": [
                    "market_opening_available",
                    "market_opening_missing",
                    "market_t10_requested_snapshot_utc",
                    "market_t10_returned_snapshot_utc",
                ],
            }
        )
        (
            output_root / f"{out_name}.manifest.json"
        ).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        rows.append(
            {
                "market": market,
                "rows": len(enriched),
                "games": enriched["game_id"].nunique(),
                "new_odds_features": len(added),
                "total_features": len(features),
                "minimum_books": int(
                    enriched["market_t10_eligible_books"].min()
                ),
                "maximum_snapshot_lag_minutes": float(
                    enriched["market_t10_snapshot_lag_minutes"].max()
                ),
                "opening_missing_rows": int(
                    enriched["market_opening_missing"].sum()
                ),
                "duplicate_game_ids": int(
                    enriched["game_id"].duplicated().sum()
                ),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(
        output_root / "compact_odds_attachment_audit.csv", index=False
    )
    return audit
