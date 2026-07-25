from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import json

import pandas as pd

from nfl_hybrid.data.io import write_frame, write_json
from nfl_hybrid.data.odds import (
    DEFAULT_HORIZON_MINUTES,
    add_market_consensus,
    build_prediction_horizons,
    devig_two_way_groups,
    match_odds_to_games,
)
from nfl_hybrid.data.provenance import SourceManifest, utc_now_iso
from nfl_hybrid.data.providers.nflverse import NflverseAdapter
from nfl_hybrid.data.providers.spreadspoke import SpreadspokeAdapter
from nfl_hybrid.data.providers.the_odds_api import OddsAPIConfig, TheOddsAPIAdapter
from nfl_hybrid.data.reconcile import (
    crosswalk_spreadspoke_to_nflverse,
    enrich_games_with_spreadspoke,
)


@dataclass(frozen=True)
class BackfillConfig:
    output_dir: Path
    seasons: tuple[int, ...] = tuple(range(2020, 2026))
    spreadspoke_locator: str | Path | None = None
    odds_horizon_minutes: tuple[int, ...] = DEFAULT_HORIZON_MINUTES
    include_large_nflverse_tables: bool = True
    include_nflverse_injuries_through: int = 2024
    max_odds_credits: int = 0
    confirm_odds_cost: bool = False


@dataclass
class HistoricalBackfill:
    config: BackfillConfig
    nflverse: NflverseAdapter = field(default_factory=NflverseAdapter)
    odds: TheOddsAPIAdapter | None = None

    def _dirs(self) -> dict[str, Path]:
        base = Path(self.config.output_dir)
        directories = {
            "raw": base / "raw",
            "canonical": base / "canonical",
            "manifests": base / "manifests",
            "plans": base / "plans",
        }
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def _save_result(
        self,
        *,
        dataset_name: str,
        result,
        directories: dict[str, Path],
        canonical: bool = False,
    ) -> Path:
        bucket = "canonical" if canonical else "raw"
        path = write_frame(result.data, directories[bucket] / dataset_name)
        manifest = SourceManifest.from_frame(
            dataset_name=dataset_name,
            source_name=str(result.metadata.get("source_name", "unknown")),
            source_locator=str(
                result.metadata.get("source_locator")
                or result.metadata.get("loader")
                or result.metadata.get("endpoint")
                or "adapter"
            ),
            frame=result.data,
            request_parameters={
                key: value
                for key, value in result.metadata.items()
                if key not in {"source_name", "source_retrieved_at_utc"}
            },
        )
        manifest.write_json(directories["manifests"] / f"{dataset_name}.json")
        return path

    def run_open_sources(self) -> dict[str, str]:
        directories = self._dirs()
        outputs: dict[str, str] = {}

        schedules = self.nflverse.load_games(self.config.seasons)
        outputs["nflverse_games"] = str(
            self._save_result(
                dataset_name="nflverse_games",
                result=schedules,
                directories=directories,
            )
        )
        canonical_games = schedules.data.copy()

        if self.config.spreadspoke_locator:
            spread = SpreadspokeAdapter(self.config.spreadspoke_locator).load_games(
                self.config.seasons
            )
            outputs["spreadspoke_games"] = str(
                self._save_result(
                    dataset_name="spreadspoke_games",
                    result=spread,
                    directories=directories,
                )
            )
            crosswalk = crosswalk_spreadspoke_to_nflverse(
                spread.data,
                schedules.data,
            )
            outputs["game_crosswalk"] = str(
                write_frame(crosswalk, directories["canonical"] / "game_crosswalk")
            )
            canonical_games = enrich_games_with_spreadspoke(
                schedules.data,
                spread.data,
                crosswalk,
            )

        outputs["games"] = str(
            write_frame(canonical_games, directories["canonical"] / "games")
        )

        player_stats = self.nflverse.load_player_stats(self.config.seasons)
        team_stats = self.nflverse.load_team_stats(self.config.seasons)
        weekly_rosters = self.nflverse.load_weekly_rosters(self.config.seasons)
        depth_charts = self.nflverse.load_depth_charts(self.config.seasons)
        snap_counts = self.nflverse.load_snap_counts(self.config.seasons)

        for name, result in (
            ("player_stats", player_stats),
            ("team_stats", team_stats),
            ("weekly_rosters", weekly_rosters),
            ("depth_charts", depth_charts),
            ("snap_counts", snap_counts),
        ):
            outputs[name] = str(
                self._save_result(
                    dataset_name=name,
                    result=result,
                    directories=directories,
                )
            )

        injury_seasons = tuple(
            season
            for season in self.config.seasons
            if season <= self.config.include_nflverse_injuries_through
        )
        if injury_seasons:
            injuries = self.nflverse.load_injuries(injury_seasons)
            outputs["nflverse_injuries"] = str(
                self._save_result(
                    dataset_name="nflverse_injuries",
                    result=injuries,
                    directories=directories,
                )
            )

        if self.config.include_large_nflverse_tables:
            pbp = self.nflverse.load_pbp(self.config.seasons)
            rosters = self.nflverse.load_rosters(self.config.seasons)
            outputs["pbp"] = str(
                self._save_result(dataset_name="pbp", result=pbp, directories=directories)
            )
            outputs["rosters"] = str(
                self._save_result(
                    dataset_name="rosters",
                    result=rosters,
                    directories=directories,
                )
            )

        write_json(
            {
                "completed_at_utc": utc_now_iso(),
                "seasons": list(self.config.seasons),
                "outputs": outputs,
            },
            Path(self.config.output_dir) / "backfill_summary.json",
        )
        return outputs

    def plan_historical_odds(self, games: pd.DataFrame) -> pd.DataFrame:
        directories = self._dirs()
        plan = build_prediction_horizons(
            games,
            horizon_minutes=self.config.odds_horizon_minutes,
        )
        # The historical endpoint returns all events at a snapshot. De-duplicate
        # timestamps before estimating quota.
        unique_requests = plan[["requested_snapshot_utc"]].drop_duplicates()
        market_count = len(self.odds.config.markets) if self.odds else 3
        region_count = len(self.odds.config.regions) if self.odds else 1
        estimated = TheOddsAPIAdapter.estimate_historical_credits(
            len(unique_requests),
            region_count=region_count,
            market_count=market_count,
        )
        plan.attrs["unique_request_count"] = len(unique_requests)
        plan.attrs["estimated_credits"] = estimated
        write_frame(plan, directories["plans"] / "odds_snapshot_plan")
        write_json(
            {
                "rows": len(plan),
                "unique_request_count": len(unique_requests),
                "estimated_credits": estimated,
                "horizon_minutes": list(self.config.odds_horizon_minutes),
            },
            directories["plans"] / "odds_snapshot_plan_summary.json",
        )
        return plan

    def run_historical_odds(
        self,
        games: pd.DataFrame,
        *,
        resume_from_utc: str | None = None,
    ) -> Path:
        if self.odds is None:
            self.odds = TheOddsAPIAdapter(OddsAPIConfig())
        plan = self.plan_historical_odds(games)
        request_times = sorted(plan["requested_snapshot_utc"].dropna().unique())
        estimated = int(plan.attrs["estimated_credits"])
        if not self.config.confirm_odds_cost:
            raise RuntimeError(
                f"Odds plan requires approximately {estimated} API credits. "
                "Set confirm_odds_cost=True only after reviewing the plan."
            )
        if self.config.max_odds_credits <= 0 or estimated > self.config.max_odds_credits:
            raise RuntimeError(
                f"Estimated odds cost {estimated} exceeds max_odds_credits="
                f"{self.config.max_odds_credits}."
            )

        if resume_from_utc:
            cutoff = pd.Timestamp(resume_from_utc)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            request_times = [
                value for value in request_times if pd.Timestamp(value) >= cutoff
            ]

        frames: list[pd.DataFrame] = []
        for snapshot in request_times:
            result = self.odds.historical_snapshot(pd.Timestamp(snapshot))
            if len(result.data):
                frames.append(result.data)

        odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(odds):
            odds = match_odds_to_games(odds, games)
            odds = devig_two_way_groups(odds)
            odds = add_market_consensus(odds)

        directories = self._dirs()
        path = write_frame(odds, directories["canonical"] / "odds_snapshots")
        SourceManifest.from_frame(
            dataset_name="odds_snapshots",
            source_name="the_odds_api",
            source_locator="historical/sports/americanfootball_nfl/odds",
            frame=odds,
            request_parameters={
                "requested_snapshots": len(request_times),
                "estimated_credits": estimated,
            },
        ).write_json(directories["manifests"] / "odds_snapshots.json")
        return path
