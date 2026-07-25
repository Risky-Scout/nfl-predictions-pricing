from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from nfl_hybrid.data.io import read_tabular
from nfl_hybrid.data.provenance import utc_now_iso
from nfl_hybrid.data.team_ids import try_canonical_team_id
from nfl_hybrid.data.providers.base import ProviderResult


_PRACTICE_NORMALIZATION = {
    "DID NOT PARTICIPATE IN PRACTICE": "DNP",
    "DID NOT PARTICIPATE": "DNP",
    "DNP": "DNP",
    "LIMITED PARTICIPATION IN PRACTICE": "LIMITED",
    "LIMITED PARTICIPATION": "LIMITED",
    "LIMITED": "LIMITED",
    "FULL PARTICIPATION IN PRACTICE": "FULL",
    "FULL PARTICIPATION": "FULL",
    "FULL": "FULL",
    "": "NOT_LISTED",
}

_GAME_NORMALIZATION = {
    "OUT": "OUT",
    "DOUBTFUL": "DOUBTFUL",
    "QUESTIONABLE": "QUESTIONABLE",
    "NO DESIGNATION": "NO_DESIGNATION",
    "": "NO_DESIGNATION",
}


def _clean_column(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _first(frame: pd.DataFrame, names: Iterable[str], default: object = pd.NA) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _normalize_practice(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    return _PRACTICE_NORMALIZATION.get(text, text.replace(" ", "_"))


def _normalize_game_status(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    return _GAME_NORMALIZATION.get(text, text.replace(" ", "_"))


@dataclass
class NFLOfficialInjuryAdapter:
    """Official NFL archive adapter.

    NFL.com exposes season/week archive pages but does not publish a stable,
    documented public injury API. For reproducibility, the preferred workflow
    is to save/export each page and ingest the resulting CSV/HTML table with
    team labels retained. Direct HTML reads are best-effort only.
    """

    source_name: str = "nfl_official_injury_archive"
    base_url: str = "https://www.nfl.com/injuries/league"

    @staticmethod
    def week_token(week: int | str, *, season_type: str = "REG") -> str:
        text = str(week).strip().lower().replace(" ", "_")
        if season_type.upper() in {"REG", "REGULAR"} and text.isdigit():
            return f"reg{int(text)}"
        aliases = {
            "wild_card": "post1",
            "wildcard": "post1",
            "post1": "post1",
            "divisional": "post2",
            "divisional_playoff": "post2",
            "post2": "post2",
            "conference": "post3",
            "conference_championship": "post3",
            "post3": "post3",
            "super_bowl": "post4",
            "superbowl": "post4",
            "post4": "post4",
        }
        if text in aliases:
            return aliases[text]
        if text.startswith(("reg", "post")):
            return text
        raise ValueError(f"Unsupported NFL injury week token: {week!r}")

    def build_url(
        self,
        season: int,
        week: int | str,
        *,
        season_type: str = "REG",
    ) -> str:
        token = self.week_token(week, season_type=season_type)
        return f"{self.base_url.rstrip('/')}/{int(season)}/{token}"

    def load_export(
        self,
        locator: str | Path,
        *,
        season: int,
        week: int | str,
        report_timestamp_utc: str | None = None,
    ) -> ProviderResult:
        raw = read_tabular(locator)
        raw.columns = [_clean_column(column) for column in raw.columns]
        retrieved = utc_now_iso()
        team = _first(raw, ("team_id", "team", "team_alias", "club"))
        frame = pd.DataFrame(
            {
                "season": int(season),
                "week": str(week),
                "team_id": team.map(try_canonical_team_id),
                "provider_team_id": _first(raw, ("provider_team_id",)),
                "player_id": _first(raw, ("player_id", "nfl_player_id")),
                "player_name": _first(raw, ("player", "player_name", "name")),
                "position": _first(raw, ("position", "pos")),
                "injury_primary": _first(
                    raw, ("injuries", "injury", "injury_primary", "body_part")
                ),
                "practice_status_raw": _first(
                    raw, ("practice_status", "practice_participation")
                ),
                "game_status_raw": _first(raw, ("game_status", "status")),
                "status_date_utc": pd.to_datetime(
                    _first(
                        raw,
                        ("status_date_utc", "report_timestamp_utc", "report_date"),
                        report_timestamp_utc,
                    ),
                    utc=True,
                    errors="coerce",
                ),
                "estimated_return_date": pd.to_datetime(
                    _first(raw, ("estimated_return_date",)),
                    utc=True,
                    errors="coerce",
                ),
                "source_page_url": _first(raw, ("source_page_url",), str(locator)),
                "source_name": self.source_name,
                "source_retrieved_at_utc": retrieved,
            }
        )
        frame["practice_status"] = frame["practice_status_raw"].map(_normalize_practice)
        frame["game_status"] = frame["game_status_raw"].map(_normalize_game_status)

        if frame["team_id"].isna().any():
            missing = int(frame["team_id"].isna().sum())
            raise ValueError(
                f"Official injury export has {missing} rows without a recognized team. "
                "Retain a team/team_id column when exporting NFL.com tables."
            )
        return ProviderResult(
            data=frame,
            metadata={
                "source_name": self.source_name,
                "source_locator": str(locator),
                "season": int(season),
                "week": str(week),
                "source_retrieved_at_utc": retrieved,
            },
        )

    def load_injuries(
        self,
        season: int,
        week: int | str,
        *,
        season_type: str = "REG",
    ) -> ProviderResult:
        url = self.build_url(season, week, season_type=season_type)
        try:
            tables = pd.read_html(url, match="Player")
        except Exception as exc:
            raise RuntimeError(
                "Direct NFL.com table ingestion failed. Save/export the official "
                f"archive page ({url}) with team labels and use load_export(), "
                "or configure the Sportradar adapter."
            ) from exc

        frames: list[pd.DataFrame] = []
        for table_number, table in enumerate(tables):
            table = table.copy()
            table.columns = [_clean_column(column) for column in table.columns]
            if "team" not in table.columns and "team_id" not in table.columns:
                # The rendered page often separates team headings from tables;
                # silently guessing a team would corrupt injury features.
                continue
            table["source_page_url"] = url
            frames.append(table)

        if not frames:
            raise RuntimeError(
                "NFL.com returned injury tables without machine-readable team labels. "
                "Use a saved export with a team column or Sportradar."
            )

        combined = pd.concat(frames, ignore_index=True)
        temporary = Path("nfl_official_injuries_temp.csv")
        combined.to_csv(temporary, index=False)
        try:
            return self.load_export(
                temporary,
                season=season,
                week=week,
                report_timestamp_utc=utc_now_iso(),
            )
        finally:
            temporary.unlink(missing_ok=True)
