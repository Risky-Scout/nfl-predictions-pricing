# Phase A Source and Ingestion Specification

## Locked source stack

| Layer | Source | Authority in the model |
|---|---|---|
| Workbook continuity | Spreadspoke | Historical game result, reference spread/total, stadium, observed weather, cross-source reconciliation |
| Football backbone | nflverse through `nflreadpy` | Schedules, scores, play-by-play, player/team stats, rosters, weekly rosters, snap counts, depth charts, player IDs |
| Timestamped market | Provider-agnostic adapter; initial implementation is The Odds API | Bookmaker, market, outcome, point, price, snapshot, update time, and kickoff |
| 2020–2024 injuries | nflverse | Historical practice and game-status records |
| 2025–2026 injuries | Official NFL archive export or Sportradar v7 | Current practice status, game status, injury, status timestamp, player/team IDs |

## Source separation rules

1. Spreadspoke lines are named `*_reference`. Their historical publication
   timestamp is unknown, so they are not valid 7-day, 72-hour, 24-hour,
   6-hour, 1-hour, or 15-minute market snapshots.
2. The Odds API is the only initial source permitted to populate
   `odds_snapshots.snapshot_utc`.
3. Observed temperature/wind fields are stored separately from forecast
   fields. An observed game-day value cannot be backfilled into a pregame
   forecast row without an explicit `proxy_flag`.
4. nflverse game and player statistics become available only after the
   relevant game is completed and the upstream release has been published.
5. Injury records are versioned by `status_date_utc`; later Friday or inactive
   reports cannot be joined into a Wednesday prediction.
6. All provider IDs are retained. Internal game IDs use nflverse `game_id` for
   2020 onward, with a Spreadspoke crosswalk.
7. Current franchise aliases are used internally: LV, LAC, LAR, TEN, and WAS.
   Historical names remain available as attributes.

## Reproducibility

Every written dataset has a JSON source manifest containing:

- dataset name;
- source name and locator;
- retrieval timestamp;
- row and column counts;
- dataframe SHA-256;
- request parameters;
- provider version/license note when supplied.

Raw provider tables are retained alongside normalized tables.

## Installation

```bash
pip install -e ".[data,dev]"
```

## Open-source backfill

```bash
nfl-hybrid-backfill \
  --output-dir data/backfill_2020_2025 \
  --seasons 2020-2025 \
  --spreadspoke-csv "$SPREADSPOKE_CSV_PATH" \
  open-sources
```

This downloads nflverse datasets and reconciles them to the user-supplied
Spreadspoke CSV.

## Odds planning and execution

Planning does not spend API credits:

```bash
nfl-hybrid-backfill \
  --output-dir data/backfill_2020_2025 \
  plan-odds \
  --games data/backfill_2020_2025/canonical/games.parquet
```

Execution requires `THE_ODDS_API_KEY`, an explicit maximum-credit ceiling, and
`--confirm-cost`.

```bash
nfl-hybrid-backfill \
  --output-dir data/backfill_2020_2025 \
  run-odds \
  --games data/backfill_2020_2025/canonical/games.parquet \
  --max-credits 25000 \
  --confirm-cost
```

The exact credit estimate is written before execution. Provider pricing and
quota rules may change, so the code never silently launches a paid backfill.

## Injuries

### Official NFL archive

The archive URL pattern is:

```text
https://www.nfl.com/injuries/league/{season}/{week_token}
```

Examples of tokens are `reg1`, `reg18`, `post1`, `post2`, `post3`, and
`post4`.

The official site does not expose a documented stable public injury API.
Therefore, the reproducible no-key method is to export the page's tables while
retaining a `team` or `team_id` column and call:

```python
from nfl_hybrid.data.providers import NFLOfficialInjuryAdapter

injuries = NFLOfficialInjuryAdapter().load_export(
    "official_nfl_2025_reg1.csv",
    season=2025,
    week=1,
)
```

The adapter refuses to guess a team from table order.

### Sportradar

For automated 2025–2026 ingestion:

```python
from nfl_hybrid.data.providers import SportradarInjuryAdapter

injuries = SportradarInjuryAdapter().load_injuries(
    season=2026,
    season_type="REG",
    week=1,
)
```

The key is read from `SPORTRADAR_API_KEY`.

## References

- Spreadspoke data dictionary and licensing: https://spreadspoke.com/data.html
- nflreadpy: https://github.com/nflverse/nflreadpy
- nflverse update schedule: https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html
- The Odds API v4: https://the-odds-api.com/liveapi/guides/v4/
- Official NFL injury archive: https://www.nfl.com/injuries/
- Sportradar weekly injuries: https://developer.sportradar.com/football/reference/nfl-weekly-injuries
