# NFL Hybrid Moneyline, ATS, Totals, and In-Play Model

Version 0.3 implements **Solution B** through Phase A and the first Phase C
research framework.

## Locked source stack

- **Spreadspoke:** workbook continuity, results, reference lines, stadium, and
  observed/reference weather.
- **nflverse through nflreadpy:** schedules, play-by-play, player/team stats,
  rosters, weekly rosters, snaps, depth charts, players, and 2020–2024 injuries.
- **Provider-agnostic odds interface:** initial implementation for The Odds API.
- **2025–2026 injuries:** official NFL archive export fallback or automated
  Sportradar-compatible feed.

Every normalized dataset retains provider IDs, retrieval timestamps, and a
SHA-256 manifest. Raw provider data is retained.

## Implemented model layers

- corrected workbook Elo, QB, scoring, totals, market, and 32-model ensemble;
- leakage-safe sequential workbook feature builder;
- direct joint margin/total model;
- market-residual joint margin/total challenger;
- separate Moneyline, ATS, and totals probability calibration;
- explicit tie and push probabilities;
- joint score simulation;
- team-game play-by-play efficiency aggregation;
- opponent-adjusted team ratings;
- empirical-Bayes team/unit priors;
- recency-weighted quarterback priors and starter mixtures;
- roster-continuity features and learned adjustments;
- hierarchical coach priors;
- provider-agnostic in-play remaining-score framework.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,dev]"
pytest
```

## Configure credentials and source files

```bash
cp .env.example .env
export SPREADSPOKE_CSV_PATH="/path/to/nfl_games.csv"
export THE_ODDS_API_KEY="..."
# Optional automated 2025–2026 injury feed
export SPORTRADAR_API_KEY="..."
```

Keys are read from environment variables and are never embedded in the project.

## Backfill open football sources

```bash
nfl-hybrid-backfill   --output-dir data/backfill_2020_2025   --seasons 2020-2025   --spreadspoke-csv "$SPREADSPOKE_CSV_PATH"   open-sources
```

## Plan timestamped odds before spending credits

```bash
nfl-hybrid-backfill   --output-dir data/backfill_2020_2025   plan-odds   --games data/backfill_2020_2025/canonical/games.parquet
```

The execution command requires both an explicit credit ceiling and
`--confirm-cost`.

## Compare model architectures

```bash
python examples/compare_phase_c_models.py
```

Outputs:

```text
outputs/phase_c_model_predictions.csv
outputs/phase_c_model_metrics.csv
```

## Build prior tables

```bash
nfl-hybrid-priors team   --history data/templates/team_metric_history.csv   --target-season 2026   --as-of-utc 2026-08-01T16:00:00Z   --output outputs/team_priors_2026.parquet
```

```bash
nfl-hybrid-priors qb-mixture   --history data/templates/quarterback_history.csv   --starter-probabilities data/templates/starter_probabilities.csv   --as-of-utc 2026-08-01T16:00:00Z   --player-output outputs/qb_player_priors_2026.parquet   --team-output outputs/qb_team_mixtures_2026.parquet
```

The included prior hyperparameters are provisional. Actual 2026 priors must be
frozen only after the complete 2020–2025 walk-forward replay.

## 2026 Season Operations

The 2026 EPA candidate tournament promoted **no** market
(`docs/model-selection/candidates-2026/results.md`), so every market is
`RETAIN_BASELINE` and the weekly card **stakes zero**. The weekly loop runs anyway
as the live search instrument, accumulating an immutable out-of-sample record.

```bash
export PYTHONPATH="$PWD/src"
export SPREADSPOKE_CSV_PATH="$PWD/data/spreadspoke_enhanced.csv"
export THE_ODDS_API_KEY="..."       # optional; enables live current-week lines

# one command per week (pre-kickoff):
scripts/run_week.sh 2026 1
```

`run_week.sh` performs: (1) incremental data refresh; (2) fetch current-week lines
via the Odds API current-odds endpoint (**3 credits/call**) or a lines CSV
template if unkeyed; (3) generate the betting card — PROVISIONAL markets print
probabilities but stake **zero**; (4) log an immutable pre-kickoff record to
`outputs/season_2026/predictions_wkNN.csv` (UTC-timestamped, never overwritten);
(5) after games resolve, score vs the market and update the drift monitor;
(6) apply the **live-promotion gate**.

**Live-promotion gate (fixed):** a PROVISIONAL market may stake (1/8 Kelly, 5%
per-bet / 15% weekly caps) only after **≥ 8 live weeks** with **cumulative
log-loss gain > 0** *and* **live pick-accuracy 95% CI lower bound > 0.524**. A
drift alarm demotes immediately. Backtests never grant live staking; only the
live record can.

Rehearse the whole loop on 2025 week-1 real data (no selection is made from 2025):

```bash
.venv/bin/python scripts/dryrun_week_loop.py
```

## Documentation

- `docs/PHASE_A_SOURCE_SPEC.md`
- `docs/SOURCE_FIELD_MAPPING.csv`
- `docs/CANONICAL_DATA_DICTIONARY.csv`
- `docs/ODDS_BACKFILL_POLICY.md`
- `docs/2026_PRIOR_DATA_PLAN.md`
- `docs/PHASE_C_MODEL_REFINEMENT.md`
- `docs/UNRESOLVED_SOURCE_GAPS.md`
