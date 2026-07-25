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

## Documentation

- `docs/PHASE_A_SOURCE_SPEC.md`
- `docs/SOURCE_FIELD_MAPPING.csv`
- `docs/CANONICAL_DATA_DICTIONARY.csv`
- `docs/ODDS_BACKFILL_POLICY.md`
- `docs/2026_PRIOR_DATA_PLAN.md`
- `docs/PHASE_C_MODEL_REFINEMENT.md`
- `docs/UNRESOLVED_SOURCE_GAPS.md`
