# NFL Hybrid Model — Phase A and Initial Phase C Build Report

**Version:** 0.3.0
**Target historical extension:** 2020–2025
**Target prior season:** 2026

## Source decision implemented

| Layer | Implemented source |
|---|---|
| Workbook continuity | Spreadspoke file adapter |
| Football backbone | nflverse through `nflreadpy` |
| Timestamped odds | Provider interface with The Odds API implementation |
| Historical injuries, 2020–2024 | nflverse |
| Injuries, 2025–2026 | Official NFL archive export or Sportradar-compatible adapter |

The code keeps reference lines and timestamped market quotes separate.
Spreadspoke and nflverse game-level line fields cannot populate a requested
pregame horizon unless their publication time is known.

## Phase A delivered

- raw and canonical storage layout;
- source manifests with retrieval time, parameters, row/column counts, and
  SHA-256 dataframe fingerprints;
- current/historical team alias normalization;
- Spreadspoke game normalization and nflverse crosswalk;
- nflverse schedules, play-by-play, stats, roster, snap, depth-chart, injury,
  and player loaders;
- canonical sportsbook spread-sign conversion;
- provider-agnostic odds interface;
- The Odds API current and historical adapters;
- bookmaker/outcome flattening, implied probability, de-vigging, consensus,
  dispersion, game matching, and standard prediction horizons;
- paid-query planning and cost guardrails;
- official NFL injury export adapter;
- Sportradar-compatible weekly injury adapter;
- as-of leakage checks;
- 378-row canonical data dictionary;
- 197-row source-to-canonical field map.

## Phase C delivered

- direct joint margin/total model;
- market-residual joint margin/total challenger;
- separate Moneyline, ATS, and total calibrators;
- explicit tie and push probabilities;
- team-game efficiency aggregation from play-by-play;
- ridge opponent adjustment;
- empirical-Bayes team/unit priors;
- recency/dropback-weighted QB priors;
- starter probability mixtures with uncertainty;
- returning-snap roster continuity;
- learned roster adjustment;
- hierarchical coach priors;
- prior-building command line interface;
- provisional 2026 hyperparameter grid and canonical input templates.

## Verification

- **26 automated tests passed**
- Python source and examples compile successfully.
- Both command-line interfaces return valid help output.
- Workbook-history Phase C comparison executed and wrote prediction/metric
  files.

## Current research result

The market-residual architecture improved the 2018 Moneyline, total, and
continuous total result, but did not improve all 2019 targets. The direct joint
model was stronger for 2019 Moneyline, ATS, and margin. The market line itself
remained difficult to beat consistently.

This is the expected decision rule:

1. retain both architectures;
2. run 2020–2025 walk-forward comparisons;
3. tune on 2021–2023;
4. select on 2024;
5. preserve 2025 as untouched final validation;
6. freeze the 2026 prior/model configuration only afterward.

## Inputs still required to execute the real backfill

The package is executable, but this environment was not supplied with:

```text
SPREADSPOKE_CSV_PATH
THE_ODDS_API_KEY
SPORTRADAR_API_KEY   # optional if official NFL exports are used
```

Therefore, no claim is made that the 2020–2025 backfill or actual 2026 priors
have already been populated. The code, contracts, and commands needed to do so
are included.

A timestamped historical forecast-weather source is also still required for a
fully as-of-correct pregame weather feature set.

## Primary commands

```bash
pip install -e ".[data,dev]"
pytest
```

```bash
nfl-hybrid-backfill \
  --output-dir data/backfill_2020_2025 \
  --seasons 2020-2025 \
  --spreadspoke-csv "$SPREADSPOKE_CSV_PATH" \
  open-sources
```

```bash
nfl-hybrid-backfill \
  --output-dir data/backfill_2020_2025 \
  plan-odds \
  --games data/backfill_2020_2025/canonical/games.parquet
```

```bash
python examples/compare_phase_c_models.py
```

## High-value files

- `README.md`
- `config/data_sources.yaml`
- `config/prior_hyperparameters.yaml`
- `docs/PHASE_A_SOURCE_SPEC.md`
- `docs/SOURCE_FIELD_MAPPING.csv`
- `docs/CANONICAL_DATA_DICTIONARY.csv`
- `docs/ODDS_BACKFILL_POLICY.md`
- `docs/2026_PRIOR_DATA_PLAN.md`
- `docs/PHASE_C_MODEL_REFINEMENT.md`
- `docs/UNRESOLVED_SOURCE_GAPS.md`
- `outputs/phase_c_model_metrics.csv`
