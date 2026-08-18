# Changelog

## 2026-08-18 — 2026 remediation: data resolution and chronological state

- **Fix 2 — authoritative chronological historical pregame state** (PR #14):
  added `build_authoritative_pregame_state` as the single entry point for the
  EPA, opponent-adjusted, QB, Elo, and market (ATS/OU) history state
  families; added new Elo-state and market-history adapters
  (`features/elo_state.py`, `features/market_history.py`); added per-family
  chronological lineage and cross-family invariant checking
  (`features/state_lineage.py`); added the machine-readable
  `docs/PREGAME_STATE_DICTIONARY.yaml` state dictionary.
- **Fix 1.5 — centralized external-data resolution** (PR #13): added
  `src/nfl_hybrid/data/external_data.py` as the single resolver for the
  private historical-data estate (`NFL_MODEL_DATA_ROOT`); unified every
  producer script and consumer onto the same registry keys, removing
  hard-coded relative-path literals; added `scripts/validate_data_registry.py`
  / `nfl-hybrid-validate-data` to verify the estate resolves correctly.
- Moved test/CI workflows to GitHub-hosted Ubuntu runners
  (`.github/workflows/tests.yml`), with a dedicated core live-feature
  parity/leakage check that must not skip.
- (Already on `main` prior to this date, included here for continuity)
  completed the canonical nullable tie/push target policy
  (`nfl_hybrid.labels.edge_to_nullable_binary`) across the pricing
  calibration, joint-score, and EPA-tournament pipelines, with zero-valid
  grading protection.

## 0.3.0 — Phase A plus initial Phase C

- locked Spreadspoke, nflverse, The Odds API, NFL archive, and Sportradar source roles;
- added raw/canonical ingestion, provenance manifests, as-of checks, and cost guardrails;
- added exact source-field mapping and 378-field canonical dictionary;
- added direct and market-residual joint-score comparison;
- added team-game efficiency and opponent-adjustment features;
- added empirical-Bayes team, quarterback, roster, and coach prior components;
- added prior CLI, templates, and provisional tuning grid;
- expanded test suite to 25 passing tests.

## 0.1.0 — Workbook reconstruction

- corrected spreadsheet baselines;
- joint-score model;
- initial walk-forward benchmark and in-play framework.
