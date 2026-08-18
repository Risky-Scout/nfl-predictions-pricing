# NFL Hybrid Moneyline, ATS, Totals, and In-Play Model

An auditable NFL forecasting and pricing system that reconstructs and corrects
a legacy workbook model, layers a leakage-safe chronological feature/state
pipeline on top, and prices Moneyline, ATS, and totals markets against the
real market rather than claiming edge it hasn't earned. As of 2026-08-18 the
weekly production card runs on the legacy/market-baseline pricing path — the
learned (EPA/opponent/QB/Elo state) engine exists, is tested, and is under
active remediation, but is **not yet** the production weekly engine. See
`STATUS.md` and `REMEDIATION_STATUS_2026.md` for exactly what is merged vs.
in progress.

## 2026 remediation status (short version)

A forensic audit of the original build surfaced correctness and leakage
defects in the historical data plumbing and the chronological feature
pipeline. Remediation is proceeding as a sequence of independently reviewed
fixes on `main`:

| Fix | Scope | Status |
|---|---|---|
| Fix 1.5 | Centralized external-data resolution, unified producer/consumer paths | **MERGED** |
| Fix 2 | Authoritative chronological historical pregame state (EPA, opponent-adjusted, QB, Elo, market history) + lineage | **MERGED** |
| Fix 3 | Chronological out-of-fold (OOF) evaluation + uncertainty | **IN PROGRESS — not yet merged to `main`** |

Do not treat anything under Fix 3 (chronological OOF membership proofs,
chronological calibration, or any uncertainty output derived from it) as
current system behavior until it lands on `main`. Full roadmap and evidence
links: `REMEDIATION_STATUS_2026.md`.

## Locked source stack

- **SpreadSpoke:** workbook continuity, results, reference lines, stadium, and
  observed/reference weather.
- **nflverse (via `nflreadpy`):** schedules, play-by-play, player/team box
  stats, rosters, weekly rosters, snaps, depth charts, and injuries.
- **The Odds API (provider-agnostic interface):** timestamped historical and
  current-week odds snapshots.
- **2025–2026 injuries:** official NFL archive export fallback or an
  automated Sportradar-compatible feed.

Every normalized dataset retains provider IDs, retrieval timestamps, and a
SHA-256 manifest. Raw provider data is retained.

## Central data architecture: `NFL_MODEL_DATA_ROOT`

All access to the private, git-ignored historical data estate — the nflverse
backfill parquets and the timestamped Odds API history snapshot stores — goes
through a single resolver, `src/nfl_hybrid/data/external_data.py`, instead of
hard-coded relative paths. The physical location is supplied entirely at
runtime:

```bash
export NFL_MODEL_DATA_ROOT=/path/to/your-nfl-data-estate
```

The resolver exposes a registry of logical dataset keys (e.g.
`backfill.games`, `backfill.pbp`, `odds_history.2020_2023`) that every
producer script and every consumer resolve identically — there is no
per-script fallback to a smoke-test or partial dataset. A small number of
already-vendored, committed datasets (e.g. the purchased closing-odds dev
parquet) resolve relative to the repo itself and don't depend on this
variable. Verify your local estate resolves correctly with:

```bash
PYTHONPATH=src python scripts/validate_data_registry.py
```

## Historical data roles

- **SpreadSpoke** — long-run (1966–2025) game results, closing reference
  lines, stadium, and observed weather; the only source with that historical
  depth, but no timestamped two-way odds and no play-by-play.
- **nflverse (play-by-play, box scores, rosters, depth charts, snaps,
  injuries)** — the football backbone for 2020–2025: EPA-ready play-by-play,
  team/player box stats, and roster/injury context, resolved through the
  `backfill.*` registry keys above.
- **Timestamped Odds API history** — three distinct acquisition runs
  (`odds_history.2020_2023`, `odds_history.2024_confirmation`,
  `odds_history.2025_final_test`), each a separate snapshot store used for
  genuine closing-line evaluation and CLV — never conflated with SpreadSpoke's
  reference lines.

## Authoritative chronological pregame-state interface

`src/nfl_hybrid/features/pregame_state.py` (`build_authoritative_pregame_state`)
is the single entry point for chronologically correct pregame state. It
orchestrates five state families — EPA/box-score, opponent-adjusted, QB,
Elo, and market (ATS/OU) history — each already leakage-safe by construction,
and adds what none of them individually provided:

- a per-family **lineage record** of exactly which prior games generated each
  row (`src/nfl_hybrid/features/state_lineage.py`);
- a cross-family **banned-column check** so no state family can smuggle in a
  target/outcome column (`src/nfl_hybrid/features/feature_manifest.py`).

The machine-readable family-by-family contract (lookback window, lag rule,
missingness behavior, historical/live reproducibility, production-wiring
status) lives in `docs/PREGAME_STATE_DICTIONARY.yaml`. Not every state family
is wired into the live weekly pricing path yet — the dictionary's
`production_wired` field is the authoritative source for which ones are.

## Leakage policy

- Every pregame feature family must emit its row's state strictly before
  incorporating that row's own outcome (shift-then-roll or a strict
  `snapshot_date <` cut, enforced in code and unit-tested per family).
- A single banned-column policy (`feature_manifest.validate_no_banned_features`)
  is checked against every state family before it can reach a caller.
- Historical replay vs. live scoring parity is tracked explicitly per family
  in `docs/PREGAME_STATE_DICTIONARY.yaml` (`historical_live_reproducibility`)
  — a family with a known gap says so rather than silently assuming parity.
- Validation is always expanding season/week walk-forward. A random train/test
  split is never used.

## Packageability and auditability

The project installs as a normal Python package (`pip install -e .`), ships
console-script entry points for backfill, priors, data-registry validation,
and weekly pricing, and keeps provenance (source manifests, SHA-256 hashes,
retrieval timestamps) attached to every derived dataset so a run can be
audited back to its raw inputs. This auditability objective — not
maximizing model complexity — is the project's primary design constraint.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,dev]"
pytest -q
```

CI (`.github/workflows/tests.yml`) runs the full suite on Python 3.11 and
3.14 on GitHub-hosted Linux runners, plus a dedicated core live-feature
parity/leakage check (`tests/test_live_features_ci.py`) that must pass and
must not skip.

## Configure credentials and data locations

```bash
cp .env.example .env
export NFL_MODEL_DATA_ROOT="/path/to/your-nfl-data-estate"
export SPREADSPOKE_CSV_PATH="/path/to/nfl_games.csv"
export THE_ODDS_API_KEY="..."
# Optional automated 2025–2026 injury feed
export SPORTRADAR_API_KEY="..."
```

Keys are read from environment variables and are never embedded in the
project.

## Backfill open football sources

```bash
nfl-hybrid-backfill --output-dir "$NFL_MODEL_DATA_ROOT/backfill-2020-2025" \
  --seasons 2020-2025 --spreadspoke-csv "$SPREADSPOKE_CSV_PATH" open-sources
```

## Plan timestamped odds before spending credits

```bash
nfl-hybrid-backfill --output-dir "$NFL_MODEL_DATA_ROOT/backfill-2020-2025" \
  plan-odds --games "$NFL_MODEL_DATA_ROOT/backfill-2020-2025/canonical/games.parquet"
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
nfl-hybrid-priors team \
  --history data/templates/team_metric_history.csv \
  --target-season 2026 --as-of-utc 2026-08-01T16:00:00Z \
  --output outputs/team_priors_2026.parquet
```

```bash
nfl-hybrid-priors qb-mixture \
  --history data/templates/quarterback_history.csv \
  --starter-probabilities data/templates/starter_probabilities.csv \
  --as-of-utc 2026-08-01T16:00:00Z \
  --player-output outputs/qb_player_priors_2026.parquet \
  --team-output outputs/qb_team_mixtures_2026.parquet
```

The included prior hyperparameters are provisional.

## 2026 season operations (current production path)

The 2026 EPA candidate tournament promoted **no** market
(`docs/model-selection/candidates-2026/results.md`), so every market is
`RETAIN_BASELINE` and the weekly card **stakes zero**. The weekly loop runs
anyway as the live search instrument, accumulating an immutable
out-of-sample record. This is the legacy/market-baseline pricing path — see
`OPERATOR_MANUAL.md`.

```bash
export PYTHONPATH="$PWD/src"
export SPREADSPOKE_CSV_PATH="$PWD/data/spreadspoke_enhanced.csv"
export THE_ODDS_API_KEY="..."       # optional; enables live current-week lines

# one command per week (pre-kickoff):
scripts/run_week.sh 2026 1
```

`run_week.sh` performs: (1) incremental data refresh; (2) fetch current-week
lines via the Odds API current-odds endpoint or a lines CSV template if
unkeyed; (3) generate the betting card — PROVISIONAL markets print
probabilities but stake **zero**; (4) log an immutable pre-kickoff record to
`outputs/season_2026/predictions_wkNN.csv` (UTC-timestamped, never
overwritten); (5) after games resolve, score vs. the market and update the
drift monitor; (6) apply the live-promotion gate.

Rehearse the whole loop on 2025 week-1 real data (no selection is made from
2025):

```bash
.venv/bin/python scripts/dryrun_week_loop.py
```

## What is implemented vs. in remediation vs. planned

**Implemented / merged to `main`:**
- corrected legacy workbook Elo, QB, scoring, totals, market, and 32-model
  ensemble;
- centralized external-data resolution (`NFL_MODEL_DATA_ROOT`, Fix 1.5);
- authoritative chronological pregame-state interface with per-family lineage
  (Fix 2);
- direct joint margin/total model and market-residual challenger;
- separate Moneyline/ATS/totals probability calibration heads;
- explicit tie and push probabilities with nullable tie/push target handling;
- market-relative evaluation, staking/abstention gate, and calibration-drift
  monitoring for the weekly loop.

**In remediation (not yet merged):**
- chronological out-of-sample (OOF) evaluation + uncertainty (Fix 3).

**Planned / not yet started:**
- chronological calibration built on top of Fix 3;
- promoting the learned engine to be the production weekly engine;
- BDL live-game/box/PBP integration;
- twice-weekly learned model refitting.

See `REMEDIATION_STATUS_2026.md` for the full roadmap with evidence links.

## Documentation

- `STATUS.md` — current status by component.
- `REMEDIATION_STATUS_2026.md` — single source of truth for remediation
  progress.
- `MODEL_ARCHITECTURE.md` — architecture, with MERGED / LEGACY / PLANNED
  labeling.
- `docs/PREGAME_STATE_DICTIONARY.yaml` — machine-readable pregame-state
  family contract.
- `docs/PHASE_A_SOURCE_SPEC.md`
- `docs/SOURCE_FIELD_MAPPING.csv`
- `docs/CANONICAL_DATA_DICTIONARY.csv`
- `docs/ODDS_BACKFILL_POLICY.md`
- `docs/2026_PRIOR_DATA_PLAN.md`
- `docs/PHASE_C_MODEL_REFINEMENT.md`
- `docs/UNRESOLVED_SOURCE_GAPS.md`
- `OPERATOR_MANUAL.md` — legacy/market-baseline weekly operating path
  (2026 learned-production runner is still under remediation).
