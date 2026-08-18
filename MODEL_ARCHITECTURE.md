# Architecture — current state, 2026-08-18

This document is organized as **CURRENT / MERGED** (on `main` today),
**LEGACY** (still present, still used, but superseded or challenger-only —
not "the" authoritative system), and **PLANNED / UNDER REMEDIATION** (does
not exist on `main` yet). See `STATUS.md` and `REMEDIATION_STATUS_2026.md`
for dates and evidence.

## 1. External/private data registry — **CURRENT / MERGED**

`src/nfl_hybrid/data/external_data.py` is the single resolver for the
private, git-ignored historical data estate. Every producer and consumer
resolves a logical dataset key (`backfill.games`, `backfill.pbp`,
`odds_history.2020_2023`, etc.) against one location, supplied at runtime via
`NFL_MODEL_DATA_ROOT` (or an explicit `root_override`) — no hard-coded
machine path, and no silent fallback to a partial/smoke dataset. A small set
of already-vendored, repo-committed datasets (`scope="repo"`, e.g. the
purchased closing-odds dev parquet) resolve relative to the repo checkout
instead and don't depend on the env var.

Two entry points:
- `resolve(key)` — strict; raises `ExternalDataUnavailableError` if the key,
  root, or path is missing. Use at the point data is actually read.
- `describe(key)` — pure, non-raising; the documented (unvalidated) location
  string, safe to call at import time without the env var set.

`scripts/validate_data_registry.py` (console script
`nfl-hybrid-validate-data`) attempts to resolve every registered key and
reports OK/MISSING per key.

## 2. Canonical historical data — **CURRENT / MERGED**

- **SpreadSpoke** (`data/spreadspoke_enhanced.csv`, vendored) — game
  results, closing reference lines, stadium, and observed weather,
  1966–2025.
- **nflverse, via `nflreadpy`** — play-by-play, team/player box stats,
  rosters, weekly rosters, snaps, depth charts, injuries, resolved through
  the `backfill.*` registry keys under `NFL_MODEL_DATA_ROOT`.
- **The Odds API, timestamped history** — three separate acquisition-run
  snapshot stores (`odds_history.2020_2023`, `odds_history.2024_confirmation`,
  `odds_history.2025_final_test`), never conflated with SpreadSpoke's
  reference lines.
- **Purchased closing odds, dev seasons 2022–2024**
  (`data/purchased/odds_closing_dev_2022_2024.parquet`) — a one-time,
  credit-limited purchase, committed to the repo, used for genuine
  closing-line calibration and evaluation.

Provenance: every normalized dataset carries provider IDs, retrieval
timestamps, and a SHA-256 manifest.

## 3. Authoritative chronological pregame-state builder — **CURRENT / MERGED**

`src/nfl_hybrid/features/pregame_state.py` (`build_authoritative_pregame_state`)
is the single entry point for chronologically correct pregame state. It does
not introduce new state math — every family it orchestrates was already
leakage-safe by construction — but it adds two things none of the individual
family builders provided on their own:

1. **Per-family lineage** (`src/nfl_hybrid/features/state_lineage.py`) —
   `sequential_prior_game_lineage` (team-direct scope: exactly the team's own
   own prior games) and `snapshot_prior_game_lineage` (leaguewide-prior-to-
   snapshot scope: every team's history before the cutoff, id list capped and
   flagged rather than silently dropped). `verify_pregame_state_lineage`
   checks, for every family: no self-reference, no unknown `game_id`, and no
   contributing game at or after the target's own kickoff.
2. **Cross-family banned-column validation**
   (`src/nfl_hybrid/features/feature_manifest.py`) — every family's non-
   identifier columns are checked against a single banned-column policy
   before the bundle is returned, so a target/outcome column can't reach a
   caller through any state family.

This orchestrator does not select a production feature set, train, tune, or
recalibrate anything, and does not change which family(ies)
`pricing/predict_week.py` already consumes — it is an additive, auditable
composition layer over pre-existing builders.

## 4. State families — **CURRENT / MERGED** (mixed production-wiring)

The full machine-readable contract per family (lookback, lag rule,
missingness behavior, historical/live reproducibility, production-wiring
status) is in `docs/PREGAME_STATE_DICTIONARY.yaml` — treat that file as
authoritative over this summary if they ever disagree.

| Family | What it represents | Lag rule | Production-wired |
|---|---|---|---|
| `epa` | Per-team offense/defense-allowed EPA, success rate, situational splits; rolling + season-to-date | `shift(1)` before any rolling/expanding/EWM aggregation | **Yes** — via `features/augmented_matrix.py` → `pricing/predict_week.py` |
| `opponent_adjusted` | Ridge-regression opponent-adjusted offense/defense ratings, partial-pooled to a league prior | Per-day snapshot fit uses only rows with `snapshot_date <` the target's; target's own day always excluded | No — correct and independently tested, no production caller yet |
| `qb` | Starter-probability-weighted QB priors (EPA/dropback, CPOE, sack/INT rate, etc.), recency-weighted | Same day-floor `snapshot_date <` rule as `opponent_adjusted` | No — same gap |
| `elo` | Pregame Elo rating, implied win probability, expected home margin, with HFA/travel/rest/QB adjustments | `LegacyElo.predict()` emitted before `LegacyElo.update()` is called with that game's own score | No — new in Fix 2, not yet wired to any live-scoring path |
| `market_history` | Each team's own rolling ATS-cover / over-under rate from its prior games (distinct from the current game's own market line) | Reuses the `epa` family's `shift(1)`-before-rolling primitive unmodified | No — new in Fix 2, same gap as `elo` |

Two of these families (`elo`, `market_history`) did not exist before Fix 2;
both are new adapter modules that reuse existing math
(`nfl_hybrid.legacy.elo.LegacyElo`, `build_team_pregame_features`) rather than
introducing new modeling logic of their own.

## 5. State lineage — **CURRENT / MERGED**

`src/nfl_hybrid/features/state_lineage.py` provides the two lineage
primitives used by every family (§3) and the invariant checker
(`verify_pregame_state_lineage`) the orchestrator runs before returning. A
worked example of the full lineage output is generated by
`scripts/generate_pregame_state_lineage_example.py` into
`outputs/pregame_state_lineage_example.json`.

## 6. Targets vs. postgame state vs. lagged pregame state — **CURRENT / MERGED**

These three are kept structurally distinct and must not be conflated:

- **Targets** — the outcome being predicted (home margin, total points, and
  the derived nullable ML/ATS/total binary labels via
  `nfl_hybrid.labels.edge_to_nullable_binary`, which excludes exact ties/
  pushes as `pd.NA` rather than mislabelling them class-zero).
- **Postgame state** — per-team-game facts computed *from* a completed
  game's own play-by-play/box score (`aggregate_advanced_team_game` in
  `src/nfl_hybrid/features/pbp_advanced.py`). This is not itself a leakage
  risk; it becomes one only if a row's own postgame facts feed that same
  row's pregame features without the lag rule in §4.
- **Lagged pregame state** — what a state family actually emits: postgame
  facts from *prior* games only, shifted or snapshot-cut per family, before
  the current row's own outcome is known. This is what §3–§4 build.

## 7. Existing model / evaluation components

**CURRENT / MERGED:**
- `src/nfl_hybrid/modern/joint_score.py` — direct joint margin/total model.
- `src/nfl_hybrid/modern/market_residual.py` — market-residual joint
  margin/total challenger.
- `src/nfl_hybrid/modern/inplay.py` — provider-agnostic in-play
  remaining-score framework (implemented; not pretrainable from the supplied
  workbooks alone — see `MODEL_ARCHITECTURE.md` §Legacy below).
- `src/nfl_hybrid/calibration/three_way.py`, `calibration/adoption.py` —
  separate Moneyline/ATS/totals calibration heads and the recalibration-
  adoption gate.
- `src/nfl_hybrid/evaluation/market_relative.py`, `evaluation/metrics.py`,
  `evaluation/walkforward.py`, `evaluation/week1_reliability.py` —
  market-relative scorecards, walk-forward harness, Week 1 reliability
  evaluation.
- `src/nfl_hybrid/staking/kelly.py` — penalized-Kelly staking + abstention
  gate.
- `src/nfl_hybrid/monitoring/calibration_drift.py` — rolling ECE + CUSUM +
  recalibration trigger.
- `src/nfl_hybrid/governance/readiness.py`, `governance/abstention.py`,
  `governance/lab_record.py`, `governance/contracts.py` — readiness
  classification, abstention policy, and the lab-record/contract discipline
  the model-selection tournaments run under.
- `src/nfl_hybrid/selection/` — the tournament/backtest harnesses
  (`epa_tournament.py`, `compact_tournament.py`, `distributional_tournament.py`,
  `unified_development_tournament.py`, `confirmation_2024.py`,
  `final_test_2025.py`, `chronological_spreadsheet_backtests.py`) used to run
  and freeze the model-selection decisions recorded under
  `docs/model-selection/`.
- `src/nfl_hybrid/pricing/betting_card.py`, `pricing/predict_week.py` — the
  weekly card builder and CLI.
- `src/nfl_hybrid/markets/quote_freshness.py` — quote-age/staleness policy
  for card audit fields.

**LEGACY (present, still used as a challenger baseline — not the sole or
current authoritative historical-state system):**
- `src/nfl_hybrid/legacy/` and `src/nfl_hybrid/spreadsheet_baselines/` —
  the reconstructed workbook constants and formulas (adjusted Elo, home-
  field/travel/bye/rest/QB adjustments, Elo-to-margin conversion,
  margin-of-victory Elo update, QB value, multiplicative offense/defense
  scoring, 70/30 H2H total blend, wind factor, American-odds conversion and
  de-vigging, the 32-model FiveThirtyEight/Yahoo/Vegas ensemble). Known
  workbook defects are corrected rather than copied. The legacy sequential
  feature builder traverses games chronologically and emits each row's
  features before updating Elo/scoring/H2H/rolling form with that row's own
  final score — this predates Fix 2 and is a separate, narrower leakage-safe
  builder than the §3 authoritative interface. It remains a challenger input
  (visible as model features), not a replacement for the spreadsheets, and
  is **not** the same thing as the Fix 2 authoritative pregame-state
  interface — do not conflate the two when describing "the" chronological
  feature system.

## 8. Current remediation boundary — **PLANNED / UNDER REMEDIATION**

Not on `main` as of 2026-08-18:

- **Fix 3 — chronological out-of-fold (OOF) evaluation + uncertainty.** In
  progress on branch `fix/chronological-oof-uncertainty`. Do not cite any
  OOF membership proof, uncertainty estimate, or chronological-OOF result
  from that branch as current system behavior.
- **Chronological calibration** built on top of Fix 3 — not started; blocked
  on Fix 3 merging.
- **Learned-engine production promotion** — the state families in §4 are not
  yet the production weekly pricing path; the weekly card prices at the
  legacy/market-baseline path (`OPERATOR_MANUAL.md`).
- **BDL live-game/box/PBP integration** — not started.
- **Twice-weekly learned model refitting** — not started.

See `REMEDIATION_STATUS_2026.md` for the complete roadmap with evidence
links, and `STATUS.md` for the current per-component status table.

## 9. Validation methodology — **CURRENT / MERGED**

Always expanding season/week walk-forward; never a random split. Primary
metrics: Brier score, log loss, and calibration for Moneyline/ATS/totals;
MAE/RMSE for margin and total; and for the betting layer, expected value at
the actual timestamped price, closing-line value, and bootstrap-CI ROI with
pushes handled via the nullable tie/push policy (§6).
