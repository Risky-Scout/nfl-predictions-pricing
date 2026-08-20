# Remediation Status — 2026

Single source of truth for the 2026 remediation effort while the rebuild is
in progress. Supersedes narrative status claims in older reports for
anything listed here — if a historical report (`FINAL_REPORT.md`,
`CALIBRATION_REPORT.md`, etc.) disagrees with this table, this table is
current and the historical report is a frozen record of a prior build state.
See `STATUS.md` for the general per-component status and `README.md` for the
architecture overview.

Statuses used: `MERGED`, `IN PROGRESS`, `PENDING`, `HISTORICAL`.

Fix 1.5, Fix 2, and Fix 3 are `MERGED` as of 2026-08-20, based on the current
`main` branch history (`git log main`). Fix 3's own OOF evidence was found
contaminated by target-feature leakage on 2026-08-20 during Fix 4 hardening
work and is being corrected on `fix/oof-feature-leakage` (Fix 3.1) before
Fix 4 may proceed — see that row below.

| Work item | Status | Evidence / PR or commit | What it establishes |
|---|---|---|---|
| Data forensic audit | **HISTORICAL** | prior audit findings (see PR #13/#14 descriptions for the defects found) | Identified the leakage and data-resolution defects Fix 1.5–3 remediate |
| Fix 1.5 — central data resolution | **MERGED** | PR #13, commit `e68a162`, merge `92ecb45` | `NFL_MODEL_DATA_ROOT`-based registry (`src/nfl_hybrid/data/external_data.py`); every producer/consumer resolves the same dataset keys, no hard-coded machine paths |
| Fix 2 — chronological pregame state | **MERGED** | PR #14, commit `4d6d04d`, merge `45cf2c4` | Authoritative chronological pregame-state interface (`src/nfl_hybrid/features/pregame_state.py`), per-family lineage, cross-family banned-column check, `docs/PREGAME_STATE_DICTIONARY.yaml`. Underlying shifted pregame-state chronology remains valid, subject to Fix 3.1's carrier/pivot remediation below (Fix 2's own state tables were never contaminated -- the leak was introduced one stage downstream, in the game-level pivot) |
| Fix 3 — chronological OOF + uncertainty | **MERGED BUT SUPERSEDED BY FIX 3.1 LEAKAGE REMEDIATION** | PR #16, commit `d41c38a`, merge `4110eb8` | Established the expanding-window chronological OOF prediction/uncertainty engine (`src/nfl_hybrid/evaluation/chronological_oof.py`) and its chronology invariant (`result_available_at_utc < target_cutoff_utc`), which remains correct. However, its `build_oof_feature_matrix` swept every numeric `home_*`/`away_*` column off the game-level pivot, including native `games`-table passthrough columns (`home_score`, `away_score`, `home_moneyline_reference`, `away_moneyline_reference`, `home_spread_reference`, `home_spread_price_reference`, `away_spread_price_reference` -- 35 leaked columns across 5 state families; confirmed example: `epa__home_score` exactly equalled the target game's own final score). This invalidates the Fix 3 OOF predictions, residuals, and uncertainty estimates persisted under `NFL_MODEL_ARTIFACT_ROOT/chronological-oof-2020-2025/` (preserved, quarantined at `NFL_MODEL_ARTIFACT_ROOT/invalidated/fix3-target-feature-leakage-2026-08-20/`) and any evidence derived from them. Do not cite Fix 3's OOF/uncertainty output as current system behavior until Fix 3.1 lands. |
| Fix 3.1 — OOF feature-provenance leakage remediation | **READY FOR PR — pending final acceptance check** | branch `fix/oof-feature-leakage` | Repairs `build_oof_feature_matrix` to a positive-provenance allowlist (`declared_state_family_columns`, `build_game_pregame_matrix(..., carrier_columns=())`) instead of a `home_*`/`away_*` name sweep, so a native `games` column can never again become a model feature merely by name. Regenerates a corrected proof-scale (season 2023 only) OOF ledger under the same artifact root. See `outputs/fix3_1_oof_feature_leakage_proof.json` for the full leak/repair proof. Fix 4 (ATS/TOTAL hardening) is blocked until this merges. |
| Chronological calibration | **PENDING** | blocked on Fix 3.1 | Not started; any prior chronological-calibration work built on Fix 3's contaminated OOF evidence must be re-validated against the Fix 3.1 corrected ledger, not treated as current |
| Fix 4 — ATS/TOTAL hardening | **BLOCKED** | — | Blocked pending corrected Fix 3.1 OOF evidence; do not begin, and do not describe as complete, until Fix 3.1 merges and its acceptance conditions (temporal eligibility, feature provenance, target/future mutation invariance, prior-history positive control, absence of target/current-market leakage) all pass |
| BDL 2026 final-game/box/PBP integration | **PENDING** | — | Not started |
| Week 1 preseason prior | **PENDING** | — | Not started |
| Compact feature deduction | **PENDING** | — | Not started |
| Model-family / frozen-spec selection (remediated pipeline) | **PENDING** | — | Not started; the existing frozen spec (`docs/model-selection/final-2025/`) predates Fix 1.5–3 and is `HISTORICAL` |
| Learned-engine production promotion | **PENDING** | — | The state families from Fix 2 are not the production weekly pricing path; weekly card still prices at the legacy/market-baseline path |
| Tuesday/Friday orchestration | **PENDING** | — | Not started |
| Immutable forecast ledger | **PENDING** | — | Not started |
| Market-relative evaluation | **HISTORICAL** (existing) / **PENDING** (re-validation) | `src/nfl_hybrid/evaluation/market_relative.py`, `FINAL_REPORT.md` | Existing evaluation code is merged and used by the legacy path; re-validating it against the Fix 1.5–3 remediated pipeline has not started |
| Mike Shackleford audit package | **PENDING** | — | Not started; depends on the full roadmap above |

## Reading this table

- `MERGED` — the work item is implemented and present on `main` as of this
  document's date.
- `IN PROGRESS` — implementation exists on a feature branch but has not
  merged to `main`. Do not describe it as current functionality.
- `PENDING` — not started, or blocked on an earlier item in this table.
- `HISTORICAL` — a prior artifact, run, or result that is frozen in time
  (e.g. an audit finding, a completed tournament, an existing evaluation
  module used by the pre-remediation pipeline). Historical items are not
  rewritten here; they are labeled so a reader doesn't mistake them for
  current claims about the remediated pipeline.

## Keeping this current

Update this file's status column and evidence links as each roadmap item
merges to `main`. When Fix 3.1 merges, update its row to `MERGED`, add the PR
reference, move Fix 4 from `BLOCKED` to `PENDING`, and re-check whether
"Chronological calibration" can move from `PENDING` to `IN PROGRESS`.
