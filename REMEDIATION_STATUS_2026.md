# Remediation Status — 2026

Single source of truth for the 2026 remediation effort while the rebuild is
in progress. Supersedes narrative status claims in older reports for
anything listed here — if a historical report (`FINAL_REPORT.md`,
`CALIBRATION_REPORT.md`, etc.) disagrees with this table, this table is
current and the historical report is a frozen record of a prior build state.
See `STATUS.md` for the general per-component status and `README.md` for the
architecture overview.

Statuses used: `MERGED`, `IN PROGRESS`, `PENDING`, `HISTORICAL`.

Fix 1.5, Fix 2, Fix 3, Fix 3.1, and Fix 4 are `MERGED` as of 2026-08-21, based
on the current `main` branch history (`git log main`). Fix 3's own OOF
evidence was found contaminated by target-feature leakage on 2026-08-20
during Fix 4 hardening work and was corrected on `fix/oof-feature-leakage`
(Fix 3.1, PR #17) — see that row below. Fix 4 (ATS/TOTAL chronological
calibration hardening), re-certified against the corrected Fix 3.1 OOF
ledger on `fix/chronological-calibration-v2`, merged to `main` via PR #18
(merge commit `d3759fd`). Fix 5 (BDL 2026 live-data integration + parity
contract) is `IN PROGRESS` on `fix/bdl-2026-live-integration` — see that row
below.

| Work item | Status | Evidence / PR or commit | What it establishes |
|---|---|---|---|
| Data forensic audit | **HISTORICAL** | prior audit findings (see PR #13/#14 descriptions for the defects found) | Identified the leakage and data-resolution defects Fix 1.5–3 remediate |
| Fix 1.5 — central data resolution | **MERGED** | PR #13, commit `e68a162`, merge `92ecb45` | `NFL_MODEL_DATA_ROOT`-based registry (`src/nfl_hybrid/data/external_data.py`); every producer/consumer resolves the same dataset keys, no hard-coded machine paths |
| Fix 2 — chronological pregame state | **MERGED** | PR #14, commit `4d6d04d`, merge `45cf2c4` | Authoritative chronological pregame-state interface (`src/nfl_hybrid/features/pregame_state.py`), per-family lineage, cross-family banned-column check, `docs/PREGAME_STATE_DICTIONARY.yaml`. Underlying shifted pregame-state chronology remains valid, subject to Fix 3.1's carrier/pivot remediation below (Fix 2's own state tables were never contaminated -- the leak was introduced one stage downstream, in the game-level pivot) |
| Fix 3 — chronological OOF + uncertainty | **MERGED BUT SUPERSEDED BY FIX 3.1 LEAKAGE REMEDIATION** | PR #16, commit `d41c38a`, merge `4110eb8` | Established the expanding-window chronological OOF prediction/uncertainty engine (`src/nfl_hybrid/evaluation/chronological_oof.py`) and its chronology invariant (`result_available_at_utc < target_cutoff_utc`), which remains correct. However, its `build_oof_feature_matrix` swept every numeric `home_*`/`away_*` column off the game-level pivot, including native `games`-table passthrough columns (`home_score`, `away_score`, `home_moneyline_reference`, `away_moneyline_reference`, `home_spread_reference`, `home_spread_price_reference`, `away_spread_price_reference` -- 35 leaked columns across 5 state families; confirmed example: `epa__home_score` exactly equalled the target game's own final score). This invalidates the Fix 3 OOF predictions, residuals, and uncertainty estimates persisted under `NFL_MODEL_ARTIFACT_ROOT/chronological-oof-2020-2025/` (preserved, quarantined at `NFL_MODEL_ARTIFACT_ROOT/invalidated/fix3-target-feature-leakage-2026-08-20/`) and any evidence derived from them. Do not cite Fix 3's OOF/uncertainty output as current system behavior until Fix 3.1 lands. |
| Fix 3.1 — OOF feature-provenance leakage remediation | **MERGED** | PR #17, merge `0184d93` (commit `d3cf41a`), branch `fix/oof-feature-leakage` | Repairs `build_oof_feature_matrix` to a positive-provenance allowlist (`declared_state_family_columns`, `build_game_pregame_matrix(..., carrier_columns=())`) instead of a `home_*`/`away_*` name sweep, so a native `games` column can never again become a model feature merely by name. Regenerated a corrected proof-scale (season 2023 only, 3,080 features) OOF ledger under the same artifact root (`NFL_MODEL_ARTIFACT_ROOT/chronological-oof-2020-2025/`, SHA256 `06744f66...`/`aa3d1111...`/`6551b34a...`). See `outputs/fix3_1_oof_feature_leakage_proof.json` for the full leak/repair proof. Fix 4 (ATS/TOTAL hardening) has been re-certified against this corrected ledger — see that row below. |
| Chronological calibration | **MERGED** | PR #18, merge `d3759fd` | Proof-scale ATS and TOTAL chronological calibration engine (conditional + push-mass) exists and is certified against the corrected Fix 3.1 OOF ledger (see Fix 4 row); still not TUE/FRI production calibration — production orchestration ("Tuesday/Friday orchestration" row below) has not started |
| Fix 4 — ATS/TOTAL hardening | **MERGED** | PR #18, merge `d3759fd`, `outputs/real_ats_total_chronological_calibration_proof.json` | Re-certified end-to-end against the corrected Fix 3.1 OOF ledger (hash-verified read-only: Fix 3.1 artifact SHA256 unchanged before/after this certification run). Real closing_t10 ATS and TOTAL market lines attached per row (>=3 eligible books, <=5min snapshot lag, snapshot returned at/before each row's own `target_cutoff_utc`); both markets independently reach a `CALIBRATED_CERTIFIED` target with `calibration_sample_count >= 100`; zero suspicious (target-outcome or target-current-market) features found in the corrected 3,080-feature manifest. Proof-scale only (`artifact_scope=FIX4_PROOF`, season 2023, `production_evidence=false`) — not production evidence, not a frozen model spec. |
| Fix 5 — BDL 2026 live-data integration + historical/live parity contract | **IN PROGRESS** | branch `fix/bdl-2026-live-integration`, `docs/BDL_LIVE_INTEGRATION.md`, `docs/BDL_PARITY_MATRIX.md`, `outputs/bdl_overlap_backtest_2025.json` | Provider-neutral BDL client, canonicalization (games/team_stats/player_stats/injuries/roster/plays), strict `status_state=="final"` finality rule, FOUR independent family-specific first-usable timestamps (`score_available_at_utc`/`box_available_at_utc`/`player_stats_available_at_utc`/`pbp_available_at_utc` via `finality.compute_family_available_at_utc`, each immutable/monotonic and proven incapable of PBP-completing-late contaminating an earlier Friday cutoff), per-family completeness gates (`can_update_score_state`/`can_update_box_state`/`can_update_player_stats_state`/`can_update_pbp_state`), a fail-closed `season_type_hint` requirement (`canonical.normalize_games` raises rather than defaulting non-postseason rows to `"REG"`), immutable raw-snapshot provenance (`NFL_LIVE_DATA_ROOT`), and a 12-family historical/live parity matrix backed by an ACTUALLY RUN 2025 overlap backtest (10-game stratified sample, 2026-08-22: game identity/score 10/10 exact; 15/16 team-box fields 100% exact incl. a corrected sign-convention bug; player/QB box 100% exact on all compared counting fields with 95.8%/100% identity coverage; PBP ingestion clean but BDL play-count ~10/game higher than nflverse's; a genuine gap found between nflverse's own wired PBP-mask attempt counts and its published box score, independent of BDL) — only `game_result` and `elo_inputs` are `EXACT`/production-eligible; every EPA/success/CPOE-dependent family remains `UNAVAILABLE`, and team-box/QB-box/turnovers/play-volume/injuries remain `APPROXIMATE_NOT_APPROVED` with concrete measured evidence (not "no overlap run") explaining why. Not wired into any production path; no feature selection performed. |
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
merges to `main`. Fix 3.1 merged via PR #17 on 2026-08-20; Fix 4 (ATS/TOTAL
chronological calibration) merged via PR #18 on 2026-08-21 (merge commit
`d3759fd`). Fix 5 (BDL 2026 live integration) is in progress on
`fix/bdl-2026-live-integration`; when it merges, update its row to
`MERGED`, add the PR reference, and re-check whether "Tuesday/Friday
orchestration" can move off `PENDING` (it still depends on Fix 5's
finality/completeness gates being wired into an actual scheduler, which
Fix 5 deliberately does not do).
