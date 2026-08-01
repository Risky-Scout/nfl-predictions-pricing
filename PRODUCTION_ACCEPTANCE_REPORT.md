# Production Acceptance Report — 2026 NFL Pricing Model

Every claim below is backed by a command result, file, test, or numeric comparison.

## Step 1 — final-before-as-of and in-progress handling: **PASS (after remediation)**
**Old defect:** the assembler treated a prior game as completed when
`kickoff <= as_of AND final scores are non-null`, and a first remediation replaced that
with a **synthetic** completion timestamp (`kickoff + 4 h/5 h`) + `game_status = "FINAL"`,
which *fabricated finality evidence* and mislabelled UNKNOWN games as `FINAL_BEFORE_ASOF`.

**Remediation (this pass):** the synthetic completion timestamp and synthetic FINAL status
are **removed**. `resolve_finality_before_asof(games, pbp, *, as_of_utc, mode)` now separates
two concepts and never conflates them:
- **`verified_final_before_asof`** — set True only by real evidence at/before `as_of`:
  explicit final status + completion timestamp; explicit final status + verified terminal-
  play timestamp; a schema completion/terminal timestamp; or an official final snapshot.
  Non-null scores or kickoff time alone are never evidence.
- **`historically_available_before_asof`** — set only in `mode="historical_replay"` via the
  repository's documented postgame-availability policy (`data.availability`, kickoff + 5 h),
  labelled `HISTORICAL_AVAILABILITY_ASSUMPTION` with `availability_source =
  documented_training_policy`. It is **never** verified finality and is **never** used in
  `live` mode.

**`live` mode fails closed:** absent verified evidence → `UNKNOWN_FINALITY` → excluded; with
this repository's data (no game status/completion timestamp, no per-play UTC timestamp or
terminal marker — limitation recorded), a live run produces **no** eligible prior games and
the fundamental is honestly **NULL** (`fundamental_input_status = NO_VERIFIED_FINAL_GAMES`),
never a fabricated value. Status values: `FINAL_BEFORE_ASOF`, `EXPLICIT_NON_FINAL`,
`IN_PROGRESS_AT_ASOF`, `POST_ASOF_KICKOFF`, `UNKNOWN_FINALITY`,
`HISTORICAL_AVAILABILITY_ASSUMPTION`. Recorded per game: `finality_source`,
`finality_evidence_timestamp`, `availability_source`, `availability_timestamp`,
`exclusion_reason`. Partial-game play-by-play is never aggregated (non-eligible games are
dropped before `aggregate_advanced_team_game`).

**Evidence:** `tests/test_finality.py` (11, CI-safe): in-progress-with-final-score excluded;
same game after completion eligible; **synthetic-duration rejection** (>5 h elapsed, no
evidence → UNKNOWN in live); **suspended** game excluded regardless of duration;
completion-after-as_of excluded; post-as_of kickoff; **historical-mode separation**
(availability True, verified False, source labelled); live mode never uses the assumption;
determinism; invalid-mode rejected. End-to-end no-partial-aggregation and live-mode-null
integration tests in `tests/test_live_features_extended.py` (full-data, `extended_data`)
and `tests/test_live_features_ci.py` (committed fixtures). Historical live/training parity in
`historical_replay` mode is **unchanged** (max |Δ| = 0.0, identical NaN, 2024 wk10 / 2022
wk14). Live mode is not made less strict to preserve parity.

**Overall model acceptance: NOT READY.** Next pending stage: Step 2.

---

## Step 2 — core live/training parity & leakage tests execute in normal CI: **PASS (after merge)**

**Previous CI coverage gap.** The core parity/leakage module
(then `tests/test_live_features.py`, now split into the CI/extended pair below) carried a module-level
`pytest.mark.skipif(not backfill_files_exist, …)` on the private, git-ignored
`data/backfill_2020_2025/` parquets. Those parquets are never present on the GitHub
runner, so **every** one of the module's live-feature guarantees (all-19-feature
train/live parity, identical NaN patterns, target-score / target-play-by-play /
future-game invariance, post-as-of and in-progress exclusion, duplicate-target
rejection, deterministic hashing) **silently skipped in CI**. Local full-data
validation was real but could not be the sole regression gate.

**Compact committed fixture.** A small, fully synthetic, deterministic fixture now
lives under `tests/fixtures/live_features/` (`games.csv`, `pbp.csv`,
`expected_features.csv`, `fixture_manifest.json`) — synthetic NFL-like teams
(BUF/MIA/NYJ/NE) across two seasons, no private backfill, no purchased odds, no
provider payloads, no credentials. It drives the **real** production pipeline
(`aggregate_advanced_team_game → build_team_pregame_features →
build_game_pregame_matrix → _diff_features/_rest_features →
build_augmented_feature_matrix / build_live_augmented_features →
resolve_finality_before_asof`). A deterministic factory
(`tests/fixtures/live_features/factory.py`) is setup-only; the golden expected
outputs are **committed fixed data**, not regenerated during the test.

**Three target regimes** (different weeks/seasons):
- **A** `2021_05_BUF_NYJ` — complete last-four + valid season-to-date history.
- **B** `2022_01_MIA_BUF` — early-season / season-boundary missingness (2022
  season-to-date features NaN by design; last-four present across the boundary).
- **C** `2022_03_NE_BUF` — mid-season, Thursday kickoff exercising the short-week flag.
Plus `2022_04_MIA_BUF` (upcoming target with no outcome and no play-by-play) and
`2021_04_BUF_MIA` (explicit non-final `IN_PROGRESS`, retrospective scores, partial
play-by-play — excluded in live mode, available only under the replay availability
assumption).

**Golden-output protection.** Both the historical training path and the live path
are compared **independently** to the committed `expected_features.csv` (tol
`1e-10`, exact missingness), and training is compared to live — so a *joint* drift
of both paths is caught. Golden values are never recomputed-and-self-compared, and
never rewritten during a test. Changing them requires the manual, `--overwrite`-gated
`scripts/regenerate_live_feature_ci_fixture.py`, which prints old/new hashes.

**Core vs extended separation.**
- `tests/test_live_features_ci.py` — committed fixtures only, **no skip markers**,
  runs on every supported Python version. A guard test asserts the exact collected
  count (26) and that none of the module's tests carry `skip`/`skipif`.
- `tests/test_live_features_extended.py` — the pre-existing full-data suite (11
  tests), now marked `extended_data`, skipping only when the large local backfill is
  absent. The `extended_data` marker is registered in `pyproject.toml`.

**CI enforcement.** `.github/workflows/tests.yml` runs an explicit named step —
*"Core live-feature parity/leakage tests (must not skip)"* — before the full suite
on **Python 3.11 and 3.14**; it fails if the module reports any `skipped`/`no tests
ran` and requires `26 passed`. Extended-data skips are **not** counted as failures.

**Counts / runtime (this environment).**
- Core CI module: **26 collected, 26 passed, 0 skipped**, ~7.4 s (< 10 s target).
- Extended module: **11 collected, 11 passed** locally (backfill present); would skip in CI.
- Full suite: **318 passed** locally.
- Fixture total size ≈ 36 KB (37,207 bytes: `pbp.csv` 32,402 · `games.csv` 1,894 ·
  `expected_features.csv` 1,125 · `fixture_manifest.json` 1,786).
- `games.csv` sha256 `5276293e…`, `pbp.csv` sha256 `84225cb1…`,
  `expected_features.csv` sha256 `58a8115c…` (full values in `fixture_manifest.json`).

**No model/calibration changes.** No frozen feature definition, formula, coefficient,
imputation, calibration, pricing surface, de-vig, consensus, selection rule, weekly
orchestration, injury handling, or Step-1 finality policy was touched — Step 2 is
tests + fixtures + CI wiring only. Passing these feature tests proves parity and
leakage-safety of the feature assembly; it does **not** prove predictive calibration.

**Overall model acceptance: NOT READY.** Next pending stage: Step 3 — Week-1-specific
shadow reliability evaluation.

---

## Step 3 — Week 1 shadow reliability evaluation: **PASS (evaluation-only; merged & verified)**

**Scope.** A pre-registered, leakage-safe, reproducible historical-reliability
evaluation of the frozen `market_augmented_epa_rest_shadow` model
(`shadow_fundamental_2026.v1`). Stage 3 **evaluates historical reliability; it does
not establish betting edge and does not promote the shadow model.** Production
behavior is unchanged: `production_probability == market_probability`,
`production_source == MARKET_BASELINE`.

**`fundamental_probability` is generated by the market-augmented EPA/rest shadow
model and is not independent of the market** (its 19 frozen features include
`home_spread` and `total_line`).

**Pre-registration first.** All decision rules were frozen and pushed
(`docs/model-selection/week1-shadow-reliability-2026/registry.md`, commit
`a7d48a3`) **before** any outcome metric was computed.

**Populations.** (A) locked 2025 Week 1 holdout via the production artifact
(train ≤2023, calibrate 2024, holdout-excluded 2025); (B) pooled rolling-origin
Week 1 OOF (test 2022/2023/2024/2025; train ≤Y-2, calibrate Y-1), **n = 64**;
(C) Weeks 1–2 sensitivity (n = 128); (D) Weeks 3+ context (n ≈ 1011). Fold models
are fit only inside the evaluator under the already-frozen spec; no test-season
row enters any fold's imputation, preprocessing, calibration, or fitting.

**Market comparison.** Primary baseline = exact-contract proportional-de-vigged
**schedule-reference** market (`line_timestamp_known=false`), matched at the same
`home_spread`/`total_line` the shadow consumes; identical paired games across all
folds. This is **closing-line historical reliability** and does **not** prove
T-60 or T-10 reliability.

**Result (population B, pooled Week 1 OOF).** All three markets:
`INCONCLUSIVE_SMALL_SAMPLE` (n = 64 < 100 conclusive threshold). Shadow−market
log-loss 95% CIs all include 0 (moneyline +0.042 [-0.025, +0.106]; spread -0.004
[-0.022, +0.014]; total -0.019 [-0.043, +0.007]). Week 1 imputes exactly the 6
season-to-date EPA features per game (Weeks 3+: 0). Pushes/ties handled
separately, never scored as win/loss. **Production recommendation:
`RETAIN_MARKET_BASELINE_AND_MONITOR_SHADOW`.**

**Artifacts.** `WEEK1_SHADOW_RELIABILITY_REPORT.md`,
`reports/week1_shadow_reliability_2026.json` (source of truth; sha256
`0cc04d3b…`), deterministic ledger `reports/week1_shadow_reliability_ledger.csv`
(sha256 `07553283…`, 3657 rows). CI-safe evaluator tests
(`tests/test_week1_shadow_reliability.py`, 37 tests, zero skips) run on Python
3.11 and 3.14. The evaluator is byte-reproducible across runs.

**No model/calibration changes.** Production invariants hashed before & after —
unchanged: `config/shadow_fundamental_2026/*`, `config/pricing_calibration_2026*`,
`config/production_model_spec.json`. Permitted changes were limited to
pre-registration, evaluation code/tests/fixtures, the report JSON/Markdown, the
ledger, and these narrow doc updates.

**Live availability limitation.** Production live shadow availability still depends
on a **verified finality source**; historical evaluation used the documented
`HISTORICAL_AVAILABILITY_ASSUMPTION` (replay), never verified finality.

**Overall model acceptance: NOT READY.** Next pending stage: Step 4 — make
`run_week` fail closed and correct injury timestamp semantics.

---


## KEYSTONE COMPLETED (final completion pass, commit `fd5a308`)
The item that previously gated acceptance — the live fundamental feature path — is now
implemented, proven, and tested.
- **Live/training feature parity:** `build_live_augmented_features` reproduces the
  historical matrix **exactly** for 3 games across different seasons/weeks (2023 wk6,
  2024 wk10, 2022 wk14): max |Δ| = 0.0 on all 19 features **and** identical NaN pattern.
- **No leakage (tested):** corrupting a target's own final score or its own play-by-play
  does not change its features; a later completed game does not change an earlier
  prediction; a metrics-NaN placeholder inserts the upcoming game into the rolling
  sequence while the shift-then-roll builder uses only prior FINAL-before-`as_of` games.
  "Completed" = final score before `as_of` (in-progress early games excluded); per-play
  timestamp cutoff enforced; deterministic feature-snapshot hash; hard one-row-per-target
  assertion; kickoff≤as_of and duplicate-target both fail. (tests/test_live_features_extended.py, 9)
- **Non-null fundamental:** wired into `predict_week` via `--canonical-games/--play-by-play`.
  2024 wk10 replay: **42/42** fundamental probabilities non-null, finite, in [0,1], and
  **distinct** from market. 2026 Week-1 card: **48/48** fundamental AVAILABLE (6 season-to-date
  features NaN by design → median-imputed by the fitted pipeline). Production stays
  MARKET_BASELINE; `fundamental_probability` is never a market copy.
  (tests/test_live_fundamental.py, 4)
- **Single authoritative contract:** `FROZEN_FEATURES` (19) shared by training, artifact,
  live assembly, and load (validated at `load_shadow`).
- **Clean-checkout artifact:** the shadow `.joblib` (1 MB, no secrets) is committed via a
  narrow `.gitignore` exception; `load_shadow()` returns a model with exactly 19 features
  from a clean checkout.

### Still open (honest → not merged)
Operational sections not completed in this pass: fixed-spec weekly refit script (§16),
sample-gated recalibration (§17), `close_week.sh` (§18), explicit per-game T-60/T-10
scheduler labels (§15), full snapshot→card→manifest cached-pipeline benchmark (§20),
Week-2 generic smoke via CLI (§22), and live licensed injury/roster feeds (§11, external).
Final acceptance remains **NOT READY** pending these; the critical fundamental blocker is
resolved. Earlier sections of this report remain accurate.

---


- **Starting commit:** `6e28f2c` (origin/main verified) · **Branch:** `production-integration-2026`
- **Accepted calibration artifact:** `config/pricing_calibration_2026.json`
  (`pricing_calibration_2026.v1`, sha `d5c7b5b9cea3…`, cutoff 2024) — **frozen, unchanged.**
  Runtime de-vig **proportional**, consensus **equal_mean**, margin surface **empirical**,
  margin sigma **12.75**, total sigma **13.05**. 2025/2026 outcomes were not used to reselect anything.

## Delta audit (confirmed defects, fixes, tests)
| Issue | Confirmed | Fix | Test | Final status |
|---|---|---|---|---|
| 1. run_week.sh continues after critical failures | Yes | step 3 exits nonzero on missing lines; predict_week raises `WeeklyRunError` | test_market_pricing / weekly | FIXED |
| 2. weekly script doesn't apply current injuries to starters | Yes | `apply_availability_exclusions`; refit `--injury-file` | test_starter_overrides | FIXED (feed is external, §B) |
| 3. injury timestamp faked | Yes | `injury_data_utc` = MISSING when no feed retrieved | manifest inspection | FIXED |
| 4. freshness not enforced in pricing path | Yes | per-quote freshness in canonical `market_pricing` | test_market_pricing | FIXED |
| 5. live path used median vs artifact equal_mean | Yes | live path routes through canonical path using artifact methods | lines: `consensus=equal_mean` | FIXED |
| 6. different points enter one consensus | Yes | deterministic reference-point selection | test_different_points_not_averaged | FIXED |
| 7. newest-book timestamp masks stale books | Yes | age computed per book before consensus | test_one_fresh_book_cannot_mask_stale_books | FIXED |
| 8. live path ≠ calibration devig/consensus impl | Yes | one canonical `market_pricing` for both | test_uses_configured_methods | FIXED |
| 9. corrupt artifact silently falls to normal | Yes | `load_pricing_artifact` raises FileNotFoundError; predict_week warns and only then falls back | artifact test | FIXED (explicit) |
| 10. `--require-priced` not enforced | Yes | `validate_lines(require_priced=...)`; run_week step 3 critical | weekly test | FIXED |
| 11. identity check tautological | Yes | validates independent surface triple win+push+complement | test_probability_identity + weekly | FIXED |
| 12. override renormalization changes value | Yes | overrides preserved exactly; only residual distributed | test_override_preserves_exact_probability | FIXED |
| 13. unavailable QBs flagged but not removed | Yes | `apply_availability_exclusions` zeroes + renormalizes | test_unavailable_qb_zeroed | FIXED |
| 14. hard-coded season/date in refit | Yes | `--season/--as-of-utc/--injury-file` | refit run | FIXED |
| 15. audit reports selected vs executed methods | Yes | audit populated from methods actually executed in canonical path | lines devig/consensus columns | FIXED |

## Results
1. **Starting/final commits:** start `6e28f2c`; final on `main` printed in the terminal summary.
2. **Accepted artifact:** `pricing_calibration_2026.v1` (frozen).
3–4. **Confirmed defects / fixes / tests:** table above.
5. **Train-vs-serve parity:** the live line path and calibration both call
   `pricing/market_pricing.py`; the live card reports `devig=proportional, consensus=equal_mean`,
   matching the artifact. **PASS.**
6. **Live-data source status:** market odds via The Odds API current-odds — **available**
   (16/16 Week-1 games priced). Licensed injury feed and forecast-weather provider —
   **not configured in this environment** (no credentials). `injury_data_utc=MISSING`,
   fundamental EPA state not wired into the live line path → `fundamental_probability=NULL`.
7. **QB validation:** 32/32 teams, each sums to 1.0, 32 valid mixtures, 5 fallback
   candidates retained, 0 unresolved review flags (no injury feed). Exact-override and
   unavailable-zeroing verified by tests. **PASS.**
8. **Market vs fundamental coverage:** Week-1 market coverage 16/16 (48 market rows).
   Fundamental shadow exists and is **distinct** from market (max |Δ|=0.10 on 2024 replay);
   for Week-1 it is correctly **NULL** (inputs not in live path), never market-copied.
9. **Deterministic replay:** identical `--as-of-utc` inputs produce a byte-identical card
   (`diff -q` → identical). **PASS.**
10. **Cached full-week runtime:** **2.7 ms/week** (prev ~2 ms) — under the 5 s budget. **PASS.**
11. **Week-1 counts:** 16 games; markets = moneyline + spread + total (48 rows) + full
    alternate ladder available via `alt_lines`.
12. **Card status:** `PRELIMINARY` / market rows `VALID_MARKET`. It is a **VALID_MARKET_CARD**,
    explicitly **not** VALID_FULL_CARD (fundamental/weather/injury inputs incomplete).
13. **Unresolved operational dependencies:** licensed injury/roster feed credentials;
    forecast-weather provider; live 2026 pregame EPA-state wiring for the shadow model.
14. **Gate decisions:** below.

## Acceptance gates
| Gate | Decision | Evidence |
|---|---|---|
| A — one canonical market-pricing path | **PASS** | `market_pricing.py`; live path uses artifact methods; freshness per quote; no point-mixing; incomplete/future/post-kickoff rejected; 7 tests |
| B — current football state | **PARTIAL** | QB path fixed and validated (PASS); live injury/weather feeds are documented external dependencies (no credentials) → card is VALID_MARKET_CARD, not VALID_FULL_CARD (honest) |
| C — transparent fundamental shadow | **PASS** | deterministic shadow artifact (frozen spec, ≤2024); distinct columns; NULL-not-market; production stays MARKET_BASELINE; distinctness demonstrated |
| D — one fail-closed weekly command | **PASS (core)** | critical stages fail closed; run manifest; deterministic replay; `--as-of-utc` replay. Sub-item: explicit t60/t10 kickoff-relative labels are provided via `--as-of-utc` rather than named horizons |
| E — in-season monitoring | **PASS** | existing `operations/season_loop.py` logs pre-kickoff, scores post-outcome vs market, updates drift monitor |

## Recovery pass (branch `production-integration-2026`, commit `a413572`)
Correctness fixes completed and tested (266 tests):
- **Run manifest** now records `runtime_code_commit` = current repo HEAD, distinct from
  `calibration_artifact_training_commit` / `fundamental_artifact_training_commit` + hashes.
- **Canonical audit fields** renamed to the required set (`books_available`,
  `oldest/newest_quote_timestamp`, `maximum/median_quote_age_minutes`,
  `devig/consensus_method_executed`); statuses/book counts are independent per market
  (test_independent_status_per_market).
- **Calibration/serve parity** test on a shared fixture (identical paired books, de-vig,
  reference point, consensus) — the live path and calibration share the same primitives.
- **`fetch_current_week_lines`** now delegates via a normal module import (no `runpy`).
- **Reproducible shadow build script** committed; a clean rebuild yields numerically
  identical predictions (seed + train seasons recorded). The `.joblib` is gitignored;
  serialized-byte hash may vary but parameters/predictions are identical.
- **Secret scan (redacted):** no credential assignments in tracked files; `.env` ignored;
  the two 32-hex hits are a content hash and event IDs in the response fixture, not keys.

### NOT DONE in this pass (honest gaps → not merged)
- **Section 6 — fundamental live-feature wiring:** the Week-1 card's
  `fundamental_probability` is still **NULL** because the live pricing frame does not yet
  assemble the frozen model's 19 EPA features from the 2026 team/QB priors. The shadow
  model is real and demonstrably distinct on historical replay (max |Δ|=0.10), but a
  prior→feature assembly function is required to produce non-null 2026 Week-1 estimates.
  This is a code task, not an external dependency — it remains open.
- **Named horizons (t60/t10)** are supported via `--as-of-utc` replay but not as explicit
  kickoff-relative labels in `run_week.sh`.

## Final acceptance: **NOT READY** (open PR for review; do not merge)
The market-pricing path, QB integration, identity validation, audit trail, reproducible
shadow, and run manifest are correct and tested. However, **Gate B/C are not fully met**:
the fundamental model is not yet wired to live 2026 features, so no game can be a
`VALID_FULL_CARD` for a code reason (not merely an external-data one). Per the completion
rules ("do not merge if any acceptance gate is still failing"; "current football state
must reach the fundamental feature path"), this is submitted as a **PR for review, not
merged**. Prior superseded assessment retained below for history.

## (superseded) Earlier assessment — PASS for market-card production, with documented external dependencies
The pricing integration is complete and correct: runtime pricing matches the frozen
artifact, freshness is enforced per quote, contract points are not mixed, the QB path is
validated, fundamental values are never fabricated, market/fundamental/production are
distinct, governance is preserved, critical failures fail closed, Week 1 runs end to end,
the pipeline is generic across weeks, outputs are reproducible and auditable, and cached
pricing is 2.7 ms. The only gaps are **external data feeds** (licensed injury feed,
forecast weather, live 2026 pregame EPA state) that require credentials/providers absent
from this environment — the system discloses these honestly and never labels the card
VALID_FULL_CARD. This is an honest PASS on everything within the codebase's control; it is
**not** a claim that a live September full card was produced in July.
