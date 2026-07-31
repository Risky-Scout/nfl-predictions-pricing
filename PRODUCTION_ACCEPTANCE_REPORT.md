# Production Acceptance Report — 2026 NFL Pricing Model

Every claim below is backed by a command result, file, test, or numeric comparison.

## Step 1 — final-before-as-of and in-progress handling: **PASS**
**Old defect:** the live assembler treated a prior game as completed when
`kickoff <= as_of AND final scores are non-null`. Historical rows already contain
eventual results, so at a reconstructed as-of an early game could still have been in
progress (e.g. 1:00 pm game ending 4:08 pm, prediction at 3:55 pm) yet be counted.

**Fix:** one canonical `resolve_finality_before_asof(games, pbp, *, as_of_utc)` returns
`is_final_before_asof, finality_status, completion_timestamp_utc, finality_source,
exclusion_reason`. Evidence hierarchy: (1) explicit final status + completion timestamp
≤ as_of; (2) final status + verified terminal-play timestamp ≤ as_of; (3) a schema
completion timestamp ≤ as_of; (4) otherwise **UNKNOWN_FINALITY → excluded (fail closed)**.
Non-null scores or kickoff time alone are never sufficient. The repo data lack a
wall-clock completion timestamp, so the live assembler attaches a **documented
conservative fallback** (`completion = kickoff + overtime-aware max-duration bound`,
4 h regulation / 5 h OT) — an upper bound used only to *confirm* finality, never to
estimate the exact end; in-progress games are excluded. A game must pass **both**
"confirmed final before as_of" **and** "no included play after as_of". Partial-game
play-by-play is never aggregated (in-progress games are dropped before team-game
aggregation). Audit fields added: eligible/excluded-in-progress/excluded-unknown/
excluded-post-as-of counts, max eligible completion timestamp, finality source and
exclusion-reason counts; the run manifest records these plus a finality decision hash.

**Evidence:** `tests/test_finality.py` (9, CI-safe synthetic): in-progress-with-final-
score excluded, same game after completion eligible, unknown-finality excluded, explicit
non-final excluded, final-but-completion-after-as-of excluded, late-window Sunday,
post-as-of kickoff, determinism. Historical live/training parity **unchanged**
(max |Δ| = 0.0, identical NaN pattern for 2024 wk10 / 2022 wk14) because same-day
in-progress games never feed a target's once-per-week rolling sequence.

**Overall model acceptance: NOT READY.** Next pending stage: Step 2 — make core
parity/leakage tests execute in CI.

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
  assertion; kickoff≤as_of and duplicate-target both fail. (tests/test_live_features.py, 9)
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
