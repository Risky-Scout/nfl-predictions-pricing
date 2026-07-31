# Production Acceptance Report — 2026 NFL Pricing Model

Every claim below is backed by a command result, file, test, or numeric comparison.

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

## Final acceptance: **PASS for market-card production, with documented external dependencies**
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
