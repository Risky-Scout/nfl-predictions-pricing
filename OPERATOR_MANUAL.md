# OPERATOR MANUAL — NFL Pricing Engine, 2026 Season

For a non-author running the season. Read the box first.

> **State of the system (read this):** All three markets — moneyline, ATS, totals —
> open the 2026 season at **`RETAIN_BASELINE`**. The pre-registered EPA tournament
> beat the real closing market on **nothing** (0/15). **The card stakes ZERO.** It
> prices every game at the de-vigged market and prints `no-bet: edge not established`.
> Nothing stakes until the **live record** earns it through the gate in §6. This is
> by design, not a bug. A model that bets unproven markets is a donation machine.

---

## 0. One-time setup
```bash
cd nfl-predictions-pricing
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[data,dev]"
export PYTHONPATH="$PWD/src"
export SPREADSPOKE_CSV_PATH="$PWD/data/spreadspoke_enhanced.csv"
export THE_ODDS_API_KEY="<your key>"      # optional but needed for live lines/capture
.venv/bin/python -m pytest -q               # expect: all green
```
(On Python 3.14 the editable import hook can fail silently — use 3.11/3.12, which
this manual assumes. `PYTHONPATH=src` is the reliable path for scripts.)

---

## 1. The weekly sequence
Run **pre-kickoff** each week. One command does steps 1b–4:
```bash
scripts/run_week.sh 2026 <WEEK>
```
It performs, in order:

| Step | Command (what run_week.sh calls) | Cost | Output |
|---|---|---|---|
| 1. Refresh | `nfl-hybrid-backfill … open-sources` | 0 | updated `data/backfill_2020_2025/` |
| 1b. **Capture** | `scripts/capture_lines.py` | **3 credits** | immutable snapshot in `data/line_history/` |
| 2. Lines | `scripts/build_week1_2026_lines.py` | **3 credits** | `outputs/season_2026/lines_wkNN.csv` |
| 3. Price | `predict_week …` | 0 | `outputs/season_2026/card_wkNN.csv` |
| 4. Log | `log_week_predictions` | 0 | `outputs/season_2026/predictions_wkNN.csv` (immutable) |

**After the games resolve** (Monday/Tuesday), run scoring + gate (step 5–6, §5).

### Capture cadence (optional but recommended — builds your edge dataset for ~free)
Run `scripts/capture_lines.py --label <tag>` at four points each week:
**Tue open / Fri / Sun T-60 / Sun T-10**. Each call = **3 credits** for the whole
slate across all 3 markets. That is **12 credits/week**, **~216 credits/season**, for
complete opening-to-closing line movement on every game — the #1 ranked data
acquisition, self-built.

---

## 2. What each output means
- **`lines_wkNN.csv`** — one row per game. `status = PRICED` (posted lines found,
  `n_books` books) or `UNPRICED-AWAITING-LINES` (no lines yet — the game is *not*
  invented; it simply is not priced until lines post).
- **`card_wkNN.csv`** — one row per (game, market). Key columns:
  - `model_spread` / `predicted_home_margin` — the model's view (= market, in production).
  - `calibrated_probability` — win/cover/over probability.
  - `market_fair_probability` — de-vigged market probability.
  - `penalized_edge` — edge after the KL penalty. **0.0 in production by design.**
  - `recommended_stake` / `should_bet` — **0 / False** for every market right now.
  - `readiness_status` — `RETAIN_BASELINE` for all markets at season open.
  - `no_bet_reason` — `no-bet: edge not established`.
  - `calibration_statement` — the standing disclaimer (see §6).
- **`predictions_wkNN.csv`** — the immutable, UTC-timestamped pre-kickoff record.
  **Never edit it.** It is the ground truth for live validation.
- **`data/line_history/*.parquet`** — accumulating book-attributed, de-vigged snapshots.

### Alternate-line quotes (bookmaker surface)
To quote any spread/total ladder for a game, use `pricing.alt_lines`
(`price_spread_ladder` / `price_total_ladder`) — fair probability, fair decimal, and
offered decimal at a configurable hold. See `tests/test_alt_lines.py` for usage.

---

## 3. Credit costs per week
- Card generation: **3 credits** (one current-odds call for lines).
- Recommended capture cadence: **12 credits** (4 calls).
- **Total ≈ 15 credits/week** if you capture; **3/week** if you only price.
- Historical purchases are *not* needed for operations — the season builds its own
  movement dataset via capture.

---

## 4. Human-review points (do these, they are not automated)
1. **August QB battles.** `data/templates/starter_probabilities.csv` rows with
   `season = 2026` and `source = depth_chart_rank` carry `needs_human_review = True`.
   These are rank-based projections from the latest depth chart. Before Week 1,
   confirm each team's actual Week-1 starter and correct the probabilities, then
   re-run `scripts/refit_2026_priors.py`. A wrong starter is worth several points.
2. **Coordinators / play-callers.** Not in the data (external-only). If a team has a
   new OC/DC/play-caller, note it — the coach prior only captures head coaches.
3. **`UNPRICED-AWAITING-LINES` games.** Re-run the line fetch closer to kickoff; do
   not manufacture a number.

---

## 5. After games resolve — score + drift
```bash
# score the logged record against the market and update the cumulative scorecard
.venv/bin/python - <<'PY'
import pandas as pd
from nfl_hybrid.operations import score_resolved_week, cumulative_live_scorecard, live_promotion_decision
# load outputs/season_2026/predictions_wkNN.csv (logged) and a resolved-results frame
# (game_id, home_win, home_cover, over), then:
# weekly = score_resolved_week(logged, resolved)
# cum = cumulative_live_scorecard(all_weekly_scores)
# decision = live_promotion_decision(cum, drift_alarm_markets=<from drift monitor>)
PY
```
Update the **drift monitor** (`monitoring.calibration_drift.monitor_calibration_drift`)
on resolved games. A **drift alarm** on any market → that market is **demoted to
`RETAIN_BASELINE` immediately** (stakes zero), and recalibration is requested through
the calibration-adoption gate. See `scripts/dryrun_week_loop.py` for a full worked
example on 2025 week-1 data.

---

## 6. The fixed promotion / demotion rules (do not change mid-season)
A market is **PROVISIONAL** only if a candidate cleared the backtest gate — none did,
so all markets are `RETAIN_BASELINE` today. Even a PROVISIONAL market **stakes zero**
until the **live gate** passes:

**PROMOTE_TO_LIVE_STAKE** (then stakes 1/8 Kelly, 5% per-bet / 15% weekly caps) requires
**all** of:
- **≥ 8 live weeks** scored, AND
- **cumulative live log-loss gain vs market > 0**, AND
- **live pick-accuracy 95% CI lower bound > 0.524**.

**DEMOTE_TO_RETAIN_BASELINE** — immediate — on any **drift alarm**.

Backtests never grant live staking. Only the immutable live record can. If a season
of live evidence never clears the gate, the honest outcome is: keep pricing at the
market and stake nothing. That is a successful sharp book, not a failure.

---

## 7. If something breaks
- **Card won't generate:** check `lines_wkNN.csv` exists and has `PRICED` rows.
- **`FileExistsError` on logging:** the week's record already exists and is immutable
  by design — do not overwrite it.
- **Odds API error / no key:** operations continue on the CSV template; games show
  `UNPRICED-AWAITING-LINES`. No numbers are invented.
- **Tests red after an update:** stop and fix before trusting any output.
