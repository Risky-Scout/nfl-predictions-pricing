# 2026 Production Runbook

Operational runbook for the certified `v2026.1-fix8-certified` football
model (ELO_ONLY, six frozen Elo features, RIDGE_ALPHA_100, card-scoped
TUE/FRI, Fix-8 ATS/TOTAL calibration) in live 2026 production.

This runbook covers **operations only**. For the model's statistical
strength, limitations, and what is/isn't demonstrated, see
[`MODEL_STRENGTH_AND_LIMITATIONS.md`](MODEL_STRENGTH_AND_LIMITATIONS.md).

## 1. Environment setup (no secret values here)

Set these environment variables (see `.env.example` for the full list; no
value is ever printed by any script in this pipeline):

| Variable | Purpose |
|---|---|
| `NFL_MODEL_DATA_ROOT` | Private historical data estate (nflverse backfill, Odds API history) |
| `NFL_MODEL_ARTIFACT_ROOT` | Derived pipeline output (certification evidence, production ledgers) |
| `NFL_LIVE_DATA_ROOT` | Immutable raw BallDontLie live-provider snapshot cache |
| `THE_ODDS_API_KEY` | Live/current market odds (The Odds API) |
| `BALLDONTLIE_API_KEY` | Live schedule/results (BallDontLie) |

Install the live-data optional dependency group (`nflreadpy`, required for
the live 2026 schedule):

```bash
pip install -e '.[data,dev]'
```

## 2. Preflight

```bash
python3 scripts/run_2026_production_card.py --preflight
```

Checks (SET/UNSET only, never a value): required env vars, the four
certified scientific hashes, Fix-8 calibration seed presence,
`backfill.games` schema resolution, output directory writability, current
git commit.

The result separates **infrastructure readiness** from **live-input
readiness**:

| Field | Meaning |
|---|---|
| `infra_ready` | Certified hashes match, calibration seed present, ledger dirs writable, git commit identified |
| `schedule_2026_available` | The games/schedule source actually contains season 2026 |
| `live_2026_market_source_registered` | A live 2026 market source is registered for `raw_market_reconstruction` (an `odds_history.2026` key) |
| `production_run_ready` | `true` **only** when `infra_ready` **and** both live inputs above are available |
| `blocking_problems` | Every unmet requirement, named -- infra blockers plus `schedule_2026_unavailable` / `live_2026_market_source_unregistered` |

`overall_status` is `READY` (all of the above), `BLOCKED_ON_LIVE_INPUTS`
(infra ready, a required live input missing), or `NOT_READY` (an infra
blocker). **`--preflight` exits 0 only when `production_run_ready` is
`true`**; `BLOCKED_ON_LIVE_INPUTS` exits non-zero so an operator or
scheduler cannot mistake it for readiness. There is no
`READY_WAITING_FOR_FIRST_DUE_CUTOFF` status -- a system missing a required
live input is blocked, not "waiting".

**A real run of this command in this environment currently reports
`BLOCKED_ON_LIVE_INPUTS`** (`production_run_ready: false`): infrastructure
is genuinely ready, but `blocking_problems` lists
`schedule_2026_unavailable` (the historical `backfill.games` source tops
out at season 2025; no live 2026 schedule adapter is wired into
`run_horizon_batch`) and `live_2026_market_source_unregistered` (no
`odds_history.2026` key is registered in
`nfl_hybrid.data.external_data`). `THE_ODDS_API_KEY` and
`NFL_LIVE_DATA_ROOT` are also unset here and will need to be set before
the first real live forecast. See
[`MODEL_STRENGTH_AND_LIMITATIONS.md`](MODEL_STRENGTH_AND_LIMITATIONS.md#production-readiness)
for the full status.

## 3. Manual TUE / FRI runs (testing and recovery only)

```bash
# Force a run regardless of the due window -- for historical/integration
# testing or manual recovery only. Requires real as-of data to exist; never
# fabricates a future market snapshot.
python3 scripts/run_2026_production_card.py --horizon TUE --as-of 2026-09-01T16:00:00Z
python3 scripts/run_2026_production_card.py --horizon FRI --as-of 2026-09-04T16:00:00Z
```

Omit `--as-of` to use the current UTC time (production default).

## 4. `--run-due` (the production mode)

```bash
python3 scripts/run_2026_production_card.py --run-due
```

Decides from local America/New_York time whether TUE or FRI is due (the
12:00-12:20 local window, DST-aware, on the correct weekday). Outside the
window: prints `NOT_DUE` and exits 0, no writes. This is the command the
scheduler (Section 6 below) invokes.

## 5. Result attachment

```bash
python3 scripts/run_2026_production_card.py --attach-results --result-file results.json
```

`results.json` is a JSON array of
`{"game_id", "horizon", "target_cutoff_utc", "result": {...}, "result_available_at_utc", "result_source_hash"}`.
Attaching a result **never mutates** the original forecast or evaluation
record -- it writes a separate, sibling `*.result.json` file. Rejected
(`IDENTIFIER_FAILURE`) if `result_available_at_utc` is not strictly before
the attachment run time, or if no matching forecast exists.

## 6. Prospective performance report

```bash
python3 scripts/report_2026_prospective_performance.py
```

Reads only immutable forecast + attached-result records. Reports
`INSUFFICIENT_PROSPECTIVE_SAMPLE` rather than a conclusion when there
aren't enough games yet. Never optimizes a threshold or mines a strategy.

## 7. Ledger paths

All under `$NFL_MODEL_ARTIFACT_ROOT/production-2026/`:

| Path | Contents | Mutability |
|---|---|---|
| `forecast-ledger/{horizon}/{game_id}__{cutoff}.json` | The forecast of record | Immutable: first write wins; identical replay is a no-op; a differing replay is a hard stop (`FORECAST_IMMUTABILITY_VIOLATION`) |
| `run-manifests/{run_id}.json` | One file per attempted batch | Append-only: a `run_id` may never be reused |
| `evaluation-ledger/{horizon}/{game_id}__{cutoff}.json` (+ `.result.json`) | Prospective shadow evidence | Forecast side immutable at write; result side attached separately, never edits the forecast |

## 8. Fail-closed status meanings

| Status | Meaning |
|---|---|
| `NOT_DUE` | Outside the TUE/FRI due window; correct, expected, no action needed |
| `SCHEDULE_UNAVAILABLE` | Live 2026 schedule could not be loaded (e.g. `nflreadpy` missing, or the provider call failed) |
| `GAME_RESULT_SOURCE_UNAVAILABLE` | Live result source unreachable |
| `ELO_SOURCE_UNAVAILABLE` | Historical Elo update-event source unreachable |
| `MARKET_SOURCE_UNAVAILABLE` | Live odds provider unreachable/unauthorized (e.g. missing `THE_ODDS_API_KEY`) |
| `MARKET_NOT_READY` | Market data present but insufficient (e.g. too few fresh coherent books) |
| `MODEL_NOT_READY` | Below `min_training_games` for this batch -- no fabricated prediction |
| `UNCERTAINTY_NOT_READY` | Insufficient expanding-window warmup for same-horizon uncertainty |
| `CALIBRATION_NOT_READY` | No production calibration seed available for this stream |
| `SCHEMA_DRIFT` | An upstream source's shape no longer matches what this pipeline expects |
| `IDENTIFIER_FAILURE` | A required identity (run_id, forecast identity, result timing) was invalid |
| `HASH_MISMATCH` | A certified scientific hash did not match live recomputation -- HARD STOP |
| `FORECAST_IMMUTABILITY_VIOLATION` | An attempted write conflicted with an existing, different forecast/result -- HARD STOP |

**None of these is ever silently substituted** with a market prediction in
place of a football prediction, a future result, stale data, a synthetic
-110 price, a later market snapshot, or an actual result before it was
available.

## 9. Recovering from a source outage

1. Do **not** retry into a synthetic substitute for the missing source.
2. Fix the source (install the dependency, set the credential, wait for
   the provider to recover).
3. Re-run `--preflight` until `READY`.
4. Re-run `--run-due` (or the specific `--horizon`, with the *original*
   `--as-of` if this is a delayed catch-up for a specific due window that
   already passed and its real as-of data now exist). The immutable ledger
   guarantees a delayed run cannot double-forecast a game already written.

## 10. How NOT to rewrite prior forecasts

- Never edit a file under `forecast-ledger/` or `evaluation-ledger/` by
  hand.
- Never delete a forecast file to "fix" a bad prediction -- if a genuine
  defect is found, fix the code, and let the (now-different) output hit
  the immutability gate on replay so the discrepancy is visible, rather
  than silently overwriting history.
- `--attach-results` is the only sanctioned way to add information to an
  existing forecast record, and it only ever adds a separate sibling file.

## 11. Scheduler

Two options are provided; see
[`MODEL_STRENGTH_AND_LIMITATIONS.md`](MODEL_STRENGTH_AND_LIMITATIONS.md#production-readiness)
for which is currently active (`OPERATOR_SCHEDULE_READY`, not yet
`AUTOMATED`, as of this certification).

### 11a. GitHub Actions (`.github/workflows/production-2026.yml`)

Targets a self-hosted runner label -- **not** `ubuntu-latest`, because
this pipeline needs the local private data estates and live credentials a
GitHub-hosted runner cannot reach. Triggers `workflow_dispatch` (manual)
plus two redundant `schedule:` crons at `16:05 UTC` and `17:05 UTC` on
Tuesdays/Fridays, bracketing America/New_York noon across both DST offsets
-- the non-matching UTC firing exits `NOT_DUE` with no writes, and the
immutable ledger makes a genuine double-firing an idempotent no-op.

**This requires an operator-provisioned self-hosted runner with the
workflow's target label, running as a persistent service.** A local
runner agent (`joseph-nfl-mac-stable`, this machine's
`/Users/josephshackelford/actions-runner-nfl-2.335.1/`) is registered but
(a) is not currently running as a service here (`ps aux | grep
Runner.Listener` returns nothing, no launchd agent is loaded), and (b) was
not registered with a label specific to this production workflow --
registering it generically (bare `self-hosted`) would make it pick up
*any* self-hosted-targeted workflow in this repo, which is not desired
for a job that writes to production ledgers. See
`.github/workflows/production-2026.yml`'s own header comment for the
exact target label and the promotion checklist. To activate this existing
agent (after confirming/adding the correct label via `./config.sh` or the
GitHub UI):

```bash
cd /Users/josephshackelford/actions-runner-nfl-2.335.1
./run.sh   # foreground; for a persistent service, install as a launchd
           # service per GitHub's ./svc.sh installer for this runner version
```

### 11b. launchd (operator fallback -- works without the GH Actions runner)

Create `~/Library/LaunchAgents/com.nfl-hybrid.production-2026.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.nfl-hybrid.production-2026</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/Users/josephshackelford/actions-runner-nfl-2.335.1/_work/nfl-predictions-pricing/nfl-predictions-pricing/scripts/run_2026_production_card.py</string>
    <string>--run-due</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <!-- Every hour at :05 on Tue/Fri; --run-due itself decides NOT_DUE
         outside the real 12:00-12:20 America/New_York window, so an
         hourly cadence is a safe, simple superset. -->
    <dict><key>Weekday</key><integer>2</integer><key>Minute</key><integer>5</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Minute</key><integer>5</integer></dict>
  </array>
  <key>StandardOutPath</key>
  <string>/Users/josephshackelford/Library/Logs/nfl-hybrid-production-2026.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/josephshackelford/Library/Logs/nfl-hybrid-production-2026.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>NFL_MODEL_DATA_ROOT</key><string>/path/to/your-nfl-data-estate</string>
    <key>NFL_MODEL_ARTIFACT_ROOT</key><string>/Users/josephshackelford/NFL-Model-Artifacts</string>
    <key>NFL_LIVE_DATA_ROOT</key><string>/path/to/your-live-data-cache</string>
  </dict>
</dict>
</plist>
```

(API keys are deliberately omitted from the plist's `EnvironmentVariables`
-- load them from the operator's own shell profile / keychain-backed
mechanism instead, never hardcode a credential into a checked-in or
world-readable file.)

```bash
launchctl load ~/Library/LaunchAgents/com.nfl-hybrid.production-2026.plist
```

### 11c. cron (alternative operator fallback)

```cron
5 * * * 2,5 cd /Users/josephshackelford/actions-runner-nfl-2.335.1/_work/nfl-predictions-pricing/nfl-predictions-pricing && /usr/bin/python3 scripts/run_2026_production_card.py --run-due >> ~/Library/Logs/nfl-hybrid-production-2026.log 2>&1
```

## 12. Prospective 2026 strength scorecard + shadow ledger

The frozen promotion contract is
[`PROSPECTIVE_VALIDATION_2026.md`](PROSPECTIVE_VALIDATION_2026.md) /
`outputs/prospective_2026_strength_preregistration.json` (schema
`PROSPECTIVE_2026_STRENGTH_V1`, hash
`a8bfca90d97c54ad42064854d4ed0a1c7115820cae998c5b282a2f9a0dd468e9`).

### 12a. Prospective strength scorecard

```bash
# Applies the frozen contract to the immutable prospective ledgers and
# writes $NFL_MODEL_ARTIFACT_ROOT/production-2026/prospective-strength/
#   PROSPECTIVE_2026_STATUS_SCORECARD.{json,md}
# Safe to run now: with no attached results every performance row reads
# INSUFFICIENT_PROSPECTIVE_SAMPLE / NOT_DEMONSTRATED / NOT_ESTABLISHED and
# no row is promoted. Runs even with $NFL_MODEL_ARTIFACT_ROOT unset
# (prints the empty-estate scorecard, writes nothing).
PYTHONPATH=src python3 scripts/report_2026_strength_scorecard.py \
  --data-through-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

### 12b. Shadow model-family ledger (already-elapsed games only)

```bash
# Replays the certified chronological OOF batching for the six frozen
# Fix-7 candidates and writes immutable first-write-wins records under
# $NFL_MODEL_ARTIFACT_ROOT/production-2026/shadow-model-family-ledger/.
# NEVER touches the forecast / evaluation ledger or the calibration seed;
# shadow output has no production-selection authority.
PYTHONPATH=src python3 scripts/run_2026_shadow_model_family.py --horizon ALL
```

### 12c. Executable-book policy activation (profitability gate)

Profitability is DISABLED (`BETTING_RULE_STATUS =
NOT_ACTIVATED_FOR_PROFITABILITY`) until `config/executable_books_2026.json`
exists and is hash-frozen. That file is **not** committed here and must
not invent a sportsbook. When ready, author it as a fixed ordered list of
eligible books (or another deterministic predeclared selection rule); the
reporter records its SHA-256, and the hash locks on the first eligible
live wager and is immutable for the season. Until then the reporter
reports `NOT_ESTABLISHED` / `EXECUTABLE_BOOK_POLICY_NOT_FROZEN`.

### 12d. Result attachment (unchanged)

Outcomes still attach only through the certified immutable path
(`scripts/run_2026_production_card.py --attach-results` /
`nfl_hybrid.production.run_2026.attach_result`): strict
`result_available_at_utc < attachment_run_time`, a separate sibling
`*.result.json`, never a rewrite of the original forecast. The prospective
strength scorecard reads those attached results; it never writes them.
