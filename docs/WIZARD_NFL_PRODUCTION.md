# NFL 2026 production on the Wizard server

Operational reference for the automated 2026 NFL production system: GitHub is
the source of truth and the scheduler, the Wizard SportsOdds server is the
permanent NFL runtime / data root / artifact root / web destination, and no
step depends on the operator's Mac.

This document covers **operations and topology only**. The model itself is
unchanged and is documented in
[`PRODUCTION_RUNBOOK_2026.md`](PRODUCTION_RUNBOOK_2026.md) and
[`MODEL_STRENGTH_AND_LIMITATIONS.md`](MODEL_STRENGTH_AND_LIMITATIONS.md).

---

## 1. The defect this system fixes

Production preflight reported:

```
infra_ready=true
live_2026_market_source_registered=true
schedule_2026_available=false
```

`run_2026.py` resolved its games/schedule population from
`external_data.resolve("backfill.games")` alone. That table is the certified
**historical** estate and its maximum season is 2025, so
`schedule_2026_available` was structurally `false` — permanently
`BLOCKED_ON_LIVE_INPUTS`, even on a host holding a valid, verified 2026
BallDontLie capture.

The repository already contained every piece needed to fix it:

| Piece | Location |
|---|---|
| Captured BallDontLie `/games` rows | `scripts/capture_bdl_2026_asof.py` writes them into the live observation log |
| Verified reader for those rows | `nfl_hybrid.data.bdl_market_bridge.read_games_rows` |
| Canonicalizer | `nfl_hybrid.providers.balldontlie.canonical.normalize_games` |
| Canonical game identity + team crosswalk | `nfl_hybrid.providers.balldontlie.team_crosswalk`, `canonical._canonical_game_id` |

Production simply never consumed them: nothing joined the canonical BDL path
to the games population that preflight and `run_horizon_batch` read.

`nfl_hybrid.data.games_population_2026` is that join, and only that join. It
adds **no second normalizer** — no team mapping, no game-id construction, no
kickoff parsing, no season-type inference. It validates a capture with the
existing validator, reads it with the existing reader, canonicalizes it with
the existing canonicalizer, and renames the resulting columns onto the
population contract.

---

## 2. The composite games population

```
certified historical backfill.games (2020-2025)
  + durable canonical BallDontLie 2026 REG/POST games
  = the ONE population read by both preflight and real production
```

Both `run_preflight` and `run_horizon_batch` call
`run_2026.load_games_population_with_provenance`, and both record the
population's `reg_post_content_sha256` (in `checks.schedule_source` and in
`input_hashes.games_population_content_sha256` respectively). Comparing those
two values is the proof that a preflight and the run that followed it saw the
same games.

Rules, all fail-closed:

- **PRESEASON is never included.** Twice over: the capture validator refuses a
  `PRE`-labelled capture outright, and `canonical_games_to_population_rows`
  drops any `PRE` row that reaches it. This matters because BDL restarts
  preseason week numbering at 1, so a preseason Week 1 could otherwise collide
  with a regular-season Week 1 canonical `game_id`.
- **Upcoming games carry null scores.** That is correct: the certified Elo
  replay only enqueues an update event for a game with both scores present,
  and a future game's `result_available_at_utc` is after every current cutoff.
- **A score transitions NULL → final exactly once.** A contradicted final score
  is a hard `GamesPopulationConflict`; a later, less complete read never
  un-records a result.
- **Conflicting canonical identity fails closed.** Same `game_id`, different
  teams or kickoff — between two evidence rows, two captures, or evidence and
  the historical estate — raises rather than picking a winner.
- **Stale unresolved rows are excluded and counted.** A game whose result was
  already chronologically available as of the evidence instant but whose scores
  are still unknown cannot honestly be a training observation or an upcoming
  forecast target, so it is dropped and named in the manifest.

Durable store (outside git, outside the historical estate):

```
$NFL_MODEL_ARTIFACT_ROOT/games-population-2026/
├── canonical_games_2026.parquet
└── canonical_games_2026.manifest.json    # row counts, content hash, contributing captures
```

---

## 3. Wizard server layout

Defined once, in [`ops/wizard/nfl_production_layout.sh`](../ops/wizard/nfl_production_layout.sh),
and sourced by every other ops script and every remote workflow step.

```
$WIZARD_NFL_HOME              default: $HOME/nfl-production-2026
├── repo/                     application source, installed at an exact commit
├── venv/                     Python env built from THIS repo's pyproject
├── data/                     NFL_MODEL_DATA_ROOT
├── artifacts/                NFL_MODEL_ARTIFACT_ROOT
├── live/                     NFL_LIVE_DATA_ROOT (raw BDL provider cache)
├── staging/                  pre-publication staging
├── state/                    resolved web root, operational state
└── logs/                     preflight/run/population logs (never predictions, never secrets)
```

### Why this shape

- **Everything hangs off `$HOME` by default.** Nothing assumes `/opt` or `/srv`
  is writable. The Wizard SSH account is a restricted deployment account, and a
  bootstrap that began with `mkdir -p /opt/...` would fail on its first
  unattended run. `WIZARD_NFL_HOME` overrides the root end to end (set the
  `WIZARD_NFL_HOME` environment secret in the `wizard-production` GitHub
  environment to relocate the whole installation).
- **Writability is proven, not assumed.** The bootstrap writes and removes a
  probe file in every directory; `[ -w ]` can disagree with reality on a
  read-only mount or under a restrictive ACL, and an unwritable artifact root
  must fail at bootstrap rather than half-way through a certified run.
- **NFL and NCAAF are never mixed.** Every path is under one NFL-only root and
  the virtualenv is built from this repository's own dependency declaration. No
  NCAAF checkout, venv, data directory or artifact directory is referenced
  anywhere in this repository.
- **Source is pushed, never pulled.** `git archive` on the GitHub-hosted runner
  produces the exact tree of the triggering commit, and it is copied in over
  SSH. The server needs no GitHub deploy key, no token and no egress to
  github.com, and production provably runs the commit the workflow ran on.
- **The web directory is discovered, never guessed.**
  [`ops/wizard/resolve_web_root.sh`](../ops/wizard/resolve_web_root.sh) resolves
  it from an explicit override, then the live nginx configuration (`nginx -T`,
  falling back to `/etc/nginx`), then a bounded search for the already-deployed
  `tools/odds-scanner/predictions/NFL/index.html`. It records the answer in
  `state/web_root`. If nothing resolves, it records nothing and exits non-zero:
  publishing into a path nginx does not serve looks exactly like success while
  publishing nothing.

### Publication

`scripts/publish_wizard_nfl_local.py` runs **on the server** and:

1. validates the card with the existing frozen `wizard-nfl-pricing-v2` gate
   (`publish_sportsodds_nfl.load_and_validate_json`) — no second schema;
2. writes the validated bytes to a temporary file **inside** the served
   directory (same filesystem, so the rename below is genuinely atomic);
3. retains the currently published card as `latest.json.prev`;
4. `os.replace`s the temporary file onto `latest.json`;
5. reads the served path back and fails if the bytes do not hash to what was
   validated.

There is no FTP in this path. `scripts/publish_sportsodds_nfl.py` (FTP) is left
untouched as a manual fallback and is never invoked by automation.

The existing NFL HTML shell already on the server is never modified.

#### The one-time permission grant this requires

Because the temporary file is created inside the served directory and renamed
onto `latest.json`, `wizard-deploy` needs write **and** search permission on
that single directory — creating, replacing and unlinking entries are
directory operations, not file operations. Nothing else in `/var/www` needs to
become writable, and `resolve_web_root.sh` reports the outcome as
`WIZARD_NFL_WEB_DIR_WRITABLE`.

`wizard-deploy` has no sudo, so this is a one-time change for root. A POSIX ACL
is the surgical way to do it: the directory keeps its existing owner, no group
memberships change, and no other directory is touched.

```bash
# Run ONCE, as root, on the Wizard server.
set -euo pipefail
NFL_DIR=/var/www/sportsodds/tools/odds-scanner/predictions/NFL

# The directory keeps its current owner (woo); only an ACL is added.
setfacl -m u:wizard-deploy:rwx "$NFL_DIR"
# New files wizard-deploy creates there stay world-readable for nginx.
setfacl -d -m u:wizard-deploy:rw-,u::rw-,g::r--,o::r-- "$NFL_DIR"

# Verify: wizard-deploy can write, nginx can still read, nothing else changed.
getfacl "$NFL_DIR"
sudo -u wizard-deploy test -w "$NFL_DIR" \
  && echo "WIZARD_NFL_WEB_DIR_WRITABLE=yes" \
  || echo "WIZARD_NFL_WEB_DIR_WRITABLE=no"
# A sticky bit would still block replacing another user's latest.json.
[ -k "$NFL_DIR" ] && echo "WARNING: sticky bit set on $NFL_DIR" || true
```

`publish_wizard_nfl_local.py` explicitly `chmod`s the published card to `0644`,
so the served `latest.json` is world-readable regardless of the creating
process's umask.

---

## 4. The one authoritative workflow

[`.github/workflows/nfl_2026_production.yml`](../.github/workflows/nfl_2026_production.yml)

| Job | Runner | Environment | Gate |
|---|---|---|---|
| `resolve` | GitHub-hosted | none | always; pure decision, reviewable on any ref |
| `daily-maintenance` | GitHub-hosted → SSH | `wizard-production` | `run_daily` **and** ref is `main` |
| `certified-production` | GitHub-hosted → SSH | `wizard-production` | `run_certified` **and** ref is `main` **and** daily did not fail |
| `report` | GitHub-hosted | none | `always()` |

### Scheduling

```
20 11 * * *                    daily maintenance
 5 16 * * 2,5                  certified TUE/FRI, correct instant during EDT
 5 17 * * 2,5                  certified TUE/FRI, correct instant during EST
*/15 * * 9,10,11,12,1 0,1,4    OPEN/MID/CLOSE sweep, game days (Sun/Mon/Thu)
25 * * 9,10,11,12,1 2,3,5,6    OPEN/MID/CLOSE sweep, other days
```

The cron expression is never the gate. `scripts/resolve_production_schedule.py`
calls the certified `run_2026.is_within_due_window` and
`run_2026.current_or_recent_cutoff` (built on `horizon_elo`'s own
`_card_monday_date` / `_card_noon_cutoff_utc`) to compute the real DST-aware
12:00–12:20 America/New_York window. Whichever of the two firings falls outside
that window resolves `certified_due=false` and the certified job is skipped —
a clean no-op, not a failure. There is no DAILY forecast horizon;
`certified_horizon` is only ever `TUE` or `FRI`.

The two sweep crons are a poll, not a timetable, because CLOSE is per game —
kickoff minus 60 minutes — so there is no single clock time to schedule it at.
The sweep asks `snapshot_stages_2026` which instants have passed for the
current week's games and executes those; a firing with nothing due exits 0
having done nothing. That is also why they deliberately pin no hour: dueness
comes from each game's own kickoff and from the same `America/New_York` card
arithmetic, so the November transition needs no second cron. They are bounded
to the season months to keep runner minutes and capture volume bounded, dense
on game days and hourly otherwise (which is what picks up first-board OPEN
discovery and the Friday-noon MID).

A snapshot instant is exact even though its execution is not: the cutoff the
model is fitted at and the market is reconstructed at is exactly kickoff minus
60 minutes, while the execution happens on the first sweep after it. It has to
be that way round — a market cannot be priced before it has been observed —
and a future instant is never executed.

### Snapshot sweep ([`run_stage_snapshots.sh`](../ops/wizard/run_stage_snapshots.sh))

Refresh, retrain point-in-time, snapshot, publish — in that order, reusing the
daily pass's own entrypoints for the first two and for recalibration so the two
paths cannot disagree about the population or the active calibrator:

1. `refresh_bdl_2026_games_evidence.py` — same script as the daily pass.
2. `update_2026_games_population.py` — same script; fails closed on bad
   evidence.
3. `create_official_capture.sh` — a capture whose nominal cutoff IS this
   sweep's instant.
4. `run_2026_stage_snapshots.py` — every due OPEN/MID/CLOSE batch. Games that
   share a kickoff share a CLOSE, so the 1:00pm ET group is one batch.
5. `generate_2026_recalibration_candidate.py --promote-if-eligible` — same
   script, same merged policy, same maturity firewall.
6. `attach_2026_results_from_population.py` then
   `report_2026_snapshot_performance.py` — results attach as their own records
   and the season CSV/JSON is rewritten.
7. `publish_2026_current_week.py` then `publish_wizard_nfl_local.py` — the
   current-week feed is assembled, each game frozen at its own CLOSE, and
   `latest.json` atomically replaced.
8. `prune_replaceable_artifacts.py` — dry run unless `--prune-apply`.

Exit codes are distinct per failure class: 2 usage, 3 evidence/population, 4
stage execution, 5–9 the five recalibration failure classes, 10 feed assembly.
Nothing is published when any of them fires.

### Why a game's CLOSE is a one-way door

Only a CLOSE snapshot is final and public for a game. Once published it is
recorded in
`production-2026/published-close-state/season=YYYY/week=WW/published_close_state.json`
and reused verbatim on every later publication. A later pass that would change
an already-closed game fails the whole publication rather than rewriting
history, so a published close cannot drift even if its forecast were somehow
re-derived. Games that have not closed publish from their best pregame
snapshot (MID, else OPEN) and keep updating until their own CLOSE.

Which stage a game was published from is operational bookkeeping and stays in
that sidecar: the public payload remains exactly `wizard-nfl-pricing-v2`, with
the same eleven game keys in the same order.

### Daily pass ([`run_daily_maintenance.sh`](../ops/wizard/run_daily_maintenance.sh))

1. `refresh_bdl_2026_games_evidence.py` — immutable, append-only `/games`
   capture (soft-fails on a provider outage; the rest of the pass still runs);
2. `update_2026_games_population.py --latest-games-evidence` — durable
   population update, fail-closed on conflicts;
3. `attach_2026_results_from_population.py` + `report_2026_prospective_performance.py`
   — prospective evaluation;
4. `generate_2026_recalibration_candidate.py --promote-if-eligible` — generate
   the candidate, then automatically promote it when the operational policy and
   the preregistered maturity firewall both allow it (section 6).

It publishes nothing, then the runner verifies the public feed with no
expectations (a liveness and contract check of whatever is currently served).

#### A legitimate no-op is not a missing input

For an unattended season these two states must never be confused, so stage 4
distinguishes them by exit code and the pass prints
`daily_maintenance_status=OK` only when every stage succeeded.

Exit 0 — healthy, nothing to do. The candidate was generated and the promotion
decision was `PROMOTED`, `NOT_YET_MATURE`, `ALREADY_ACTIVE`, `NO_CANDIDATE` or
`POLICY_DISABLED`. **A `NOT_YET_MATURE` day is a success**, and for most of the
season it is the expected outcome.

Everything else is a fail-closed day, because it means production could not
build a candidate or could not resolve a calibrator at all — the state where
the historical estate, the certified baseline seed or the active pointer is
missing or corrupt. Silently continuing would let production drift for weeks
with nobody seeing it.

| pass exit | meaning |
|---|---|
| 3 | the 2026 games population rejected the available evidence |
| 4 | promotion policy, policy-lock or candidate integrity violation |
| 5 | candidate generation itself failed (e.g. a required historical estate is absent on this host) |
| 6 | the artifact root could not be resolved |
| 7 | the active calibrator could not be resolved (certified baseline seed missing, or a corrupt active pointer) |

Exit 7 is checked independently of the promotion decision on purpose: below the
maturity floor, promotion returns `NOT_YET_MATURE` without ever needing to
resolve a calibrator, so a host missing the certified baseline seed would
otherwise report a perfectly healthy day while being unable to price a single
card.

### Certified pass ([`run_certified_card.sh`](../ops/wizard/run_certified_card.sh))

1. verify the official capture exists and hashes to exactly what was declared;
2. `run_2026_production_card.py --preflight` — **must** report `READY`;
3. `run_2026_production_card.py --horizon TUE|FRI --market-capture-manifest …`;
4. `export_wizard_nfl_pricing.py` — the frozen `wizard-nfl-pricing-v2` contract;
5. `publish_wizard_nfl_local.py` — atomic publication.

The runner then verifies the public feed with `--expect-sha256` set to the
card that was just published. A server-side rename proves the bytes landed;
only this proves the public site serves them.

---

## 5. Automatic retraining

No new retraining engine exists and the Ridge fit behaviour is untouched.

Retraining is a consequence of the population, not a separate mechanism. When a
completed 2026 game is added to the durable population, it becomes part of the
composite population that `run_horizon_batch` reads. The **existing**
chronological rules then decide when it is used:

- the certified Elo replay enqueues an update event for it because both scores
  are present, so it moves the ratings later cards are built on;
- the training mask admits it to a fit once
  `result_available_at_utc < target_cutoff_utc` (STRICT), where
  `result_available_at_utc` is the certified conservative kickoff+5h floor.

A future game therefore cannot enter a fit (its availability is after every
current cutoff), and a completed game cannot leak backward into an earlier
cutoff's fit. Both directions are covered by tests.

Automatic recalibration (section 6) rides on the same property: a candidate is
fit over the same composite population, so newly completed games reach the
calibrator through exactly the chronology that already governs the Ridge fit.

---

## 6. Automatic recalibration, and automatic promotion once mature

### Candidate generation

`generate_2026_recalibration_candidate.py` runs every day as stage 4 of the
daily pass. It reuses the certified chronological calibration machinery
(`chronological_calibration`, `three_way`) to produce a candidate seed and
writes it — with exact per-stream memberships, thresholds and hashes — to

```
$NFL_MODEL_ARTIFACT_ROOT/production-2026/recalibration-candidates/<candidate_id>/
├── candidate_calibration_seed.json
└── candidate_manifest.json
```

outside git, content-addressed and immutable. Regenerating identical evidence
is an idempotent no-op rather than a new near-duplicate, because the candidate
id *is* the hash of its own seed.

### The two separate gates

There are two questions, and conflating them is what would break the science.

`promotion_authorization` reads the historical preregistration
`outputs/prospective_2026_strength_preregistration.json` and answers only this:
does that document authorize a **scientific refit** — a new calibrator family
or a new calibrator config? It answers no, permanently, because the document
asserts `invariance.no_scientific_refit = true`. That document is not edited,
reinterpreted or superseded.

`config/recalibration_promotion_2026.json` is a **versioned operational
overlay** and answers a different question: may the *already-certified*
calibrator family and config be re-fitted on more 2026 evidence once the
*already-preregistered* sample-maturity firewall is satisfied? It answers yes.
The two answers are compatible because the policy is only accepted at all if it
requires every candidate's calibrator family and both config hashes to **equal
the certified ones**. A promotion is therefore the same estimator and the same
config on more data — a recalibration, not a refit.

There is deliberately **no** "the candidate must beat the active calibrator by
X" threshold anywhere. Any such threshold would have to be chosen after
observing 2026 results. Promotion is a deterministic scheduled recalibration
behind a preregistered sample-size firewall, not a selection event.

### The maturity firewall

The game floor lives in exactly one place: `prospective_strength_2026.`
`PROMOTION_ELIGIBLE_MIN_GAMES` (currently 200 unique completed 2026 games). The
policy records *where* to read it, never *what* it is — `numeric_threshold_`
`duplicated_here` is `false`, and the number appears nowhere in the policy
bytes or in the promotion code. `PromotionPolicy.minimum_prospective_games()`
imports the frozen module and reads the symbol, so the contract and the
operational overlay cannot drift apart.

Below the floor, `evaluate_promotion` returns `NOT_YET_MATURE`. That is a
**success**: candidates keep being generated, Ridge retraining keeps following
the existing chronology, and the current calibrator stays active. The daily
pass exits 0.

### The eleven requirements

Above the floor, a candidate becomes active only when all of these hold.

| # | Requirement |
|---|---|
| 1 | prospective maturity is `PROMOTION_ELIGIBLE` under the frozen rule |
| 2 | the policy is enabled and its hash matches the policy lock |
| 3 | the candidate carries exactly the four production streams |
| 4 | every conditional calibrator is fitted |
| 5 | every push calibration state is well-formed |
| 6 | every parameter is finite, including each push bucket scale |
| 7 | family and both config hashes equal the frozen certified ones |
| 8 | the seed content hash and the manifest hashes verify |
| 9 | training membership is internally consistent and chronologically possible |
| 10 | membership has not regressed against the active candidate |
| 11 | the candidate differs from the currently active one |

Requirements 4 and 5 are checked by the **same** frozen predicate production
pricing uses (`run_2026._frozen_stream_calibration_ready`), so a candidate can
never pass promotion and then be rejected at pricing time.

The order matters: maturity is checked before the candidate is validated.
Before the floor a candidate legitimately has unfitted streams (too few
labelled rows yet) and must report `NOT_YET_MATURE`; after the floor the same
unfitted stream is genuine corruption and fails closed.

### The active-calibrator pointer

The certified baseline is never overwritten and never copied. Instead:

```
$NFL_MODEL_ARTIFACT_ROOT/production-2026/recalibration/
├── active_calibrator.json          <- a tiny POINTER
└── promotion-events/<event_hash>.json
```

**Absence of the pointer means the immutable Fix-8 baseline is active.** A
promotion atomically rewrites that one small file (temp file plus
`os.replace` in the same directory) to *reference* a candidate seed that
already exists under `recalibration-candidates/`. The seed bytes are never
duplicated, so there is no second calibration estate to keep in sync.

The pointer records the schema version, the candidate id (or `BASELINE`), the
exact seed path, the seed SHA256, the candidate manifest SHA256, the
promotion-policy SHA256, `promoted_at_utc`, the git commit, the previous active
candidate and its hash, the prospective unique completed-game count at
promotion, and the training-membership hashes for all four streams.

A promotion event is written **only when the active calibrator actually
changes** — append-only and content-addressed. Re-running promotion against the
already-active candidate writes nothing at all: no pointer rewrite, no second
event, no seed copy.

### One resolver, and no silent fallback

`run_2026` no longer hardcodes the Fix-8 seed as the only possible live
calibrator. Both `run_preflight` and `run_horizon_batch` call
`recalibration_2026.resolve_active_calibrator`, which has exactly three
outcomes:

- **no pointer** → the verified immutable baseline;
- **valid pointer** → the referenced immutable candidate, with the seed hash,
  the manifest hash, the policy hash against the lock, and all four streams'
  readiness re-verified on *every* read;
- **anything else** → `ActiveCalibratorError`.

A pointer that exists but does not verify **never** degrades to the baseline.
Preflight turns that into the blocking problem `active_calibrator_invalid` (so
a certified run cannot reach `READY`), and a batch turns it into the
fail-closed status `ACTIVE_CALIBRATOR_INVALID` before a single forecast is
written. A silent fallback would publish a card that claims one calibrator and
used another, which is worse than publishing nothing.

Only a genuinely unavailable artifact root (a hermetic test, or a host with no
`NFL_MODEL_ARTIFACT_ROOT`) degrades to an empty seed — the pre-existing
uncalibrated state every stream already fails closed on, and not a fallback
from a broken promotion.

### The policy lock

On first operational use, one small first-write-wins lock is written to

```
$NFL_MODEL_ARTIFACT_ROOT/production-2026/recalibration-policy/policy_lock.json
```

recording the policy SHA256, the policy schema version, the git commit and the
creation timestamp. The create is `O_CREAT|O_EXCL`, so two concurrent runs
cannot both claim to have written it. Every later run verifies the same hash; a
policy that changed after being locked raises `PolicyLockViolation` and the
daily pass exits 4. A new policy is never silently accepted.

An operator cannot manufacture a promotion. Nothing in the module consults an
environment variable, a CLI flag or a filesystem marker, the daily script has
no `--force` or `--bypass` option, and a policy that *declared* such a control
— or that relaxed any requirement in the table above, omitted a stream,
declared itself not fail-closed, made the baseline mutable, or restated the
maturity threshold as a literal — is rejected by `_validate_policy_document`
before any candidate is even examined.

### Provenance

Every forecast of record, every run manifest and every evaluation-ledger
provenance block now carries `active_calibrator_source` (`BASELINE` or
`CANDIDATE`), `active_calibrator_candidate_id`,
`active_calibrator_seed_sha256` and `recalibration_policy_sha256`. A card is
only reproducible if the calibrator it used is named.

This is internal evidence only. The published `wizard-nfl-pricing-v2` contract
is unchanged and structurally cannot absorb these fields: the exporter builds an
explicit allow-list and asserts `tuple(public_game.keys()) == GAME_KEY_ORDER`.

---

## 7. Runtime assets and the one-time migration

`scripts/audit_production_runtime_assets.py` enumerates every non-git runtime
dependency under `NFL_MODEL_DATA_ROOT`, `NFL_MODEL_ARTIFACT_ROOT` and
`NFL_LIVE_DATA_ROOT`, and classifies each:

| Class | Meaning | Migration |
|---|---|---|
| `REBUILDABLE_PUBLIC` | Free public source with an in-repo producer (nflverse backfill) | not required |
| `REGENERABLE_FROM_ESTATE` | Deterministically recomputable by a certified in-repo script | not required |
| `IRREPLACEABLE_PAID_HISTORY` | Purchased point-in-time Odds API snapshots | **required** |
| `IRREPLACEABLE_LIVE_EVIDENCE` | BDL observation log, forecast/evaluation ledgers, run manifests | **required** |
| `IRREPLACEABLE_CERTIFIED` | The frozen production calibration seed (its hash is part of the contract) | **required** |

### The migration path

Direct Mac → Wizard connectivity has failed before, so the migration goes
through GitHub:

```
Mac self-hosted runner
  -> encrypted/private GitHub Actions artifact
  -> GitHub-hosted runner
  -> verified SSH transfer to Wizard
```

The Mac self-hosted runner is used **only** for this migration.

1. On the Mac: `python scripts/audit_production_runtime_assets.py --hash
   --output mac_manifest.json`, and archive the assets whose
   `migration_required` is true.
2. Upload as a private workflow artifact; download on a GitHub-hosted runner;
   transfer over the proven SSH channel into `$WIZARD_NFL_DATA_DIR` /
   `$WIZARD_NFL_ARTIFACT_DIR` / `$WIZARD_NFL_LIVE_DIR`.
3. On Wizard: `python scripts/audit_production_runtime_assets.py --hash
   --output wizard_manifest.json`.
4. `python scripts/audit_production_runtime_assets.py --compare
   mac_manifest.json wizard_manifest.json`.

**Nothing is deleted on the Mac until that comparison reports
`safe_to_delete_source: true`,** which requires every migration-required asset
to be byte-identical on Wizard and both manifests to have been produced with
`--hash`. No live data is ever committed to git.

---

## 8. Running the first real public forecast from the existing Week-1 capture

The official Week-1 TUE capture must not be recaptured or modified.

### A capture manifest has two different SHA256 integrity objects

This distinction is not pedantic — conflating the two is what produced the
Week-1 TUE "hash discrepancy", where one recorded value
(`59d1b46e…`) did not match the manifest file's own hash (`337c71a2…`).
[`scripts/capture_bdl_2026_asof.py`](../scripts/capture_bdl_2026_asof.py)
creates both, and they can never be equal:

| object | how it is computed | what it identifies |
|---|---|---|
| `MANIFEST_CONTENT_SHA256` | `sha256(deterministic_json(body))` over the body **without** its own `manifest_sha256` key, serialized compactly (`separators=(",", ":")`, `sort_keys=True`). Stored **inside** the file as `manifest_sha256`. | the capture's CONTENT. Immune to pretty-printing. |
| `MANIFEST_FILE_SHA256` | `sha256sum manifest.json` | the exact BYTES on disk |

They differ for two independent reasons: the file is written with `indent=2`
rather than compact separators, and the file contains the `manifest_sha256`
field that did not exist when the content hash was computed — a
self-referential hash cannot cover itself.

The practical consequence: **re-serializing a manifest changes its file hash
while leaving its content hash identical.** So a file hash that no longer
matches a recorded value is not by itself evidence that the evidence was
altered, and a content hash that fails to recompute *is*.

### Resolving and verifying a capture's identity

[`scripts/verify_capture_manifest_integrity.py`](../scripts/verify_capture_manifest_integrity.py)
is the one place that decides this, and
[`run_certified_card.sh`](../ops/wizard/run_certified_card.sh) calls it before
anything else runs. It always does three things and fails closed on any of
them:

1. **self-verifies** the capture by recomputing `MANIFEST_CONTENT_SHA256` from
   the body and comparing it with the `manifest_sha256` the file carries. This
   is what detects a mutated capture, it requires no external reference value,
   and nothing in production performed it before;
2. **resolves a declared hash** to whichever object it is, so a correctly
   recorded value is never rejected merely for referring to the other object,
   and a value matching neither is never accepted;
3. **records both objects** in the run evidence.

To determine which object any recorded value refers to, run it against the
capture — this is read-only and never rewrites the manifest:

```bash
python scripts/verify_capture_manifest_integrity.py \
  "<capture>/manifest.json" \
  --declared-sha256 59d1b46e488a1e80c695cb77081f80b346f29a808422439187aabcd8981b10ab
```

`capture_manifest_declared_object` then reports `MANIFEST_FILE_SHA256`,
`MANIFEST_CONTENT_SHA256`, or `NO_MATCH`. Only the last is a genuine integrity
failure, and it fails the certified run closed.

After migration the capture lives on Wizard at:

```
$WIZARD_NFL_HOME/data/live-observation-log/balldontlie-2026/season=2026/week=01/horizon=TUE/capture=20260908T153615Z/manifest.json
```

Dispatch, from `main`:

```bash
gh workflow run nfl_2026_production.yml \
  --ref main \
  -f mode=certified_only \
  -f horizon=TUE \
  -f market_capture_manifest="$HOME/nfl-production-2026/data/live-observation-log/balldontlie-2026/season=2026/week=01/horizon=TUE/capture=20260908T153615Z/manifest.json" \
  -f market_capture_sha256=59d1b46e488a1e80c695cb77081f80b346f29a808422439187aabcd8981b10ab \
  -f publish=publish
```

Substitute the real absolute path if `WIZARD_NFL_HOME` was overridden. Run it
once with `-f publish=dry_run` first: that verifies the capture's integrity,
requires a `READY` preflight, runs the certified card, exports the
`wizard-nfl-pricing-v2` contract and reports the destination without writing
`latest.json`.

Because `market_capture_manifest` is supplied, the workflow uses that manifest
verbatim and creates no new capture. The declared `market_capture_sha256` is
resolved to one of the two integrity objects on the server before anything else
runs, the capture's own content hash is self-verified, and either check failing
aborts before preflight.

---

## 9. Required GitHub configuration

`wizard-production` environment (restricted to `main`):

| Secret | Purpose | Status |
|---|---|---|
| `WIZARD_SSH_HOST` | Wizard SSH endpoint | configured, proven |
| `WIZARD_SSH_PORT` | Wizard SSH port | configured, proven |
| `WIZARD_SSH_USER` | deployment account | configured, proven |
| `WIZARD_SSH_PRIVATE_KEY` | deployment key | configured, proven |
| `WIZARD_SSH_KNOWN_HOSTS` | pinned host key | configured, proven |
| `BALLDONTLIE_API_KEY` | live schedule/results/odds capture | required |
| `THE_ODDS_API_KEY` | historical market reconstruction | required |
| `WIZARD_NFL_HOME` | override the installation root | optional |
| `WIZARD_NFL_WEB_DIR` | override the served directory | optional, recommended once known |
