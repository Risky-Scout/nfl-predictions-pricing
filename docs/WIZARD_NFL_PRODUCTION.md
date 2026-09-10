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
20 11 * * *      daily maintenance
 5 16 * * 2,5    certified TUE/FRI, correct instant during EDT
 5 17 * * 2,5    certified TUE/FRI, correct instant during EST
```

The cron expression is never the gate. `scripts/resolve_production_schedule.py`
calls the certified `run_2026.is_within_due_window` and
`run_2026.current_or_recent_cutoff` (built on `horizon_elo`'s own
`_card_monday_date` / `_card_noon_cutoff_utc`) to compute the real DST-aware
12:00–12:20 America/New_York window. Whichever of the two firings falls outside
that window resolves `certified_due=false` and the certified job is skipped —
a clean no-op, not a failure. There is no DAILY forecast horizon;
`certified_horizon` is only ever `TUE` or `FRI`.

### Daily pass ([`run_daily_maintenance.sh`](../ops/wizard/run_daily_maintenance.sh))

1. `refresh_bdl_2026_games_evidence.py` — immutable, append-only `/games`
   capture (soft-fails on a provider outage; the rest of the pass still runs);
2. `update_2026_games_population.py --latest-games-evidence` — durable
   population update, fail-closed on conflicts;
3. `attach_2026_results_from_population.py` + `report_2026_prospective_performance.py`
   — prospective evaluation;
4. `generate_2026_recalibration_candidate.py` — candidate + promotion state.

It publishes nothing, then the runner verifies the public feed with no
expectations (a liveness and contract check of whatever is currently served).

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

---

## 6. Automatic recalibration and the promotion gate

`generate_2026_recalibration_candidate.py` runs every day. It reuses the
certified chronological calibration machinery
(`chronological_calibration`, `three_way`) to produce a candidate seed, and
writes it — with exact stream memberships, thresholds and hashes — to

```
$NFL_MODEL_ARTIFACT_ROOT/production-2026/recalibration-candidates/<candidate_id>/
├── candidate_calibration_seed.json
└── candidate_manifest.json
```

outside git.

Promotion is decided **only** by the frozen machine-readable preregistration
`outputs/prospective_2026_strength_preregistration.json`. Today that document:

- carries no enabled `certified_calibrator_promotion_authorization` block, and
- asserts `invariance.no_scientific_refit = true`.

So `promotion_authorization` returns `PROMOTION_NOT_AUTHORIZED`, and
`promote_candidate` raises `PromotionRefused` without touching the certified
seed. An operator cannot manufacture an authorization: no environment variable,
CLI flag or filesystem marker is consulted.

Steady state, which is the expected state:

- candidate generation runs automatically;
- the certified calibrator remains active and byte-identical;
- the workflow reports the candidate state;
- promotion fails closed.

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

The official Week-1 TUE capture must not be recaptured or modified. Its
manifest hash is:

```
59d1b46e488a1e80c695cb77081f80b346f29a808422439187aabcd8981b10ab
```

After migration it lives on Wizard at:

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
once with `-f publish=dry_run` first: that validates the capture hash, requires
a `READY` preflight, runs the certified card, exports the
`wizard-nfl-pricing-v2` contract and reports the destination without writing
`latest.json`.

Because `market_capture_manifest` is supplied, the workflow uses that manifest
verbatim and creates no new capture. The declared `market_capture_sha256` is
recomputed on the server before anything else runs, and a mismatch aborts
before preflight.

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
