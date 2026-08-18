# Remediation Status — 2026

Single source of truth for the 2026 remediation effort while the rebuild is
in progress. Supersedes narrative status claims in older reports for
anything listed here — if a historical report (`FINAL_REPORT.md`,
`CALIBRATION_REPORT.md`, etc.) disagrees with this table, this table is
current and the historical report is a frozen record of a prior build state.
See `STATUS.md` for the general per-component status and `README.md` for the
architecture overview.

Statuses used: `MERGED`, `IN PROGRESS`, `PENDING`, `HISTORICAL`.

Only Fix 1.5 and Fix 2 are `MERGED` as of 2026-08-18, based on the current
`main` branch history (`git log main`). Nothing later in the roadmap has
started implementation on `main`.

| Work item | Status | Evidence / PR or commit | What it establishes |
|---|---|---|---|
| Data forensic audit | **HISTORICAL** | prior audit findings (see PR #13/#14 descriptions for the defects found) | Identified the leakage and data-resolution defects Fix 1.5–3 remediate |
| Fix 1.5 — central data resolution | **MERGED** | PR #13, commit `e68a162`, merge `92ecb45` | `NFL_MODEL_DATA_ROOT`-based registry (`src/nfl_hybrid/data/external_data.py`); every producer/consumer resolves the same dataset keys, no hard-coded machine paths |
| Fix 2 — chronological pregame state | **MERGED** | PR #14, commit `4d6d04d`, merge `45cf2c4` | Authoritative chronological pregame-state interface (`src/nfl_hybrid/features/pregame_state.py`), per-family lineage, cross-family banned-column check, `docs/PREGAME_STATE_DICTIONARY.yaml` |
| Fix 3 — chronological OOF + uncertainty | **IN PROGRESS — not yet merged** | branch `fix/chronological-oof-uncertainty` | Not established on `main`. No chronological OOF membership proof or uncertainty output from this branch should be cited as current system behavior |
| Chronological calibration | **PENDING** | blocked on Fix 3 | Not started |
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
merges to `main`. When Fix 3 merges, update its row to `MERGED`, add the PR
reference, and re-check whether "Chronological calibration" can move from
`PENDING` to `IN PROGRESS`.
