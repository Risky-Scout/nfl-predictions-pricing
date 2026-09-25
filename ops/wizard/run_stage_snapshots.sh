#!/usr/bin/env bash
# SNAPSHOT SWEEP: execute whatever OPEN/MID/CLOSE stages have become due for
# the current week, then publish the current-week public feed.
#
# Invoked over SSH by the snapshot-sweep job of
# .github/workflows/nfl_2026_production.yml. ORCHESTRATOR ONLY: every step
# shells out to an existing, certified entrypoint. No model logic, no feature
# construction, no calibration, no market rule, no schema.
#
#   1. refresh schedule/results evidence          (existing daily script)
#   2. update the durable canonical 2026 population (existing daily script)
#   3. capture the market AT this sweep's instant   (existing capture script)
#   4. execute every due stage                      (run_2026_stage_snapshots)
#   5. recalibration candidate + promotion          (existing daily script)
#   6. attach provider-final results, write season reports
#   7. publish the current-week feed and freeze each game at its CLOSE
#   8. prune replaceable intermediates             (dry run unless asked)
#
# Steps 1, 2, 5 are the SAME scripts the daily pass runs. That is deliberate:
# the point-in-time contract says refresh-then-retrain-then-snapshot, and
# reusing the daily entrypoints is what guarantees the snapshot path and the
# daily path can never disagree about what the population says or which
# calibrator is active.
#
# Storage: step 3 writes one capture per sweep. Captures are immutable
# evidence and are never deleted here; step 8 prunes only the re-materializable
# per-book quotes artifacts derived from them.
#
# Usage (on the server):
#   bash run_stage_snapshots.sh [--stage ALL|OPEN|MID|CLOSE] [--as-of ISO8601]
#       [--skip-capture] [--dry-run-publish] [--prune-apply]
set -euo pipefail

STAGE="ALL"
AS_OF=""
SKIP_CAPTURE=0
DRY_RUN_PUBLISH=0
PRUNE_APPLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --stage)            STAGE="$2"; shift 2 ;;
        --as-of)            AS_OF="$2"; shift 2 ;;
        --skip-capture)     SKIP_CAPTURE=1; shift ;;
        --dry-run-publish)  DRY_RUN_PUBLISH=1; shift ;;
        --prune-apply)      PRUNE_APPLY=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${STAGE}" in
    ALL|OPEN|MID|CLOSE) ;;
    *) echo "FAIL CLOSED: --stage must be ALL, OPEN, MID or CLOSE; got '${STAGE}'" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ops/wizard/nfl_production_layout.sh
. "${HERE}/nfl_production_layout.sh"

PY="${WIZARD_NFL_VENV_DIR}/bin/python"
cd "${WIZARD_NFL_REPO_DIR}"
mkdir -p "${WIZARD_NFL_LOG_DIR}"

# ONE as-of for the whole sweep, for the same reason the certified card
# orchestrator builds one: every stage below must be asked about the same
# instant, or they would disagree about what is due.
as_of_args=()
[ -n "${AS_OF}" ] && as_of_args=(--as-of "${AS_OF}")

# --- 1-2. refresh evidence and the canonical population --------------------
echo "=== 1. games evidence ==="
set +e
"${PY}" scripts/refresh_bdl_2026_games_evidence.py --season 2026
evidence_exit=$?
set -e
if [ "${evidence_exit}" -ne 0 ]; then
    # A provider blip must not stop a due CLOSE from being priced off the
    # capture it already has; the population step below still fails closed on
    # genuinely bad evidence.
    echo "games_evidence_refresh=PROVIDER_UNAVAILABLE"
else
    echo "games_evidence_refresh=OK"
fi

echo "=== 2. canonical games population ==="
set +e
"${PY}" scripts/update_2026_games_population.py --latest-games-evidence
population_exit=$?
set -e
if [ "${population_exit}" -eq 2 ]; then
    echo "FAIL CLOSED: the canonical 2026 population rejected the newest evidence" >&2
    exit 3
elif [ "${population_exit}" -ne 0 ]; then
    echo "FAIL CLOSED: population update failed (exit ${population_exit})" >&2
    exit 3
fi
echo "games_population=OK"

# --- 3. market capture at THIS instant -------------------------------------
capture_args=()
if [ "${SKIP_CAPTURE}" -eq 0 ]; then
    echo "=== 3. point-in-time stage market observation ==="
    # --stage-observation, NOT a TUE/FRI capture. A sweep runs whenever a
    # stage instant has passed -- Wednesday, Friday noon, an hour before any
    # kickoff -- and the certified TUE/FRI capture window would reject all of
    # those. The observation records the board at THIS instant and is priced
    # at a stage cutoff at or after it. The certified capture path is
    # untouched and still fails closed outside its own window.
    set +e
    capture_out="$(bash "${HERE}/create_official_capture.sh" \
        --horizon TUE --stage-observation "${as_of_args[@]}" 2>&1)"
    capture_exit=$?
    set -e
    echo "${capture_out}"
    if [ "${capture_exit}" -ne 0 ]; then
        echo "market_capture=UNAVAILABLE"
    else
        # The capture script's own machine-readable key. Parsing a different
        # one silently dropped every successful capture.
        manifest="$(printf '%s\n' "${capture_out}" | sed -n 's/^OFFICIAL_CAPTURE_MANIFEST=//p' | tail -1)"
        if [ -n "${manifest}" ] && [ -f "${manifest}" ]; then
            capture_args=(--market-capture-manifest "${manifest}")
            echo "market_capture=${manifest}"
        else
            echo "market_capture=UNAVAILABLE"
        fi
    fi
else
    echo "=== 3. market capture SKIPPED ==="
fi

# --- 4. execute every due stage --------------------------------------------
echo "=== 4. due stage snapshots (${STAGE}) ==="
sweep_json="${WIZARD_NFL_LOG_DIR}/stage-sweep-$(date -u +%Y%m%dT%H%M%SZ).json"
set +e
"${PY}" scripts/run_2026_stage_snapshots.py \
    --stage "${STAGE}" \
    "${as_of_args[@]}" \
    "${capture_args[@]}" \
    > "${sweep_json}"
sweep_exit=$?
set -e
cat "${sweep_json}"
if [ "${sweep_exit}" -ne 0 ]; then
    echo "FAIL CLOSED: due stage execution failed (exit ${sweep_exit}); nothing is published" >&2
    exit 4
fi
echo "stage_sweep=OK"

# Did anything actually execute? A sweep that fires with no stage instant due
# is the declared clean no-op, and it must not go on to assemble a feed: with
# no snapshot for any game, assembly correctly refuses to publish an empty
# card and would turn a legitimate no-op into exit 10.
#
# This is deliberately narrow. It asks only whether zero batches were DUE --
# not whether execution failed (that already exited 4 above), and not whether
# assembly would be happy. A sweep that did execute something always proceeds
# to publication and still fails closed there on partial or ambiguous state.
due_batches="$("${PY}" -c '
import json,sys
doc=json.load(open(sys.argv[1]))
print(sum(stage.get("due_batches",0) for stage in doc.get("stages",[])))
' "${sweep_json}")"
echo "due_batches=${due_batches}"

# --- 5. recalibration candidate + promotion --------------------------------
# The SAME entrypoint and the SAME merged policy the daily pass uses, so the
# existing maturity firewall decides promotion here exactly as it does there.
#
# ONLY WHEN THIS SWEEP ACTUALLY EXECUTED A STAGE.
#
# This is the most expensive thing the sweep does: seven of the seventeen
# minutes an idle poll takes, measured across live runs. On an idle poll it is
# also pure repetition -- no snapshot was written, so it re-reads the same
# prospective ledger and re-derives the same candidate and the same verdict.
#
# That repetition is not free, because the dense windows fire every fifteen
# minutes and the concurrency group serialises production on purpose. A
# seventeen-minute idle sweep cannot finish before the next one is due, so the
# queue grows and the CLOSE that the dense window exists to catch is executed
# later and later. Dropping the idle path to ten minutes is what makes the
# fifteen-minute cadence actually hold in wall-clock terms rather than only in
# cron terms.
#
# Nothing here becomes manual or less automatic:
#   - the once-daily maintenance pass runs this same entrypoint with the same
#     --promote-if-eligible flag, after its own results attachment;
#   - any sweep that DID execute a stage still runs it exactly as before, and
#     executing a stage is the only thing a sweep does that can move the
#     prospective ledger itself;
#   - the 200-game maturity gate is untouched -- it lives inside the
#     entrypoint, not in this decision.
#
# The one thing that genuinely waits is a promotion that becomes eligible
# purely because a game finished and was graded during an idle stretch. That
# is picked up by the next executing sweep or by the daily pass, whichever
# comes first. It was already the case that a sweep's own stage 6 result
# attachment lands AFTER this step, so the freshest results were never visible
# to the same sweep's recalibration anyway.
echo "=== 5. recalibration ==="
if [ "${due_batches}" -eq 0 ]; then
    echo "recalibration=SKIPPED_NOTHING_EXECUTED"
else
    set +e
    "${PY}" scripts/generate_2026_recalibration_candidate.py --promote-if-eligible
    recalibration_exit=$?
    set -e
    case "${recalibration_exit}" in
        0) echo "recalibration=OK" ;;
        4) echo "FAIL CLOSED: recalibration policy/integrity violation" >&2; exit 5 ;;
        1) echo "FAIL CLOSED: recalibration candidate generation failed" >&2; exit 6 ;;
        2) echo "FAIL CLOSED: artifact root unresolvable" >&2; exit 7 ;;
        3) echo "FAIL CLOSED: active calibrator unresolvable" >&2; exit 8 ;;
        *) echo "FAIL CLOSED: unexpected recalibration failure (exit ${recalibration_exit})" >&2; exit 9 ;;
    esac
fi

# NOTHING NEW TO EXECUTE IS NOT THE SAME AS NOTHING TO PUBLISH.
#
# This used to `exit 0` right here whenever due_batches was zero. That was
# added when feed assembly still failed closed on a week with no CLOSE yet, so
# skipping publication was the only way an idle sweep could succeed. PR #58
# removed that reason -- an empty current week now publishes a valid envelope
# -- and the guard then became the thing PREVENTING the fix from ever running:
# publication was unreachable on exactly the sweeps that needed it, and the
# public page kept serving a Week-1 card from 2026-09-15.
#
# So the sweep no longer stops here. Stage execution is still a no-op and
# still reports NOOP_NOTHING_DUE; the difference is that the ordinary
# publication stage below now runs either way and decides for itself what the
# current week should show. Nothing is fabricated to force it: no stage is
# invented, no capture is taken, and the publisher still publishes CLOSE-only.
if [ "${due_batches}" -eq 0 ]; then
    echo "stage_execution=NOOP_NOTHING_DUE"
fi

# --- 6. results and season reporting ---------------------------------------
echo "=== 6. results + season reporting ==="
"${PY}" scripts/attach_2026_results_from_population.py
"${PY}" scripts/report_2026_snapshot_performance.py \
    --artifact-root "${NFL_MODEL_ARTIFACT_ROOT}" \
    --output-dir "${NFL_MODEL_ARTIFACT_ROOT}/production-2026/snapshot-performance-reports"
echo "season_reporting=OK"

# --- 7. current-week publication -------------------------------------------
echo "=== 7. current-week publication ==="
public_json="${NFL_MODEL_ARTIFACT_ROOT}/public/wizardofodds/nfl-pricing/latest.json"
set +e
"${PY}" scripts/publish_2026_current_week.py \
    "${as_of_args[@]}" \
    --artifact-root "${NFL_MODEL_ARTIFACT_ROOT}" \
    --output "${public_json}"
publish_exit=$?
set -e
if [ "${publish_exit}" -ne 0 ]; then
    echo "FAIL CLOSED: current-week feed assembly failed (exit ${publish_exit})" >&2
    exit 10
fi

publish_args=(--json "${public_json}")
[ "${DRY_RUN_PUBLISH}" -eq 1 ] && publish_args+=(--dry-run)
"${PY}" scripts/publish_wizard_nfl_local.py "${publish_args[@]}"

echo "PUBLISHED_SHA256=$(sha256sum "${public_json}" | awk '{print $1}')"

# --- 8. bounded storage ----------------------------------------------------
echo "=== 8. retention ==="
prune_args=(--artifact-root "${NFL_MODEL_ARTIFACT_ROOT}")
[ "${PRUNE_APPLY}" -eq 1 ] && prune_args+=(--apply)
"${PY}" scripts/prune_replaceable_artifacts.py "${prune_args[@]}"

# Both paths reach here now. snapshot_action describes what STAGE EXECUTION
# did, which is the thing that was or was not due; whether anything was
# published is a separate fact, reported by the publisher itself and by
# PUBLISHED_SHA256 above.
if [ "${due_batches}" -eq 0 ]; then
    echo "snapshot_action=NOOP_NOTHING_DUE"
else
    echo "snapshot_action=EXECUTED"
fi
echo "stage_snapshot_status=OK"
