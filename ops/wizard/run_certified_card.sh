#!/usr/bin/env bash
# CERTIFIED TUE/FRI: run one certified prediction card end to end on the Wizard
# server, export it and archive it. It does NOT write the public endpoint --
# see stage 4.
#
# Executed over SSH by the certified job of
# .github/workflows/nfl_2026_production.yml. It is an ORCHESTRATOR ONLY: every
# step below shells out to an existing, certified entrypoint. It contains no
# model logic, no feature construction, no calibration, no market rule and no
# schema.
#
#   1. verify the official BallDontLie capture manifest exists, self-verifies
#      its own recorded content hash, and resolves the hash the caller declared
#      to one of its two integrity objects (both retained in the run record);
#   2. production preflight against THAT capture, at the SAME as-of instant
#      the card will be priced at -- must report READY;
#   3. the certified six-Elo / Ridge-alpha-100 card for the horizon, priced
#      from that capture, via scripts/run_2026_production_card.py;
#   4. export the frozen wizard-nfl-pricing-v2 contract for the run and write
#      its immutable archive.
#
# THERE IS NO PUBLICATION STAGE. The public NFL endpoint serves exactly one
# contract -- the current-week pregame board written by run_stage_snapshots.sh --
# and this orchestrator is not a public writer. Stage 4 explains why.
#
# Every stage fails closed: a hash mismatch, a preflight that is not READY, a
# non-SUCCESS batch or a failed export aborts, leaving the archive and the
# published board untouched.
#
# Usage (on the server):
#   bash run_certified_card.sh --horizon TUE \
#       --capture-manifest /path/to/manifest.json \
#       --capture-sha256 <hex> \
#       [--as-of ISO8601] [--dry-run-publish]
#
# --dry-run-publish is still ACCEPTED and does nothing: there is no publication
# left for it to withhold. It is kept so an existing caller -- including this
# workflow's own dry-run path -- is not failed by an unknown argument.
set -euo pipefail

HORIZON=""
CAPTURE_MANIFEST=""
CAPTURE_SHA256=""
AS_OF=""

while [ $# -gt 0 ]; do
    case "$1" in
        --horizon)           HORIZON="$2"; shift 2 ;;
        --capture-manifest)  CAPTURE_MANIFEST="$2"; shift 2 ;;
        --capture-sha256)    CAPTURE_SHA256="$2"; shift 2 ;;
        --as-of)             AS_OF="$2"; shift 2 ;;
        # Accepted and inert -- see the usage note above.
        --dry-run-publish)   shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ops/wizard/nfl_production_layout.sh
. "${HERE}/nfl_production_layout.sh"

PY="${WIZARD_NFL_VENV_DIR}/bin/python"
cd "${WIZARD_NFL_REPO_DIR}"

case "${HORIZON}" in
    TUE|FRI) ;;
    *) echo "FAIL CLOSED: --horizon must be TUE or FRI (there is no DAILY forecast horizon); got '${HORIZON}'" >&2; exit 2 ;;
esac
if [ -z "${CAPTURE_MANIFEST}" ]; then
    echo "FAIL CLOSED: --capture-manifest is required; a capture is never auto-selected" >&2
    exit 2
fi

# --- 1. capture identity ----------------------------------------------------
echo "=== 1. official capture ==="
if [ ! -f "${CAPTURE_MANIFEST}" ]; then
    echo "FAIL CLOSED: capture manifest not found on the server: ${CAPTURE_MANIFEST}" >&2
    exit 3
fi
# A capture manifest has TWO integrity objects that are different by
# construction -- the file hash, and the content hash the capture stores inside
# itself as .manifest_sha256 (see scripts/verify_capture_manifest_integrity.py).
# The verifier ALWAYS self-verifies the content hash, which is what detects a
# mutated capture, and resolves a declared hash to whichever object it is, so a
# correctly recorded value is never rejected for referring to the other object
# and an unrecognised value is never accepted.
set +e
"${PY}" scripts/verify_capture_manifest_integrity.py \
    "${CAPTURE_MANIFEST}" \
    ${CAPTURE_SHA256:+--declared-sha256 "${CAPTURE_SHA256}"}
capture_integrity_exit=$?
set -e
if [ "${capture_integrity_exit}" -ne 0 ]; then
    echo "FAIL CLOSED: capture manifest integrity check failed for ${CAPTURE_MANIFEST}" >&2
    exit 3
fi
# The manifest is READ here and never rewritten: the capture is immutable
# evidence and this run only points at it.

# --- 2. preflight -- must be READY -----------------------------------------
# ONE as-of for the whole run. Preflight cross-checks the capture against the
# certified cutoff it resolves from this instant, so a replay that passes
# --as-of to the card but not to preflight would have preflight judge the
# capture against the CURRENT card's cutoff and reject a correct capture as an
# unregistered live source. Both stages must be asked about the same instant,
# so the value is built once here and reused verbatim below.
as_of_args=()
[ -n "${AS_OF}" ] && as_of_args=(--as-of "${AS_OF}")

echo "=== 2. preflight ==="
preflight_json="${WIZARD_NFL_LOG_DIR}/preflight-$(date -u +%Y%m%dT%H%M%SZ).json"
set +e
"${PY}" scripts/run_2026_production_card.py \
    --preflight \
    --horizon "${HORIZON}" \
    --market-capture-manifest "${CAPTURE_MANIFEST}" \
    "${as_of_args[@]}" \
    > "${preflight_json}"
preflight_exit=$?
set -e
cat "${preflight_json}"
overall="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("overall_status"))' "${preflight_json}")"
echo "preflight_overall_status=${overall}"
if [ "${preflight_exit}" -ne 0 ] || [ "${overall}" != "READY" ]; then
    echo "FAIL CLOSED: preflight is '${overall}' (exit ${preflight_exit}); READY is required before a public card" >&2
    exit 4
fi

# --- 3. certified card ------------------------------------------------------
echo "=== 3. certified ${HORIZON} card ==="
run_json="${WIZARD_NFL_LOG_DIR}/run-${HORIZON}-$(date -u +%Y%m%dT%H%M%SZ).json"
set +e
"${PY}" scripts/run_2026_production_card.py \
    --horizon "${HORIZON}" \
    --market-capture-manifest "${CAPTURE_MANIFEST}" \
    "${as_of_args[@]}" \
    > "${run_json}"
run_exit=$?
set -e
cat "${run_json}"
run_status="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${run_json}")"
run_id="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "${run_json}")"
echo "run_status=${run_status}"
echo "run_id=${run_id}"
if [ "${run_exit}" -ne 0 ] || [ "${run_status}" != "SUCCESS" ]; then
    echo "FAIL CLOSED: certified batch status '${run_status}' (exit ${run_exit}); nothing is exported or published" >&2
    exit 5
fi

# --- 4. export and archive wizard-nfl-pricing-v2 ----------------------------
#
# The certified card is EXPORTED AND ARCHIVED, NOT PUBLISHED.
#
# export_wizard_nfl_pricing.py writes the immutable archive at
# archive/season=<YYYY>/week=<NN>/horizon=<TUE|FRI>.json first and refuses to
# overwrite a differing one, so this step is the audit record of the certified
# card and is unchanged. It also writes the card into the ARTIFACT tree, which
# is a staging path, not the served directory.
#
# What is gone is the public publication that used to follow. A certified card
# covers one cutoff for the whole slate; the public product is the current-week
# pregame board, where each game is shown at the latest pregame snapshot it has
# reached. Publishing the certified card replaced that board with a
# single-cutoff card twice a week and dropped every game that was being shown
# from OPEN or MID, until the next snapshot sweep happened to restore it. The
# board assembler in run_stage_snapshots.sh is now the sole public writer, and
# scripts/publish_wizard_nfl_local.py refuses a non-authoritative contract, so
# this orchestrator could not perform that downgrade even if it tried.
#
# Nothing else about the certified path changes: the same capture, the same
# preflight, the same six-Elo / Ridge-alpha-100 card, the same forecast ledger
# the OPEN/MID/CLOSE stage machinery reads, the same archive.
echo "=== 4. export + archive wizard-nfl-pricing-v2 ==="
run_manifest="${NFL_MODEL_ARTIFACT_ROOT}/production-2026/run-manifests/${run_id}.json"
forecast_dir="${NFL_MODEL_ARTIFACT_ROOT}/production-2026/forecast-ledger/${HORIZON}"
certified_json="${NFL_MODEL_ARTIFACT_ROOT}/public/wizardofodds/nfl-pricing/latest.json"
"${PY}" scripts/export_wizard_nfl_pricing.py \
    --run-manifest "${run_manifest}" \
    --forecast-dir "${forecast_dir}" \
    --output "${certified_json}"
echo "exported=${certified_json}"
echo "exported_sha256=$(sha256sum "${certified_json}" | awk '{print $1}')"

# The machine-readable lines the workflow parses. There is deliberately no
# PUBLISHED_SHA256: this orchestrator publishes nothing, and the public bytes
# are proved by the snapshot sweep that does.
echo "CERTIFIED_RUN_ID=${run_id}"
echo "CERTIFIED_HORIZON=${HORIZON}"
echo "certified_card_published=NO_PUBLIC_WRITE_BY_DESIGN"
echo "certified_card_status=OK"
