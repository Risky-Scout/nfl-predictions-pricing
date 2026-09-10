#!/usr/bin/env bash
# CERTIFIED TUE/FRI: run one public prediction card end to end on the Wizard
# server, then publish it atomically.
#
# Executed over SSH by the certified job of
# .github/workflows/nfl_2026_production.yml. It is an ORCHESTRATOR ONLY: every
# step below shells out to an existing, certified entrypoint. It contains no
# model logic, no feature construction, no calibration, no market rule and no
# schema.
#
#   1. verify the official BallDontLie capture manifest exists and hashes to
#      exactly what the caller declared (retained verbatim in the run record);
#   2. production preflight against THAT capture -- must report READY;
#   3. the certified six-Elo / Ridge-alpha-100 card for the horizon, priced
#      from that capture, via scripts/run_2026_production_card.py;
#   4. export the frozen wizard-nfl-pricing-v2 contract for the run;
#   5. publish it atomically into the nginx-served NFL directory over the
#      local filesystem (never FTP).
#
# Every stage fails closed: a hash mismatch, a preflight that is not READY, a
# non-SUCCESS batch, a failed export or a failed validation aborts before
# anything is published, leaving the currently published card untouched.
#
# Usage (on the server):
#   bash run_certified_card.sh --horizon TUE \
#       --capture-manifest /path/to/manifest.json \
#       --capture-sha256 <hex> \
#       [--as-of ISO8601] [--dry-run-publish]
set -euo pipefail

HORIZON=""
CAPTURE_MANIFEST=""
CAPTURE_SHA256=""
AS_OF=""
DRY_RUN_PUBLISH=0

while [ $# -gt 0 ]; do
    case "$1" in
        --horizon)           HORIZON="$2"; shift 2 ;;
        --capture-manifest)  CAPTURE_MANIFEST="$2"; shift 2 ;;
        --capture-sha256)    CAPTURE_SHA256="$2"; shift 2 ;;
        --as-of)             AS_OF="$2"; shift 2 ;;
        --dry-run-publish)   DRY_RUN_PUBLISH=1; shift ;;
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
actual_sha="$(sha256sum "${CAPTURE_MANIFEST}" | awk '{print $1}')"
echo "capture_manifest=${CAPTURE_MANIFEST}"
echo "capture_manifest_sha256=${actual_sha}"
if [ -n "${CAPTURE_SHA256}" ] && [ "${actual_sha}" != "${CAPTURE_SHA256}" ]; then
    echo "FAIL CLOSED: capture manifest sha256 ${actual_sha} != declared ${CAPTURE_SHA256}" >&2
    exit 3
fi
# The manifest is READ here and never rewritten: the capture is immutable
# evidence and this run only points at it.

# --- 2. preflight -- must be READY -----------------------------------------
echo "=== 2. preflight ==="
preflight_json="${WIZARD_NFL_LOG_DIR}/preflight-$(date -u +%Y%m%dT%H%M%SZ).json"
set +e
"${PY}" scripts/run_2026_production_card.py \
    --preflight \
    --horizon "${HORIZON}" \
    --market-capture-manifest "${CAPTURE_MANIFEST}" \
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
as_of_args=()
[ -n "${AS_OF}" ] && as_of_args=(--as-of "${AS_OF}")
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

# --- 4. export wizard-nfl-pricing-v2 ---------------------------------------
echo "=== 4. export wizard-nfl-pricing-v2 ==="
run_manifest="${NFL_MODEL_ARTIFACT_ROOT}/production-2026/run-manifests/${run_id}.json"
forecast_dir="${NFL_MODEL_ARTIFACT_ROOT}/production-2026/forecast-ledger/${HORIZON}"
public_json="${NFL_MODEL_ARTIFACT_ROOT}/public/wizardofodds/nfl-pricing/latest.json"
"${PY}" scripts/export_wizard_nfl_pricing.py \
    --run-manifest "${run_manifest}" \
    --forecast-dir "${forecast_dir}" \
    --output "${public_json}"
echo "exported=${public_json}"
echo "exported_sha256=$(sha256sum "${public_json}" | awk '{print $1}')"

# --- 5. atomic publication --------------------------------------------------
echo "=== 5. publish ==="
publish_args=(--json "${public_json}")
[ "${DRY_RUN_PUBLISH}" -eq 1 ] && publish_args+=(--dry-run)
"${PY}" scripts/publish_wizard_nfl_local.py "${publish_args[@]}"

# The single machine-readable line the workflow parses to verify the PUBLIC
# site is serving exactly this card.
echo "PUBLISHED_SHA256=$(sha256sum "${public_json}" | awk '{print $1}')"
echo "PUBLISHED_RUN_ID=${run_id}"
echo "PUBLISHED_HORIZON=${HORIZON}"
echo "certified_card_status=OK"
