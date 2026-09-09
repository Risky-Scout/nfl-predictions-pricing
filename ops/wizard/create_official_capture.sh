#!/usr/bin/env bash
# Create ONE official BallDontLie pregame capture for the certified horizon
# that is due, on the Wizard server.
#
# It captures data and nothing else: no fit, no price, no calibration, no
# export, no publication. The capture directory it creates is immutable and
# append-only -- this script never touches an existing capture.
#
# The card identity (season, week, season_type) is RESOLVED, never guessed:
# scripts/resolve_current_card_identity.py derives it from the certified
# card-scoped cutoff plus the composite games population. A wrong week here
# would be rejected downstream anyway (the certified market path cross-checks
# the capture's recorded identity against the run's own), so resolving it
# correctly is what makes an unattended capture possible at all.
#
# Emits one machine-readable line the workflow parses:
#   OFFICIAL_CAPTURE_MANIFEST=<absolute path>
#   OFFICIAL_CAPTURE_SHA256=<hex>
#
# Usage (on the server):
#   bash create_official_capture.sh --horizon TUE [--as-of ISO8601]
set -euo pipefail

HORIZON=""
AS_OF=""
while [ $# -gt 0 ]; do
    case "$1" in
        --horizon) HORIZON="$2"; shift 2 ;;
        --as-of)   AS_OF="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${HORIZON}" in
    TUE|FRI) ;;
    *) echo "FAIL CLOSED: --horizon must be TUE or FRI; got '${HORIZON}'" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ops/wizard/nfl_production_layout.sh
. "${HERE}/nfl_production_layout.sh"

PY="${WIZARD_NFL_VENV_DIR}/bin/python"
cd "${WIZARD_NFL_REPO_DIR}"

as_of_args=()
[ -n "${AS_OF}" ] && as_of_args=(--as-of "${AS_OF}")

card_json="$("${PY}" scripts/resolve_current_card_identity.py --horizon "${HORIZON}" "${as_of_args[@]}")"
printf '%s\n' "${card_json}"

season="$(printf '%s' "${card_json}" | "${PY}" -c 'import json,sys; print(json.load(sys.stdin)["season"])')"
week="$(printf '%s' "${card_json}" | "${PY}" -c 'import json,sys; print(json.load(sys.stdin)["week"])')"
season_type="$(printf '%s' "${card_json}" | "${PY}" -c 'import json,sys; print(json.load(sys.stdin)["season_type"])')"
cutoff="$(printf '%s' "${card_json}" | "${PY}" -c 'import json,sys; print(json.load(sys.stdin)["target_cutoff_utc"])')"

capture_json="$("${PY}" scripts/capture_bdl_2026_asof.py \
    --season "${season}" \
    --week "${week}" \
    --season-type "${season_type}" \
    --horizon "${HORIZON}" \
    --nominal-cutoff "${cutoff}")"
printf '%s\n' "${capture_json}"

# scripts/capture_bdl_2026_asof.py reports human-readable lines, not JSON; the
# capture directory is the one authoritative value needed from it.
capture_dir="$(printf '%s\n' "${capture_json}" | sed -n 's/^capture directory *: *//p' | tail -n 1)"
if [ -z "${capture_dir}" ]; then
    echo "FAIL CLOSED: the capture reported no capture_dir" >&2
    exit 3
fi
manifest="${capture_dir%/}/manifest.json"
if [ ! -f "${manifest}" ]; then
    echo "FAIL CLOSED: the capture wrote no manifest at ${manifest}" >&2
    exit 3
fi

echo "OFFICIAL_CAPTURE_MANIFEST=${manifest}"
echo "OFFICIAL_CAPTURE_SHA256=$(sha256sum "${manifest}" | awk '{print $1}')"
