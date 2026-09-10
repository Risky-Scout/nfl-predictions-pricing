#!/usr/bin/env bash
# DAILY: the every-day maintenance pass, executed over SSH on the Wizard
# server by .github/workflows/nfl_2026_production.yml.
#
# It produces NO public forecast and publishes nothing. There is deliberately
# no DAILY forecast horizon: public prediction cards are created only under
# the existing certified TUE/FRI semantics, by run_certified_card.sh.
#
#   1. refresh immutable 2026 completed-game/schedule evidence from
#      BallDontLie (append-only capture; no fitting, no pricing);
#   2. update the durable 2026 canonical games population from that evidence
#      (fail-closed on conflicting identities or changed scores) -- this is
#      also what makes completed games available to later eligible refits;
#   3. update prospective evaluation by attaching newly available results to
#      existing forecasts (never mutating a forecast) and report prospective
#      performance;
#   4. generate/update the recalibration candidate and report its promotion
#      state (promotion itself remains fail-closed).
#
# Each stage is idempotent and safe to repeat. Stage 1 is allowed to fail
# softly when BallDontLie is unreachable -- the rest of the pass still runs
# against the evidence already on disk -- but every other stage failing fails
# the pass.
#
# Usage (on the server):
#   bash run_daily_maintenance.sh [--skip-capture]
set -euo pipefail

SKIP_CAPTURE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-capture) SKIP_CAPTURE=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ops/wizard/nfl_production_layout.sh
. "${HERE}/nfl_production_layout.sh"

PY="${WIZARD_NFL_VENV_DIR}/bin/python"
cd "${WIZARD_NFL_REPO_DIR}"

capture_status="SKIPPED"
if [ "${SKIP_CAPTURE}" -eq 0 ]; then
    echo "=== 1. refresh 2026 BallDontLie games evidence ==="
    if "${PY}" scripts/refresh_bdl_2026_games_evidence.py --season 2026; then
        capture_status="OK"
    else
        # A provider outage must not take down population maintenance,
        # prospective evaluation or candidate generation, all of which work
        # from evidence already captured. It is reported, not hidden.
        capture_status="PROVIDER_UNAVAILABLE"
        echo "WARNING: BallDontLie games-evidence refresh failed; continuing with existing evidence" >&2
    fi
fi
echo "games_evidence_refresh=${capture_status}"

echo "=== 2. update durable 2026 canonical games population ==="
population_json="${WIZARD_NFL_LOG_DIR}/population-$(date -u +%Y%m%dT%H%M%SZ).json"
set +e
"${PY}" scripts/update_2026_games_population.py --latest-games-evidence > "${population_json}"
population_exit=$?
set -e
cat "${population_json}" 2>/dev/null || true
if [ "${population_exit}" -eq 2 ]; then
    echo "FAIL CLOSED: the 2026 games population update rejected the available evidence" >&2
    exit 3
fi
# Exit 1 is "no COMPLETE evidence found yet" -- legitimate before the first
# successful capture, and not a failure of the daily pass.
echo "games_population_update_exit=${population_exit}"

echo "=== 3. update prospective evaluation ==="
"${PY}" scripts/attach_2026_results_from_population.py
"${PY}" scripts/report_2026_prospective_performance.py || true

echo "=== 4. recalibration candidate ==="
set +e
"${PY}" scripts/generate_2026_recalibration_candidate.py
recalibration_exit=$?
set -e
if [ "${recalibration_exit}" -ne 0 ]; then
    # Candidate GENERATION can legitimately fail before enough labelled 2026
    # evidence exists. The certified calibrator stays active either way, so the
    # pass reports the state instead of failing -- but it always reports it.
    echo "WARNING: recalibration candidate generation failed (exit ${recalibration_exit}); reporting state only" >&2
    "${PY}" scripts/generate_2026_recalibration_candidate.py --report-only
fi
echo "recalibration_candidate_exit=${recalibration_exit}"

echo "daily_maintenance_status=OK"
