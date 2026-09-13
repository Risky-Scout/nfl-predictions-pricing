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
#   4. generate/update the recalibration candidate, then automatically promote
#      it when the versioned operational promotion policy and the frozen
#      preregistered maturity firewall both allow it. Before that firewall is
#      satisfied the decision is NOT_YET_MATURE, which is a success.
#
# Each stage is idempotent and safe to repeat. Stage 1 is allowed to fail
# softly when BallDontLie is unreachable -- the rest of the pass still runs
# against the evidence already on disk -- but every other stage failing fails
# the pass.
#
# Stage 4 in particular fails the pass on ANY non-zero exit, and the pass
# prints daily_maintenance_status=OK only when every stage succeeded. The
# distinction that matters for an unattended season is between a legitimate
# no-op and a missing input: NOT_YET_MATURE is a success, whereas "no candidate
# could be generated" means the historical estate, the certified baseline seed
# or the active pointer is missing or corrupt, and that must be loud.
# Pass exit codes are distinct so a scheduler log says what broke:
#   exit 3  the 2026 games population rejected the available evidence
#   exit 4  promotion policy / policy-lock / candidate integrity violation
#   exit 5  candidate generation failed (e.g. required historical estate absent)
#   exit 6  the artifact root could not be resolved
#   exit 7  the active calibrator could not be resolved (baseline seed missing,
#           or a corrupt active pointer)
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

echo "=== 4. recalibration candidate + automatic promotion when eligible ==="
set +e
"${PY}" scripts/generate_2026_recalibration_candidate.py --promote-if-eligible
recalibration_exit=$?
set -e
echo "recalibration_candidate_exit=${recalibration_exit}"
# Exit 0 is the ONLY healthy outcome, and it already covers every legitimate
# steady state: the candidate was generated, and promotion decided PROMOTED,
# NOT_YET_MATURE, ALREADY_ACTIVE, NO_CANDIDATE or POLICY_DISABLED. A
# NOT_YET_MATURE day is a success and must stay one.
#
# Every other exit code is a FAIL CLOSED. An unattended season cannot treat
# "production could not build a recalibration candidate at all" as a healthy
# day: that is precisely the state where the required historical estate, the
# certified baseline seed or the active pointer is missing or corrupt, and
# silently continuing would let production drift for weeks without anyone
# seeing it. The state is always re-reported before exiting so the failure
# arrives with its own evidence.
if [ "${recalibration_exit}" -ne 0 ]; then
    # Remapped onto codes the pass owns, so stage 4's failures never collide
    # with stage 2's population rejection (exit 3).
    case "${recalibration_exit}" in
        1) reason="candidate generation itself failed (required historical estate missing, or an unexpected exception)"
           pass_exit=5 ;;
        2) reason="the artifact root could not be resolved"
           pass_exit=6 ;;
        3) reason="the active calibrator could not be resolved (certified baseline seed missing, or a corrupt active pointer)"
           pass_exit=7 ;;
        4) reason="promotion policy, policy-lock or candidate integrity violation"
           pass_exit=4 ;;
        *) reason="unexpected recalibration failure"
           pass_exit=8 ;;
    esac
    echo "FAIL CLOSED: ${reason} (candidate script exit ${recalibration_exit})" >&2
    "${PY}" scripts/generate_2026_recalibration_candidate.py --report-only || true
    echo "daily_maintenance_status=FAIL_CLOSED"
    exit "${pass_exit}"
fi

echo "daily_maintenance_status=OK"
