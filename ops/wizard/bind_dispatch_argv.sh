#!/usr/bin/env bash
# Bind NFL production dispatch inputs as data for a fixed remote argument list.
#
# Sourced on the GitHub-hosted runner by .github/workflows/nfl_2026_production.yml.
# It is not an SSH command and ops/wizard/remote.sh does not copy it to the
# Wizard host. Nothing here opens a connection.
#
# Grammar, taken from the workflow input descriptions and the CLIs that
# consume the values. Invalid input fails closed. Nothing is trimmed, cased,
# or otherwise coerced into a nearby accepted value.
#
#   horizon
#     workflow_dispatch choice "" | TUE | FRI, and the only horizons
#     scripts/run_2026_production_card.py and create_official_capture.sh
#     accept on a certified run. Blank means "the due window decides" and is
#     not a remote --horizon word. A remote --horizon word must be TUE or FRI.
#   as_of
#     The --as-of form documented by scripts/resolve_production_schedule.py
#     and docs/PRODUCTION_RUNBOOK_2026.md: YYYY-MM-DDTHH:MM:SSZ.
#     pandas would accept dates, offsets, and other spellings and coerce them.
#     Those spellings are rejected here, before any remote command is built.
#   market_capture_manifest
#     An absolute path of an official BallDontLie capture manifest, the shape
#     scripts/capture_bdl_2026_asof.py writes and docs/WIZARD_NFL_PRODUCTION.md
#     records:
#       <absolute-prefix>/live-observation-log/balldontlie-2026/season=YYYY/week=NN/horizon=<TUE|FRI|STAGE|SMOKE>/capture=YYYYMMDDTHHMMSSZ/manifest.json
#     Traversal, a relative path, a newline, and any shell metacharacter are
#     rejected. The prefix is the absolute data root (the documented
#     $WIZARD_NFL_HOME/data, or the operator's real absolute root).
#   market_capture_sha256
#     64 hexadecimal characters, the declared manifest hash. Empty means the
#     flag is omitted. Case is preserved; it is not rewritten.
#
# After a value passes, printf %q must be a no-op on it. Only that word is
# placed in a name=value assignment. The remote shell program that follows
# is a fixed argument list; the assignment is data, not concatenated syntax.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "source ops/wizard/bind_dispatch_argv.sh; it is not a remote command" >&2
    exit 2
fi

_DISPATCH_MANIFEST_RE='^(/[A-Za-z0-9_-][A-Za-z0-9._-]*)+/live-observation-log/balldontlie-2026/season=[0-9]{4}/week=[0-9]{2}/horizon=(TUE|FRI|STAGE|SMOKE)/capture=[0-9]{8}T[0-9]{6}Z/manifest\.json$'

dispatch_fail() {
    local message="$1"
    if [ "$#" -ge 2 ]; then
        printf 'FAIL CLOSED: %s: %q\n' "$message" "$2" >&2
    else
        printf 'FAIL CLOSED: %s\n' "$message" >&2
    fi
    exit 1
}

validate_dispatch_horizon() {
    case "$1" in
        TUE|FRI) ;;
        *) dispatch_fail "horizon must be TUE or FRI" "$1" ;;
    esac
}

validate_dispatch_as_of() {
    local value="$1" round
    [[ "$value" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] \
        || dispatch_fail "as_of must be YYYY-MM-DDTHH:MM:SSZ" "$value"
    round="$(date -u -d "$value" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)" \
        || dispatch_fail "as_of is not a real UTC instant" "$value"
    [ "$round" = "$value" ] \
        || dispatch_fail "as_of is not a real UTC instant" "$value"
}

validate_dispatch_manifest() {
    local value="$1"
    case "$value" in
        *..*|*"//"*)
            dispatch_fail "manifest path contains traversal" "$value"
            ;;
    esac
    [[ "$value" =~ $_DISPATCH_MANIFEST_RE ]] \
        || dispatch_fail "manifest path is not an official capture manifest.json" "$value"
}

validate_dispatch_sha256() {
    [[ "$1" =~ ^[0-9a-fA-F]{64}$ ]] \
        || dispatch_fail "capture sha256 must be 64 hexadecimal characters" "$1"
}

# Print one `name=value` assignment for a value that has already passed its
# grammar. printf %q must reproduce the value unchanged; a value that needs
# bash quoting is rejected instead of being spliced into the remote script.
dispatch_checked_assignment() {
    local name="$1" value="$2" quoted
    case "$name" in
        horizon) validate_dispatch_horizon "$value" ;;
        as_of) validate_dispatch_as_of "$value" ;;
        manifest) validate_dispatch_manifest "$value" ;;
        capture_sha256) validate_dispatch_sha256 "$value" ;;
        *) dispatch_fail "refusing to bind an unknown remote word" ;;
    esac
    quoted="$(printf '%q' "$value")"
    if [ "$quoted" != "$value" ]; then
        dispatch_fail "${name} cannot be passed as one remote word" "$value"
    fi
    case "$value" in
        ''|*[!A-Za-z0-9._:/=-]*)
            dispatch_fail "${name} contains a shell metacharacter" "$value"
            ;;
    esac
    printf '%s=%s\n' "$name" "$value"
}
