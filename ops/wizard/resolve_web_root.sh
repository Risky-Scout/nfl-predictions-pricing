#!/usr/bin/env bash
# Discover, on the Wizard server, the filesystem directory nginx serves the
# public NFL predictions path (WIZARD_NFL_PUBLIC_URL) from, and record it so
# publication never has to guess.
#
# READ-ONLY except for the one recorded path under $WIZARD_NFL_STATE_DIR.
# It changes no nginx configuration, restarts nothing, creates no web
# directory and publishes nothing.
#
# RESOLUTION ORDER (first hit wins, and every step is reported):
#
#   1. $WIZARD_NFL_WEB_DIR, if the operator/environment already supplied it.
#      An explicit answer always beats a discovered one.
#   2. The live nginx configuration: `nginx -T` (falling back to reading
#      /etc/nginx) is parsed for `root`/`alias` directives, and each candidate
#      root is joined with the public URL path. This is the authoritative
#      source when it is readable.
#   3. A filesystem search for an ALREADY-DEPLOYED
#      `tools/odds-scanner/predictions/NFL/index.html`. The NFL HTML shell is
#      already present on the server (deployed over FTP), so the directory
#      containing it is the served directory by construction. Searched only
#      under conventional web roots, never `/`.
#
# If nothing resolves, this script exits non-zero and records nothing:
# publication must fail closed, never invent a destination, and never create
# a new directory tree that nginx does not serve.
#
# Usage (on the server):
#   bash ops/wizard/resolve_web_root.sh            # resolve + record
#   bash ops/wizard/resolve_web_root.sh --print    # resolve + print, no write
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ops/wizard/nfl_production_layout.sh
. "${HERE}/nfl_production_layout.sh"

PRINT_ONLY=0
[ "${1:-}" = "--print" ] && PRINT_ONLY=1

# The logical URL path, without leading/trailing slashes. Derived from the
# public URL in the layout file so there is one definition of it.
URL_PATH="$(printf '%s' "${WIZARD_NFL_PUBLIC_URL}" | sed -E 's#^https?://[^/]+/##; s#/+$##')"

CONVENTIONAL_ROOTS="
/var/www
/var/www/html
/usr/share/nginx/html
/srv/www
/srv/http
/home
"

resolution_method=""
resolved=""

is_candidate() {
    # A candidate is a directory that already contains the deployed NFL page.
    # Requiring index.html (rather than mere existence) is what stops a
    # coincidentally-named empty directory from being accepted as the served
    # path.
    [ -d "$1" ] && [ -f "$1/index.html" ]
}

# --- 1. explicit override ---------------------------------------------------
if [ -n "${WIZARD_NFL_WEB_DIR:-}" ]; then
    resolved="${WIZARD_NFL_WEB_DIR}"
    resolution_method="EXPLICIT_OVERRIDE"
fi

# --- 2. live nginx configuration -------------------------------------------
if [ -z "${resolved}" ]; then
    nginx_dump=""
    if command -v nginx >/dev/null 2>&1; then
        nginx_dump="$(nginx -T 2>/dev/null || true)"
    fi
    if [ -z "${nginx_dump}" ] && [ -d /etc/nginx ]; then
        nginx_dump="$(grep -rhoE '^[[:space:]]*(root|alias)[[:space:]]+[^;]+;' /etc/nginx 2>/dev/null || true)"
    fi
    if [ -n "${nginx_dump}" ]; then
        while read -r candidate_root; do
            [ -z "${candidate_root}" ] && continue
            candidate="${candidate_root%/}/${URL_PATH}"
            if is_candidate "${candidate}"; then
                resolved="${candidate}"
                resolution_method="NGINX_CONFIGURATION"
                break
            fi
        done <<EOF
$(printf '%s\n' "${nginx_dump}" \
    | grep -E '^[[:space:]]*(root|alias)[[:space:]]+' \
    | sed -E 's/^[[:space:]]*(root|alias)[[:space:]]+//; s/;[[:space:]]*$//; s/^"//; s/"$//' \
    | sort -u)
EOF
    fi
fi

# --- 3. already-deployed page ----------------------------------------------
if [ -z "${resolved}" ]; then
    for base in ${CONVENTIONAL_ROOTS}; do
        [ -d "${base}" ] || continue
        candidate="${base%/}/${URL_PATH}"
        if is_candidate "${candidate}"; then
            resolved="${candidate}"
            resolution_method="CONVENTIONAL_ROOT"
            break
        fi
    done
fi

if [ -z "${resolved}" ]; then
    for base in ${CONVENTIONAL_ROOTS}; do
        [ -d "${base}" ] || continue
        found="$(find "${base}" -maxdepth 8 -type d -path "*/${URL_PATH}" 2>/dev/null | sort | head -n 1 || true)"
        if [ -n "${found}" ] && is_candidate "${found}"; then
            resolved="${found}"
            resolution_method="DEPLOYED_PAGE_SEARCH"
            break
        fi
    done
fi

if [ -z "${resolved}" ]; then
    cat >&2 <<EOF
FAIL CLOSED: could not resolve the nginx-served directory for /${URL_PATH}.
Nothing was recorded and nothing was created. Publication is refused rather
than writing a public card into a path nginx does not serve.

Resolve this by either:
  * setting the WIZARD_NFL_WEB_DIR environment secret in the wizard-production
    GitHub environment to the exact served directory, or
  * running this script on the server with WIZARD_NFL_WEB_DIR exported once so
    it is recorded at ${WIZARD_NFL_WEB_ROOT_RECORD}.
EOF
    exit 2
fi

printf 'WIZARD_NFL_WEB_DIR=%s\n' "${resolved}"
printf 'WIZARD_NFL_WEB_DIR_RESOLUTION=%s\n' "${resolution_method}"
printf 'WIZARD_NFL_WEB_DIR_WRITABLE=%s\n' "$([ -w "${resolved}" ] && echo yes || echo no)"

if [ "${PRINT_ONLY}" -eq 0 ]; then
    mkdir -p "${WIZARD_NFL_STATE_DIR}"
    printf '%s' "${resolved}" > "${WIZARD_NFL_WEB_ROOT_RECORD}.tmp"
    mv "${WIZARD_NFL_WEB_ROOT_RECORD}.tmp" "${WIZARD_NFL_WEB_ROOT_RECORD}"
    printf 'RECORDED=%s\n' "${WIZARD_NFL_WEB_ROOT_RECORD}"
fi
