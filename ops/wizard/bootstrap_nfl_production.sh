#!/usr/bin/env bash
# Idempotent bootstrap of the NFL 2026 production installation on the Wizard
# server. Safe to run on every workflow invocation: a fully provisioned host
# converges to the same state and reports IDEMPOTENT, and nothing here deletes
# data, artifacts, live captures or the published web card.
#
# What it does, in order:
#   1. creates the NFL-only directory layout (ops/wizard/nfl_production_layout.sh)
#      and PROVES each directory is writable rather than assuming it;
#   2. installs the application source at an EXACT commit, from an archive the
#      workflow uploaded over the already-proven SSH channel;
#   3. creates/updates the production Python virtualenv from THIS repository's
#      own dependency declaration (never an NCAAF environment);
#   4. reports the layout, the installed commit and the resolved web directory.
#
# What it deliberately does NOT do: touch nginx, create the web directory,
# publish anything, run the model, fetch live data, or write any secret to
# disk.
#
# SOURCE IS PUSHED, NOT PULLED
#   The server never clones from GitHub. `git archive` on the GitHub-hosted
#   runner produces the exact tree of the commit the workflow was triggered
#   on, and that archive is copied in over SSH. So the server needs no GitHub
#   deploy key, no personal access token and no network egress to github.com,
#   and the code that runs in production is provably the code the workflow
#   ran on -- not a branch tip re-resolved later on the server.
#
# Usage (executed over SSH by .github/workflows/nfl_2026_production.yml):
#   bash bootstrap_nfl_production.sh --commit <sha> --source-archive <path.tar.gz>
set -euo pipefail

COMMIT=""
SOURCE_ARCHIVE=""
SKIP_VENV=0

while [ $# -gt 0 ]; do
    case "$1" in
        --commit)         COMMIT="$2"; shift 2 ;;
        --source-archive) SOURCE_ARCHIVE="$2"; shift 2 ;;
        --skip-venv)      SKIP_VENV=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ops/wizard/nfl_production_layout.sh
. "${HERE}/nfl_production_layout.sh"

echo "=== NFL 2026 production bootstrap ==="
wizard_nfl_print_layout

# --- 1. layout --------------------------------------------------------------
for dir in \
    "${WIZARD_NFL_HOME}" \
    "${WIZARD_NFL_DATA_DIR}" \
    "${WIZARD_NFL_ARTIFACT_DIR}" \
    "${WIZARD_NFL_LIVE_DIR}" \
    "${WIZARD_NFL_STAGING_DIR}" \
    "${WIZARD_NFL_STATE_DIR}" \
    "${WIZARD_NFL_LOG_DIR}"
do
    mkdir -p "${dir}"
    # Prove writability with a real write. `[ -w ]` can disagree with reality
    # on a read-only mount or under a restrictive ACL, and an unwritable
    # artifact root must fail here rather than half-way through a certified
    # production run.
    probe="${dir}/.bootstrap_write_probe"
    if ! : > "${probe}" 2>/dev/null; then
        echo "FAIL CLOSED: ${dir} is not writable by $(id -un)" >&2
        exit 3
    fi
    rm -f "${probe}"
    echo "layout_ok=${dir}"
done

# NFL data/artifacts are private production evidence, not world-readable
# content. 700 is applied to the roots only; existing contents are untouched.
chmod 700 "${WIZARD_NFL_DATA_DIR}" "${WIZARD_NFL_ARTIFACT_DIR}" "${WIZARD_NFL_LIVE_DIR}" 2>/dev/null || true

# --- 2. application source ---------------------------------------------------
if [ -n "${SOURCE_ARCHIVE}" ]; then
    if [ ! -f "${SOURCE_ARCHIVE}" ]; then
        echo "FAIL CLOSED: source archive not found: ${SOURCE_ARCHIVE}" >&2
        exit 4
    fi
    if [ -z "${COMMIT}" ]; then
        echo "FAIL CLOSED: --source-archive requires --commit so the installed tree is identifiable" >&2
        exit 4
    fi

    staged="${WIZARD_NFL_HOME}/.repo.incoming.$$"
    rm -rf "${staged}"
    mkdir -p "${staged}"
    tar -xzf "${SOURCE_ARCHIVE}" -C "${staged}"
    if [ ! -f "${staged}/pyproject.toml" ]; then
        rm -rf "${staged}"
        echo "FAIL CLOSED: the source archive does not contain pyproject.toml at its root" >&2
        exit 4
    fi
    printf '%s' "${COMMIT}" > "${staged}/.deployed_commit"

    # Swap the whole tree rather than overlaying: an overlay would leave a
    # deleted file from a previous revision behind, and a stale script in the
    # production checkout is exactly the kind of drift this bootstrap exists
    # to prevent. The application checkout holds no generated state (data,
    # artifacts and live captures all live in their own roots), so replacing
    # it wholesale destroys nothing.
    previous="${WIZARD_NFL_HOME}/.repo.previous.$$"
    rm -rf "${previous}"
    if [ -d "${WIZARD_NFL_REPO_DIR}" ]; then
        mv "${WIZARD_NFL_REPO_DIR}" "${previous}"
    fi
    if ! mv "${staged}" "${WIZARD_NFL_REPO_DIR}"; then
        [ -d "${previous}" ] && mv "${previous}" "${WIZARD_NFL_REPO_DIR}"
        echo "FAIL CLOSED: could not install the source tree at ${WIZARD_NFL_REPO_DIR}" >&2
        exit 4
    fi
    rm -rf "${previous}"
    echo "source=INSTALLED_FROM_ARCHIVE"
elif [ -d "${WIZARD_NFL_REPO_DIR}" ]; then
    echo "source=EXISTING"
else
    echo "FAIL CLOSED: no source at ${WIZARD_NFL_REPO_DIR} and no --source-archive given" >&2
    exit 4
fi

if [ -f "${WIZARD_NFL_REPO_DIR}/.deployed_commit" ]; then
    echo "source_commit=$(cat "${WIZARD_NFL_REPO_DIR}/.deployed_commit")"
fi
echo "source_dir=${WIZARD_NFL_REPO_DIR}"

# --- 3. python environment --------------------------------------------------
if [ "${SKIP_VENV}" -eq 0 ] && [ -d "${WIZARD_NFL_REPO_DIR}" ]; then
    if [ ! -x "${WIZARD_NFL_VENV_DIR}/bin/python" ]; then
        python3 -m venv "${WIZARD_NFL_VENV_DIR}"
        echo "venv=CREATED"
    else
        echo "venv=EXISTING"
    fi
    # This repository's OWN dependency declaration. There is no reference to
    # any NCAAF virtualenv or requirements file anywhere in this path.
    "${WIZARD_NFL_VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip setuptools wheel
    # Editable, so the swap in step 2 takes effect without a reinstall: the
    # install path never changes, only the tree behind it.
    "${WIZARD_NFL_VENV_DIR}/bin/python" -m pip install --quiet -e "${WIZARD_NFL_REPO_DIR}[data]"
    echo "venv_python=$("${WIZARD_NFL_VENV_DIR}/bin/python" -V 2>&1)"
    echo "venv_dir=${WIZARD_NFL_VENV_DIR}"
fi

# --- 4. web directory (resolution only; never created here) -----------------
if bash "${HERE}/resolve_web_root.sh"; then
    echo "web_root=RESOLVED"
else
    # Not fatal for bootstrap: the data/artifact/runtime installation is still
    # valid and useful. Publication itself fails closed separately.
    echo "web_root=UNRESOLVED (publication will fail closed until this is set)"
fi

echo "bootstrap_status=OK"
