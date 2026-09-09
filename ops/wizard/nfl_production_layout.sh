#!/usr/bin/env bash
# The ONE definition of the NFL 2026 production layout on the Wizard server.
#
# Sourced (never executed) by every other ops script and by every remote step
# of .github/workflows/nfl_2026_production.yml, so the bootstrap, the model
# run and the publication step cannot drift into three different opinions
# about where anything lives.
#
# ---------------------------------------------------------------------------
# WHY EVERYTHING HANGS OFF $HOME BY DEFAULT
# ---------------------------------------------------------------------------
# Nothing here assumes /opt or /srv is writable. The Wizard SSH account is a
# restricted deployment account (the proven wizard_ssh_smoke.yml run showed
# only that it can log in and read); a bootstrap that begins with `mkdir -p
# /opt/...` would fail on the very first automated run. Every path is therefore
# rooted at WIZARD_NFL_HOME, which defaults to a single directory inside the
# SSH user's own home and is overridable end-to-end by exporting
# WIZARD_NFL_HOME before sourcing this file. bootstrap_nfl_production.sh
# proves writability explicitly rather than assuming it.
#
# ---------------------------------------------------------------------------
# NFL AND NCAAF ARE NEVER MIXED
# ---------------------------------------------------------------------------
# Every directory below is under one NFL-only root, and the Python environment
# is built from THIS repository's own dependency declaration (pyproject.toml /
# requirements*.txt). No NCAAF virtualenv, data directory, artifact directory
# or checkout is referenced anywhere in this repository.
#
# ---------------------------------------------------------------------------
# THE WEB DIRECTORY IS RESOLVED, NOT GUESSED
# ---------------------------------------------------------------------------
# The public page is served at the logical URL path /tools/odds-scanner/
# predictions/NFL/ on the Wizard SportsOdds host (WIZARD_NFL_PUBLIC_URL below),
# but the filesystem path nginx serves that from is a property of the server,
# not of this repository. WIZARD_NFL_WEB_DIR is therefore NOT defaulted to a
# guess: ops/wizard/resolve_web_root.sh discovers it on the server (from the
# live nginx configuration, falling back to locating the already-deployed
# index.html) and records it in $WIZARD_NFL_STATE_DIR/web_root. This file
# reads that record. An operator may override it by exporting
# WIZARD_NFL_WEB_DIR (or setting the WIZARD_NFL_WEB_DIR environment secret in
# the wizard-production GitHub environment), which always wins.
#
# Usage:  . ops/wizard/nfl_production_layout.sh

# Root of the entire NFL production installation. One directory, NFL only.
: "${WIZARD_NFL_HOME:=${HOME}/nfl-production-2026}"

# Application checkout/source: a plain git clone of this repository, updated
# by fetch + hard reset to an exact commit. Never edited in place on the
# server, never the place anything is generated.
: "${WIZARD_NFL_REPO_DIR:=${WIZARD_NFL_HOME}/repo}"

# Python virtual environment, built from the checkout's own dependency
# declaration. Separate from the checkout so a forced repo reset never
# destroys the environment, and so the environment can be rebuilt without
# re-cloning.
: "${WIZARD_NFL_VENV_DIR:=${WIZARD_NFL_HOME}/venv}"

# NFL_MODEL_DATA_ROOT -- the private historical estate (nflverse backfill
# parquets, Odds API history). Irreplaceable-by-default production evidence:
# see scripts/audit_production_runtime_assets.py for the per-asset
# rebuildable/irreplaceable classification.
: "${WIZARD_NFL_DATA_DIR:=${WIZARD_NFL_HOME}/data}"

# NFL_MODEL_ARTIFACT_ROOT -- generated pipeline output: certification
# evidence, the frozen production calibration seed, the forecast/evaluation
# ledgers, the durable canonical 2026 games population, recalibration
# candidates, and the exported public card tree.
: "${WIZARD_NFL_ARTIFACT_DIR:=${WIZARD_NFL_HOME}/artifacts}"

# NFL_LIVE_DATA_ROOT -- the immutable raw BallDontLie provider snapshot cache.
: "${WIZARD_NFL_LIVE_DIR:=${WIZARD_NFL_HOME}/live}"

# Staging area for a card about to be published. A file is validated here and
# then moved onto the served path with an atomic rename; nothing is ever
# written directly into the web directory.
: "${WIZARD_NFL_STAGING_DIR:=${WIZARD_NFL_HOME}/staging}"

# Small durable operational state: the resolved web root, run locks.
: "${WIZARD_NFL_STATE_DIR:=${WIZARD_NFL_HOME}/state}"

# Run logs (never predictions, never secrets).
: "${WIZARD_NFL_LOG_DIR:=${WIZARD_NFL_HOME}/logs}"

# The recorded, server-discovered nginx document directory for
# /tools/odds-scanner/predictions/NFL/. Written by resolve_web_root.sh.
: "${WIZARD_NFL_WEB_ROOT_RECORD:=${WIZARD_NFL_STATE_DIR}/web_root}"

# The public URL the deployed feed must become readable at. Used only by
# scripts/verify_public_nfl_feed.py, which runs on the GitHub-hosted runner.
: "${WIZARD_NFL_PUBLIC_URL:=https://sportsodds.wizardofodds.com/tools/odds-scanner/predictions/NFL/}"  # pragma: allowlist secret

# The published feed's fixed file name. The page fetches exactly this name;
# it is never taken from a command line.
: "${WIZARD_NFL_PUBLIC_JSON_NAME:=latest.json}"

# Deliberately NOT defaulted -- an unresolved web directory must fail closed
# rather than publish into an invented path.
if [ -z "${WIZARD_NFL_WEB_DIR:-}" ] && [ -f "${WIZARD_NFL_WEB_ROOT_RECORD}" ]; then
    WIZARD_NFL_WEB_DIR="$(cat "${WIZARD_NFL_WEB_ROOT_RECORD}")"
fi

export WIZARD_NFL_HOME WIZARD_NFL_REPO_DIR WIZARD_NFL_VENV_DIR \
       WIZARD_NFL_DATA_DIR WIZARD_NFL_ARTIFACT_DIR WIZARD_NFL_LIVE_DIR \
       WIZARD_NFL_STAGING_DIR WIZARD_NFL_STATE_DIR WIZARD_NFL_LOG_DIR \
       WIZARD_NFL_WEB_ROOT_RECORD WIZARD_NFL_PUBLIC_URL \
       WIZARD_NFL_PUBLIC_JSON_NAME
[ -n "${WIZARD_NFL_WEB_DIR:-}" ] && export WIZARD_NFL_WEB_DIR

# The environment the certified pipeline itself reads. Exported here so a
# remote step only ever has to source this one file.
export NFL_MODEL_DATA_ROOT="${WIZARD_NFL_DATA_DIR}"
export NFL_MODEL_ARTIFACT_ROOT="${WIZARD_NFL_ARTIFACT_DIR}"
export NFL_LIVE_DATA_ROOT="${WIZARD_NFL_LIVE_DIR}"

wizard_nfl_print_layout() {
    printf 'WIZARD_NFL_HOME=%s\n' "${WIZARD_NFL_HOME}"
    printf 'WIZARD_NFL_REPO_DIR=%s\n' "${WIZARD_NFL_REPO_DIR}"
    printf 'WIZARD_NFL_VENV_DIR=%s\n' "${WIZARD_NFL_VENV_DIR}"
    printf 'WIZARD_NFL_DATA_DIR=%s\n' "${WIZARD_NFL_DATA_DIR}"
    printf 'WIZARD_NFL_ARTIFACT_DIR=%s\n' "${WIZARD_NFL_ARTIFACT_DIR}"
    printf 'WIZARD_NFL_LIVE_DIR=%s\n' "${WIZARD_NFL_LIVE_DIR}"
    printf 'WIZARD_NFL_STAGING_DIR=%s\n' "${WIZARD_NFL_STAGING_DIR}"
    printf 'WIZARD_NFL_STATE_DIR=%s\n' "${WIZARD_NFL_STATE_DIR}"
    printf 'WIZARD_NFL_LOG_DIR=%s\n' "${WIZARD_NFL_LOG_DIR}"
    printf 'WIZARD_NFL_WEB_DIR=%s\n' "${WIZARD_NFL_WEB_DIR:-UNRESOLVED}"
}
