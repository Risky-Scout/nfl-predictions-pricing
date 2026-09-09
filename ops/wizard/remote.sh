#!/usr/bin/env bash
# The ONE SSH client wrapper the authoritative workflow uses to reach the
# Wizard server. Runs on the GitHub-hosted runner, never on the server.
#
# It is a thin, auditable wrapper around the EXACT connection pattern the
# merged .github/workflows/wizard_ssh_smoke.yml already proved end to end
# (PRIVATE_KEY_PARSE=PASS, KNOWN_HOSTS_PARSE=PASS, SSH_CONNECTION=PASS against
# the Wizard SportsOdds host): a 0600 private key written to ~/.ssh, a pinned
# known_hosts file, BatchMode, and StrictHostKeyChecking=yes. Host key
# checking is never relaxed and no password authentication path exists.
#
# There is no FTP anywhere in this file, and no FTP in the authoritative
# workflow. scripts/publish_sportsodds_nfl.py (FTP) remains only as a manual
# fallback and is not invoked by automation.
#
# Credentials come from the wizard-production GitHub environment and are read
# from the process environment only:
#   WIZARD_SSH_HOST WIZARD_SSH_PORT WIZARD_SSH_USER
#   WIZARD_SSH_PRIVATE_KEY WIZARD_SSH_KNOWN_HOSTS
# No secret value is ever echoed, and `set -x` is never enabled here.
#
# Subcommands:
#   prepare               write the key + known_hosts, prove both parse
#   exec <command...>     run a command on the server, failing on its exit code
#   put <local> <remote>  copy one file to the server (remote path is relative
#                         to the SSH user's home)
#   push-ops <remote-dir> copy the ops/wizard/*.sh helpers plus an optional
#                         source archive into <remote-dir>, relative to home
#
# Remote paths are deliberately HOME-RELATIVE. Nothing here assumes /opt or
# /srv is writable, and the layout the server actually uses is defined once in
# ops/wizard/nfl_production_layout.sh -- which this wrapper never needs to
# know about, because it only has to land the bootstrap in the user's home.
set -euo pipefail

KEY_PATH="${HOME}/.ssh/wizard_deploy"
KNOWN_HOSTS_PATH="${HOME}/.ssh/wizard_known_hosts"

_require_env() {
    for name in "$@"; do
        if [ -z "${!name:-}" ]; then
            echo "FAIL CLOSED: required environment variable ${name} is not set" >&2
            exit 78
        fi
    done
}

_ssh_opts() {
    printf '%s\n' \
        -i "${KEY_PATH}" \
        -p "${WIZARD_SSH_PORT}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=yes \
        -o UserKnownHostsFile="${KNOWN_HOSTS_PATH}" \
        -o ConnectTimeout=20 \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=10
}

cmd_prepare() {
    _require_env WIZARD_SSH_PRIVATE_KEY WIZARD_SSH_KNOWN_HOSTS

    install -d -m 700 "${HOME}/.ssh"

    printf '%s\n' "${WIZARD_SSH_PRIVATE_KEY}" > "${KEY_PATH}"
    chmod 600 "${KEY_PATH}"

    printf '%s\n' "${WIZARD_SSH_KNOWN_HOSTS}" > "${KNOWN_HOSTS_PATH}"
    chmod 600 "${KNOWN_HOSTS_PATH}"

    # Parsing both up front turns a malformed secret into a clear, immediate
    # failure instead of an opaque "Permission denied" later in the run.
    ssh-keygen -y -f "${KEY_PATH}" > /dev/null
    echo "PRIVATE_KEY_PARSE=PASS"
    ssh-keygen -lf "${KNOWN_HOSTS_PATH}" > /dev/null
    echo "KNOWN_HOSTS_PARSE=PASS"
}

cmd_exec() {
    _require_env WIZARD_SSH_HOST WIZARD_SSH_PORT WIZARD_SSH_USER
    mapfile -t opts < <(_ssh_opts)
    # `bash -s` with the command on stdin keeps quoting sane for multi-line
    # remote scripts, and `set -euo pipefail` on the far side means a failing
    # remote step fails the workflow step rather than being swallowed.
    printf '%s\n' 'set -euo pipefail' "$@" \
        | ssh "${opts[@]}" "${WIZARD_SSH_USER}@${WIZARD_SSH_HOST}" 'bash -s'
}

cmd_put() {
    _require_env WIZARD_SSH_HOST WIZARD_SSH_PORT WIZARD_SSH_USER
    local local_path="$1" remote_path="$2"
    scp -P "${WIZARD_SSH_PORT}" \
        -i "${KEY_PATH}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=yes \
        -o UserKnownHostsFile="${KNOWN_HOSTS_PATH}" \
        -o ConnectTimeout=20 \
        "${local_path}" "${WIZARD_SSH_USER}@${WIZARD_SSH_HOST}:${remote_path}"
}

cmd_push_ops() {
    local remote_dir="$1" source_archive="${2:-}"
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Unquoted "$HOME/..." on the far side so the SERVER expands its own home.
    cmd_exec "mkdir -p \"\$HOME/${remote_dir}\" && chmod 700 \"\$HOME/${remote_dir}\""
    for script in nfl_production_layout.sh resolve_web_root.sh bootstrap_nfl_production.sh; do
        cmd_put "${here}/${script}" "${remote_dir}/${script}"
    done
    if [ -n "${source_archive}" ]; then
        cmd_put "${source_archive}" "${remote_dir}/nfl-source.tar.gz"
    fi
    cmd_exec "ls -1 \"\$HOME/${remote_dir}\""
}

case "${1:-}" in
    prepare)   shift; cmd_prepare "$@" ;;
    exec)      shift; cmd_exec "$@" ;;
    put)       shift; cmd_put "$@" ;;
    push-ops)  shift; cmd_push_ops "$@" ;;
    *)
        echo "usage: remote.sh {prepare|exec <command>|put <local> <remote>|push-ops <remote-dir> [source-archive]}" >&2
        exit 2
        ;;
esac
