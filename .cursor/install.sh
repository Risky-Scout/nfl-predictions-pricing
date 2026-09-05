#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the nfl-hybrid-model project.
# Creates/refreshes a project virtualenv at .venv and installs the package
# in editable mode with the data + dev extras (mirrors README + CI).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The default base image ships python3 but not always the stdlib venv/ensurepip
# support that `python3 -m venv` needs. Ensure it before creating the venv.
if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1 \
  || ! python3 -Ic 'import ensurepip; ensurepip.version()' >/dev/null 2>&1; then
  sudo apt-get update -qq
  PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  sudo apt-get install -y -qq "python${PYVER}-venv" || sudo apt-get install -y -qq python3-venv
fi

# `python3 -m venv` is safe to re-run against an existing environment.
python3 -m venv .venv

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[data,dev]"

echo "install.sh: environment ready ($(.venv/bin/python --version))"
