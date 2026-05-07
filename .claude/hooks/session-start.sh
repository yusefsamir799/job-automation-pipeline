#!/bin/bash
# SessionStart hook for Claude Code on the web.
# Installs the Python dependencies the pipeline scripts import so that
# tests, linters and ad-hoc script runs work inside a remote session.
set -euo pipefail

# Only run inside the remote (web) container.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

PIP_FLAGS="--disable-pip-version-check --no-input"

# Runtime dependencies (matches README + actual imports across scrapers/, pipeline/, setup/).
python3 -m pip install $PIP_FLAGS \
  python-jobspy \
  anthropic \
  python-docx \
  gspread \
  google-auth \
  requests \
  beautifulsoup4 \
  pandas

# Tooling for linting / testing inside the session.
python3 -m pip install $PIP_FLAGS \
  ruff \
  pytest
