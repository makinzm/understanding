#!/bin/bash
# Thin wrapper so Makefile's `scripts/test-*.sh` glob picks up the
# pytest-based sync-dod-skills.py tests without changing the Makefile.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run --with pytest pytest scripts/test_sync_dod_skills.py -q
