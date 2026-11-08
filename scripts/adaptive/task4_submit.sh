#!/usr/bin/env bash
# Backward-compatible Task 4 launcher.
# Canonical source: scripts/research/historical_replay/task4/task4_submit.sh
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$repo_root/scripts/research/historical_replay/task4/task4_submit.sh" "$@"
