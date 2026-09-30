#!/usr/bin/env bash
set -euo pipefail
PRESSB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PRESSB_PYTHON="${PRESSB_PYTHON:-$PRESSB_ROOT/.conda/envs/pressb/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export TMPDIR="$PRESSB_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
cd "$PRESSB_ROOT"
exec "$PRESSB_PYTHON" "$PRESSB_ROOT/scripts/run_sim.py" "$@"
