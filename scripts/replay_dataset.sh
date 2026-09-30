#!/usr/bin/env bash
set -euo pipefail
PRESSB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PXR_WORK_THREAD_LIMIT=8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$PRESSB_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
cd "$PRESSB_ROOT"
exec "$PRESSB_ROOT/.conda/envs/pressb/bin/python" "$PRESSB_ROOT/scripts/replay_dataset.py" "$@"
