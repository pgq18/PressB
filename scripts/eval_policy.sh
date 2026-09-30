#!/usr/bin/env bash
set -euo pipefail
PRESSB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PXR_WORK_THREAD_LIMIT=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PRESSB_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
cd "$PRESSB_ROOT"
exec "$PRESSB_ROOT/.conda/envs/pressb/bin/python" "$PRESSB_ROOT/scripts/eval_policy.py" "$@"
