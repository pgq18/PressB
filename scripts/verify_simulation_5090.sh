#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
PYTHON="$PROJECT_ROOT/.conda/envs/pressb/bin/python"
OUTPUT="${1:-outputs/edge_feedback}"
if [[ -e "$OUTPUT" ]]; then
    echo "Choose a new output directory to preserve existing evidence: $OUTPUT" >&2
    exit 1
fi
mkdir -p logs
exec > >(tee -a logs/verification-5090-simulation.log) 2>&1
trap 'result=$?; printf "%s\n" "$result" > logs/verification-5090-simulation.exitcode' EXIT
"$PYTHON" scripts/verify_gpu_5090.py
PYTHONPATH="$PROJECT_ROOT/.cache/usd-inspect" "$PYTHON" scripts/prepare_wrist_asset.py
bash scripts/run.sh --headless --gpu 0 --video --output "$OUTPUT"
"$PYTHON" scripts/audit_episode.py "$OUTPUT"
"$PYTHON" scripts/audit_global_camera.py "$OUTPUT"
PYTHONPATH="$PROJECT_ROOT/.cache/usd-inspect" "$PYTHON" scripts/audit_mount_asset.py "$OUTPUT"
printf 'Full physical scene and both RGB-D camera audits passed: %s\n' "$OUTPUT"
