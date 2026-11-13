#!/usr/bin/env bash
# Install only into an activated, non-base conda environment on an ARM host.
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" || ! -f "$CONDA_PREFIX/conda-meta/history" ]]; then
  echo 'Activate a dedicated Python 3.11 conda environment first.' >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo 'conda must be available to verify that the active prefix is not base.' >&2
  exit 2
fi
CONDA_BASE_PREFIX="$(conda info --base)"
if [[ "$(realpath "$CONDA_PREFIX")" == "$(realpath "$CONDA_BASE_PREFIX")" ]]; then
  echo 'Refusing to install into conda base; activate a dedicated environment.' >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PI05_PYTHON="$CONDA_PREFIX/bin/python"
"$PI05_PYTHON" - <<'PY'
import os, platform, sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ['CONDA_PREFIX']).resolve():
    raise SystemExit('Python does not belong to the active conda environment')
if sys.version_info[:2] != (3, 11) or platform.machine().lower() != 'aarch64':
    raise SystemExit('This tested recipe requires aarch64 and Python 3.11')
PY

CONSTRAINTS="$SCRIPT_DIR/constraints-910c.txt"
"$PI05_PYTHON" -m pip install --constraint "$CONSTRAINTS" \
  'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cpu
"$PI05_PYTHON" -m pip install --constraint "$CONSTRAINTS" \
  'torch-npu==2.10.0' numpy huggingface_hub
"$PI05_PYTHON" "$PROJECT_DIR/scripts/install_pi05.py" --constraint "$CONSTRAINTS"
# Python dependencies used by the existing CANN toolkit/profiler, installed in
# this conda prefix. This script does not install or modify CANN or the driver.
"$PI05_PYTHON" -m pip install --constraint "$CONSTRAINTS" \
  decorator attrs ml-dtypes tornado absl-py pandas plotly loguru openpyxl msguard \
  'opentelemetry-exporter-otlp-proto-grpc==1.33.1' \
  'opentelemetry-exporter-otlp-proto-http==1.33.1'
"$PI05_PYTHON" -m pip check
