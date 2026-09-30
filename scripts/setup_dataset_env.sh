#!/usr/bin/env bash
# Separate CPU-only LeRobot conversion/audit environment; never alter Isaac's Python.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-$(command -v conda || true)}}"
if [[ ! -x "$CONDA_BIN" ]]; then
    echo 'Conda was not found or is not executable; set CONDA_BIN to its executable, activate Conda, or add conda to PATH.' >&2
    exit 1
fi
DATASET_ENV="${PRESSB_DATASET_ENV:-$PROJECT_ROOT/.conda/envs/lerobot}"
DATASET_ENV="$(realpath -m -- "$DATASET_ENV")"
if [[ "$DATASET_ENV" == "$(realpath -m -- "$PROJECT_ROOT/.conda/envs/pressb")" ]]; then
    echo 'The dataset environment must be separate from .conda/envs/pressb.' >&2
    exit 1
fi
export CONDA_PKGS_DIRS="$PROJECT_ROOT/.cache/conda"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${PRESSB_DATASET_PIP_INDEX_URL:-https://pypi.org/simple}"
# Do not inherit the Isaac environment's USD path, pins or install destination.
unset PYTHONPATH PYTHONHOME PIP_CONSTRAINT PIP_TARGET PIP_PREFIX PIP_EXTRA_INDEX_URL
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$PROJECT_ROOT/logs"
exec > >(tee -a "$PROJECT_ROOT/logs/install-dataset-env.log") 2>&1
cd "$PROJECT_ROOT"

if [[ ! -x "$DATASET_ENV/bin/python" ]]; then
    "$CONDA_BIN" create --prefix "$DATASET_ENV" python=3.12 pip -y
fi
DATASET_PYTHON="$DATASET_ENV/bin/python"
"$DATASET_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected a separate Python 3.12 environment: " + sys.version'
"$DATASET_PYTHON" -m pip install --upgrade pip wheel
"$DATASET_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu'
"$DATASET_PYTHON" -m pip install --requirement "$PROJECT_ROOT/requirements-dataset.txt"
"$DATASET_PYTHON" -m pip check
"$DATASET_PYTHON" -m pip freeze > "$PROJECT_ROOT/logs/install-dataset-requirements.txt"
"$DATASET_PYTHON" - "$PROJECT_ROOT/logs/install-dataset-versions.json" <<'PY'
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys

import av
import numpy
import torch
import torchvision
from lerobot.datasets.lerobot_dataset import LeRobotDataset

assert torch.version.cuda is None, "The conversion environment must use CPU PyTorch"
assert version("torch").split("+")[0] == "2.10.0"
assert version("torchvision").split("+")[0] == "0.25.0"
assert version("lerobot") == "0.6.1"
record = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "environment": sys.prefix,
    "python": sys.version,
    "torch_cuda_build": torch.version.cuda,
    "pip_check": "No broken requirements found.",
    "lerobot_dataset_import": LeRobotDataset.__name__,
    "packages": {name: version(name) for name in
                 ("torch", "torchvision", "torchcodec", "numpy", "av", "datasets", "pandas", "pyarrow", "lerobot")},
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
echo "Dataset conversion/audit Python: $DATASET_PYTHON"
echo 'Dataset operations are local; no Hugging Face Hub upload is performed.'
