#!/usr/bin/env bash
# NVIDIA's Isaac Lab 2.2 / Isaac Sim 5.0 stack for RTX 50-series (Blackwell).
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-$(command -v conda || true)}}"
PRESSB_ENV="${PRESSB_ENV:-$PROJECT_ROOT/.conda/envs/pressb}"
export CONDA_PKGS_DIRS="$PROJECT_ROOT/.cache/conda"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export PIP_INDEX_URL="${PRESSB_PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_DEFAULT_TIMEOUT=120 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export PIP_PROGRESS_BAR=off OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_CONSTRAINT PIP_EXTRA_INDEX_URL
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/vendor"
exec > >(tee -a "$PROJECT_ROOT/logs/install-5090.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$PROJECT_ROOT/logs/install-5090.exitcode"' EXIT
if [[ ! -x "$PRESSB_ENV/bin/python" ]]; then
    if [[ ! -x "$CONDA_BIN" ]]; then
        echo 'Set CONDA_BIN to an executable Conda installation.' >&2
        exit 1
    fi
    "$CONDA_BIN" create --prefix "$PRESSB_ENV" python=3.11 pip -y
fi
PYTHON="$PRESSB_ENV/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "RTX 5090 profile requires a separate Python 3.11 environment"'
"$PYTHON" -m pip install --upgrade pip 'setuptools==78.1.1' 'wheel==0.45.1'
# Install the CUDA 12.8 wheels explicitly; older PyTorch binaries lack sm_120.
"$PYTHON" -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
export PIP_CONSTRAINT="$PROJECT_ROOT/configs/constraints-5090.txt"
"$PYTHON" -m pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com
"$PYTHON" -m pip install 'scipy==1.15.3' 'opencv-python==4.10.0.84' matplotlib trimesh pytest toml 'warp-lang==1.7.1' 'av==15.1.0'
# Keep modern USD separate from Kit's USD; never install usd-core into the main environment.
"$PYTHON" -m pip install --no-deps --upgrade --target "$PROJECT_ROOT/.cache/usd-inspect" 'usd-core==24.11'
ISAACLAB_PATH="$PROJECT_ROOT/vendor/IsaacLab"
if [[ ! -d "$ISAACLAB_PATH/.git" ]]; then
    git clone --depth 1 --branch v2.2.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
fi
if [[ "$(git -C "$ISAACLAB_PATH" describe --tags --exact-match HEAD)" != v2.2.0 ]]; then
    echo 'Expected the official Isaac Lab v2.2.0 tag; preserve and inspect the existing checkout.' >&2
    exit 1
fi
"$PYTHON" -m pip install --no-build-isolation \
    -e "$ISAACLAB_PATH/source/isaaclab" \
    -e "$ISAACLAB_PATH/source/isaaclab_assets" \
    -e "$ISAACLAB_PATH/source/isaaclab_tasks" \
    -e "$ISAACLAB_PATH/source/isaaclab_rl"
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$PROJECT_ROOT"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$PROJECT_ROOT/logs/install-5090-requirements.txt"
"$PYTHON" - "$PROJECT_ROOT/logs/install-5090-versions.json" <<'PY'
import importlib.metadata as m
import json,sys
from pathlib import Path
import torch
assert torch.version.cuda == '12.8'
assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
record={'python':sys.version,'environment':sys.prefix,'packages':{name:m.version(name) for name in ('isaacsim','isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl','torch','torchvision','numpy','Pillow','gymnasium','scipy')},'torch_cuda':torch.version.cuda,'torch_architectures':torch.cuda.get_arch_list()}
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
printf 'Ready: %s\n' "$PYTHON"
