#!/usr/bin/env bash
# Reproducible, project-local Isaac Sim 4.5 / Isaac Lab 2.0.2 installation.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-$(command -v conda || true)}}"
if [[ ! -x "$CONDA_BIN" ]]; then
    echo 'Conda was not found or is not executable; set CONDA_BIN to its executable, activate Conda, or add conda to PATH.' >&2
    exit 1
fi
PRESSB_ENV="${PRESSB_ENV:-$PROJECT_ROOT/.conda/envs/pressb}"
export CONDA_PKGS_DIRS="$PROJECT_ROOT/.cache/conda"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export PIP_INDEX_URL="${PRESSB_PIP_INDEX_URL:-https://pypi.org/simple}"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export OMNI_KIT_ACCEPT_EULA=YES
if [[ -f "$PROJECT_ROOT/logs/install-pins.txt" ]]; then
    export PIP_CONSTRAINT="$PROJECT_ROOT/logs/install-pins.txt"
fi
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$PROJECT_ROOT/.cache/wheels" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/vendor"
exec > >(tee -a "$PROJECT_ROOT/logs/install-setup.log") 2>&1

if [[ ! -x "$PRESSB_ENV/bin/python" ]]; then
    "$CONDA_BIN" create --prefix "$PRESSB_ENV" python=3.10 pip -y
fi
PYTHON="$PRESSB_ENV/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
"$PYTHON" -m pip install --upgrade pip 'setuptools==78.1.1' wheel
"$PYTHON" -m pip install 'numpy==1.26.4' 'scipy==1.14.1' 'pillow==11.0.0' matplotlib trimesh pytest toml --find-links "$PROJECT_ROOT/.cache/wheels"
"$PYTHON" -m pip install 'isaacsim[all,extscache]==4.5.0' 'torch==2.5.1' 'torchvision==0.20.1' 'gymnasium==1.0.0' 'opencv-python==4.10.0.84' --extra-index-url https://pypi.nvidia.com --find-links "$PROJECT_ROOT/.cache/wheels"

# Read the upstream USD's Unicode prim names with a separate OpenUSD process.
# Never install this package into Isaac Sim's own Python/pxr package directory.
PRESSB_USD_INSPECT="$PROJECT_ROOT/.cache/usd-inspect"
if PYTHONPATH="$PRESSB_USD_INSPECT" "$PYTHON" - "$PRESSB_USD_INSPECT" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
try:
    target = Path(sys.argv[1]).resolve()
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in metadata.distributions(path=[str(target)])}
    assert versions.get('usd-core') == '24.11'
    from pxr import Usd
    assert Usd.GetVersion() == (0, 24, 11)
    assert Path(Usd.__file__).resolve().is_relative_to(target)
except Exception:
    sys.exit(1)
PY
then
    echo 'Isolated OpenUSD 24.11 already installed; preserving existing files.'
else
    "$PYTHON" -m pip install --no-deps --upgrade --target "$PRESSB_USD_INSPECT" 'usd-core==24.11'
fi
PYTHONPATH="$PRESSB_USD_INSPECT" "$PYTHON" -c 'from pxr import Usd; assert Usd.GetVersion() == (0,24,11); print("Isolated OpenUSD:", Usd.GetVersion())'

ISAACLAB_PATH="$PROJECT_ROOT/vendor/IsaacLab"
if [[ ! -d "$ISAACLAB_PATH/.git" ]]; then
    git clone --depth 1 --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
fi
if [[ "$(git -C "$ISAACLAB_PATH" describe --tags --exact-match HEAD)" != v2.0.2 ]]; then
    echo 'Expected vendor/IsaacLab at official v2.0.2; preserve local work and check its revision.' >&2
    exit 1
fi
# These pins retain the tested Isaac Lab 2.0 / PyTorch 2.5 dependency family.
"$PYTHON" -m pip install 'torch==2.5.1' 'torchvision==0.20.1' 'transformers==4.48.3' 'gymnasium==1.0.0' 'warp-lang==1.5.0' --find-links "$PROJECT_ROOT/.cache/wheels"
"$PYTHON" -m pip install --no-build-isolation --find-links "$PROJECT_ROOT/.cache/wheels" \
    -e "$ISAACLAB_PATH/source/isaaclab" \
    -e "$ISAACLAB_PATH/source/isaaclab_assets" \
    -e "$ISAACLAB_PATH/source/isaaclab_tasks" \
    -e "$ISAACLAB_PATH/source/isaaclab_rl"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$PROJECT_ROOT/logs/install-requirements.txt"
"$PYTHON" - "$PROJECT_ROOT/logs/install-pins.txt" <<'PY'
import importlib.metadata as m
from pathlib import Path
import sys
pins = []
for distribution in m.distributions():
    name = distribution.metadata['Name']
    if not name.lower().replace('_', '-').startswith('isaaclab'):
        pins.append(f'{name}=={distribution.version}')
Path(sys.argv[1]).write_text('\n'.join(sorted(set(pins), key=str.lower)) + '\n')
print('Python:', sys.version)
for package in ('isaacsim', 'isaaclab', 'isaaclab_assets', 'torch', 'numpy', 'scipy'):
    print(f'{package}: {m.version(package)}')
PY
echo "Ready: $PRESSB_ENV/bin/python"
