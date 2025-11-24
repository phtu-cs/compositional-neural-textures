#!/usr/bin/env bash
set -euo pipefail

# Dev environment bootstrap using uv (https://github.com/astral-sh/uv)
# - Creates a Python 3.9 virtualenv at .venv
# - Installs PyTorch 2.0.0 (+ CUDA 11.8 on NVIDIA Linux)
# - Installs torch-geometric with matching wheels
# - Installs repo dependencies from requirements.in via uv
# - Optionally builds CUDA extensions when nvcc is available

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

# Resolve paths relative to this script so it works from any CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
REQ_IN="$SCRIPT_DIR/requirements.in"
VENV_DIR="$REPO_ROOT/.venv"

echo "[1/7] Checking uv..."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; attempting installation via official script."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add default install location to PATH for this session if needed
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.uv/bin:$PATH"
  else
    echo "curl not available; please install uv manually: https://astral.sh/uv" >&2
    exit 1
  fi
fi

echo "[2/7] Ensuring Python 3.9 is available via uv..."
if ! uv python find 3.9 >/dev/null 2>&1; then
  uv python install 3.9
fi

echo "[3/7] Creating virtual environment ($VENV_DIR) with Python 3.9..."
uv venv --python 3.9 "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -V

echo "[4/7] Preparing environment-specific requirements..."
TORCH_VERSION="${TORCH_VERSION:-2.0.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.1}"
PYG_VERSION="${PYG_VERSION:-2.3.0}"

CUDA_TAG="cpu"
if [[ "$OS_NAME" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_TAG="cu118"
fi
# Allow manual override: export CUDA_TAG_OVERRIDE=cpu|cu118
if [[ -n "${CUDA_TAG_OVERRIDE:-}" ]]; then
  CUDA_TAG="$CUDA_TAG_OVERRIDE"
fi

TORCH_SUFFIX=""
TORCH_FIND_LINKS=""
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"
if [[ "$CUDA_TAG" == "cu118" ]]; then
  TORCH_SUFFIX="+cu118"
  TORCH_FIND_LINKS="https://download.pytorch.org/whl/torch_stable.html"
  PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu118.html"
fi

# Detectron2 pre-selection for Linux+CUDA (wheels or source)
DETECTRON2_SPEC=""
if [[ "$OS_NAME" == "Linux" ]] && [[ "$CUDA_TAG" == "cu118" ]]; then
  DETECTRON2_SPEC="detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
fi

if [[ ! -f "$REQ_IN" ]]; then
  echo "requirements.in not found at $REQ_IN. Did you forget to add it?" >&2
  exit 1
fi

RESOLVED_REQ="$(mktemp "$SCRIPT_DIR/requirements.resolved.in")"
{
  if [[ -n "$TORCH_FIND_LINKS" ]]; then
    echo "--find-links $TORCH_FIND_LINKS"
  fi
  echo "torch==${TORCH_VERSION}${TORCH_SUFFIX}"
  echo "torchvision==${TORCHVISION_VERSION}${TORCH_SUFFIX}"
  echo "--find-links $PYG_FIND_LINKS"
  echo "torch-geometric==${PYG_VERSION}"
  if [[ -n "$DETECTRON2_SPEC" ]]; then
    echo "$DETECTRON2_SPEC"
  fi
  cat "$REQ_IN"
} > "$RESOLVED_REQ"

echo "[5/7] Installing dependencies via uv from $RESOLVED_REQ..."
uv pip compile -o "$SCRIPT_DIR/requirements-lock.txt" "$RESOLVED_REQ" || true
uv pip install -r "$RESOLVED_REQ"

# On macOS, try to ensure detectron2 is importable. We prefer the vendored copy
# to avoid repeated network builds; fall back to Git if needed.
if [[ "$OS_NAME" == "Darwin" ]]; then
  echo "[6/7] macOS: ensuring detectron2 is installed (no build isolation)."
  # Ensure modern packaging/setuptools in case older pins slipped in via cache
  uv pip install -U 'packaging>=24.0' 'setuptools>=65' wheel >/dev/null 2>&1 || true
  export CC=${CC:-clang}
  export CXX=${CXX:-clang++}
  if [[ -d "$REPO_ROOT/detectron2" ]]; then
    echo "Attempting local editable install: $REPO_ROOT/detectron2"
    uv pip install --no-build-isolation -e "$REPO_ROOT/detectron2" || {
      echo "Local install failed; trying GitHub source (may take a while)..."
      uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' || {
        echo "detectron2 install failed on macOS; you can run manually:" >&2
        echo "  CC=clang CXX=clang++ uv pip install --no-build-isolation -e detectron2" >&2
      }
    }
  else
    echo "No local detectron2/ directory; installing from GitHub..."
    uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' || true
  fi
fi

echo "[7/7] Optional CUDA extension builds..."
if command -v nvcc >/dev/null 2>&1; then
  if [[ -d "$REPO_ROOT/src/models/components/deform_detr/models/ops" ]]; then
    (cd "$REPO_ROOT/src/models/components/deform_detr/models/ops" && bash make.sh)
  fi
  echo "Building pointnet2_ops (requires nvcc)..."
  uv pip install -q "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib" || {
    echo "pointnet2_ops install failed; you can retry manually later." >&2
  }
else
  echo "nvcc not found; skipping CUDA extension builds (deformable DETR, pointnet2)."
fi

cat << 'EOF'

Setup complete.

Notes:
- If you use Weights & Biases, login with: wandb login
- To activate the environment in a new shell: source .venv/bin/activate
- If detectron2 or CUDA ops fail to build, ensure compatible drivers/toolkit.
EOF
