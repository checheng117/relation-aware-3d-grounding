#!/usr/bin/env bash
# Create the canonical conda environment and install this repo in editable mode.
# Does not read or print .env / HF_TOKEN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="rag3d"
ENV_FILE="$ROOT/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found on PATH."
  echo
  echo "Install Miniconda or Miniforge, restart the terminal, then run:"
  echo "  bash scripts/setup_env.sh"
  echo
  echo "Reference: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: missing $ENV_FILE"
  exit 1
fi

# Conda's qt-main_activate.sh trips `set -u` when QT_* is unset; relax for conda only.
set +u
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"

eval "$(conda shell.bash hook)"

env_exists() {
  conda run -n "$ENV_NAME" python -c "pass" >/dev/null 2>&1
}

if env_exists; then
  echo "Conda environment '$ENV_NAME' already exists (skipping conda env create)."
else
  echo "Creating conda environment '$ENV_NAME' from environment.yml ..."
  conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"
pip install -U pip
pip install -e ".[dev,viz]"
set -u

echo
echo "Done."
echo "Activate this environment in new shells:"
echo "  conda activate $ENV_NAME"
echo
echo "Then run tests or smoke:"
echo "  make test"
echo "  make smoke"
echo
echo "Optional sanity check:"
echo "  python scripts/check_env.py"
