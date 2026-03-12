#!/usr/bin/env bash
# Run Docling CLI with numpy<2 so PyTorch/Docling don't crash.
# Usage: ./scripts/run_docling.sh [docling args...]
# Example: ./scripts/run_docling.sh data/uploads/Amalgamated_APS_01.tif

set -e
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"

if [[ ! -d "$VENV" ]]; then
  echo "No .venv found. Create one and install requirements first."
  exit 1
fi

# Force NumPy 1.x so Docling/PyTorch work (NumPy 2.x causes "Numpy is not available" and layout errors).
"$VENV/bin/pip" install 'numpy>=1.26.0,<2' --force-reinstall -q
# Run Docling CLI with all passed arguments.
exec "$VENV/bin/docling" "$@"
