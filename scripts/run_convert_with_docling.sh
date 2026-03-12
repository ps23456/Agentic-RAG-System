#!/usr/bin/env bash
# Convert a document to Markdown with Docling and save to data/structured/.
# Installs numpy<2 and opencv in one go so pip never upgrades numpy to 2.x.
#
# Usage: ./scripts/run_convert_with_docling.sh <path-to-file>
# Example: ./scripts/run_convert_with_docling.sh data/uploads/Amalgamated_APS_01.tif

set -e
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"

if [[ ! -d "$VENV" ]]; then
  echo "No .venv found. Create one and install requirements first."
  exit 1
fi

# Install numpy<2 and opencv in ONE command so pip does not upgrade numpy to 2.x.
# (Installing opencv alone causes pip to install numpy 2.x and breaks Docling.)
echo "Ensuring numpy<2 and opencv-python 4.8–4.9..."
"$VENV/bin/pip" install 'numpy>=1.26.0,<2' 'opencv-python>=4.8,<4.10' -q

echo ""
exec "$VENV/bin/python" scripts/convert_with_docling.py "$@"
