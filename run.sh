#!/usr/bin/env bash
# Run the Insurance Claim Search app using the project venv (avoids wrong Python).
# Uses .venv_bge if it exists (for BGE M3 support), otherwise falls back to .venv
cd "$(dirname "$0")"
if [ -d ".venv_bge" ]; then
  VENV_PATH=".venv_bge"
  echo "Using .venv_bge (BGE M3 support)"
elif [ -d ".venv" ]; then
  VENV_PATH=".venv"
  echo "Using .venv (standard)"
else
  echo "Creating venv and installing dependencies..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt -q
  VENV_PATH=".venv"
fi
exec ${VENV_PATH}/bin/python -m streamlit run app.py --server.port 8501 "$@"
