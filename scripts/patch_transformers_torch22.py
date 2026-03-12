#!/usr/bin/env python3
"""
One-time patch so transformers accepts PyTorch 2.2 (it normally requires 2.4+).
Run with:  .venv/bin/python scripts/patch_transformers_torch22.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Find transformers in the same env as this script
import importlib.util
spec = importlib.util.find_spec("transformers.utils.import_utils")
if spec is None or not spec.origin:
    print("transformers.utils.import_utils not found", file=sys.stderr)
    sys.exit(1)
path = spec.origin
if not os.path.isfile(path):
    print("File not found:", path, file=sys.stderr)
    sys.exit(1)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the strict 2.4 check with accepting any available torch
old = """@lru_cache
def is_torch_available() -> bool:
    try:
        is_available, torch_version = _is_package_available("torch", return_version=True)
        parsed_version = version.parse(torch_version)
        if is_available and parsed_version < version.parse("2.4.0"):
            logger.warning_once(f"Disabling PyTorch because PyTorch >= 2.4 is required but found {torch_version}")
        return is_available and version.parse(torch_version) >= version.parse("2.4.0")
    except packaging.version.InvalidVersion:
        return False"""

new = """@lru_cache
def is_torch_available() -> bool:
    try:
        is_available, torch_version = _is_package_available("torch", return_version=True)
        return is_available  # patched: accept torch 2.2+ (original required >= 2.4)
    except packaging.version.InvalidVersion:
        return False"""

if old not in content:
    if "patched: accept torch 2.2+" in content:
        print("Already patched:", path)
        sys.exit(0)
    print("Could not find the exact block to patch. Transformers version may differ.", file=sys.stderr)
    sys.exit(1)

content = content.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched:", path)
print("You can now run: ./run.sh  or  .venv/bin/python -m streamlit run app.py")
