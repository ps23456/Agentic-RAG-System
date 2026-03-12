#!/usr/bin/env python3
"""Check that the venv has all deps and PyTorch is usable for the app. Run: python scripts/check_env.py"""
import sys
import os

# project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def main():
    print("Python:", sys.executable)
    print("Version:", sys.version.split()[0])
    errors = []

    # numpy
    try:
        import numpy as np
        v = np.__version__
        if int(v.split(".")[0]) >= 2:
            errors.append(f"numpy {v} may conflict with torch; use numpy<2 (e.g. pip install 'numpy>=1.26,<2')")
        else:
            print(f"numpy {v} OK")
    except Exception as e:
        errors.append(f"numpy: {e}")

    # torch
    try:
        import torch
        if not hasattr(torch, "get_default_device"):
            torch.get_default_device = lambda: torch.device(torch._C._get_default_device())
        print(f"torch {torch.__version__} OK")
    except Exception as e:
        errors.append(f"torch: {e}")
        print("Install with: pip install torch")
        if errors:
            print("\nErrors:", *errors, sep="\n  - ")
        return 1

    # Patch transformers to accept torch 2.2 (it requires 2.4+ otherwise)
    try:
        import transformers.utils.import_utils as _tfu
        _old = _tfu.is_torch_available
        if hasattr(_old, "cache_clear"):
            _old.cache_clear()
        _tfu.is_torch_available = lambda: True
        if "torch" in getattr(_tfu, "BACKENDS_MAPPING", {}):
            _orig = _tfu.BACKENDS_MAPPING["torch"]
            _tfu.BACKENDS_MAPPING["torch"] = (lambda: True, _orig[1] if isinstance(_orig, (list, tuple)) else _orig)
        import transformers.utils as _tu
        _tu.is_torch_available = lambda: True
        import transformers as _tr
        _tr.is_torch_available = lambda: True
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        print("sentence_transformers (all-MiniLM-L6-v2) OK")
    except Exception as e:
        errors.append(f"sentence_transformers/transformers: {e}")

    if errors:
        print("\nIssues:")
        for e in errors:
            print("  -", e)
        return 1
    print("\nEnvironment OK. Run: ./run.sh  or  .venv/bin/python -m streamlit run app.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
