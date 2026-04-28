"""Stage 0 smoke / contract tests for the running RAG API.

Goals:
  - Verify each locked endpoint returns the documented status code and shape.
  - Verify protected routes reject missing/invalid API keys.
  - Stay strictly READ-ONLY (no upload, no index trigger, no deletes).
  - Run against the live `uvicorn backend.main:app` instance — this validates
    the real network + middleware + CORS surface that customers will hit.

Two ways to run:

  # 1) Standalone (no pytest installed; uses .venv with httpx already present):
  BASE_URL=http://localhost:8000 \
  BACKEND_API_KEY=$(grep ^BACKEND_API_KEY .env | cut -d= -f2) \
  /Users/pshah/testing\\ 3/.venv/bin/python tests/test_smoke.py

  # 2) Pytest (if pytest is later installed in the env):
  BASE_URL=http://localhost:8000 \
  BACKEND_API_KEY=... \
  pytest -q tests/test_smoke.py

Exit code:
  0 = all checks passed
  1 = one or more failures (details printed)
"""
from __future__ import annotations

import os
import sys
from typing import Any

import httpx

# ---- configuration ---------------------------------------------------------

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("BACKEND_API_KEY", "").strip()
SAMPLE_QUERY = os.environ.get(
    "SAMPLE_QUERY", "Financing to self help groups"
)
# Big chat queries can take 30s+ on cold cache; set generous default.
QUERY_TIMEOUT = float(os.environ.get("SMOKE_QUERY_TIMEOUT", "60"))
DEFAULT_TIMEOUT = float(os.environ.get("SMOKE_TIMEOUT", "10"))


def _client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=timeout, follow_redirects=False)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---- individual checks (also pytest-discoverable as `test_*`) --------------

def test_health_ok() -> None:
    with _client() as c:
        r = c.get("/api/health")
    _assert(r.status_code == 200, f"/api/health expected 200 got {r.status_code}")
    body = r.json()
    _assert(body.get("status") == "ok", f"/api/health body: {body!r}")


def test_index_status_shape() -> None:
    with _client(timeout=15) as c:
        r = c.get("/api/index/status")
    _assert(r.status_code == 200, f"/api/index/status expected 200 got {r.status_code}")
    body = r.json()
    required_keys = {
        "chunk_count",
        "tree_count",
        "image_count",
        "patients",
        "status",
        "indexing",
        "progress",
        "stage",
    }
    missing = required_keys - body.keys()
    _assert(not missing, f"/api/index/status missing keys: {missing}")
    _assert(isinstance(body["chunk_count"], int), "chunk_count must be int")
    _assert(isinstance(body["patients"], list), "patients must be list")
    _assert(isinstance(body["indexing"], bool), "indexing must be bool")


def test_documents_list_shape() -> None:
    if not API_KEY:
        print("  [skip] BACKEND_API_KEY not provided; skipping authed /api/documents check")
        return
    with _client() as c:
        r = c.get("/api/documents", headers={"X-API-Key": API_KEY})
    _assert(r.status_code == 200, f"/api/documents expected 200 got {r.status_code}")
    body = r.json()
    _assert("files" in body and isinstance(body["files"], list), f"unexpected body: {body!r}")
    if body["files"]:
        sample = body["files"][0]
        for key in ("name", "size", "type"):
            _assert(key in sample, f"file entry missing '{key}': {sample!r}")


def test_documents_info_shape() -> None:
    with _client(timeout=15) as c:
        r = c.get("/api/documents/info")
    _assert(r.status_code == 200, f"/api/documents/info expected 200 got {r.status_code}")
    body = r.json()
    _assert("chunk_count" in body, f"/api/documents/info body: {body!r}")


def test_metrics_requires_api_key() -> None:
    with _client() as c:
        r_missing = c.get("/api/metrics")
        r_bad = c.get("/api/metrics", headers={"X-API-Key": "definitely-wrong"})
    _assert(
        r_missing.status_code in (401, 503),
        f"/api/metrics without key expected 401/503 got {r_missing.status_code}",
    )
    _assert(
        r_bad.status_code == 401,
        f"/api/metrics with bad key expected 401 got {r_bad.status_code}",
    )


def test_metrics_with_valid_key() -> None:
    if not API_KEY:
        print("  [skip] BACKEND_API_KEY not provided; skipping authed /api/metrics check")
        return
    with _client() as c:
        r = c.get("/api/metrics", headers={"X-API-Key": API_KEY})
    _assert(r.status_code == 200, f"/api/metrics expected 200 got {r.status_code} body={r.text!r}")
    body = r.json()
    for key in ("uptime_s", "request_count", "error_count", "avg_latency_ms", "top_routes"):
        _assert(key in body, f"/api/metrics missing key '{key}'")


def test_query_validation_too_long() -> None:
    if not API_KEY:
        print("  [skip] BACKEND_API_KEY not provided; skipping /query check")
        return
    long_q = "a" * 3000
    with _client() as c:
        r = c.post("/query", headers={"X-API-Key": API_KEY}, json={"question": long_q})
    _assert(
        r.status_code == 422,
        f"/query overlong question expected 422 got {r.status_code}: {r.text!r}",
    )


def test_query_validation_empty() -> None:
    if not API_KEY:
        print("  [skip] BACKEND_API_KEY not provided; skipping /query check")
        return
    with _client() as c:
        r = c.post("/query", headers={"X-API-Key": API_KEY}, json={"question": ""})
    _assert(
        r.status_code == 422,
        f"/query empty question expected 422 got {r.status_code}: {r.text!r}",
    )


def test_query_happy_path() -> None:
    if not API_KEY:
        print("  [skip] BACKEND_API_KEY not provided; skipping live /query")
        return
    if os.environ.get("SMOKE_SKIP_QUERY") == "1":
        print("  [skip] SMOKE_SKIP_QUERY=1 set; skipping live /query")
        return
    with _client(timeout=QUERY_TIMEOUT) as c:
        r = c.post("/query", headers={"X-API-Key": API_KEY}, json={"question": SAMPLE_QUERY})
    _assert(r.status_code == 200, f"/query expected 200 got {r.status_code}: {r.text[:300]!r}")
    body = r.json()
    for key in ("answer", "sources", "intent", "elapsed_ms"):
        _assert(key in body, f"/query missing key '{key}'")
    _assert(isinstance(body["sources"], list), "sources must be list")
    _assert(len(body["answer"]) > 0, "answer must not be empty")
    print(f"  /query elapsed_ms={body.get('elapsed_ms')} sources={len(body.get('sources', []))}")


# ---- standalone runner -----------------------------------------------------

def _checks() -> list[tuple[str, Any]]:
    return [
        ("health_ok", test_health_ok),
        ("index_status_shape", test_index_status_shape),
        ("documents_list_shape", test_documents_list_shape),
        ("documents_info_shape", test_documents_info_shape),
        ("metrics_requires_api_key", test_metrics_requires_api_key),
        ("metrics_with_valid_key", test_metrics_with_valid_key),
        ("query_validation_too_long", test_query_validation_too_long),
        ("query_validation_empty", test_query_validation_empty),
        ("query_happy_path", test_query_happy_path),
    ]


def main() -> int:
    print(f"Smoke target: {BASE_URL} (api key set: {bool(API_KEY)})")
    failures: list[tuple[str, str]] = []
    for name, fn in _checks():
        try:
            print(f"-> {name}")
            fn()
            print(f"   PASS")
        except AssertionError as e:
            print(f"   FAIL: {e}")
            failures.append((name, str(e)))
        except httpx.RequestError as e:
            print(f"   FAIL (network): {e}")
            failures.append((name, f"network: {e}"))
        except Exception as e:  # noqa: BLE001 — surface unexpected errors
            print(f"   FAIL (unexpected {type(e).__name__}): {e}")
            failures.append((name, f"{type(e).__name__}: {e}"))
    print()
    total = len(_checks())
    passed = total - len(failures)
    print(f"Summary: {passed}/{total} passed")
    if failures:
        print("Failures:")
        for n, msg in failures:
            print(f"  - {n}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
