#!/usr/bin/env python3
"""Automated cross-tenant isolation check for same filename uploads.

Runs an end-to-end check:
  1) creates (or rotates) keys for tenant-a and tenant-b
  2) uploads same filename with different contents under each key
  3) verifies each tenant reads only its own content

Usage:
  BASE_URL=http://localhost:8000 .venv/bin/python scripts/tenant_isolation_check.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.tenant_store import tenant_store


def _mk_key(tenant_slug: str, tenant_name: str, email: str) -> str:
    tenant_store.initialize_schema()
    raw, _ctx = tenant_store.create_or_rotate_api_key(
        tenant_slug=tenant_slug,
        tenant_name=tenant_name,
        user_email=email,
        user_display_name=email,
        key_label=f"{tenant_slug}-integration-key",
    )
    return raw


def main() -> int:
    base_url = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
    filename = "tenant_isolation_probe.txt"
    content_a = "tenant A isolated content\n"
    content_b = "tenant B isolated content\n"

    key_a = _mk_key("tenant-a", "Tenant A", "owner@tenant-a.local")
    key_b = _mk_key("tenant-b", "Tenant B", "owner@tenant-b.local")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        with httpx.Client(base_url=base_url, timeout=60) as c:
            p.write_text(content_a, encoding="utf-8")
            with p.open("rb") as fp:
                r = c.post("/api/upload", headers={"X-API-Key": key_a}, files={"files": (filename, fp, "text/plain")})
            if r.status_code != 200:
                print("FAIL upload A", r.status_code, r.text)
                return 1

            p.write_text(content_b, encoding="utf-8")
            with p.open("rb") as fp:
                r = c.post("/api/upload", headers={"X-API-Key": key_b}, files={"files": (filename, fp, "text/plain")})
            if r.status_code != 200:
                print("FAIL upload B", r.status_code, r.text)
                return 1

            r = c.get("/api/documents/text", headers={"X-API-Key": key_a}, params={"file": filename})
            if r.status_code != 200:
                print("FAIL read A", r.status_code, r.text)
                return 1
            got_a = r.json().get("content", "")

            r = c.get("/api/documents/text", headers={"X-API-Key": key_b}, params={"file": filename})
            if r.status_code != 200:
                print("FAIL read B", r.status_code, r.text)
                return 1
            got_b = r.json().get("content", "")

    ok_a = content_a.strip() in got_a
    ok_b = content_b.strip() in got_b
    cross_bad = (content_b.strip() in got_a) or (content_a.strip() in got_b)

    summary = {
        "base_url": base_url,
        "filename": filename,
        "tenant_a_ok": ok_a,
        "tenant_b_ok": ok_b,
        "cross_tenant_leak": cross_bad,
    }
    print(json.dumps(summary, ensure_ascii=True))

    if ok_a and ok_b and not cross_bad:
        print("PASS tenant isolation")
        return 0
    print("FAIL tenant isolation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

