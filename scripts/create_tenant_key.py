#!/usr/bin/env python3
"""Create or rotate a tenant API key in the local registry.

Usage:
  .venv/bin/python scripts/create_tenant_key.py \
    --tenant-slug tenant-a \
    --tenant-name "Tenant A" \
    --user-email owner@tenant-a.local \
    --user-name "Tenant A Owner" \
    --label "tenant-a-key"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.db.tenant_store import tenant_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or rotate tenant API key.")
    parser.add_argument("--tenant-slug", required=True)
    parser.add_argument("--tenant-name", required=True)
    parser.add_argument("--user-email", required=True)
    parser.add_argument("--user-name", default="")
    parser.add_argument("--label", default="service-key")
    parser.add_argument("--role", default="owner")
    parser.add_argument(
        "--scopes",
        default="admin:*",
        help="Comma-separated scopes, e.g. 'docs:read,query:run'",
    )
    parser.add_argument(
        "--raw-key",
        default="",
        help="Optional explicit key value. If omitted, a strong random key is generated.",
    )
    parser.add_argument(
        "--expires-at",
        default="",
        help="Optional ISO-8601 UTC timestamp, e.g. 2026-12-31T23:59:59+00:00",
    )
    args = parser.parse_args()

    tenant_store.initialize_schema()
    scopes = tuple(s.strip() for s in args.scopes.split(",") if s.strip())
    raw_key, ctx = tenant_store.create_or_rotate_api_key(
        tenant_slug=args.tenant_slug,
        tenant_name=args.tenant_name,
        user_email=args.user_email,
        user_display_name=args.user_name or args.user_email,
        key_label=args.label,
        raw_api_key=args.raw_key or None,
        role=args.role,
        scopes=scopes or ("admin:*",),
        expires_at=(args.expires_at.strip() or None),
    )
    print(
        json.dumps(
            {
                "tenant_id": ctx.tenant_id,
                "tenant_slug": ctx.tenant_slug,
                "user_id": ctx.user_id,
                "user_email": ctx.user_email,
                "key_id": ctx.key_id,
                "key_label": ctx.key_label,
                "role": ctx.role,
                "scopes": list(ctx.scopes),
                "api_key": raw_key,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

