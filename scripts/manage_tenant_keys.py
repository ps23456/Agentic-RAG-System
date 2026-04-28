#!/usr/bin/env python3
"""Manage tenant API keys: list, revoke, and set expiry.

Examples:
  # list all keys
  .venv/bin/python scripts/manage_tenant_keys.py list

  # list keys for one tenant
  .venv/bin/python scripts/manage_tenant_keys.py list --tenant-slug tenant-a

  # revoke one key
  .venv/bin/python scripts/manage_tenant_keys.py revoke --key-id key_abc123

  # set expiry
  .venv/bin/python scripts/manage_tenant_keys.py expire --key-id key_abc123 --expires-at 2026-12-31T23:59:59+00:00
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


def cmd_list(args) -> int:
    tenant_store.initialize_schema()
    rows = tenant_store.list_api_keys(
        tenant_slug=args.tenant_slug or None,
        user_email=args.user_email or None,
    )
    print(json.dumps({"count": len(rows), "keys": rows}, ensure_ascii=True))
    return 0


def cmd_revoke(args) -> int:
    tenant_store.initialize_schema()
    ok = tenant_store.revoke_api_key(args.key_id)
    print(json.dumps({"ok": ok, "key_id": args.key_id}, ensure_ascii=True))
    return 0 if ok else 1


def cmd_expire(args) -> int:
    tenant_store.initialize_schema()
    ok = tenant_store.set_key_expiry(args.key_id, args.expires_at)
    print(
        json.dumps(
            {"ok": ok, "key_id": args.key_id, "expires_at": args.expires_at},
            ensure_ascii=True,
        )
    )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage tenant API keys.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--tenant-slug", default="")
    p_list.add_argument("--user-email", default="")
    p_list.set_defaults(func=cmd_list)

    p_revoke = sub.add_parser("revoke")
    p_revoke.add_argument("--key-id", required=True)
    p_revoke.set_defaults(func=cmd_revoke)

    p_expire = sub.add_parser("expire")
    p_expire.add_argument("--key-id", required=True)
    p_expire.add_argument("--expires-at", required=True)
    p_expire.set_defaults(func=cmd_expire)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

