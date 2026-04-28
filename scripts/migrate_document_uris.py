#!/usr/bin/env python3
"""Normalize legacy document storage URIs to tenant-scoped paths.

This script updates `documents.storage_uri` when rows still point to the old
shared uploads path (e.g. `data/uploads/<file>`), while the tenant-scoped file
already exists at:
  data/uploads/<tenant_slug>/<user_id>/<file_name>

Usage:
  .venv/bin/python scripts/migrate_document_uris.py --dry-run
  .venv/bin/python scripts/migrate_document_uris.py --apply
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_FOLDER


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate document storage URIs to tenant paths.")
    parser.add_argument("--apply", action="store_true", help="Apply changes.")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed changes only.")
    args = parser.parse_args()
    if not args.apply and not args.dry_run:
        args.dry_run = True

    db_path = os.path.join(PROJECT_ROOT, "storage", "tenant_registry.db")
    uploads_root = os.path.join(DATA_FOLDER, "uploads")
    if not os.path.isfile(db_path):
        print(f"ERROR: db not found at {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT d.id, d.tenant_id, d.user_id, d.file_name, d.storage_uri, t.slug AS tenant_slug
        FROM documents d
        JOIN tenants t ON t.id = d.tenant_id
        WHERE d.status = 'active'
        ORDER BY d.tenant_id, d.user_id, d.file_name
        """
    ).fetchall()

    updates: list[tuple[str, str, str]] = []
    for r in rows:
        file_name = r["file_name"]
        current = (r["storage_uri"] or "").strip()
        tenant_path = os.path.join(uploads_root, r["tenant_slug"], r["user_id"], file_name)
        if not os.path.isfile(tenant_path):
            continue
        if current == tenant_path:
            continue
        # Only rewrite likely legacy/shared URIs or empty URIs.
        shared_path = os.path.join(uploads_root, file_name)
        if (not current) or (current == shared_path):
            updates.append((r["id"], current, tenant_path))

    print(f"db_path={db_path}")
    print(f"candidate_rows={len(rows)}")
    print(f"proposed_updates={len(updates)}")
    for doc_id, old_uri, new_uri in updates[:20]:
        print(f"- {doc_id}: {old_uri or '<empty>'} -> {new_uri}")
    if len(updates) > 20:
        print(f"... and {len(updates) - 20} more")

    if args.apply and updates:
        for doc_id, _old_uri, new_uri in updates:
            cur.execute(
                "UPDATE documents SET storage_uri = ?, updated_at = datetime('now') WHERE id = ?",
                (new_uri, doc_id),
            )
        conn.commit()
        print(f"applied_updates={len(updates)}")
    else:
        print("no changes applied")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

