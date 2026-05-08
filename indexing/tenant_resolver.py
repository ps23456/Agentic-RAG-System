"""Tenant-resolver helpers for the indexing pipeline.

Indexers walk `data_folder` from disk; they don't know who owns each file.
This module bridges that gap by querying the `documents` table once at the
start of an index run and producing fast in-memory lookup maps:

    by_path   — full storage_uri -> {tenant_id, user_id, customer_id}
    by_base   — basename -> {tenant_id, user_id, customer_id} (only when
                that basename has exactly one owner; ambiguous basenames
                stay out of this map and force a path-based lookup).

Callers always prefer `by_path` when they have the full path, falling
back to `by_base` only when they don't.

The lookups intentionally return empty strings (not None) for missing
owners, so Chroma metadata stays string-typed and easily filterable.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class TenantOwnershipResolver:
    """In-memory snapshot of (path -> owner) and (basename -> owner).

    Build one of these once per indexing run; reuse for every file. Cheap
    to construct (single SELECT against `documents`).
    """

    __slots__ = ("by_path", "by_base", "_ambiguous_bases")

    def __init__(self) -> None:
        self.by_path: dict[str, dict[str, str]] = {}
        self.by_base: dict[str, dict[str, str]] = {}
        self._ambiguous_bases: set[str] = set()

    @classmethod
    def from_tenant_store(cls) -> "TenantOwnershipResolver":
        """Snapshot all active documents into memory.

        Imported lazily so this module has no hard dependency on the FastAPI
        app — it can be used from scripts and tests too.
        """
        inst = cls()
        try:
            from backend.db.tenant_store import tenant_store
            rows = tenant_store.list_active_documents_for_indexing()
        except Exception as e:
            logger.warning("Tenant resolver: could not load documents table (%s)", e)
            return inst

        seen_base: dict[str, dict[str, str]] = {}
        for r in rows:
            owner = {
                "tenant_id": (r.get("tenant_id") or "").strip(),
                "user_id": (r.get("user_id") or "").strip(),
                "customer_id": (r.get("customer_id") or "").strip(),
            }
            path = (r.get("storage_uri") or "").strip()
            base = (r.get("file_name") or "").strip()
            if path:
                inst.by_path[path] = owner
            if base:
                if base in inst._ambiguous_bases:
                    continue
                prev = seen_base.get(base)
                if prev is None:
                    seen_base[base] = owner
                elif prev != owner:
                    # Same basename owned by two different (tenant,user,customer)
                    # combos — basename lookup would be unsafe.
                    inst._ambiguous_bases.add(base)
                    seen_base.pop(base, None)
        inst.by_base = seen_base
        if inst._ambiguous_bases:
            logger.info(
                "Tenant resolver: %d ambiguous basenames; use full storage_uri",
                len(inst._ambiguous_bases),
            )
        return inst

    def lookup(self, path: str = "", basename: str = "") -> dict[str, str]:
        """Return owner dict (always string-valued) for the given file.

        Strategy: full path first (always unique), then basename if
        unambiguous, else empty owner. Empty owner is fine — retrieval
        will still see those rows, but they're flagged for backfill.
        """
        if path and path in self.by_path:
            return self.by_path[path]
        if basename:
            base = os.path.basename(basename).strip()
            if base and base in self.by_base:
                return self.by_base[base]
        return {"tenant_id": "", "user_id": "", "customer_id": ""}


_resolver_singleton: Optional[TenantOwnershipResolver] = None


def get_resolver(refresh: bool = False) -> TenantOwnershipResolver:
    """Process-level lazy resolver. Pass refresh=True at the start of every
    index run so newly uploaded documents are visible.
    """
    global _resolver_singleton
    if refresh or _resolver_singleton is None:
        _resolver_singleton = TenantOwnershipResolver.from_tenant_store()
    return _resolver_singleton
