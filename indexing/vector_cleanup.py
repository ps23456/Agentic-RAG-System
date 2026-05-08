"""Vector-store cleanup helpers.

Removes orphaned vectors from Chroma after a document is deleted or when
reindex needs to drop stale entries that no longer correspond to active docs.

Two entry points:

    purge_file_from_vectors(file_name, doc_id=None)
        Best-effort delete of all vector rows for one file from both the text
        chunk collection and the image collection. Used by DELETE
        /api/documents to keep retrieval in sync with the documents table.

    prune_vectors_to_active_set(active_file_names)
        Walks both collections and removes any row whose `file_name` metadata
        is NOT in the provided active set. Safety net for cases where a
        previous delete ran before purge was wired up, or files were removed
        out-of-band.

Both functions are idempotent and never raise — failures are logged and
swallowed so the caller's primary flow (delete request, reindex job) is
never blocked by a vector-store hiccup.
"""
from __future__ import annotations

import logging
from typing import Iterable

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_IMAGE_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


def _open_collections():
    """Return (text_collection, image_collection) — either may be None.

    Uses get_collection (not get_or_create) so we don't accidentally create
    an empty collection on a fresh server.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        logger.warning("chromadb not installed; skip vector cleanup")
        return None, None

    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception as e:
        logger.warning("Could not open Chroma at %s: %s", CHROMA_PERSIST_DIR, e)
        return None, None

    text_col = None
    image_col = None
    try:
        text_col = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        text_col = None
    try:
        image_col = client.get_collection(name=CHROMA_IMAGE_COLLECTION_NAME)
    except Exception:
        image_col = None
    return text_col, image_col


def _build_purge_where(file_name: str, tenant_id: str | None, customer_id: str | None) -> dict:
    """Compose the Chroma where-clause for a tenant-scoped per-file purge.

    Always pins file_name. Adds tenant_id / customer_id when provided so a
    delete in tenant A cannot remove rows owned by tenant B even if both
    happen to have a file with the same basename.
    """
    clauses: list[dict] = [{"file_name": {"$eq": file_name}}]
    if tenant_id:
        clauses.append({"tenant_id": {"$eq": tenant_id}})
    if customer_id:
        clauses.append({"customer_id": {"$eq": customer_id}})
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def purge_file_from_vectors(
    file_name: str,
    doc_id: str | None = None,
    tenant_id: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """Delete all vector rows for a single file from text + image collections,
    scoped to the given tenant when provided.

    Strongly recommend passing tenant_id from the calling route — same
    basename across tenants would otherwise be ambiguous and could remove
    another tenant's vectors.

    Returns a dict with per-collection counts removed (best-effort).
    """
    out = {"text_removed": 0, "image_removed": 0, "errors": []}
    if not file_name:
        return out

    where = _build_purge_where(file_name, tenant_id, customer_id)
    text_col, image_col = _open_collections()

    if text_col is not None:
        try:
            existing = text_col.get(where=where, include=[])
            ids = list(existing.get("ids") or [])
            if ids:
                text_col.delete(ids=ids)
                out["text_removed"] = len(ids)
        except Exception as e:
            logger.warning("Text-collection purge failed for %s: %s", file_name, e)
            out["errors"].append(f"text: {e}")

    if image_col is not None:
        try:
            existing = image_col.get(where=where, include=[])
            ids = list(existing.get("ids") or [])
            if ids:
                image_col.delete(ids=ids)
                out["image_removed"] = len(ids)
        except Exception as e:
            logger.warning("Image-collection purge failed for %s: %s", file_name, e)
            out["errors"].append(f"image: {e}")

    if out["text_removed"] or out["image_removed"]:
        logger.info(
            "Purged vectors for %s tenant=%s customer=%s — text=%d image=%d",
            file_name, tenant_id or "*", customer_id or "*",
            out["text_removed"], out["image_removed"],
        )
    return out


def backfill_tenant_metadata(
    force: bool = False,
    limit: int | None = None,
) -> dict:
    """Stamp tenant_id / user_id / customer_id onto existing Chroma rows.

    For every row in both collections, we look up its owner via the
    `documents` table — first by full storage path (`path` for image rows,
    matched against documents.storage_uri) then by basename when path is
    missing or non-matching. Rows that already have a tenant_id are
    skipped unless `force=True`.

    Returns {text_updated, image_updated, skipped, unmatched, errors}.

    Idempotent and chunked: call again to drain remaining unmatched rows
    (e.g. after fixing a missing storage_uri or after running upserts).
    """
    out = {
        "text_updated": 0,
        "image_updated": 0,
        "skipped": 0,
        "unmatched": 0,
        "errors": [],
    }
    try:
        from backend.db.tenant_store import tenant_store
    except Exception as e:
        out["errors"].append(f"tenant_store_unavailable: {e}")
        return out

    rows = tenant_store.list_active_documents_for_indexing()
    by_path: dict[str, dict] = {}
    by_base: dict[str, dict] = {}
    ambiguous: set[str] = set()
    for r in rows:
        owner = {
            "tenant_id": r.get("tenant_id") or "",
            "user_id": r.get("user_id") or "",
            "customer_id": r.get("customer_id") or "",
        }
        path = (r.get("storage_uri") or "").strip()
        base = (r.get("file_name") or "").strip()
        if path:
            by_path[path] = owner
        if base:
            if base in ambiguous:
                continue
            prev = by_base.get(base)
            if prev is None:
                by_base[base] = owner
            elif prev != owner:
                ambiguous.add(base)
                by_base.pop(base, None)

    text_col, image_col = _open_collections()

    def _patch(col, key: str) -> None:
        if col is None:
            return
        try:
            existing = col.get(include=["metadatas"])
        except Exception as e:
            out["errors"].append(f"{key}_get_failed: {e}")
            return
        ids = existing.get("ids") or []
        metas = existing.get("metadatas") or []
        patch_ids: list[str] = []
        patch_metas: list[dict] = []
        for i, uid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            meta = meta or {}
            if (meta.get("tenant_id") or "").strip() and not force:
                out["skipped"] += 1
                continue
            row_path = (meta.get("path") or "").strip()
            row_base = (meta.get("file_name") or "").strip()
            owner = by_path.get(row_path) if row_path else None
            if not owner and row_base:
                owner = by_base.get(row_base)
            if not owner:
                out["unmatched"] += 1
                continue
            new_meta = dict(meta)
            new_meta["tenant_id"] = owner.get("tenant_id", "") or ""
            new_meta["user_id"] = owner.get("user_id", "") or ""
            new_meta["customer_id"] = owner.get("customer_id", "") or ""
            patch_ids.append(uid)
            patch_metas.append(new_meta)
            if limit is not None and len(patch_ids) >= int(limit):
                break
        if patch_ids:
            try:
                col.update(ids=patch_ids, metadatas=patch_metas)
                out[key] += len(patch_ids)
            except Exception as e:
                out["errors"].append(f"{key}_update_failed: {e}")

    _patch(text_col, "text_updated")
    _patch(image_col, "image_updated")

    if out["text_updated"] or out["image_updated"]:
        logger.info(
            "Backfill tenant metadata: text=%d image=%d skipped=%d unmatched=%d",
            out["text_updated"], out["image_updated"],
            out["skipped"], out["unmatched"],
        )
    return out


def prune_vectors_to_active_set(
    active_file_names: Iterable[str],
    active_tenant_files: dict[str, set[str]] | None = None,
) -> dict:
    """Remove vector rows that no longer correspond to an active document.

    Two modes:

    1. Tenant-aware (preferred): pass `active_tenant_files` as
       {tenant_id: {file_name, ...}}. A row survives only if its
       (tenant_id, file_name) is present. Rows with empty tenant_id are
       still pruned via `active_file_names` membership (legacy data).

    2. Legacy: only `active_file_names` provided — file_name-only check.
       Kept for backward compatibility but unsafe in true multi-tenant
       deployments because two tenants can share a basename.

    Safety net used by reindex paths to clean up entries left over from
    deletes that pre-dated tenant metadata wiring.
    """
    active = {(n or "").strip() for n in active_file_names if n}
    tenant_active = active_tenant_files or {}
    out = {"text_removed": 0, "image_removed": 0, "errors": []}

    if not active and not tenant_active:
        # Defensive: refuse to wipe everything when caller passes nothing.
        logger.warning(
            "prune_vectors_to_active_set called with empty active set; skipping",
        )
        return out

    text_col, image_col = _open_collections()

    for col, key in ((text_col, "text_removed"), (image_col, "image_removed")):
        if col is None:
            continue
        try:
            existing = col.get(include=["metadatas"])
            ids = existing.get("ids") or []
            metas = existing.get("metadatas") or []
            stale_ids = []
            for i, uid in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                fname = ((meta or {}).get("file_name") or "").strip()
                if not fname:
                    continue
                row_tenant = ((meta or {}).get("tenant_id") or "").strip()
                if row_tenant and tenant_active:
                    # Row carries tenant metadata — judge against that
                    # tenant's active set only.
                    owned = tenant_active.get(row_tenant) or set()
                    if fname not in owned:
                        stale_ids.append(uid)
                else:
                    # Legacy / tenant-less row — fall back to global file_name
                    # membership, but only if we have one.
                    if active and fname not in active:
                        stale_ids.append(uid)
            if stale_ids:
                col.delete(ids=stale_ids)
                out[key] = len(stale_ids)
        except Exception as e:
            logger.warning("Prune failed for %s: %s", key, e)
            out["errors"].append(f"{key}: {e}")

    if out["text_removed"] or out["image_removed"]:
        logger.info(
            "Pruned stale vectors — text=%d image=%d (tenants=%d global_active=%d)",
            out["text_removed"], out["image_removed"],
            len(tenant_active), len(active),
        )
    return out
