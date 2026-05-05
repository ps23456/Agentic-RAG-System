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


def purge_file_from_vectors(file_name: str, doc_id: str | None = None) -> dict:
    """Delete all vector rows for a single file from text + image collections.

    Returns a dict with per-collection counts removed (best-effort, may be 0
    when Chroma cannot report counts).
    """
    out = {"text_removed": 0, "image_removed": 0, "errors": []}
    if not file_name:
        return out

    text_col, image_col = _open_collections()

    if text_col is not None:
        try:
            existing = text_col.get(
                where={"file_name": {"$eq": file_name}},
                include=[],
            )
            ids = list(existing.get("ids") or [])
            if ids:
                text_col.delete(ids=ids)
                out["text_removed"] = len(ids)
        except Exception as e:
            logger.warning("Text-collection purge failed for %s: %s", file_name, e)
            out["errors"].append(f"text: {e}")

    if image_col is not None:
        try:
            existing = image_col.get(
                where={"file_name": {"$eq": file_name}},
                include=[],
            )
            ids = list(existing.get("ids") or [])
            if ids:
                image_col.delete(ids=ids)
                out["image_removed"] = len(ids)
        except Exception as e:
            logger.warning("Image-collection purge failed for %s: %s", file_name, e)
            out["errors"].append(f"image: {e}")

    if out["text_removed"] or out["image_removed"]:
        logger.info(
            "Purged vectors for %s — text=%d image=%d",
            file_name, out["text_removed"], out["image_removed"],
        )
    return out


def prune_vectors_to_active_set(active_file_names: Iterable[str]) -> dict:
    """Remove vector rows whose file_name is NOT in `active_file_names`.

    Safety net used by reindex paths to clean up entries left over from
    deletes that pre-dated the purge wiring or external cleanup.
    """
    active = {(n or "").strip() for n in active_file_names if n}
    out = {"text_removed": 0, "image_removed": 0, "errors": []}
    if not active:
        # Defensive: refuse to wipe everything when caller passes no active
        # files. That would happen on a fresh tenant where no docs exist —
        # but we don't want a config bug to nuke the index.
        logger.warning("prune_vectors_to_active_set called with empty active set; skipping")
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
                fname = (meta or {}).get("file_name") or ""
                if fname and fname not in active:
                    stale_ids.append(uid)
            if stale_ids:
                col.delete(ids=stale_ids)
                out[key] = len(stale_ids)
        except Exception as e:
            logger.warning("Prune failed for %s: %s", key, e)
            out["errors"].append(f"{key}: {e}")

    if out["text_removed"] or out["image_removed"]:
        logger.info(
            "Pruned stale vectors — text=%d image=%d (active=%d)",
            out["text_removed"], out["image_removed"], len(active),
        )
    return out
