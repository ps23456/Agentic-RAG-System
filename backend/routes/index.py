"""Indexing endpoints: trigger reindex (docs, images, or both), check status."""
import threading
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from backend.security import require_scopes

router = APIRouter()


class IndexRequest(BaseModel):
    """Optional body for POST /api/index/* endpoints.

    `files`: when provided, only these basenames are (re)indexed; the rest of the
    existing index is preserved untouched. When omitted or empty, the endpoint
    performs the usual incremental scan over the entire data folder.
    """
    files: Optional[List[str]] = None
    customer_id: Optional[str] = None


def _normalize_filter(body: Optional[IndexRequest]) -> Optional[set[str]]:
    if body and body.files:
        names = {f for f in (body.files or []) if isinstance(f, str) and f}
        return names or None
    return None


@router.post("/api/index")
async def trigger_reindex(_auth = Depends(require_scopes("index:run"))):
    """Re-index everything (docs + images)."""
    from backend.services.rag_service import rag
    if rag._indexing:
        return {"status": "already_indexing"}
    thread = threading.Thread(target=rag.reindex, daemon=True)
    thread.start()
    return {"status": "started"}


@router.post("/api/index/docs")
async def trigger_reindex_docs(
    body: Optional[IndexRequest] = None,
    auth = Depends(require_scopes("index:run")),
):
    """Re-index documents only (text chunks + trees).

    Accepts optional JSON body: ``{ "files": ["foo.pdf", "bar.md"] }`` to index just
    those files. When omitted, performs a full incremental re-scan.
    """
    from backend.services.rag_service import rag
    from backend.db.tenant_store import tenant_store

    if rag._indexing:
        return {"status": "already_indexing"}
    file_filter = _normalize_filter(body)
    customer_id = (body.customer_id.strip() if body and body.customer_id else "") or None
    tenant_store.update_documents_index_status(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        status="indexing",
        customer_id=customer_id,
        file_names=file_filter,
        index_error="",
    )

    def _run():
        try:
            rag.reindex_docs(file_filter=file_filter)
            if str(rag._index_status).startswith("error"):
                tenant_store.update_documents_index_status(
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    status="failed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error=str(rag._index_status),
                )
            else:
                tenant_store.update_documents_index_status(
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    status="indexed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error="",
                )
        except Exception as e:
            tenant_store.update_documents_index_status(
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
                status="failed",
                customer_id=customer_id,
                file_names=file_filter,
                index_error=str(e),
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "targeted": bool(file_filter), "count": len(file_filter) if file_filter else 0}


@router.post("/api/index/images")
async def trigger_reindex_images(
    body: Optional[IndexRequest] = None,
    auth = Depends(require_scopes("index:run")),
):
    """Re-index images only.

    Accepts optional JSON body: ``{ "files": ["photo.png"] }`` to index just those
    files. When omitted, performs a full incremental re-scan.
    """
    from backend.services.rag_service import rag
    from backend.db.tenant_store import tenant_store

    if rag._indexing:
        return {"status": "already_indexing"}
    file_filter = _normalize_filter(body)
    customer_id = (body.customer_id.strip() if body and body.customer_id else "") or None
    tenant_store.update_documents_index_status(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        status="indexing",
        customer_id=customer_id,
        file_names=file_filter,
        index_error="",
    )

    def _run():
        try:
            rag.reindex_images(file_filter=file_filter)
            if str(rag._index_status).startswith("error"):
                tenant_store.update_documents_index_status(
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    status="failed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error=str(rag._index_status),
                )
            else:
                tenant_store.update_documents_index_status(
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    status="indexed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error="",
                )
        except Exception as e:
            tenant_store.update_documents_index_status(
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
                status="failed",
                customer_id=customer_id,
                file_names=file_filter,
                index_error=str(e),
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "targeted": bool(file_filter), "count": len(file_filter) if file_filter else 0}


@router.get("/api/index/status")
async def index_status():
    from backend.services.rag_service import rag
    return rag.get_index_info()


class BackfillCaptionsRequest(BaseModel):
    """Body for POST /api/index/backfill_captions.

    `customer_id`: required-ish — when omitted, scope is the caller's full
        tenant/user storage directory.
    `files`: when set, only patch rows whose file_name is in this list.
    `force`: re-caption rows that already have auto_caption set.
    `include_pdf_pages`: include rows from PDF pages (off by default — those
        usually have rich OCR text and captioning is expensive).
    `limit`: cap on rows updated in this call. Useful to keep request time
        bounded; call repeatedly to drain remaining candidates.
    """
    customer_id: Optional[str] = None
    files: Optional[List[str]] = None
    force: Optional[bool] = False
    include_pdf_pages: Optional[bool] = False
    limit: Optional[int] = None


@router.post("/api/index/backfill_captions")
async def trigger_backfill_captions(
    body: Optional[BackfillCaptionsRequest] = None,
    auth=Depends(require_scopes("index:run")),
):
    """Re-run auto-caption on already-indexed images and patch their
    `auto_caption` metadata in place. Does NOT re-encode CLIP embeddings.

    Use after enabling IMAGE_AUTO_CAPTION=true so existing image rows pick
    up captions without needing a re-upload or full reindex. Runs in a
    background thread; the response returns immediately. Inspect logs or
    call again with the same body to see remaining candidates drop.
    """
    import os as _os
    from backend.services.rag_service import rag
    from backend.db.tenant_store import tenant_store

    customer_id = (
        body.customer_id.strip() if body and body.customer_id else ""
    ) or None
    file_set: set[str] | None = None
    if body and body.files:
        file_set = {f for f in body.files if isinstance(f, str) and f}
        if not file_set:
            file_set = None

    # Restrict to this tenant's storage directory so a backfill request
    # can never patch another tenant's image rows.
    uploads_root = rag.uploads_folder
    tenant_slug = getattr(auth, "tenant_slug", "") or ""
    user_id = getattr(auth, "user_id", "") or ""
    if customer_id:
        prefix = _os.path.join(
            uploads_root, tenant_slug, user_id, customer_id
        )
    else:
        prefix = _os.path.join(uploads_root, tenant_slug, user_id)
    # Trailing separator so /tenant/userA/foo doesn't match /tenant/userAB/...
    prefix = prefix.rstrip(_os.sep) + _os.sep

    # Cross-check against the tenant's own document table so we never
    # operate on a row whose file is no longer owned by this caller.
    if file_set is None:
        owned = tenant_store.list_file_names_for_owner(
            auth.tenant_id, auth.user_id, customer_id=customer_id
        )
        # If the tenant has no documents, there's nothing to do.
        if not owned:
            return {"status": "nothing_to_do", "scope": prefix}
        file_set = owned
    else:
        owned = tenant_store.list_file_names_for_owner(
            auth.tenant_id, auth.user_id, customer_id=customer_id
        )
        # Drop any file the caller listed that they don't actually own.
        file_set = {f for f in file_set if f in owned}
        if not file_set:
            return {"status": "nothing_to_do", "scope": prefix}

    force = bool(body.force) if body else False
    include_pdf_pages = bool(body.include_pdf_pages) if body else False
    limit = body.limit if body else None

    def _run() -> dict | None:
        try:
            from indexing.image_indexer import backfill_auto_captions
            result = backfill_auto_captions(
                file_names=file_set,
                allowed_path_prefix=prefix,
                include_pdf_pages=include_pdf_pages,
                force=force,
                limit=limit,
            )
            try:
                rag._last_backfill_result = result
            except Exception:
                pass
            return result
        except Exception as e:
            try:
                rag._last_backfill_result = {"errors": [str(e)]}
            except Exception:
                pass
            return None

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {
        "status": "started",
        "scope": prefix,
        "files_in_scope": len(file_set),
        "force": force,
        "include_pdf_pages": include_pdf_pages,
        "limit": limit,
        "hint": "Re-call this endpoint to see candidates drop, or check rag._last_backfill_result via logs.",
    }
