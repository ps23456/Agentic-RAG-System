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
