"""Indexing endpoints: trigger reindex (docs, images, or both), check status."""
import threading
from fastapi import APIRouter

router = APIRouter()


@router.post("/api/index")
async def trigger_reindex():
    """Re-index everything (docs + images)."""
    from backend.services.rag_service import rag
    if rag._indexing:
        return {"status": "already_indexing"}
    thread = threading.Thread(target=rag.reindex, daemon=True)
    thread.start()
    return {"status": "started"}


@router.post("/api/index/docs")
async def trigger_reindex_docs():
    """Re-index documents only (text chunks + trees)."""
    from backend.services.rag_service import rag
    if rag._indexing:
        return {"status": "already_indexing"}
    thread = threading.Thread(target=rag.reindex_docs, daemon=True)
    thread.start()
    return {"status": "started"}


@router.post("/api/index/images")
async def trigger_reindex_images():
    """Re-index images only."""
    from backend.services.rag_service import rag
    if rag._indexing:
        return {"status": "already_indexing"}
    thread = threading.Thread(target=rag.reindex_images, daemon=True)
    thread.start()
    return {"status": "started"}


@router.get("/api/index/status")
async def index_status():
    from backend.services.rag_service import rag
    return rag.get_index_info()
