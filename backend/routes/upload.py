"""File upload endpoint."""
import logging
import os
import mimetypes
import threading
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi import Depends
from typing import List
from backend.db.tenant_store import tenant_store
from backend.security import require_scopes

logger = logging.getLogger(__name__)

router = APIRouter()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
DOC_EXTENSIONS = {".pdf", ".md", ".txt", ".json"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | DOC_EXTENSIONS


def _auto_index_enabled() -> bool:
    """Auto-trigger docs+images indexing after upload (default on).

    Set AUTO_INDEX_ON_UPLOAD=false to keep the legacy manual-index flow.
    """
    return os.environ.get("AUTO_INDEX_ON_UPLOAD", "true").strip().lower() != "false"


def _kickoff_post_upload_index(
    file_basenames: set[str],
    customer_id: str | None,
    tenant_id: str,
    user_id: str,
) -> bool:
    """Spawn a background thread that runs docs + images indexing for the
    files just uploaded.

    Returns True if the thread was started, False if another index job is
    already running (in which case the caller should rely on the next manual
    /api/index call to pick these files up — they're still on disk).
    """
    if not file_basenames:
        return False

    from backend.services.rag_service import rag

    if rag._indexing:
        logger.info(
            "Auto-index skipped: another index job is in progress (files=%s)",
            sorted(file_basenames),
        )
        return False

    def _run() -> None:
        try:
            tenant_store.update_documents_index_status(
                tenant_id=tenant_id,
                user_id=user_id,
                status="indexing",
                customer_id=customer_id,
                file_names=file_basenames,
                index_error="",
            )

            # Pass ALL uploaded basenames to both indexers. Each pipeline
            # only acts on the extensions it cares about (text_indexer skips
            # non-text/non-image files; image_indexer skips non-images and
            # non-PDFs internally), so this is safe and incremental.
            rag.reindex_docs(file_filter=set(file_basenames))
            docs_status = str(rag._index_status)

            rag.reindex_images(file_filter=set(file_basenames))
            img_status = str(rag._index_status)

            if docs_status.startswith("error") or img_status.startswith("error"):
                err = docs_status if docs_status.startswith("error") else img_status
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="failed",
                    customer_id=customer_id,
                    file_names=file_basenames,
                    index_error=err,
                )
            else:
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="indexed",
                    customer_id=customer_id,
                    file_names=file_basenames,
                    index_error="",
                )
        except Exception as e:
            logger.exception("Auto-index failed for files=%s", sorted(file_basenames))
            try:
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="failed",
                    customer_id=customer_id,
                    file_names=file_basenames,
                    index_error=str(e),
                )
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()
    return True


@router.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    customer_id: str = Form("default"),
    auth = Depends(require_scopes("docs:write")),
):
    from backend.services.rag_service import rag
    customer_slug = (customer_id or "").strip()
    if not customer_slug:
        customer_slug = "default"
    if any(ch in customer_slug for ch in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="Invalid customer_id")
    uploads_root = rag.uploads_folder
    tenant_dir = os.path.join(uploads_root, auth.tenant_slug, auth.user_id, customer_slug)
    os.makedirs(tenant_dir, exist_ok=True)

    saved: list[str] = []
    uploaded_docs: list[dict[str, str]] = []
    images: list[str] = []
    docs: list[str] = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        safe_name = os.path.basename(f.filename or "").strip()
        if not safe_name:
            continue
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        doc_dir = os.path.join(tenant_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        path = os.path.join(doc_dir, safe_name)
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)
        mime = (f.content_type or "").strip() or (mimetypes.guess_type(safe_name)[0] or "")
        stored_doc_id = tenant_store.upsert_document(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            customer_id=customer_slug,
            file_name=safe_name,
            size_bytes=len(content),
            storage_uri=path,
            mime_type=mime,
        )
        saved.append(safe_name)
        uploaded_docs.append(
            {
                "doc_id": stored_doc_id,
                "file_name": safe_name,
                "customer_id": customer_slug,
            }
        )
        if ext in IMAGE_EXTENSIONS:
            images.append(safe_name)
        elif ext in DOC_EXTENSIONS:
            docs.append(safe_name)

    # Auto-index the uploaded files so external callers don't need to remember
    # to call /api/index/docs and /api/index/images separately. Runs in a
    # background thread; the upload response returns immediately.
    auto_index_started = False
    if saved and _auto_index_enabled():
        auto_index_started = _kickoff_post_upload_index(
            file_basenames=set(saved),
            customer_id=customer_slug,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
        )

    return {
        "uploaded": saved,
        "uploaded_docs": uploaded_docs,
        "customer_id": customer_slug,
        "count": len(saved),
        "images": images,
        "docs": docs,
        "images_count": len(images),
        "docs_count": len(docs),
        "auto_index_started": auto_index_started,
    }
