"""File upload endpoint."""
import logging
import os
import mimetypes
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

# ---- Upload limits (off by default; set UPLOAD_LIMITS_ENABLED=true to enforce) ----


def _upload_limits_enabled() -> bool:
    return os.environ.get("UPLOAD_LIMITS_ENABLED", "false").strip().lower() in (
        "true",
        "1",
        "yes",
    )


def _max_upload_files_per_request() -> int:
    try:
        n = int(os.environ.get("MAX_UPLOAD_FILES_PER_REQUEST", "3"))
        return max(1, min(n, 50))
    except ValueError:
        return 3


def _max_upload_bytes_per_file() -> int:
    try:
        n = int(os.environ.get("MAX_UPLOAD_BYTES_PER_FILE", str(30 * 1024 * 1024)))
        return max(1024, min(n, 500 * 1024 * 1024))
    except ValueError:
        return 30 * 1024 * 1024


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
) -> str | None:
    """Enqueue docs+images indexing for the files just uploaded (Gap 2 queue).

    Returns the new ``job_id`` when enqueued, or ``None`` on failure.
    Multiple uploads queue multiple jobs; the global worker runs them
    serially in FIFO order.
    """
    if not file_basenames:
        return None
    try:
        return tenant_store.create_index_job(
            tenant_id,
            user_id,
            "upload_auto",
            {
                "file_basenames": sorted(file_basenames),
                "customer_id": (customer_id or "").strip(),
            },
        )
    except Exception as e:
        logger.warning("enqueue upload_auto index job failed: %s", e)
        return None


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

    limits_on = _upload_limits_enabled()
    max_files = _max_upload_files_per_request()
    max_bytes = _max_upload_bytes_per_file()
    if limits_on and len(files) > max_files:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many files in one upload (max {max_files} per request). "
                "Split into multiple requests or disable UPLOAD_LIMITS_ENABLED."
            ),
        )

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
        if limits_on and len(content) > max_bytes:
            mb = max_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: {safe_name} ({len(content) / (1024 * 1024):.1f} MB). "
                    f"Maximum is {mb:.0f} MB per file when upload limits are enabled."
                ),
            )
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
    index_job_id: str | None = None
    auto_index_started = False
    if saved and _auto_index_enabled():
        index_job_id = _kickoff_post_upload_index(
            file_basenames=set(saved),
            customer_id=customer_slug,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
        )
        auto_index_started = bool(index_job_id)

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
        "index_job_id": index_job_id or "",
        "upload_limits_active": limits_on,
    }
