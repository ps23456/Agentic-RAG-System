"""File upload endpoint."""
import os
import mimetypes
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi import Depends
from typing import List
from backend.db.tenant_store import tenant_store
from backend.security import require_scopes

router = APIRouter()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
DOC_EXTENSIONS = {".pdf", ".md", ".txt", ".json"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | DOC_EXTENSIONS


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

    return {
        "uploaded": saved,
        "uploaded_docs": uploaded_docs,
        "customer_id": customer_slug,
        "count": len(saved),
        "images": images,
        "docs": docs,
        "images_count": len(images),
        "docs_count": len(docs),
    }
