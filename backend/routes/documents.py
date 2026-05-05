"""Document serving: list files, render PDF pages, serve text/images."""
import base64
import json
import io
import os
import re
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from backend.db.tenant_store import tenant_store
from backend.security import require_scopes

# Devanagari Unicode block U+0900 to U+097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _is_latin_preferred(text_window: str, threshold: float = 0.35) -> bool:
    """True if the text is mostly Latin/ASCII (not predominantly Devanagari)."""
    if not text_window or not text_window.strip():
        return True
    n_devanagari = len(_DEVANAGARI_RE.findall(text_window))
    n_total = len([c for c in text_window if not c.isspace()])
    if n_total == 0:
        return True
    return (n_devanagari / n_total) < threshold


router = APIRouter()


def _find_file(file_name: str) -> str | None:
    from backend.services.rag_service import rag
    for root, _dirs, files in os.walk(rag.data_folder):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


def _require_owned_document(auth, file_name: str, customer_id: str | None = None) -> None:
    if not tenant_store.has_document_access(auth.tenant_id, auth.user_id, file_name, customer_id=customer_id):
        raise HTTPException(404, f"File not found: {file_name}")


def _owned_path(auth, file_name: str, customer_id: str | None = None) -> str:
    row = tenant_store.get_document_for_owner(auth.tenant_id, auth.user_id, file_name, customer_id=customer_id)
    if row and row.get("storage_uri"):
        p = row["storage_uri"]
        if os.path.isfile(p):
            return p
    # Backward-compat fallback for older rows without storage URI.
    path = _find_file(file_name)
    if not path:
        raise HTTPException(404, f"File not found: {file_name}")
    return path


def _resolve_doc(auth, file_name: str | None, doc_id: str | None, customer_id: str | None) -> dict:
    if doc_id:
        row = tenant_store.get_document_by_id_for_owner(
            auth.tenant_id,
            auth.user_id,
            doc_id=doc_id,
            customer_id=customer_id,
        )
        if not row:
            raise HTTPException(404, f"Document not found: {doc_id}")
        return row
    if not file_name:
        raise HTTPException(400, "Provide either file or doc_id")
    row = tenant_store.get_document_for_owner(
        auth.tenant_id,
        auth.user_id,
        file_name=file_name,
        customer_id=customer_id,
    )
    if not row:
        raise HTTPException(404, f"File not found: {file_name}")
    return row


@router.delete("/api/documents")
async def delete_document(
    file: str = Query(""),
    doc_id: str = Query(""),
    customer_id: str = Query(""),
    auth=Depends(require_scopes("docs:write")),
):
    """Delete an uploaded document by filename."""
    if file and (".." in file or "/" in file or "\\" in file):
        raise HTTPException(400, "Invalid filename")
    cid = customer_id.strip() or None
    row = _resolve_doc(auth, (file or None), (doc_id.strip() or None), cid)
    path = row.get("storage_uri") or _owned_path(auth, row["file_name"], customer_id=cid)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    try:
        os.remove(path)
        # Best-effort empty directory cleanup for doc_id storage layout.
        removed_dirs = 0
        parent = os.path.dirname(path)
        for _ in range(3):
            if not parent:
                break
            try:
                os.rmdir(parent)
                removed_dirs += 1
                parent = os.path.dirname(parent)
            except OSError:
                break
        if row.get("id"):
            tenant_store.soft_delete_document_by_id(auth.tenant_id, auth.user_id, row["id"], customer_id=cid)
        else:
            tenant_store.soft_delete_document(auth.tenant_id, auth.user_id, row["file_name"], customer_id=cid)

        # Purge vectors so retrieval no longer surfaces this file.
        # Best-effort: failures are logged inside the helper but never raised.
        try:
            from indexing.vector_cleanup import purge_file_from_vectors
            purge_result = purge_file_from_vectors(row["file_name"], doc_id=row.get("id"))
        except Exception:
            purge_result = {"text_removed": 0, "image_removed": 0, "errors": ["purge_helper_unavailable"]}

        # Refresh in-memory text-index chunks so get_index_info() reports the
        # post-delete chunk count immediately. Image count is read from Chroma
        # on demand by rag_service.get_index_info, so it self-heals.
        try:
            from backend.services.rag_service import rag
            rag.drop_text_chunks_for_file(row["file_name"])
        except Exception:
            pass

        audit_id = tenant_store.record_delete_audit(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            customer_id=cid or "",
            document_id=row.get("id", "") or "",
            file_name=row["file_name"],
            result="success",
            details_json=json.dumps(
                {
                    "storage_uri": path,
                    "removed_empty_dirs": removed_dirs,
                    "vectors_text_removed": purge_result.get("text_removed", 0),
                    "vectors_image_removed": purge_result.get("image_removed", 0),
                },
                ensure_ascii=True,
            ),
        )
        return {
            "deleted": row["file_name"],
            "doc_id": row.get("id", ""),
            "audit_id": audit_id,
            "storage_removed": True,
            "removed_empty_dirs": removed_dirs,
            "vectors_removed": {
                "text": purge_result.get("text_removed", 0),
                "image": purge_result.get("image_removed", 0),
            },
        }
    except OSError as e:
        tenant_store.record_delete_audit(
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            customer_id=cid or "",
            document_id=row.get("id", "") or "",
            file_name=row["file_name"],
            result="failed",
            details_json=json.dumps({"error": str(e)}, ensure_ascii=True),
        )
        raise HTTPException(500, str(e))


@router.get("/api/documents/mistral-ocr-md")
async def download_mistral_ocr_markdown(
    file: str = Query(""),
    doc_id: str = Query(""),
    customer_id: str = Query(""),
    auth=Depends(require_scopes("docs:read")),
):
    """
    Run Mistral OCR on an uploaded PDF and return markdown as a download.
    User can re-upload the .md next to the .pdf (same basename) so indexing uses this text.
    """
    if file and (".." in file or "/" in file or "\\" in file):
        raise HTTPException(400, "Invalid filename")
    cid = customer_id.strip() or None
    row = _resolve_doc(auth, (file or None), (doc_id.strip() or None), cid)
    file = row["file_name"]
    if not file.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    path = row.get("storage_uri") or _owned_path(auth, file, customer_id=cid)
    try:
        from document_loader import mistral_ocr_pdf_to_markdown

        md = mistral_ocr_pdf_to_markdown(path)
    except ValueError as e:
        raise HTTPException(503, str(e)) from e
    stem = os.path.splitext(file)[0]
    out_name = f"{stem}.md"
    ascii_name = out_name.encode("ascii", "replace").decode("ascii")
    cd = f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(out_name)}'
    return Response(
        content=md.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": cd},
    )


# Allowed extensions in the documents page; anything else (dotfiles, .DS_Store, .gitkeep, ...) is hidden.
_LISTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp", ".md", ".txt", ".json"}


@router.get("/api/documents")
async def list_documents(
    customer_id: str = Query(""),
    auth=Depends(require_scopes("docs:read")),
):
    cid = customer_id.strip() or None
    files = []
    for row in tenant_store.list_documents_for_owner(auth.tenant_id, auth.user_id, customer_id=cid):
        f = row["file_name"]
        path = row["storage_uri"] or _find_file(f) or ""
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in _LISTED_EXTENSIONS:
            continue
        files.append({
            "doc_id": row.get("id", ""),
            "name": f,
            "size": os.path.getsize(path),
            "type": ext,
            "customer_id": row.get("customer_id") or "default",
            "index_status": row.get("index_status") or "pending",
            "index_error": row.get("index_error") or "",
            "indexed_at": row.get("indexed_at"),
        })
    return {"files": files}


@router.get("/api/documents/info")
async def document_info():
    from backend.services.rag_service import rag
    return rag.get_index_info()


@router.get("/api/documents/page")
async def get_document_page(
    file: str = Query(""),
    doc_id: str = Query(""),
    page: int = Query(1),
    customer_id: str = Query(""),
    auth=Depends(require_scopes("docs:read")),
):
    """Render a PDF page as PNG and return base64."""
    cid = customer_id.strip() or None
    row = _resolve_doc(auth, (file or None), (doc_id.strip() or None), cid)
    file = row["file_name"]
    path = row.get("storage_uri") or _owned_path(auth, file, customer_id=cid)

    ext = os.path.splitext(file)[1].lower()
    if ext != ".pdf":
        raise HTTPException(400, "Only PDF pages can be rendered")

    try:
        import fitz
        doc = fitz.open(path)
        total = len(doc)
        pg_idx = max(0, min(page - 1, total - 1))
        pix = doc.load_page(pg_idx).get_pixmap(dpi=150, alpha=False)
        buf = pix.tobytes("png")
        doc.close()
        img_b64 = base64.b64encode(buf).decode()
        return {"image": img_b64, "page": pg_idx + 1, "total_pages": total}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/documents/text")
async def get_document_text(
    file: str = Query(""),
    doc_id: str = Query(""),
    search: str = Query(""),
    page: int = Query(1),
    customer_id: str = Query(""),
    auth=Depends(require_scopes("docs:read")),
):
    """Return text/markdown file content with optional search highlight offset.
    For PDF: use page to pick match closest to cited location.
    For .md/.txt: page does NOT map to lines—use FIRST relevant match (avoids wrong scroll to end)."""
    cid = customer_id.strip() or None
    row = _resolve_doc(auth, (file or None), (doc_id.strip() or None), cid)
    file = row["file_name"]
    path = row.get("storage_uri") or _owned_path(auth, file, customer_id=cid)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        scroll_line = 0
        matched_exact = ""
        ext = os.path.splitext(file)[1].lower()
        is_pdf_like = ext == ".pdf"  # Only PDF has physical pages; .md/.txt use logical sections

        if search:
            lower = content.lower()
            parts = [p.strip() for p in search.strip().split("\n") if p.strip()]
            exact_phrases = parts[:1] if len(parts) > 1 else []
            fallback_phrases = parts[1:] if len(parts) > 1 else parts
            for phase, phrases in enumerate([exact_phrases, fallback_phrases]):
                for phrase in phrases:
                    if len(phrase) < 8:
                        continue
                    use_exact_only = phase == 0 and exact_phrases
                    try_order = [phrase] if use_exact_only else [phrase, phrase[:80], phrase[:60], phrase[:40]]
                    for attempt in try_order:
                        if len(attempt) < 8:
                            continue
                        needle = attempt.lower()
                        all_matches = []
                        start = 0
                        while True:
                            idx = lower.find(needle, start)
                            if idx < 0:
                                break
                            ln = content[:idx].count("\n")
                            all_matches.append((idx, ln, content[idx : idx + len(attempt)]))
                            start = idx + 1
                        if all_matches:
                            window_len = 80
                            latin_matches = [
                                m
                                for m in all_matches
                                if _is_latin_preferred(
                                    content[max(0, m[0] - window_len) : m[0]]
                                )
                            ]
                            candidates = latin_matches if latin_matches else all_matches
                            if len(candidates) == 1:
                                scroll_line = candidates[0][1]
                                matched_exact = candidates[0][2]
                            else:
                                # PDF: page N -> ~line (N-1)*20; .md/.txt: use FIRST match (page is logical, not line-based)
                                if is_pdf_like:
                                    target_line = (page - 1) * 20
                                    best = min(
                                        candidates,
                                        key=lambda m: abs(m[1] - target_line),
                                    )
                                else:
                                    best = candidates[0]  # First match = earliest occurrence
                                scroll_line = best[1]
                                matched_exact = best[2]
                            break
                    if matched_exact:
                        break
                if matched_exact:
                    break
        return {
            "content": content,
            "file_name": file,
            "scroll_line": scroll_line,
            "matched_text": matched_exact,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/documents/image")
async def get_document_image(
    file: str = Query(""),
    doc_id: str = Query(""),
    customer_id: str = Query(""),
    auth=Depends(require_scopes("docs:read")),
):
    """Serve an image file as binary response."""
    cid = customer_id.strip() or None
    row = _resolve_doc(auth, (file or None), (doc_id.strip() or None), cid)
    file = row["file_name"]
    path = row.get("storage_uri") or _owned_path(auth, file, customer_id=cid)
    ext = os.path.splitext(file)[1].lower()
    media_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff"}
    media = media_map.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=media)
