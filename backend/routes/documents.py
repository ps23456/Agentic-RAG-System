"""Document serving: list files, render PDF pages, serve text/images."""
import base64
import io
import os
import re
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

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


@router.delete("/api/documents")
async def delete_document(file: str = Query(...)):
    """Delete an uploaded document by filename."""
    from backend.services.rag_service import rag
    if ".." in file or "/" in file or "\\" in file:
        raise HTTPException(400, "Invalid filename")
    uploads = rag.uploads_folder
    path = os.path.join(uploads, file)
    if not os.path.isfile(path):
        raise HTTPException(404, f"File not found: {file}")
    try:
        os.remove(path)
        return {"deleted": file}
    except OSError as e:
        raise HTTPException(500, str(e))


@router.get("/api/documents/mistral-ocr-md")
async def download_mistral_ocr_markdown(file: str = Query(...)):
    """
    Run Mistral OCR on an uploaded PDF and return markdown as a download.
    User can re-upload the .md next to the .pdf (same basename) so indexing uses this text.
    """
    if ".." in file or "/" in file or "\\" in file:
        raise HTTPException(400, "Invalid filename")
    if not file.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    path = _find_file(file)
    if not path:
        raise HTTPException(404, f"File not found: {file}")
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
async def list_documents():
    from backend.services.rag_service import rag
    uploads = rag.uploads_folder
    if not os.path.isdir(uploads):
        return {"files": []}
    files = []
    for f in sorted(os.listdir(uploads)):
        if f.startswith("."):
            continue
        path = os.path.join(uploads, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in _LISTED_EXTENSIONS:
            continue
        files.append({
            "name": f,
            "size": os.path.getsize(path),
            "type": ext,
        })
    return {"files": files}


@router.get("/api/documents/info")
async def document_info():
    from backend.services.rag_service import rag
    return rag.get_index_info()


@router.get("/api/documents/page")
async def get_document_page(
    file: str = Query(...),
    page: int = Query(1),
):
    """Render a PDF page as PNG and return base64."""
    path = _find_file(file)
    if not path:
        raise HTTPException(404, f"File not found: {file}")

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
    file: str = Query(...),
    search: str = Query(""),
    page: int = Query(1),
):
    """Return text/markdown file content with optional search highlight offset.
    For PDF: use page to pick match closest to cited location.
    For .md/.txt: page does NOT map to lines—use FIRST relevant match (avoids wrong scroll to end)."""
    path = _find_file(file)
    if not path:
        raise HTTPException(404, f"File not found: {file}")
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
async def get_document_image(file: str = Query(...)):
    """Serve an image file as binary response."""
    path = _find_file(file)
    if not path:
        raise HTTPException(404, f"File not found: {file}")
    ext = os.path.splitext(file)[1].lower()
    media_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff"}
    media = media_map.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type=media)
