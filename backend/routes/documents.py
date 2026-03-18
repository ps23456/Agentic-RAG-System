"""Document serving: list files, render PDF pages, serve text/images."""
import base64
import io
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter()


def _find_file(file_name: str) -> str | None:
    from backend.services.rag_service import rag
    for root, _dirs, files in os.walk(rag.data_folder):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


@router.get("/api/documents")
async def list_documents():
    from backend.services.rag_service import rag
    uploads = rag.uploads_folder
    if not os.path.isdir(uploads):
        return {"files": []}
    files = []
    for f in sorted(os.listdir(uploads)):
        path = os.path.join(uploads, f)
        if os.path.isfile(path):
            ext = os.path.splitext(f)[1].lower()
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
async def get_document_text(file: str = Query(...), search: str = Query("")):
    """Return text/markdown file content with optional search highlight offset."""
    path = _find_file(file)
    if not path:
        raise HTTPException(404, f"File not found: {file}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        scroll_line = 0
        if search:
            lower = content.lower()
            idx = lower.find(search.lower())
            if idx >= 0:
                scroll_line = content[:idx].count("\n")
        return {"content": content, "file_name": file, "scroll_line": scroll_line}
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
