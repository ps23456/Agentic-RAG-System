"""File upload endpoint."""
import os
from fastapi import APIRouter, UploadFile, File
from typing import List

router = APIRouter()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
DOC_EXTENSIONS = {".pdf", ".md", ".txt", ".json"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | DOC_EXTENSIONS


@router.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    from backend.services.rag_service import rag
    uploads = rag.uploads_folder
    os.makedirs(uploads, exist_ok=True)

    saved: list[str] = []
    images: list[str] = []
    docs: list[str] = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        path = os.path.join(uploads, f.filename)
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)
        saved.append(f.filename)
        if ext in IMAGE_EXTENSIONS:
            images.append(f.filename)
        elif ext in DOC_EXTENSIONS:
            docs.append(f.filename)

    return {
        "uploaded": saved,
        "count": len(saved),
        "images": images,
        "docs": docs,
        "images_count": len(images),
        "docs_count": len(docs),
    }
