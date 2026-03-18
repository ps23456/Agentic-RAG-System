"""File upload endpoint."""
import os
from fastapi import APIRouter, UploadFile, File
from typing import List

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".md", ".txt", ".json"}


@router.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    from backend.services.rag_service import rag
    uploads = rag.uploads_folder
    os.makedirs(uploads, exist_ok=True)

    saved = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        path = os.path.join(uploads, f.filename)
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)
        saved.append(f.filename)

    return {"uploaded": saved, "count": len(saved)}
