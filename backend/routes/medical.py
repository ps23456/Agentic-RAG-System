"""Medical analysis endpoints."""
import os
import re

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


def _medical_base():
    from backend.services.rag_service import rag
    base = os.path.join(rag.data_folder, "medical_reports")
    os.makedirs(base, exist_ok=True)
    return base


@router.get("/api/medical/patients")
async def list_patients():
    from backend.services.rag_service import rag
    patients = set(rag.get_patients())
    base = _medical_base()
    if os.path.isdir(base):
        for d in os.listdir(base):
            if os.path.isdir(os.path.join(base, d)):
                patients.add(d.replace("_", " "))
    return {"patients": sorted(patients)}


@router.get("/api/medical/files")
async def list_patient_files(patient: str = ""):
    """List medical file paths for a patient (for analyzing previously uploaded files)."""
    if not patient.strip():
        return {"paths": [], "files": []}
    base = _medical_base()
    safe = re.sub(r"[^\w\-]", "_", patient.strip())
    patient_dir = os.path.join(base, safe)
    paths = []
    files = []
    if os.path.isdir(patient_dir):
        for root, _, filenames in os.walk(patient_dir):
            for f in sorted(filenames):
                ext = os.path.splitext(f)[1].lower()
                if ext in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}:
                    full_path = os.path.join(root, f)
                    paths.append(full_path)
                    files.append({"path": full_path, "name": f, "page": 1})
    return {"paths": paths, "files": files}


@router.get("/api/medical/image")
async def serve_medical_image(path: str = Query(...)):
    """Serve a medical image or PDF page for preview. path = full server path (validated)."""
    base = _medical_base()
    base_real = os.path.realpath(base)
    path_clean = os.path.normpath(path)
    path_real = os.path.realpath(path_clean)
    if not path_real.startswith(base_real) or ".." in path:
        raise HTTPException(403, "Invalid path")
    if not os.path.isfile(path_real):
        raise HTTPException(404, "File not found")
    ext = os.path.splitext(path_real)[1].lower()
    if ext == ".pdf":
        try:
            import fitz
            from PIL import Image as PILImage
            doc = fitz.open(path_real)
            pix = doc.load_page(0).get_pixmap(dpi=120, alpha=False)
            buf = pix.tobytes("png")
            doc.close()
            return Response(content=buf, media_type="image/png")
        except Exception as e:
            raise HTTPException(500, str(e))
    media_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    media = media_map.get(ext, "application/octet-stream")
    with open(path_real, "rb") as f:
        return Response(content=f.read(), media_type=media)


class ClassifyRequest(BaseModel):
    image_path: str
    page: int = 1


@router.post("/api/medical/classify")
async def classify_document(req: ClassifyRequest):
    from backend.services.rag_service import rag
    from llm_insight import classify_medical_document
    key, provider = rag._llm_key_and_provider()
    doc_type, err = classify_medical_document(provider, key, req.image_path, req.page)
    if err:
        return {"error": err}
    return {"type": doc_type}


class AnalyzeRequest(BaseModel):
    query: str
    image_paths: List[str]
    pages: List[Optional[int]] = []
    patient: str = ""


@router.post("/api/medical/analyze")
async def analyze(req: AnalyzeRequest):
    from backend.services.rag_service import rag
    from llm_insight import medical_analysis
    key, provider = rag._llm_key_and_provider()

    history_ctx = ""
    if req.patient:
        safe = re.sub(r"[^\w\-]", "_", req.patient.strip())
        base = _medical_base()
        patient_dir = os.path.join(base, safe)
        if os.path.isdir(patient_dir):
            prev = []
            for root, _, files in os.walk(patient_dir):
                for f in sorted(files):
                    prev.append(os.path.join(root, f))
            if prev:
                history_ctx = f"Patient {req.patient} has {len(prev)} previous reports on file."

    pages = req.pages if req.pages else [None] * len(req.image_paths)
    report, err = medical_analysis(provider, key, req.query, req.image_paths, pages, history_ctx)
    if err:
        return {"error": err}
    return {"report": report}


@router.post("/api/medical/upload")
async def upload_medical(
    patient: str = Form(...),
    doc_type: str = Form("Medical Report"),
    files: List[UploadFile] = File(...),
):
    base = _medical_base()
    safe_patient = re.sub(r"[^\w\-]", "_", patient.strip())
    safe_type = re.sub(r"[^\w\-]", "_", doc_type.strip())
    dest = os.path.join(base, safe_patient, safe_type)
    os.makedirs(dest, exist_ok=True)

    saved = []
    for f in files:
        path = os.path.join(dest, f.filename)
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)
        saved.append(path)

    return {"saved": saved, "patient": patient, "type": doc_type}
