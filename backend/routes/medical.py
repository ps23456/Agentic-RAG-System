"""Medical analysis endpoints."""
import os
import re
from fastapi import APIRouter, UploadFile, File, Form
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
