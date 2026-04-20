"""
Image indexer: index standalone image files + images extracted from PDFs
into a separate Chroma collection (image_collection) with CLIP image embeddings only.
No text chunks in this collection; do not mix embedding spaces.
"""
import io
import logging
import os
from pathlib import Path
from typing import List, Any, Optional, Callable

from config import (
    DATA_FOLDER,
    CHROMA_PERSIST_DIR,
    CHROMA_IMAGE_COLLECTION_NAME,
    MULTIMODAL_CLIP_MODEL,
    PDF_EXTENSIONS,
    IMAGE_EXTENSIONS,
    STRUCTURED_DOC_MIN_PAGES,
)
from document_loader import (
    get_document_metadata_for_path,
    extract_text_from_image,
    extract_text_from_pdf,
    get_mistral_ocr_key,
)

_VLM_DESCRIPTION_MIN_OCR_LEN = 150

# Keywords that suggest a medical/insurance form (triggers section-header vision description)
_FORM_LIKE_KEYWORDS = ("patient", "diagnosis", "claim", "physician", "disability", "restriction", "primary", "secondary")


def _vlm_describe_image(pil_image) -> str:
    """Use Groq VLM to generate a text description of an image when OCR text is too short."""
    import base64, io, os
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return ""
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        import requests
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in detail: what objects, shapes, text, arrows, diagrams, tables, or visual elements are present? Be specific about layout and visual elements like arrows, flowcharts, boxes, connections. Keep under 200 words."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                "max_tokens": 300,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            desc = resp.json()["choices"][0]["message"]["content"]
            logger.info("VLM description generated (%d chars)", len(desc))
            return desc
    except Exception as e:
        logger.warning("VLM describe failed: %s", e)
    return ""


def _vlm_describe_form_sections(pil_image) -> str:
    """Use vision model to list section headers in a form – helps queries like 'primary diagnosis' find the right page."""
    import base64, io, os
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return ""
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        import requests
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "This image appears to be a medical or insurance form. List the main section headers and key fields visible (e.g. Primary Diagnosis, Secondary Diagnosis, Patient Name, Physical Capacities, restrictions, dates). Include exact phrases you see. Keep under 150 words."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                "max_tokens": 200,
            },
            timeout=25,
        )
        if resp.status_code == 200:
            desc = resp.json()["choices"][0]["message"]["content"]
            logger.info("VLM form sections generated (%d chars)", len(desc))
            return desc
    except Exception as e:
        logger.warning("VLM form sections failed: %s", e)
    return ""


def _looks_like_form(ocr_text: str) -> bool:
    """Heuristic: OCR contains form-like keywords."""
    if not ocr_text or len(ocr_text) < 20:
        return False
    lower = ocr_text.lower()
    return any(kw in lower for kw in _FORM_LIKE_KEYWORDS)


def _normalize_name(name: str) -> str:
    """Normalize to Title Case for consistent filtering (e.g. 'TERESA BROWN' → 'Teresa Brown')."""
    if not name:
        return ""
    return " ".join(w.capitalize() for w in name.strip().split())


# Map folder names (medical_reports subdirs) to display report types for ChromaDB filtering
_FOLDER_TO_REPORT_TYPE = {
    "xray": "X-ray",
    "mri": "MRI",
    "ct_scan": "CT Scan",
    "medical_report": "Medical Report",
    "ultrasound": "Ultrasound",
    "other": "Other",
}


def _extract_medical_report_metadata(path: str, data_folder: str) -> tuple[str, str]:
    """
    If path is under data/medical_reports/{patient}/{report_type}/, return (patient_display, report_type).
    Otherwise return ("", "").
    """
    path_norm = os.path.normpath(path)
    data_norm = os.path.normpath(data_folder)
    prefix = os.path.join(data_norm, "medical_reports")
    if not path_norm.startswith(prefix):
        return "", ""
    rel = os.path.relpath(path_norm, prefix)
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return "", ""
    patient_sanitized = parts[0]
    folder = parts[1].lower()
    report_type = _FOLDER_TO_REPORT_TYPE.get(folder, "")
    patient_display = _normalize_name(patient_sanitized.replace("_", " "))
    return patient_display, report_type

logger = logging.getLogger(__name__)

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_processor = CLIPProcessor.from_pretrained(MULTIMODAL_CLIP_MODEL)
        _clip_model = CLIPModel.from_pretrained(MULTIMODAL_CLIP_MODEL)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _collect_image_items(
    data_folder: str,
    existing_mtimes: dict[str, float] | None = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> List[dict]:
    """
    Collect indexable image items: image files + PDF pages rendered as images.
    Returns list of {"id", "file_name", "page", "path", "pil_image", "file_type", "is_pdf_page"}.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    items = []
    seen = set()
    processed = 0
    if not os.path.isdir(data_folder):
        return items

    for root, _dirs, files in os.walk(data_folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            ext = Path(path).suffix.lower()
            base = os.path.basename(path)
            
            # Incremental check
            mtime = os.path.getmtime(path)
            if existing_mtimes and base in existing_mtimes:
                if abs(existing_mtimes[base] - mtime) < 1.0:
                    logger.debug("Skipping unchanged image file: %s", base)
                    continue

            if ext in IMAGE_EXTENSIONS:
                try:
                    doc_meta = get_document_metadata_for_path(path)
                    ocr_pages = extract_text_from_image(path)
                    ocr_text = ocr_pages[0][1] if ocr_pages else ""
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        w, h = img.size
                        if max(w, h) > 1024:
                            img = img.resize((min(1024, w), min(1024, h)), Image.Resampling.LANCZOS)
                        uid = f"img_{base}_{1}".replace(" ", "_").replace(".", "_")
                        if uid in seen:
                            continue
                        seen.add(uid)
                        if len(ocr_text.strip()) < _VLM_DESCRIPTION_MIN_OCR_LEN:
                            vlm_desc = _vlm_describe_image(img)
                            if vlm_desc:
                                ocr_text = ocr_text + "\n\n" + vlm_desc if ocr_text.strip() else vlm_desc
                        elif _looks_like_form(ocr_text):
                            # Vision model lists section headers for form-like images – helps "primary diagnosis" etc.
                            form_desc = _vlm_describe_form_sections(img)
                            if form_desc:
                                ocr_text = ocr_text + "\n\n[Section headers: " + form_desc + "]"
                        patient_from_path, report_type = _extract_medical_report_metadata(path, data_folder)
                        patient_name = patient_from_path or (doc_meta.get("patient_name", "") or "")
                        items.append({
                            "id": uid,
                            "file_name": base,
                            "page": 1,
                            "path": path,
                            "pil_image": img.copy(),
                            "file_type": "image",
                            "is_pdf_page": False,
                            "ocr_text": ocr_text[:2000],
                            "patient_name": patient_name,
                            "report_type": report_type,
                            "claim_number": doc_meta.get("claim_number", "") or "",
                            "policy_number": doc_meta.get("policy_number", "") or "",
                            "group_number": doc_meta.get("group_number", "") or "",
                            "doctor_name": doc_meta.get("doctor_name", "") or "",
                            "last_modified": mtime,
                        })
                        processed += 1
                        if progress_callback:
                            pct = min(35, 10 + processed * 3)
                            progress_callback(pct, f"Processing {base}")
                except Exception as e:
                    logger.warning("Skip image %s: %s", path, e)

            elif ext in PDF_EXTENSIONS:
                try:
                    doc = fitz.open(path)
                    page_count = len(doc)
                    
                    if page_count >= STRUCTURED_DOC_MIN_PAGES:
                        logger.info("Skip image index for large PDF (%d pages): %s", page_count, base)
                        doc.close()
                        continue
                    
                    logger.info("Indexing PDF as images: %s (%d pages)", base, page_count)
                    doc_meta = get_document_metadata_for_path(path)
                    pdf_text_pages = extract_text_from_pdf(path)
                    pdf_text_map = {p: t for p, t in pdf_text_pages}
                    
                    for page_num in range(page_count):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(dpi=150, alpha=False)
                        buf = pix.tobytes("png")
                        pil_img = Image.open(io.BytesIO(buf)).convert("RGB")
                        w, h = pil_img.size
                        if max(w, h) > 1024:
                            pil_img = pil_img.resize((min(1024, w), min(1024, h)), Image.Resampling.LANCZOS)
                        uid = f"pdf_{base}_p{page_num + 1}".replace(" ", "_").replace(".", "_")
                        if uid in seen:
                            continue
                        seen.add(uid)
                        page_ocr = pdf_text_map.get(page_num + 1, "")
                        if len(page_ocr.strip()) < _VLM_DESCRIPTION_MIN_OCR_LEN:
                            vlm_desc = _vlm_describe_image(pil_img)
                            if vlm_desc:
                                page_ocr = page_ocr + "\n\n" + vlm_desc if page_ocr.strip() else vlm_desc
                        elif _looks_like_form(page_ocr):
                            form_desc = _vlm_describe_form_sections(pil_img)
                            if form_desc:
                                page_ocr = page_ocr + "\n\n[Section headers: " + form_desc + "]"
                        patient_from_path, report_type = _extract_medical_report_metadata(path, data_folder)
                        patient_name = patient_from_path or (doc_meta.get("patient_name", "") or "")
                        items.append({
                            "id": uid,
                            "file_name": base,
                            "page": page_num + 1,
                            "path": path,
                            "pil_image": pil_img,
                            "file_type": "pdf_page",
                            "is_pdf_page": True,
                            "ocr_text": page_ocr[:2000],
                            "patient_name": patient_name,
                            "report_type": report_type,
                            "claim_number": doc_meta.get("claim_number", "") or "",
                            "policy_number": doc_meta.get("policy_number", "") or "",
                            "doctor_name": doc_meta.get("doctor_name", "") or "",
                            "last_modified": mtime,
                        })
                        processed += 1
                        if progress_callback and processed % 3 == 0:
                            pct = min(35, 10 + processed * 2)
                            progress_callback(pct, f"Processing {base} (page {page_num + 1})")
                    doc.close()
                except Exception as e:
                    logger.warning("Skip PDF %s: %s", path, e)

    return items


def _encode_images(model, processor, images: List[Any]) -> List[List[float]]:
    """Encode PIL images with CLIP image encoder (L2 normalized for cosine in Chroma)."""
    import torch
    inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
    return outputs.cpu().numpy().tolist()


def build_image_index(
    data_folder: str | None = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> int:
    """
    Build the image-only Chroma collection (CLIP image embeddings).
    Does not index text; does not touch claim_chunks.
    Returns number of image items indexed.
    """
    data_folder = data_folder or DATA_FOLDER
    if progress_callback:
        progress_callback(2, "Loading CLIP model (first time may take 1-2 min)...")
    model, processor = _load_clip()
    if progress_callback:
        progress_callback(5, "Scanning for images...")

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError("chromadb required for image index. pip install chromadb")

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    # Remove delete_collection to allow incremental upsert
    collection = client.get_or_create_collection(
        name=CHROMA_IMAGE_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 1. Fetch existing metadata for incremental check
    existing_mtimes = {}
    try:
        results = collection.get(include=["metadatas"])
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                fname = meta.get("file_name")
                mtime = meta.get("last_modified")
                if fname and mtime is not None:
                    existing_mtimes[fname] = float(mtime)
    except Exception as e:
        logger.debug("Could not fetch existing image metadata: %s", e)

    # 2. Collect only new/changed items
    if progress_callback:
        progress_callback(8, "Collecting images...")
    image_items = _collect_image_items(
        data_folder, existing_mtimes=existing_mtimes, progress_callback=progress_callback
    )
    if image_items:
        from retrieval.agentic_rag import normalize_patient_names_in_items
        existing_patients = []
        try:
            existing = collection.get(include=["metadatas"])
            for m in (existing.get("metadatas") or []):
                pn = (m or {}).get("patient_name", "")
                if pn:
                    existing_patients.append(pn)
        except Exception:
            pass
        normalize_patient_names_in_items(image_items, existing_patient_names=existing_patients)
    if not image_items:
        logger.info("Image index: no image items found in %s", data_folder)
        if progress_callback:
            progress_callback(100, "No new images to index")
        return 0

    images = [x["pil_image"] for x in image_items]
    embeddings = _encode_images(model, processor, images)
    if progress_callback:
        progress_callback(85, "Encoding with CLIP...")
    ids = [x["id"] for x in image_items]
    metadatas = [
        {
            "file_name": x["file_name"],
            "page": x["page"],
            "path": x["path"],
            "doc_id": x["file_name"],
            "file_type": x["file_type"],
            "is_pdf_page": str(x.get("is_pdf_page", False)),
            "ocr_text": x.get("ocr_text", "") or "",
            "patient_name": _normalize_name(x.get("patient_name", "") or ""),
            "report_type": x.get("report_type", "") or "",
            "claim_number": x.get("claim_number", "") or "",
            "policy_number": x.get("policy_number", "") or "",
            "group_number": x.get("group_number", "") or "",
            "doctor_name": _normalize_name(x.get("doctor_name", "") or ""),
            "last_modified": float(x.get("last_modified", 0.0)),
        }
        for x in image_items
    ]
    if progress_callback:
        progress_callback(95, "Saving to index...")
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    if progress_callback:
        progress_callback(100, "Done")
    logger.info("Image index: added/updated %d item(s) to %s", len(ids), CHROMA_IMAGE_COLLECTION_NAME)
    return len(ids)


def get_image_index_count() -> int:
    """Return number of items in image_collection (0 if not built)."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        return 0
    if not os.path.isdir(CHROMA_PERSIST_DIR):
        return 0
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        coll = client.get_collection(name=CHROMA_IMAGE_COLLECTION_NAME)
        return coll.count()
    except Exception:
        return 0
