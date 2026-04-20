"""
Document loader: read files from local folder, extract text (PDF or OCR), chunk.
"""
import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterator, Callable, Optional

import pdfplumber
import fitz  # PyMuPDF for fallback and page images for OCR
import json
from config import DATA_FOLDER, MISTRAL_OCR_FORCE_FILENAMES, STRUCTURED_DOC_MIN_PAGES

# Lazy OCR: don't import pytesseract at module level (it can crash the process on some systems).
# We'll try to enable OCR when first needed.
OCR_AVAILABLE = None  # None = not tried yet, True/False after _check_ocr()

# Mistral OCR: set via set_mistral_ocr_key() from the app before indexing.
_MISTRAL_OCR_KEY: str = ""

def _is_mistral_disabled() -> bool:
    """Check if Mistral OCR is persistently disabled for this run (due to 401)."""
    return os.path.exists(".mistral_disabled")

def _disable_mistral_persistently():
    """Disable Mistral OCR persistently across processes for this run."""
    try:
        with open(".mistral_disabled", "w") as f:
            f.write("disabled")
    except:
        pass

def _reset_mistral_status():
    """Re-enable Mistral OCR (call when key changes)."""
    if os.path.exists(".mistral_disabled"):
        try:
            os.remove(".mistral_disabled")
        except:
            pass

def _find_tesseract_cmd() -> str | None:
    """Try common install paths so OCR works even if tesseract is not on PATH."""
    import shutil
    candidates = [
        "tesseract",  # on PATH
        "/opt/homebrew/bin/tesseract",  # Apple Silicon Homebrew
        "/usr/local/bin/tesseract",  # Intel Homebrew / Linux
    ]
    for cmd in candidates:
        if cmd == "tesseract":
            if shutil.which("tesseract"):
                return "tesseract"
        elif os.path.isfile(cmd) and os.access(cmd, os.X_OK):
            return cmd
    return None


def _check_ocr() -> bool:
    global OCR_AVAILABLE
    if OCR_AVAILABLE is not None:
        return OCR_AVAILABLE
    try:
        import pytesseract
        # from pdf2image import convert_from_path  # noqa: F401  # poppler usage commented out
        tesseract_cmd = _find_tesseract_cmd()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # Quick test that tesseract runs
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
    except Exception:
        OCR_AVAILABLE = False
    return bool(OCR_AVAILABLE)


def get_ocr_status() -> tuple[bool, str]:
    """Return (available, message) for UI."""
    if _MISTRAL_OCR_KEY:
        return True, "Mistral OCR is active (cloud API). High-accuracy parsing for images and PDFs."
    ok = _check_ocr()
    if ok:
        return True, "OCR (Tesseract) is available. Image text will be indexed."
    return False, "Install Tesseract to index text from images (e.g. brew install tesseract on macOS), then re-index."


def set_mistral_ocr_key(key: str):
    """Set the Mistral OCR API key (call before indexing)."""
    global _MISTRAL_OCR_KEY
    new_key = (key or "").strip()
    if new_key != _MISTRAL_OCR_KEY:
        _MISTRAL_OCR_KEY = new_key
        _reset_mistral_status()


def _ensure_mistral_key_from_env() -> None:
    """If the key was never set (e.g. CLI index without Streamlit), read MISTRAL_OCR_API_KEY."""
    if _MISTRAL_OCR_KEY:
        return
    k = os.environ.get("MISTRAL_OCR_API_KEY", "").strip()
    if k:
        set_mistral_ocr_key(k)


def _should_force_mistral_ocr(path: str) -> bool:
    """
    Force full-document Mistral OCR for PDFs that mix printed template text with handwriting:
    normal heuristics keep enough embedded text that per-page OCR is skipped.

    - Sidecar: same path + '.mistralocr' or '{stem}.mistralocr' next to the file (empty file is ok).
    - Env MISTRAL_OCR_FORCE_FILENAMES: comma-separated basenames (case-insensitive).
    """
    base = os.path.basename(path)
    stem = Path(path).stem
    if os.path.isfile(path + ".mistralocr"):
        return True
    side2 = os.path.join(os.path.dirname(path) or ".", f"{stem}.mistralocr")
    if os.path.isfile(side2):
        return True
    raw = (MISTRAL_OCR_FORCE_FILENAMES or "").strip()
    if not raw:
        return False
    base_lower = base.lower()
    for part in raw.split(","):
        p = part.strip()
        if p and p.lower() == base_lower:
            return True
    return False


def _mistral_ocr_pdf_page_render(path: str, page_num: int) -> str:
    """Render one PDF page to PNG and run Mistral image OCR. Returns text or ''."""
    if not _MISTRAL_OCR_KEY or _is_mistral_disabled():
        return ""
    try:
        import tempfile

        doc = fitz.open(path)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=150, alpha=False)
        buf = pix.tobytes("png")
        doc.close()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(buf)
            tmp_path = tmp.name
        try:
            pages = _mistral_ocr_image(tmp_path)
            if pages:
                return pages[0][1] or ""
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as e:
        logger.debug("Mistral OCR for PDF page failed: %s", e)
    return ""


def get_mistral_ocr_key() -> str:
    return _MISTRAL_OCR_KEY


def _mistral_ocr_pdf(path: str) -> List[tuple[int, str]]:
    """OCR a PDF using Mistral's cloud OCR API. Returns [(page_num, markdown_text), ...]."""
    if not _MISTRAL_OCR_KEY or _is_mistral_disabled():
        return []
    try:
        import base64
        from mistralai import Mistral

        with open(path, "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        client = Mistral(api_key=_MISTRAL_OCR_KEY)
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64}",
            },
        )

        pages = []
        for page in response.pages:
            page_num = page.index + 1
            text = page.markdown or ""
            pages.append((page_num, text))
        logger.info("Mistral OCR: %s → %d pages extracted", os.path.basename(path), len(pages))
        return pages
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Unauthorized" in err_msg:
            _disable_mistral_persistently()
            logger.error("Mistral OCR Failed (401 Unauthorized): API key invalid. Disabling Mistral OCR for this run.")
        else:
            logger.warning("Mistral OCR failed for PDF %s: %s", path, e)
        return []


def mistral_ocr_pdf_to_markdown(path: str) -> str:
    """
    Run Mistral OCR on a PDF and return one markdown string (per-page ## sections).
    Use for manual export; place a companion .md next to the PDF with the same stem to prefer it at index time.
    Raises ValueError if Mistral is unavailable or returns nothing.
    """
    _ensure_mistral_key_from_env()
    if not _MISTRAL_OCR_KEY or _is_mistral_disabled():
        raise ValueError(
            "Mistral OCR is not available. Set MISTRAL_OCR_API_KEY in .env and restart the server."
        )
    pages = _mistral_ocr_pdf(path)
    if not pages:
        raise ValueError(
            "Mistral OCR returned no text. Check the PDF, API key, and quota; see server logs."
        )
    parts = [f"## Page {num}\n\n{text.strip()}" for num, text in pages]
    return "\n\n".join(parts)


def _mistral_ocr_image(path: str) -> List[tuple[int, str]]:
    """OCR an image using Mistral's cloud OCR API. Returns [(1, markdown_text)]."""
    if not _MISTRAL_OCR_KEY or _is_mistral_disabled():
        return []
    try:
        import base64
        from mistralai import Mistral

        ext = Path(path).suffix.lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff",
        }
        mime = mime_map.get(ext, "image/jpeg")

        with open(path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        client = Mistral(api_key=_MISTRAL_OCR_KEY)
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:{mime};base64,{b64}",
            },
        )

        pages = []
        for page in response.pages:
            page_num = page.index + 1
            text = page.markdown or ""
            pages.append((page_num, text))
        logger.info("Mistral OCR: %s → %d pages, %d chars", os.path.basename(path), len(pages), sum(len(t) for _, t in pages))
        return pages
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Unauthorized" in err_msg:
            _disable_mistral_persistently()
            logger.error("Mistral OCR Failed (401 Unauthorized): API key invalid. Disabling Mistral OCR for this run.")
        else:
            logger.warning("Mistral OCR failed for image %s: %s", path, e)
        return []


from config import (
    DATA_FOLDER,
    PDF_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    JSON_EXTENSIONS,
    TEXT_EXTENSIONS,
    MISTRAL_OCR_FORCE_FILENAMES,
    get_doc_type,
    CHUNK_BY,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    STRUCTURED_DOC_MIN_PAGES,
    STRUCTURED_DOC_TEXT_RATIO,
    STRUCTURED_MAX_CHUNK_CHARS,
)

logger = logging.getLogger(__name__)


def get_document_metadata_for_path(path: str) -> dict:
    """
    Extract metadata from a single document file (for image indexer to associate PDF pages).
    Returns dict with patient_name, claim_number, policy_number, group_number.
    For images: uses OCR to extract text first, then parses metadata.
    """
    ext = Path(path).suffix.lower()
    pages = []
    if ext in PDF_EXTENSIONS:
        pages = extract_text_from_pdf(path)
    elif ext in MARKDOWN_EXTENSIONS:
        pages = extract_text_from_markdown(path)
    elif ext in TEXT_EXTENSIONS:
        pages = extract_text_from_txt(path)
    elif ext in JSON_EXTENSIONS:
        pages = extract_text_from_json(path)
    elif ext in IMAGE_EXTENSIONS:
        pages = extract_text_from_image(path)
    if not pages:
        return {"patient_name": "", "claim_number": "", "policy_number": "", "group_number": "", "doctor_name": ""}
    full_text = "\n\n".join(t for _, t in pages)
    return extract_chunk_metadata(full_text)


def _normalize_metadata_value(val: str) -> str:
    """Normalize metadata for consistent filtering: collapse spaces, strip."""
    if not val:
        return ""
    return re.sub(r"\s+", " ", val.strip())


def _normalize_name_to_title(name: str) -> str:
    """Normalize a person name to Title Case for consistent matching everywhere.
    'TERESA BROWN' → 'Teresa Brown', 'teresa brown' → 'Teresa Brown'."""
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", name.strip())
    return " ".join(w.capitalize() for w in cleaned.split())


def extract_chunk_metadata(text: str) -> dict:
    """
    Extract insurance/claim metadata from chunk text for ChromaDB filtering.
    Returns dict with patient_name, claim_number, policy_number, group_number (empty str if not found).
    Uses multiple patterns for robustness across form variations and OCR.
    """
    meta = {"patient_name": "", "claim_number": "", "policy_number": "", "group_number": "", "doctor_name": ""}
    if not text or not isinstance(text, str):
        return meta

    # Patient name: multiple patterns for form variations (Patient Name, Full Name, Insured)
    # Mistral/markdown forms: |  Name *Teresa Brown* |
    patient_patterns = [
        r"Name\s+\*([A-Za-z][A-Za-z\s\-']+)\*",
        r"\|\s*Name\s+\*([A-Za-z][A-Za-z\s\-']+)\*\s*\|",
        r"(?:Patient\s+Name|Insured/Patient\s+Name|Insured\s+Name)\s*:?\s*([A-Za-z][A-Za-z\s\-]+?)(?:\s{2,}|\n|Claim|$|Policy|Group)",
        r"(?:Patient\s+Name|Insured/Patient\s+Name)\s*:?\s*([A-Za-z][A-Za-z\s\-]+)",
        r"Patient\s+Name\s+([A-Z][A-Za-z\s]+?)(?:\s+Claim|\s{2,}|\n|$)",
        r"Full\s+Name\s+([A-Z][A-Za-z\s\-]+?)(?:\s{2,}|\n|Birth|Address|$)",
    ]
    for pat in patient_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = _normalize_name_to_title(m.group(1))
            if len(name) >= 2 and not name.isdigit():
                meta["patient_name"] = name
                break

    # Claim number — specific patterns first (avoid r"Claim\s+(\w+)" matching "Claim Number" -> "Number").
    for pat in [
        r"Claim\s+Number\s+\*([^\*\s\|]+)\*",
        r"Claim\s+Number\s*:?\s*([\w\-.+]+)",
        r"Claim\s*#\s*([\w\-.+]+)",
        r"Claim\s+(\d[\w\-.+]*)(?=\s*\||\s*\n)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = _normalize_metadata_value(m.group(1))
            if raw.lower() in ("number", "no", "na", "n/a") or len(raw) < 2:
                continue
            meta["claim_number"] = raw
            break

    # Policy number (avoid matching literal "Number" as value)
    for pat in [
        r"Policy\s+Number\s+\*([^\*\s\|]+)\*",
        r"Policy\s+Number\s*:?\s*([\d\w\-]+)",
        r"Policy\s*#\s*([\d\w\-]+)",
        r"Policy\s+(\d[\d\w\-]*)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = _normalize_metadata_value(m.group(1))
            if raw.lower() == "number" or len(raw) < 2:
                continue
            meta["policy_number"] = raw
            break

    # Group number
    for pat in [
        r"Group\s+Number\s*:?\s*([\d\w\-]+)",
        r"Group\s*#\s*([\d\w\-]+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            meta["group_number"] = _normalize_metadata_value(m.group(1))
            break

    # Doctor/physician name: "Dr. Smith", "Attending Physician Name ... Dr. Faisal de Valle"
    for pat in [
        r"(?:Dr\.|Doctor)\s+([A-Za-z][A-Za-z\.\s\-]+?)(?=\s+Degree|\s+Signature|\s+M\.D\.|\s{2,}|\n|$)",
        r"Attending\s+Physician\s+Name[^:]*:?\s*(?:Dr\.\s+)?([A-Za-z][A-Za-z\.\s\-]+?)(?=\s+Degree|\s+Signature|\s{2,}|\n|$)",
        r"Physician\s+Name[^:]*:?\s*(?:Dr\.\s+)?([A-Za-z][A-Za-z\.\s\-]+?)(?=\s+Degree|\s+Signature|\s{2,}|\n|$)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = _normalize_name_to_title(m.group(1))
            if len(name) >= 3 and not name.isdigit():
                meta["doctor_name"] = name
                break

    return meta


@dataclass
class Chunk:
    """One indexed unit of text with metadata."""
    chunk_id: str
    text: str
    file_name: str
    page_number: int | None
    document_type: str
    start_char: int
    end_char: int
    # Extracted metadata for ChromaDB filtering (empty str when not found)
    patient_name: str = ""
    claim_number: str = ""
    policy_number: str = ""
    group_number: str = ""
    doctor_name: str = ""
    # Structured document support
    doc_quality: str = ""          # "structured" for auto-detected clean PDFs, "" for standard
    embedding_text: str = ""       # summary text for vector embedding (used instead of full text for large chunks)
    last_modified: float = 0.0     # file mtime for incremental indexing


def _extract_text_pdfplumber(path: str) -> List[tuple[int, str]]:
    """Extract text per page using pdfplumber. Returns [(page_num, text), ...]."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                pages.append((i, text or ""))
    except Exception as e:
        logger.warning("pdfplumber failed for %s: %s", path, e)
    return pages


def _extract_text_docling(path: str) -> List[tuple[int, str]]:
    """
    Extract text using IBM's Docling for high-quality structure and bilingual support.
    Uses a local cache (sidecar file) to avoid re-processing 500+ page PDFs.
    """
    cache_dir = os.path.join(DATA_FOLDER, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{os.path.basename(path)}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                logger.info("Using cached Docling extraction: %s", cache_path)
                return [(int(p), t) for p, t in data.items()]
        except Exception as e:
            logger.warning("Failed to read Docling cache: %s", e)

    pages = []
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        
        logger.info("Starting Docling conversion (lightweight mode): %s", path)
        
        # Disable heavy models that cause 'meta tensor' errors on some systems
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = False
        pipeline_options.do_picture_description = False
        pipeline_options.do_chart_extraction = False
        pipeline_options.do_ocr = True # Keep OCR for bilingual/complex text
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(path)
        
        doc = result.document
        page_map = {}
        
        # Iterate through items and assign them to pages
        for item, level in doc.iterate_items():
            if hasattr(item, 'prov') and item.prov:
                for p_prov in item.prov:
                    page_no = p_prov.page_no
                    if page_no not in page_map:
                        page_map[page_no] = ""
                    if hasattr(item, 'text'):
                        page_map[page_no] += item.text + "\n"
        
        # Sort and return
        for p_no in sorted(page_map.keys()):
            pages.append((p_no, page_map[p_no].strip()))
            
        if not pages:
            # Fallback if page-level iteration failed: get the whole markdown and treat as one
            full_md = doc.export_to_markdown()
            pages.append((1, full_md))
            
        # Save to cache
        try:
            with open(cache_path, 'w') as f:
                json.dump({str(p): t for p, t in pages}, f)
        except:
            pass
            
    except Exception as e:
        logger.warning("Docling extraction failed for %s: %s", path, e)
    return pages


def _extract_text_pymupdf(path: str) -> List[tuple[int, str]]:
    """Extract text per page using PyMuPDF. Returns [(page_num, text), ...]."""
    pages = []
    try:
        doc = fitz.open(path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()
            pages.append((i + 1, text or ""))
        doc.close()
    except Exception as e:
        logger.warning("PyMuPDF text extraction failed for %s: %s", path, e)
    return pages


def _needs_ocr(text: str, min_chars: int = 50) -> bool:
    """Heuristic: if page has very little text, treat as scanned and use OCR."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return len(cleaned) < min_chars


def extract_text_from_pdf_page(path: str, page_num: int) -> str:
    """
    Extract text from a single PDF page. Uses PyMuPDF first (fast); if empty or very short,
    uses Mistral OCR (when configured) or Tesseract for scanned pages. Used at display time
    to show query-relevant context (e.g. "primary diagnosis").
    """
    try:
        _ensure_mistral_key_from_env()
        if _should_force_mistral_ocr(path) and _MISTRAL_OCR_KEY and not _is_mistral_disabled():
            t = _mistral_ocr_pdf_page_render(path, page_num)
            if t.strip():
                return t.strip()
        doc = fitz.open(path)
        page = doc.load_page(page_num - 1)
        text = page.get_text() or ""
        doc.close()
        if text.strip() and len(text.strip()) >= 30:
            return text.strip()
        # Scanned page: prefer Mistral OCR when configured (matches index quality)
        if _MISTRAL_OCR_KEY and not _is_mistral_disabled():
            t = _mistral_ocr_pdf_page_render(path, page_num)
            if t.strip():
                return t.strip()
        return _ocr_pdf_page(path, page_num)
    except Exception as e:
        logger.warning("extract_text_from_pdf_page failed for %s p.%s: %s", path, page_num, e)
    return ""


def find_pdf_page_containing_phrases(path: str, phrases: list[str], start_page: int = 1, max_pages: int = 6) -> tuple[int, str]:
    """
    Find the first page in a PDF that contains any of the given phrases.
    Tries start_page first, then nearby pages (start-1..1, start+1..end). Limited to max_pages to avoid cost.
    Returns (page_num, text) or (start_page, text) if no phrase found.
    """
    if not phrases:
        text = extract_text_from_pdf_page(path, start_page)
        return start_page, text
    try:
        doc = fitz.open(path)
        total = len(doc)
        doc.close()
    except Exception:
        return start_page, extract_text_from_pdf_page(path, start_page)
    # Order: start_page, then pages before (start-1..1), then after (start+1..total). Cap at max_pages.
    to_try = [start_page]
    for p in range(start_page - 1, 0, -1):
        to_try.append(p)
        if len(to_try) >= max_pages:
            break
    for p in range(start_page + 1, total + 1):
        if len(to_try) >= max_pages:
            break
        to_try.append(p)
    for p in to_try:
        text = extract_text_from_pdf_page(path, p)
        if not text:
            continue
        lower = text.lower()
        if any(ph and len(ph) >= 3 and ph in lower for ph in phrases):
            return p, text
    return start_page, extract_text_from_pdf_page(path, start_page)


def _ocr_pdf_page(path: str, page_num: int) -> str:
    """Run OCR on a single PDF page (1-based). Returns extracted text."""
    if not _check_ocr():
        return ""
    try:
        import io
        from PIL import Image
        import pytesseract
        tesseract_cmd = _find_tesseract_cmd()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        doc = fitz.open(path)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=200, alpha=False)
        buf = pix.tobytes("png")
        doc.close()
        img = Image.open(io.BytesIO(buf)).convert("RGB")
        w, h = img.size
        if max(w, h) > 2000:
            img = img.resize((min(2000, w), min(2000, h)), Image.Resampling.LANCZOS)
        text = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6") or ""
        return text.strip()
    except Exception as e:
        logger.warning("OCR failed for %s page %s: %s", path, page_num, e)
    return ""


def _ocr_image_pages(path: str) -> List[tuple[int, str]]:
    """
    Run OCR on an image file.

    If the image is multi-frame (e.g. multi-page TIFF), run OCR on each frame and
    return [(page_number, text), ...]. For single-frame images, this is just [(1, text)].
    """
    if not _check_ocr():
        return []
    try:
        from PIL import Image, ImageSequence, ImageOps
        import pytesseract

        texts: List[tuple[int, str]] = []
        with Image.open(path) as img:
            # Multi-page TIFFs expose frames via ImageSequence
            frames = list(ImageSequence.Iterator(img))
            if not frames:
                return []
            for idx, frame in enumerate(frames, start=1):
                # Normalize for OCR: grayscale, boosted contrast, slight upscale.
                if frame.mode not in ("L", "RGB"):
                    frame = frame.convert("RGB")
                gray = frame.convert("L")
                gray = ImageOps.autocontrast(gray)
                # Upscale to help Tesseract read small text
                w, h = gray.size
                scale = 1.5
                gray = gray.resize((int(w * scale), int(h * scale)))

                # Use a configuration tuned for block text (forms)
                config = "--oem 3 --psm 6"
                text = pytesseract.image_to_string(gray, lang="eng", config=config) or ""
                texts.append((idx, text))
        return texts
    except Exception as e:
        logger.warning("OCR failed for image %s: %s", path, e)
        return []


def extract_text_from_pdf(path: str) -> List[tuple[int, str]]:
    """
    Extract text from PDF. Priority: pdfplumber → PyMuPDF → Mistral OCR → Tesseract.
    For large structured PDFs, prefer the companion .md file (pre-converted via Mistral OCR).
    Returns [(page_number, text), ...].
    """
    # 0. If a pre-converted markdown companion file exists, skip PDF extraction entirely.
    # e.g. pnb.md is the Mistral-OCR output for PNB AR 2024-25_Web.pdf
    md_companion = os.path.splitext(path)[0] + ".md"
    if os.path.exists(md_companion):
        logger.info("Using pre-converted markdown companion for %s: %s", os.path.basename(path), md_companion)
        return extract_text_from_markdown(md_companion)

    # 1. Try internal text extraction first (Small docs)
    pages = _extract_text_pdfplumber(path)
    if not pages:
        pages = _extract_text_pymupdf(path)
    
    if not pages:
        # No text content at all (broken PDF or purely image-based), try full document Mistral OCR
        if _MISTRAL_OCR_KEY:
            return _mistral_ocr_pdf(path)
        return []

    # 2. Check how many pages need OCR
    pages_needing_ocr = [p_num for p_num, text in pages if _needs_ocr(text)]
    
    # If more than 90% of pages have NO text content, try full document Mistral OCR
    # (High threshold: if even 10% have text, we prefer page-by-page local/selective OCR)
    zero_text_pages = [p_num for p_num, text in pages if not text.strip()]
    ratio_zero_text = len(zero_text_pages) / len(pages)
    
    if ratio_zero_text > 0.9 and _MISTRAL_OCR_KEY and not _is_mistral_disabled():
        # Document is almost entirely scanned/image based, use Mistral for global extraction
        mistral_pages = _mistral_ocr_pdf(path)
        if mistral_pages:
            return mistral_pages

    # 3. Process page-by-page (Mixed extraction)
    result = []
    for page_num, text in pages:
        if _needs_ocr(text):
            # Try Mistral for this specific page if doc is small, else fallback to local Tesseract
            # Cloud OCR on every page of a large doc is too slow/expensive
            if _MISTRAL_OCR_KEY and not _is_mistral_disabled() and len(pages) < 10:
                ocr_text = _mistral_ocr_pdf_page_render(path, page_num)
                if not ocr_text.strip():
                    ocr_text = _ocr_pdf_page(path, page_num)
                result.append((page_num, ocr_text if ocr_text else text))
            else:
                ocr_text = _ocr_pdf_page(path, page_num)
                result.append((page_num, ocr_text if ocr_text else text))
        else:
            result.append((page_num, text))
    return result


def extract_text_from_image(
    path: str,
    enable_vision: bool = False,
    vision_provider: str = "",
    vision_key: str = ""
) -> List[tuple[int, str]]:
    """
    Extract text from an image file.
    Priority: Mistral OCR (cloud, high accuracy) → Tesseract (local).
    For single images, this is one logical page. For multi-page images (like TIFF),
    each frame becomes its own page so searches can show the correct page number.
    If enable_vision is true, additionally hit the chosen LLM for a semantic description.
    """
    if _MISTRAL_OCR_KEY:
        pages = _mistral_ocr_image(path)
        if not pages:
            pages = [(1, "")]
    else:
        pages = _ocr_image_pages(path)
        if not pages:
            pages = [(1, "")]

    if enable_vision and vision_provider and vision_key:
        try:
            from llm_insight import generate_image_description
            logger.info("Generating vision caption for %s using %s", os.path.basename(path), vision_provider)
            caption, err = generate_image_description(vision_provider, vision_key, path)
            if caption:
                # Append to the first page's text
                page_idx, text = pages[0]
                pages[0] = (page_idx, text + f"\n\n[AI Image Description: {caption}]")
            elif err:
                logger.warning("Vision captioning failed for %s: %s", path, err)
        except Exception as e:
            logger.warning("Error calling vision captioning for %s: %s", path, e)

    return pages



def extract_text_from_markdown(path: str) -> List[tuple[int, str]]:
    """Read a Markdown file as plain text.
    For large files (4MB+), split by top-level headings into logical sections,
    each assigned a page number — so pnb.md gets proper page attribution.
    Returns [(page_number, text), ...].
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not text:
            return []
        # For large files, split by heading boundaries into logical pages
        if len(text) > 100_000:
            sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
            pages = []
            for i, section in enumerate(sections, start=1):
                section = section.strip()
                if section:
                    pages.append((i, section))
            if pages:
                return pages
        return [(1, text)]
    except Exception as e:
        logger.warning("Failed to read markdown %s: %s", path, e)
        return []


def extract_text_from_json(path: str) -> List[tuple[int, str]]:
    """Read a JSON file as plain text so it can be indexed and used as LLM context."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [(1, text or "")]
    except Exception as e:
        logger.warning("Failed to read json %s: %s", path, e)
        return []


def extract_text_from_txt(path: str) -> List[tuple[int, str]]:
    """Read a plain text (.txt) file. Returns [(1, full_text)]."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return [(1, text or "")]
    except Exception as e:
        logger.warning("Failed to read txt %s: %s", path, e)
        return []


# ---------------------------------------------------------------------------
# Structured document detection & section-based chunking
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    re.compile(r"^\s*#{1,4}\s+.+", re.MULTILINE),                          # Markdown headings
    re.compile(r"^\s*(?:Chapter|CHAPTER)\s+\d+", re.MULTILINE),             # Chapter 1, CHAPTER 2
    re.compile(r"^\s*(?:Section|SECTION)\s+\d+", re.MULTILINE),             # Section 1
    re.compile(r"^\s*\d{1,2}\.\s+[A-Z]", re.MULTILINE),                    # 1. Introduction, 12. Appendix
    re.compile(r"^\s*\d{1,2}\.\d{1,2}\s+[A-Z]", re.MULTILINE),             # 1.1 Overview
    re.compile(r"^[A-Z][A-Z\s&,\-]{8,}$", re.MULTILINE),                   # ALL CAPS headings (>=10 chars)
]


def _classify_document(pages: List[tuple[int, str]]) -> str:
    """Classify a PDF as 'structured' (clean text, headings) or 'standard'.

    Structured docs get section-based chunking (few large chunks).
    Standard docs go through the existing per-page paragraph chunking.
    """
    if len(pages) < STRUCTURED_DOC_MIN_PAGES:
        return "standard"

    text_pages = sum(1 for _, t in pages if len(re.sub(r"\s+", " ", t).strip()) >= 50)
    ratio = text_pages / len(pages) if pages else 0
    if ratio < STRUCTURED_DOC_TEXT_RATIO:
        return "standard"

    sample_text = "\n\n".join(t for _, t in pages[:50])
    heading_count = 0
    for pat in _HEADING_PATTERNS:
        heading_count += len(pat.findall(sample_text))
    if heading_count < 3:
        return "standard"

    return "structured"


def _detect_sections(
    pages: List[tuple[int, str]],
) -> List[tuple[str, int, int]]:
    """Find section boundaries by detecting headings in page text.

    Returns [(section_title, start_page_idx, end_page_idx), ...].
    Indices are into the *pages* list (0-based).
    If fewer than 3 sections detected, falls back to evenly-spaced groups of ~10.
    """
    section_heading_re = re.compile(
        r"(?:^\s*#{1,4}\s+(.+))"              # Markdown headings
        r"|(?:^\s*(?:Chapter|CHAPTER)\s+\d+[.:\s]*(.*))"
        r"|(?:^\s*(?:Section|SECTION)\s+\d+[.:\s]*(.*))"
        r"|(?:^\s*(\d{1,2}\.\s+[A-Z].{3,}))"  # 1. Introduction
        r"|(?:^([A-Z][A-Z\s&,\-]{8,})$)",      # ALL CAPS heading
        re.MULTILINE,
    )

    boundaries: List[tuple[str, int]] = []
    for idx, (_, text) in enumerate(pages):
        first_500 = text[:500]
        m = section_heading_re.search(first_500)
        if m:
            title = next((g for g in m.groups() if g), "").strip()
            if title and len(title) >= 3:
                boundaries.append((title[:120], idx))

    if len(boundaries) < 3:
        n_groups = min(10, len(pages))
        group_size = max(1, len(pages) // n_groups)
        boundaries = []
        for i in range(0, len(pages), group_size):
            page_num = pages[i][0]
            boundaries.append((f"Pages {page_num}–{pages[min(i + group_size - 1, len(pages) - 1)][0]}", i))

    sections: List[tuple[str, int, int]] = []
    for i, (title, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else len(pages) - 1
        sections.append((title, start, end))
    return sections


def chunk_structured_document(
    pages: List[tuple[int, str]],
    file_name: str,
    document_type: str,
    base_id: str,
    document_metadata: dict | None = None,
) -> List[Chunk]:
    """Section-based chunking for structured PDFs (annual reports, etc.).

    Groups pages by detected sections, producing ~5-15 large chunks.
    Each chunk stores the full section text for BM25 and a short summary
    in embedding_text for the vector index.
    """
    doc_meta = document_metadata or {}
    sections = _detect_sections(pages)
    chunks: List[Chunk] = []
    chunk_index = 0

    for title, start_idx, end_idx in sections:
        section_pages = pages[start_idx : end_idx + 1]
        section_text = "\n\n".join(t for _, t in section_pages if t.strip())
        if not section_text.strip():
            continue

        prefixed_text = f"[{title}]\n\n{section_text}"

        page_range_start = section_pages[0][0]
        page_range_end = section_pages[-1][0]

        text_parts = [prefixed_text]
        if len(prefixed_text) > STRUCTURED_MAX_CHUNK_CHARS:
            text_parts = _split_large_section(prefixed_text, title, STRUCTURED_MAX_CHUNK_CHARS)

        for part in text_parts:
            chunk_id = f"{base_id}_sec{chunk_index}"
            embed_text = f"{title}\n\n{part[:1500]}"
            c_meta = extract_chunk_metadata(part)
            for k, v in doc_meta.items():
                if v and not c_meta.get(k):
                    c_meta[k] = v

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    file_name=file_name,
                    page_number=page_range_start,
                    document_type=document_type,
                    start_char=0,
                    end_char=len(part),
                    patient_name=c_meta.get("patient_name", ""),
                    claim_number=c_meta.get("claim_number", ""),
                    policy_number=c_meta.get("policy_number", ""),
                    group_number=c_meta.get("group_number", ""),
                    doctor_name=c_meta.get("doctor_name", ""),
                    doc_quality="structured",
                    embedding_text=embed_text,
                )
            )
            chunk_index += 1

    logger.info(
        "Structured chunking: %s → %d sections, %d chunks (from %d pages)",
        file_name, len(sections), len(chunks), len(pages),
    )
    return chunks


def _split_large_section(text: str, title: str, max_chars: int) -> List[str]:
    """Split an oversized section at paragraph boundaries, keeping the title prefix."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2
        if current_len + p_len > max_chars and current:
            parts.append("\n\n".join(current))
            current = [f"[{title} (cont.)]"]
            current_len = len(current[0]) + 2
        current.append(p)
        current_len += p_len

    if current:
        parts.append("\n\n".join(current))
    return parts if parts else [text]


def chunk_text(
    page_num: int | None,
    text: str,
    file_name: str,
    document_type: str,
    base_id: str,
    document_metadata: dict | None = None,
    doc_quality: str = "",
    embedding_text: str = "",
) -> List[Chunk]:
    """
    Chunk text by paragraph (or keep as single page chunk if CHUNK_BY is page).
    Splits on double newlines; merges small paragraphs up to MAX_CHUNK_CHARS.
    document_metadata: optional dict with patient_name, claim_number, policy_number, group_number
    from document-level extraction (used when chunk text doesn't contain it).
    """
    if not text or not text.strip():
        return []
    doc_meta = document_metadata or {}
    # Normalize whitespace but keep paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current = []
    current_len = 0
    chunk_index = 0

    for p in paragraphs:
        p_len = len(p) + 2
        if current_len + p_len > MAX_CHUNK_CHARS and current:
            combined = "\n\n".join(current)
            if len(combined) >= MIN_CHUNK_CHARS:
                chunk_id = f"{base_id}_p{page_num or 0}_c{chunk_index}" if base_id else f"c{chunk_index}"
                c_meta = extract_chunk_metadata(combined)
                for k, v in doc_meta.items():
                    if v and (not c_meta.get(k)):
                        c_meta[k] = v
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=combined,
                        file_name=file_name,
                        page_number=page_num,
                        document_type=document_type,
                        start_char=0,
                        end_char=len(combined),
                        patient_name=c_meta.get("patient_name", ""),
                        claim_number=c_meta.get("claim_number", ""),
                        policy_number=c_meta.get("policy_number", ""),
                        group_number=c_meta.get("group_number", ""),
                        doctor_name=c_meta.get("doctor_name", ""),
                    )
                )
                chunk_index += 1
            current = []
            current_len = 0
        current.append(p)
        current_len += p_len

    if current:
        combined = "\n\n".join(current)
        if len(combined) >= MIN_CHUNK_CHARS or not chunks:
            chunk_id = f"{base_id}_p{page_num or 0}_c{chunk_index}" if base_id else f"c{chunk_index}"
            c_meta = extract_chunk_metadata(combined)
            for k, v in doc_meta.items():
                if v and (not c_meta.get(k)):
                    c_meta[k] = v
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=combined,
                    file_name=file_name,
                    page_number=page_num,
                    document_type=document_type,
                    start_char=0,
                    end_char=len(combined),
                    patient_name=c_meta.get("patient_name", ""),
                    claim_number=c_meta.get("claim_number", ""),
                    policy_number=c_meta.get("policy_number", ""),
                    group_number=c_meta.get("group_number", ""),
                    doctor_name=c_meta.get("doctor_name", ""),
                    doc_quality=doc_quality,
                    embedding_text=embedding_text,
                )
            )
    return chunks


def load_and_chunk_folder(
    folder: str | None = None, 
    existing_chunks: List[Chunk] | None = None,
    enable_vision: bool = False,
    vision_provider: str = "",
    vision_api_key: str = "",
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> List[Chunk]:
    """
    Read supported files from folder, extract text (with OCR or Docling), chunk.
    Incremental: if existing_chunks provided and file mtime hasn't changed, skip extraction.
    Returns list of Chunks with metadata.
    """
    folder = folder or DATA_FOLDER
    if not os.path.isdir(folder):
        logger.warning("Data folder does not exist: %s", folder)
        return []

    all_chunks: List[Chunk] = []
    seen_bases: dict[str, int] = {}
    processed_count = 0

    # Map existing chunks by filename for quick comparison
    file_map: dict[str, List[Chunk]] = {}
    if existing_chunks:
        for c in existing_chunks:
            if c.file_name not in file_map:
                file_map[c.file_name] = []
            file_map[c.file_name].append(c)

    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            ext = Path(path).suffix.lower()
            if (
                ext not in PDF_EXTENSIONS
                and ext not in IMAGE_EXTENSIONS
                and ext not in MARKDOWN_EXTENSIONS
                and ext not in JSON_EXTENSIONS
                and ext not in TEXT_EXTENSIONS
            ):
                continue
            base = os.path.basename(path)
            doc_type = get_doc_type(path)
            base_id = base.replace(" ", "_").replace(".", "_")
            if base_id in seen_bases:
                seen_bases[base_id] += 1
                base_id = f"{base_id}_{seen_bases[base_id]}"
            else:
                seen_bases[base_id] = 0

            # Incremental check
            mtime = os.path.getmtime(path)
            if base in file_map:
                cached = file_map[base]
                if cached and abs(cached[0].last_modified - mtime) < 1.0:
                    logger.debug("Skipping unchanged file: %s", base)
                    all_chunks.extend(cached)
                    continue

            if progress_callback:
                pct = min(55, 15 + processed_count * 4)
                progress_callback(pct, f"Processing {base}")
                processed_count += 1

            if ext in PDF_EXTENSIONS:
                pages = extract_text_from_pdf(path)
            elif ext in MARKDOWN_EXTENSIONS:
                pages = extract_text_from_markdown(path)
            elif ext in JSON_EXTENSIONS:
                pages = extract_text_from_json(path)
            elif ext in TEXT_EXTENSIONS:
                pages = extract_text_from_txt(path)
            else:
                pages = extract_text_from_image(path, enable_vision, vision_provider, vision_api_key)

            # Extract document-level metadata from full text for ChromaDB filtering
            full_text = "\n\n".join(t for _, t in pages)
            document_metadata = extract_chunk_metadata(full_text)

            # Route structured PDFs through section-based chunking (fast path)
            if ext in PDF_EXTENSIONS and _classify_document(pages) == "structured":
                logger.info("Structured PDF detected: %s (%d pages) → section-based chunking", base, len(pages))
                for c in chunk_structured_document(pages, base, doc_type, base_id, document_metadata):
                    all_chunks.append(c)
            else:
                current_quality = "structured" if ext in MARKDOWN_EXTENSIONS else ""
                for page_num, text in pages:
                    for c in chunk_text(page_num, text, base, doc_type, base_id, document_metadata, doc_quality=current_quality):
                        all_chunks.append(c)

            # Assign mtime to all new chunks for this file
            for c in all_chunks:
                if c.file_name == base and c.last_modified == 0:
                    c.last_modified = mtime

    return all_chunks


def count_image_files_in_folder(folder: str | None = None) -> int:
    """Count image files under folder (for UI: show OCR message when > 0)."""
    folder = folder or DATA_FOLDER
    if not os.path.isdir(folder):
        return 0
    n = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                n += 1
    return n
