"""
Multimodal RAG index: CLIP embeddings for text + images in a separate Chroma collection.
Does not touch the Hybrid RAG index (claim_chunks). Use for "Multimodal RAG" mode only.
"""
import io
import logging
import os
from pathlib import Path
from typing import List, Any

from config import (
    DATA_FOLDER,
    CHROMA_PERSIST_DIR,
    CHROMA_MULTIMODAL_COLLECTION_NAME,
    MULTIMODAL_CLIP_MODEL,
    MULTIMODAL_TOP_K,
    MULTIMODAL_RETRIEVE_CANDIDATES,
    PDF_EXTENSIONS,
    IMAGE_EXTENSIONS,
    UPLOADS_SUBDIR,
)
from document_loader import load_and_chunk_folder, Chunk

logger = logging.getLogger(__name__)

_clip_model = None
_clip_processor = None


def _load_clip():
    """Lazy-load CLIP model and processor for text + image encoding in same space."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            _clip_processor = CLIPProcessor.from_pretrained(MULTIMODAL_CLIP_MODEL)
            _clip_model = CLIPModel.from_pretrained(MULTIMODAL_CLIP_MODEL)
            _clip_model.eval()
        except Exception as e:
            logger.error("Failed to load CLIP: %s", e)
            raise
    return _clip_model, _clip_processor


def _collect_image_items(data_folder: str, existing_mtimes: dict[str, float] | None = None) -> List[dict]:
    """
    Collect indexable image items: image files + PDF pages rendered as images.
    Returns list of {"id", "file_name", "page", "path", "pil_image", "is_pdf_page"}.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    items = []
    seen = set()
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
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # Resize if huge to avoid OOM
                        w, h = img.size
                        if max(w, h) > 1024:
                            img = img.resize((min(1024, w), min(1024, h)), Image.Resampling.LANCZOS)
                        uid = f"img_{base}_{1}".replace(" ", "_").replace(".", "_")
                        if uid in seen:
                            continue
                        seen.add(uid)
                        items.append({
                            "id": uid,
                            "file_name": base,
                            "page": 1,
                            "path": path,
                            "pil_image": img.copy(),
                        })
                except Exception as e:
                    logger.warning("Skip image %s: %s", path, e)

            elif ext in PDF_EXTENSIONS:
                try:
                    doc = fitz.open(path)
                    for page_num in range(len(doc)):
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
                        items.append({
                            "id": uid,
                            "file_name": base,
                            "page": page_num + 1,
                            "path": path,
                            "pil_image": pil_img,
                            "is_pdf_page": True,
                            "last_modified": mtime,
                        })
                    doc.close()
                except Exception as e:
                    logger.warning("Skip PDF %s: %s", path, e)

    return items


def _encode_images(model, processor, images: List[Any]) -> List[List[float]]:
    """Encode a list of PIL images to CLIP embeddings (normalized)."""
    import torch
    inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
        if not hasattr(outputs, "norm"):
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                outputs = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                outputs = outputs.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported image feature output type: {type(outputs)}")
        # L2 normalize for cosine similarity in Chroma
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
    return outputs.cpu().numpy().tolist()


def _encode_texts(model, processor, texts: List[str]) -> List[List[float]]:
    """Encode a list of text strings to CLIP embeddings (normalized)."""
    import torch
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        outputs = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        if not hasattr(outputs, "norm"):
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                outputs = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                outputs = outputs.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported text feature output type: {type(outputs)}")
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
    return outputs.cpu().numpy().tolist()


def build_multimodal_index(
    data_folder: str | None = None,
    enable_vision: bool = False,
    vision_provider: str = "",
    vision_api_key: str = ""
) -> int:
    """
    Build the multimodal Chroma collection (text + image embeddings).
    Uses collection CHROMA_MULTIMODAL_COLLECTION_NAME only; does not touch claim_chunks.
    Returns total number of items indexed.
    """
    data_folder = data_folder or DATA_FOLDER
    model, processor = _load_clip()

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError("chromadb is required for Multimodal RAG. Install with: pip install chromadb")

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    
    collection = client.get_or_create_collection(
        name=CHROMA_MULTIMODAL_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 1. Fetch existing metadata for incremental check
    existing_mtimes = {}
    try:
        results = collection.get(include=["metadatas"])
        if results and results["metadatas"]:
            for i, meta in enumerate(results["metadatas"]):
                fname = meta.get("file_name")
                mtime = meta.get("last_modified")
                if fname and mtime is not None:
                    # For multimodal, multiple chunks (ids) can have same file_name
                    # We store the latest mtime seen for that file
                    existing_mtimes[fname] = max(existing_mtimes.get(fname, 0.0), float(mtime))
    except Exception as e:
        logger.debug("Could not fetch existing multimodal metadata: %s", e)

    total = 0

    # 2) Index image items (image files + PDF pages)
    image_items = _collect_image_items(data_folder, existing_mtimes=existing_mtimes)
    if image_items:
        images = [x["pil_image"] for x in image_items]
        embeddings = _encode_images(model, processor, images)
        ids = [x["id"] for x in image_items]
        metadatas = [
            {
                "file_name": x["file_name"],
                "page": x["page"],
                "path": x["path"],
                "chunk_type": "image",
                "is_pdf_page": str(x.get("is_pdf_page", False)),
                "last_modified": float(x.get("last_modified", 0.0)),
            }
            for x in image_items
        ]
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        total += len(ids)
        logger.info("Multimodal index: added/updated %d image(s)", len(ids))

    # 3) Index text chunks (incremental)
    # Load existing chunks to pass to load_and_chunk_folder for text-level mtime skipping
    existing_chunks = []
    try:
        results = collection.get(where={"chunk_type": "text"}, include=["metadatas", "documents"])
        if results and results["metadatas"]:
            for i, meta in enumerate(results["metadatas"]):
                existing_chunks.append(Chunk(
                    chunk_id=results["ids"][i],
                    text=results["documents"][i] or "",
                    file_name=meta.get("file_name", ""),
                    page_number=meta.get("page_number", 0),
                    document_type=meta.get("document_type", ""),
                    start_char=0,
                    end_char=0,
                    last_modified=float(meta.get("last_modified", 0.0))
                ))
    except Exception:
        pass

    all_chunks: List[Chunk] = load_and_chunk_folder(
        data_folder, 
        existing_chunks=existing_chunks,
        enable_vision=enable_vision,
        vision_provider=vision_provider,
        vision_api_key=vision_api_key
    )
    
    # Filter only new or changed text chunks
    existing_text_ids = {c.chunk_id: c.last_modified for c in existing_chunks}
    chunks_to_encode = [
        c for c in all_chunks 
        if c.chunk_id not in existing_text_ids or abs(existing_text_ids[c.chunk_id] - c.last_modified) > 0.1
    ]

    if chunks_to_encode:
        texts = [c.text for c in chunks_to_encode]
        embeddings = _encode_texts(model, processor, texts)
        ids = [c.chunk_id for c in chunks_to_encode]
        metadatas = [
            {
                "file_name": c.file_name,
                "page_number": c.page_number or 0,
                "document_type": c.document_type,
                "chunk_type": "text",
                "patient_name": getattr(c, "patient_name", "") or "",
                "claim_number": getattr(c, "claim_number", "") or "",
                "policy_number": getattr(c, "policy_number", "") or "",
                "group_number": getattr(c, "group_number", "") or "",
                "doctor_name": getattr(c, "doctor_name", "") or "",
                "last_modified": float(c.last_modified),
            }
            for c in chunks_to_encode
        ]
        documents = [c.text for c in chunks_to_encode]
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        total += len(ids)
        logger.info("Multimodal index: added/updated %d text chunk(s)", len(ids))
    else:
        total += (collection.count() - total) # Count existing text
        logger.info("Multimodal index: no text changes detected.")

    return total


def search_multimodal(query: str, top_k: int = MULTIMODAL_TOP_K, data_folder: str | None = None) -> List[dict]:
    """
    Search the multimodal collection with a text query. Returns mixed text + image results.
    Each result: {"type": "text"|"image", "id", "score", "file_name", "page", "text" (if text), "path", "is_pdf_page" (if image)}.
    """
    data_folder = data_folder or DATA_FOLDER
    model, processor = _load_clip()

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        return []

    if not os.path.isdir(CHROMA_PERSIST_DIR):
        return []

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(name=CHROMA_MULTIMODAL_COLLECTION_NAME)
    except Exception:
        return []

    # Retrieve more candidates so we can boost image results and re-rank (surfaces diagrams for visual queries)
    n_retrieve = min(MULTIMODAL_RETRIEVE_CANDIDATES, collection.count())
    q_emb = _encode_texts(model, processor, [query])
    results = collection.query(
        query_embeddings=q_emb,
        n_results=n_retrieve,
        include=["metadatas", "documents", "distances"],
    )

    out = []
    ids = results["ids"][0] if results["ids"] else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    documents = results["documents"][0] if results.get("documents") else []
    distances = results["distances"][0] if results.get("distances") else []

    for i, uid in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        doc_text = documents[i] if i < len(documents) else None
        dist = float(distances[i]) if i < len(distances) and distances[i] is not None else 0
        score = max(0.0, 1.0 - float(dist))

        chunk_type = meta.get("chunk_type", "text")
        if chunk_type == "image":
            out.append({
                "type": "image",
                "id": uid,
                "score": score,
                "file_name": meta.get("file_name", ""),
                "page": meta.get("page", 0),
                "path": meta.get("path", ""),
                "is_pdf_page": meta.get("is_pdf_page", "False") == "True",
            })
        else:
            out.append({
                "type": "text",
                "id": uid,
                "score": score,
                "file_name": meta.get("file_name", ""),
                "page": meta.get("page_number", 0),
                "text": doc_text or "",
            })

    # When query suggests visual content, boost image results so diagrams/charts rank higher
    _VISUAL_KEYWORDS = (
        "diagram", "image", "picture", "chart", "figure", "photo", "visual",
        "arrow", "arrows", "box", "boxes", "flow", "drawing", "screenshot",
        "architecture", "schema", "layout", "graph", "illustration",
    )
    q_lower = (query or "").lower()
    if any(kw in q_lower for kw in _VISUAL_KEYWORDS):
        for item in out:
            if item.get("type") == "image":
                item["score"] = item["score"] * 1.5
        out.sort(key=lambda x: x["score"], reverse=True)

    return out[:top_k]


def get_multimodal_index_count() -> int:
    """Return number of items in the multimodal collection (0 if not built)."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        return 0
    if not os.path.isdir(CHROMA_PERSIST_DIR):
        return 0
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
        coll = client.get_collection(name=CHROMA_MULTIMODAL_COLLECTION_NAME)
        return coll.count()
    except Exception:
        return 0
