"""
Optional image reranker for Multimodal Hybrid RAG.
Uses SigLIP (or similar) to re-score (query, image) pairs and improve ranking.
Retrieve more images with CLIP, then rerank with this model and return top-K.
"""
import io
import logging
import os
from typing import List, Tuple, Dict, Any

from config import IMAGE_RERANKER_MODEL, IMAGE_RERANKER_CANDIDATES

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, AutoModel
        import torch
        _processor = AutoProcessor.from_pretrained(IMAGE_RERANKER_MODEL)
        _model = AutoModel.from_pretrained(IMAGE_RERANKER_MODEL)
        _model.eval()
    return _model, _processor


def _load_pil_image(item: Dict[str, Any]) -> "Image.Image | None":
    """Load a PIL Image from image_item (path, is_pdf_page, page)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    path = item.get("path", "")
    if not path or not os.path.isfile(path):
        return None
    is_pdf = item.get("is_pdf_page", False)
    page_num = int(item.get("page", 1))
    try:
        if is_pdf:
            import fitz
            doc = fitz.open(path)
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=150, alpha=False)
            buf = pix.tobytes("png")
            doc.close()
            img = Image.open(io.BytesIO(buf)).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        w, h = img.size
        # Preserve aspect ratio; SigLIP expects ~224px, processor handles final resize
        max_side = 384
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.debug("Failed to load image for rerank: %s", e)
    return None


def rerank(
    query: str,
    candidates: List[Tuple[Dict[str, Any], float]],
    top_k: int,
    batch_size: int = 8,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Rerank image candidates using SigLIP (query–image relevance).
    candidates: list of (image_item, clip_score).
    Returns list of (image_item, reranker_score) sorted by score desc, length top_k.
    """
    if not candidates:
        return []
    try:
        model, processor = _load_model()
    except Exception as e:
        logger.warning("Image reranker not available: %s. Returning unreranked list.", e)
        return [(item, score) for item, score in candidates[:top_k]]

    import torch
    images = []
    valid_indices = []
    for i, (item, _) in enumerate(candidates):
        pil = _load_pil_image(item)
        if pil is not None:
            images.append(pil)
            valid_indices.append(i)
    if not images:
        return [(item, score) for item, score in candidates[:top_k]]

    scores_list = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        try:
            inputs = processor(
                text=[query] * len(batch_images),
                images=batch_images,
                padding="max_length",
                return_tensors="pt",
            )
            device = next(model.parameters(), None)
            if device is not None:
                device = device.device
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            logits = out.logits_per_image
            if hasattr(logits, "sigmoid"):
                probs = logits.sigmoid()
            else:
                import torch.nn.functional as F
                probs = F.sigmoid(logits)
            row = probs[:, 0].cpu().float().tolist()
            scores_list.extend(row)
        except Exception as e:
            logger.warning("Image reranker batch failed: %s", e)
            scores_list.extend([0.0] * len(batch_images))

    idx_to_score = {valid_indices[i]: scores_list[i] for i in range(len(valid_indices))}
    reranked = []
    for i, (item, original_score) in enumerate(candidates):
        siglip_score = idx_to_score.get(i, 0.0)
        # Blend SigLIP visual score with original (CLIP + OCR text boost).
        # This ensures OCR text relevance isn't discarded by visual-only reranking.
        score = 0.5 * siglip_score + 0.5 * original_score
        reranked.append((item, float(score)))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]
