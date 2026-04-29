"""
Image retrieval: CLIP text encoder for query -> search image_collection only.
Separate Chroma collection; no mixing with text embeddings.
OCR-text re-ranking: when images have stored OCR text, use keyword overlap
to boost images whose content actually matches the query.
"""
import os
import re
import logging
from typing import List, Tuple, Dict, Any

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_IMAGE_COLLECTION_NAME,
    MULTIMODAL_CLIP_MODEL,
    MULTIMODAL_HYBRID_IMAGE_TOP_N,
    USE_IMAGE_RERANKER,
    IMAGE_RERANKER_CANDIDATES,
)

logger = logging.getLogger(__name__)

# Pattern to detect filenames in queries (e.g. "flower.png", "report.pdf")
_FILENAME_PATTERN = re.compile(r"\b([a-zA-Z0-9_\-]+\.(png|jpg|jpeg|gif|bmp|tiff|tif|pdf|webp))\b", re.IGNORECASE)

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and",
    "or", "not", "no", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "have", "has", "had",
    "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "his", "her", "they",
    "them", "their", "what", "which", "who", "whom", "where", "when",
    "how", "show", "give", "find", "get", "see", "tell", "image",
})


def _ocr_text_relevance(query: str, ocr_text: str) -> float:
    """Score how relevant the OCR text is to the query using keyword overlap + phrase proximity + exact phrase boost."""
    if not ocr_text or not query:
        return 0.0
    q_words = list(dict.fromkeys(w for w in re.findall(r"[a-z]{2,}", query.lower()) if w not in _STOPWORDS))
    if not q_words:
        return 0.0
    ocr_lower = ocr_text.lower()
    matched = sum(1 for w in q_words if w in ocr_lower)
    base_score = matched / len(q_words)

    # Phrase proximity bonus: check if consecutive query words appear near each other in OCR text
    phrase_bonus = 0.0
    if len(q_words) >= 2:
        bigrams_found = 0
        bigrams_total = 0
        for i in range(len(q_words) - 1):
            w1, w2 = q_words[i], q_words[i + 1]
            if w1 in ocr_lower and w2 in ocr_lower:
                bigrams_total += 1
                # Check if w1 and w2 appear within 50 chars of each other
                for m in re.finditer(re.escape(w1), ocr_lower):
                    window = ocr_lower[m.start():m.start() + 50]
                    if w2 in window:
                        bigrams_found += 1
                        break
        if bigrams_total > 0:
            phrase_bonus = 0.3 * (bigrams_found / bigrams_total)

    # Exact phrase boost: when a multi-word phrase from the query appears verbatim in OCR, strong boost
    # (e.g. "primary diagnosis" in query and OCR -> image with exact words ranks higher)
    exact_phrase_bonus = 0.0
    if len(q_words) >= 2:
        for n in range(min(4, len(q_words)), 1, -1):  # Check 4-word, 3-word, 2-word phrases
            for i in range(len(q_words) - n + 1):
                phrase = " ".join(q_words[i:i + n])
                if phrase in ocr_lower:
                    exact_phrase_bonus = 0.5 + 0.1 * n  # 2-word: +0.6, 3-word: +0.7, 4-word: +0.8
                    break
            if exact_phrase_bonus > 0:
                break

    return base_score + phrase_bonus + exact_phrase_bonus

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        _clip_processor = CLIPProcessor.from_pretrained(MULTIMODAL_CLIP_MODEL)
        _clip_model = CLIPModel.from_pretrained(MULTIMODAL_CLIP_MODEL)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _encode_query(model, processor, query: str) -> List[List[float]]:
    import torch
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    with torch.no_grad():
        out = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        if not hasattr(out, "norm"):
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                out = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported query feature output type: {type(out)}")
        out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().numpy().tolist()


class ImageRetriever:
    """
    Retrieves from image_collection only (CLIP image embeddings).
    Query is encoded with CLIP text encoder; search is cosine in Chroma.
    """

    def __init__(self):
        pass

    def retrieve(
        self,
        query: str,
        top_n: int | None = None,
        metadata_filter: dict | None = None,
        boost_ocr: bool = False,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Returns list of (image_item, similarity_score).
        image_item: {file_name, path, page, doc_id, file_type, is_pdf_page}.
        metadata_filter: e.g. {"patient_name": "Rika Popper"} for ChromaDB where clause.
        Scores are raw similarity in [0,1] (from CLIP, or from SigLIP reranker if enabled).
        """
        top_n = top_n or MULTIMODAL_HYBRID_IMAGE_TOP_N
        if not os.path.isdir(CHROMA_PERSIST_DIR):
            return []
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.warning("chromadb not installed")
            return []
        try:
            client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection(name=CHROMA_IMAGE_COLLECTION_NAME)
        except Exception:
            return []

        # When query explicitly mentions a filename (e.g. "flower.png"), fetch by metadata first
        filename_hits = []
        for m in _FILENAME_PATTERN.finditer(query):
            fname = m.group(1)
            try:
                got = collection.get(where={"file_name": {"$eq": fname}}, include=["metadatas"])
                ids_list = got.get("ids") or []
                metas_list = got.get("metadatas") or []
                if ids_list:
                    for i, uid in enumerate(ids_list):
                        meta = metas_list[i] if i < len(metas_list) else {}
                        ocr_text = meta.get("ocr_text", "") or ""
                        item = {
                            "id": uid,
                            "file_name": meta.get("file_name", ""),
                            "path": meta.get("path", ""),
                            "page": meta.get("page", 0),
                            "doc_id": meta.get("doc_id", meta.get("file_name", "")),
                            "file_type": meta.get("file_type", "image"),
                            "is_pdf_page": meta.get("is_pdf_page", "False") == "True",
                            "patient_name": meta.get("patient_name", "") or "",
                            "report_type": meta.get("report_type", "") or "",
                            "claim_number": meta.get("claim_number", "") or "",
                            "ocr_text": ocr_text,
                        }
                        filename_hits.append((item, 1.0))
                    break
            except Exception as e:
                logger.debug("Filename lookup for %s failed: %s", fname, e)

        model, processor = _load_clip()
        q_emb = _encode_query(model, processor, query)
        # Retrieve more if we will rerank; otherwise just top_n
        n_retrieve = (
            min(IMAGE_RERANKER_CANDIDATES, collection.count())
            if USE_IMAGE_RERANKER
            else min(top_n, collection.count())
        )
        if n_retrieve <= 0:
            return []
        kwargs = {
            "query_embeddings": q_emb,
            "n_results": n_retrieve,
            "include": ["metadatas", "distances"],
        }
        if metadata_filter:
            clauses = []
            for k, v in metadata_filter.items():
                if v is None:
                    continue
                # List value: use $in (e.g. report_type=["Other", ""] for unclassified docs)
                if isinstance(v, (list, tuple)):
                    if not v:
                        continue
                    clauses.append({k: {"$in": list(v)}})
                # Try multiple case variants for name fields to handle inconsistent casing
                elif k in ("patient_name", "doctor_name"):
                    variants = list({v, v.upper(), v.title(), " ".join(w.capitalize() for w in str(v).split())})
                    if len(variants) == 1:
                        clauses.append({k: {"$eq": variants[0]}})
                    else:
                        clauses.append({"$or": [{k: {"$eq": var}} for var in variants]})
                else:
                    clauses.append({k: {"$eq": v}})
            if clauses:
                kwargs["where"] = {"$and": clauses} if len(clauses) > 1 else clauses[0]
        results = collection.query(**kwargs)
        ids = results["ids"][0] if results["ids"] else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []
        out = list(filename_hits)
        for i, uid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = float(distances[i]) if i < len(distances) and distances[i] is not None else 0
            clip_score = max(0.0, 1.0 - dist)
            ocr_text = meta.get("ocr_text", "") or ""
            ocr_rel = _ocr_text_relevance(query, ocr_text)
            # Blend CLIP with OCR: higher OCR weight when boost_ocr (text-heavy queries) so images with exact words rank first
            ocr_weight = 0.65 if boost_ocr else 0.4
            score = clip_score + ocr_weight * ocr_rel if ocr_text else clip_score
            item = {
                "id": uid,
                "file_name": meta.get("file_name", ""),
                "path": meta.get("path", ""),
                "page": meta.get("page", 0),
                "doc_id": meta.get("doc_id", meta.get("file_name", "")),
                "file_type": meta.get("file_type", "image"),
                "is_pdf_page": meta.get("is_pdf_page", "False") == "True",
                "patient_name": meta.get("patient_name", "") or "",
                "report_type": meta.get("report_type", "") or "",
                "claim_number": meta.get("claim_number", "") or "",
                "ocr_text": ocr_text,
            }
            key = (item.get("file_name", ""), item.get("page", ""))
            if key not in {(x[0].get("file_name"), x[0].get("page")) for x in out}:
                out.append((item, score))
        # Re-sort by blended score (CLIP + OCR relevance)
        out.sort(key=lambda x: x[1], reverse=True)
        if USE_IMAGE_RERANKER and out:
            from .image_reranker import rerank as image_rerank
            out = image_rerank(query, out, top_k=top_n)
        return out
