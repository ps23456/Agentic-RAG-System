"""
BGE reranker for text passages. Used by text retrieval pipeline.
Keeps retrieval module self-contained (does not depend on search_index for rerank).
"""
import logging
from typing import List, Tuple, Any

from config import RERANKER_MODEL, RERANKER_TOP_K, RERANKER_MAX_PASSAGE_CHARS

logger = logging.getLogger(__name__)

_reranker = None


def _load_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def rerank(
    query: str,
    candidates: List[Tuple[Any, float]],
    top_k: int = RERANKER_TOP_K,
    text_extractor: callable = None,
) -> List[Tuple[Any, float]]:
    """
    Rerank candidates using BGE CrossEncoder.
    candidates: list of (item, score); item must have .text or pass text_extractor(item) -> str.
    Returns list of (item, reranker_score) sorted by score desc, length top_k.
    """
    if not candidates:
        return []
    try:
        model = _load_reranker()
    except Exception as e:
        logger.warning("Reranker not available: %s", e)
        return candidates[:top_k]
    items = [c[0] for c in candidates]
    texts = []
    for item in items:
        if text_extractor:
            t = text_extractor(item)
        elif hasattr(item, "text"):
            t = (item.text or "")[:RERANKER_MAX_PASSAGE_CHARS]
        else:
            t = ""
        texts.append(t)
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    else:
        scores = list(scores)
    out = [(items[i], float(scores[i])) for i in range(len(items))]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]
