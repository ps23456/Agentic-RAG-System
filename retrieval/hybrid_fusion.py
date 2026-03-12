"""
Score normalization (min-max) and weighted fusion for Multimodal Hybrid RAG.
No heuristic boosting; proper normalized score fusion.
"""
from typing import List, Tuple, Any, Dict

from config import MULTIMODAL_HYBRID_WEIGHTS, MULTIMODAL_HYBRID_TOP_K


def normalize_scores(scores: List[float], eps: float = 1e-8) -> List[float]:
    """
    Min-max normalization into [0, 1].
    normalized_score = (score - min_score) / (max_score - min_score + eps)
    """
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s + eps
    return [(s - min_s) / denom for s in scores]


def fuse_results(
    text_results: List[Tuple[Any, float]],
    image_results: List[Tuple[Dict[str, Any], float]],
    query_type: str,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Identity-based score fusion using Reciprocal Rank Fusion (RRF).
    Aggregates text and image ranks by (file_name, page_number).
    This ensures proper weighted fusion without min-max distortion.
    """
    top_k = top_k or MULTIMODAL_HYBRID_TOP_K
    weights = MULTIMODAL_HYBRID_WEIGHTS.get(query_type, MULTIMODAL_HYBRID_WEIGHTS["hybrid"])
    w_text, w_image = weights

    # Map key -> {text_rrf: float, image_rrf: float, content: Best_Content_Obj}
    registry = {}
    RRF_K = 60

    # text_results is already sorted by score descending
    for i, (chunk, _) in enumerate(text_results):
        fn = getattr(chunk, "file_name", "unknown")
        pg = getattr(chunk, "page_number", 0)
        key = (fn, pg)
        
        rrf_score = 1.0 / (RRF_K + i + 1)
        if key not in registry:
            registry[key] = {"text_rrf": 0.0, "image_rrf": 0.0, "content": chunk}
        
        if rrf_score > registry[key]["text_rrf"]:
            registry[key]["text_rrf"] = rrf_score

    # image_results is already sorted by score descending
    for i, (img_item, _) in enumerate(image_results):
        fn = img_item.get("file_name", "unknown")
        pg = img_item.get("page", 0)
        key = (fn, pg)
        
        rrf_score = 1.0 / (RRF_K + i + 1)
        if key not in registry:
            registry[key] = {"text_rrf": 0.0, "image_rrf": rrf_score, "content": img_item}
        else:
            if rrf_score > registry[key]["image_rrf"]:
                registry[key]["image_rrf"] = rrf_score
            # Prefer image item as primary content for display
            registry[key]["content"] = img_item

    # Calculate final scores
    fused = []
    
    # Scaling factor to make top possible score 1.0 for the UI
    SCALE = (RRF_K + 1)

    for (fn, pg), data in registry.items():
        t_rrf = data["text_rrf"]
        i_rrf = data["image_rrf"]
        
        final = (w_text * t_rrf) + (w_image * i_rrf)
        
        fused.append({
            "type": "image" if isinstance(data["content"], dict) else "text",
            "content": data["content"],
            "normalized_text_score": t_rrf * SCALE,
            "normalized_image_score": i_rrf * SCALE,
            "normalized_score": max(t_rrf, i_rrf) * SCALE,
            "final_score": final * SCALE,
            "file_name": fn,
            "page": pg
        })

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused[:top_k]
