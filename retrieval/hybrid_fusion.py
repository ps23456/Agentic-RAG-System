"""
Score normalization (min-max) and weighted fusion for Multimodal Hybrid RAG.
Includes phrase-match boost: chunks containing query key phrases rank higher (universal solution).
"""
import re
from typing import Any, Dict, List, Tuple

from config import MULTIMODAL_HYBRID_WEIGHTS, MULTIMODAL_HYBRID_TOP_K


def _extract_query_phrases(query: str) -> List[str]:
    """Extract meaningful 2–4 word phrases for retrieval boosting (skip stopwords)."""
    stop = {"a", "an", "the", "of", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "to", "for", "in", "on", "at",
            "by", "with", "from", "as", "into", "through", "during", "before", "after"}
    words = re.findall(r"\b[a-zA-Z0-9]+\b", (query or "").lower())
    phrases = []
    for n in range(4, 1, -1):
        for i in range(len(words) - n + 1):
            span = words[i : i + n]
            if not any(w in stop for w in span) or n >= 3:
                phrases.append(" ".join(span))
    return phrases


def _extract_main_intent_phrases(query: str) -> Tuple[List[str], List[str]]:
    """
    Parse query to separate MAIN INTENT (what user wants) vs FILTER/SCOPE (who/which).
    E.g. "primary diagnosis of teresa brown" -> main=["primary diagnosis"], filter=["teresa brown"]
    Chunks matching main intent rank above those matching only filter.
    """
    q = (query or "").strip().lower()
    main_phrases: List[str] = []
    filter_phrases: List[str] = []
    all_phrases = _extract_query_phrases(query)

    # Patterns: "X of Y", "X for Y", "Y's X" -> X is main intent, Y is filter
    m = re.search(r"^(.+?)\s+of\s+(.+)$", q)
    if m:
        main_phrases.extend(p for p in all_phrases if p and p in m.group(1).lower() and len(p) >= 3)
        filter_phrases.extend(p for p in all_phrases if p and p in m.group(2).lower() and len(p) >= 3)
    if not main_phrases and not filter_phrases:
        m = re.search(r"^(.+?)\s+for\s+(.+)$", q)
        if m:
            main_phrases.extend(p for p in all_phrases if p and p in m.group(1).lower() and len(p) >= 3)
            filter_phrases.extend(p for p in all_phrases if p and p in m.group(2).lower() and len(p) >= 3)
    if not main_phrases and not filter_phrases:
        m = re.search(r"^(.+?)'s\s+(.+)$", q)
        if m:
            main_phrases.extend(p for p in all_phrases if p and p in m.group(2).lower() and len(p) >= 3)
            filter_phrases.extend(p for p in all_phrases if p and p in m.group(1).lower() and len(p) >= 3)

    if not main_phrases:
        main_phrases = [p for p in all_phrases if p and len(p) >= 3]
    if not filter_phrases:
        filter_phrases = []

    return main_phrases, filter_phrases


def boost_phrase_matching(
    text_results: List[Tuple[Any, float]],
    query: str,
    main_phrases_override: List[str] | None = None,
    filter_phrases_override: List[str] | None = None,
) -> List[Tuple[Any, float]]:
    """
    Intent-aware boost: MAIN PRIORITY = chunks matching what user wants;
    NEXT PRIORITY = chunks matching filter/scope. Non-matching last.
    When main_phrases_override/filter_phrases_override are provided (from LLM), use those
    for dynamic, query-agnostic understanding. Otherwise fall back to regex parsing.
    """
    if not query or not text_results:
        return text_results
    if main_phrases_override is not None or filter_phrases_override is not None:
        main_phrases = main_phrases_override or []
        filter_phrases = filter_phrases_override or []
        all_phrases = main_phrases + [p for p in filter_phrases if p and p not in main_phrases]
        if not main_phrases and not filter_phrases:
            main_phrases, filter_phrases = _extract_main_intent_phrases(query)
            all_phrases = main_phrases + [p for p in filter_phrases if p not in main_phrases]
    else:
        main_phrases, filter_phrases = _extract_main_intent_phrases(query)
        all_phrases = main_phrases + [p for p in filter_phrases if p not in main_phrases]
    if not all_phrases and not main_phrases and not filter_phrases:
        return text_results

    query_norm = re.sub(r"\s+", " ", (query or "").strip().lower())
    query_min_len = 10
    # Core query: longest phrase from all_phrases (handles "i want X" -> X is core)
    core_phrases = sorted([p for p in all_phrases if len(p) >= 15], key=len, reverse=True)

    tier0 = []  # verbatim: chunk contains full query or core phrase (exact chunk for any query)
    tier1 = []  # matches main intent
    tier2 = []  # matches filter only
    tier3 = []  # no match

    for item in text_results:
        chunk, score = item
        text = (getattr(chunk, "text", "") or "").lower()
        main_matched = [p for p in main_phrases if p in text]
        filter_matched = [p for p in filter_phrases if p in text and p not in main_phrases]

        verbatim = (
            (len(query_norm) >= query_min_len and query_norm in text)
            or any(cp in text for cp in core_phrases[:3])
        )

        if verbatim:
            tier0.append((item, len(query_norm), score))
        elif main_matched:
            longest_main = max(len(p) for p in main_matched)
            n_main = len(main_matched)
            tier1.append((item, longest_main, n_main, len(filter_matched), score))
        elif filter_matched:
            longest_filter = max(len(p) for p in filter_matched)
            tier2.append((item, longest_filter, score))
        else:
            tier3.append(item)

    tier0.sort(key=lambda x: (-x[1], -x[2]))
    tier1.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4]))
    tier2.sort(key=lambda x: (-x[1], -x[2]))
    return [t[0] for t in tier0] + [t[0] for t in tier1] + [t[0] for t in tier2] + tier3


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

    # Map key -> {text_rrf: float, image_rrf: float, content: Best_Content_Obj, best_score: float}
    registry = {}
    RRF_K = 60

    # text_results is already sorted by score descending (best first)
    # Use BEST occurrence per (file, page): keep highest RRF and content from highest-scoring chunk
    for i, (chunk, score) in enumerate(text_results):
        fn = getattr(chunk, "file_name", "unknown")
        pg = getattr(chunk, "page_number", 0)
        key = (fn, pg)
        rrf_score = 1.0 / (RRF_K + i + 1)
        if key not in registry:
            registry[key] = {
                "text_rrf": rrf_score,
                "image_rrf": 0.0,
                "content": chunk,
                "best_score": float(score),
            }
        else:
            # Same page from different source (e.g. tree vs chunk): keep best RRF and best-scoring content
            if rrf_score > registry[key]["text_rrf"]:
                registry[key]["text_rrf"] = rrf_score
            if score > registry[key]["best_score"]:
                registry[key]["content"] = chunk
                registry[key]["best_score"] = float(score)

    # image_results is already sorted by score descending
    for i, (img_item, _) in enumerate(image_results):
        fn = img_item.get("file_name", "unknown")
        pg = img_item.get("page", 0)
        key = (fn, pg)
        
        rrf_score = 1.0 / (RRF_K + i + 1)
        if key not in registry:
            registry[key] = {
                "text_rrf": 0.0,
                "image_rrf": rrf_score,
                "content": img_item,
                "best_score": -1.0,
            }
        else:
            if rrf_score > registry[key]["image_rrf"]:
                registry[key]["image_rrf"] = rrf_score
            # For text_heavy queries, prefer text so user sees actual content (e.g. from Markdown)
            # For image_heavy/hybrid, prefer image for visual context
            if query_type != "text_heavy":
                registry[key]["content"] = img_item

    # Calculate final scores
    fused = []
    
    # Scaling factor to make top possible score 1.0 for the UI
    SCALE = (RRF_K + 1)

    for (fn, pg), data in registry.items():
        t_rrf = data["text_rrf"]
        i_rrf = data["image_rrf"]
        
        # For text_heavy: exclude image-only results so we never show images for text queries
        if query_type == "text_heavy" and t_rrf <= 0:
            continue
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
