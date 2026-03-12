"""
Query classifier for Multimodal Hybrid RAG.
Determines modality weights: text_heavy, image_heavy, or hybrid.
Modular: rule-based now; replace with LLM-based classifier later if needed.
"""
from typing import Tuple

from config import QUERY_CLASSIFIER_IMAGE_KEYWORDS


class QueryClassifier:
    """
    Classify query intent for modality weighting.
    Returns one of: "text_heavy", "image_heavy", "hybrid".
    """

    def __init__(self, image_keywords: Tuple[str, ...] | None = None):
        self.image_keywords = image_keywords or QUERY_CLASSIFIER_IMAGE_KEYWORDS

    def classify(self, query: str) -> str:
        """
        Rule-based classification.
        If query contains image-related keywords -> image_heavy.
        Else -> text_heavy.
        Optional: hybrid when query is long and has some image keywords (not implemented here).
        """
        if not (query or "").strip():
            return "text_heavy"
        q_lower = query.strip().lower()
        for kw in self.image_keywords:
            if kw in q_lower:
                return "image_heavy"
        return "text_heavy"


def classify_query(query: str, classifier: QueryClassifier | None = None) -> str:
    """Convenience: classify query using default or provided classifier."""
    if classifier is None:
        classifier = QueryClassifier()
    return classifier.classify(query)
