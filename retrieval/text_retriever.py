"""
Text retrieval pipeline: BM25 + dense (SentenceTransformer) -> hybrid fusion -> BGE rerank.
Uses existing SearchIndex; returns (chunk, score) for normalization in hybrid_fusion.
"""
from typing import List, Tuple, Any

from config import (
    RERANKER_CANDIDATES,
    RERANKER_TOP_K,
    MULTIMODAL_HYBRID_TEXT_CANDIDATES,
    STRUCTURED_SKIP_RERANKER,
)


class TextRetriever:
    """
    Runs the full text pipeline on an existing SearchIndex.
    Does NOT mix embedding spaces; uses only the text collection.
    """

    def __init__(self, search_index: Any):
        """
        search_index: SearchIndex from search_index.py (has hybrid_search, rerank).
        """
        self.index = search_index

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        fusion: str = "rrf",
        metadata_filter: dict | None = None,
    ) -> List[Tuple[Any, float]]:
        """
        BM25 + dense -> RRF fusion -> BGE rerank.
        metadata_filter: e.g. {"patient_name": "Rika Popper"} for ChromaDB/BM25 filtering.
        Returns list of (chunk, reranker_score). Scores are raw (not normalized).
        """
        top_k = top_k or RERANKER_TOP_K
        n_candidates = max(
            min(MULTIMODAL_HYBRID_TEXT_CANDIDATES, RERANKER_CANDIDATES),
            top_k,
        )
        if not self.index or not getattr(self.index, "chunks", None):
            return []
        hybrid_hits = self.index.hybrid_search(
            query, top_k=n_candidates, fusion=fusion, metadata_filter=metadata_filter
        )
        if not hybrid_hits:
            return []

        if STRUCTURED_SKIP_RERANKER and self._majority_structured(hybrid_hits):
            hybrid_hits.sort(key=lambda x: x[1], reverse=True)
            return hybrid_hits[:top_k]

        reranked = self.index.rerank(
            query,
            hybrid_hits,
            top_k=top_k,
            prioritize_exact_phrase=False,
        )
        return reranked

    @staticmethod
    def _majority_structured(hits) -> bool:
        """True if >50% of candidate chunks come from structured documents."""
        if not hits:
            return False
        n_structured = sum(1 for c, _ in hits if getattr(c, "doc_quality", "") == "structured")
        return n_structured > len(hits) / 2

    def retrieve_one_per_patient(
        self,
        patients: list[str],
        query: str = "",
        top_per_patient: int = 1,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve best chunk per patient. Guarantees coverage for list queries.
        Tries ChromaDB metadata filter first; falls back to text scan when metadata is empty.
        Returns [(chunk, score), ...] one per patient, ordered by patient list.
        """
        if not self.index or not patients:
            return []
        out = []
        found_via_metadata = set()

        for p in patients:
            if not p:
                continue
            hits = self.index.hybrid_search(
                query or "Patient Name",
                top_k=top_per_patient,
                fusion="rrf",
                metadata_filter={"patient_name": p},
            )
            if hits:
                reranked = self.index.rerank(
                    query or "Patient Name",
                    hits,
                    top_k=top_per_patient,
                    prioritize_exact_phrase=True,
                )
                out.extend(reranked)
                found_via_metadata.add(p)

        # Fallback: when ChromaDB has no patient_name metadata, find chunks by text
        missing = [p for p in patients if p and p not in found_via_metadata]
        if missing and self.index and getattr(self.index, "chunks", None):
            for p in missing:
                for c in self.index.chunks:
                    text = (getattr(c, "text", "") or "").lower()
                    if p.lower() in text:
                        out.append((c, 0.9))
                        break
        return out

    def get_unique_metadata_values(self, key: str) -> list[str]:
        """Return sorted unique non-empty values for metadata key (e.g. patient_name)."""
        if not self.index or not getattr(self.index, "chunks", None):
            return []
        seen = set()
        for c in self.index.chunks:
            v = getattr(c, key, "") or ""
            if v and v not in seen:
                seen.add(v)
        return sorted(seen)
