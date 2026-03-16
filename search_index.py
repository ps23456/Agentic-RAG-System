"""
Search index: BM25, Vector (sentence-transformers + FAISS), and Hybrid fusion.
"""
import re
import logging
from typing import List, Tuple

from rank_bm25 import BM25Okapi
import numpy as np

from document_loader import Chunk, load_and_chunk_folder
from config import (
    DATA_FOLDER,
    BM25_TOP_K,
    VECTOR_TOP_K,
    HYBRID_TOP_K,
    RRF_K,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_CANDIDATES,
    RERANKER_TOP_K,
    RERANKER_MAX_PASSAGE_CHARS,
    VECTOR_BACKEND,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

# Lazy load heavy deps
_sentence_transformers = None
_faiss = None
_reranker_model = None


def _get_tokenizer():
    """Simple tokenizer: lowercase, alphanumeric + keep hyphens for claim IDs."""
    def tokenize(text: str) -> List[str]:
        text = re.sub(r"[^\w\s\-]", " ", text.lower())
        return [t for t in text.split() if len(t) > 1 or t.isdigit()]
    return tokenize


def _load_embedding_model():
    global _sentence_transformers
    if _sentence_transformers is None:
        import torch  # noqa: F401
        import torch.nn  # noqa: F401
        from sentence_transformers import SentenceTransformer
        _sentence_transformers = SentenceTransformer(EMBEDDING_MODEL)
    return _sentence_transformers


def _load_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _load_reranker():
    """Lazy-load BGE reranker (CrossEncoder)."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker_model


class SearchIndex:
    """
    Holds chunks, BM25 index, and vector index (FAISS in-memory or Chroma persistent).
    Supports BM25 search, vector search, and hybrid (RRF or score fusion).
    """

    def __init__(self, chunks: List[Chunk] | None = None):
        self.chunks: List[Chunk] = chunks or []
        self.chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: List[List[str]] = []
        self._vector_index = None  # FAISS index when backend=faiss
        self._chroma_collection = None  # Chroma collection when backend=chroma
        self._embedding_model = None
        self._embedding_dim: int = 0

    def build_bm25(self) -> None:
        tokenizer = _get_tokenizer()
        self._bm25_corpus = [tokenizer(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def build_vector_index(self) -> None:
        if not self.chunks:
            self._vector_index = None
            self._chroma_collection = None
            return

        # 1. Fetch existing IDs and mtimes from Chroma (if available)
        existing_mtimes = {}
        if VECTOR_BACKEND == "chroma":
            self.load_chroma_collection()
            if self._chroma_collection:
                try:
                    # Fetching all IDs and metadatas to check for modification
                    results = self._chroma_collection.get(include=["metadatas"])
                    if results and results["metadatas"]:
                        for i, meta in enumerate(results["metadatas"]):
                            chid = results["ids"][i]
                            mtime = meta.get("last_modified", 0.0)
                            existing_mtimes[chid] = float(mtime)
                except Exception as e:
                    logger.debug("Incremental check failed: %s", e)

        # 2. Filter chunks that need encoding
        chunks_to_encode = []
        for c in self.chunks:
            if c.chunk_id not in existing_mtimes:
                chunks_to_encode.append(c)
            elif abs(existing_mtimes[c.chunk_id] - c.last_modified) > 0.1:
                chunks_to_encode.append(c)

        if not chunks_to_encode:
            logger.info("All text chunks are already indexed and unchanged. Skipping encoding.")
            if VECTOR_BACKEND == "chroma" and self._chroma_collection:
                return # Already loaded
            # If FAISS or not loaded yet, we still need to build/load (though FAISS is usually transient)
        
        model = _load_embedding_model()
        self._embedding_model = model
        
        if chunks_to_encode:
            texts = [
                (c.embedding_text if getattr(c, "embedding_text", "") else c.text)
                for c in chunks_to_encode
            ]
            logger.info(f"Encoding {len(texts)} new/changed chunks...")
            new_embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
            new_embeddings = np.array(new_embeddings, dtype=np.float32)
        else:
            new_embeddings = np.array([], dtype=np.float32).reshape(0, 0)

        # 3. Build/Update
        if VECTOR_BACKEND == "chroma":
            self._incremental_build_chroma(chunks_to_encode, new_embeddings)
        else:
            # For FAISS, we still rebuild the whole thing as it's in-memory
            all_texts = [(c.embedding_text if getattr(c, "embedding_text", "") else c.text) for c in self.chunks]
            embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=64)
            embeddings = np.array(embeddings, dtype=np.float32)
            self._build_faiss_index(embeddings)

    def _incremental_build_chroma(self, chunks_to_encode: List[Chunk], new_embeddings: np.ndarray) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.warning("chromadb not installed.")
            return

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._chroma_collection = collection

        if not chunks_to_encode:
            return

        ids = [c.chunk_id for c in chunks_to_encode]
        metadatas = [
            {
                "file_name": c.file_name,
                "page_number": c.page_number or 0,
                "document_type": c.document_type,
                "patient_name": getattr(c, "patient_name", "") or "",
                "claim_number": getattr(c, "claim_number", "") or "",
                "policy_number": getattr(c, "policy_number", "") or "",
                "group_number": getattr(c, "group_number", "") or "",
                "doctor_name": getattr(c, "doctor_name", "") or "",
                "doc_quality": getattr(c, "doc_quality", "") or "",
                "embedding_text": getattr(c, "embedding_text", "") or "",
                "last_modified": float(getattr(c, "last_modified", 0.0)),
            }
            for c in chunks_to_encode
        ]
        documents = [c.text for c in chunks_to_encode]
        collection.upsert(ids=ids, embeddings=new_embeddings.tolist(), metadatas=metadatas, documents=documents)
        logger.info(f"Upserted {len(ids)} chunks to ChromaDB.")

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        # --- FAISS setup (commented out - using Chroma). Uncomment below to use FAISS again. ---
        # faiss = _load_faiss()
        # index = faiss.IndexFlatIP(embeddings.shape[1])
        # faiss.normalize_L2(embeddings)
        # index.add(embeddings)
        # self._vector_index = index
        # self._chroma_collection = None
        pass

    def load_chroma_collection(self) -> None:
        """Connect to existing ChromaDB collection without re-embedding."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            return
        import os
        if not os.path.exists(CHROMA_PERSIST_DIR):
            return
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
        try:
            self._chroma_collection = client.get_collection(CHROMA_COLLECTION_NAME)
            self._embedding_model = _load_embedding_model()
        except Exception:
            self._chroma_collection = None

    def _build_chroma_index(self, embeddings: np.ndarray) -> None:
        """Legacy rebuild method - redirected to incremental internally now."""
        self._incremental_build_chroma(self.chunks, embeddings)

    def vector_search(
        self,
        query: str,
        top_k: int = VECTOR_TOP_K,
        metadata_filter: dict | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        metadata_filter: dict e.g. {"patient_name": "Rika Popper"} for ChromaDB where clause.
        """
        if not self.chunks:
            return []
        model = self._embedding_model or _load_embedding_model()
        q_emb = model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype=np.float32)

        if self._chroma_collection is not None:
            return self._vector_search_chroma(q_emb, top_k, query, metadata_filter)
        if self._vector_index is None:
            return []
        return self._vector_search_faiss(q_emb, top_k, query)

    def _vector_search_faiss(self, q_emb: np.ndarray, top_k: int, query: str) -> List[Tuple[Chunk, float]]:
        # --- FAISS search (commented out - using Chroma). Uncomment below to use FAISS again. ---
        # faiss = _load_faiss()
        # faiss.normalize_L2(q_emb)
        # scores, indices = self._vector_index.search(q_emb, min(top_k, len(self.chunks)))
        # hits: List[Tuple[Chunk, float]] = []
        # for s, i in zip(scores[0], indices[0]):
        #     if i < 0:
        #         break
        #     hits.append((self.chunks[i], float(s)))
        # return self._rerank_vector_hits(hits, query, top_k)
        return []

    def _vector_search_chroma(
        self,
        q_emb: np.ndarray,
        top_k: int,
        query: str,
        metadata_filter: dict | None = None,
    ) -> List[Tuple[Chunk, float]]:
        where = None
        if metadata_filter:
            clauses = []
            for k, v in metadata_filter.items():
                if v is None or (isinstance(v, (list, str)) and not v):
                    continue
                if k in ("patient_name", "doctor_name"):
                    # v can be str or list (OCR variants: ["Rita Pepper", "Rika Popper", "Rita Peyer"])
                    if isinstance(v, list):
                        variants = list(dict.fromkeys(str(x).strip() for x in v if x))
                    else:
                        variants = list({v, str(v).upper(), str(v).title(), " ".join(w.capitalize() for w in str(v).split())})
                    if not variants:
                        continue
                    if len(variants) == 1:
                        clauses.append({k: {"$eq": variants[0]}})
                    else:
                        clauses.append({"$or": [{k: {"$eq": var}} for var in variants]})
                else:
                    clauses.append({k: {"$eq": v}})
            if clauses:
                where = {"$and": clauses} if len(clauses) > 1 else clauses[0]

        kwargs = {
            "query_embeddings": q_emb.tolist(),
            "n_results": min(top_k, len(self.chunks)),
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self._chroma_collection.query(**kwargs)
        hits: List[Tuple[Chunk, float]] = []
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results.get("distances") else []
        for chunk_id, dist in zip(ids, distances):
            chunk = self.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            # Chroma cosine distance: 0 = identical, 2 = opposite. Convert to similarity (higher = better).
            similarity = 1.0 - (float(dist) if dist is not None else 0)
            hits.append((chunk, similarity))
        return self._rerank_vector_hits(hits, query, top_k)

    def _rerank_vector_hits(self, hits: List[Tuple[Chunk, float]], query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        """Apply token-overlap and page boost, then return top_k."""
        tokenizer = _get_tokenizer()
        q_tokens = tokenizer(query)
        q_token_set = set(q_tokens)
        reranked: List[Tuple[Chunk, float]] = []
        for chunk, base_score in hits:
            boosted = base_score
            if boosted > 0 and q_token_set:
                chunk_tokens = set(tokenizer(chunk.text))
                overlap = len(q_token_set & chunk_tokens)
                if overlap >= len(q_token_set):
                    boosted *= 1.4
                elif overlap >= max(1, len(q_token_set) - 1):
                    boosted *= 1.15
                if chunk.page_number and chunk.page_number > 0:
                    boosted *= 1.0 + 0.03 / float(chunk.page_number)
            reranked.append((chunk, float(boosted)))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def verbatim_search(
        self,
        query: str,
        metadata_filter: dict | None = None,
        max_results: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """
        Universal: find chunks containing the query (exact or all-keywords).
        Exact substring first; then soft match (all significant words) so similar
        queries like "RBI provisioning divergence" find "Divergence in Asset Classification
        and Provisioning" without requiring exact words. No bias—understands intent.
        """
        if not query or not self.chunks:
            return []
        q = re.sub(r"\s+", " ", (query or "").strip().lower())
        if len(q) < 4:
            return []
        q_words = [w for w in q.split() if len(w) >= 3]
        # Skip negation words for soft match: documents may use "un-X" or "non-X" instead of "not X"
        NEGATION_WORDS = frozenset({"not", "without", "no", "non", "none"})
        required_words = [w for w in q_words if w not in NEGATION_WORDS] or q_words
        exact_out = []
        soft_out = []
        for c in self.chunks:
            if not self._chunk_matches_filter(c, metadata_filter):
                continue
            text = (c.text or "").lower()
            pos = text.find(q)
            if pos != -1:
                exact_out.append((c, 1.0, pos))
            elif required_words and all(w in text for w in required_words):
                first_pos = min(text.find(w) for w in required_words)
                soft_out.append((c, 0.9, first_pos))
        exact_out.sort(key=lambda x: (x[2], -len((x[0].text or ""))))
        soft_out.sort(key=lambda x: (x[2], -len((x[0].text or ""))))
        seen = set()
        out = []
        for c, s, _ in exact_out + soft_out:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                out.append((c, s))
                if len(out) >= max_results:
                    break
        return out

    def _chunk_matches_filter(self, chunk: Chunk, metadata_filter: dict | None) -> bool:
        """Return True if chunk matches metadata_filter (or filter is empty).
        For patient_name/doctor_name, v can be list of OCR variants."""
        if not metadata_filter:
            return True
        for k, v in metadata_filter.items():
            if v is None or (isinstance(v, (list, str)) and not v):
                continue
            chunk_val = getattr(chunk, k, "") or ""
            if k in ("patient_name", "doctor_name") and isinstance(v, list):
                if chunk_val not in v:
                    return False
            elif chunk_val != v:
                return False
        return True

    def bm25_search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        metadata_filter: dict | None = None,
    ) -> List[Tuple[Chunk, float]]:
        if not self._bm25 or not self.chunks:
            return []
        tokenizer = _get_tokenizer()
        q_tokens = tokenizer(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)

        # Heuristic re-ranking:
        # - Exact phrase boost: if chunk contains the query as a phrase (e.g. "chest pain"), strong boost.
        # - Boost chunks that contain *all* query tokens (good for headings like
        #   "INSURED/PATIENT INFORMATION" even if punctuation differs).
        # - Slightly prefer earlier pages within the same document so page 1
        #   headers rank ahead of boilerplate on later pages.
        q_token_set = set(q_tokens)
        # Normalize query for exact phrase check: lowercase, collapse spaces (chunk text normalized same way)
        query_phrase = re.sub(r"\s+", " ", query.lower().strip())
        scores = np.array(scores, dtype=float)
        for i, chunk in enumerate(self.chunks):
            boosted = scores[i]
            if boosted <= 0:
                continue

            chunk_lower = (chunk.text or "").lower()
            # Exact phrase boost: chunk contains the full query as a phrase (e.g. "chest pain", "subjective symptoms")
            if len(query_phrase) >= 2 and query_phrase in chunk_lower:
                boosted *= 2.5

            # Token coverage boost
            chunk_tokens = set(self._bm25_corpus[i]) if i < len(self._bm25_corpus) else set()
            overlap = len(q_token_set & chunk_tokens)
            if overlap >= len(q_token_set):
                # All query tokens present: strong boost
                boosted *= 1.8
            elif overlap >= max(1, len(q_token_set) - 1):
                # Most tokens present: mild boost
                boosted *= 1.3

            # Page priority boost
            if chunk.page_number and chunk.page_number > 0:
                boosted *= 1.0 + 0.05 / float(chunk.page_number)

            scores[i] = boosted

        indices = np.argsort(scores)[::-1]
        out = []
        for i in indices:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            if self._chunk_matches_filter(chunk, metadata_filter):
                out.append((chunk, float(scores[i])))
                if len(out) >= top_k:
                    break
        return out

    def _reciprocal_rank_fusion(
        self,
        bm25_hits: List[Tuple[Chunk, float]],
        vector_hits: List[Tuple[Chunk, float]],
        k: int = RRF_K,
    ) -> List[Tuple[Chunk, float]]:
        """
        Reciprocal Rank Fusion: score(d) = sum 1/(k + rank(d)).
        Returns list of (chunk, fusion_score) sorted by score descending.
        """
        scores: dict[str, float] = {}
        for rank, (chunk, _) in enumerate(bm25_hits, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank)
        for rank, (chunk, _) in enumerate(vector_hits, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank)
        # Sort by fusion score
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [(self.chunk_by_id[cid], scores[cid]) for cid in sorted_ids]

    def hybrid_search(
        self,
        query: str,
        top_k: int = HYBRID_TOP_K,
        fusion: str = "rrf",
        metadata_filter: dict | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Run BM25 and vector search, then fuse with RRF or weighted score.
        metadata_filter: e.g. {"patient_name": "Rika Popper"} for ChromaDB + BM25 filtering.
        """
        bm25_hits = self.bm25_search(query, top_k=BM25_TOP_K, metadata_filter=metadata_filter)
        vector_hits = self.vector_search(query, top_k=VECTOR_TOP_K, metadata_filter=metadata_filter)
        if fusion == "rrf":
            fused = self._reciprocal_rank_fusion(bm25_hits, vector_hits, k=RRF_K)
        else:
            # Weighted score fusion with min-max normalization
            fused = self._weighted_fusion(bm25_hits, vector_hits)
        return fused[:top_k]

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = RERANKER_TOP_K,
        prioritize_exact_phrase: bool = False,
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank candidates using BGE reranker (CrossEncoder).
        candidates: list of (chunk, score) from e.g. hybrid_search.
        prioritize_exact_phrase: if True (BM25 only), chunks containing the query as a phrase rank first.
        Returns list of (chunk, reranker_score) sorted by relevance, length top_k.
        """
        if not candidates:
            return []
        try:
            model = _load_reranker()
        except Exception as e:
            logger.warning("Reranker not available: %s. Returning unreranked candidates.", e)
            return candidates[:top_k]
        pairs = []
        chunks = []
        for chunk, _ in candidates:
            raw = (chunk.text or "").strip()
            if len(raw) > RERANKER_MAX_PASSAGE_CHARS:
                # Truncate at word boundary to avoid cutting mid-word
                cut = raw[:RERANKER_MAX_PASSAGE_CHARS].rsplit(maxsplit=1)
                text = cut[0] if cut else raw[:RERANKER_MAX_PASSAGE_CHARS]
            else:
                text = raw or ""
            pairs.append((query, text))
            chunks.append(chunk)
        scores = model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        else:
            scores = list(scores)
        out = [(chunks[i], float(scores[i])) for i in range(len(chunks))]
        # Exact phrase first only for BM25; Vector/Hybrid keep pure reranker order
        if prioritize_exact_phrase:
            query_phrase = re.sub(r"\s+", " ", (query or "").lower().strip())
            query_exact_case = re.sub(r"\s+", " ", (query or "").strip())
            if len(query_phrase) >= 2:
                exact = [(c, s) for c, s in out if query_phrase in (c.text or "").lower()]
                other = [(c, s) for c, s in out if query_phrase not in (c.text or "").lower()]
                # Sort exact by reranker score desc; tie-break: prefer exact-case phrase (e.g. "Chest Pain" in doc)
                exact.sort(
                    key=lambda x: (
                        -x[1],
                        0 if (query_exact_case and query_exact_case in (x[0].text or "")) else 1,
                    )
                )
                other.sort(key=lambda x: x[1], reverse=True)
                out = exact + other
                return [(c, float(s)) for c, s in out[:top_k]]
        out.sort(key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in out[:top_k]]

    def hybrid_search_with_rerank(
        self,
        query: str,
        top_k: int = RERANKER_TOP_K,
        fusion: str = "rrf",
        metadata_filter: dict | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve more hybrid candidates, then rerank with BGE reranker.
        Returns top_k results after reranking (more relevant docs on top).
        """
        fused = self.hybrid_search(
            query, top_k=RERANKER_CANDIDATES, fusion=fusion, metadata_filter=metadata_filter
        )
        return self.rerank(query, fused, top_k=top_k)

    def _weighted_fusion(
        self,
        bm25_hits: List[Tuple[Chunk, float]],
        vector_hits: List[Tuple[Chunk, float]],
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> List[Tuple[Chunk, float]]:
        bm25_scores = {c.chunk_id: s for c, s in bm25_hits}
        vector_scores = {c.chunk_id: s for c, s in vector_hits}
        all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        bm25_vals = [bm25_scores.get(i, 0) for i in all_ids]
        vector_vals = [vector_scores.get(i, 0) for i in all_ids]
        bm25_min, bm25_max = min(bm25_vals), max(bm25_vals) or 1
        vec_min, vec_max = min(vector_vals), max(vector_vals) or 1
        combined = []
        for cid in all_ids:
            b = (bm25_scores.get(cid, 0) - bm25_min) / (bm25_max - bm25_min + 1e-9)
            v = (vector_scores.get(cid, 0) - vec_min) / (vec_max - vec_min + 1e-9)
            combined.append((self.chunk_by_id[cid], bm25_weight * b + vector_weight * v))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined
