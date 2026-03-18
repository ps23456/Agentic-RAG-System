"""Singleton RAG service wrapping existing retrieval/indexing modules."""
import logging
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_FOLDER

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.search_index = None
        self.page_trees: list[dict] = []
        self.image_count = 0
        self._lock = threading.Lock()
        self._indexing = False
        self._index_status = "idle"

    @property
    def data_folder(self) -> str:
        return DATA_FOLDER

    @property
    def uploads_folder(self) -> str:
        return os.path.join(DATA_FOLDER, "uploads")

    def _groq_key(self) -> str:
        return os.environ.get("GROQ_API_KEY", "")

    def _openai_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    def _mistral_ocr_key(self) -> str:
        return os.environ.get("MISTRAL_OCR_API_KEY", "")

    def _llm_key_and_provider(self) -> tuple[str, str]:
        k = self._groq_key()
        if k:
            return k, "groq"
        k = self._openai_key()
        if k:
            return k, "openai"
        return "", "groq"

    def initialize(self):
        try:
            from indexing.text_indexer import load_existing_index
            self.search_index = load_existing_index()
            if self.search_index:
                logger.info("Loaded existing text index: %d chunks", len(self.search_index.chunks))
        except Exception as e:
            logger.warning("Could not load text index: %s", e)

        try:
            from indexing.page_tree import load_all_trees
            self.page_trees = load_all_trees()
            logger.info("Loaded %d tree(s)", len(self.page_trees))
        except Exception as e:
            logger.warning("Could not load trees: %s", e)

        try:
            from indexing.image_indexer import get_image_index_count
            self.image_count = get_image_index_count()
        except Exception:
            self.image_count = 0

    def get_patients(self) -> list[str]:
        if not self.search_index or not hasattr(self.search_index, "chunks"):
            return []
        try:
            from retrieval.agentic_rag import get_robust_catalog
            catalog = get_robust_catalog(self.search_index.chunks)
            return catalog.get("known_patients", [])
        except Exception:
            return []

    def _merge_tree_and_rag(self, query: str, results: list[dict]) -> list[dict]:
        """Unified merge: tree method and agentic RAG reinforce each other.

        - Results found by BOTH methods get a corroboration boost
        - Tree-only PDF hits not in fusion get injected
        - Keyword relevance is scored dynamically from the query
        """
        import re

        try:
            from indexing.page_tree import tree_keyword_retrieve
        except ImportError:
            return results

        tree_hits = tree_keyword_retrieve(query, self.page_trees, self.data_folder, top_k=10)

        tree_scores: dict[tuple, float] = {}
        tree_data: dict[tuple, dict] = {}
        for tc, raw_score in tree_hits:
            key = (tc.file_name, tc.page_number)
            tree_scores[key] = raw_score
            tree_data[key] = {
                "type": "text",
                "file_name": tc.file_name,
                "page": tc.page_number,
                "score": 0.0,
                "snippet": (tc.text or "")[:500],
                "patient_name": getattr(tc, "patient_name", "") or "",
                "section_title": getattr(tc, "_section_title", "") or "",
            }

        q_lower = query.lower().strip()
        q_words = [w for w in re.split(r'\W+', q_lower) if len(w) > 2]

        def _compute_final(r: dict) -> float:
            base = r.get("score", 0)
            key = (r.get("file_name", ""), r.get("page"))
            snippet = (r.get("snippet", "") or "").lower()
            title = (r.get("section_title", "") or "").lower()
            text = snippet + " " + title

            tree_sc = tree_scores.get(key, 0)
            if tree_sc > 0 and base > 0:
                base += 0.3 * tree_sc
            elif tree_sc > 0:
                base += 0.15 * tree_sc

            if q_words:
                if q_lower in text:
                    base += 0.4
                matched = sum(1 for w in q_words if w in text)
                base += (matched / len(q_words)) * 0.25

            return base

        for r in results:
            r["score"] = round(_compute_final(r), 4)

        existing_keys = {(r["file_name"], r.get("page")) for r in results}
        max_score = max((r["score"] for r in results), default=1.0)

        for key, tree_sc in tree_scores.items():
            if key in existing_keys:
                continue
            r = tree_data[key]
            r["score"] = round(_compute_final(r) + max_score * 0.7, 4)
            results.append(r)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def chat(self, query: str, patient_filter: str | None = None, web_search: bool = False) -> dict:
        """Run the full search pipeline and return structured results."""
        from retrieval.agentic_rag import run_agentic_rag, get_robust_catalog
        from retrieval.text_retriever import TextRetriever
        from retrieval.image_retriever import ImageRetriever
        from indexing.page_tree import generate_summary_from_results

        if not self.search_index:
            return {"summary": "No documents indexed yet. Please upload files and reindex.", "sources": [], "results": []}

        text_retriever = TextRetriever(self.search_index)
        image_retriever = ImageRetriever()
        catalog = get_robust_catalog(self.search_index.chunks)

        user_meta = {}
        if patient_filter and patient_filter != "All":
            aliases = catalog.get("patient_name_aliases", {})
            user_meta["patient_name"] = aliases.get(patient_filter, [patient_filter])

        llm_key, provider = self._llm_key_and_provider()

        understanding, fused, direct_answer = run_agentic_rag(
            query,
            self.search_index,
            text_retriever,
            image_retriever,
            catalog=catalog,
            user_metadata_filter=user_meta or None,
            use_llm=bool(llm_key),
            llm_api_key=llm_key,
            llm_provider=provider,
            page_trees=self.page_trees if self.page_trees else None,
            data_folder=self.data_folder,
        )

        summary = ""
        sources: list[dict] = []
        if fused:
            try:
                summary, sources = generate_summary_from_results(query, fused, llm_key, provider)
            except Exception as e:
                logger.warning("Summary generation failed: %s", e)
                if direct_answer:
                    summary = direct_answer

        results = []
        for r in (fused or [])[:15]:
            content = r["content"]
            if r["type"] == "text":
                results.append({
                    "type": "text",
                    "file_name": getattr(content, "file_name", ""),
                    "page": getattr(content, "page_number", None),
                    "score": r.get("final_score", 0),
                    "snippet": (getattr(content, "text", "") or "")[:500],
                    "patient_name": getattr(content, "patient_name", "") or "",
                    "section_title": getattr(content, "_section_title", "") or "",
                })
            else:
                results.append({
                    "type": "image",
                    "file_name": content.get("file_name", ""),
                    "page": content.get("page"),
                    "score": r.get("final_score", 0),
                    "snippet": (content.get("ocr_text", "") or "")[:500],
                    "patient_name": content.get("patient_name", "") or "",
                    "is_pdf_page": content.get("is_pdf_page", False),
                    "path": content.get("path", ""),
                })

        if self.page_trees:
            results = self._merge_tree_and_rag(query, results)

        if web_search:
            try:
                from backend.services.web_search import web_search as do_web_search
                web_hits = do_web_search(query, max_results=5)
                web_context = ""
                for i, wh in enumerate(web_hits):
                    results.append({
                        "type": "web",
                        "file_name": wh.get("source", "") or wh.get("title", "Web"),
                        "page": None,
                        "score": round(0.5 - i * 0.05, 4),
                        "snippet": wh.get("snippet", ""),
                        "patient_name": "",
                        "section_title": wh.get("title", ""),
                        "url": wh.get("url", ""),
                    })
                    web_context += f"\n[Web: {wh.get('title', '')}] {wh.get('snippet', '')}"

                if web_context and summary:
                    llm_key, prov = self._llm_key_and_provider()
                    if llm_key:
                        try:
                            from retrieval.agentic_rag import _call_llm
                            enhanced = _call_llm(
                                f"You previously answered a query with document-based information. "
                                f"Now incorporate relevant web search results into your answer. "
                                f"Keep the existing answer structure but add web insights where helpful.\n\n"
                                f"Query: {query}\n\nExisting answer:\n{summary}\n\n"
                                f"Web results:\n{web_context}\n\n"
                                f"Return the enhanced answer in markdown.",
                                llm_key, prov,
                            )
                            if enhanced and len(enhanced) > 50:
                                summary = enhanced
                        except Exception as e:
                            logger.warning("Web summary enhancement failed: %s", e)
            except Exception as e:
                logger.warning("Web search integration failed: %s", e)

        if direct_answer and not summary:
            summary = direct_answer

        score_sources: list[dict] = []
        seen_src: set[tuple] = set()
        for r in results:
            fn = r.get("file_name", "")
            pg = r.get("page")
            key = (fn, pg)
            if fn and key not in seen_src:
                score_sources.append({"file_name": fn, "page": pg, "title": fn})
                seen_src.add(key)
            if len(score_sources) >= 8:
                break

        if score_sources:
            sources = score_sources

        return {
            "summary": summary,
            "sources": sources,
            "results": results,
            "intent": (understanding or {}).get("intent", ""),
            "reasoning": (understanding or {}).get("reasoning", ""),
        }

    def reindex_docs(self):
        """Re-index documents: text chunks + tree indexes."""
        if self._indexing:
            return
        self._indexing = True
        self._index_status = "indexing_docs"
        try:
            from indexing.text_indexer import build_text_index
            from indexing.page_tree import build_trees_for_folder

            llm_key, provider = self._llm_key_and_provider()

            self.search_index = build_text_index(
                self.data_folder,
                enable_vision=True,
                vision_provider=provider,
                vision_api_key=llm_key,
            )

            trees, _n = build_trees_for_folder(self.data_folder, "both", llm_key, provider)
            self.page_trees = trees

            self._index_status = "done"
        except Exception as e:
            logger.error("Reindex docs failed: %s", e)
            self._index_status = f"error: {e}"
        finally:
            self._indexing = False

    def reindex_images(self):
        """Re-index images only."""
        if self._indexing:
            return
        self._indexing = True
        self._index_status = "indexing_images"
        try:
            from indexing.image_indexer import build_image_index, get_image_index_count

            build_image_index(self.data_folder)
            self.image_count = get_image_index_count()

            self._index_status = "done"
        except Exception as e:
            logger.error("Reindex images failed: %s", e)
            self._index_status = f"error: {e}"
        finally:
            self._indexing = False

    def reindex(self):
        """Re-index everything (docs + images)."""
        if self._indexing:
            return
        self._indexing = True
        self._index_status = "indexing"
        try:
            from indexing.text_indexer import build_text_index
            from indexing.image_indexer import build_image_index, get_image_index_count
            from indexing.page_tree import build_trees_for_folder

            llm_key, provider = self._llm_key_and_provider()

            self.search_index = build_text_index(
                self.data_folder,
                enable_vision=True,
                vision_provider=provider,
                vision_api_key=llm_key,
            )

            build_image_index(self.data_folder)
            self.image_count = get_image_index_count()

            trees, _n = build_trees_for_folder(self.data_folder, "both", llm_key, provider)
            self.page_trees = trees

            self._index_status = "done"
        except Exception as e:
            logger.error("Reindex failed: %s", e)
            self._index_status = f"error: {e}"
        finally:
            self._indexing = False

    def get_index_info(self) -> dict:
        chunk_count = len(self.search_index.chunks) if self.search_index and hasattr(self.search_index, "chunks") else 0
        return {
            "chunk_count": chunk_count,
            "tree_count": len(self.page_trees),
            "image_count": self.image_count,
            "patients": self.get_patients(),
            "status": self._index_status,
            "indexing": self._indexing,
        }


rag = RAGService()
