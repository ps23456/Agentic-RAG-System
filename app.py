"""
Insurance Claim Search - Streamlit app.
Single query input; three result sections: BM25, Vector, Hybrid.
Retrieval-only, explainable results.
"""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env so GEMINI_API_KEY / OPENAI_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# Fix TLS cert path when venv moved or path mismatch (e.g. testing vs testing 3)
# Force use of current Python's certifi; setdefault fails when .env has wrong path
try:
    import certifi
    _ca = certifi.where()
    if _ca and os.path.isfile(_ca):
        os.environ["SSL_CERT_FILE"] = _ca
        os.environ["REQUESTS_CA_BUNDLE"] = _ca
except Exception:
    pass

# Patch transformers to accept PyTorch 2.2 (transformers 4.44+ requires 2.4).
def _patch_transformers_accept_torch_22():
    try:
        import torch  # noqa: F401
        import transformers.utils.import_utils as _tfu
        _old = _tfu.is_torch_available
        if hasattr(_old, "cache_clear"):
            _old.cache_clear()
        _tfu.is_torch_available = lambda: True
        if "torch" in getattr(_tfu, "BACKENDS_MAPPING", {}):
            _orig = _tfu.BACKENDS_MAPPING["torch"]
            _tfu.BACKENDS_MAPPING["torch"] = (lambda: True, _orig[1] if isinstance(_orig, (list, tuple)) else _orig)
        import transformers.utils as _tu
        _tu.is_torch_available = lambda: True
        import transformers as _tr
        _tr.is_torch_available = lambda: True
    except Exception:
        pass

try:
    import torch  # noqa: F401
    # PyTorch 2.2 has no torch.get_default_device(); transformers 5.x expects it
    if not hasattr(torch, "get_default_device"):
        torch.get_default_device = lambda: torch.device(torch._C._get_default_device())
    _patch_transformers_accept_torch_22()
except ImportError:
    pass

import streamlit as st

from config import (
    DATA_FOLDER,
    HYBRID_TOP_K,
    UPLOADS_SUBDIR,
    VECTOR_MIN_SIMILARITY,
    HYBRID_MIN_FUSION_SCORE,
    RERANKER_TOP_K,
    RERANKER_CANDIDATES,
    MULTIMODAL_TOP_K,
    MULTIMODAL_HYBRID_TOP_K,
    MULTIMODAL_HYBRID_WEIGHTS,
    AUTO_METADATA_FILTER,
    PREFER_USER_METADATA_FILTER,
    METADATA_DIVERSITY_ENABLED,
    METADATA_DIVERSITY_MAX_PER_ENTITY,
    METADATA_DIVERSITY_CANDIDATES,
    LLM_QUERY_UNDERSTANDING,
)
from document_loader import (
    load_and_chunk_folder,
    get_ocr_status,
    count_image_files_in_folder,
    set_mistral_ocr_key,
    get_mistral_ocr_key,
    extract_text_from_pdf_page,
    find_pdf_page_containing_phrases,
    extract_chunk_metadata,
)
from search_index import SearchIndex
from llm_insight import (
    get_insight,
    explain_image,
    medical_analysis,
    build_context,
    classify_medical_document,
)
from multimodal_index import build_multimodal_index, search_multimodal, get_multimodal_index_count
from indexing.text_indexer import build_text_index, load_existing_index
from indexing.image_indexer import build_image_index, get_image_index_count
from retrieval.query_classifier import classify_query
from retrieval.query_metadata_extractor import (
    extract_metadata_from_query,
    merge_metadata_filters,
    get_index_metadata_catalog,
)
from retrieval.text_retriever import TextRetriever
from retrieval.image_retriever import ImageRetriever
from retrieval.hybrid_fusion import fuse_results, boost_phrase_matching
from retrieval.result_diversifier import diversify_by_metadata, diversify_fused_results
from retrieval.llm_query_understanding import understand_query_llm, detect_list_patients_intent
from retrieval.agentic_rag import run_agentic_rag, get_robust_catalog


def _check_pytorch():
    """Ensure PyTorch is installed so sentence-transformers and vector search work."""
    try:
        import torch  # noqa: F401
        return True, None
    except ImportError as e:
        return False, str(e)


def _extract_query_phrases(query: str) -> list[str]:
    """Extract meaningful 2–4 word phrases from query for snippet matching (skip stopwords)."""
    stop = {"a", "an", "the", "of", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "to", "for", "in", "on", "at",
            "by", "with", "from", "as", "into", "through", "during", "before", "after"}
    words = re.findall(r"\b[a-zA-Z0-9]+\b", (query or "").lower())
    phrases = []
    for n in range(4, 1, -1):  # Prefer longer phrases first
        for i in range(len(words) - n + 1):
            span = words[i : i + n]
            if not any(w in stop for w in span) or n >= 3:
                phrases.append(" ".join(span))
    return phrases


def _resolve_llm_for_tree(llm_provider: str, llm_api_key: str) -> tuple[str, str]:
    """Resolve which API key and provider to use for tree indexing/search.
    Uses whichever provider the user selected in the sidebar."""
    provider_lower = (llm_provider or "").lower()
    if "openai" in provider_lower or "chatgpt" in provider_lower:
        key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        return key.strip(), "openai"
    elif "gemini" in provider_lower:
        key = llm_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        return key.strip(), "gemini"
    elif "groq" in provider_lower:
        key = llm_api_key or os.environ.get("GROQ_API_KEY", "")
        return key.strip(), "groq"
    elif "mistral" in provider_lower:
        key = llm_api_key or os.environ.get("MISTRAL_API_KEY", "")
        return key.strip(), "openai"  # Mistral uses OpenAI-compatible API
    # Fallback: try all env keys
    for env_key, prov in [("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"), ("GEMINI_API_KEY", "gemini"), ("GOOGLE_API_KEY", "gemini")]:
        k = os.environ.get(env_key, "")
        if k:
            return k.strip(), prov
    return "", "groq"


def _count_tree_nodes(nodes) -> int:
    """Count total nodes in a tree (including nested children)."""
    if isinstance(nodes, list):
        return sum(_count_tree_nodes(n) for n in nodes)
    elif isinstance(nodes, dict):
        c = 1
        if "nodes" in nodes:
            c += _count_tree_nodes(nodes["nodes"])
        return c
    return 0


def snippet(text: str, query: str | None = None, max_len: int = 400, prioritize_phrase: str | None = None) -> str:
    """
    Build a short snippet for display.

    If a query is provided, try to center the snippet around the first occurrence
    of any meaningful phrase from the query (e.g. "primary diagnosis", "teresa brown").
    prioritize_phrase: when set (e.g. patient name), center snippet around it so it's visible.
    """
    if not text:
        return ""
    flat = text.replace("\n", " ").strip()
    if len(flat) <= max_len:
        return flat

    phrases = _extract_query_phrases(query) if query else []
    if prioritize_phrase and prioritize_phrase.strip():
        phrases = [prioritize_phrase.strip()] + [p for p in phrases if p != prioritize_phrase.strip()]
    if phrases:
        flat_lower = flat.lower()
        best_idx = -1
        best_len = 0
        for p in phrases:
            if len(p) < 3:
                continue
            idx = flat_lower.find(p.lower() if isinstance(p, str) else p)
            if idx != -1 and len(p) > best_len:
                best_idx = idx
                best_len = len(p)
        if best_idx != -1:
            start = max(0, best_idx - max_len // 3)
            end = min(len(flat), start + max_len)
            window = flat[start:end]
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(flat) else ""
            return prefix + window.strip() + suffix

    # Fallback: simple prefix
    return flat[: max_len - 3].rsplit(" ", 1)[0] + "..."


def render_result(chunk, score: float, score_label: str, idx: int, query: str | None = None, data_folder: str | None = None):
    st.markdown(f"**{idx}. {chunk.file_name}**" + (f" (p.{chunk.page_number})" if chunk.page_number else ""))
    chunk_text = chunk.text or ""
    display_text = chunk_text
    found_page = chunk.page_number
    if data_folder and query and chunk.file_name and chunk.file_name.lower().endswith(".pdf") and chunk.page_number:
        phrases = _extract_query_phrases(query)
        if not any(p and len(p) >= 3 and p in chunk_text.lower() for p in phrases):
            for root, _dirs, files in os.walk(data_folder):
                if chunk.file_name in files:
                    pdf_path = os.path.join(root, chunk.file_name)
                    if os.path.isfile(pdf_path):
                        found_page, page_text = find_pdf_page_containing_phrases(pdf_path, phrases, start_page=chunk.page_number)
                        if page_text:
                            display_text = page_text
                    break
    ctx_hint = f" · Context from p.{found_page}" if found_page and found_page != chunk.page_number else ""
    st.caption(f"{score_label}: {score:.4f}  ·  Type: {chunk.document_type}  ·  ID: `{chunk.chunk_id}`{ctx_hint}")
    st.text(snippet(display_text, query=query))
    st.divider()


def main():
    st.set_page_config(page_title="Insurance Claim Search", layout="wide")
    st.title("Insurance Claim Search")

    ok, err = _check_pytorch()
    if not ok:
        st.error(
            "**PyTorch is not installed.** Vector search (and Hybrid) need PyTorch. "
            "In your terminal, activate the project venv and run:\n\n"
            "```bash\npip install torch\n```\n\n"
            "Or reinstall all dependencies:\n\n"
            "```bash\npip install -r requirements.txt\n```"
        )
        st.caption("After installing, restart the Streamlit app (stop and run `streamlit run app.py` again).")
        return

    # Ensure patch is applied (transformers 4.44+ requires torch>=2.4; we accept 2.2)
    _patch_transformers_accept_torch_22()

    st.markdown("Search over **policy PDFs**, **claim PDFs**, and **medical reports** from a local folder or **uploaded files**. "
                "Compare **BM25** (exact/keyword), **Vector** (semantic), and **Hybrid** (fusion) results.")

    # RAG mode: Hybrid (BM25+Vector+RRF) vs Multimodal (CLIP text+image in separate index)
    st.sidebar.markdown("---")
    st.sidebar.subheader("RAG mode")
    rag_mode = st.sidebar.radio(
        "Choose RAG",
        ["Hybrid RAG", "Multimodal RAG", "Multimodal Hybrid RAG", "Medical Analysis"],
        key="rag_mode",
        help="Hybrid: BM25+Vector+reranker. Multimodal: single CLIP index. Multimodal Hybrid: text+image fusion. Medical Analysis: Specialized clinical review & temporal comparison.",
    )

    data_folder = st.sidebar.text_input("Data folder", value=DATA_FOLDER, help="Folder to read documents from. Uploads are saved here under an 'uploads' subfolder.")
    uploads_folder = os.path.join(data_folder, UPLOADS_SUBDIR)
    if not os.path.isdir(data_folder):
        try:
            os.makedirs(data_folder, exist_ok=True)
            os.makedirs(uploads_folder, exist_ok=True)
        except OSError:
            pass
    if not os.path.isdir(data_folder):
        st.sidebar.warning(f"Folder not found: {data_folder}. Create it or use the default, then upload or add files.")

    # --- Upload documents (real data) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload documents")
    st.sidebar.caption("Add your own PDFs, images, Markdown (.md), Text (.txt), or JSON files to search on.")
    uploaded = st.sidebar.file_uploader(
        "Choose PDF, image, Markdown (.md), Text (.txt), or JSON files",
        type=["md", "txt", "json", "pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
        accept_multiple_files=True,
        key="uploader",
    )
    if uploaded:
        os.makedirs(uploads_folder, exist_ok=True)
        saved = 0
        for f in uploaded:
            if f.name:
                path = os.path.join(uploads_folder, f.name)
                try:
                    with open(path, "wb") as out:
                        out.write(f.getvalue())
                    saved += 1
                except OSError:
                    st.sidebar.error(f"Could not save {f.name}")
        if saved:
            st.sidebar.success(f"Saved {saved} file(s) to `{uploads_folder}`. Click **Index / Re-index** to search them.")
        st.sidebar.caption("Upload more files or click **Index / Re-index documents** below to include them in search.")

    # --- Mistral OCR (cloud-powered, high accuracy) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("OCR Engine")
    mistral_ocr_key = st.sidebar.text_input(
        "Mistral OCR API key",
        value=os.environ.get("MISTRAL_API_KEY", ""),
        type="password",
        key="mistral_ocr_key",
        help="Mistral OCR provides high-accuracy text extraction from images and scanned PDFs. Get a key at https://console.mistral.ai/",
    )
    set_mistral_ocr_key(mistral_ocr_key)

    num_images = count_image_files_in_folder(data_folder)
    if num_images > 0:
        ocr_ok, ocr_msg = get_ocr_status()
        if ocr_ok:
            st.sidebar.success(f"✓ {ocr_msg}")
        else:
            st.sidebar.warning(
                "**Your uploaded image is not in the index yet.** To get real matches from it:\n\n"
                "1. Add a Mistral OCR API key above (recommended), **or**\n"
                "2. Install Tesseract: `brew install tesseract` (macOS)\n\n"
                "Then click **Index / Re-index documents** below."
            )

    # --- AI Insight (optional: ChatGPT or Gemini) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("AI -- RAG")
    st.sidebar.caption("Use an LLM to summarize or answer from retrieved chunks.")
    _gemini_available = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    _groq_available = bool(os.environ.get("GROQ_API_KEY"))
    _mistral_available = bool(os.environ.get("MISTRAL_API_KEY") or get_mistral_ocr_key())
    _hf_available = bool(os.environ.get("HUGGINGFACE_API_KEY"))
    llm_provider = st.sidebar.selectbox(
        "Provider",
        ["Off", "OpenAI (ChatGPT)", "Gemini", "Groq", "Mistral", "Hugging Face"],
        index=4 if _mistral_available else (3 if _groq_available else (5 if _hf_available else (2 if _gemini_available else 0))),
        key="llm_provider",
    )
    llm_api_key = ""
    if llm_provider == "OpenAI (ChatGPT)":
        llm_api_key = st.sidebar.text_input(
            "OpenAI API key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            key="openai_key",
            help="Or set OPENAI_API_KEY in the environment.",
        )
    elif llm_provider == "Gemini":
        llm_api_key = st.sidebar.text_input(
            "Gemini API key",
            value=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
            key="gemini_key",
            help="Loaded from .env (GEMINI_API_KEY) if set. You can also paste here.",
        )
    elif llm_provider == "Groq":
        llm_api_key = st.sidebar.text_input(
            "Groq API key",
            value=os.environ.get("GROQ_API_KEY", ""),
            type="password",
            key="groq_key",
            help="Loaded from .env (GROQ_API_KEY) if set. You can also paste here.",
        )
    elif llm_provider == "Mistral":
        llm_api_key = st.sidebar.text_input(
            "Mistral API key (vision + chat)",
            value=os.environ.get("MISTRAL_API_KEY") or get_mistral_ocr_key(),
            type="password",
            key="mistral_key",
            help="Same key as Mistral OCR. Used for vision (image classification) and chat.",
        )
    elif llm_provider == "Hugging Face":
        llm_api_key = st.sidebar.text_input(
            "Hugging Face API key",
            value=os.environ.get("HUGGINGFACE_API_KEY", ""),
            type="password",
            key="hf_key",
            help="Loaded from .env (HUGGINGFACE_API_KEY) if set. Used for Inference API.",
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Indexing Settings")
    indexing_mode = st.sidebar.radio(
        "For PDFs:",
        ["Chunk + embedding", "Tree structure (PageIndex)", "Both"],
        index=0,
        key="indexing_mode",
        help="Chunk: BM25 + ChromaDB (default). Tree: vectorless, LLM reasons over section structure. Both: build both.",
    )
    indexing_mode_key = {"Chunk + embedding": "chunk", "Tree structure (PageIndex)": "tree", "Both": "both"}.get(indexing_mode, "chunk")
    enable_vision_captioning = st.sidebar.checkbox(
        "Vision LLM Captioning",
        value=False,
        help="Use the active LLM (above) to generate semantic descriptions for images during indexing. Costs API credits.",
        key="enable_vision_captioning"
    )

    if "search_index" not in st.session_state:
        st.session_state.search_index = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "multimodal_count" not in st.session_state:
        st.session_state.multimodal_count = 0
    if "multimodal_hybrid_image_count" not in st.session_state:
        st.session_state.multimodal_hybrid_image_count = 0
    if "page_trees" not in st.session_state:
        st.session_state.page_trees = []
    if "page_tree_count" not in st.session_state:
        st.session_state.page_tree_count = 0

    if rag_mode == "Hybrid RAG":
        if st.session_state.search_index is None:
            with st.spinner("Loading existing text index from ChromaDB..."):
                existing = load_existing_index()
                if existing:
                    st.session_state.search_index = existing
                    st.session_state.chunk_count = len(existing.chunks)
                    st.sidebar.success(f"Loaded {st.session_state.chunk_count} chunks from existing index.")
                else:
                    st.sidebar.info("No existing index found. Click 'Re-index' to build.")

        if st.sidebar.button("Re-index documents (full rebuild)", type="primary"):
            with st.spinner("Loading documents and building Hybrid index..."):
                chunks = load_and_chunk_folder(
                    data_folder,
                    enable_vision=enable_vision_captioning,
                    vision_provider=llm_provider,
                    vision_api_key=llm_api_key
                )
                if not chunks:
                    st.sidebar.warning("No chunks extracted. Add PDF or image files to the data folder.")
                    st.session_state.search_index = None
                    st.session_state.chunk_count = 0
                else:
                    index = SearchIndex(chunks=chunks)
                    index.build_bm25()
                    try:
                        index.build_vector_index()
                    except ImportError as e:
                        if "torch" in str(e).lower() or "pytorch" in str(e).lower():
                            _patch_transformers_accept_torch_22()
                            try:
                                index.build_vector_index()
                            except ImportError:
                                st.error(
                                    "**PyTorch not detected.** Run the app with the project venv:\n\n"
                                    "```bash\n./run.sh\n```\n\nor\n\n"
                                    "```bash\n.venv/bin/python -m streamlit run app.py\n```"
                                )
                                st.stop()
                        else:
                            st.error(f"Import error building vector index: {e}")
                            st.stop()
                    st.session_state.search_index = index
                    st.session_state.chunk_count = len(chunks)
                    st.sidebar.success(f"Rebuilt index: {st.session_state.chunk_count} chunks.")
    elif rag_mode == "Multimodal Hybrid RAG":
        # Use different key than widget; radio owns st.session_state.indexing_mode
        st.session_state.mh_indexing_mode = indexing_mode_key
        # Load trees when tree or both
        if indexing_mode_key in ("tree", "both"):
            from indexing.page_tree import load_all_trees
            st.session_state.page_trees = load_all_trees()
            st.session_state.page_tree_count = len(st.session_state.page_trees)
        else:
            st.session_state.page_trees = []
            st.session_state.page_tree_count = 0

        # Text index (same as Hybrid); image index (separate collection)
        if st.session_state.search_index is None and indexing_mode_key != "tree":
            with st.spinner("Loading existing text index from ChromaDB..."):
                existing = load_existing_index()
                if existing:
                    st.session_state.search_index = existing
                    st.session_state.chunk_count = len(existing.chunks)
                    st.sidebar.success(f"Loaded {st.session_state.chunk_count} chunks from existing index.")
                else:
                    st.sidebar.info("No existing index found. Click 'Re-index' to build.")
        elif indexing_mode_key == "tree":
            st.sidebar.info("Tree mode: using section structure (no chunks). Click 'Re-index' to build trees.")

        if st.sidebar.button("Re-index documents (full rebuild)", type="primary", key="mh_index_text"):
            with st.spinner("Building index..."):
                try:
                    if indexing_mode_key in ("tree", "both"):
                        from indexing.page_tree import build_trees_for_folder
                        llm_key, _provider = _resolve_llm_for_tree(llm_provider, llm_api_key)
                        trees, n_trees = build_trees_for_folder(data_folder, indexing_mode_key, llm_key, _provider)
                        st.session_state.page_trees = trees
                        st.session_state.page_tree_count = n_trees
                        st.sidebar.success(f"Tree index: {n_trees} PDF(s) with section structure.")
                    if indexing_mode_key in ("chunk", "both"):
                        index = build_text_index(
                            data_folder,
                            enable_vision=enable_vision_captioning,
                            vision_provider=llm_provider,
                            vision_api_key=llm_api_key
                        )
                        st.session_state.search_index = index
                        st.session_state.chunk_count = len(index.chunks)
                        st.sidebar.success(f"Text index rebuilt: {st.session_state.chunk_count} chunks.")
                    elif indexing_mode_key == "tree":
                        st.session_state.search_index = None
                        st.session_state.chunk_count = 0
                except Exception as e:
                    st.sidebar.error(f"Index failed: {e}")
        if st.sidebar.button("Index / Re-index (Image)", type="secondary", key="mh_index_image"):
            with st.spinner("Building image index (CLIP)..."):
                try:
                    n = build_image_index(data_folder)
                    st.session_state.multimodal_hybrid_image_count = n
                    st.sidebar.success(f"Image index: {n} items.")
                except Exception as e:
                    st.sidebar.error(f"Image index failed: {e}")
        st.session_state.multimodal_hybrid_image_count = get_image_index_count()
        cap_parts = []
        if indexing_mode_key != "tree":
            cap_parts.append(f"Text chunks: **{st.session_state.chunk_count}**")
        if indexing_mode_key in ("tree", "both"):
            tree_node_count = sum(
                _count_tree_nodes(t.get("nodes", [])) for t in st.session_state.page_trees
            ) if st.session_state.page_trees else 0
            cap_parts.append(f"Tree index: **{st.session_state.page_tree_count}** PDF(s) · **{tree_node_count}** nodes")
        cap_parts.append(f"Image collection: **{st.session_state.multimodal_hybrid_image_count}**")
        st.sidebar.caption(" · ".join(cap_parts))
    elif rag_mode == "Multimodal RAG":
        # Multimodal RAG: separate index only
        if st.sidebar.button("Index / Re-index (Multimodal)", type="primary"):
            with st.spinner("Building Multimodal index (CLIP text + images)..."):
                try:
                    n = build_multimodal_index(
                        data_folder,
                        enable_vision=enable_vision_captioning,
                        vision_provider=llm_provider,
                        vision_api_key=llm_api_key
                    )
                    st.session_state.multimodal_count = n
                    st.sidebar.success(f"Multimodal index: {n} items (text + images).")
                except Exception as e:
                    st.sidebar.error(f"Multimodal index failed: {e}")
        else:
            st.session_state.multimodal_count = get_multimodal_index_count()
    elif rag_mode == "Medical Analysis":
        # No specific indexing needed here, uses existing ones.
        pass

    if rag_mode == "Hybrid RAG":
        if st.session_state.search_index is None:
            st.info("Upload documents in the sidebar (or add files to the data folder), then click **Index / Re-index documents** to start searching.")
            return

        index = st.session_state.search_index
        st.sidebar.caption(f"Chunks in index: **{st.session_state.chunk_count}**")
        use_reranker = st.sidebar.checkbox(
            "Use BGE reranker",
            value=True,
            key="use_bge_reranker",
            help="Rerank hybrid results with BGE reranker for more relevant top results (recommended).",
        )
        # Metadata filters for ChromaDB / BM25
        st.sidebar.markdown("---")
        st.sidebar.subheader("Metadata filters")
        st.sidebar.caption("Filter results by patient or document type (ChromaDB + BM25).")
        catalog_sidebar = get_index_metadata_catalog(index.chunks)
        unique_patients = catalog_sidebar.get("known_patients", []) or sorted(
            set(getattr(c, "patient_name", "") or "" for c in index.chunks if getattr(c, "patient_name", ""))
        )
        filter_patient = st.sidebar.selectbox(
            "Patient",
            ["All"] + unique_patients,
            key="filter_patient_hybrid",
            help="Restrict results to documents for this patient. Overrides auto-detection from query.",
        )
        use_llm_understanding = st.sidebar.checkbox(
            "LLM query understanding",
            value=LLM_QUERY_UNDERSTANDING,
            key="use_llm_understanding",
            help="Use LLM to parse any query (intent, filters, expanded search). Requires Groq/Gemini/OpenAI key.",
        )
        auto_metadata = st.sidebar.checkbox(
            "Auto-detect filter from query",
            value=AUTO_METADATA_FILTER,
            key="auto_metadata_hybrid",
            help="Infer patient/claim/policy from query when LLM understanding is off.",
        )
        user_metadata_filter = {}
        if filter_patient and filter_patient != "All":
            aliases = catalog_sidebar.get("patient_name_aliases", {})
            user_metadata_filter["patient_name"] = aliases.get(filter_patient, [filter_patient])
        # Show metadata catalog (patients, claims) - useful for any query
        if catalog_sidebar.get("known_patients"):
            st.sidebar.caption("Patients in index: " + ", ".join(catalog_sidebar["known_patients"]))
        if catalog_sidebar.get("known_claims"):
            st.sidebar.caption("Claims: " + ", ".join(catalog_sidebar["known_claims"][:5]) + (" …" if len(catalog_sidebar["known_claims"]) > 5 else ""))
        if catalog_sidebar.get("known_doctors"):
            st.sidebar.caption("Doctors: " + ", ".join(catalog_sidebar["known_doctors"][:5]) + (" …" if len(catalog_sidebar["known_doctors"]) > 5 else ""))
        # Show which files are in the index (so user sees the image is included)
        indexed_files = sorted(set(c.file_name for c in index.chunks))
        if indexed_files:
            st.sidebar.caption("Indexed files: " + ", ".join(f[:25] + ("…" if len(f) > 25 else "") for f in indexed_files))

        query = st.text_input(
        "Query",
        placeholder='e.g. "Why was claim CLM-8891 rejected?" or "Show policy clause for pre-existing conditions"',
            key="query",
        )
        if num_images > 0 and not get_ocr_status()[0]:
            st.caption("💡 Your uploaded image is not searchable yet. Install Tesseract and click **Index / Re-index** (see sidebar) to include its text in results.")
        if not query.strip():
            st.caption("Enter a query above to see BM25, Vector, and Hybrid results side by side.")
            return

        catalog = get_index_metadata_catalog(index.chunks) if index.chunks else {}
        metadata_filter = user_metadata_filter or None
        search_query = query  # Default: use raw query
        is_list_patients = detect_list_patients_intent(query) and catalog.get("known_patients")

        # List-patients: ALWAYS show direct answer + one chunk per patient (no LLM needed)
        understanding = None
        if is_list_patients:
            understanding = {
                "intent": "list_entities",
                "metadata_filter": {},
                "search_query": "Patient Name",
                "direct_answer": "Patients in index: " + ", ".join(catalog["known_patients"]),
                "target_attribute": "patient_names",
            }

        # LLM query understanding (for non-list queries)
        if use_llm_understanding and catalog and not is_list_patients:
            _uq_key = (llm_api_key or "").strip() or os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
            _uq_prov = "groq" if (llm_provider == "Groq" and llm_api_key) or os.environ.get("GROQ_API_KEY") else ("gemini" if (llm_provider == "Gemini" and llm_api_key) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") else ("openai" if (llm_provider == "OpenAI (ChatGPT)" and llm_api_key) or os.environ.get("OPENAI_API_KEY") else "groq"))
            if _uq_key:
                with st.spinner("Understanding query..."):
                    understanding = understand_query_llm(query, catalog, _uq_key.strip(), provider=_uq_prov)
                if understanding:
                    search_query = understanding.get("search_query") or query
                    u_meta = understanding.get("metadata_filter") or {}
                    if u_meta:
                        metadata_filter = merge_metadata_filters(metadata_filter, u_meta, prefer_user=PREFER_USER_METADATA_FILTER)

        # Fallback: rule-based metadata extraction (when LLM off or failed)
        if not understanding and auto_metadata and index.chunks:
            query_extracted = extract_metadata_from_query(
                query,
                known_patients=catalog.get("known_patients"),
                known_claims=catalog.get("known_claims"),
                known_policies=catalog.get("known_policies"),
                known_groups=catalog.get("known_groups"),
            )
            metadata_filter = merge_metadata_filters(
                user_metadata_filter or None,
                query_extracted,
                prefer_user=PREFER_USER_METADATA_FILTER,
            )

        if metadata_filter:
            st.caption(
                "🔍 Filtering by: "
                + ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in metadata_filter.items())
            )

        max_per_entity_hybrid = METADATA_DIVERSITY_MAX_PER_ENTITY
        retrieve_k = METADATA_DIVERSITY_CANDIDATES if METADATA_DIVERSITY_ENABLED else RERANKER_CANDIDATES
        out_k = RERANKER_TOP_K if use_reranker else HYBRID_TOP_K

        text_retriever_hybrid = TextRetriever(index) if index else None

        with st.spinner("Searching..."):
            if is_list_patients and text_retriever_hybrid:
                # One chunk per patient - guaranteed coverage
                one_per = text_retriever_hybrid.retrieve_one_per_patient(
                    catalog["known_patients"], query="Patient Name",
                    patient_name_aliases=catalog.get("patient_name_aliases"),
                )
                bm25_hits = vector_hits = hybrid_hits = one_per
            elif use_reranker:
                bm25_hits = index.bm25_search(
                    search_query, top_k=retrieve_k, metadata_filter=metadata_filter or None
                )
                bm25_hits = index.rerank(query, bm25_hits, top_k=retrieve_k, prioritize_exact_phrase=True)
                vector_hits = index.vector_search(
                    search_query, top_k=retrieve_k, metadata_filter=metadata_filter or None
                )
                vector_hits = index.rerank(query, vector_hits, top_k=retrieve_k)
                hybrid_fused = index.hybrid_search(
                    search_query, top_k=retrieve_k, fusion="rrf", metadata_filter=metadata_filter or None
                )
                hybrid_hits = index.rerank(query, hybrid_fused, top_k=retrieve_k)
            else:
                bm25_hits = index.bm25_search(search_query, top_k=retrieve_k, metadata_filter=metadata_filter or None)
                vector_hits = index.vector_search(search_query, top_k=retrieve_k, metadata_filter=metadata_filter or None)
                hybrid_hits = index.hybrid_search(
                    search_query, top_k=retrieve_k, fusion="rrf", metadata_filter=metadata_filter or None
                )

        # Filter out low-confidence matches only when reranker is off (and not list query)
        if not use_reranker and not is_list_patients:
            vector_hits = [(c, s) for c, s in vector_hits if s >= VECTOR_MIN_SIMILARITY]
            hybrid_hits = [(c, s) for c, s in hybrid_hits if s >= HYBRID_MIN_FUSION_SCORE]

        # Metadata diversity (skip for list query - we already have one per patient)
        if METADATA_DIVERSITY_ENABLED and not is_list_patients:
            bm25_hits = diversify_by_metadata(bm25_hits, top_k=out_k, max_per_entity=max_per_entity_hybrid)
            vector_hits = diversify_by_metadata(vector_hits, top_k=out_k, max_per_entity=max_per_entity_hybrid)
            hybrid_hits = diversify_by_metadata(hybrid_hits, top_k=out_k, max_per_entity=max_per_entity_hybrid)

        st.subheader("Results")
        if understanding and understanding.get("direct_answer"):
            st.success("**" + understanding["direct_answer"] + "**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 1️⃣ BM25 Results" + (" + Reranked" if use_reranker else ""))
            st.caption("Exact keyword matches." + (" Reranked with BGE for relevance." if use_reranker else " claim IDs, clause names, legal terms."))
            if not bm25_hits:
                st.info("No exact keyword matches for this query. Try terms that appear in your documents (e.g. claim ID, policy clause).")
            else:
                for i, (chunk, score) in enumerate(bm25_hits, start=1):
                    render_result(chunk, score, "Rerank score" if use_reranker else "BM25 score", i, query=query, data_folder=data_folder)

        with col2:
            st.markdown("### 2️⃣ Vector Results" + (" + Reranked" if use_reranker else ""))
            st.caption("Semantic matches." + (" Reranked with BGE for relevance." if use_reranker else " medical meaning, paraphrases."))
            if not vector_hits:
                st.info("No confident semantic matches. Try different terms or add more documents (including image text via Tesseract).")
            else:
                for i, (chunk, score) in enumerate(vector_hits, start=1):
                    render_result(chunk, score, "Rerank score" if use_reranker else "Similarity", i, query=query, data_folder=data_folder)

        with col3:
            st.markdown("### 3️⃣ Hybrid Results (RRF)" + (" + Reranked" if use_reranker else ""))
            st.caption("Fusion of BM25 + Vector (Reciprocal Rank Fusion)." + (" Reranked with BGE for relevance." if use_reranker else ""))
            if not hybrid_hits:
                st.info("No confident hybrid matches. Use keywords from your docs or index more files (e.g. image text with Tesseract).")
            else:
                for i, (chunk, score) in enumerate(hybrid_hits, start=1):
                    render_result(chunk, score, "Rerank score" if use_reranker else "Fusion score", i, query=query, data_folder=data_folder)

        st.divider()
        st.caption("Retrieval-only prototype. Results are from your indexed documents; no generated answers.")
        if any(c.file_name.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")) for c in index.chunks):
            st.caption("💡 Your uploaded image is in the index. If it didn’t appear above, try words that actually appear on the form (e.g. **policy number**, **employee name**, **certification**).")

        # --- AI -- RAG: answer from retrieved chunks via ChatGPT or Gemini ---
        if llm_provider != "Off" and llm_api_key.strip():
            st.markdown("---")
            st.subheader("🤖 AI Insight")
            _prov_label = (
                "OpenAI (ChatGPT)" if llm_provider == "OpenAI (ChatGPT)"
                else ("Gemini" if llm_provider == "Gemini" else ("Groq" if llm_provider == "Groq" else "Hugging Face"))
            )
            st.caption("Answer generated from the retrieved chunks above using " + _prov_label + ". For audit, always check the source snippets.")
            context_chunks = []
            seen_ids = set()
            for chunk, _ in hybrid_hits + vector_hits + bm25_hits:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    context_chunks.append(chunk)
            if not context_chunks:
                st.info("No chunks to use as context. Try a query that returns results above, then ask for AI insight.")
            else:
                available_files = sorted(set(c.file_name for c in context_chunks))
                if len(available_files) > 1:
                    st.markdown("**Select documents to use for AI insight:**")
                    selected_files = st.multiselect(
                        "Choose one or more documents",
                        options=available_files,
                        default=available_files,
                        key="insight_file_selector",
                        help="Select which documents should be used to generate the AI insight.",
                    )
                    if not selected_files:
                        st.warning("Please select at least one document to generate insights.")
                        st.stop()
                    filtered_chunks = [c for c in context_chunks if c.file_name in selected_files]
                    st.caption(f"Using {len(filtered_chunks)} chunk(s) from {len(selected_files)} document(s): {', '.join(selected_files)}")
                else:
                    filtered_chunks = context_chunks
                    st.caption(f"Using {len(filtered_chunks)} chunk(s) from: {available_files[0]}")
                include_summary = st.checkbox(
                    "Include brief summary",
                    value=False,
                    key="insight_include_summary",
                    help="Add a 1–3 sentence summary at the end of the insight.",
                )
                if st.button("Generate AI insight", key="gen_insight"):
                    provider_key = (
                        "openai" if llm_provider == "OpenAI (ChatGPT)"
                        else ("gemini" if llm_provider == "Gemini" else ("groq" if llm_provider == "Groq" else "huggingface"))
                    )
                    with st.spinner("Calling LLM..."):
                        answer, err = get_insight(provider_key, llm_api_key.strip(), query.strip(), filtered_chunks, max_context_chars=30000, include_summary=include_summary)
                    if err:
                        st.error(f"**Error:** {err}")
                    else:
                        st.markdown(answer, unsafe_allow_html=False)
                        st.caption("**Sources used:** " + ", ".join(set(c.file_name for c in filtered_chunks)))
        elif llm_provider != "Off":
            st.caption("Set an API key in the sidebar to enable AI insight.")

        # --- Explain image (Hybrid RAG): when results include chunks from image files, vision LLM can describe the image ---
        def _find_file_in_data_folder(fname: str, folder: str) -> str | None:
            for root, _dirs, files in os.walk(folder):
                if fname in files:
                    return os.path.join(root, fname)
            return None

        all_hit_chunks = [c for c, _ in hybrid_hits + vector_hits + bm25_hits]
        image_files_from_hits = []
        seen_fnames = set()
        for c in all_hit_chunks:
            fname = getattr(c, "file_name", "") or ""
            if not fname or fname in seen_fnames:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
                path = _find_file_in_data_folder(fname, data_folder)
                if path:
                    seen_fnames.add(fname)
                    image_files_from_hits.append({"file_name": fname, "page": getattr(c, "page_number", None) or 1, "path": path})
        if image_files_from_hits and llm_provider != "Off" and llm_api_key.strip() and llm_provider in ("OpenAI (ChatGPT)", "Gemini", "Groq"):
            st.markdown("---")
            st.subheader("🖼️ Explain what's in the image")
            st.caption("Some results are from image files. Use a vision-capable LLM (OpenAI, Gemini, or Groq) to describe or explain the image using your query.")
            choice_hybrid = st.selectbox(
                "Which image to explain?",
                range(len(image_files_from_hits)),
                format_func=lambda i: f"{image_files_from_hits[i]['file_name']} (p.{image_files_from_hits[i]['page']})",
                key="hybrid_explain_which",
            )
            if st.button("Explain this image", key="btn_hybrid_explain_image"):
                item = image_files_from_hits[choice_hybrid]
                provider_key = (
                    "openai" if llm_provider == "OpenAI (ChatGPT)"
                    else ("gemini" if llm_provider == "Gemini" else "groq")
                )
                with st.spinner("Vision LLM is describing the image..."):
                    explanation, err = explain_image(provider_key, llm_api_key.strip(), query.strip(), item["path"], page=None)
                if err:
                    st.error(f"**Error:** {err}")
                else:
                    st.markdown(explanation)

    elif rag_mode == "Multimodal Hybrid RAG":
        # Separate text + image collections; query classification; normalized score fusion
        indexing_mode_mh = getattr(st.session_state, "mh_indexing_mode", "chunk")
        has_chunks = st.session_state.search_index is not None and (st.session_state.search_index.chunks if st.session_state.search_index else False)
        has_trees = indexing_mode_mh in ("tree", "both") and st.session_state.page_trees
        if not has_chunks and not has_trees:
            st.info("Click **Index / Re-index documents** in the sidebar to build the index. For Tree mode: select 'Tree structure' then re-index. Optionally click **Index / Re-index (Image)** for image results.")
            return
        index = st.session_state.search_index
        # Metadata filter for Multimodal Hybrid (in sidebar)
        catalog_mh_sidebar = get_robust_catalog(index.chunks) if index and index.chunks else {}
        unique_patients_mh = catalog_mh_sidebar.get("known_patients", [])
        if catalog_mh_sidebar.get("known_patients"):
            st.sidebar.caption("Patients in index: " + ", ".join(catalog_mh_sidebar["known_patients"]))
        st.sidebar.caption("Metadata filter (Multimodal Hybrid):")
        filter_patient_mh = st.sidebar.selectbox(
            "Patient",
            ["All"] + unique_patients_mh,
            key="filter_patient_mh",
            help="Restrict text results to this patient. Overrides auto-detection.",
        )
        use_agentic_rag = st.sidebar.checkbox(
            "Agentic RAG (recommended)",
            value=True,
            key="use_agentic_rag",
            help="Understand query intent first, then route to the right retrieval (list patients, semantic search, etc.).",
        )
        use_llm_understanding_mh = st.sidebar.checkbox(
            "LLM query understanding",
            value=LLM_QUERY_UNDERSTANDING,
            key="use_llm_understanding_mh",
            help="Use LLM to parse any query. Requires Groq/Gemini/OpenAI key.",
        )
        auto_metadata_mh = st.sidebar.checkbox(
            "Auto-detect filter from query",
            value=AUTO_METADATA_FILTER,
            key="auto_metadata_mh",
            help="Infer patient/claim/policy when LLM understanding is off.",
        )
        user_metadata_filter_mh = {}
        if filter_patient_mh and filter_patient_mh != "All":
            aliases_mh = catalog_mh_sidebar.get("patient_name_aliases", {})
            user_metadata_filter_mh["patient_name"] = aliases_mh.get(filter_patient_mh, [filter_patient_mh])
        query_mh = st.text_input(
            "Query",
            placeholder='e.g. "insurance claim" or "diagram with boxes and arrows"',
            key="query_multimodal_hybrid",
        )
        if not query_mh.strip():
            st.caption("Enter a query. Use image-related words (diagram, chart, flow, …) for image-heavy weighting.")
            return

        catalog_mh = get_index_metadata_catalog(index.chunks) if index and index.chunks else {}
        robust_catalog = get_robust_catalog(index.chunks) if index and index.chunks else {}
        metadata_filter_mh = user_metadata_filter_mh or None

        # Flow diagnostic: show why list-patients or normal path was chosen
        if st.sidebar.checkbox("Show flow diagnostic", value=False, key="mh_flow_diag"):
            with st.sidebar.expander("Flow diagnostic"):
                st.write("**Catalog (from attributes):**", catalog_mh)
                st.write("**Robust catalog (from text if needed):**", robust_catalog)
                st.write("**List-patients intent:**", detect_list_patients_intent(query_mh))
                st.write("**known_patients:**", robust_catalog.get("known_patients") or catalog_mh.get("known_patients"))

        # Tree search: run in parallel with chunk retrieval when "both" for speed
        # Cache results in session_state so source-button clicks don't re-trigger search
        tree_fused = []
        tree_future = None
        _tree_exec = None
        tree_summary = ""
        tree_sources = []
        _tree_cache_key = f"_tree_cache_{hash(query_mh)}"
        _tree_cached = st.session_state.get(_tree_cache_key)
        if indexing_mode_mh in ("tree", "both") and st.session_state.page_trees:
            if _tree_cached and _tree_cached.get("query") == query_mh:
                tree_fused = _tree_cached.get("fused", [])
                tree_summary = _tree_cached.get("summary", "")
                tree_sources = _tree_cached.get("sources", [])
            else:
                from indexing.page_tree import tree_search
                llm_key_mh, _prov = _resolve_llm_for_tree(llm_provider, llm_api_key)
                if indexing_mode_mh == "both":
                    _tree_exec = ThreadPoolExecutor(max_workers=2)
                    tree_future = _tree_exec.submit(tree_search, query_mh, st.session_state.page_trees, llm_key_mh, _prov, data_folder)
                else:
                    tree_result = tree_search(query_mh, st.session_state.page_trees, llm_key_mh, _prov, data_folder)
                    tree_fused = [{"type": "text", "content": tc, "final_score": sc} for tc, sc in tree_result.get("chunks", [])]
                    tree_summary = tree_result.get("summary", "")
                    tree_sources = tree_result.get("sources", [])
                    st.session_state[_tree_cache_key] = {
                        "query": query_mh, "fused": tree_fused,
                        "summary": tree_summary, "sources": tree_sources,
                    }

        if indexing_mode_mh == "tree":
            fused = tree_fused
            n_tree_nodes = sum(
                _count_tree_nodes(t.get("nodes", [])) for t in st.session_state.page_trees
            ) if st.session_state.page_trees else 0
            understanding_mh = {
                "intent": "tree_search",
                "query_type": "text_heavy",
                "reasoning": f"LLM navigated {n_tree_nodes} tree nodes across {len(st.session_state.page_trees)} document(s) to find relevant sections.",
            }
            direct_answer = None
            text_results = [(r["content"], r["final_score"]) for r in fused]
            image_results = []
        elif use_agentic_rag and index:
            # Agentic RAG: understand query → route to correct retrieval
            # When agentic is ON, always use LLM if any key is available
            with st.spinner("Agentic RAG: understanding query & retrieving..."):
                text_retriever = TextRetriever(index)
                image_retriever = ImageRetriever()
                understanding_mh, fused, direct_answer = run_agentic_rag(
                    query_mh,
                    index,
                    text_retriever,
                    image_retriever,
                    catalog=robust_catalog or catalog_mh,
                    user_metadata_filter=user_metadata_filter_mh or None,
                    use_llm=True,
                    llm_api_key=llm_api_key or "",
                    llm_provider=llm_provider or "",
                )
                if understanding_mh and understanding_mh.get("metadata_filter"):
                    st.caption(
                        "🔍 Filtering by: "
                        + ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in understanding_mh["metadata_filter"].items())
                    )
        else:
            # Legacy flow (no agentic)
            search_query_mh = query_mh
            understanding_mh = None
            is_list_patients = detect_list_patients_intent(query_mh) and (robust_catalog.get("known_patients") or catalog_mh.get("known_patients"))

            if is_list_patients:
                patients = robust_catalog.get("known_patients") or catalog_mh.get("known_patients", [])
                understanding_mh = {
                    "intent": "list_entities",
                    "metadata_filter": {},
                    "search_query": "Patient Name",
                    "direct_answer": "Patients in index: " + ", ".join(patients),
                    "target_attribute": "patient_names",
                }
            elif use_llm_understanding_mh and (robust_catalog or catalog_mh):
                _uq_key_mh = (llm_api_key or "").strip() or os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
                _uq_prov_mh = "groq" if (llm_provider == "Groq" and llm_api_key) or os.environ.get("GROQ_API_KEY") else ("gemini" if (llm_provider == "Gemini" and llm_api_key) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") else ("openai" if (llm_provider == "OpenAI (ChatGPT)" and llm_api_key) or os.environ.get("OPENAI_API_KEY") else "groq"))
                if _uq_key_mh:
                    with st.spinner("Understanding query..."):
                        understanding_mh = understand_query_llm(query_mh, robust_catalog or catalog_mh, _uq_key_mh.strip(), provider=_uq_prov_mh)
                if understanding_mh:
                    search_query_mh = understanding_mh.get("search_query") or query_mh
                    u_meta = understanding_mh.get("metadata_filter") or {}
                    if u_meta:
                        metadata_filter_mh = merge_metadata_filters(metadata_filter_mh, u_meta, prefer_user=PREFER_USER_METADATA_FILTER)

            if not understanding_mh and auto_metadata_mh and index.chunks:
                query_extracted_mh = extract_metadata_from_query(
                    query_mh,
                    known_patients=(robust_catalog or catalog_mh).get("known_patients"),
                    known_claims=(robust_catalog or catalog_mh).get("known_claims"),
                    known_policies=(robust_catalog or catalog_mh).get("known_policies"),
                    known_groups=(robust_catalog or catalog_mh).get("known_groups"),
                )
                metadata_filter_mh = merge_metadata_filters(
                    user_metadata_filter_mh or None,
                    query_extracted_mh,
                    prefer_user=PREFER_USER_METADATA_FILTER,
                )

            if metadata_filter_mh:
                st.caption(
                    "🔍 Filtering by: "
                    + ", ".join(f"{k.replace('_', ' ')}={v}" for k, v in metadata_filter_mh.items())
                )

            retrieve_k = METADATA_DIVERSITY_CANDIDATES if METADATA_DIVERSITY_ENABLED else MULTIMODAL_HYBRID_TOP_K
            with st.spinner("Running Multimodal Hybrid retrieval..."):
                text_retriever = TextRetriever(index)
                image_retriever = ImageRetriever()
                query_type = classify_query(query_mh)
                if is_list_patients and (robust_catalog.get("known_patients") or catalog_mh.get("known_patients")):
                    patients = robust_catalog.get("known_patients") or catalog_mh.get("known_patients")
                    text_results = text_retriever.retrieve_one_per_patient(
                        patients, query="Patient Name",
                        patient_name_aliases=(robust_catalog or catalog_mh).get("patient_name_aliases"),
                    )
                    image_results = []
                    fused = [{"type": "text", "content": c, "final_score": s} for c, s in text_results]
                else:
                    text_results = text_retriever.retrieve(search_query_mh, top_k=retrieve_k, metadata_filter=metadata_filter_mh or None)
                    if index and hasattr(index, "verbatim_search"):
                        verbatim = index.verbatim_search(query_mh, metadata_filter_mh or None, max_results=10)
                        seen_v = {c.chunk_id for c, _ in verbatim}
                        text_results = list(verbatim) + [(c, s) for c, s in text_results if c.chunk_id not in seen_v]
                    text_results = boost_phrase_matching(text_results, query_mh)
                    image_results = image_retriever.retrieve(search_query_mh, top_n=30, metadata_filter=metadata_filter_mh or None)
                    fuse_top_k = retrieve_k if METADATA_DIVERSITY_ENABLED else MULTIMODAL_HYBRID_TOP_K
                    fused = fuse_results(text_results, image_results, query_type, top_k=fuse_top_k)
                    if METADATA_DIVERSITY_ENABLED and fused:
                        fused = diversify_fused_results(fused, entity_key="patient_name", top_k=MULTIMODAL_HYBRID_TOP_K, max_per_entity=METADATA_DIVERSITY_MAX_PER_ENTITY)
            direct_answer = understanding_mh.get("direct_answer") if understanding_mh else None

        # Both mode: get tree results (from parallel run) and prepend to fused
        if indexing_mode_mh == "both" and tree_future is not None:
            try:
                tree_result = tree_future.result(timeout=90)
                tree_fused = [{"type": "text", "content": tc, "final_score": sc} for tc, sc in tree_result.get("chunks", [])]
                tree_summary = tree_result.get("summary", "")
                tree_sources = tree_result.get("sources", [])
                st.session_state[_tree_cache_key] = {
                    "query": query_mh, "fused": tree_fused,
                    "summary": tree_summary, "sources": tree_sources,
                }
            except Exception:
                tree_fused = []
            if _tree_exec:
                _tree_exec.shutdown(wait=False)
        if indexing_mode_mh == "both" and tree_fused and fused:
            seen_ids = {getattr(r["content"], "chunk_id", None) for r in fused if r["type"] == "text"}
            for tr in tree_fused:
                cid = getattr(tr["content"], "chunk_id", None)
                if cid and cid not in seen_ids:
                    fused.insert(0, tr)
                    seen_ids.add(cid)
            fused = fused[:15]

        # For debug display: derive text_results/image_results from fused
        if indexing_mode_mh == "tree" or use_agentic_rag:
            text_results = [(r["content"], r["final_score"]) for r in fused if r["type"] == "text"]
            image_results = [(r["content"], r["final_score"]) for r in fused if r["type"] == "image"]

        query_type = (
            understanding_mh.get("query_type")
            if (use_agentic_rag and understanding_mh and understanding_mh.get("query_type") in ("text_heavy", "image_heavy", "hybrid"))
            else classify_query(query_mh)
        )
        weights = MULTIMODAL_HYBRID_WEIGHTS.get(query_type, (0.5, 0.5))
        # --- PageIndex-style side-by-side: Summary (left) + PDF viewer (right) ---
        if indexing_mode_mh in ("tree", "both") and tree_summary:
            # Deduplicate sources
            seen_sources = set()
            unique_sources = []
            for src in tree_sources:
                key = (src.get("file_name", ""), src.get("page", ""))
                if key not in seen_sources:
                    seen_sources.add(key)
                    unique_sources.append(src)

            # Auto-select the first source if none selected yet
            if unique_sources and "_tree_view_src" not in st.session_state:
                st.session_state["_tree_view_src"] = {
                    "file_name": unique_sources[0].get("file_name", ""),
                    "page": unique_sources[0].get("page", 1),
                }

            view_src = st.session_state.get("_tree_view_src")

            # Side-by-side layout: summary left (55%), PDF viewer right (45%)
            col_summary, col_viewer = st.columns([55, 45], gap="medium")

            with col_summary:
                st.subheader("PageIndex Results")
                st.markdown(tree_summary)

                # Source buttons
                if unique_sources:
                    st.markdown("---")
                    src_cols = st.columns(min(len(unique_sources), 3))
                    for idx, src in enumerate(unique_sources[:3]):
                        fname = src.get("file_name", "")
                        pg = src.get("page", "")
                        short_name = fname[:18] + "..." if len(fname) > 21 else fname
                        is_active = view_src and view_src.get("file_name") == fname and str(view_src.get("page")) == str(pg)
                        btn_label = f"{'📖' if is_active else '📄'} {short_name} p.{pg}"
                        btn_key = f"tree_src_{idx}_{fname}_{pg}"
                        with src_cols[idx]:
                            if st.button(btn_label, key=btn_key, use_container_width=True, type="primary" if is_active else "secondary"):
                                st.session_state["_tree_view_src"] = {"file_name": fname, "page": pg}
                                st.rerun()

                with st.expander("Tree search details", expanded=False):
                    n_tree_nodes = sum(
                        _count_tree_nodes(t.get("nodes", [])) for t in st.session_state.page_trees
                    ) if st.session_state.page_trees else 0
                    st.caption(f"Navigated {n_tree_nodes} tree nodes across {len(st.session_state.page_trees)} doc(s) · Found {len(fused)} sections")
                    for i, row in enumerate(fused[:5], 1):
                        if row["type"] == "text":
                            chunk = row["content"]
                            pg = getattr(chunk, "page_number", 0) or "?"
                            title = getattr(chunk, "_section_title", "") or ""
                            st.caption(f"{i}. {getattr(chunk, 'file_name', '')} (p.{pg})" + (f" — {title}" if title else ""))

            with col_viewer:
                if view_src:
                    _vs_fname = view_src.get("file_name", "")
                    _vs_page = int(view_src.get("page", 1) or 1)
                    _vs_path = None
                    for root, _dirs, files in os.walk(data_folder):
                        if _vs_fname in files:
                            _vs_path = os.path.join(root, _vs_fname)
                            break
                    if _vs_path and os.path.isfile(_vs_path) and _vs_path.lower().endswith(".pdf"):
                        try:
                            import fitz as _fitz_tree
                            import io as _io_tree
                            from PIL import Image as _PILImage_tree
                            doc = _fitz_tree.open(_vs_path)
                            total_pages = len(doc)

                            # Page navigation
                            nav_cols = st.columns([1, 3, 1])
                            with nav_cols[0]:
                                if _vs_page > 1:
                                    if st.button("◀ Prev", key="tree_pg_prev", use_container_width=True):
                                        st.session_state["_tree_view_src"] = {"file_name": _vs_fname, "page": _vs_page - 1}
                                        st.rerun()
                            with nav_cols[1]:
                                st.markdown(f"<div style='text-align:center'><b>{_vs_fname}</b><br>Page {_vs_page} / {total_pages}</div>", unsafe_allow_html=True)
                            with nav_cols[2]:
                                if _vs_page < total_pages:
                                    if st.button("Next ▶", key="tree_pg_next", use_container_width=True):
                                        st.session_state["_tree_view_src"] = {"file_name": _vs_fname, "page": _vs_page + 1}
                                        st.rerun()

                            pg_idx = max(0, _vs_page - 1)
                            if pg_idx < total_pages:
                                page_obj = doc.load_page(pg_idx)
                                pix = page_obj.get_pixmap(dpi=150, alpha=False)
                                buf = pix.tobytes("png")
                                img = _PILImage_tree.open(_io_tree.BytesIO(buf)).convert("RGB")
                                st.image(img, use_container_width=True)
                            doc.close()
                        except Exception as _e_tree:
                            st.warning(f"Could not render page: {_e_tree}")
                    else:
                        st.info(f"PDF not found: {_vs_fname}")
                else:
                    st.caption("Click a source to view the PDF page here.")

            if indexing_mode_mh == "both":
                st.markdown("---")

        # --- Standard results display (skip in tree-only mode when summary is present) ---
        _tree_only_with_summary = indexing_mode_mh == "tree" and bool(tree_summary)
        if not _tree_only_with_summary:
            st.subheader("Multimodal Hybrid Results" + (" (Agentic)" if use_agentic_rag else "") + (" (Tree)" if indexing_mode_mh in ("tree", "both") else ""))
            st.caption("Fused ranking: text (BM25 + dense + rerank) + image (CLIP). Query type: **%s** · Weights: text=%.2f, image=%.2f" % (query_type, weights[0], weights[1]))
            if (use_agentic_rag or indexing_mode_mh == "tree") and understanding_mh:
                with st.expander("Agent understanding", expanded=False):
                    if understanding_mh.get("reasoning"):
                        st.write("**Reasoning:**", understanding_mh["reasoning"])
                    st.write("**Intent:**", understanding_mh.get("intent", "—"))
                    if understanding_mh.get("main_intent_keywords"):
                        st.write("**Main intent:**", ", ".join(understanding_mh["main_intent_keywords"]))
                    if understanding_mh.get("search_queries"):
                        st.write("**Search queries:**", understanding_mh["search_queries"])
                    if indexing_mode_mh == "tree":
                        st.caption("Tree mode: LLM reasoned over sections to find relevant nodes.")
            answer_to_show = direct_answer
            if answer_to_show:
                st.success("**" + answer_to_show + "**")
        if not fused and not tree_summary:
            st.info("No results. Index documents and optionally the image collection (sidebar).")
        if fused and not _tree_only_with_summary:
            debug = st.checkbox("Show debug (query type, top text/image, fusion)", value=False, key="mh_debug")
            if debug:
                with st.expander("Debug: retrieval and fusion"):
                    st.write("**Query type:**", query_type)
                    st.write("**Fusion weights:**", weights)
                    st.write("**Top 5 text (score):**")
                    for i, (chunk, score) in enumerate(text_results[:5], 1):
                        st.caption("%d. %.4f — %s" % (i, score, (chunk.text or "")[:150].replace(chr(10), " ") + "…"))
                    st.write("**Top 5 image (score):**")
                    for i, (item, score) in enumerate(image_results[:5], 1):
                        st.caption("%d. %.4f — %s (p.%s)" % (i, score, item.get("file_name", ""), item.get("page", "")))
                    st.write("**Final fused (top 10):**")
                    for i, row in enumerate(fused[:10], 1):
                        if row["type"] == "text":
                            st.caption("%d. [text] %.4f — %s" % (i, row["final_score"], (row["content"].text or "")[:80].replace(chr(10), " ") + "…"))
                        else:
                            st.caption("%d. [image] %.4f — %s" % (i, row["final_score"], row["content"].get("file_name", "")))
            for i, row in enumerate(fused[:10], start=1):
                if row["type"] == "text":
                    chunk = row["content"]
                    chunk_text = (getattr(chunk, "text", "") or "").lower()
                    query_norm = re.sub(r"\s+", " ", (query_mh or "").strip().lower())
                    is_exact = i == 1 and len(query_norm) >= 10 and query_norm in chunk_text
                    exact_badge = " · **Exact match**" if is_exact else ""
                    patient_label = ""
                    pn = getattr(chunk, "patient_name", "") or ""
                    if not pn and (chunk.text or ""):
                        from document_loader import extract_chunk_metadata as _ecm
                        pn = _ecm(chunk.text).get("patient_name", "") or ""
                    if pn:
                        patient_label = " · **Patient: %s**" % pn
                    st.markdown("**%d. %s**" % (i, getattr(chunk, "file_name", "")) + (f" (p.{getattr(chunk, 'page_number', 0)})" if getattr(chunk, "page_number", 0) else "") + patient_label + exact_badge)
                    # If this text comes from an image file (e.g. JPG), show the image as well as the OCR text
                    fname = getattr(chunk, "file_name", "") or ""
                    if fname and os.path.splitext(fname)[1].lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
                        for root, _dirs, files in os.walk(data_folder):
                            if fname in files:
                                img_path = os.path.join(root, fname)
                                if os.path.isfile(img_path):
                                    st.image(img_path, use_container_width=True)
                                    st.caption("Image (source of text below)")
                                break
                    display_text = getattr(chunk, "text", "") or ""
                    page_num = getattr(chunk, "page_number", 0) or 1
                    found_page = page_num
                    phrases = _extract_query_phrases(query_mh)
                    has_phrase = any(p and len(p) >= 3 and p in chunk_text for p in phrases)
                    if not has_phrase and fname and fname.lower().endswith(".pdf") and page_num:
                        for root, _dirs, files in os.walk(data_folder):
                            if fname in files:
                                pdf_path = os.path.join(root, fname)
                                if os.path.isfile(pdf_path):
                                    found_page, page_text = find_pdf_page_containing_phrases(pdf_path, phrases, start_page=page_num)
                                    if page_text:
                                        display_text = page_text
                                break
                    st.caption("Fused score: %.4f · Text" % row["final_score"] + (f" · Context from p.{found_page}" if found_page != page_num else ""))
                    st.text(snippet(display_text, query=query_mh, prioritize_phrase=pn or None))
                else:
                    item = row["content"]
                    img_patient = item.get("patient_name", "") or ""
                    img_patient_label = " · **Patient: %s**" % img_patient if img_patient else ""
                    st.markdown("**%d. %s**" % (i, item.get("file_name", "")) + (f" (p.{item.get('page', '')})" if item.get("page") else "") + img_patient_label)
                    st.caption("Fused score: %.4f · Image" % row["final_score"])
                    path = item.get("path", "")
                    if item.get("is_pdf_page") and path:
                        try:
                            import fitz
                            import io as _io
                            from PIL import Image
                            doc = fitz.open(path)
                            page = doc.load_page(int(item.get("page", 1)) - 1)
                            pix = page.get_pixmap(dpi=120, alpha=False)
                            buf = pix.tobytes("png")
                            img = Image.open(_io.BytesIO(buf)).convert("RGB")
                            st.image(img, use_container_width=True)
                            doc.close()
                        except Exception:
                            st.caption("Image: %s" % path)
                    elif path and os.path.isfile(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.caption("Image: %s" % path)
                    ocr_text = item.get("ocr_text", "")
                    if ocr_text:
                        with st.expander("OCR extracted text", expanded=False):
                            st.text(ocr_text[:1500])
                st.divider()

            # --- AI Insight (Multimodal Hybrid RAG): answer from retrieved text chunks ---
            if llm_provider != "Off" and llm_api_key.strip():
                st.markdown("---")
                st.subheader("🤖 AI Insight")
                _prov_label = (
                    "OpenAI (ChatGPT)" if llm_provider == "OpenAI (ChatGPT)"
                    else ("Gemini" if llm_provider == "Gemini" else ("Groq" if llm_provider == "Groq" else "Hugging Face"))
                )
                st.caption("Answer generated from the retrieved text chunks above using " + _prov_label + ". For audit, always check the source snippets.")
                context_chunks_mh = []
                seen_ids_mh = set()
                for row in fused:
                    if row["type"] != "text":
                        continue
                    c = row["content"]
                    cid = getattr(c, "chunk_id", id(c))
                    if cid not in seen_ids_mh:
                        seen_ids_mh.add(cid)
                        context_chunks_mh.append(c)
                if not context_chunks_mh:
                    st.info("No text chunks in results to use as context. Try a query that returns text results for AI insight.")
                else:
                    available_files_mh = sorted(set(getattr(c, "file_name", "") for c in context_chunks_mh if getattr(c, "file_name", "")))
                    if len(available_files_mh) > 1:
                        st.markdown("**Select documents to use for AI insight:**")
                        selected_files_mh = st.multiselect(
                            "Choose one or more documents",
                            options=available_files_mh,
                            default=available_files_mh,
                            key="mh_insight_file_selector",
                            help="Select which documents should be used to generate the AI insight.",
                        )
                        if not selected_files_mh:
                            st.warning("Please select at least one document to generate insights.")
                            filtered_chunks_mh = []
                        else:
                            filtered_chunks_mh = [c for c in context_chunks_mh if getattr(c, "file_name", "") in selected_files_mh]
                            st.caption(f"Using {len(filtered_chunks_mh)} chunk(s) from {len(selected_files_mh)} document(s): {', '.join(selected_files_mh)}")
                    else:
                        filtered_chunks_mh = context_chunks_mh
                        st.caption(f"Using {len(filtered_chunks_mh)} chunk(s) from: {available_files_mh[0] if available_files_mh else 'results'}")
                    if filtered_chunks_mh:
                        include_summary_mh = st.checkbox(
                            "Include brief summary",
                            value=False,
                            key="mh_insight_include_summary",
                            help="Add a 1–3 sentence summary at the end of the insight.",
                        )
                        if st.button("Generate AI insight", key="mh_gen_insight"):
                            provider_key = (
                                "openai" if llm_provider == "OpenAI (ChatGPT)"
                                else ("gemini" if llm_provider == "Gemini" else ("groq" if llm_provider == "Groq" else "huggingface"))
                            )
                            with st.spinner("Calling LLM..."):
                                answer, err = get_insight(provider_key, llm_api_key.strip(), query_mh.strip(), filtered_chunks_mh, max_context_chars=30000, include_summary=include_summary_mh)
                            if err:
                                st.error(f"**Error:** {err}")
                            else:
                                st.markdown(answer, unsafe_allow_html=False)
                                st.caption("**Sources used:** " + ", ".join(set(getattr(c, "file_name", "") for c in filtered_chunks_mh)))
            elif llm_provider != "Off":
                st.caption("Set an API key in the sidebar to enable AI insight.")

            # --- Explain image (Multimodal Hybrid RAG): vision LLM for image results ---
            image_items_mh = [row["content"] for row in fused if row["type"] == "image" and row["content"].get("path")]
            # Also include text results that came from image files (so we can explain e.g. when image ranks as text)
            def _find_image_path_mh(fname: str) -> str | None:
                for root, _dirs, files in os.walk(data_folder):
                    if fname in files:
                        return os.path.join(root, fname)
                return None
            for row in fused:
                if row["type"] != "text":
                    continue
                fname = getattr(row["content"], "file_name", "") or ""
                if not fname:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
                    path = _find_image_path_mh(fname)
                    if path and not any(
                        path == im.get("path") and fname == im.get("file_name") for im in image_items_mh
                    ):
                        image_items_mh.append({
                            "file_name": fname,
                            "page": getattr(row["content"], "page_number", None) or 1,
                            "path": path,
                            "is_pdf_page": False,
                        })
            if image_items_mh and llm_provider != "Off" and llm_api_key.strip() and llm_provider in ("OpenAI (ChatGPT)", "Gemini", "Groq"):
                st.markdown("---")
                st.subheader("🖼️ Explain what's in the image")
                st.caption("Use a vision-capable LLM (OpenAI, Gemini, or Groq) to describe or explain the image using your query.")
                choice_mh = st.selectbox(
                    "Which image to explain?",
                    range(len(image_items_mh)),
                    format_func=lambda i: f"{image_items_mh[i].get('file_name', '')} (p.{image_items_mh[i].get('page', 1)})",
                    key="mh_explain_which",
                )
                if st.button("Explain this image", key="btn_mh_explain_image"):
                    item = image_items_mh[choice_mh]
                    path = item.get("path", "")
                    page = item.get("page") if item.get("is_pdf_page") else None
                    provider_key = (
                        "openai" if llm_provider == "OpenAI (ChatGPT)"
                        else ("gemini" if llm_provider == "Gemini" else "groq")
                    )
                    with st.spinner("Vision LLM is describing the image..."):
                        explanation, err = explain_image(provider_key, llm_api_key.strip(), query_mh.strip(), path, page)
                    if err:
                        st.error(f"**Error:** {err}")
                    else:
                        st.markdown(explanation)
            elif image_items_mh and llm_provider != "Off":
                st.caption("Vision explanation requires **OpenAI**, **Gemini**, or **Groq**. Select one in the sidebar.")

    elif rag_mode == "Multimodal RAG":
        # --- Multimodal RAG: separate index (text + images via CLIP), Hybrid index never touched ---
        st.sidebar.caption(f"Multimodal index: **{st.session_state.multimodal_count}** items (text + images).")
        if st.session_state.multimodal_count == 0:
            st.info("Select **Multimodal RAG** and click **Index / Re-index (Multimodal)** in the sidebar to build the CLIP index (text + images). Then search here.")
            return
        query_m = st.text_input(
            "Query",
            placeholder='e.g. "chest pain" or "policy clause"',
            key="query_multimodal",
        )
        if not query_m.strip():
            st.caption("Enter a query to search over text and images (CLIP).")
            return
        with st.spinner("Searching multimodal index..."):
            hits = search_multimodal(query_m, top_k=MULTIMODAL_TOP_K, data_folder=data_folder)
        st.subheader("Multimodal Results (text + images)")
        st.caption("CLIP retrieval over text chunks and images. Hybrid RAG index is unchanged.")
        if not hits:
            st.info("No results. Try different terms or index more documents (Index / Re-index Multimodal).")
        else:
            for i, h in enumerate(hits, start=1):
                if h["type"] == "text":
                    st.markdown(f"**{i}. {h['file_name']}**" + (f" (p.{h['page']})" if h.get("page") else ""))
                    st.caption(f"Similarity: {h['score']:.4f}  ·  Text chunk")
                    st.text(snippet(h.get("text", ""), query=query_m))
                else:
                    st.markdown(f"**{i}. {h['file_name']}**" + (f" (p.{h['page']})" if h.get("page") else ""))
                    st.caption(f"Similarity: {h['score']:.4f}  ·  Image")
                    path = h.get("path", "")
                    if h.get("is_pdf_page") and path:
                        try:
                            import fitz
                            import io as _io
                            from PIL import Image
                            doc = fitz.open(path)
                            page = doc.load_page(int(h.get("page", 1)) - 1)
                            pix = page.get_pixmap(dpi=120, alpha=False)
                            buf = pix.tobytes("png")
                            img = Image.open(_io.BytesIO(buf)).convert("RGB")
                            st.image(img, use_container_width=True)
                            doc.close()
                        except Exception:
                            st.caption(f"Image: {path} (p.{h.get('page')})")
                    elif path and os.path.isfile(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.caption(f"Image: {path}")
                st.divider()

            # --- Explain image with vision LLM (what's in the diagram) ---
            # Include both (1) image hits and (2) text chunks that came from an image file
            def _find_image_path(fname: str) -> str | None:
                for root, _dirs, files in os.walk(data_folder):
                    if fname in files:
                        return os.path.join(root, fname)
                return None

            image_hits = [h for h in hits if h.get("type") == "image" and h.get("path")]
            text_from_images = []
            for h in hits:
                if h.get("type") != "text":
                    continue
                fname = h.get("file_name", "")
                if not fname:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
                    path = _find_image_path(fname)
                    if path:
                        text_from_images.append({
                            "type": "image",
                            "file_name": fname,
                            "page": h.get("page") or 1,
                            "path": path,
                            "is_pdf_page": False,
                        })
            explainable = image_hits + [x for x in text_from_images if not any(
                x["path"] == ih.get("path") and x.get("file_name") == ih.get("file_name") for ih in image_hits
            )]
            if explainable and llm_provider != "Off" and llm_api_key.strip():
                st.markdown("---")
                st.subheader("🖼️ Explain what's in the image")
                st.caption("Use a vision-capable LLM (OpenAI, Gemini, or Groq) to describe or explain the image using your query.")
                if llm_provider not in ("OpenAI (ChatGPT)", "Gemini", "Groq"):
                    st.caption("Vision explanation requires **OpenAI**, **Gemini**, or **Groq**. Select one in the sidebar.")
                else:
                    choice = st.selectbox(
                        "Which image to explain?",
                        range(len(explainable)),
                        format_func=lambda i: f"{explainable[i].get('file_name', '')} (p.{explainable[i].get('page', 1)})",
                        key="multimodal_explain_which",
                    )
                    if st.button("Explain this image", key="btn_explain_image"):
                        h = explainable[choice]
                        path = h.get("path", "")
                        page = h.get("page") if h.get("is_pdf_page") else None
                        provider_key = (
                            "openai" if llm_provider == "OpenAI (ChatGPT)"
                            else ("gemini" if llm_provider == "Gemini" else "groq")
                        )
                        with st.spinner("Vision LLM is describing the image..."):
                            explanation, err = explain_image(provider_key, llm_api_key.strip(), query_m.strip(), path, page)
                        if err:
                            st.error(f"**Error:** {err}")
                        else:
                            st.markdown(explanation)

    elif rag_mode == "Medical Analysis":
        st.header("🩺 Specialized Medical Analysis Hub")
        st.markdown("Perform clinical-grade analysis on medical images with **Temporal Comparison** to historical patient records.")

        # Medical reports storage (patient-specific, only used on this page)
        medical_reports_base = os.path.join(data_folder, "medical_reports")
        os.makedirs(medical_reports_base, exist_ok=True)

        def _sanitize_patient(name):
            return re.sub(r"[^\w\-]", "_", (name or "").strip()) or "unknown"

        def _get_patients_with_medical_reports():
            """Patients from index + patients who have medical_reports folders."""
            all_chunks = st.session_state.search_index.chunks if (st.session_state.search_index and hasattr(st.session_state.search_index, "chunks")) else []
            catalog = get_robust_catalog(all_chunks)
            from_index = set(catalog.get("known_patients", []))
            if os.path.isdir(medical_reports_base):
                for d in os.listdir(medical_reports_base):
                    pdir = os.path.join(medical_reports_base, d)
                    if os.path.isdir(pdir):
                        from_index.add(d.replace("_", " "))
            return sorted(from_index)

        # --- UPLOAD: Up to 5 docs → Select patient → AI classifies → Validate (✓/✗) → Save ---
        st.markdown("---")
        st.subheader("📤 Upload Medical Reports (up to 5)")
        with st.expander("Upload → AI classifies → You validate (✓ correct / ✗ wrong) → Save", expanded=False):
            patients_for_upload = _get_patients_with_medical_reports()
            patient_for_upload = st.selectbox(
                "Select patient (or None to enter new)",
                ["None", "-- Add new patient --"] + (patients_for_upload or []),
                key="ma_patient_upload",
            )
            if patient_for_upload == "-- Add new patient --":
                new_patient = st.text_input("Enter new patient name", key="ma_new_patient")
                patient_for_upload = new_patient.strip() if new_patient else "None"
            elif patient_for_upload == "None":
                new_patient = st.text_input("Patient is None — write patient name here", key="ma_none_patient")
                if new_patient and new_patient.strip():
                    patient_for_upload = new_patient.strip()

            uploaded_files = st.file_uploader(
                "Browse files (up to 5: X-ray, MRI, PDF, etc.)",
                type=["jpg", "jpeg", "png", "tiff", "tif", "bmp", "pdf"],
                accept_multiple_files=True,
                key="ma_file_upload",
            )
            if len(uploaded_files) > 5:
                uploaded_files = uploaded_files[:5]
                st.caption("Only first 5 files will be used.")

            if "ma_classifications" not in st.session_state:
                st.session_state.ma_classifications = {}

            if uploaded_files and patient_for_upload and patient_for_upload not in ("None", "-- Add new patient --"):
                if st.button("🤖 Classify with AI", key="ma_classify_btn"):
                    import tempfile
                    provider_key = (
                        "openai" if llm_provider == "OpenAI (ChatGPT)"
                        else "gemini" if llm_provider == "Gemini"
                        else "groq" if llm_provider == "Groq"
                        else "mistral" if llm_provider == "Mistral"
                        else "groq"
                    )
                    api_key_val = (
                        llm_api_key.strip()
                        or (get_mistral_ocr_key() if llm_provider == "Mistral" else "")
                        or os.environ.get("MISTRAL_API_KEY")
                        or os.environ.get("GROQ_API_KEY")
                        or os.environ.get("GEMINI_API_KEY")
                        or os.environ.get("OPENAI_API_KEY", "")
                    )
                    if not api_key_val:
                        st.error("Set an API key (OpenAI, Gemini, Groq, or Mistral) in the sidebar to use AI classification.")
                    else:
                        st.session_state.ma_classifications = {}
                        with st.spinner("AI is classifying each document..."):
                            for uf in uploaded_files:
                                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uf.name)[1], delete=False) as tmp:
                                    tmp.write(uf.getvalue())
                                    tmp_path = tmp.name
                                try:
                                    cat, err = classify_medical_document(provider_key, api_key_val, tmp_path, page=1)
                                    st.session_state.ma_classifications[uf.name] = {"ai_says": cat or "Other", "user_ok": None, "corrected": None}
                                except Exception as e:
                                    st.session_state.ma_classifications[uf.name] = {"ai_says": "Other", "user_ok": None, "corrected": None, "err": str(e)}
                                finally:
                                    try:
                                        os.unlink(tmp_path)
                                    except Exception:
                                        pass
                        st.rerun()

                if st.session_state.ma_classifications:
                    st.markdown("**Is this classification correct?** Tick ✓ or ✗ and fix if wrong.")
                    type_to_folder = {"X-ray": "xray", "MRI": "mri", "CT Scan": "ct_scan", "Medical Report": "medical_report", "Ultrasound": "ultrasound", "Other": "other"}
                    all_types = ["X-ray", "MRI", "CT Scan", "Medical Report", "Ultrasound", "Other"]
                    for fname, data in list(st.session_state.ma_classifications.items()):
                        ai_cat = data.get("ai_says", "Other")
                        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
                        with c1:
                            st.caption(f"**{fname}** → AI says: **{ai_cat}**")
                        with c2:
                            if st.button("✓ Correct", key=f"ma_ok_{fname}"):
                                st.session_state.ma_classifications[fname]["user_ok"] = True
                                st.session_state.ma_classifications[fname]["corrected"] = None
                                st.rerun()
                        with c3:
                            if st.button("✗ Wrong", key=f"ma_wrong_{fname}"):
                                st.session_state.ma_classifications[fname]["user_ok"] = False
                                st.rerun()
                        if data.get("user_ok") is False:
                            with c4:
                                corr = st.selectbox("Correct type", all_types, key=f"ma_corr_{fname}")
                                if st.button("Apply", key=f"ma_apply_{fname}"):
                                    st.session_state.ma_classifications[fname]["corrected"] = corr
                                    st.session_state.ma_classifications[fname]["user_ok"] = True
                                    st.rerun()
                        elif data.get("user_ok") is True:
                            with c4:
                                final = data.get("corrected") or ai_cat
                                st.success(f"✓ {final}")

                    if st.button("💾 Save all to patient folder", key="ma_save_all_btn"):
                        patient_sanitized = _sanitize_patient(patient_for_upload)
                        saved = 0
                        for uf in uploaded_files:
                            data = st.session_state.ma_classifications.get(uf.name, {})
                            final_type = data.get("corrected") or data.get("ai_says", "Other")
                            folder = type_to_folder.get(final_type, "other")
                            dest_dir = os.path.join(medical_reports_base, patient_sanitized, folder)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_path = os.path.join(dest_dir, uf.name)
                            with open(dest_path, "wb") as f:
                                f.write(uf.getvalue())
                            saved += 1
                        st.success(f"Saved {saved} file(s) to {patient_for_upload}.")
                        st.session_state.ma_classifications = {}
                        st.rerun()

        # --- SELECT & ANALYZE ---
        st.markdown("---")
        st.subheader("Select & Analyze")
        all_chunks = st.session_state.search_index.chunks if (st.session_state.search_index and hasattr(st.session_state.search_index, "chunks")) else []
        catalog = get_robust_catalog(all_chunks)
        known_patients = sorted(catalog.get("known_patients", []))
        patients_with_reports = _get_patients_with_medical_reports()
        patient_options = ["None"] + (patients_with_reports or known_patients)

        patient_choice = st.selectbox("Select Patient", patient_options, key="ma_patient_choice")

        # Filter by report type (not all JPGs)
        report_type_filter = st.selectbox(
            "Filter by report type",
            ["All", "X-ray", "MRI", "CT Scan", "Medical Report", "Ultrasound", "Other"],
            key="ma_report_filter",
            help="Show only this type of medical report for the selected patient.",
        )

        images_to_analyze = []

        def _get_patient_medical_files(patient, rtype_filter):
            """Get files from medical_reports storage for this patient + type."""
            if not patient or patient == "None":
                return []
            patient_sanitized = _sanitize_patient(patient)
            base = os.path.join(medical_reports_base, patient_sanitized)
            if not os.path.isdir(base):
                return []
            type_map = {"X-ray": "xray", "MRI": "mri", "CT Scan": "ct_scan", "Medical Report": "medical_report", "Ultrasound": "ultrasound", "Other": "other"}
            folders = [type_map[rtype_filter]] if rtype_filter != "All" else list(type_map.values())
            files = []
            for folder in folders:
                fdir = os.path.join(base, folder)
                if os.path.isdir(fdir):
                    for f in os.listdir(fdir):
                        fp = os.path.join(fdir, f)
                        if os.path.isfile(fp):
                            ext = os.path.splitext(f)[1].lower()
                            if ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf"):
                                files.append({"path": fp, "file_name": f, "page": 1, "is_pdf_page": ext == ".pdf"})
            return files

        def select_image(label_suffix=""):
            selected_img = None
            if patient_choice != "None":
                # Prefer patient's medical_reports (filtered by type)
                local_files = _get_patient_medical_files(patient_choice, report_type_filter)
                if local_files:
                    st.subheader(f"Medical reports for {patient_choice} ({report_type_filter}) {label_suffix}")
                    opts = [f"{x['file_name']}" for x in local_files]
                    idx = st.selectbox(f"Choose report {label_suffix}", range(len(opts)), format_func=lambda i: opts[i], key=f"ma_local_{label_suffix}")
                    selected_img = local_files[idx]
                else:
                    # Fallback: indexed images (ImageRetriever) from ChromaDB
                    st.subheader(f"Imaging History for {patient_choice} ({report_type_filter}) {label_suffix}")
                    ir = ImageRetriever()
                    aliases_ma = catalog.get("patient_name_aliases", {})
                    mf = {"patient_name": aliases_ma.get(patient_choice, [patient_choice])}
                    if report_type_filter != "All":
                        # "Other" includes unclassified docs (report_type="") from uploads/
                        mf["report_type"] = [report_type_filter, ""] if report_type_filter == "Other" else report_type_filter
                    history_hits = ir.retrieve("", top_n=50, metadata_filter=mf)
                    # Fallback: if no hits by metadata, search by patient name in OCR text
                    # (JPGs in uploads/ may have patient_name in OCR but not in metadata if extraction failed)
                    if not history_hits:
                        fallback_hits = ir.retrieve(patient_choice, top_n=100, metadata_filter=None)
                        patient_lower = patient_choice.lower()
                        def _patient_match(item):
                            return patient_lower in (item.get("patient_name") or "").lower() or patient_lower in (item.get("ocr_text") or "").lower()
                        def _report_type_ok(item):
                            if report_type_filter == "All":
                                return True
                            rt = (item.get("report_type") or "").strip()
                            if report_type_filter == "Other":
                                return rt in ("", "Other")
                            return rt == report_type_filter
                        history_hits = [
                            (item, score) for item, score in fallback_hits
                            if _patient_match(item) and _report_type_ok(item)
                        ][:50]
                    if not history_hits:
                        st.info(f"No medical reports found for {patient_choice}. Upload files above or index documents.")
                    else:
                        img_options = [f"{h[0]['file_name']} (p.{h[0].get('page', 1)})" for h in history_hits]
                        selected_img_idx = st.selectbox(f"Choose image {label_suffix}", range(len(img_options)), format_func=lambda i: img_options[i], key=f"img_sel_{label_suffix}")
                        selected_img = history_hits[selected_img_idx][0]
            else:
                search_q = st.text_input(f"Search for image {label_suffix}", placeholder="e.g. 'skull x-ray'", key=f"search_{label_suffix}")
                if search_q:
                    ir = ImageRetriever()
                    hits = ir.retrieve(search_q, top_n=10)
                    if hits:
                        img_options = [f"{h[0]['file_name']} (p.{h[0].get('page', 1)})" for h in hits]
                        selected_img_idx = st.selectbox(f"Choose image {label_suffix} from search", range(len(img_options)), format_func=lambda i: img_options[i], key=f"hit_sel_{label_suffix}")
                        selected_img = hits[selected_img_idx][0]
            return selected_img

        st.markdown("### Select Primary Image (Image A)")
        img_a = select_image("A")
        if img_a:
            images_to_analyze.append(img_a)

        compare_mode = st.checkbox("➕ Compare with a second image (Image B)?", key="ma_compare")
        img_b = None
        if compare_mode:
            st.markdown("---")
            st.markdown("### Select Comparative Image (Image B)")
            img_b = select_image("B")
            if img_b:
                images_to_analyze.append(img_b)

        # Show selected images side-by-side
        if images_to_analyze:
            cols = st.columns(len(images_to_analyze))
            for i, img in enumerate(images_to_analyze):
                with cols[i]:
                    path = img.get("path")
                    label = "Image A" if i == 0 else "Image B"
                    if path and os.path.exists(path):
                        if path.lower().endswith(".pdf"):
                            try:
                                import fitz
                                import io as _io
                                from PIL import Image as _PILImage
                                doc = fitz.open(path)
                                page = doc.load_page(int(img.get("page", 1)) - 1)
                                pix = page.get_pixmap(dpi=120, alpha=False)
                                buf = pix.tobytes("png")
                                pil_img = _PILImage.open(_io.BytesIO(buf)).convert("RGB")
                                st.image(pil_img, caption=f"{label}: {img.get('file_name', '')}", use_container_width=True)
                                doc.close()
                            except Exception:
                                st.caption(f"{label}: PDF (page {img.get('page', 1)})")
                        else:
                            st.image(path, caption=f"{label}: {img.get('file_name', '')}", use_container_width=True)
        
        # 3. Clinical Query & Analyze
        if images_to_analyze:
            default_q = "Compare these two images and describe changes." if len(images_to_analyze) > 1 else "Describe this image and compare to history if available."
            clinical_query = st.text_input("Clinical Query / Questions", value=default_q, placeholder="e.g. 'Is there any change in bone alignment?'")
            if st.button("🚀 Analyze & Compare", type="primary"):
                # Fetch historical text context for temporal comparison
                history_context = ""
                if patient_choice != "None":
                    # Get text chunks for this patient (include OCR variants)
                    tr = TextRetriever(st.session_state.search_index)
                    aliases_ma = catalog.get("patient_name_aliases", {})
                    mf_ma = {"patient_name": aliases_ma.get(patient_choice, [patient_choice])}
                    text_hits = tr.retrieve(clinical_query, top_k=20, metadata_filter=mf_ma)
                    history_context = build_context([h[0] for h in text_hits])
                
                provider_key = (
                    "openai" if llm_provider == "OpenAI (ChatGPT)"
                    else "gemini" if llm_provider == "Gemini"
                    else "groq" if llm_provider == "Groq"
                    else "mistral" if llm_provider == "Mistral"
                    else "groq"
                )
                api_key_for_analysis = (
                    llm_api_key.strip()
                    or (get_mistral_ocr_key() if llm_provider == "Mistral" else "")
                    or os.environ.get("MISTRAL_API_KEY")
                    or os.environ.get("GROQ_API_KEY")
                    or os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("OPENAI_API_KEY", "")
                )
                with st.spinner("Specialized Medical Agent is reviewing..."):
                    report, err = medical_analysis(
                        provider=provider_key,
                        api_key=api_key_for_analysis,
                        query=clinical_query,
                        image_paths=[img["path"] for img in images_to_analyze],
                        pages=[img.get("page") for img in images_to_analyze],
                        history_context=history_context
                    )
                
                if err:
                    st.error(f"Analysis failed: {err}")
                else:
                    st.markdown("---")
                    st.success("Analysis Complete")
                    st.markdown(report)

if __name__ == "__main__":
    main()
