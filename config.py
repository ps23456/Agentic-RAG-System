"""Configuration for Insurance Claim Search."""
import os

# Data source: local folder only
DATA_FOLDER = os.environ.get("CLAIM_SEARCH_DATA", os.path.join(os.path.dirname(__file__), "data"))
# Subfolder under data folder where uploaded files are stored
UPLOADS_SUBDIR = "uploads"

# Supported extensions
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
# Markdown (e.g. from Docling conversion) – indexed as plain text
MARKDOWN_EXTENSIONS = {".md"}
# JSON (e.g. Docling structured export) – treated as plain text for search/LLM
JSON_EXTENSIONS = {".json"}
# Plain text files (e.g. formatted output from OCR conversion scripts) – indexed as plain text
TEXT_EXTENSIONS = {".txt"}

# Document type detection by filename prefix (case-insensitive)
DOC_TYPE_POLICY = "policy"
DOC_TYPE_CLAIM = "claim"
DOC_TYPE_MEDICAL = "medical"
DOC_TYPE_UNKNOWN = "unknown"

def get_doc_type(filename: str) -> str:
    base = os.path.basename(filename).lower()
    if base.startswith("policy") or "policy" in base.split("_")[0]:
        return DOC_TYPE_POLICY
    if base.startswith("claim") or "claim" in base.split("_")[0] or "clm-" in base:
        return DOC_TYPE_CLAIM
    if base.startswith("medical") or "medical" in base or "report" in base:
        return DOC_TYPE_MEDICAL
    return DOC_TYPE_UNKNOWN

# Mistral OCR: comma-separated PDF basenames to always OCR via Mistral (e.g. handwritten forms with embedded template text).
# Example: MISTRAL_OCR_FORCE_FILENAMES="TEE_TBrown (1).pdf,scan.pdf"
MISTRAL_OCR_FORCE_FILENAMES = os.environ.get("MISTRAL_OCR_FORCE_FILENAMES", "")

# Chunking
CHUNK_BY = "paragraph"  # or "page"
MAX_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 50

# Structured document chunking (auto-detected: text-rich PDFs with headings, e.g. annual reports)
STRUCTURED_DOC_MIN_PAGES = 20        # only apply section chunking for PDFs with this many pages
STRUCTURED_DOC_TEXT_RATIO = 0.80     # 80%+ pages must have extractable text to qualify
STRUCTURED_MAX_CHUNK_CHARS = 50000   # large section-based chunks
STRUCTURED_SKIP_RERANKER = True      # skip BGE CrossEncoder reranker for structured-doc results

# Search
# Larger BM25/vector top-K gives RRF more candidates so reranker can surface the best (e.g. chunk with doctor name).
BM25_TOP_K = 40
VECTOR_TOP_K = 40
HYBRID_TOP_K = 15
RRF_K = 60  # constant for Reciprocal Rank Fusion
# Minimum score to show (avoid weak/dummy matches; 0.18 allows OCR form text to appear)
VECTOR_MIN_SIMILARITY = 0.18
HYBRID_MIN_FUSION_SCORE = 0.02  # RRF; filters very weak fusion results

# Metadata filtering: auto-extract from query for better accuracy at scale (100+ docs)
AUTO_METADATA_FILTER = True  # Infer patient/claim/policy from query (e.g. "Rika Popper's claim")
PREFER_USER_METADATA_FILTER = True  # Sidebar selection overrides auto-extracted when both present

# Metadata diversity: prevent same-patient dominance; improves ALL queries
METADATA_DIVERSITY_ENABLED = True  # Apply diversity by patient_name so results cover multiple patients
METADATA_DIVERSITY_MAX_PER_ENTITY = 2  # Max chunks per patient in top results (1 for list queries)
METADATA_DIVERSITY_CANDIDATES = 60  # Retrieve this many before diversity (need pool for round-robin)

# LLM query understanding: parse any query for intent, filters, expanded search (dynamic, accurate)
LLM_QUERY_UNDERSTANDING = True  # Use LLM to understand query (requires Groq/Gemini/OpenAI key)

# Embedding model for vector (semantic) search. BGE M3 (multilingual, better retrieval). See docs/BGE_M3_SETUP.md.
# Temporarily using MiniLM for faster indexing - switch back to "BAAI/bge-m3" when needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector backend: "chroma" (persistent) or "faiss" (in-memory; FAISS code commented out for now).
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "chroma")
# Chroma: directory to persist the vector DB (used when VECTOR_BACKEND=chroma).
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "data", "chroma"))
CHROMA_COLLECTION_NAME = "claim_chunks"

# BGE Reranker: second-stage reranking of hybrid results (query + passage -> score).
# Uses sentence-transformers CrossEncoder; first load may download the model.
RERANKER_MODEL = "BAAI/bge-reranker-base"
# How many candidates to pass to the reranker. Tuned down from 80 -> 30 because:
#   (a) BM25+vector RRF upstream already filters weak matches,
#   (b) BGE rerank quality on 25-30 candidates is statistically ~equal to 80,
#   (c) CPU rerank time scales linearly with candidate count — 30 is ~2.5x faster.
# Raise back to 60-80 only if you observe the right chunk consistently missing.
RERANKER_CANDIDATES = 30
# How many to return after reranking.
RERANKER_TOP_K = 15
# Max chars per passage sent to reranker (avoids token limit).
RERANKER_MAX_PASSAGE_CHARS = 512
# Below this total candidate count, skip the cross-encoder entirely — BM25+vector
# RRF ordering is already a strong signal when the candidate pool is small
# (e.g. patient-scoped queries). Saves ~5-10s on small/scoped queries.
RERANKER_MIN_CANDIDATES = 8

# --- Multimodal RAG (separate from Hybrid; uses its own Chroma collection) ---
CHROMA_MULTIMODAL_COLLECTION_NAME = "multimodal_only"
# Larger model = better image-text alignment (e.g. "diagram with boxes" -> diagram image). Re-index after changing.
MULTIMODAL_CLIP_MODEL = "openai/clip-vit-large-patch14"
MULTIMODAL_TOP_K = 15
# Retrieve this many candidates then re-rank (so image boost can surface diagrams)
MULTIMODAL_RETRIEVE_CANDIDATES = 50

# --- Multimodal Hybrid RAG (production): separate text + image collections, score fusion ---
CHROMA_IMAGE_COLLECTION_NAME = "image_collection"
# Query-type fusion weights (text_weight, image_weight). Do NOT mix embedding spaces.
MULTIMODAL_HYBRID_WEIGHTS = {
    "text_heavy": (0.92, 0.08),   # Strongly favor text (Markdown, PDF text) for queries like "primary diagnosis"
    "image_heavy": (0.3, 0.7),
    "hybrid": (0.5, 0.5),
}
# Rule-based query classifier: if query contains any of these -> image_heavy (modular for future LLM classifier).
QUERY_CLASSIFIER_IMAGE_KEYWORDS = (
    "diagram", "image", "chart", "arrow", "arrows", "box", "boxes", "screenshot",
    "flow", "architecture", "picture", "photo", "figure", "visual", "drawing",
    "schema", "layout", "graph", "illustration",
)
MULTIMODAL_HYBRID_TOP_K = 15
# Retrieve more text candidates so reranker can pick the most relevant (e.g. chunk that contains the answer).
# Tuned down in step with RERANKER_CANDIDATES — a larger value here is wasted when
# the reranker only sees 30 anyway (the pipeline takes the min of the two).
MULTIMODAL_HYBRID_TEXT_CANDIDATES = 40
# Retrieve more images so relevant form/diagram images can appear in fused top-K.
MULTIMODAL_HYBRID_IMAGE_TOP_N = 30
# Optional image reranker: second-stage rerank of CLIP results with SigLIP (query–image relevance). Off by default.
USE_IMAGE_RERANKER = False
IMAGE_RERANKER_MODEL = "google/siglip-base-patch16-224"
# How many CLIP results to pass to the image reranker; then return top MULTIMODAL_HYBRID_IMAGE_TOP_N.
IMAGE_RERANKER_CANDIDATES = 50

# --- PageIndex-style tree indexing (vectorless, no chunking) ---
# Where to store tree JSON files (one per PDF)
PAGE_TREE_CACHE_DIR = os.path.join(DATA_FOLDER, "cache", "trees")
# Pages to check for Table of Contents (PageIndex: TOC is usually in first 20-50 pages)
PAGE_TREE_TOC_PAGES = 50
# Max pages per LLM chunk when chunked processing (no TOC)
PAGE_TREE_MAX_PAGES_PER_LLM = 20
# Max pages per leaf node (PageIndex uses 10). Larger sections are split into sub-sections.
PAGE_TREE_MAX_PAGES_PER_NODE = 10
