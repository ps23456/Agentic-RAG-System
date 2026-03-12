# Agentic Multimodal Hybrid RAG – System Flow

## Complete Query-to-Results Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER ENTERS QUERY                                   │
│                 "disability restrictions for Rika Popper"                     │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        1. LLM AGENT (Brain)                                  │
│                     Groq / Gemini / OpenAI                                    │
│                                                                              │
│  Receives: query + catalog of entities + sample document content             │
│                                                                              │
│  Outputs:                                                                    │
│    • intent: "general_search"                                                │
│    • search_queries: [                                                        │
│        "disability restrictions Rika Popper physical capacities",            │
│        "activity restrictions limitations stand walk sit Rika Popper"        │
│      ]                                                                       │
│    • patient_filter: "Rika Popper"                                           │
│    • direct_answer: null                                                     │
│                                                                              │
│  (If no LLM key → rule-based fallback with domain synonym expansion)         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               │  For EACH search query generated:
                               │
              ┌────────────────┴────────────────┐
              │                                  │
              ▼                                  ▼
┌─────────────────────────────┐    ┌─────────────────────────────────┐
│     TEXT RETRIEVAL PATH      │    │      IMAGE RETRIEVAL PATH        │
│                              │    │                                  │
│                              │    │                                  │
│   ┌──────────────────────┐   │    │   ┌──────────────────────────┐   │
│   │  Query (raw text)    │   │    │   │  Query (raw text)        │   │
│   └──────────┬───────────┘   │    │   └──────────┬───────────────┘   │
│              │               │    │              │                   │
│     ┌────────┴────────┐      │    │              ▼                   │
│     │                 │      │    │   ┌──────────────────────────┐   │
│     ▼                 ▼      │    │   │  CLIP Text Encoder       │   │
│  ┌───────┐    ┌────────────┐ │    │   │  (openai/clip-vit-base)  │   │
│  │ BM25  │    │  Embedding │ │    │   │  query → 512-dim vector  │   │
│  │Search │    │   Model    │ │    │   └──────────┬───────────────┘   │
│  │       │    │(MiniLM-L6) │ │    │              │                   │
│  │keyword│    │query → 384 │ │    │              ▼                   │
│  │match  │    │dim vector  │ │    │   ┌──────────────────────────┐   │
│  └───┬───┘    └─────┬──────┘ │    │   │  ChromaDB                │   │
│      │              │        │    │   │  (image_collection)      │   │
│      │              ▼        │    │   │  cosine similarity       │   │
│      │     ┌────────────────┐│    │   │  25 images (PDF pages    │   │
│      │     │   ChromaDB     ││    │   │  + standalone images)    │   │
│      │     │(claim_chunks)  ││    │   └──────────┬───────────────┘   │
│      │     │cosine search   ││    │              │                   │
│      │     │54 text chunks  ││    │              ▼                   │
│      │     └───────┬────────┘│    │   ┌──────────────────────────┐   │
│      │             │         │    │   │  Metadata Filtering      │   │
│      │    ┌────────┘         │    │   │  patient_name="Rika..."  │   │
│      │    │                  │    │   └──────────┬───────────────┘   │
│      │    ▼                  │    │              │                   │
│      │  ┌──────────────────┐ │    │              ▼                   │
│      │  │Metadata Filtering│ │    │   ┌──────────────────────────┐   │
│      │  │patient_name=     │ │    │   │  SigLIP Image Reranker   │   │
│      │  │"Rika Popper"     │ │    │   │  (google/siglip-base)    │   │
│      │  └────────┬─────────┘ │    │   │  re-scores (query,image) │   │
│      │           │           │    │   │  pairs for relevance     │   │
│      ▼           ▼           │    │   └──────────┬───────────────┘   │
│  ┌──────────────────────┐    │    │              │                   │
│  │    RRF Fusion         │    │    │              ▼                   │
│  │ (Reciprocal Rank      │    │    │     Ranked image results        │
│  │  Fusion)              │    │    │     with reranker scores         │
│  │ Merges BM25 + Vector  │    │    │                                  │
│  │ rankings              │    │    └─────────────┬───────────────────┘
│  └──────────┬───────────┘    │                   │
│             │                │                   │
│             ▼                │                   │
│  ┌──────────────────────┐    │                   │
│  │   BGE Reranker        │    │                   │
│  │ (BAAI/bge-reranker)   │    │                   │
│  │ CrossEncoder rescores │    │                   │
│  │ (query, chunk) pairs  │    │                   │
│  └──────────┬───────────┘    │                   │
│             │                │                   │
│             ▼                │                   │
│     Ranked text results      │                   │
│     with reranker scores     │                   │
│                              │                   │
└──────────────┬───────────────┘                   │
               │                                   │
               │    Merge across all search queries │
               │    (keep highest score per chunk)  │
               │                                   │
               └───────────────┬───────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      2. QUERY TYPE AUTO-DETECTION                            │
│                                                                              │
│  classify_query("disability restrictions for Rika Popper")                   │
│    → "text_heavy"  (no image keywords found)                                 │
│    → weights: text = 0.8, image = 0.2                                        │
│                                                                              │
│  classify_query("diagram with boxes and arrows")                             │
│    → "image_heavy" (found: diagram, boxes, arrows)                           │
│    → weights: text = 0.3, image = 0.7                                        │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      3. SCORE FUSION                                         │
│                                                                              │
│  Text scores:  [0.95, 0.72, 0.41, ...]  → min-max normalize → [1.0, 0.57..] │
│  Image scores: [0.003, 0.001, ...]      → min-max normalize → [1.0, 0.33..] │
│                                                                              │
│  final_score = w_text × norm_text  +  w_image × norm_image                   │
│                                                                              │
│  Example (text_heavy):                                                       │
│    Text chunk:  0.8 × 1.0  + 0.2 × 0.0 = 0.80                               │
│    Image item:  0.8 × 0.0  + 0.2 × 1.0 = 0.20                               │
│                                                                              │
│  Sort all results by final_score descending                                  │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      4. DIVERSITY & MINIMUM GUARANTEE                        │
│                                                                              │
│  If METADATA_DIVERSITY_ENABLED:                                              │
│    → Max 2 results per patient (ensures coverage across patients)            │
│                                                                              │
│  If results < 3:                                                             │
│    → Retry with original query + no metadata filter (broader search)         │
│    → Append until at least 3 results                                         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      5. TOP-K RESULTS RETURNED                               │
│                                                                              │
│  1. [0.800] markdown 4.md        │ Patient: Rika Popper  │ Text              │
│  2. [0.504] markdown 4.txt       │ Patient: Rika Popper  │ Text              │
│  3. [0.200] real estate app.jpg  │                        │ Image             │
│  ...                                                                         │
│                                                                              │
│  + Direct answer (if list query): "Patients: Alyson Jude, Rika Popper, ..."  │
│  + AI Insight (optional): LLM summarizes from retrieved chunks               │
│  + Image Explanation (optional): Vision LLM describes image content          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Index Building Flow

```
┌─────────────────┐     ┌──────────────────────────────────────────────────────┐
│  Data Folder     │     │  TEXT INDEX ("Index / Re-index documents")            │
│                  │     │                                                      │
│  PDFs            │────▶│  load_and_chunk_folder()                             │
│  Markdown (.md)  │     │    → Read files (pdfplumber / OCR / raw)             │
│  Text (.txt)     │     │    → Extract full document text                      │
│  Images (.jpg)   │     │    → extract_chunk_metadata() → patient, claim, etc. │
│  JSON (.json)    │     │    → chunk_text() → ~500-char chunks                 │
│                  │     │    → Each chunk inherits document metadata            │
│                  │     │                                                      │
│                  │     │  SearchIndex(54 chunks)                              │
│                  │     │    → BM25 index (keyword)                            │
│                  │     │    → SentenceTransformer embeddings → ChromaDB       │
│                  │     │      (claim_chunks collection, cosine similarity)    │
│                  │     └──────────────────────────────────────────────────────┘
│                  │
│                  │     ┌──────────────────────────────────────────────────────┐
│                  │     │  IMAGE INDEX ("Index / Re-index Image")               │
│                  │────▶│                                                      │
│                  │     │  For each image / PDF page:                          │
│                  │     │    → Render to pixels                                │
│                  │     │    → CLIP image encoder → 512-dim embedding          │
│                  │     │    → Store in ChromaDB (image_collection)            │
│                  │     │    → 25 items total                                  │
│                  │     └──────────────────────────────────────────────────────┘
└─────────────────┘
```

---

## Models Used

| Model | Purpose | What It Does |
|-------|---------|-------------|
| `all-MiniLM-L6-v2` | Text embeddings | Query/chunk → 384-dim vector for semantic search |
| `BAAI/bge-reranker-base` | Text reranking | CrossEncoder rescores (query, chunk) pairs |
| `openai/clip-vit-base-patch32` | Image embeddings | Query text → 512-dim vector to search images |
| `google/siglip-base-patch16-224` | Image reranking | Rescores (query, image) pairs for relevance |
| `llama-3.3-70b-versatile` (Groq) | LLM brain | Query understanding, multi-query generation |

## Run

```bash
cd "/Users/pshah/testing 3"
.venv_bge/bin/python3 -m streamlit run app.py
```
