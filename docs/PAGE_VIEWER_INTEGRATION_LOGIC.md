# Page Viewer & PageIndex-Style Integration Logic

**Goal:** Add "click to open exact page" and optional page-tree retrieval **without affecting** the existing Agentic RAG, Multimodal Hybrid RAG, Vision, Mistral OCR, reranker, or query understanding.

---

## Current Stack (Unchanged)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT MULTIMODAL HYBRID RAG STACK                             │
│                                    (DO NOT MODIFY)                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INDEXING (document_loader.py, indexing.text_indexer.py, image_indexer.py)          │
│    • load_and_chunk_folder() → chunk by paragraph/section                               │
│    • Mistral OCR (pdf/images) → high-accuracy text                                       │
│    • Vision LLM captioning (image_indexer) → form sections, diagrams                     │
│    • Chunk: chunk_id, text, file_name, page_number, document_type, patient_name, etc.      │
│    • BM25 corpus + ChromaDB (claim_chunks) + ChromaDB (image_collection)                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. QUERY UNDERSTANDING (Agentic RAG)                                                     │
│    • LLM analyzes query → intent, search_queries, main_intent_keywords, patient_filter   │
│    • Rule-based fallback when no LLM key                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. RETRIEVAL (multimodal hybrid)                                                         │
│    • Text: BM25 + Dense (Chromadb) → RRF fusion → BGE Reranker                           │
│    • Verbatim search (exact + soft match) + main_intent_keywords → phrase boost           │
│    • Image: CLIP + SigLIP reranker (optional)                                             │
│    • Query type: text_heavy / image_heavy / hybrid → weights                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. FUSION (hybrid_fusion.py)                                                             │
│    • RRF: text + image by (file_name, page_number)                                        │
│    • boost_phrase_matching (tiered by intent)                                             │
│    • diversify_fused_results (by patient)                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. DISPLAY (app.py)                                                                      │
│    • Fused results: file_name, page_number, snippet, score                               │
│    • find_pdf_page_containing_phrases() for better context when chunk lacks it            │
│    • AI Insight (LLM), Explain image (Vision LLM)                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Principle: Additive Only

| Component | Role | Change for Page Viewer |
|-----------|------|------------------------|
| **Agentic RAG** | Query understanding, multi-query, intent | **None** |
| **Multimodal Hybrid** | Text + image retrieval, fusion | **None** |
| **Vision / Mistral OCR** | Indexing, image captioning | **None** |
| **Reranker (BGE)** | Rescore text chunks | **None** |
| **Chunk metadata** | file_name, page_number, text | **Already has page_number** |

**Key insight:** Chunks already carry `file_name` and `page_number`. The page viewer is a **display-layer enhancement** that consumes those fields. No retrieval logic changes.

---

## Where the New Feature Plugs In

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           ADDITIVE LAYER (NEW)                                           │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  EXISTING: Fused results → display (file_name, page_number, snippet)            │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                                 │
│                                        │  Same data                                      │
│                                        ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  NEW: Page Viewer UI                                                             │   │
│   │  • Input: file_name, page_number (from result chunk)                             │   │
│   │  • Resolve: data_folder + file_name → file_path                                  │   │
│   │  • Render: PDF page as image (PyMuPDF) or Markdown section                       │   │
│   │  • Trigger: user clicks "View source" or result citation                         │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   No changes to: retrieval, fusion, agentic, indexing, OCR, vision                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Logic Flow: Current vs. With Page Viewer

### Current (unchanged)

```
User Query
    → Agentic RAG (LLM plan)
    → Multi-query retrieval (BM25 + dense + verbatim + main_intent_keywords)
    → Phrase boost
    → Fusion (text + image)
    → Diversify
    → Display: "1. pnb.md (p.426) · Fused score: 0.92 · Text"
    → Snippet
```

### With Page Viewer (additive)

```
User Query
    → [same flow as above]
    → Display: "1. pnb.md (p.426) · Fused score: 0.92 · Text"
    → Snippet
    → + "View source" button / expander
        → On click: resolve_path(file_name) → render_pdf_page(path, page_number)
        → Show PDF page image or Markdown section in viewer
```

---

## Implementation Options

### Option A: Same Page, Additive UI (Recommended)

**Location:** Multimodal Hybrid RAG results section in `app.py`

**Changes:**
- Add `_resolve_path(file_name, data_folder)` helper
- Add `render_pdf_page(path, page)` using PyMuPDF
- For each result: add expander "View source" → when expanded, render page

**Impact:** None on existing retrieval, indexing, or agentic logic.

**Files to touch:**
- `app.py` only (add UI, helper, optional viewer column)

---

### Option B: New Page / Tab

**Location:** New Streamlit page or tab (e.g. "Document Viewer" or "Source View")

**Flow:**
- User searches in Multimodal Hybrid RAG (unchanged)
- Results show `file_name` and `page_number`. Each result has a link/button.
- On click: navigate to new page or set `st.session_state.selected_citation = {file, page}`
- New page: reads `selected_citation`, renders PDF viewer

**Impact:** None on existing retrieval. Only adds routing and a new page.

---

### Option C: Right Panel (PageIndex-Style Layout)

**Location:** Same page, layout change: `st.columns([2, 1])` – left results, right viewer

**Flow:**
- Left: results (unchanged)
- Right: when user clicks a result, `selected_citation` updates → right panel re-renders with PDF page

**Impact:** Layout change only. Retrieval logic unchanged.

---

## Optional: Page Tree (PageIndex-Style) – Future Enhancement

If you later add **reasoning-based tree retrieval** (like PageIndex):

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  PARALLEL INDEX (optional, not replacing existing)                                        │
│                                                                                          │
│  • Build: page tree JSON per PDF (when indexing)                                          │
│    - Extract pages → LLM infers TOC / sections → node_id, title, start_page, end_page     │
│  • Store: {file_id: tree.json} in data/cache or alongside ChromaDB                       │
│  • Retrieve: LLM reasons over tree (titles, summaries) → node_ids → fetch pages           │
│                                                                                          │
│  • Integration: Run as ADDITIONAL retrieval path when query_type suggests structure      │
│    - Merge tree results with fused results (optional)                                     │
│    - Or: use tree only for "open exact page" when user clicks                             │
│                                                                                          │
│  • Key: Tree is separate from chunk index. Chunks stay for BM25/dense. Tree is optional.  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**When to add:** Only if you want structure-aware retrieval for very long PDFs. Not required for "click to open page".

---

## Summary Table

| Feature | Current | Add Page Viewer | Add Page Tree (future) |
|---------|---------|-----------------|------------------------|
| Agentic RAG | ✅ | ✅ | ✅ |
| Multimodal Hybrid | ✅ | ✅ | ✅ |
| BM25 + Dense + Rerank | ✅ | ✅ | ✅ |
| Vision / Mistral OCR | ✅ | ✅ | ✅ |
| Phrase boost, verbatim | ✅ | ✅ | ✅ |
| Display results | ✅ | ✅ + clickable | ✅ |
| PDF page on click | ❌ | ✅ | ✅ |
| Tree-based retrieval | ❌ | ❌ | Optional |

---

## Implementation Checklist (Minimal)

1. **Add `_resolve_path(file_name, data_folder)`** – walk data folder to find file
2. **Add `render_pdf_page(path, page)`** – PyMuPDF page → image → `st.image()`
3. **In each result row:** add expander `"View source: {file_name} p.{page}"` → when expanded, call `render_pdf_page`
4. **For Markdown:** if `file_name` ends with `.md`, show `st.markdown(section_content)` instead of PDF

**No changes to:** `retrieval/`, `document_loader.py` (chunking), `search_index.py`, `indexing/`, `config.py` (unless adding viewer config).

---

## Data Flow Summary

```
Indexing (unchanged)
    → Chunks with file_name, page_number, text
    → BM25 + ChromaDB (text) + ChromaDB (image)

Query (unchanged)
    → Agentic RAG → search_queries, main_intent_keywords
    → Retrieval + fusion + diversify
    → Fused results: [{type, content, final_score}]

Display (additive)
    → For each result: show file_name, page_number, snippet
    → NEW: "View source" → resolve_path → render_pdf_page
```

**The page viewer is a pure consumer of the existing `file_name` and `page_number` in each result. No upstream changes.**
