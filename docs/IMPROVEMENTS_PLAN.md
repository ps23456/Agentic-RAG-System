# System Improvements Plan - Insurance Claim Search RAG

## Current System Assessment

**✅ What's Working Well:**
- Hybrid retrieval (BM25 + Vector + RRF + reranker)
- Persistent vector storage (Chroma)
- Multiple LLM providers (OpenAI, Gemini, Groq, Hugging Face)
- OCR support for images
- Document selection for AI insight

**⚠️ Areas for Improvement:**
1. **Page numbers** - Paragraph chunking can span pages; page_number only shows starting page
2. **Image previews** - No visual attachment/preview from source documents
3. **Embedding model** - MiniLM is fast but BGE M3 has better retrieval quality
4. **Retrieval accuracy** - Could use query expansion, better reranking

---

## Priority Improvements

### 1. **Accurate Page Numbers** ⭐ HIGH PRIORITY

**Problem:** When chunking by paragraph, a chunk can span multiple pages. Currently `page_number` only stores the starting page.

**Solution:**
- Track `start_page` and `end_page` in Chunk dataclass
- When chunking, detect if text spans pages (by tracking character positions per page)
- Display: "p.3-4" instead of "p.3" when chunk spans pages
- Store page ranges in Chroma metadata

**Code Changes:**
- `document_loader.py`: Update `Chunk` to have `start_page` and `end_page`
- `chunk_text()`: Detect page boundaries when merging paragraphs
- `app.py`: Display page range in `render_result()`

---

### 2. **Image Preview/Attachment** ⭐ HIGH PRIORITY

**Problem:** When a result comes from a PDF/image, users can't see the visual context.

**Solution:**
- Extract and store page images from PDFs during indexing
- For image files, store reference to original image
- In search results, show thumbnail/preview of the page/image
- Click to view full-size image

**Implementation:**
- Add `page_image_path` or `image_reference` to Chunk metadata
- Extract PDF pages as images: `pdf2image` or `PyMuPDF` (already have both)
- Store images in `data/page_images/{file_name}/page_{num}.png`
- In `render_result()`, show image thumbnail if available
- Use Streamlit's `st.image()` for preview

**Code Changes:**
- `document_loader.py`: Extract page images during PDF processing
- `app.py`: Add image preview in `render_result()`
- Store image paths in Chroma metadata or separate mapping

---

### 3. **Better Embedding Model** ⭐ MEDIUM PRIORITY

**Problem:** MiniLM is fast but BGE M3 has better multilingual/retrieval quality.

**Solution:**
- Use BGE M3 for production (slower but better)
- OR: Use BGE M3 for vector search, keep MiniLM for reranker
- OR: Hybrid embeddings (MiniLM + BGE M3, fuse scores)

**Implementation:**
- Set `EMBEDDING_MODEL = "BAAI/bge-m3"` in config
- Re-index documents (will take longer)
- Monitor performance; if too slow, revert or optimize

---

### 4. **Query Expansion** ⭐ MEDIUM PRIORITY

**Problem:** User queries might miss synonyms or related terms.

**Solution:**
- Expand query with synonyms (WordNet, medical/legal term dictionaries)
- Add query rewriting (e.g., "noisy" → "noise", "noisy environment")
- Use LLM to generate query variations before retrieval

**Implementation:**
- Add `expand_query()` function using `nltk` WordNet or domain dictionaries
- Apply expansion before BM25 and vector search
- Keep original query for reranker (reranker handles semantics)

---

### 5. **Better Reranking** ⭐ MEDIUM PRIORITY

**Problem:** BGE reranker is good but could be improved.

**Solution:**
- Use larger reranker model (e.g., `BAAI/bge-reranker-large-v2`)
- Increase `RERANKER_CANDIDATES` to 50-100 for better recall
- Add cross-encoder reranking with query + multiple passages at once

**Implementation:**
- Update `RERANKER_MODEL` in config
- Increase `RERANKER_CANDIDATES` to 50
- Test quality vs speed tradeoff

---

### 6. **Metadata Filtering** ⭐ LOW PRIORITY (Future)

**Problem:** Can't filter by document type, date, patient at query time.

**Solution:**
- Add UI dropdown: "Filter by document type" (Policy/Claim/Medical)
- Use Chroma's `where` clause for metadata filtering
- Filter chunks before retrieval (or after, for display)

**Implementation:**
- Add filter UI in sidebar
- Pass `where` dict to Chroma `query()` when filter is set
- Apply same filter to BM25 results (post-filter)

---

## Quick Wins (Easy to Implement)

1. **Display page ranges** - Update `render_result()` to show "p.3-4" when chunk spans pages
2. **Add image thumbnails** - Extract PDF page images, show in results
3. **Switch to BGE M3** - Change config, re-index (if speed is acceptable)
4. **Increase reranker candidates** - Change `RERANKER_CANDIDATES` from 30 to 50

---

## Implementation Order

1. **Page ranges** (1-2 hours) - Most visible improvement
2. **Image previews** (2-3 hours) - High user value
3. **BGE M3** (30 min + re-index time) - Better quality
4. **Query expansion** (2-3 hours) - Better retrieval
5. **Better reranking** (30 min) - Easy config change

---

## Testing Checklist

After each improvement:
- [ ] Search for "noisy environment" - should find markdown 4 with correct page
- [ ] Search for exact claim ID - should find exact match
- [ ] Search for patient name - should find all occurrences
- [ ] Check page numbers are accurate (especially for multi-page chunks)
- [ ] Verify image previews show correct page
- [ ] Test AI insight still works with all providers
