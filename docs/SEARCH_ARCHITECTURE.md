# How BM25, Vector, and Hybrid Search Work in This Project

All three methods search over the **same set of text chunks**. They differ in how those chunks are indexed and how the query is matched.

---

## 1. Shared pipeline: documents → chunks

**Where:** `document_loader.py` + `config.py`

1. **Load:** All PDFs and images under the data folder (e.g. `data/`) are read.
2. **Extract text:** PDFs use pdfplumber/PyMuPDF; images and scanned pages use **Tesseract OCR**.
3. **Chunk:** Text is split by paragraph into chunks (max ~1500 chars, min ~50). Each chunk keeps:
   - `chunk_id`, `text`, `file_name`, `page_number`, `document_type`.
4. **Single list:** One list of `Chunk` objects is passed to `SearchIndex(chunks=chunks)`.

So **BM25 and Vector both use the same chunks**; they do not have separate “databases” of documents—only different **indexes** over the same chunks.

---

## 2. BM25 (keyword / lexical search)

**Where:** `search_index.py` — `build_bm25()`, `bm25_search()`, and `_get_tokenizer()`.

**Library:** `rank_bm25.BM25Okapi`.

### Index build (`build_bm25`)

1. **Tokenize each chunk:**  
   - Lowercase, keep letters, digits, hyphens (for IDs like `CLM-8891`).  
   - Split on whitespace, drop single non-digit characters.  
   - Result: one list of tokens per chunk, e.g. `["claim", "clm", "8891", "rejected", ...]`.
2. **Build BM25 index:**  
   - `BM25Okapi(self._bm25_corpus)` builds the inverted index and term stats (IDF, etc.) over that tokenized corpus.  
   - No embeddings; only token counts and document lengths.

### Search (`bm25_search`)

1. **Tokenize the query** with the same tokenizer.
2. **Score all chunks:** `self._bm25.get_scores(q_tokens)` returns one BM25 score per chunk (how well the chunk matches the query terms).
3. **Rank:** Sort by score descending, take top `BM25_TOP_K` (e.g. 20).
4. **Return:** List of `(Chunk, score)` for chunks with score &gt; 0.

**In short:** BM25 ranks by **exact (token) overlap** and term frequency. Good for claim IDs, clause names, and exact phrases; no understanding of meaning.

---

## 3. Vector DB (semantic search)

**Where:** `search_index.py` — `build_vector_index()`, `vector_search()`, and `_load_embedding_model()`.

**Libraries:** `sentence_transformers.SentenceTransformer`, `faiss.IndexFlatIP` (FAISS).

### Index build (`build_vector_index`)

1. **Embed all chunk texts:**  
   - Model: `all-MiniLM-L6-v2` (config: `EMBEDDING_MODEL`).  
   - `model.encode(texts)` → one vector per chunk (e.g. 384-dim).
2. **Normalize:** L2-normalize vectors so that inner product = cosine similarity.
3. **FAISS index:**  
   - `faiss.IndexFlatIP(dim)` = exact search with **inner product**.  
   - `index.add(embeddings)` stores all chunk vectors.  
   - No separate “vector DB” process; it’s an in-memory FAISS index over the same chunks.

### Search (`vector_search`)

1. **Embed the query:** `model.encode([query])` → one vector, same dimension.
2. **Normalize** the query vector.
3. **FAISS search:** `index.search(q_emb, top_k)` returns top‑k chunk indices and **similarity scores** (inner product = cosine similarity).
4. **Return:** List of `(Chunk, similarity)`.

**In short:** Vector search ranks by **meaning similarity** (paraphrases, related concepts). Good for “rejection reason” matching “denied because…”; not for exact IDs unless the model happens to embed them similarly.

---

## 4. Hybrid (BM25 + Vector combined)

**Where:** `search_index.py` — `hybrid_search()`, `_reciprocal_rank_fusion()`, and optionally `_weighted_fusion()`.

**Idea:** Run BM25 and Vector **separately**, then merge their rankings so chunks that do well in **both** (or either) get a single combined score.

### Flow

1. **Run both retrievers (same query, same chunks):**
   - `bm25_hits = self.bm25_search(query, top_k=BM25_TOP_K)`
   - `vector_hits = self.vector_search(query, top_k=VECTOR_TOP_K)`
2. **Fuse rankings.** Two options in code:
   - **RRF (default):** Reciprocal Rank Fusion.
   - **Weighted:** Min‑max normalize BM25 and vector scores, then `0.5 * norm_bm25 + 0.5 * norm_vector`.

### RRF (Reciprocal Rank Fusion)

- For each chunk that appears in either list, compute:
  - `score(chunk) = 1/(k + rank_in_bm25) + 1/(k + rank_in_vector)`
  - `k = RRF_K` (e.g. 60) so ranks are smoothed.
  - If a chunk is only in one list, it still gets one term; if in both, it gets two terms (so it tends to rank higher).
- **Sort** all chunks by this fusion score descending.
- **Return** top `HYBRID_TOP_K` (e.g. 15).

So **hybrid does not train a model**; it only **combines the two rankings** (BM25 + vector) so you get both keyword hits and semantic hits, with a boost for chunks that appear in both.

---

## 5. Summary table

| Component   | What is indexed        | How the query is used           | Output              |
|------------|------------------------|----------------------------------|---------------------|
| **BM25**   | Tokenized chunk text   | Tokenized → BM25 score per chunk | Top‑k by BM25 score  |
| **Vector** | Embeddings of chunks   | Query → embedding → FAISS search | Top‑k by similarity |
| **Hybrid** | Same BM25 + Vector     | Same query → both → RRF (or weighted) | Top‑k by fusion score |

All three use the **same chunks** from `load_and_chunk_folder()`; only the **index structures** (BM25 corpus vs FAISS vectors) and the **scoring/ranking** logic differ. The app then applies optional **minimum-score filters** (e.g. `VECTOR_MIN_SIMILARITY`, `HYBRID_MIN_FUSION_SCORE`) before showing results in the UI.
