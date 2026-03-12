# Vector Database Guide – Open Source Options

This app currently uses **FAISS** (in-memory) for vector search. If you want **persistent storage**, **metadata filtering**, or a **separate vector service**, use an open-source vector database.

---

## Quick comparison

| Database   | Setup              | Persistence | Metadata filter | Best for                    |
|-----------|--------------------|------------|------------------|-----------------------------|
| **Chroma** | `pip install chromadb` | ✅ Disk     | ✅ Yes           | **Easiest; local/dev**      |
| **Qdrant**  | Docker or `pip` run   | ✅ Optional | ✅ Yes           | Production, flexible        |
| **pgvector**| PostgreSQL extension | ✅ DB       | ✅ SQL filters   | One DB for all data         |
| **LanceDB** | `pip install lancedb` | ✅ Disk     | ✅ Yes           | Embedded, no server         |

---

## Recommended: Chroma

**Why Chroma**

- **No server**: Runs in-process; data stored in a folder on disk.
- **Python-native**: Same process as your Streamlit app; no Docker required.
- **Metadata filtering**: Filter by `file_name`, `document_type`, etc. at query time.
- **Same embeddings**: Use your existing sentence-transformers model; Chroma only stores vectors and metadata.

**Install**

```bash
pip install chromadb
```

**Connection (in this app)**

1. Install: `pip install chromadb` (or use the project’s `requirements.txt`, which includes it).
2. Set the backend to Chroma (one of):
   - **Environment**: `export VECTOR_BACKEND=chroma` and optionally `export CHROMA_PERSIST_DIR=/path/to/data/chroma`.
   - **config.py**: Set `VECTOR_BACKEND = "chroma"` and, if you like, `CHROMA_PERSIST_DIR` to your desired folder (default: `./data/chroma`).
3. Run the app and click **Index / Re-index documents**. Vectors are stored in Chroma; re-indexing replaces the collection.

On **Index / Re-index**, the app will:

1. Load and chunk documents (unchanged).
2. Build BM25 in memory (unchanged).
3. Compute embeddings and **add them to Chroma** (instead of FAISS), with metadata: `file_name`, `page_number`, `document_type`.

Queries then hit Chroma for vector search; BM25 and hybrid (RRF + reranker) behave as before.

---

### How to check that files are stored in Chroma

1. **Use Chroma backend and re-index**
   - Set `VECTOR_BACKEND=chroma` (in `config.py` or env).
   - Start the app and click **Index / Re-index documents**.

2. **Run the check script** (from project root):
   ```bash
   .venv_bge/bin/python scripts/check_chroma.py
   ```
   It prints:
   - Whether the backend is Chroma and where the DB is stored.
   - Total number of chunks in the collection.
   - A sample of stored chunk IDs and metadata (`file_name`, `page_number`, `document_type`).

3. **Inspect the persist directory**
   - Chroma stores files under `CHROMA_PERSIST_DIR` (default: `data/chroma`).
   - If that folder exists and has files (e.g. `chroma.sqlite3`, subdirs), the DB is in use.

### How it works with Chroma (same behavior as FAISS)

| Step | What happens |
|------|-------------------------------|
| **Index / Re-index** | Documents → chunks (in memory). BM25 built in memory. Embeddings computed → **added to Chroma** (one row per chunk: `id`, vector, `file_name`, `page_number`, `document_type`). Collection is replaced on each re-index. |
| **Search** | Your query is embedded. **Chroma** returns the nearest chunk IDs and distances. App converts distance → similarity and maps IDs back to chunks (from in-memory `chunk_by_id`). Same token/page rerank as with FAISS. |
| **AI Insight** | Same as before: uses hybrid + reranked results and “full top doc first”; no change when using Chroma. |

Chroma only stores **vectors and metadata**. Chunk text stays in app memory for the session; re-indexing repopulates both in-memory chunks and the Chroma collection.

---

**Optional: filter by document**

Chroma supports `where` on metadata, e.g. “only chunks from `markdown 4.md`” or `document_type == "claim"`. That can be exposed later in the UI (e.g. dropdown “Search in: All files / Claim forms only”).

---

## Alternative: Qdrant

**When to use**

- You want a **separate service** (e.g. Docker container).
- You need **scaling** or **multi-process** access to the same index.

**Setup**

```bash
# Run Qdrant server (Docker)
docker run -p 6333:6333 qdrant/qdrant
```

**Connect (Python)**

```bash
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
# Create collection, add points with vector + payload (metadata), query with filter
```

You would then point this app’s vector backend to Qdrant (similar to the Chroma integration: config flag + a `search_index` backend that uses `qdrant_client` instead of FAISS).

---

## Alternative: pgvector (PostgreSQL)

**When to use**

- You already use **PostgreSQL**.
- You want **one database** for metadata, users, and vectors (e.g. filter by patient_id, date, doc type in SQL).

**Setup**

- Install PostgreSQL and the **pgvector** extension.
- Create a table with a vector column; store chunk text and metadata in other columns.
- Use your embedding model in Python and `INSERT`/`SELECT` with vector similarity.

This requires more wiring (migrations, connection string, SQL) and is best when you’re ready to move full metadata and access control into Postgres.

---

## Summary

- **Start with Chroma**: add `chromadb`, set `VECTOR_BACKEND=chroma` and `CHROMA_PERSIST_DIR`, then use the app’s existing “Index / Re-index” and search. No server, persistence and metadata filtering built in.
- **Later**: Add UI for metadata filters (e.g. by file or doc type) that translate to Chroma `where`.
- **If you outgrow Chroma**: Consider Qdrant (separate service) or pgvector (single SQL DB).
