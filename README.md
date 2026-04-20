# Insurance Claim Search

Local-first prototype for searching **policy PDFs**, **claim documents**, and **medical reports** in a folder you control. It combines **BM25** (keyword), **dense vector** (semantic) search, and **hybrid fusion** (Reciprocal Rank Fusion), with optional **RAG chat**, **multimodal** (text + image) retrieval, and **PageIndex-style hierarchical tree** indexing over PDFs.

## Architecture

| Layer | Stack |
|--------|--------|
| **Web UI (recommended)** | React 19 + Vite + TypeScript + Tailwind — `frontend/` |
| **API** | FastAPI — `backend/` (`uvicorn backend.main:app`) |
| **Legacy UI** | Streamlit — `app.py` (same retrieval stack, different surface) |

The Vite dev server proxies `/api` to the backend (see `frontend/vite.config.ts`).

## Features (current)

- **Hybrid retrieval**: BM25 + embeddings, fused with RRF; optional **CrossEncoder** reranking (`config.py`).
- **Vector store**: **Chroma** persistent DB by default (`data/chroma/`); configurable via `VECTOR_BACKEND` / `CHROMA_*` in `config.py`.
- **Multimodal**: Separate text and image (CLIP) collections and weighted fusion for diagram/visual queries.
- **Tree index**: PageIndex-style TOC → hierarchical section tree (custom implementation in `indexing/page_tree.py`, inspired by [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)). Cached JSON under `data/cache/trees/`.
- **Document types**: Inferred from filenames (`policy*`, `claim*` / `CLM-*`, `medical*` / `*report*`, etc.) — see `config.py` `get_doc_type`.
- **Formats**: PDF, images (OCR), Markdown, JSON, plain text uploads under `data/uploads/` by default.
- **LLM**: Groq / OpenAI / Gemini for query understanding, chat, and tree navigation — keys via `.env` (details in [docs/LLM_SETUP.md](docs/LLM_SETUP.md)).

## Requirements

- **Python** 3.10+ (3.12 used in some setups)
- **Node.js** 20+ (for the React app)

## Setup

### 1. Python environment

```bash
cd /path/to/repo
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Some teams use a separate **`.venv_bge`** for BGE M3 and heavier models — see [docs/BGE_M3_SETUP.md](docs/BGE_M3_SETUP.md). Either venv is fine if dependencies install cleanly.

**NumPy / PyTorch:** The repo pins `numpy>=1.26.0,<2` for compatibility with PyTorch and several ML stacks. If you see NumPy ABI errors, reinstall with that constraint.

Optional one-time helper (only if your environment needs it):

```bash
.venv/bin/python scripts/patch_transformers_torch22.py
.venv/bin/python scripts/check_env.py
```

### 2. Frontend dependencies

```bash
cd frontend
npm install
```

### 3. Environment variables

Create a **`.env`** file in the project root (same level as `config.py`). It is gitignored. Common variables:

- **LLM / APIs**: e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `GROQ_API_KEY`, `MISTRAL_OCR_API_KEY` (see [docs/LLM_SETUP.md](docs/LLM_SETUP.md)).
- **Data path**: `CLAIM_SEARCH_DATA` — overrides the default `data/` folder.
- **Vector DB**: `VECTOR_BACKEND`, `CHROMA_PERSIST_DIR` (defaults in `config.py`).

### 4. Data folder

By default, documents live under **`data/`** (or `CLAIM_SEARCH_DATA`). Place PDFs and other supported files there; uploads typically go to `data/uploads/`. That folder is **gitignored** so clones get an empty uploads directory and each installation keeps its own files locally. After adding files, trigger indexing from the UI or API (see below).

## Run the app

### Option A — React UI + FastAPI (recommended)

**Terminal 1 — backend**

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — frontend**

```bash
cd frontend
npm run dev
```

- **Frontend:** http://localhost:3000  
- **API:** http://localhost:8000  
- **Health check:** `GET http://localhost:8000/api/health`

There is a **`run_app.sh`** script that starts both processes; it may assume a specific venv path on your machine — adjust the script or use the two-terminal flow above for portability.

### Option B — Streamlit UI

```bash
./run.sh
```

Uses the project venv (prefers `.venv_bge` if present, else `.venv`) and runs Streamlit on port **8501**:

```bash
python -m streamlit run app.py --server.port 8501
```

Indexing modes (chunk-only, tree-only, or both) are exposed in the Streamlit sidebar; the React app uses the FastAPI `RAGService` configuration from `config.py`.

## API indexing (FastAPI)

- `POST /api/index` — full reindex (documents + images)  
- `POST /api/index/docs` — documents only  
- `POST /api/index/images` — image index only  
- `GET /api/index/status` — indexing status  

Chat, documents, medical, upload, and fields routes live under `backend/routes/`.

## Documentation (repo)

| Doc | Topic |
|-----|--------|
| [docs/LLM_SETUP.md](docs/LLM_SETUP.md) | API keys and RAG / LLM usage |
| [docs/BGE_M3_SETUP.md](docs/BGE_M3_SETUP.md) | BGE M3 embedding setup |
| [docs/VECTOR_DB_GUIDE.md](docs/VECTOR_DB_GUIDE.md) | Chroma / vector configuration |
| [docs/SEARCH_ARCHITECTURE.md](docs/SEARCH_ARCHITECTURE.md) | Retrieval design |
| [docs/MULTIMODAL_HYBRID_FLOW.md](docs/MULTIMODAL_HYBRID_FLOW.md) | Multimodal hybrid RAG |

## Optional tooling

- **Sample PDFs:** `python scripts/create_sample_docs.py`  
- **RAG evaluation:** `scripts/run_ragas_eval.py` (requires optional RAG eval deps in `requirements.txt`)  
- **OCR:** Tesseract locally, or Mistral OCR via API — see [docs/LLM_SETUP.md](docs/LLM_SETUP.md)

## Troubleshooting

- **Wrong Python / missing packages:** Always activate the same venv you used for `pip install`, and run `uvicorn`/`streamlit` from that environment.
- **CORS / API errors in the browser:** Ensure the backend is on port 8000 and the frontend dev server proxies `/api` (default Vite config).
- **PyTorch / OCR import issues:** Lazy loading is used where possible; if native code crashes on import, install Tesseract on PATH or restrict to text PDFs; see [docs/LLM_SETUP.md](docs/LLM_SETUP.md) for Mistral OCR.

## Objective

Demonstrate production-style **insurance and medical document search**: explainable retrieval (sources, pages), hybrid ranking, and optional grounded generation — all against **local data** you provide.
