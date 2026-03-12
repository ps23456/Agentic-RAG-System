# Insurance Claim Search (Local Prototype)

Self-contained prototype for searching over **policy PDFs**, **claim PDFs**, and **medical reports** in a local folder. Supports **BM25** (keyword), **Vector** (semantic), and **Hybrid** (fusion) search with side-by-side comparison.

## Tech Stack

- **Python** + **Streamlit**
- **BM25**: rank-bm25
- **Vector**: sentence-transformers (all-MiniLM-L6-v2) + FAISS
- **PDF**: pdfplumber, PyMuPDF
- **OCR**: Tesseract (pytesseract + pdf2image for scanned PDFs/images)

## Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install mistralai  # Required for Mistral OCR
   ```

2. **NumPy / PyTorch**

   - The project pins `numpy>=1.26.0,<2` to avoid the "module compiled with NumPy 1.x" error when importing PyTorch. If you see that error, run: `pip install "numpy>=1.26.0,<2"`.
   - **PyTorch 2.2:** Run `.venv/bin/python scripts/patch_transformers_torch22.py` once after installing (or after creating a new venv). Then run the app with the project venv (e.g. `./run.sh`).

   To **search text inside uploaded images** (e.g. claim form photos, scanned docs), you can use:
   - **Mistral OCR (Recommended):** High-accuracy cloud API. Set `MISTRAL_API_KEY` in `.env` or the sidebar. (Requires `pip install mistralai`).
   - **Tesseract (Local):** 
     - **macOS:** `brew install tesseract`
     - **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
     - **Windows:** Install from [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

   Then click **Index / Re-index** in the app.

4. **Add documents to the data folder**

   - Default folder: `data/` in the project root.
   - Put **policy**, **claim**, and **medical** PDFs (or images) there.
   - Document type is inferred from filename:
     - `policy*` → policy  
     - `claim*`, `*CLM-*` → claim  
     - `medical*`, `*report*` → medical  

5. **Optional: Generate sample PDFs for testing**

   ```bash
   python scripts/create_sample_docs.py
   ```

   This creates `policy_terms.pdf`, `claim_CLM-8891.pdf`, and `medical_report_John_Doe.pdf` in `data/`.

## Troubleshooting

- **Python exits with code 136 or crashes on import**  
  This can happen when the **pytesseract** (Tesseract) native library is loaded. The app now loads OCR only when needed (lazy import), so the app should start even if Tesseract isn’t installed. If it still crashes, try: `pip uninstall pytesseract` and use text PDFs only, or install Tesseract and ensure it’s on your PATH (`brew install tesseract` on macOS).

- **"PyTorch is not installed"**  
  Use the same venv for both `pip install` and `streamlit run app.py`, and run `pip install "numpy>=1.26.0,<2"` and `pip install torch` in that venv.

## Run the app

**Recommended (uses the project venv’s Python):**
```bash
./run.sh
```

Or manually:
```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python -m streamlit run app.py
```

To check that the venv has all dependencies and PyTorch works:
```bash
.venv/bin/python scripts/check_env.py
```

If you reinstall or create a new venv and see "PyTorch is not installed" in the app, run the one-time patch:
```bash
.venv/bin/python scripts/patch_transformers_torch22.py
```

- In the sidebar: set **Data folder** (default: `data/`), then click **Index / Re-index documents**.
- Enter a query, e.g.:
  - *"Why was claim CLM-8891 rejected?"*
  - *"Show policy clause for pre-existing conditions"*
- View results in three columns:
  1. **BM25** – exact/keyword matches (IDs, clauses, legal terms).
  2. **Vector** – semantic matches (medical/insurance meaning, paraphrases).
  3. **Hybrid** – combined ranking via Reciprocal Rank Fusion (RRF).
- **Optional – AI -- RAG:** Use **ChatGPT** (OpenAI) or **Gemini** to get an answer generated from the retrieved chunks. See [docs/LLM_SETUP.md](docs/LLM_SETUP.md) for API keys and steps.

## Data source and indexing

- **Data source**: Only a local folder (e.g. `data/`). No cloud or external DB.
- **On Index**: All supported files in the folder are read; text is extracted (PDF or OCR for images/scanned pages), chunked by paragraph, then indexed for BM25 and vector search.
- **Metadata** per chunk: file name, page number, document type (policy/claim/medical).

## Hybrid logic

- BM25 and vector search run over the same query.
- Results are combined with **Reciprocal Rank Fusion** (RRF): `score(d) = 1/(k+rank_bm25) + 1/(k+rank_vector)`.
- Hybrid list is sorted by this fusion score so that hits that appear in both BM25 and vector results rank higher.

## Output (retrieval-only)

- Each result shows: **text snippet**, **source file**, **page number** (if available), **BM25 score** or **vector similarity** or **fusion score**.
- No generated answers; results are explainable and auditable from your documents.

## Environment

- Override data folder: `export CLAIM_SEARCH_DATA=/path/to/your/docs`
- First run will download the sentence-transformers model (~80MB).

## Objective

This prototype shows that:

- **BM25** reliably finds exact policy clauses and claim IDs.
- **Vector** search captures medical and insurance meaning.
- **Hybrid** retrieval works on real, newly added documents and is suitable for production-style insurance claim search.

Focus is on correctness and real-data behavior; the UI is kept simple and local.
