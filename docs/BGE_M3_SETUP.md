# Guide: Use BGE M3 Embedding Model

BGE M3 requires **PyTorch (torch) >= 2.6**. If your current environment has torch 2.2.2, follow these steps to get torch 2.6+ and then use BGE M3.

---

## Step 1: Check your Python version

```bash
python3 --version
```

- If you see **Python 3.11** or **3.12**: go to Step 2.
- If you see **Python 3.10** or older: you need a newer Python first (Step 1b).

### Step 1b (only if Python is 3.10 or older): Install Python 3.11 or 3.12

**On macOS (Homebrew):**
```bash
brew install python@3.12
```

**On Windows:** Download Python 3.12 from [python.org](https://www.python.org/downloads/) and install.

**Or use pyenv:**
```bash
pyenv install 3.12
pyenv local 3.12
```

Then use that Python for the project (e.g. create a new venv with it in Step 2).

---

## Step 2: Create a new virtual environment with Python 3.11+

From your project folder:

```bash
cd /Users/pshah/testing

# Use Python 3.11 or 3.12 (adjust name if needed)
python3.12 -m venv .venv_bge

# Activate it
# macOS/Linux:
source .venv_bge/bin/activate
# Windows:
# .venv_bge\Scripts\activate
```

---

## Step 3: Install PyTorch 2.6+ from PyTorch’s index

With the new venv activated:

**CPU only (recommended if you don’t use GPU):**
```bash
pip install --upgrade pip
pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cpu
```

**macOS (Apple Silicon M1/M2/M3):**
```bash
pip install torch>=2.6 --index-url https://download.pytorch.org/whl
```

Check that torch is 2.6 or higher:
```bash
python -c "import torch; print(torch.__version__)"
```
You should see something like `2.6.0` or higher.

---

## Step 4: Install the rest of the project dependencies

Still in the same venv:

```bash
pip install -r requirements.txt
```

If anything fails, install the main ones manually:
```bash
pip install sentence-transformers faiss-cpu streamlit rank-bm25 numpy
```

---

## Step 5: Set the embedding model to BGE M3

In `config.py`, set:
```python
EMBEDDING_MODEL = "BAAI/bge-m3"
```
(This is already set if you followed the project’s BGE M3 setup.)

---

## Step 6: Pre-download BGE M3 (optional but recommended)

So the first app run doesn’t wait on a big download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3'); print('BGE M3 ready')"
```

Wait until you see `BGE M3 ready`. The model is cached for future runs.

---

## Step 7: Run the app and re-index

```bash
streamlit run app.py
```

In the app, click **“Index / Re-index documents”**. The vector index will be built with BGE M3. After that, semantic and hybrid search use BGE M3.

---

## Quick reference

| Step | Action |
|------|--------|
| 1 | `python3 --version` → need 3.11+ |
| 2 | New venv: `python3.12 -m venv .venv_bge` then activate |
| 3 | `pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cpu` |
| 4 | `pip install -r requirements.txt` |
| 5 | `config.py`: `EMBEDDING_MODEL = "BAAI/bge-m3"` |
| 6 | `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3'); print('BGE M3 ready')"` |
| 7 | `streamlit run app.py` → Re-index in the app |

---

## If you prefer to stay in your current environment

Try forcing a newer torch from PyTorch’s index **without** a new Python/venv:

```bash
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
```

If the version is still 2.2.x, your current Python or platform likely doesn’t have torch 2.6 on PyPI/PyTorch index; then use the new venv + Python 3.11/3.12 path above.
