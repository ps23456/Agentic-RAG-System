# Adding ChatGPT or Gemini to the RAG System (AI Insight)

The app can use an LLM (OpenAI ChatGPT or Google Gemini) to **generate an answer** from the **retrieved chunks** (RAG: retrieval-augmented generation). The LLM only sees the text you retrieved, so answers stay grounded in your documents.

---

## Step 1: Install dependencies

From the project root with the venv activated:

```bash
cd /path/to/testing
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install openai google-generativeai python-dotenv
```

Or install everything:

```bash
pip install -r requirements.txt
```

---

## Step 2: Get an API key

### Option A: OpenAI (ChatGPT)

1. Go to [OpenAI API](https://platform.openai.com/) and sign in or create an account.
2. Open **API keys** (e.g. [platform.openai.com/api-keys](https://platform.openai.com/api-keys)).
3. Create a new secret key and copy it. You will use **gpt-4o-mini** (or another model) for cost-effective answers.

### Option B: Google Gemini

1. Go to [Google AI Studio](https://aistudio.google.com/) or [Google AI for Developers](https://ai.google.dev/).
2. Get an API key for the Gemini API (e.g. “Get API key” in AI Studio).
3. Copy the key. The app uses **gemini-1.5-flash** by default.

---

### Option C: Mistral OCR (High-accuracy)

1. Go to [Mistral AI Console](https://console.mistral.ai/) and sign in.
2. Go to **API Keys** and create a new secret key.
3. This key allows high-accuracy OCR for images and scanned PDFs.

---

## Step 3: Set the API key

**Option 1 – `.env` file (recommended):**

Create a file named `.env` in the project root (same folder as `app.py`). The app loads it automatically via `python-dotenv`:

```bash
# .env (do not commit this file; it is in .gitignore)
GEMINI_API_KEY=your_gemini_key_here
# OPENAI_API_KEY=sk-your_openai_key_here
```

If `GEMINI_API_KEY` is set in `.env`, the sidebar will default to **Gemini** and the key field will be pre-filled. You can still override or paste the key in the UI.

**Option 2 – Environment variable (shell):**

```bash
# For OpenAI (ChatGPT)
export OPENAI_API_KEY="sk-..."

# For Gemini
export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY="..."
```

Then start the app:

```bash
./run.sh
# or: streamlit run app.py
```

**Option 3 – In the app:**

1. In the sidebar, open **“AI -- RAG”**.
2. Choose **OpenAI (ChatGPT)** or **Gemini**.
3. Paste your API key in the password field. (Leaving it empty will use the env var if set.)

---

## Step 4: Use AI Insight in the app

1. Run a search (e.g. “Why was claim CLM-8891 rejected?”) so that BM25 / Vector / Hybrid results appear.
2. In the sidebar, select **OpenAI (ChatGPT)** or **Gemini** and ensure the API key is set (or in the environment).
3. Below the three result columns, click **“Generate AI insight”**.
4. The LLM will answer using only the **retrieved chunks** as context. The **Sources used** line shows which files were passed to the LLM.

You can try different queries and click **“Generate AI insight”** again each time.

---

## How it works (RAG flow)

1. **Retrieve:** Your query runs against BM25, Vector, and Hybrid as before. Top chunks (from hybrid, vector, and BM25) are merged and deduplicated.
2. **Context:** Those chunks are turned into a single text block (with source labels like `[Source: claim_CLM-8891.pdf]`).
3. **Prompt:** The app sends to the LLM:
   - A system prompt: “Answer only from the provided excerpts; do not make up details.”
   - The context (excerpts) and the user question.
4. **Answer:** The model returns a short answer. The UI shows it and lists the **sources used** so you can check against the snippets above.

No extra indexing or “vector DB” is used for the LLM; it only sees the same chunks that are already shown in the three result columns.

---

## Optional: Change model or context length

- **OpenAI:** In `llm_insight.py`, change the `model` argument in `get_insight_openai()` (e.g. to `gpt-4o`).
- **Gemini:** In `llm_insight.py`, change the `model` argument in `get_insight_gemini()` (e.g. to `gemini-1.5-pro`).
- **Context size:** The default max context length is 12,000 characters. You can change `max_context_chars` in the `get_insight()` call in `app.py` or in `llm_insight.py`’s `build_context()`.

---

## Security

- Do **not** commit API keys to git. The `.env` file is in `.gitignore`. Use `.env`, environment variables, or enter the key only in the app (the app does not save the key to disk if you paste it in the UI).
- Keys are sent over the network to OpenAI or Google; use HTTPS (default) and keep your key secret.
