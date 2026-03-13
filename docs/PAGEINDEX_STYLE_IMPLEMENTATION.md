# PageIndex-Style Implementation Plan
## Agentic Multimodal Hybrid RAG – Clickable Page References & Source Viewer

**Goal:** Like PageIndex.ai, provide exact page references and click-to-view so users see the exact source page, building confidence that the answer comes from that location.

---

## Current State vs. Target

| Feature | Current | Target (PageIndex-style) |
|---------|---------|--------------------------|
| Page reference in results | ✅ `filename.pdf (p.123)` | ✅ Same, but **clickable** |
| Click to open source | ❌ No | ✅ Opens PDF to that exact page in a viewer |
| PDF viewer panel | ❌ No | ✅ Right panel shows source document |
| Large PDF handling | ❌ Fails → user uses Markdown | ✅ Support Markdown + optional PDF preview |
| AI answer citations | ✅ "Sources used: file1, file2" | ✅ Inline citations: `[filename p.406]` |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User Query → Agentic RAG → Retrieval → Fused Results → AI Insight          │
│                                                                             │
│  For each result chunk:                                                     │
│    • file_name, page_number, document_type (already in Chunk)                │
│    • file_path (need to add: resolve from data_folder + file_name)          │
│                                                                             │
│  For each citation in AI answer:                                            │
│    • LLM instructed to cite as [filename p.N]                               │
│    • Parse citations → render as clickable links                            │
│    • On click → open PDF viewer to page N (or show Markdown section)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Store & Resolve File Paths (Foundation)

**Problem:** Chunks have `file_name` but not full path. We need path to open PDFs.

**Solution:**
1. Add `file_path` to Chunk metadata (optional; can be computed from `file_name` + `data_folder`).
2. Add helper: `resolve_file_path(file_name, data_folder)` – walks data folder to find file.

**Code changes:**
- `document_loader.py`: When creating Chunk, pass `path` or store relative path.
- `search_index.py` / ChromaDB: Add `file_path` to metadata if not too long (or store relative path).
- `app.py`: Add `def _resolve_path(fname, data_folder) -> str | None`.

**Markdown handling:** For `.md` files, "page" = section number. We can show the Markdown section content in an expander instead of a PDF viewer.

---

### Phase 2: Clickable Page References in Results

**Current:** `**1. filename.pdf (p.123)**` – plain text.

**Target:** `**1. [filename.pdf p.123](#) **` – clickable; on click, scroll to viewer and show that page.

**Implementation:**
1. Use Streamlit `st.session_state` to store `selected_citation = {"file": "x.pdf", "page": 123}`.
2. Render each result as a button or link that sets `selected_citation` on click.
3. Use `st.container()` with a key for the viewer; when `selected_citation` changes, re-render viewer.

**Streamlit approach:** Use `st.button` or `st.link` with `on_click` callback. Or use `st.markdown` with custom HTML (Streamlit allows limited HTML). Simpler: use `st.expander` with "View source page" – when expanded, show the page.

---

### Phase 3: PDF Viewer & Page Jump

**Problem:** Streamlit has no built-in PDF viewer with page parameter.

**Options:**

| Option | Pros | Cons |
|--------|------|-----|
| **A. Render PDF page as image** | Simple, works with PyMuPDF (already have it) | No native PDF feel; one page at a time |
| **B. `streamlit-pdf-viewer`** | Real PDF viewer | May need page jump; check package support |
| **C. Embed PDF.js in iframe** | Full PDF viewer | Complex; need to serve PDF + page param |
| **D. Open in new tab** | Browser native PDF viewer | Can't pass page number reliably |

**Recommended: Option A – Render page as image**
- Use PyMuPDF: `doc.load_page(page_num - 1).get_pixmap()` → PNG
- Display with `st.image()`
- Add "Page 406 / 560" caption
- Works for all PDFs; no extra dependencies

**Code:**
```python
def render_pdf_page(path: str, page: int | None = 1) -> None:
    import fitz
    doc = fitz.open(path)
    page = min(max(1, page or 1), len(doc))
    pix = doc.load_page(page - 1).get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    st.image(img, caption=f"Page {page} / {len(doc)}")
    doc.close()
```

---

### Phase 4: Layout – Results Left, Viewer Right (PageIndex-style)

**Layout:**
```
┌─────────────────────────────┬─────────────────────────────┐
│  Query + Results            │  Source Viewer               │
│  • AI answer with citations │  • PDF page image or         │
│  • Clickable [file p.N]     │    Markdown section          │
│  • Result list              │  • Page 406 / 560             │
└─────────────────────────────┴─────────────────────────────┘
```

**Streamlit:** Use `st.columns([2, 1])` – left for results, right for viewer. When user clicks a citation, update the right column.

---

### Phase 5: LLM Citations with Page Numbers

**Current:** `build_context()` adds `[Source: filename, Page N]` before each chunk. LLM sees it but may not include in answer.

**Target:** LLM should cite inline: `[PNB AR 2024-25_Web.pdf p.406]` so we can parse and make clickable.

**Prompt change in `llm_insight.py`:**
```
When citing sources, use the format: [filename p.N] (e.g. [PNB_AR_2024.pdf p.406]).
Include the exact page number for each fact.
```

**Then:**
- Parse response for `[filename p.N]` pattern.
- Replace with clickable link: `[filename p.N](?file=...&page=N)` or use a custom component.

---

### Phase 6: Large PDF + Markdown Workflow

**Problem:** Large PDFs don't index → user uploads Markdown instead.

**Current flow:**
- `extract_text_from_pdf()` checks for companion `.md` (e.g. `pnb.md` next to `pnb.pdf`).
- If companion exists, uses Markdown (which has page splits from conversion).
- Chunks get `page_number` from Markdown sections.

**For Markdown-only (no PDF):**
- No PDF to show. Instead: show the **Markdown section** that contains the chunk.
- Store `section_content` or re-extract: given `file_name` + `page_number`, read the Markdown and extract that section.

**Implementation:**
- Add `get_markdown_section(path, section_num)` – returns the text for that section.
- In viewer: when source is `.md`, show `st.markdown(section_content)` instead of PDF image.

**For PDF + Markdown companion:**
- User has both: `pnb.pdf` and `pnb.md` (converted from PDF).
- Index from Markdown (fast).
- When showing source: use **PDF** if it exists, else Markdown section.
- Resolve: `path = pdf_path if exists else md_path`.

---

## Implementation Order

| Step | Task | Effort | Blocks |
|------|------|--------|--------|
| 1 | Add `_resolve_path` helper | 30 min | - |
| 2 | Render PDF page as image (viewer) | 1 hr | - |
| 3 | Add right-panel viewer to Multimodal Hybrid RAG | 1 hr | 1, 2 |
| 4 | Make result citations clickable → update viewer | 1.5 hr | 3 |
| 5 | Update LLM prompt for citations | 30 min | - |
| 6 | Parse citations in answer | 1 hr | 5 |
| 7 | Markdown section viewer | 1 hr | 1 |
| 8 | Large PDF: use Markdown + optional PDF preview | 1 hr | 7 |

---

## File Changes Summary

| File | Changes |
|------|---------|
| `app.py` | `_resolve_path()`, viewer column, clickable citations, `st.session_state` for selected citation |
| `llm_insight.py` | Add citation instruction to prompt; optional `parse_citations()` helper |
| `document_loader.py` | Optional: add `file_path` to Chunk; `get_markdown_section()` |
| `config.py` | Optional: `PAGE_IMAGE_DPI`, `VIEWER_COLUMN_WIDTH` |

---

## Quick Win (Minimal First Step)

**1–2 hours:** Add a "View source" button next to each result. On click, show an expander with the PDF page rendered as image (or Markdown section). No layout change, no citation parsing yet. This gives user confidence immediately.

```python
# In render_result or fused result loop:
for i, row in enumerate(fused[:10], 1):
    ...
    path = _resolve_path(chunk.file_name, data_folder)
    if path and path.lower().endswith(".pdf"):
        with st.expander(f"📄 View source: {chunk.file_name} p.{chunk.page_number}"):
            render_pdf_page(path, chunk.page_number)
```

---

## Summary

- **Page references:** Already in chunks; make them clickable.
- **PDF viewer:** Render page as image with PyMuPDF; no new deps.
- **Markdown:** Show section content when no PDF.
- **Large PDFs:** Use Markdown for indexing; show PDF when available, else Markdown section.
- **Citations:** Update prompt; parse `[file p.N]`; make clickable.

Start with the Quick Win, then expand to full PageIndex-style layout.
