"""
PageIndex-style tree indexing: vectorless, no chunking, reasoning-based retrieval.

Builds a HIERARCHICAL tree from PDFs (sections → sub-sections with page ranges).
LLM reasons over the tree for retrieval — multi-step navigation into broad sections.

Follows the approach from https://github.com/VectifyAI/PageIndex:
1. Detect TOC in first N pages
2. Extract hierarchical structure with page numbers
3. Convert flat list (with "structure" field like 1, 1.1, 1.2) into nested tree
4. Tree search: LLM navigates tree → drills into sections → extracts page text
"""
import concurrent.futures
import json
import logging
import os
import re
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from config import DATA_FOLDER, PAGE_TREE_CACHE_DIR, PAGE_TREE_MAX_PAGES_PER_LLM, PAGE_TREE_TOC_PAGES, PAGE_TREE_MAX_PAGES_PER_NODE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF page-text cache
# ---------------------------------------------------------------------------
# Opening a PDF and extracting every page with fitz is expensive — for a
# 560-page file it takes several seconds. The retrieval pipeline used to do
# this THREE times per query (tree_keyword_retrieve, _merge_tree_and_rag, and
# _direct_pdf_scan), which was the main source of the multi-minute "thinking"
# time. We now cache the extracted texts keyed by (absolute_path, mtime) so
# repeat scans are instant. The cache can be warmed at server startup via
# `preload_pdf_pages` so even the first query is fast.
_PDF_PAGES_CACHE: dict[tuple[str, float], list[str]] = {}
_PDF_CACHE_LOCK = threading.Lock()


def get_pdf_page_texts(path: str) -> list[str]:
    """Return plain text for every page of a PDF, using a process-wide cache.

    The cache key includes the file's mtime so the cache invalidates
    automatically when the file is re-uploaded. Failures return an empty list
    but are NOT cached, so transient I/O errors self-heal on retry.
    """
    if not path:
        return []
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    key = (os.path.abspath(path), mtime)
    with _PDF_CACHE_LOCK:
        cached = _PDF_PAGES_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import fitz
        doc = fitz.open(path)
        pages = [doc.load_page(i).get_text() for i in range(len(doc))]
        doc.close()
    except Exception as e:
        logger.warning("PDF text extract failed for %s: %s", path, e)
        return []
    with _PDF_CACHE_LOCK:
        _PDF_PAGES_CACHE[key] = pages
    return pages


def preload_pdf_pages(paths: list[str], max_workers: int = 4) -> None:
    """Warm the PDF page-text cache in parallel.

    Called once at server startup so the first user query does not pay the
    cold-read cost for large PDFs (e.g. annual reports). Fitz releases the GIL
    during page extraction, so a thread pool gives a near-linear speedup.
    """
    paths = [p for p in (paths or []) if p and os.path.isfile(p)]
    if not paths:
        return
    t0 = _time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(get_pdf_page_texts, paths))
    logger.info(
        "Preloaded %d PDF(s) into page-text cache in %.2fs",
        len(paths), _time.time() - t0,
    )


@dataclass
class TreeChunk:
    """Chunk-like wrapper for tree search results (display-compatible with Chunk)."""
    chunk_id: str
    text: str
    file_name: str
    page_number: int | None
    document_type: str = "tree"
    start_char: int = 0
    end_char: int = 0
    patient_name: str = ""
    claim_number: str = ""
    policy_number: str = ""
    group_number: str = ""
    doctor_name: str = ""


def _is_groq_limit_error(e: Exception) -> bool:
    """True if error is Groq rate limit / quota exceeded."""
    msg = str(e).lower()
    if "rate" in msg and "limit" in msg:
        return True
    if "429" in msg or "quota" in msg or "limit exceeded" in msg:
        return True
    try:
        from groq import RateLimitError
        return isinstance(e, RateLimitError)
    except ImportError:
        pass
    return getattr(e, "status_code", None) == 429


def _call_llm_tree(prompt: str, api_key: str, provider: str, max_tokens: int = 4000) -> str | None:
    """Call LLM for tree building/search. Falls back to OpenAI when Groq limit exceeded."""
    api_key = (api_key or "").strip() or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        return None
    provider = (provider or "").strip().lower()
    if not provider:
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            provider = "groq"
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            groq_max = min(max_tokens, 4096)
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=groq_max,
                temperature=0.1,
            )
            return r.choices[0].message.content if r.choices else None
        elif provider in ("gemini", "google"):
            try:
                from google import genai
                client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
                r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                return getattr(r, "text", None)
            except Exception:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                r = model.generate_content(prompt)
                return getattr(r, "text", None)
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return r.choices[0].message.content if r.choices else None
    except Exception as e:
        # When Groq rate limit exceeded, fallback to OpenAI if available
        if provider == "groq" and _is_groq_limit_error(e):
            openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if openai_key:
                logger.info("Groq limit exceeded, falling back to OpenAI")
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    r = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.1,
                    )
                    return r.choices[0].message.content if r.choices else None
                except Exception as fallback_err:
                    logger.warning("OpenAI fallback also failed: %s", fallback_err)
        logger.warning("LLM call for tree failed: %s", e)
    return None


def _extract_json(text: str) -> Any:
    """Extract JSON from LLM response (handles ```json blocks)."""
    text = (text or "").strip()
    # Try ```json blocks first
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    start = text.find("[")
    if start == -1:
        start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("]") + 1 if "[" in text else text.rfind("}") + 1
    if end <= start:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        # Clean common issues
        raw = text[start:end]
        raw = raw.replace(",]", "]").replace(",}", "}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


# ─────────────────────────────────────────────────────────
# Tree building: hierarchical TOC extraction
# ─────────────────────────────────────────────────────────

def _detect_toc_pages(pages: List[tuple[int, str]], api_key: str, provider: str) -> List[int]:
    """Detect which pages contain a Table of Contents (like PageIndex find_toc_pages)."""
    toc_pages = []
    check_range = min(len(pages), PAGE_TREE_TOC_PAGES)

    for i in range(check_range):
        page_num, text = pages[i]
        sample = text[:2000]
        # Heuristic: TOC pages have many "..." or repeated dots/numbers
        dot_lines = len(re.findall(r"\.{3,}", sample))
        number_lines = len(re.findall(r"\b\d{1,3}\s*$", sample, re.MULTILINE))
        has_toc_header = bool(re.search(r"(?:table\s+of\s+contents|contents|index)", sample, re.IGNORECASE))

        if has_toc_header or dot_lines >= 3 or number_lines >= 5:
            toc_pages.append(i)
        elif toc_pages and i == toc_pages[-1] + 1:
            # Continue if consecutive with previous TOC page (multi-page TOC)
            if dot_lines >= 1 or number_lines >= 3:
                toc_pages.append(i)
        elif toc_pages:
            break  # Stop after gap
    return toc_pages


def _extract_toc_hierarchical(
    pages: List[tuple[int, str]],
    toc_page_indices: List[int],
    api_key: str,
    provider: str,
) -> List[dict]:
    """
    Extract hierarchical TOC with structure numbers (1, 1.1, 1.2, etc.) and page numbers.
    This is the key step from PageIndex: we get the full tree structure from the TOC.
    """
    toc_text = ""
    for idx in toc_page_indices:
        if idx < len(pages):
            pnum, text = pages[idx]
            # Replace dots with colons for cleaner parsing
            cleaned = re.sub(r"\.{3,}", " : ", text[:2000])
            cleaned = re.sub(r"(?:\. ){3,}\.?", " : ", cleaned)
            toc_text += f"\n--- Page {pnum} ---\n{cleaned}\n"
    # Cap total TOC text to stay within context limits (especially Groq)
    if len(toc_text) > 20000:
        toc_text = toc_text[:20000] + "\n... (TOC truncated)"

    prompt = f"""You are given pages from a document's Table of Contents.
Extract the COMPLETE hierarchical table of contents as a flat JSON array.

CRITICAL: Use the "structure" field to encode hierarchy:
- Top-level sections: "1", "2", "3", ...
- Sub-sections: "1.1", "1.2", "2.1", ...
- Sub-sub-sections: "1.1.1", "1.1.2", ...

Include page numbers from the TOC as "page" (integer).

Table of Contents pages:
{toc_text}

Return JSON array:
[
  {{"structure": "1", "title": "Introduction", "page": 1}},
  {{"structure": "1.1", "title": "Overview", "page": 3}},
  {{"structure": "2", "title": "Risk Management", "page": 45}},
  {{"structure": "2.1", "title": "Credit Risk", "page": 46}},
  {{"structure": "2.1.1", "title": "Qualitative Disclosures", "page": 48}},
  ...
]

Extract ALL items from the TOC. Include every section and sub-section.
Return ONLY the JSON array, no other text."""

    raw = _call_llm_tree(prompt, api_key, provider, max_tokens=8000)
    if not raw:
        return []
    parsed = _extract_json(raw)
    if not isinstance(parsed, list) or not parsed:
        return []

    # Clean and validate
    items = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        structure = str(item.get("structure", str(len(items) + 1)))
        page = item.get("page")
        if page is not None:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = None
        items.append({
            "structure": structure,
            "title": str(item["title"]).strip()[:200],
            "page": page,
        })

    return items


def _calculate_page_offset(toc_items: List[dict], pages: List[tuple[int, str]], api_key: str, provider: str) -> int:
    """
    PageIndex-style: calculate offset between TOC page numbers and physical page indices.
    E.g., TOC says "page 1" but that's actually physical page 5 in the PDF.
    """
    # Find items with page numbers
    items_with_pages = [it for it in toc_items if it.get("page") is not None]
    if not items_with_pages:
        return 0

    # Check a few items to find the offset
    sample_items = items_with_pages[:5]
    page_texts = ""
    checked_pages = set()
    for it in sample_items:
        toc_page = it["page"]
        # Check nearby physical pages
        for offset_try in range(-5, 10):
            phys = toc_page + offset_try
            if phys < 1 or phys > len(pages) or phys in checked_pages:
                continue
            checked_pages.add(phys)
            idx = phys - 1  # 0-based index
            if idx < len(pages):
                page_texts += f"\n<page_{phys}>\n{pages[idx][1][:800]}\n</page_{phys}>\n"
            if len(checked_pages) > 25:
                break

    titles_str = "\n".join(f'- "{it["title"]}" (TOC says page {it["page"]})' for it in sample_items)
    prompt = f"""You are given section titles from a Table of Contents (with their listed page numbers)
and the actual text from several physical pages of the PDF.

Sections from TOC:
{titles_str}

Physical page texts:
{page_texts}

Find where these section titles actually appear in the physical pages.
Calculate the OFFSET: offset = physical_page - toc_page

Return JSON: {{"offset": <integer>}}

If the TOC page numbers match physical pages exactly, offset is 0.
If TOC says "page 1" but it's on physical page 5, offset is 4."""

    raw = _call_llm_tree(prompt, api_key, provider, max_tokens=500)
    if raw:
        parsed = _extract_json(raw)
        if isinstance(parsed, dict) and "offset" in parsed:
            try:
                return int(parsed["offset"])
            except (ValueError, TypeError):
                pass
    return 0


def _apply_offset_and_compute_ranges(toc_items: List[dict], offset: int, total_pages: int) -> List[dict]:
    """Apply page offset and compute start_index/end_index for each item."""
    for item in toc_items:
        if item.get("page") is not None:
            item["physical_index"] = item["page"] + offset
        else:
            item["physical_index"] = None

    # Forward-fill None physical indices
    last_valid = 1
    for item in toc_items:
        if item["physical_index"] is not None and item["physical_index"] >= 1:
            last_valid = item["physical_index"]
        else:
            item["physical_index"] = last_valid

    # Compute start_index and end_index
    for i, item in enumerate(toc_items):
        item["start_index"] = item["physical_index"]
        # end_index = next item's start - 1 (or total_pages)
        next_start = total_pages
        for j in range(i + 1, len(toc_items)):
            ns = toc_items[j].get("physical_index")
            if ns and ns > item["start_index"]:
                next_start = ns - 1
                break
        item["end_index"] = min(next_start, total_pages)
        if item["end_index"] < item["start_index"]:
            item["end_index"] = item["start_index"]

    return toc_items


def _list_to_tree(flat_items: List[dict]) -> List[dict]:
    """
    Convert flat list with "structure" field (1, 1.1, 1.2, 2, 2.1, ...) into nested tree.
    Same logic as PageIndex list_to_tree.
    """
    def get_parent_structure(structure: str) -> str | None:
        parts = structure.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    nodes = {}
    root_nodes = []

    for item in flat_items:
        structure = item.get("structure", "")
        node = {
            "title": item.get("title", ""),
            "node_id": item.get("node_id", ""),
            "start_index": item.get("start_index", 1),
            "end_index": item.get("end_index", 1),
            "summary": item.get("summary", item.get("title", "")),
            "nodes": [],
        }
        nodes[structure] = node

        parent = get_parent_structure(structure)
        if parent and parent in nodes:
            nodes[parent]["nodes"].append(node)
            # Expand parent's range to cover children
            if node["end_index"] > nodes[parent]["end_index"]:
                nodes[parent]["end_index"] = node["end_index"]
        else:
            root_nodes.append(node)

    # Clean empty nodes arrays
    def _clean(n):
        if not n["nodes"]:
            del n["nodes"]
        else:
            for child in n["nodes"]:
                _clean(child)
        return n

    return [_clean(n) for n in root_nodes]


def _assign_node_ids(tree: list | dict, counter: list | None = None) -> None:
    """Recursively assign node_ids like 0000, 0001, etc."""
    if counter is None:
        counter = [0]
    if isinstance(tree, list):
        for node in tree:
            _assign_node_ids(node, counter)
    elif isinstance(tree, dict):
        tree["node_id"] = str(counter[0]).zfill(4)
        counter[0] += 1
        if "nodes" in tree:
            _assign_node_ids(tree["nodes"], counter)


def _detect_sections_fallback(pages: List[tuple[int, str]]) -> List[dict]:
    """Fallback: regex-based section detection when no LLM. Returns flat list."""
    section_re = re.compile(
        r"(?:^\s*#{1,4}\s+(.+))"
        r"|(?:^\s*(?:Chapter|CHAPTER)\s+\d+[.:\s]*(.*))"
        r"|(?:^\s*(?:Section|SECTION)\s+\d+[.:\s]*(.*))"
        r"|(?:^\s*(\d{1,2}\.\d{0,2}\s+[A-Z].{3,}))"
        r"|(?:^([A-Z][A-Z\s&,\-]{8,})$)",
        re.MULTILINE,
    )
    boundaries = []
    for idx, (page_num, text) in enumerate(pages):
        first_800 = text[:800]
        for m in section_re.finditer(first_800):
            title = next((g for g in m.groups() if g), "").strip()
            if title and len(title) >= 3 and len(title) <= 150:
                boundaries.append((title, page_num, idx))
                break  # One per page

    nodes = []
    for i, (title, start_page, idx) in enumerate(boundaries):
        end_page = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else (pages[-1][0] if pages else start_page)
        if end_page < start_page:
            end_page = start_page
        nodes.append({
            "title": title,
            "start_index": start_page,
            "end_index": end_page,
            "summary": title,
            "node_id": str(len(nodes)).zfill(4),
        })

    if not nodes:
        chunk_size = max(5, len(pages) // 20)
        for i in range(0, len(pages), chunk_size):
            block = pages[i : i + chunk_size]
            if block:
                start_p, end_p = block[0][0], block[-1][0]
                nodes.append({
                    "node_id": str(len(nodes)).zfill(4),
                    "title": f"Pages {start_p}–{end_p}",
                    "start_index": start_p,
                    "end_index": end_p,
                    "summary": f"Content from pages {start_p} to {end_p}.",
                })
    return nodes


def _generate_structure_no_toc(
    pages: List[tuple[int, str]],
    api_key: str,
    provider: str,
) -> List[dict]:
    """
    For docs without TOC: process in chunks, extract hierarchical structure.
    Like PageIndex process_no_toc: generate_toc_init + generate_toc_continue.
    """
    chunk_size = PAGE_TREE_MAX_PAGES_PER_LLM
    all_items = []

    for start in range(0, len(pages), chunk_size):
        chunk = pages[start : start + chunk_size]
        content = ""
        for pnum, text in chunk:
            content += f"\n<page_{pnum}>\n{text[:2000]}\n</page_{pnum}>\n"

        if not all_items:
            prompt = f"""You are an expert at extracting hierarchical document structure.
Extract sections and sub-sections from this document text.

Use the "structure" field for hierarchy: "1", "1.1", "1.2", "2", "2.1", etc.
Use the <page_N> tags to determine physical page numbers.

Document:
{content}

Return JSON array:
[
  {{"structure": "1", "title": "...", "physical_index": <page_number>}},
  {{"structure": "1.1", "title": "...", "physical_index": <page_number>}},
  ...
]
Return ONLY the JSON array."""
        else:
            prev_json = json.dumps(all_items[-10:], indent=1)
            prompt = f"""Continue extracting the hierarchical document structure.
Previous sections (for context):
{prev_json}

New document pages:
{content}

Continue the structure numbering from where you left off.
Return JSON array of NEW sections only:
[{{"structure": "...", "title": "...", "physical_index": <page_number>}}, ...]
Return ONLY the JSON array."""

        raw = _call_llm_tree(prompt, api_key, provider, max_tokens=4000)
        if raw:
            parsed = _extract_json(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("title"):
                        pi = item.get("physical_index")
                        if isinstance(pi, str):
                            nums = re.findall(r"\d+", pi)
                            pi = int(nums[0]) if nums else None
                        elif isinstance(pi, (int, float)):
                            pi = int(pi)
                        all_items.append({
                            "structure": str(item.get("structure", str(len(all_items) + 1))),
                            "title": str(item["title"]).strip()[:200],
                            "page": pi,
                            "physical_index": pi,
                        })

    return all_items


def build_tree_from_pdf(
    path: str,
    pages: List[tuple[int, str]],
    api_key: str = "",
    provider: str = "openai",
) -> dict:
    """
    Build a hierarchical tree from PDF pages (PageIndex approach).
    1. Detect TOC pages → extract hierarchical TOC with structure numbers
    2. Calculate page offset (TOC page vs physical page)
    3. Convert flat structure to nested tree
    4. Fallback: chunk-based LLM extraction or regex
    """
    file_name = os.path.basename(path)
    if not pages:
        return {"file_name": file_name, "nodes": [], "page_count": 0, "file_path": path}

    total_pages = len(pages)
    tree_nodes = []

    # Step 1: Try TOC-based extraction (best for annual reports, long docs)
    if api_key and total_pages > 10:
        toc_indices = _detect_toc_pages(pages, api_key, provider)
        logger.info("TOC pages detected: %s for %s", toc_indices, file_name)

        if toc_indices:
            toc_items = _extract_toc_hierarchical(pages, toc_indices, api_key, provider)
            if toc_items:
                offset = _calculate_page_offset(toc_items, pages, api_key, provider)
                logger.info("Page offset: %d for %s", offset, file_name)
                toc_items = _apply_offset_and_compute_ranges(toc_items, offset, total_pages)
                tree_nodes = _list_to_tree(toc_items)
                _assign_node_ids(tree_nodes)
                logger.info("Tree from TOC: %s → %d top-level nodes (hierarchical)", file_name, len(tree_nodes))

    # Step 2: No TOC → chunk-based LLM extraction
    if not tree_nodes and api_key:
        flat_items = _generate_structure_no_toc(pages, api_key, provider)
        if flat_items:
            flat_items = _apply_offset_and_compute_ranges(flat_items, 0, total_pages)
            tree_nodes = _list_to_tree(flat_items)
            _assign_node_ids(tree_nodes)
            logger.info("Tree from LLM chunks: %s → %d top-level nodes", file_name, len(tree_nodes))

    # Step 3: Regex fallback
    if not tree_nodes:
        tree_nodes = _detect_sections_fallback(pages)
        logger.info("Tree from regex: %s → %d nodes", file_name, len(tree_nodes))

    # Step 4: Split any leaf node > MAX_PAGES_PER_NODE into sub-sections
    # (PageIndex enforces max 10 pages per node)
    if tree_nodes and api_key:
        tree_nodes = _split_large_leaves(tree_nodes, path, pages, api_key, provider)

    return {
        "file_name": file_name,
        "file_path": path,
        "nodes": tree_nodes,
        "page_count": total_pages,
    }


def _split_large_leaves(
    nodes: list, pdf_path: str, pages: list, api_key: str, provider: str,
    max_pages: int = PAGE_TREE_MAX_PAGES_PER_NODE,
) -> list:
    """
    PageIndex-style enforcement: any leaf node with > max_pages pages
    gets split into sub-sections using LLM analysis of its content.
    """
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            node["nodes"] = _split_large_leaves(node["nodes"], pdf_path, pages, api_key, provider, max_pages)
            continue
        start_p = node.get("start_index", 1)
        end_p = node.get("end_index", start_p)
        span = end_p - start_p + 1
        if span <= max_pages:
            continue
        # This leaf is too large — extract sub-sections
        logger.info("Splitting large leaf '%s' (%d pages) into sub-sections", node.get("title", "?"), span)
        sub_pages = _read_pdf_page_range(pdf_path, start_p, end_p)
        if not sub_pages:
            continue
        # Build page-tagged text (first 500 chars per page for efficiency)
        tagged_parts = []
        for pnum, ptxt in sub_pages:
            tagged_parts.append(f"<page_{pnum}>\n{ptxt[:500]}\n</page_{pnum}>")
        tagged_text = "\n".join(tagged_parts)
        if len(tagged_text) > 25000:
            tagged_text = tagged_text[:25000]

        prompt = f"""You are analyzing a document section to identify its sub-sections.

Section: "{node.get('title', '')}" (pages {start_p}-{end_p})

Below is the beginning of each page in this section. Identify the major sub-sections/topics within it.

{tagged_text}

Extract the sub-sections as JSON:
[
  {{"title": "Sub-section title", "start_page": <page_number>}},
  ...
]

INSTRUCTIONS:
- Look for headings, numbered sections (e.g. "7.c.i"), topic changes
- Each sub-section should be a distinct topic
- Include the page number where each sub-section starts
- Return at least 3 sub-sections for large sections
- Return ONLY valid JSON array"""

        raw = _call_llm_tree(prompt, api_key, provider, max_tokens=1500)
        if not raw:
            continue
        parsed = _extract_json(raw)
        if not isinstance(parsed, list) or len(parsed) < 2:
            continue

        sub_nodes = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict) or "title" not in item:
                continue
            s = int(item.get("start_page", start_p))
            e = int(parsed[i + 1].get("start_page", end_p)) - 1 if i + 1 < len(parsed) else end_p
            s = max(s, start_p)
            e = min(e, end_p)
            if e < s:
                continue
            sub_nodes.append({
                "title": item["title"],
                "start_index": s,
                "end_index": e,
                "summary": item["title"],
            })

        if sub_nodes:
            base = int(node.get("node_id", "0")) * 100
            _assign_node_ids(sub_nodes, counter=[base])
            node["nodes"] = sub_nodes
            logger.info("Split '%s' into %d sub-sections", node.get("title", "?"), len(sub_nodes))

    return nodes


# ─────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────

def save_tree(tree: dict, cache_dir: str | None = None) -> str:
    """Save tree to JSON."""
    cache_dir = cache_dir or PAGE_TREE_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    fname = tree.get("file_name", "unknown").replace(" ", "_").replace(".", "_")
    out_path = os.path.join(cache_dir, f"{fname}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    n_nodes = _count_nodes(tree.get("nodes", []))
    logger.info("Saved tree to %s (%d total nodes)", out_path, n_nodes)
    return out_path


def _count_nodes(nodes) -> int:
    """Count total nodes in tree (including nested)."""
    if isinstance(nodes, list):
        return sum(_count_nodes(n) for n in nodes)
    elif isinstance(nodes, dict):
        c = 1
        if "nodes" in nodes:
            c += _count_nodes(nodes["nodes"])
        return c
    return 0


def load_tree(file_name: str, cache_dir: str | None = None) -> dict | None:
    """Load tree by file_name."""
    cache_dir = cache_dir or PAGE_TREE_CACHE_DIR
    fname = file_name.replace(" ", "_").replace(".", "_")
    path = os.path.join(cache_dir, f"{fname}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load tree %s: %s", path, e)
    return None


def load_all_trees(cache_dir: str | None = None) -> List[dict]:
    """Load all tree JSON files from cache."""
    cache_dir = cache_dir or PAGE_TREE_CACHE_DIR
    if not os.path.isdir(cache_dir):
        return []
    trees = []
    for f in os.listdir(cache_dir):
        if f.endswith(".json"):
            try:
                with open(os.path.join(cache_dir, f), "r", encoding="utf-8") as fp:
                    trees.append(json.load(fp))
            except Exception as e:
                logger.warning("Failed to load tree %s: %s", f, e)
    return trees


# ─────────────────────────────────────────────────────────
# Tree Search: multi-step LLM reasoning over tree
# ─────────────────────────────────────────────────────────

def _tree_to_toc_string(nodes: list, indent: int = 0) -> str:
    """Convert tree to human-readable TOC string for LLM."""
    lines = []
    for n in nodes:
        prefix = "  " * indent
        page_range = f"p.{n.get('start_index', '?')}-{n.get('end_index', '?')}"
        nid = n.get("node_id", "?")
        lines.append(f"{prefix}- [{nid}] {n.get('title', '')} ({page_range})")
        if "nodes" in n and n["nodes"]:
            lines.append(_tree_to_toc_string(n["nodes"], indent + 1))
    return "\n".join(lines)


def _find_node_by_id(nodes: list, node_id: str) -> dict | None:
    """Find node by node_id in tree."""
    for n in nodes:
        if str(n.get("node_id", "")) == str(node_id) or str(n.get("node_id", "")) == str(node_id).zfill(4):
            return n
        if "nodes" in n:
            found = _find_node_by_id(n["nodes"], node_id)
            if found:
                return found
    return None


def _get_page_text(tree: dict, start_p: int, end_p: int, data_folder: str) -> str:
    """Extract page text for a page range from the PDF file."""
    file_path = tree.get("file_path")
    fname = tree.get("file_name", "")
    if not file_path or not os.path.isfile(file_path):
        file_path = os.path.join(data_folder, fname)
    if not os.path.isfile(file_path):
        # Try uploads subfolder
        file_path = os.path.join(data_folder, "uploads", fname)
    if not os.path.isfile(file_path) or not file_path.lower().endswith(".pdf"):
        return ""
    try:
        from document_loader import extract_text_from_pdf
        pages = extract_text_from_pdf(file_path)
        parts = [t for p, t in pages if start_p <= p <= end_p]
        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning("Failed to extract pages %d-%d from %s: %s", start_p, end_p, fname, e)
    return ""


_pdf_cache: dict[str, list[tuple[int, str]]] = {}


def _get_page_text_cached(tree: dict, start_p: int, end_p: int, data_folder: str) -> str:
    """Extract page text with caching to avoid re-reading the same PDF."""
    file_path = tree.get("file_path")
    fname = tree.get("file_name", "")
    if not file_path or not os.path.isfile(file_path):
        file_path = os.path.join(data_folder, fname)
    if not os.path.isfile(file_path):
        file_path = os.path.join(data_folder, "uploads", fname)
    if not os.path.isfile(file_path) or not file_path.lower().endswith(".pdf"):
        return ""

    if file_path not in _pdf_cache:
        try:
            from document_loader import extract_text_from_pdf
            _pdf_cache[file_path] = extract_text_from_pdf(file_path)
        except Exception as e:
            logger.warning("Failed to extract PDF %s: %s", fname, e)
            return ""

    pages = _pdf_cache[file_path]
    parts = [t for p, t in pages if start_p <= p <= end_p]
    return "\n\n".join(parts) if parts else ""


def _keyword_scan_pdf(file_path: str, query: str, start_p: int, end_p: int) -> list[tuple[int, str, int]]:
    """
    Fast keyword scan: search pages for query terms using fitz.
    Returns [(page_num, page_text, match_count), ...] for pages with matches.
    """
    try:
        import fitz
    except ImportError:
        return []

    q_lower = query.lower().strip()
    # Build search terms: full query + individual meaningful words
    terms = [q_lower]
    words = [w for w in re.findall(r'[a-z]+', q_lower) if len(w) >= 4]
    terms.extend(words)

    hits = []
    try:
        doc = fitz.open(file_path)
        for pg_idx in range(max(0, start_p - 1), min(end_p, len(doc))):
            text = doc.load_page(pg_idx).get_text()
            text_lower = text.lower()
            count = 0
            for term in terms:
                if term in text_lower:
                    count += 1
            if count > 0:
                hits.append((pg_idx + 1, text, count))
        doc.close()
    except Exception as e:
        logger.warning("keyword scan failed: %s", e)
    # Sort by match count descending
    hits.sort(key=lambda x: -x[2])
    return hits


def _collect_all_leaf_ranges(nodes: list) -> list[dict]:
    """Collect all leaf nodes (deepest sections) with their page ranges."""
    leaves = []
    for n in nodes:
        children = n.get("nodes", [])
        if children:
            leaves.extend(_collect_all_leaf_ranges(children))
        else:
            leaves.append(n)
    return leaves


def tree_keyword_retrieve(
    query: str,
    trees: List[dict],
    data_folder: str | None = None,
    top_k: int = 15,
    allowed_files: set[str] | None = None,
) -> list[tuple["TreeChunk", float]]:
    """
    Fast, LLM-free keyword retrieval from tree-indexed PDFs.
    Uses fitz to scan pages for query terms and returns (TreeChunk, score) pairs
    compatible with text_results in the agentic RAG fusion pipeline.

    `allowed_files`: when provided (e.g. the patient's known files, or the
    single file the user clicked on), unrelated trees are skipped *before* the
    expensive page scan. This is a major speed win for targeted queries like
    "Alyson Jude's activity restrictions" — only her files are touched.
    """
    if not trees:
        return []

    data_folder = data_folder or DATA_FOLDER
    q_lower = query.lower().strip()
    q_words = [w for w in re.findall(r'[a-z]+', q_lower) if len(w) >= 4]

    all_hits: list[tuple[TreeChunk, float]] = []
    max_raw_score = 1  # track for normalization

    for tree in trees:
        # Upfront scoping: skip trees not in the allowed set before any I/O.
        if allowed_files is not None and tree.get("file_name", "") not in allowed_files:
            continue
        file_path = _resolve_pdf_path(tree, data_folder)
        if not file_path:
            continue
        pages = get_pdf_page_texts(file_path)
        if not pages:
            continue
        page_scores: list[tuple[int, str, int]] = []
        for pg_idx, text in enumerate(pages):
            text_lower = text.lower()
            score = 0
            if q_lower in text_lower:
                score += 10
            for w in q_words:
                if w in text_lower:
                    score += 1
            if score > 0:
                page_scores.append((pg_idx + 1, text, score))
                if score > max_raw_score:
                    max_raw_score = score

        page_scores.sort(key=lambda x: -x[2])

        leaves = _collect_all_leaf_ranges(tree.get("nodes", []))

        def _section_for_page(pg: int) -> str:
            for lf in leaves:
                if lf.get("start_index", 0) <= pg <= lf.get("end_index", 0):
                    return lf.get("title", "")
            return ""

        for pg_num, pg_text, raw_score in page_scores[:top_k]:
            tc = TreeChunk(
                chunk_id=f"tree_kw_{tree.get('file_name', '')}_{pg_num}",
                text=pg_text[:12000],
                file_name=tree.get("file_name", ""),
                page_number=pg_num,
            )
            tc._section_title = _section_for_page(pg_num)
            normalized = min(1.0, raw_score / max(max_raw_score, 1))
            all_hits.append((tc, normalized))

    all_hits.sort(key=lambda x: -x[1])
    return all_hits[:top_k]


def build_summary_prompt_and_sources(
    query: str,
    fused_results: list[dict],
) -> tuple[str, list[dict]]:
    """Build the summary LLM prompt and the ordered citation sources.

    Pure function — no LLM call. Used by both the streaming and non-streaming
    summary helpers so the UI can show citation chips before tokens arrive.
    """
    top_items = fused_results[:8]
    if not top_items:
        return "", []

    def _normalize_for_summary(text: str) -> str:
        """
        Normalize OCR/markdown table noise so the LLM can read form rows reliably.
        Converts pipe-heavy markdown rows to plain text and trims separator clutter.
        """
        if not text:
            return ""
        t = text
        # Collapse markdown table separators (| --- | --- |) which add noise.
        t = re.sub(r"^\s*\|?\s*(?:-+\s*\|)+\s*-+\s*\|?\s*$", "", t, flags=re.MULTILINE)
        # Replace remaining table pipes with spaced delimiters.
        t = t.replace("|", " ; ")
        # Normalize repeated punctuation/separators from OCR output.
        t = re.sub(r"[;]{2,}", ";", t)
        t = re.sub(r"\s{2,}", " ", t)
        # Keep line structure for section/header detection.
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    content_parts = []
    sources = []
    seen_sources: dict[tuple, int] = {}  # (fname, pg) -> 1-based index
    for r in top_items:
        content = r["content"]
        if r.get("type") == "image" and isinstance(content, dict):
            fname = content.get("file_name", "")
            pg = content.get("page", "?")
            ocr = content.get("ocr_text", "") or ""
            page_text = ""
            if fname and str(pg).isdigit() and fname.lower().endswith(".pdf"):
                img_path = content.get("path", "")
                pdf_dir = os.path.dirname(img_path) if img_path else ""
                pdf_path = os.path.join(pdf_dir, fname) if pdf_dir else ""
                if not pdf_path or not os.path.isfile(pdf_path):
                    pdf_path = os.path.join(DATA_FOLDER, fname)
                if not os.path.isfile(pdf_path):
                    pdf_path = os.path.join(DATA_FOLDER, "uploads", fname)
                if os.path.isfile(pdf_path):
                    nearby = _read_pdf_page_range(pdf_path, int(pg), int(pg))
                    page_text = nearby[0][1][:2500] if nearby else ""
            txt = page_text or ocr[:3000] or f"[Image from {fname} page {pg}]"
            txt = _normalize_for_summary(txt)
            src_key = (fname, str(pg))
            if src_key not in seen_sources:
                seen_sources[src_key] = len(sources) + 1
                sources.append({"file_name": fname, "page": pg, "title": fname})
            idx = seen_sources[src_key]
            content_parts.append(f"[Source {idx}: {fname} p.{pg}]\n{txt}")
        else:
            fname = getattr(content, "file_name", "") or ""
            pg = getattr(content, "page_number", None) or "?"
            txt = (getattr(content, "text", "") or "")[:3000]
            txt = _normalize_for_summary(txt)
            src_key = (fname, str(pg))
            if src_key not in seen_sources:
                seen_sources[src_key] = len(sources) + 1
                section = getattr(content, "_section_title", "") or ""
                sources.append({
                    "file_name": fname,
                    "page": pg,
                    "title": section or fname,
                })
            idx = seen_sources[src_key]
            content_parts.append(f"[Source {idx}: {fname} p.{pg}]\n{txt}")

    summary_input = "\n\n---\n\n".join(content_parts)
    if len(summary_input) > 20000:
        summary_input = summary_input[:20000]

    # Build numbered source list for citation instructions
    source_nums = "\n".join(f"- [{i+1}] {s['file_name']} (page {s['page']})" for i, s in enumerate(sources))

    prompt = f"""Based on the following document excerpts, answer the user's question with a clear, well-structured summary.

Question: "{query}"

Document excerpts (each prefixed with its source number):
{summary_input}

Source reference (use these numbers for citations):
{source_nums}

INSTRUCTIONS:
- Write a clear, structured answer using ONLY information from the excerpts
- Use markdown headers (##) for major topics
- Use bullet points for key details
- Include specific data, numbers, and facts
- If content is in a non-English language, summarize the key points in English
- PRIORITIZE excerpts that contain section titles or lines matching the query terms.
- For OCR/form tables, interpret delimiters (';', '*', broken spacing) and extract concrete values (job title, employer, dates, salary, duties) when present.
- DO NOT say "not available" or "not provided" if any excerpt contains partial relevant details; report available details and clearly mark unknown fields as "not visible in excerpt".
- CRITICAL: Add citation markers at the end of sentences or facts that come from a source. Use [1], [2], [3], etc. to match the source numbers above. Example: "Shri D. Surendran holds an MBA [1]." For facts from multiple sources use [1,2]. Add citations so users can verify every claim.
- Be concise but comprehensive"""

    return prompt, sources


def generate_summary_from_results(
    query: str,
    fused_results: list[dict],
    api_key: str,
    provider: str = "openai",
    max_tokens: int = 2000,
) -> tuple[str, list[dict]]:
    """Generate a PageIndex-style markdown summary from fused results.

    Non-streaming path (used by legacy `/api/chat`).
    """
    prompt, sources = build_summary_prompt_and_sources(query, fused_results)
    if not prompt:
        return "", []
    summary = _call_llm_tree(prompt, api_key, provider, max_tokens=max_tokens) or ""
    return summary, sources


def stream_summary_from_results(
    query: str,
    fused_results: list[dict],
    api_key: str,
    provider: str = "openai",
):
    """Stream summary tokens. Yields str deltas; caller accumulates the final text.

    Sources are already known before streaming; get them from
    `build_summary_prompt_and_sources` first if the UI needs them up front.
    """
    prompt, _ = build_summary_prompt_and_sources(query, fused_results)
    if not prompt:
        return
    from retrieval.agentic_rag import _call_llm_stream
    for delta in _call_llm_stream(prompt, api_key, provider, max_tokens=2000):
        if delta:
            yield delta


def tree_search(
    query: str,
    trees: List[dict],
    api_key: str = "",
    provider: str = "openai",
    data_folder: str | None = None,
) -> dict:
    """
    PageIndex-style tree search: keyword scan + LLM summarization.

    Returns dict with chunks, summary, and sources.

    Approach (inspired by real PageIndex):
    1. KEYWORD SCAN: fast fitz-based search across all sections for query terms
       — this finds the exact pages (e.g., page 429 with "Qualitative Disclosures")
    2. LLM FALLBACK: if keyword scan finds nothing, use LLM to pick sections
    3. LLM SUMMARY: generate structured answer from the found pages
    """
    _empty = {"chunks": [], "summary": "", "sources": []}
    if not trees or not api_key:
        return _empty

    data_folder = data_folder or DATA_FOLDER

    results = []
    seen_pages = set()

    # ── Step 1: KEYWORD SCAN — single pass through each PDF ──
    # Uses fitz (C-based, very fast) to search every page for query terms.
    # Scoring: full phrase match = 10 points, each word = 1 point.
    # This ensures pages with the exact phrase rank highest.
    q_lower = query.lower().strip()
    q_words = [w for w in re.findall(r'[a-z]+', q_lower) if len(w) >= 4]

    for tree in trees:
        file_path = _resolve_pdf_path(tree, data_folder)
        if not file_path:
            continue
        try:
            import fitz
            doc = fitz.open(file_path)
            page_hits = []
            for pg_idx in range(len(doc)):
                text = doc.load_page(pg_idx).get_text()
                text_lower = text.lower()
                score = 0
                # Full phrase match = 10 points (highest priority)
                if q_lower in text_lower:
                    score += 10
                # Each individual word match = 1 point
                for w in q_words:
                    if w in text_lower:
                        score += 1
                if score > 0:
                    page_hits.append((pg_idx + 1, text, score))
            doc.close()
        except Exception as e:
            logger.warning("keyword scan failed for %s: %s", file_path, e)
            continue

        # Sort by score — full phrase matches first, then by word count
        page_hits.sort(key=lambda x: -x[2])

        # Find which section each hit page belongs to
        leaves = _collect_all_leaf_ranges(tree.get("nodes", []))
        def _section_for_page(pg: int) -> str:
            for lf in leaves:
                if lf.get("start_index", 0) <= pg <= lf.get("end_index", 0):
                    return lf.get("title", "")
            return ""

        for pg_num, pg_text, match_count in page_hits[:5]:
            page_key = (tree.get("file_name", ""), pg_num)
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            # Read surrounding pages for context
            nearby = _read_pdf_page_range(file_path, max(1, pg_num - 1), pg_num + 1)
            full_text = "\n\n".join(t for _, t in nearby) if nearby else pg_text
            tc = TreeChunk(
                chunk_id=f"tree_kw_{tree.get('file_name', '')}_{pg_num}",
                text=full_text[:12000],
                file_name=tree.get("file_name", ""),
                page_number=pg_num,
            )
            tc._section_title = _section_for_page(pg_num)
            # Full phrase match (score>=10) gets highest ranking
            final_score = 1.0 if match_count >= 10 else min(0.9, 0.5 + match_count * 0.1)
            results.append((tc, final_score))

    # Sort by score descending, keep top 5
    results.sort(key=lambda x: -x[1])
    results = results[:5]

    # ── Step 2: LLM FALLBACK if keyword scan found nothing ──
    if not results:
        tree_tocs = []
        for t in trees:
            nodes = t.get("nodes", [])
            if not nodes:
                continue
            toc_str = _tree_to_toc_string(nodes)
            tree_tocs.append(f"Document: {t.get('file_name', '?')} ({t.get('page_count', '?')} pages)\n{toc_str}")

        if tree_tocs:
            full_toc = "\n\n".join(tree_tocs)
            prompt = f"""You are a document retrieval expert. Given a user question and a table of contents, identify which section(s) most likely contain the answer.

User question: "{query}"

Document tree:
{full_toc}

INSTRUCTIONS:
1. Reason about WHERE in the document this topic would appear.
2. Pick the DEEPEST matching node — child over parent.
3. Return up to 3 selections.

Return JSON:
{{"selections": [{{"file_name": "...", "node_id": "...", "title": "...", "reason": "..."}}]}}
Return ONLY valid JSON."""

            raw = _call_llm_tree(prompt, api_key, provider, max_tokens=800)
            parsed = _extract_json(raw) if raw else None
            if isinstance(parsed, list):
                parsed = {"selections": parsed}
            if isinstance(parsed, dict) and "selections" in parsed:
                for sel in parsed["selections"][:3]:
                    if not isinstance(sel, dict):
                        continue
                    node_id = sel.get("node_id", "")
                    fname = sel.get("file_name", "")
                    tree = None
                    for t in trees:
                        tname = t.get("file_name", "")
                        if fname and (fname == tname or fname.replace(" ", "_").lower() in tname.replace(" ", "_").lower()):
                            tree = t
                            break
                    if not tree:
                        for t in trees:
                            if _find_node_by_id(t.get("nodes", []), node_id):
                                tree = t
                                break
                    if not tree:
                        continue
                    node = _find_node_by_id(tree.get("nodes", []), node_id)
                    if not node:
                        title_lower = sel.get("title", "").lower()
                        if title_lower:
                            node = _find_node_by_title(tree.get("nodes", []), title_lower)
                    if not node:
                        continue
                    s = node.get("start_index", 1)
                    e = node.get("end_index", s)
                    if e - s > 5:
                        drill = _drill_into_section(query, tree, node, api_key, provider, data_folder)
                        if drill:
                            results.extend(drill)
                            continue
                    fp = _resolve_pdf_path(tree, data_folder)
                    pages = _read_pdf_page_range(fp, s, e) if fp else []
                    text = "\n\n".join(t for _, t in pages) if pages else node.get("title", "")
                    tc = TreeChunk(
                        chunk_id=f"tree_llm_{tree.get('file_name', '')}_{node_id}",
                        text=text[:12000],
                        file_name=tree.get("file_name", ""),
                        page_number=s,
                    )
                    tc._section_title = node.get("title", "")
                    results.append((tc, 0.8 - len(results) * 0.05))

    if not results:
        return _empty

    # ── Step 3: Build source citations ──
    sources = []
    for tc, _score in results:
        sources.append({
            "file_name": tc.file_name,
            "page": tc.page_number,
            "title": getattr(tc, "_section_title", "") or tc.file_name,
        })

    # ── Step 4: LLM summarizes the found pages ──
    content_for_summary = []
    for tc, _score in results[:5]:
        pg = tc.page_number or "?"
        txt = (tc.text or "")[:3000]
        content_for_summary.append(f"[Source: {tc.file_name} p.{pg}]\n{txt}")

    summary_input = "\n\n---\n\n".join(content_for_summary)
    if len(summary_input) > 20000:
        summary_input = summary_input[:20000]

    summary_prompt = f"""Based on the following document excerpts, answer the user's question with a clear, well-structured summary.

Question: "{query}"

Document excerpts:
{summary_input}

INSTRUCTIONS:
- Write a clear, structured answer using ONLY information from the excerpts
- Use markdown headers (##) for major topics
- Use bullet points for key details
- Include specific data, numbers, and facts
- If content is in a non-English language, summarize the key points in English
- Be concise but comprehensive"""

    summary = _call_llm_tree(summary_prompt, api_key, provider, max_tokens=2000) or ""

    return {"chunks": results, "summary": summary, "sources": sources}


def _find_node_by_title(nodes: list, title_lower: str) -> dict | None:
    """Find node by fuzzy title match."""
    best = None
    best_score = 0
    for n in nodes:
        n_title = n.get("title", "").lower()
        if title_lower in n_title or n_title in title_lower:
            score = len(n_title)
            if score > best_score:
                best = n
                best_score = score
        if "nodes" in n:
            found = _find_node_by_title(n["nodes"], title_lower)
            if found:
                f_title = found.get("title", "").lower()
                score = len(f_title)
                if score > best_score:
                    best = found
                    best_score = score
    return best


def _resolve_pdf_path(tree: dict, data_folder: str) -> str | None:
    """Resolve the actual file path for a tree's PDF."""
    file_path = tree.get("file_path", "")
    fname = tree.get("file_name", "")
    if file_path and os.path.isfile(file_path):
        return file_path
    fp = os.path.join(data_folder, fname)
    if os.path.isfile(fp):
        return fp
    fp = os.path.join(data_folder, "uploads", fname)
    if os.path.isfile(fp):
        return fp
    return None


def _read_pdf_page_range(file_path: str, start_p: int, end_p: int) -> list[tuple[int, str]]:
    """Read only specific pages from a PDF using PyMuPDF (fast, no full-PDF load)."""
    try:
        import fitz
        doc = fitz.open(file_path)
        pages = []
        for pg_idx in range(max(0, start_p - 1), min(end_p, len(doc))):
            text = doc.load_page(pg_idx).get_text()
            pages.append((pg_idx + 1, text))
        doc.close()
        return pages
    except Exception as e:
        logger.warning("fitz page read failed for %s: %s", file_path, e)
        return []


def _get_pdf_pages(tree: dict, data_folder: str) -> list[tuple[int, str]]:
    """Get cached PDF pages for a tree (reads ALL pages — use sparingly)."""
    file_path = _resolve_pdf_path(tree, data_folder)
    if not file_path:
        return []
    if file_path not in _pdf_cache:
        try:
            from document_loader import extract_text_from_pdf
            _pdf_cache[file_path] = extract_text_from_pdf(file_path)
        except Exception as e:
            logger.warning("Failed to extract PDF %s: %s", file_path, e)
            return []
    return _pdf_cache[file_path]


def _drill_into_section(
    query: str,
    tree: dict,
    node: dict,
    api_key: str,
    provider: str,
    data_folder: str,
) -> list[tuple] | None:
    """
    PageIndex-style multi-step drill-down: when a section spans many pages
    with no sub-nodes, build a compact page-level index covering ALL pages,
    then ask the LLM to pick the specific pages that answer the query.
    """
    start_p = node.get("start_index", 1)
    end_p = node.get("end_index", start_p)

    file_path = _resolve_pdf_path(tree, data_folder)
    if not file_path:
        return None

    # Read ONLY the section's pages (not the full 560-page PDF)
    section_pages = _read_pdf_page_range(file_path, start_p, end_p)
    if not section_pages:
        return None

    # Build compact page-level index: first ~250 chars per page
    page_previews = []
    for pnum, ptxt in section_pages:
        preview = ptxt[:300].replace('\n', ' ').strip()
        page_previews.append(f"Page {pnum}: {preview}")

    if not page_previews:
        return None

    page_index = "\n".join(page_previews)
    # Cap to ~25k chars for LLM context safety
    if len(page_index) > 25000:
        page_index = page_index[:25000]

    prompt = f"""You are drilling down into a document section to find specific pages relevant to a question.

Question: "{query}"

Section: "{node.get('title', '')}" (pages {start_p}-{end_p})

Below is a compact index showing the first ~250 characters of EVERY page in this section. Use these previews to identify which pages contain content relevant to the question.

{page_index}

INSTRUCTIONS:
- Look for pages whose preview text mentions the topic: "{query}"
- Pick the 1-5 most relevant page numbers
- Be precise — pick the exact pages, not surrounding ones

Return JSON: {{"pages": [<page_number>, ...], "reason": "brief explanation"}}
Return ONLY valid JSON."""

    raw = _call_llm_tree(prompt, api_key, provider, max_tokens=500)
    if not raw:
        return None
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict) or "pages" not in parsed:
        return None

    results = []
    selected_pages = parsed.get("pages", [])[:5]
    if not selected_pages:
        return None

    for pg in selected_pages:
        try:
            pg = int(pg)
        except (ValueError, TypeError):
            continue
        if pg < start_p or pg > end_p:
            continue
        # Get 2-3 pages around the target for context
        p_start = max(start_p, pg - 1)
        p_end = min(end_p, pg + 1)
        nearby = _read_pdf_page_range(file_path, p_start, p_end)
        page_text = "\n\n".join(t for _, t in nearby) if nearby else ""
        if not page_text:
            continue

        metadata = {}
        try:
            from document_loader import extract_chunk_metadata
            metadata = extract_chunk_metadata(page_text[:3000])
        except Exception:
            pass

        chunk_id = f"tree_{tree.get('file_name', '')}_{node.get('node_id', '')}_{pg}"
        tc = TreeChunk(
            chunk_id=chunk_id,
            text=page_text[:12000],
            file_name=tree.get("file_name", ""),
            page_number=pg,
            patient_name=metadata.get("patient_name", ""),
        )
        tc._section_title = node.get("title", "")
        results.append((tc, 1.0 - len(results) * 0.05))

    return results if results else None


# ─────────────────────────────────────────────────────────
# Build trees for folder
# ─────────────────────────────────────────────────────────

def build_trees_for_folder(
    folder: str,
    indexing_mode: str,
    api_key: str = "",
    provider: str = "openai",
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> tuple[List[dict], int]:
    """
    Build trees for all PDFs in folder when indexing_mode is "tree" or "both".
    Returns (list of trees, count of PDFs processed).
    """
    from document_loader import extract_text_from_pdf

    folder = folder or DATA_FOLDER
    if not os.path.isdir(folder):
        return [], 0

    # Count PDFs first for progress
    pdf_paths = []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(".pdf"):
                pdf_paths.append((root, name))

    trees = []
    count = 0
    total = len(pdf_paths)
    for i, (root, name) in enumerate(pdf_paths):
        path = os.path.join(root, name)
        try:
            pdf_mtime = os.path.getmtime(path)
            fname_key = name.replace(" ", "_").replace(".", "_")
            out_json = os.path.join(PAGE_TREE_CACHE_DIR, f"{fname_key}.json")
            # Skip LLM tree rebuild if cache exists and is newer than the PDF file
            if os.path.isfile(out_json) and os.path.getmtime(out_json) + 0.5 >= pdf_mtime:
                cached = load_tree(name)
                if cached and cached.get("nodes"):
                    if progress_callback and total > 0:
                        pct = 70 + int(28 * (i + 1) / total)
                        progress_callback(pct, f"Tree (cached): {name}")
                    trees.append(cached)
                    count += 1
                    logger.debug("Using cached page tree for %s", name)
                    continue

            if progress_callback and total > 0:
                pct = 70 + int(28 * (i + 1) / total)
                progress_callback(pct, f"Building tree {i + 1}/{total}: {name}")
            pages = extract_text_from_pdf(path)
            if not pages:
                continue
            tree = build_tree_from_pdf(path, pages, api_key, provider)
            save_tree(tree)
            trees.append(tree)
            count += 1
            n_total = _count_nodes(tree.get("nodes", []))
            logger.info("Built tree for %s: %d total nodes", name, n_total)
        except Exception as e:
            logger.warning("Tree build failed for %s: %s", name, e)
    return trees, count
