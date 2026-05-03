"""
Agentic RAG: truly autonomous retrieval powered by LLM reasoning.

The LLM is the BRAIN — not a helper, not a classifier.
It understands ANY query (like a human would), generates multiple search
strategies, evaluates results, and retries if needed.

No hardcoded keyword patterns for query types.
The LLM decides everything; rule-based is fallback only when no API key.
"""
import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, List, Tuple, Optional

from document_loader import extract_chunk_metadata

# OCR often misreads names: "Rita Pepper" -> "Rika Popper", "Rita Peyer". Merge similar names.
_PATIENT_NAME_SIMILARITY_THRESHOLD = 0.82

# Detect filenames in queries so we can filter retrieval to that file (Chroma metadata file_name).
# Supports uploads like "TEE_TBrown (1).pdf" — single-token stem + optional "(n)" suffix.
_FILENAME_PATTERN = re.compile(
    r"\b([A-Za-z0-9_\-]+(?:\s*\(\d+\))?\.(?:png|jpg|jpeg|gif|bmp|tiff|tif|pdf|webp|md|txt|json))\b",
    re.IGNORECASE,
)
# Separate pattern: spaced basenames ("Safety test.pdf"). Two words; first word >=3 chars avoids
# matching "at APS_TBrown.pdf" over the real single-token filename when both match.
_FILENAME_SPACE_PATTERN = re.compile(
    r"\b([\w\-]{3,}\s+[\w\-]{2,}\.(?:png|jpg|jpeg|gif|bmp|tiff|tif|pdf|webp|md|txt|json))\b",
    re.IGNORECASE,
)

# Paths like data/uploads/TEE_TBrown (1).md — same basename rules as _FILENAME_PATTERN after a slash
_PATH_AFTER_SLASH = re.compile(
    r"[/\\]([A-Za-z0-9_\-]+(?:\s*\(\d+\))?\.(?:png|jpg|jpeg|gif|bmp|tiff|tif|pdf|webp|md|txt|json))\b",
    re.IGNORECASE,
)
# @data/uploads/TEE_TBrown (1).md — allow spaces in path before extension
_AT_PATH_FILE = re.compile(
    r"@([^@\n]+\.(?:png|jpg|jpeg|gif|bmp|tiff|tif|pdf|webp|md|txt|json))\b",
    re.IGNORECASE,
)


def _extract_query_filename(query: str) -> Optional[str]:
    """
    Infer which uploaded file (Chroma file_name) the user refers to.
    Collects all plausible filenames, normalizes to basename, picks the most specific
    (longest basename wins — avoids first-match bugs like report.md vs TEE_TBrown (1).md).
    """
    if not (query and query.strip()):
        return None
    cands: list[str] = []

    for m in _FILENAME_PATTERN.finditer(query):
        cands.append(m.group(1).strip())

    for m in _FILENAME_SPACE_PATTERN.finditer(query):
        cands.append(m.group(1).strip())

    for m in _PATH_AFTER_SLASH.finditer(query):
        cands.append(m.group(1).strip())

    for m in _AT_PATH_FILE.finditer(query):
        cands.append(os.path.basename(m.group(1).strip()))

    seen_lower: set[str] = set()
    uniq: list[str] = []
    for c in cands:
        base = os.path.basename(c.strip().strip('"').strip("'"))
        if not base or base.lower() in seen_lower:
            continue
        seen_lower.add(base.lower())
        uniq.append(base)

    if not uniq:
        return None

    def score(name: str) -> tuple:
        pos = query.lower().rfind(name.lower())
        if pos < 0:
            pos = 0
        return (len(name), pos)

    return max(uniq, key=score)


# ──────────────────────────────────────────────────────────────
# Catalog & metadata helpers
# ──────────────────────────────────────────────────────────────

_METADATA_VALIDATORS = {
    "claim_number": re.compile(r"^[\d][\w\-]+$|^\d{3,}"),
    "policy_number": re.compile(r"^\d[\d\w\-]*$"),
    "group_number": re.compile(r"^\d[\d\w\-]*$"),
}

_GARBAGE_VALUES = frozenset({
    "form", "containing", "denial", "information", "services", "type",
    "and", "cancellation", "documents", "the", "for", "continuously",
    "or", "any", "this", "that", "with", "from", "not", "are", "was",
    "been", "have", "has", "will", "would", "should", "could", "can",
})


def _normalize_name(name: str) -> str:
    if not name:
        return name
    return " ".join(w.capitalize() for w in name.strip().split())


def _is_valid_metadata(key: str, value: str) -> bool:
    if not value or len(value) < 2:
        return False
    if value.lower() in _GARBAGE_VALUES or key.startswith("_"):
        return False
    validator = _METADATA_VALIDATORS.get(key)
    if validator and not validator.match(value):
        return False
    return True


def _name_similarity(a: str, b: str) -> float:
    """Jaro-Winkler-like: high for OCR variants (Rika Popper vs Rita Pepper)."""
    if not a or not b:
        return 0.0
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _merge_similar_patient_names(
    patient_counts: dict[str, int],
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Merge OCR variants (Rika Popper, Rita Peyer -> Rita Pepper).
    Returns (canonical_list, alias_map) where alias_map[canonical] = [all variants including canonical].
    """
    names = list(patient_counts.keys())
    if not names:
        return [], {}

    # Union-find: group similar names
    parent: dict[str, str] = {n: n for n in names}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa == pb:
            return
        # Merge into the one with higher count (likely correct spelling)
        ca, cb = patient_counts.get(pa, 0), patient_counts.get(pb, 0)
        if ca >= cb:
            parent[pb] = pa
        else:
            parent[pa] = pb

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if _name_similarity(a, b) >= _PATIENT_NAME_SIMILARITY_THRESHOLD:
                union(a, b)

    # Build canonical (rep with max count per group) and alias map
    groups: dict[str, list[str]] = {}
    for n in names:
        rep = find(n)
        if rep not in groups:
            groups[rep] = []
        groups[rep].append(n)

    canonical_list = []
    alias_map: dict[str, list[str]] = {}
    for rep, members in groups.items():
        # Canonical = member with highest chunk count
        best = max(members, key=lambda m: patient_counts.get(m, 0))
        canonical_list.append(best)
        alias_map[best] = sorted(set(members))

    return sorted(canonical_list), alias_map


def _build_variant_to_canonical(alias_map: dict[str, list[str]]) -> dict[str, str]:
    """Map each variant (including canonical) to its canonical form."""
    out: dict[str, str] = {}
    for canonical, variants in alias_map.items():
        for v in variants:
            if v:
                out[v] = canonical
    return out


def normalize_patient_names_in_chunks(chunks: list) -> None:
    """
    At index time: rewrite chunk.patient_name to canonical form.
    Prevents OCR variants (Rika Popper, Rita Peyer) from being stored as separate patients.
    Modifies chunks in place.
    """
    import logging
    _log = logging.getLogger(__name__)
    if not chunks:
        return
    patient_counts: dict[str, int] = {}
    for c in chunks or []:
        p = getattr(c, "patient_name", "") or ""
        if not p:
            text = getattr(c, "text", "") or ""
            if text:
                p = extract_chunk_metadata(text).get("patient_name", "")
        if p:
            pn = _normalize_name(p)
            patient_counts[pn] = patient_counts.get(pn, 0) + 1
    _, alias_map = _merge_similar_patient_names(patient_counts)
    variant_to_canonical = _build_variant_to_canonical(alias_map)
    n_rewritten = 0
    for c in chunks:
        p = getattr(c, "patient_name", "") or ""
        if not p:
            text = getattr(c, "text", "") or ""
            if text:
                p = extract_chunk_metadata(text).get("patient_name", "")
        if p:
            pn = _normalize_name(p)
            canonical = variant_to_canonical.get(pn, pn)
            if canonical != pn:
                n_rewritten += 1
            c.patient_name = canonical
    if n_rewritten:
        _log.info("Normalized %d chunk(s) patient_name (OCR variants -> canonical)", n_rewritten)


def normalize_patient_names_in_items(
    items: list[dict],
    existing_patient_names: list[str] | None = None,
) -> None:
    """
    At index time: rewrite item['patient_name'] to canonical form (for image index).
    existing_patient_names: from Chroma so we merge new OCR variants with already-indexed names.
    Modifies items in place.
    """
    patient_counts: dict[str, int] = {}
    for p in existing_patient_names or []:
        pn = _normalize_name(p or "")
        if pn:
            patient_counts[pn] = patient_counts.get(pn, 0) + 10  # weight existing
    for x in items or []:
        p = _normalize_name(x.get("patient_name", "") or "")
        if p:
            patient_counts[p] = patient_counts.get(p, 0) + 1
    if not patient_counts:
        return
    _, alias_map = _merge_similar_patient_names(patient_counts)
    variant_to_canonical = _build_variant_to_canonical(alias_map)
    for x in items or []:
        p = _normalize_name(x.get("patient_name", "") or "")
        if p:
            x["patient_name"] = variant_to_canonical.get(p, p)


def get_robust_catalog(chunks: list) -> dict:
    """Build a clean catalog of entities from chunks (attribute + text fallback).
    Merges OCR variants (Rika Popper, Rita Peyer -> Rita Pepper) to avoid duplicates."""
    patient_counts: dict[str, int] = {}
    claims, policies, groups, doctors = set(), set(), set(), set()

    for c in chunks or []:
        p = getattr(c, "patient_name", "") or ""
        cl = getattr(c, "claim_number", "") or ""
        pol = getattr(c, "policy_number", "") or ""
        g = getattr(c, "group_number", "") or ""
        d = getattr(c, "doctor_name", "") or ""

        if not p or not cl:
            text = getattr(c, "text", "") or ""
            if text:
                meta = extract_chunk_metadata(text)
                p = p or meta.get("patient_name", "")
                cl = cl or meta.get("claim_number", "")
                pol = pol or meta.get("policy_number", "")
                g = g or meta.get("group_number", "")
                d = d or meta.get("doctor_name", "")

        if p:
            pn = _normalize_name(p)
            patient_counts[pn] = patient_counts.get(pn, 0) + 1
        if cl and _is_valid_metadata("claim_number", cl):
            claims.add(cl)
        if pol and _is_valid_metadata("policy_number", pol):
            policies.add(pol)
        if g and _is_valid_metadata("group_number", g):
            groups.add(g)
        if d and _is_valid_metadata("doctor_name", d) and len(d.split()) <= 6:
            doctors.add(_normalize_name(d))

    known_patients, patient_name_aliases = _merge_similar_patient_names(patient_counts)

    return {
        "known_patients": known_patients,
        "patient_name_aliases": patient_name_aliases,
        "known_claims": sorted(claims),
        "known_policies": sorted(policies),
        "known_groups": sorted(groups),
        "known_doctors": sorted(doctors),
    }


def _get_document_context(chunks: list, max_chunks: int = 8) -> str:
    """Build a representative sample of document content for the LLM."""
    if not chunks:
        return "No documents indexed."

    seen_files = set()
    samples = []
    for c in chunks:
        fn = getattr(c, "file_name", "")
        if fn in seen_files:
            continue
        seen_files.add(fn)
        text = (getattr(c, "text", "") or "")[:300].replace("\n", " ").strip()
        if len(text) < 20:
            continue
        patient = getattr(c, "patient_name", "") or ""
        samples.append(f"- File: {fn}" + (f" | Patient: {patient}" if patient else "") + f"\n  Content: {text}...")
        if len(samples) >= max_chunks:
            break

    return "\n".join(samples) if samples else "No readable document samples."


# ──────────────────────────────────────────────────────────────
# LLM Agent: the brain
# ──────────────────────────────────────────────────────────────

def _build_agent_prompt(query: str, catalog: dict, doc_context: str) -> str:
    """
    The core prompt that makes the LLM think like a retrieval agent.
    It generates multiple search queries for maximum coverage.
    """
    catalog_str = json.dumps({
        "patients": catalog.get("known_patients", []),
        "claims": catalog.get("known_claims", []),
        "doctors": catalog.get("known_doctors", []),
    }, indent=0)

    return f"""You are an autonomous retrieval agent for a multi-domain document search system.
Your job: understand the user's query and generate the BEST search strategy to find relevant documents or images.

USER QUERY: "{query}"

KNOWN ENTITIES IN THE INDEX (only use if explicitly mentioned in the query):
{catalog_str}

SAMPLE CONTENT ACROSS DOCUMENTS:
{doc_context}

YOUR TASK: Analyze the query and output a JSON retrieval plan. Think step by step:
1. What is the user actually asking for?
2. What keywords or visual descriptors (if seeking images) would appear in the target content?
3. Identity check: Does the query mention a specific patient, claim, or person?
4. Generate 2-3 different search queries to maximize coverage across modalities (text and visual).

OUTPUT FORMAT (valid JSON only, no markdown):
{{
  "reasoning": "Brief explanation of your strategy",
  "intent": "list_entities | get_info | compare | general_search",
  "search_queries": ["query1", "query2", "query3"],
  "main_intent_keywords": ["phrase1", "phrase2"],
  "scope": "all_patients | specific_patient | unscoped",
  "patient_filter": null or "EXACT patient name from catalog ONLY if explicitly mentioned",
  "query_type": "text_heavy | image_heavy | hybrid",
  "direct_answer": null,
  "target_attribute": null
}}

CRITICAL RULES:
- If the query mentions a specific file (e.g. flower.png, report.pdf), ALWAYS include the exact filename in search_queries.
- ONLY set patient_filter if the patient's name is EXPLICITLY MENTIONED in the query.
- If the query is general (e.g., "diagrams", "arrows", "forms"), do NOT apply any patient filters.
- Use words that appear in the documents (PHYSICAL CAPACITIES, icd10, ICD-9, treatment, stand, walk, hours, etc.) for text queries.
- Use visual descriptors (diagram, chart, arrow, screenshot) for image-seeking queries.
- main_intent_keywords: CRITICAL for retrieval accuracy. Include BOTH (a) the user's phrasing AND (b) the formal/document way this concept appears (section headers, regulatory terms). Examples:
  * User: "forex exposure that is not hedged" → include ["unhedged foreign currency exposure", "currency induced credit risk", "forex exposure not hedged"]
  * User: "RBI provisioning divergence" → include ["divergence in asset classification and provisioning", "RBI provisioning", "provisioning"]
  * User: "how does the bank handle X" → include the exact section title or regulatory term for X (e.g. "Unhedged Foreign Currency Exposure")
  Without document-equivalent phrases, similar queries will miss the right chunk. Think: "How would this appear in a formal report or policy document?"
- query_type: "text_heavy" if answer is in text/narrative, "image_heavy" if in forms/diagrams/charts, "hybrid" if both (e.g. physical capacities form).
- NEVER refuse or say you can't help."""


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


def _call_llm(prompt: str, api_key: str, provider: str) -> str | None:
    """Call the LLM and return raw text response. Falls back to OpenAI when Groq limit exceeded."""
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1,
            )
            return r.choices[0].message.content if r.choices else None
        elif provider == "gemini":
            try:
                from google import genai
                client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
                r = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                return getattr(r, "text", None)
            except Exception:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                r = model.generate_content(prompt)
                return getattr(r, "text", None)
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1,
            )
            return r.choices[0].message.content if r.choices else None
    except Exception as e:
        # When Groq rate limit exceeded, fallback to OpenAI if available
        if provider == "groq" and _is_groq_limit_error(e):
            openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if openai_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    r = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=600,
                        temperature=0.1,
                    )
                    return r.choices[0].message.content if r.choices else None
                except Exception:
                    pass
        return None
    return None


def _call_llm_stream(prompt: str, api_key: str, provider: str, max_tokens: int = 2000):
    """Stream LLM tokens for Groq and OpenAI. Yields str deltas.

    Gemini and other providers: yields the full response once (non-streaming fallback).
    On any error, silently falls back to the non-streaming `_call_llm`.
    """
    provider = (provider or "").strip().lower() or "groq"
    try:
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=min(max_tokens, 4096),
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                except Exception:
                    delta = None
                if delta:
                    yield delta
            return
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                except Exception:
                    delta = None
                if delta:
                    yield delta
            return
    except Exception as e:
        # Groq rate-limit fallback: stream from OpenAI instead.
        if provider == "groq" and _is_groq_limit_error(e):
            openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if openai_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.1,
                        stream=True,
                    )
                    for chunk in stream:
                        try:
                            delta = chunk.choices[0].delta.content if chunk.choices else None
                        except Exception:
                            delta = None
                        if delta:
                            yield delta
                    return
                except Exception as fb_err:
                    logger.warning("OpenAI streaming fallback failed: %s", fb_err)
        logger.warning("Streaming LLM call failed (%s); falling back to non-streaming: %s", provider, e)

    # Final fallback: one-shot non-streaming.
    out = _call_llm(prompt, api_key, provider)
    if out:
        yield out


def _parse_llm_plan(raw: str | None) -> dict | None:
    """Parse the LLM's JSON response into a retrieval plan."""
    if not raw:
        return None
    cleaned = re.sub(r"^```\w*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    try:
        plan = json.loads(cleaned)
        if isinstance(plan.get("search_queries"), list) and plan["search_queries"]:
            return plan
    except json.JSONDecodeError:
        pass
    return None


def _resolve_api_key(llm_api_key: str) -> str:
    key = (llm_api_key or "").strip()
    if key:
        return key
    # Groq first, then OpenAI as fallback, then Gemini
    for env in ["GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        k = os.environ.get(env)
        if k and str(k).strip():
            return str(k).strip()
    return ""


def _resolve_provider(llm_provider: str) -> str:
    p = (llm_provider or "").strip().lower()
    if "groq" in p:
        return "groq"
    if "gemini" in p or "google" in p:
        return "gemini"
    if "openai" in p or "chatgpt" in p:
        return "openai"
    # Prefer Groq if key exists, else fallback to OpenAI, then Gemini
    if os.environ.get("GROQ_API_KEY") and str(os.environ.get("GROQ_API_KEY", "")).strip():
        return "groq"
    if os.environ.get("OPENAI_API_KEY") and str(os.environ.get("OPENAI_API_KEY", "")).strip():
        return "openai"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "openai"  # default fallback when no keys


# ──────────────────────────────────────────────────────────────
# Rule-based fallback (no LLM key available)
# ──────────────────────────────────────────────────────────────

def _fallback_plan(query: str, catalog: dict) -> dict:
    """
    When no LLM is available, generate a reasonable plan using rules.
    Still generates multiple search queries for coverage.
    """
    q_lower = (query or "").strip().lower()
    queries = [query]

    # Check if listing entities
    list_words = ["list", "show", "all", "every", "existing", "who are", "name",
                  "enumerate", "display", "how many", "give me"]
    entity_words = {"patient": "known_patients", "doctor": "known_doctors",
                    "physician": "known_doctors", "claim": "known_claims"}

    for entity, cat_key in entity_words.items():
        if entity in q_lower and any(lw in q_lower for lw in list_words):
            values = catalog.get(cat_key, [])
            if values:
                label = cat_key.replace("known_", "").replace("_", " ").title()
                return {
                    "intent": "list_entities",
                    "search_queries": [f"{entity} name list", query],
                    "main_intent_keywords": [entity, "name", "list"],
                    "scope": "all_patients",
                    "patient_filter": None,
                    "query_type": "text_heavy",
                    "direct_answer": f"{label} in index: " + ", ".join(values),
                    "target_attribute": cat_key,
                }

    # Check for specific patient mention
    patient_filter = None
    for p in catalog.get("known_patients", []):
        if p.lower() in q_lower:
            patient_filter = p
            break

    # Generate additional search queries by adding related terms
    extra = query
    term_map = {
        "disab": "disability restrictions physical capacities limitations",
        "restrict": "restrictions activity restrictions physical capacities",
        "doctor": "attending physician physician name doctor",
        "physician": "attending physician physician name",
        "diagnos": "diagnosis primary diagnosis ICD10 condition",
        "treatment": "treatment medications prognosis therapy",
        "symptom": "symptoms subjective symptoms complaints",
        "pain": "pain symptoms chest pain subjective",
        "work": "work capabilities return to work occupation",
        "surgery": "surgery hospitalization procedure operation",
        "medication": "medications treatment prescribed drug",
        "claim": "claim claim number claim status benefits",
        "policy": "policy number policy coverage premium",
        "capacity": "physical capacities stand walk sit lift carry hours",
        "stand": "stand walk physical capacities hours total",
        "hour": "hours stand walk physical capacities total",
        "patient": "Patient Name insured patient information",
        "insurance": "policy number group number claim insured",
        "walk": "walk stand sit lift carry physical capacities",
        "lift": "lift carry handling physical capacities",
    }
    added = set()
    for trigger, expansion in term_map.items():
        if trigger in q_lower:
            for term in expansion.split():
                if term.lower() not in q_lower and term not in added:
                    extra += " " + term
                    added.add(term)

    if extra != query:
        queries.append(extra)

    if patient_filter:
        queries.append(f"{patient_filter} {query}")

    from .hybrid_fusion import _extract_query_phrases
    main_kw = _extract_query_phrases(query)
    return {
        "intent": "general_search",
        "search_queries": queries[:3],
        "main_intent_keywords": main_kw[:6] if main_kw else [],
        "scope": "all_patients" if "all" in q_lower or "every" in q_lower or "each" in q_lower or "patients" in q_lower else "unscoped",
        "patient_filter": patient_filter,
        "query_type": "text_heavy",
        "direct_answer": None,
        "target_attribute": None,
    }


# ──────────────────────────────────────────────────────────────
# Retrieval execution
# ──────────────────────────────────────────────────────────────

def _find_best_chunk_per_patient(
    patient: str, chunks: list, top_n: int = 1, variants: list | None = None
) -> list:
    """Find chunks for a patient by text or metadata. Prefers header mentions. variants = OCR aliases."""
    check_names = [patient] + (variants or [])
    check_lower = {re.sub(r"\s+", " ", n.lower()) for n in check_names if n}
    scored = []
    for c in chunks or []:
        chunk_patient = re.sub(r"\s+", " ", (getattr(c, "patient_name", "") or "").lower())
        if chunk_patient and chunk_patient not in check_lower:
            continue
        text = (getattr(c, "text", "") or "")
        text_norm = re.sub(r"\s+", " ", text.lower())  # collapse spaces for matching
        matched_via_tokens = False
        if not chunk_patient and not any(p in text_norm for p in check_lower):
            # Also try: both first+last name tokens present (handles "Jude, Alyson", extra spaces)
            tokens = {w for w in re.findall(r"[a-z]+", text_norm) if len(w) >= 2}
            patient_tokens = {w for n in check_names if n for w in n.lower().split() if len(w) >= 2}
            if not (patient_tokens and patient_tokens <= tokens):
                continue
            matched_via_tokens = True
        text_lower = text.lower()
        text_norm_for_pos = re.sub(r"\s+", " ", text_lower)
        if matched_via_tokens:
            pos = min((text_lower.find(t) for t in patient_tokens if t in text_lower), default=0)
        else:
            pos = min((text_norm_for_pos.find(p) for p in check_lower if p in text_norm_for_pos), default=len(text_lower))
        position_score = max(0.0, 1.0 - (pos / max(len(text_lower), 1)))
        has_label = 1.0 if "patient name" in text_lower[: pos + 50] else 0.0
        score = position_score + has_label
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def _expand_queries_with_intent_keywords(
    search_queries: list[str],
    main_intent_keywords: list[str] | None,
    max_extra: int = 3,
) -> list[str]:
    """
    Add document-style phrases from LLM as additional search queries.
    When user says "forex exposure not hedged", the LLM produces "unhedged foreign currency
    exposure" — running retrieval for that finds the right chunk. Dynamic, no hardcoding.
    """
    if not main_intent_keywords:
        return search_queries
    seen = {q.strip().lower() for q in search_queries if q}
    extra = []
    # Prefer longer, more specific phrases (likely section headers / formal terms)
    for kw in sorted(main_intent_keywords, key=lambda x: -len(x)):
        if not kw or len(kw) < 8:
            continue
        k = kw.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        extra.append(kw.strip())
        if len(extra) >= max_extra:
            break
    return search_queries + extra


def _multi_query_retrieve(
    search_queries: list[str],
    text_retriever: Any,
    image_retriever: Any,
    index: Any,
    metadata_filter: dict | None,
    retrieve_k: int,
    boost_ocr: bool = False,
    main_intent_keywords: list[str] | None = None,
    rerank_query: str | None = None,
) -> tuple[list, list]:
    """
    Run multiple search queries, merge their candidates, then rerank ONCE.

    Previous behaviour reranked per-query (3 queries × 80 candidates = 240 CPU
    cross-encoder scorings, ~30s on CPU). We now:
      1. Gather BM25+vector RRF hybrid hits from each query (no rerank per query).
      2. Deduplicate across queries, keeping the max hybrid score per chunk.
      3. Rerank the UNION a single time against `rerank_query` (the user's
         original question) so scores are consistent across all candidates.
      4. If the deduped union is small (<= RERANKER_MIN_CANDIDATES), skip the
         cross-encoder — RRF ordering is already a strong signal for small
         pools (typical for patient-scoped queries).
    """
    from config import RERANKER_CANDIDATES, RERANKER_MIN_CANDIDATES, STRUCTURED_SKIP_RERANKER

    queries = _expand_queries_with_intent_keywords(
        search_queries[:2], main_intent_keywords, max_extra=1
    )
    rerank_q = (rerank_query or (search_queries[0] if search_queries else "")).strip()

    all_text: dict[str, tuple[Any, float]] = {}
    all_image: dict[tuple[str, Any], tuple[dict, float]] = {}

    # Step 1: gather hybrid candidates (no rerank) from every query variant.
    for sq in queries:
        hybrid_hits = index.hybrid_search(
            sq, top_k=RERANKER_CANDIDATES, fusion="rrf", metadata_filter=metadata_filter,
        ) if hasattr(index, "hybrid_search") else []
        for chunk, score in hybrid_hits:
            cid = getattr(chunk, "chunk_id", None)
            if cid is None:
                continue
            if cid not in all_text or score > all_text[cid][1]:
                all_text[cid] = (chunk, score)

        image_hits = image_retriever.retrieve(
            sq, top_n=20, metadata_filter=metadata_filter, boost_ocr=boost_ocr,
        )
        for item, score in image_hits:
            key = (item.get("file_name", ""), item.get("page", ""))
            if key not in all_image or score > all_image[key][1]:
                all_image[key] = (item, score)

    # Cap union size for rerank, preferring highest RRF scores.
    union = sorted(all_text.values(), key=lambda x: x[1], reverse=True)
    union = union[:RERANKER_CANDIDATES]

    # Step 2: single rerank pass against the user's original question — unless
    # the pool is already small (fast path) or contains mostly structured docs
    # (structured-doc skip is honored to match text_retriever behaviour).
    text_results: list[tuple[Any, float]]
    skip_rerank = (
        len(union) <= RERANKER_MIN_CANDIDATES
        or not rerank_q
        or (STRUCTURED_SKIP_RERANKER and union and sum(
            1 for c, _ in union if getattr(c, "doc_quality", "") == "structured"
        ) > len(union) / 2)
    )
    if skip_rerank or not hasattr(index, "rerank"):
        text_results = union
    else:
        try:
            text_results = index.rerank(
                rerank_q, union, top_k=retrieve_k, prioritize_exact_phrase=True,
            )
        except Exception:
            text_results = union

    image_results = sorted(all_image.values(), key=lambda x: x[1], reverse=True)
    return text_results[:retrieve_k], image_results[:20]


def _list_entities_retrieve(
    patients: list[str],
    search_queries: list[str],
    text_retriever: Any,
    index: Any,
    patient_name_aliases: dict | None = None,
) -> list[dict]:
    """Guaranteed one-chunk-per-patient retrieval. Uses aliases to match OCR variants."""
    fused = []
    rerank_query = search_queries[0] if search_queries else "Patient Name"
    aliases = patient_name_aliases or {}

    for p in patients:
        found = False
        variants = aliases.get(p, [p, p.upper(), _normalize_name(p)])
        mf = {"patient_name": variants if isinstance(variants, list) else [variants]}
        hits = text_retriever.index.hybrid_search(
            "Patient Name", top_k=3, fusion="rrf",
            metadata_filter=mf,
        )
        if hits:
            reranked = text_retriever.index.rerank(rerank_query + " " + p, hits, top_k=1)
            if reranked:
                fused.append({"type": "text", "content": reranked[0][0], "final_score": reranked[0][1]})
                found = True
        if not found:
            best = _find_best_chunk_per_patient(p, index.chunks, top_n=1, variants=aliases.get(p))
            if best:
                fused.append({"type": "text", "content": best[0][0], "final_score": best[0][1]})

    return fused


def _per_patient_attribute_retrieve(
    patients: list[str],
    search_queries: list[str],
    text_retriever: Any,
    index: Any,
    patient_name_aliases: dict | None = None,
) -> list[dict]:
    """Search for a specific attribute across all patients. Uses aliases for OCR variants."""
    fused = []
    rerank_query = " ".join(search_queries[:2]) if search_queries else "patient information"
    aliases = patient_name_aliases or {}

    for p in patients:
        hits = []
        variants = aliases.get(p, [p, p.upper(), _normalize_name(p)])
        mf = {"patient_name": variants if isinstance(variants, list) else [variants]}
        for sq in search_queries[:2]:
            h = text_retriever.index.hybrid_search(
                sq, top_k=5, fusion="rrf",
                metadata_filter=mf,
            )
            if h:
                hits.extend(h)
                break

        if not hits:
            best = _find_best_chunk_per_patient(p, index.chunks, top_n=3, variants=aliases.get(p))
            hits = [(c, s) for c, s in best]

        if hits:
            # Deduplicate
            seen = set()
            unique = []
            for c, s in hits:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    unique.append((c, s))

            reranked = text_retriever.index.rerank(rerank_query, unique, top_k=1)
            if reranked:
                fused.append({"type": "text", "content": reranked[0][0], "final_score": reranked[0][1]})
            else:
                fused.append({"type": "text", "content": hits[0][0], "final_score": hits[0][1]})

    return fused


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def run_agentic_rag(
    query: str,
    index: Any,
    text_retriever: Any,
    image_retriever: Any,
    catalog: dict,
    user_metadata_filter: Optional[dict] = None,
    use_llm: bool = True,
    llm_api_key: str = "",
    llm_provider: str = "groq",
    page_trees: list | None = None,
    data_folder: str | None = None,
) -> tuple[dict | None, list[dict], str | None]:
    """
    Autonomous agentic RAG:
    1. LLM analyzes query → generates retrieval plan (multiple search queries)
    2. Executes plan (multi-query retrieval, per-patient, or list)
    3. Guarantees minimum results via fallback
    """
    from .query_metadata_extractor import merge_metadata_filters
    from .hybrid_fusion import fuse_results, boost_phrase_matching
    from .result_diversifier import diversify_fused_results
    from .query_classifier import classify_query

    from config import (
        MULTIMODAL_HYBRID_TOP_K,
        METADATA_DIVERSITY_ENABLED,
        METADATA_DIVERSITY_MAX_PER_ENTITY,
        METADATA_DIVERSITY_CANDIDATES,
        PREFER_USER_METADATA_FILTER,
    )

    MIN_RESULTS = 3

    # Step 1: Build context for the agent
    robust_catalog = get_robust_catalog(index.chunks) if index and index.chunks else {}
    catalog_final = robust_catalog if robust_catalog.get("known_patients") else catalog or {}

    # Step 2: Let the LLM think (or fall back to rules)
    plan = None
    api_key = _resolve_api_key(llm_api_key)
    provider = _resolve_provider(llm_provider)

    if use_llm and api_key:
        doc_context = _get_document_context(index.chunks) if index and index.chunks else ""
        prompt = _build_agent_prompt(query, catalog_final, doc_context)
        raw = _call_llm(prompt, api_key, provider)
        plan = _parse_llm_plan(raw)

    if not plan:
        plan = _fallback_plan(query, catalog_final)

    # Extract plan fields (LLM-derived when available; dynamic for any query/patient)
    intent = plan.get("intent", "general_search")
    # Override: if query clearly asks for list of patients and catalog has them, ensure list_entities
    q_lower = (query or "").strip().lower()
    if "patient" in q_lower and any(w in q_lower for w in ["list", "show", "all", "every", "who", "name", "enumerate"]):
        if catalog_final.get("known_patients"):
            intent = "list_entities"
            plan["target_attribute"] = plan.get("target_attribute") or "known_patients"
    search_queries = plan.get("search_queries") or [query]
    main_intent_keywords = plan.get("main_intent_keywords") or []
    scope = plan.get("scope", "unscoped")
    patient_filter = plan.get("patient_filter")
    plan_query_type = plan.get("query_type")
    direct_answer = plan.get("direct_answer")
    target_attr = plan.get("target_attribute")
    main_phrases = main_intent_keywords if main_intent_keywords else None
    filter_phrases = [patient_filter] if patient_filter else None

    # When query mentions a specific file (e.g. "@data/uploads/foo.md"), filter retrieval to that basename
    query_filename = _extract_query_filename(query)
    if query_filename and query_filename not in search_queries:
        search_queries = [query_filename] + list(search_queries)

    # Build metadata filter (expand patient to OCR variants for matching)
    metadata_filter = {}
    if query_filename:
        metadata_filter["file_name"] = query_filename
    if patient_filter:
        aliases = (catalog_final or {}).get("patient_name_aliases", {})
        metadata_filter["patient_name"] = aliases.get(patient_filter, [patient_filter])
    metadata_filter = merge_metadata_filters(
        user_metadata_filter or {},
        metadata_filter,
        prefer_user=PREFER_USER_METADATA_FILTER,
    ) or {}

    # Scoped fast-path: when retrieval is already narrowed to a specific file
    # or patient, reduce fan-out and candidate breadth to keep latency low.
    has_file_scope = bool((metadata_filter or {}).get("file_name"))
    has_patient_scope = bool((metadata_filter or {}).get("patient_name"))
    if (has_file_scope or has_patient_scope) and len(search_queries) > 2:
        search_queries = search_queries[:2]

    # Build understanding dict for UI display (agent understanding page)
    understanding = {
        "intent": intent,
        "metadata_filter": metadata_filter,
        "search_query": search_queries[0] if search_queries else query,
        "search_queries": search_queries,
        "main_intent_keywords": main_intent_keywords,
        "query_type": plan_query_type,
        "direct_answer": direct_answer,
        "target_attribute": target_attr,
        "reasoning": plan.get("reasoning", ""),
    }

    retrieve_k = METADATA_DIVERSITY_CANDIDATES if METADATA_DIVERSITY_ENABLED else MULTIMODAL_HYBRID_TOP_K
    if has_file_scope:
        retrieve_k = min(retrieve_k, 12)
    elif has_patient_scope:
        retrieve_k = min(retrieve_k, 20)

    # Active patient filter can be str or list (aliases); normalize once.
    patient_vals = (metadata_filter or {}).get("patient_name") or []
    active_patients = {
        str(p).strip().lower()
        for p in (patient_vals if isinstance(patient_vals, (list, tuple)) else [patient_vals])
        if p
    }

    # Step 3: Execute the plan
    aliases = catalog_final.get("patient_name_aliases", {})
    if intent == "list_entities" and catalog_final.get("known_patients") and "patient" in (target_attr or "patient"):
        fused = _list_entities_retrieve(
            catalog_final["known_patients"], search_queries, text_retriever, index,
            patient_name_aliases=aliases,
        )

    elif intent == "list_entities" and catalog_final.get("known_doctors") and "doctor" in (target_attr or ""):
        fused = _list_entities_retrieve(
            catalog_final["known_patients"], search_queries, text_retriever, index,
            patient_name_aliases=aliases,
        )
        understanding["direct_answer"] = "Doctors: " + ", ".join(catalog_final["known_doctors"])

    elif scope == "all_patients" and catalog_final.get("known_patients") and intent != "general_search":
        fused = _per_patient_attribute_retrieve(
            catalog_final["known_patients"], search_queries, text_retriever, index,
            patient_name_aliases=aliases,
        )

    else:
        # Multi-query semantic search — the core autonomous path
        query_type = plan_query_type if plan_query_type in ("text_heavy", "image_heavy", "hybrid") else classify_query(query)
        text_results, image_results = _multi_query_retrieve(
            search_queries, text_retriever, image_retriever, index,
            metadata_filter=metadata_filter or None,
            retrieve_k=retrieve_k,
            boost_ocr=(query_type == "text_heavy"),
            main_intent_keywords=main_intent_keywords,
            rerank_query=query,
        )

        # If patient metadata filtering returns context that is semantically off-topic
        # (common when one file has missing patient metadata), do a guarded semantic
        # backoff: retrieve without patient filter and keep only query-overlap chunks.
        if active_patients and text_results:
            q_tokens = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) >= 4]
            q_token_set = set(q_tokens)

            def _overlap_hits(triples: list[tuple]) -> int:
                n = 0
                for c, _ in triples[:12]:
                    t = (getattr(c, "text", "") or "").lower()
                    overlap = sum(1 for tok in q_token_set if tok in t)
                    if overlap >= 2:
                        n += 1
                return n

            q_phrase = (query or "").strip().lower()

            def _has_phrase(triples: list[tuple]) -> bool:
                if len(q_phrase) < 10:
                    return False
                for c, _ in triples[:12]:
                    t = (getattr(c, "text", "") or "").lower()
                    if q_phrase in t:
                        return True
                return False

            # Very low lexical overlap OR no phrase hit means filtered hits are likely wrong section/file.
            if q_token_set and (_overlap_hits(text_results) <= 1 or not _has_phrase(text_results)):
                mf_backoff = dict(metadata_filter or {})
                mf_backoff.pop("patient_name", None)
                backoff_text, _ = _multi_query_retrieve(
                    search_queries,
                    text_retriever,
                    image_retriever,
                    index,
                    metadata_filter=mf_backoff or None,
                    retrieve_k=retrieve_k,
                    boost_ocr=(query_type == "text_heavy"),
                    main_intent_keywords=main_intent_keywords,
                    rerank_query=query,
                )
                seen_ids = {c.chunk_id for c, _ in text_results}
                for c, s in backoff_text:
                    t = (getattr(c, "text", "") or "").lower()
                    overlap = sum(1 for tok in q_token_set if tok in t)
                    has_phrase = len(q_phrase) >= 10 and q_phrase in t
                    if overlap < 2 and not has_phrase:
                        continue
                    if c.chunk_id in seen_ids:
                        continue
                    # Slight boost: content overlap rescue should be visible to fusion.
                    text_results.append((c, max(s, 1.2 if has_phrase else 0.95)))
                    seen_ids.add(c.chunk_id)

        if page_trees:
            from indexing.page_tree import tree_keyword_retrieve
            # Compute the allowed-file set BEFORE scanning so unrelated trees
            # are never opened. For a query like "Alyson Jude's activity
            # restrictions" this means only her files are scanned.
            # file_name may be a str (single file) or list/tuple (tenant scope ≤8 files — see rag_service).
            target_file_raw = (metadata_filter or {}).get("file_name")
            allowed_files: set[str] | None = None
            if target_file_raw:
                if isinstance(target_file_raw, (list, tuple)):
                    allowed_files = {str(f).strip() for f in target_file_raw if f}
                    if not allowed_files:
                        allowed_files = None
                else:
                    tf = str(target_file_raw).strip()
                    allowed_files = {tf} if tf else None
            elif active_patients and index and getattr(index, "chunks", None):
                allowed_files = {
                    getattr(c, "file_name", "")
                    for c in index.chunks
                    if (getattr(c, "patient_name", "") or "").strip().lower() in active_patients
                }
                allowed_files = {f for f in allowed_files if f} or None
            tree_hits = tree_keyword_retrieve(
                query, page_trees, data_folder, top_k=25,
                allowed_files=allowed_files,
            )
            q_lower = (query or "").strip().lower()
            # Prioritize tree hits with EXACT phrase match (e.g. "Concentration of Funding Sources")
            # so PDF page 381 ranks above irrelevant pages 384, 107, etc.
            exact_tree = []
            other_tree = []
            for tc, sc in tree_hits:
                text = (getattr(tc, "text", "") or "").lower()
                if len(q_lower) >= 10 and q_lower in text:
                    exact_tree.append((tc, max(sc, 1.0)))  # Boost exact matches
                else:
                    other_tree.append((tc, sc))
            seen_tree = {c.chunk_id for c, _ in text_results}
            for tc, sc in exact_tree + other_tree:
                if tc.chunk_id not in seen_tree:
                    text_results.append((tc, sc))
                    seen_tree.add(tc.chunk_id)
            text_results.sort(key=lambda x: x[1], reverse=True)

        verbatim = []
        if index and hasattr(index, "verbatim_search"):
            verbatim = index.verbatim_search(query, metadata_filter or None, max_results=10)
            # Also search for document-style phrases from LLM (section headers, formal terms)
            for kw in (main_intent_keywords or [])[:3]:
                if kw and len(kw) >= 8:
                    v_kw = index.verbatim_search(kw, metadata_filter or None, max_results=5)
                    for c, s in v_kw:
                        if c.chunk_id not in {x.chunk_id for x, _ in verbatim}:
                            verbatim.append((c, s))
        seen_ids = {c.chunk_id for c, _ in verbatim}
        # Prepend exact-phrase tree hits FIRST so they get best RRF (PDF page 381 outranks .md)
        exact_tree_front = [(c, s) for c, s in text_results if c.chunk_id not in seen_ids and getattr(c, "chunk_id", "").startswith("tree_kw_") and len(q_lower) >= 10 and q_lower in (getattr(c, "text", "") or "").lower()]
        seen_ids.update(c.chunk_id for c, _ in exact_tree_front)
        text_results = exact_tree_front + list(verbatim) + [(c, s) for c, s in text_results if c.chunk_id not in seen_ids]
        main_phrases = main_intent_keywords if main_intent_keywords else None
        filter_phrases = [patient_filter] if patient_filter else None
        text_results = boost_phrase_matching(text_results, query, main_phrases_override=main_phrases, filter_phrases_override=filter_phrases)
        fuse_top_k = retrieve_k if METADATA_DIVERSITY_ENABLED else MULTIMODAL_HYBRID_TOP_K
        fused = fuse_results(text_results, image_results, query_type, top_k=fuse_top_k)

        # Only apply diversity for broad queries (all_patients, unscoped without a filter).
        # For specific queries (patient filter, specific intent), preserve score ordering.
        apply_diversity = (
            METADATA_DIVERSITY_ENABLED
            and fused
            and scope in ("all_patients",)
            and not patient_filter
        )
        if apply_diversity:
            fused = diversify_fused_results(
                fused, entity_key="patient_name",
                top_k=MULTIMODAL_HYBRID_TOP_K, max_per_entity=METADATA_DIVERSITY_MAX_PER_ENTITY,
            )

    # Step 4: Guarantee minimum results — retry broader if too few
    if len(fused) < MIN_RESULTS and index and index.chunks:
        query_type = classify_query(query)  # Ensure set for fallback (list_entities path may not set it)
        existing_ids = set()
        for r in fused:
            c = r["content"]
            existing_ids.add(c.chunk_id if r["type"] == "text" else (c.get("file_name", ""), c.get("page", "")))

        fallback_text, fallback_image = _multi_query_retrieve(
            [query] + search_queries[:1], text_retriever, image_retriever, index,
            metadata_filter=metadata_filter or None, retrieve_k=retrieve_k,
            boost_ocr=(query_type == "text_heavy"),
            main_intent_keywords=main_intent_keywords,
            rerank_query=query,
        )
        if index and hasattr(index, "verbatim_search"):
            v_fb = index.verbatim_search(query, metadata_filter or None, max_results=10)
            for kw in (main_intent_keywords or [])[:3]:
                if kw and len(kw) >= 8:
                    v_kw = index.verbatim_search(kw, metadata_filter or None, max_results=5)
                    for c, s in v_kw:
                        if c.chunk_id not in {x.chunk_id for x, _ in v_fb}:
                            v_fb.append((c, s))
            seen_fb = {c.chunk_id for c, _ in v_fb}
            fallback_text = list(v_fb) + [(c, s) for c, s in fallback_text if c.chunk_id not in seen_fb]
        fallback_text = boost_phrase_matching(fallback_text, query, main_phrases_override=main_phrases, filter_phrases_override=filter_phrases)
        fallback_fused = fuse_results(fallback_text, fallback_image, query_type, top_k=retrieve_k)

        # patient_name can be str or list (OCR aliases)
        pn_val = (metadata_filter or {}).get("patient_name") or []
        active_patients = {p.lower() for p in (pn_val if isinstance(pn_val, (list, tuple)) else [pn_val]) if p}

        for r in fallback_fused:
            c = r["content"]
            key = c.chunk_id if r["type"] == "text" else (c.get("file_name", ""), c.get("page", ""))
            if key in existing_ids:
                continue

            # When a specific patient was requested, skip results that clearly
            # belong to a DIFFERENT patient (prevents cross-patient contamination).
            if active_patients:
                if r["type"] == "text":
                    result_patient = (getattr(c, "patient_name", "") or "").lower()
                else:
                    result_patient = (c.get("patient_name", "") or "").lower()
                if result_patient and result_patient not in active_patients:
                    continue

            fused.append(r)
            existing_ids.add(key)
            if len(fused) >= MIN_RESULTS:
                break

    return understanding, fused, direct_answer
