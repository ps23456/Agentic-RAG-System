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
from typing import Any, List, Tuple, Optional

from document_loader import extract_chunk_metadata

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


def get_robust_catalog(chunks: list) -> dict:
    """Build a clean catalog of entities from chunks (attribute + text fallback)."""
    patients, claims, policies, groups, doctors = set(), set(), set(), set(), set()

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
            patients.add(_normalize_name(p))
        if cl and _is_valid_metadata("claim_number", cl):
            claims.add(cl)
        if pol and _is_valid_metadata("policy_number", pol):
            policies.add(pol)
        if g and _is_valid_metadata("group_number", g):
            groups.add(g)
        if d and _is_valid_metadata("doctor_name", d) and len(d.split()) <= 6:
            doctors.add(_normalize_name(d))

    return {
        "known_patients": sorted(patients),
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
  "scope": "all_patients | specific_patient | unscoped",
  "patient_filter": null or "EXACT patient name from catalog ONLY if explicitly mentioned",
  "direct_answer": null,
  "target_attribute": null
}}

CRITICAL RULES:
- ONLY set patient_filter if the patient's name is EXPLICITLY MENTIONED in the query.
- If the query is general (e.g., "diagrams", "arrows", "forms"), do NOT apply any patient filters.
- Use words that appear in the documents (PHYSICAL CAPACITIES, icd10, ICD-9, treatment, etc.) for text queries.
- Use visual descriptors (diagram, chart, arrow, screenshot) for image-seeking queries.
- NEVER refuse or say you can't help."""


def _call_llm(prompt: str, api_key: str, provider: str) -> str | None:
    """Call the LLM and return raw text response."""
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
    except Exception:
        return None
    return None


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
    for env in ["GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]:
        k = os.environ.get(env)
        if k:
            return k.strip()
    return ""


def _resolve_provider(llm_provider: str) -> str:
    p = (llm_provider or "").strip().lower()
    if "groq" in p:
        return "groq"
    if "gemini" in p or "google" in p:
        return "gemini"
    if "openai" in p or "chatgpt" in p:
        return "openai"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "groq"


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
                    "scope": "all_patients",
                    "patient_filter": None,
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
        "capacity": "physical capacities stand walk sit lift carry",
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

    return {
        "intent": "general_search",
        "search_queries": queries[:3],
        "scope": "all_patients" if "all" in q_lower or "every" in q_lower or "each" in q_lower or "patients" in q_lower else "unscoped",
        "patient_filter": patient_filter,
        "direct_answer": None,
        "target_attribute": None,
    }


# ──────────────────────────────────────────────────────────────
# Retrieval execution
# ──────────────────────────────────────────────────────────────

def _find_best_chunk_per_patient(patient: str, chunks: list, top_n: int = 1) -> list:
    """Find chunks for a patient by text scan. Prefers header mentions."""
    p_lower = patient.lower()
    scored = []
    for c in chunks or []:
        text = (getattr(c, "text", "") or "")
        text_lower = text.lower()
        if p_lower not in text_lower:
            continue
        pos = text_lower.find(p_lower)
        position_score = max(0.0, 1.0 - (pos / max(len(text_lower), 1)))
        has_label = 1.0 if "patient name" in text_lower[:pos + len(p_lower) + 20] else 0.0
        score = position_score + has_label
        scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def _multi_query_retrieve(
    search_queries: list[str],
    text_retriever: Any,
    image_retriever: Any,
    index: Any,
    metadata_filter: dict | None,
    retrieve_k: int,
) -> tuple[list, list]:
    """
    Run multiple search queries and merge results.
    This is the key to handling any phrasing — different queries catch different angles.
    """
    all_text = {}
    all_image = {}

    for sq in search_queries:
        text_hits = text_retriever.retrieve(sq, top_k=retrieve_k, metadata_filter=metadata_filter)
        for chunk, score in text_hits:
            cid = chunk.chunk_id
            if cid not in all_text or score > all_text[cid][1]:
                all_text[cid] = (chunk, score)

        image_hits = image_retriever.retrieve(sq, top_n=20, metadata_filter=metadata_filter)
        for item, score in image_hits:
            key = (item.get("file_name", ""), item.get("page", ""))
            if key not in all_image or score > all_image[key][1]:
                all_image[key] = (item, score)

    text_results = sorted(all_text.values(), key=lambda x: x[1], reverse=True)
    image_results = sorted(all_image.values(), key=lambda x: x[1], reverse=True)

    return text_results[:retrieve_k], image_results[:20]


def _list_entities_retrieve(
    patients: list[str],
    search_queries: list[str],
    text_retriever: Any,
    index: Any,
) -> list[dict]:
    """Guaranteed one-chunk-per-patient retrieval."""
    fused = []
    rerank_query = search_queries[0] if search_queries else "Patient Name"

    for p in patients:
        found = False
        for name_variant in {p, p.upper(), _normalize_name(p)}:
            hits = text_retriever.index.hybrid_search(
                "Patient Name", top_k=3, fusion="rrf",
                metadata_filter={"patient_name": name_variant},
            )
            if hits:
                reranked = text_retriever.index.rerank(rerank_query + " " + p, hits, top_k=1)
                if reranked:
                    fused.append({"type": "text", "content": reranked[0][0], "final_score": reranked[0][1]})
                    found = True
                    break

        if not found:
            best = _find_best_chunk_per_patient(p, index.chunks, top_n=1)
            if best:
                fused.append({"type": "text", "content": best[0][0], "final_score": best[0][1]})

    return fused


def _per_patient_attribute_retrieve(
    patients: list[str],
    search_queries: list[str],
    text_retriever: Any,
    index: Any,
) -> list[dict]:
    """Search for a specific attribute across all patients."""
    fused = []
    rerank_query = " ".join(search_queries[:2]) if search_queries else "patient information"

    for p in patients:
        hits = []
        for name_variant in {p, p.upper(), _normalize_name(p)}:
            for sq in search_queries[:2]:
                h = text_retriever.index.hybrid_search(
                    sq, top_k=5, fusion="rrf",
                    metadata_filter={"patient_name": name_variant},
                )
                if h:
                    hits.extend(h)
                    break
            if hits:
                break

        if not hits:
            best = _find_best_chunk_per_patient(p, index.chunks, top_n=3)
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
) -> tuple[dict | None, list[dict], str | None]:
    """
    Autonomous agentic RAG:
    1. LLM analyzes query → generates retrieval plan (multiple search queries)
    2. Executes plan (multi-query retrieval, per-patient, or list)
    3. Guarantees minimum results via fallback
    """
    from .query_metadata_extractor import merge_metadata_filters
    from .hybrid_fusion import fuse_results
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

    # Extract plan fields
    intent = plan.get("intent", "general_search")
    search_queries = plan.get("search_queries") or [query]
    scope = plan.get("scope", "unscoped")
    patient_filter = plan.get("patient_filter")
    direct_answer = plan.get("direct_answer")
    target_attr = plan.get("target_attribute")

    # Build metadata filter
    metadata_filter = {}
    if patient_filter:
        metadata_filter["patient_name"] = patient_filter
    metadata_filter = merge_metadata_filters(
        user_metadata_filter or {},
        metadata_filter,
        prefer_user=PREFER_USER_METADATA_FILTER,
    ) or {}

    # Build understanding dict for UI display
    understanding = {
        "intent": intent,
        "metadata_filter": metadata_filter,
        "search_query": search_queries[0] if search_queries else query,
        "search_queries": search_queries,
        "direct_answer": direct_answer,
        "target_attribute": target_attr,
        "reasoning": plan.get("reasoning", ""),
    }

    retrieve_k = METADATA_DIVERSITY_CANDIDATES if METADATA_DIVERSITY_ENABLED else MULTIMODAL_HYBRID_TOP_K

    # Step 3: Execute the plan
    if intent == "list_entities" and catalog_final.get("known_patients") and "patient" in (target_attr or "patient"):
        fused = _list_entities_retrieve(
            catalog_final["known_patients"], search_queries, text_retriever, index,
        )

    elif intent == "list_entities" and catalog_final.get("known_doctors") and "doctor" in (target_attr or ""):
        fused = _list_entities_retrieve(
            catalog_final["known_patients"], search_queries, text_retriever, index,
        )
        understanding["direct_answer"] = "Doctors: " + ", ".join(catalog_final["known_doctors"])

    elif scope == "all_patients" and catalog_final.get("known_patients") and intent != "general_search":
        fused = _per_patient_attribute_retrieve(
            catalog_final["known_patients"], search_queries, text_retriever, index,
        )

    else:
        # Multi-query semantic search — the core autonomous path
        text_results, image_results = _multi_query_retrieve(
            search_queries, text_retriever, image_retriever, index,
            metadata_filter=metadata_filter or None,
            retrieve_k=retrieve_k,
        )
        query_type = classify_query(query)
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
        existing_ids = set()
        for r in fused:
            c = r["content"]
            existing_ids.add(c.chunk_id if r["type"] == "text" else (c.get("file_name", ""), c.get("page", "")))

        fallback_text, fallback_image = _multi_query_retrieve(
            [query] + search_queries[:1], text_retriever, image_retriever, index,
            metadata_filter=None, retrieve_k=retrieve_k,
        )
        query_type = classify_query(query)
        fallback_fused = fuse_results(fallback_text, fallback_image, query_type, top_k=retrieve_k)

        active_patient = (metadata_filter or {}).get("patient_name", "").lower()

        for r in fallback_fused:
            c = r["content"]
            key = c.chunk_id if r["type"] == "text" else (c.get("file_name", ""), c.get("page", ""))
            if key in existing_ids:
                continue

            # When a specific patient was requested, skip results that clearly
            # belong to a DIFFERENT patient (prevents cross-patient contamination).
            if active_patient:
                if r["type"] == "text":
                    result_patient = (getattr(c, "patient_name", "") or "").lower()
                else:
                    result_patient = (c.get("patient_name", "") or "").lower()
                if result_patient and result_patient != active_patient:
                    continue

            fused.append(r)
            existing_ids.add(key)
            if len(fused) >= MIN_RESULTS:
                break

    return understanding, fused, direct_answer
