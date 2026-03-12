"""
LLM-based query understanding for dynamic, accurate retrieval.
Parses ANY user query to extract intent, metadata filters, and search strategy.
Works for: list patients, doctor names, restrictions, claim status, etc.
"""
import json
import os
import re
from typing import Any, Optional


def _build_understanding_prompt(query: str, catalog: dict) -> str:
    catalog_str = json.dumps({
        "known_patients": catalog.get("known_patients", [])[:50],
        "known_claims": catalog.get("known_claims", [])[:20],
        "known_doctors": catalog.get("known_doctors", [])[:20],
    }, indent=0)
    return f"""You understand natural language like a human. The user is searching insurance claim documents (PDFs, forms with Patient Name, Claim #, disability, restrictions, physician, etc.).

User query: "{query}"

Available entities in the index (use EXACT values for metadata_filter):
{catalog_str}

Understand the CONTEXT and INTENT. Return ONLY valid JSON (no markdown):
- "intent": "list_entities" | "get_attribute" | "filter_and_search" | "general_search"
- "metadata_filter": {{}} or {{"patient_name": "X", "claim_number": "Y"}} - use exact catalog values
- "search_query": REWRITE for semantic search. Include field names + synonyms that appear in forms. Examples:
  * "list the patients" -> "Patient Name list of patients"
  * "who are the patients" -> "Patient Name Insured"
  * "show me disability stuff" -> "disability restrictions physical capacities limitations"
  * "Rika's claim" -> "claim Rika Popper disability"
  * "doctor name" -> "Attending Physician Name doctor physician"
  * "what can the patient do" -> "physical capacities stand walk sit lift carry"
  * "restrictions" -> "restrictions limitations physical capacities"
- "direct_answer": if listing entities and catalog has data, e.g. "Patients: A, B, C"; else null
- "target_attribute": patient_names | doctor_names | claim_status | restrictions | dates | null

Rules:
- Understand paraphrases: "stuff about X" = "X", "who is" = list, "tell me about" = general
- search_query must match words that appear in insurance forms (Patient Name, Claim, disability, physician, restrictions, etc.)
- Be generous with search_query - include related terms so we find relevant docs"""


def _get_llm_provider_and_key(api_key: str, provider: str) -> tuple[str, str]:
    """Resolve provider and key - prefer explicit, else try env vars."""
    key = (api_key or "").strip()
    if key and provider in ("groq", "gemini", "openai"):
        return provider, key
    import os
    for p, env in [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"), ("gemini", "GOOGLE_API_KEY"), ("openai", "OPENAI_API_KEY")]:
        k = os.environ.get(env)
        if k:
            return p, k.strip()
    return "groq", key or ""


def understand_query_llm(
    query: str,
    catalog: dict,
    api_key: str,
    provider: str = "groq",
) -> dict:
    """
    Use LLM to understand query. Returns dict with:
    intent, metadata_filter, search_query, direct_answer, target_attribute
    Falls back to rule-based if LLM fails.
    """
    if not query or not (query or "").strip():
        return {"intent": "general_search", "metadata_filter": {}, "search_query": query, "direct_answer": None, "target_attribute": None}

    prompt = _build_understanding_prompt(query, catalog)
    raw = None
    prov, key = _get_llm_provider_and_key(api_key, provider)
    if not key:
        return _understand_query_fallback(query, catalog)

    try:
        if prov == "groq":
            from groq import Groq
            client = Groq(api_key=key)
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            raw = r.choices[0].message.content if r.choices else None
        elif prov == "gemini":
            try:
                from google import genai
                client = genai.Client(api_key=key, http_options={"api_version": "v1"})
                r = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                raw = getattr(r, "text", None)
            except Exception:
                import google.generativeai as genai
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                r = model.generate_content(prompt)
                raw = getattr(r, "text", None)
        elif prov == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=key)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            raw = r.choices[0].message.content if r.choices else None
    except Exception:
        raw = None

    if raw:
        raw = re.sub(r"^```\w*\n?", "", raw).strip()
        raw = re.sub(r"\n?```\s*$", "", raw).strip()
        try:
            out = json.loads(raw)
            # Validate and clean
            meta = out.get("metadata_filter") or {}
            meta = {k: v for k, v in meta.items() if v and isinstance(v, str)}
            out["metadata_filter"] = meta
            out["search_query"] = (out.get("search_query") or query).strip() or query
            if out.get("intent") == "list_entities" and out.get("target_attribute") == "patient_names" and catalog.get("known_patients"):
                out["direct_answer"] = "Patients in index: " + ", ".join(catalog["known_patients"])
            return out
        except json.JSONDecodeError:
            pass

    # Fallback: rule-based
    return _understand_query_fallback(query, catalog)


def detect_list_patients_intent(query: str) -> bool:
    """Fast check: does query ask for list of patients? Works without LLM."""
    q = (query or "").strip().lower()
    if not q:
        return False
    patterns = ["list", "name all", "show all", "existing", "who are the", "all patients", "name the", "list the"]
    return any(p in q for p in patterns) and "patient" in q


def _expand_search_query(query: str, q_lower: str) -> str:
    """Expand query with synonyms/field names for better retrieval when LLM is off."""
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
        "walk": "walk stand sit lift carry physical capacities",
        "lift": "lift carry handling physical capacities",
    }
    expanded = query
    for trigger, expansion in term_map.items():
        if trigger in q_lower:
            for term in expansion.split():
                if term.lower() not in q_lower:
                    expanded += " " + term
    return expanded


def _understand_query_fallback(query: str, catalog: dict) -> dict:
    """Rule-based fallback when LLM unavailable. Understands context like natural language."""
    from .query_metadata_extractor import extract_metadata_from_query

    q_lower = (query or "").strip().lower()
    meta_filter = extract_metadata_from_query(
        query,
        known_patients=catalog.get("known_patients"),
        known_claims=catalog.get("known_claims"),
        known_policies=catalog.get("known_policies"),
        known_groups=catalog.get("known_groups"),
    )

    # List intent: "list patients", "existing patients", "name all patients"
    list_patterns = ["list", "name all", "show all", "existing", "who are the", "all patients", "all claims", "which patients"]
    is_list = any(p in q_lower for p in list_patterns)
    if is_list and "patient" in q_lower and catalog.get("known_patients"):
        return {
            "intent": "list_entities",
            "metadata_filter": {},
            "search_query": "Patient Name list of patients",
            "direct_answer": "Patients in index: " + ", ".join(catalog["known_patients"]),
            "target_attribute": "patient_names",
        }

    # Doctor/physician query
    if any(w in q_lower for w in ["doctor", "physician", "attending", "provider"]):
        return {
            "intent": "get_attribute",
            "metadata_filter": meta_filter or {},
            "search_query": query + " Attending Physician Name doctor physician",
            "direct_answer": None,
            "target_attribute": "doctor_names",
        }

    # Expand search_query for better retrieval (understand context)
    search_query = _expand_search_query(query, q_lower)
    return {
        "intent": "general_search",
        "metadata_filter": meta_filter or {},
        "search_query": search_query,
        "direct_answer": None,
        "target_attribute": None,
    }
