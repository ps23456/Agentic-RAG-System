"""
Query-aware metadata extraction: infer metadata filters from user query.
Works for varied phrasings - "Rika's claim", "policy 503", "documents for Teresa Brown".
Uses exact, partial, and fuzzy matching against known index entities.
Scales to 100+ documents; no hardcoded query patterns.
"""
import re
from difflib import SequenceMatcher
from typing import Optional

FUZZY_RATIO_THRESHOLD = 0.85  # Min similarity for fuzzy match (0-1)


# Patterns for explicit metadata in query (case-insensitive)
# Claim: alphanumeric with dashes (e.g. 503-WOP-01, CLM-8891); exclude common words like "status", "form"
PATTERNS = {
    "claim_number": [
        r"claim\s*(?:#|number)?\s*([\w\-]+)",
        r"clm[-\s]*([\w\-]+)",
        # Avoid "claim status" -> status; require digit or uppercase (claim IDs often have)
        r"claim\s+([A-Z0-9][\w\-]*)",
    ],
    "policy_number": [
        r"policy\s*(?:#|number)?\s*([\d][\d\w\-]*)",  # Must start with digit
        r"policy\s+([\d][\d\w\-]*)",
    ],
    "group_number": [
        r"group\s*(?:#|number)?\s*([\d\w\-]+)",
    ],
}

# Words that are not claim numbers (avoid false positives)
CLAIM_STOPWORDS = frozenset({"status", "form", "number", "info", "information", "details"})


def _normalize_claim(s: str) -> str:
    """Normalize claim number for comparison (uppercase, collapse spaces)."""
    return re.sub(r"\s+", "", (s or "").strip().upper())


def _normalize_policy(s: str) -> str:
    """Normalize policy/group number."""
    return re.sub(r"\s+", "", (s or "").strip())


def extract_metadata_from_query(
    query: str,
    known_patients: Optional[list[str]] = None,
    known_claims: Optional[list[str]] = None,
    known_policies: Optional[list[str]] = None,
    known_groups: Optional[list[str]] = None,
) -> dict:
    """
    Extract metadata filter from user query by:
    1. Matching explicit patterns (claim #, policy #, etc.)
    2. Matching patient names (exact or partial: "Rika" -> "Rika Popper")

    Returns dict with keys patient_name, claim_number, policy_number, group_number.
    Only includes non-empty values. Use result as metadata_filter for ChromaDB.
    """
    out = {}
    q = (query or "").strip()
    if not q:
        return out

    q_lower = q.lower()
    known_patients = known_patients or []
    known_claims = known_claims or []
    known_policies = known_policies or []
    known_groups = known_groups or []

    # 1. Explicit claim number
    for pat in PATTERNS["claim_number"]:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            if raw.lower() in CLAIM_STOPWORDS:
                continue
            if known_claims:
                norm = _normalize_claim(raw)
                for c in known_claims:
                    if _normalize_claim(c) == norm or norm in _normalize_claim(c):
                        out["claim_number"] = c
                        break
                if "claim_number" not in out:
                    out["claim_number"] = raw
            else:
                out["claim_number"] = raw
            if "claim_number" in out:
                break

    # 2. Explicit policy number
    for pat in PATTERNS["policy_number"]:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            if known_policies:
                norm = _normalize_policy(raw)
                for p in known_policies:
                    if _normalize_policy(p) == norm or norm in _normalize_policy(p):
                        out["policy_number"] = p
                        break
                if "policy_number" not in out:
                    out["policy_number"] = raw
            else:
                out["policy_number"] = raw
            break

    # 3. Explicit group number
    for pat in PATTERNS["group_number"]:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            out["group_number"] = raw
            break

    # 4. Patient name: match query against known patients (any phrasing)
    # - Exact substring: "Rika Popper" in query
    # - Partial: "Rika" or "Popper" in query
    # - Fuzzy: "Rika P", "Teresa B" -> high similarity to known name
    # - Token scan: any query token (3+ chars) that matches/fuzzy-matches a known patient
    if known_patients:
        best_match = None
        best_score = 0.0

        for patient in known_patients:
            if not patient:
                continue
            p_lower = patient.lower()
            score = 0.0

            # Exact: full name in query
            if p_lower in q_lower:
                score = 1.0
            # Partial: first or last name (3+ chars)
            elif any(len(p) >= 3 and p.lower() in q_lower for p in patient.split()):
                score = 0.9
            # Patient contains query substring (e.g. query "Rika" in "Rika Popper")
            elif any(q_lower in p_lower for p in patient.split() if len(p) >= 3):
                score = 0.85

            if score > best_score:
                best_score = score
                best_match = patient

        # Fuzzy: scan query tokens (words 3+ chars) against known patients
        if best_score < FUZZY_RATIO_THRESHOLD:
            tokens = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", q)
            for tok in tokens:
                t_lower = tok.lower()
                if t_lower in ("the", "and", "for", "all", "claim", "policy", "patient", "document"):
                    continue
                for patient in known_patients:
                    if not patient:
                        continue
                    # Token matches start of first/last name
                    for part in patient.split():
                        if len(part) >= 3:
                            r = SequenceMatcher(None, t_lower, part.lower()).ratio()
                            if r >= FUZZY_RATIO_THRESHOLD:
                                if r > best_score:
                                    best_score = r
                                    best_match = patient
                            # Substring: "Rika" in "Rika Popper"
                            elif t_lower in part.lower() and len(t_lower) >= 3:
                                if 0.8 > best_score:
                                    best_score = 0.8
                                    best_match = patient

        if best_match and best_score >= 0.8:
            out["patient_name"] = best_match

    return out


def merge_metadata_filters(
    user_filter: dict | None,
    query_extracted: dict | None,
    prefer_user: bool = True,
) -> dict | None:
    """
    Merge user-selected filter with auto-extracted from query.
    prefer_user: if True, user selection overrides query extraction for same key.
    Returns merged dict or None if empty.
    """
    merged = dict(query_extracted or {})
    uf = user_filter or {}
    for k, v in uf.items():
        if v:
            if prefer_user:
                merged[k] = v
            elif k not in merged:
                merged[k] = v
    return {k: v for k, v in merged.items() if v} or None


def get_index_metadata_catalog(chunks: list) -> dict:
    """
    Build catalog of unique metadata values from index chunks.
    Merges OCR variants (Rika Popper, Rita Peyer -> Rita Pepper).
    """
    from retrieval.agentic_rag import _merge_similar_patient_names, _normalize_name

    patient_counts: dict[str, int] = {}
    claims = set()
    policies = set()
    groups = set()
    for c in chunks or []:
        p = getattr(c, "patient_name", "") or ""
        if p:
            pn = _normalize_name(p)
            patient_counts[pn] = patient_counts.get(pn, 0) + 1
        cl = getattr(c, "claim_number", "") or ""
        if cl:
            claims.add(cl)
        pol = getattr(c, "policy_number", "") or ""
        if pol:
            policies.add(pol)
        g = getattr(c, "group_number", "") or ""
        if g:
            groups.add(g)
    doctors = set()
    for c in chunks or []:
        d = getattr(c, "doctor_name", "") or ""
        if d:
            doctors.add(d)
    known_patients, patient_name_aliases = _merge_similar_patient_names(patient_counts)
    return {
        "known_patients": known_patients,
        "patient_name_aliases": patient_name_aliases,
        "known_claims": sorted(claims),
        "known_policies": sorted(policies),
        "known_groups": sorted(groups),
        "known_doctors": sorted(doctors),
    }
