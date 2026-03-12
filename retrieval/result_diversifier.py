"""
Metadata-aware result diversification.
Prevents same-patient/same-claim dominance in results so "list the patients"
and other queries return diverse, representative chunks across entities.
Improves ALL queries by ensuring coverage across patients/claims.
"""
from typing import List, Tuple, Any, Callable


def get_entity_key(chunk: Any, entity_key: str = "patient_name") -> str:
    """Extract entity value from chunk for diversity grouping."""
    val = getattr(chunk, entity_key, "") or ""
    return (val or "").strip() or "_no_entity_"


def diversify_by_metadata(
    results: List[Tuple[Any, float]],
    entity_key: str = "patient_name",
    top_k: int = 15,
    max_per_entity: int = 2,
    min_entity_coverage: bool = True,
) -> List[Tuple[Any, float]]:
    """
    Reorder results to favor diversity by entity (e.g. patient_name).
    Ensures no single patient dominates top results; surfaces chunks from different patients.

    - max_per_entity: max chunks from same patient in top_k (default 2)
    - min_entity_coverage: if True, prefer at least one chunk per known entity in pool
    """
    if not results:
        return []
    if top_k >= len(results):
        top_k = len(results)

    # Build entity -> [(chunk, score), ...] sorted by score desc
    by_entity: dict[str, List[Tuple[Any, float]]] = {}
    for chunk, score in results:
        key = get_entity_key(chunk, entity_key)
        if key not in by_entity:
            by_entity[key] = []
        by_entity[key].append((chunk, score))

    for key in by_entity:
        by_entity[key].sort(key=lambda x: x[1], reverse=True)

    # Greedy selection: round-robin by entity, taking best from each
    # First pass: take 1 from each entity (best score)
    # Second pass: take 2nd from each entity if we have room
    out: List[Tuple[Any, float]] = []
    entity_counts: dict[str, int] = {k: 0 for k in by_entity}
    pool = list(results)
    pool.sort(key=lambda x: x[1], reverse=True)

    # _no_entity_ (docs without a patient_name, e.g. annual reports) should NOT be
    # capped by max_per_entity — their scores should speak for themselves.
    NO_ENTITY = "_no_entity_"
    effective_max: dict[str, int] = {
        k: (top_k if k == NO_ENTITY else max_per_entity) for k in by_entity
    }

    # Round-robin: alternate entities, each taking best remaining
    entities_with_items = [k for k in by_entity if by_entity[k]]
    round_idx = 0
    while len(out) < top_k:
        added = False
        for key in entities_with_items:
            if entity_counts[key] >= effective_max[key]:
                continue
            items = by_entity[key]
            taken = entity_counts[key]
            if taken < len(items):
                chunk, score = items[taken]
                if (chunk, score) not in [(c, s) for c, s in out]:
                    out.append((chunk, score))
                    entity_counts[key] += 1
                    added = True
                    if len(out) >= top_k:
                        break
        if not added:
            break
        round_idx += 1

    # If we have room, fill with remaining best by score (any entity)
    if len(out) < top_k:
        seen_ids = {getattr(c, "chunk_id", id(c)) for c, _ in out}
        for chunk, score in pool:
            if len(out) >= top_k:
                break
            cid = getattr(chunk, "chunk_id", id(chunk))
            if cid not in seen_ids:
                key = get_entity_key(chunk, entity_key)
                if entity_counts.get(key, 0) < effective_max.get(key, max_per_entity):
                    out.append((chunk, score))
                    seen_ids.add(cid)
                    entity_counts[key] = entity_counts.get(key, 0) + 1

    return out[:top_k]


def diversify_fused_results(
    fused: List[dict],
    entity_key: str = "patient_name",
    top_k: int = 15,
    max_per_entity: int = 2,
) -> List[dict]:
    """
    Diversify fused Multimodal Hybrid results (list of {type, content, final_score}).
    content can be chunk (has patient_name) or image dict (may have patient_name in metadata).
    """
    if not fused:
        return fused[:top_k]

    def get_entity(r: dict) -> str:
        c = r.get("content")
        if c is None:
            return "_no_entity_"
        if hasattr(c, entity_key):
            return (getattr(c, entity_key, "") or "").strip() or "_no_entity_"
        if isinstance(c, dict):
            return (c.get(entity_key, "") or "").strip() or "_no_entity_"
        return "_no_entity_"

    by_entity: dict[str, List[dict]] = {}
    for r in fused:
        key = get_entity(r)
        if key not in by_entity:
            by_entity[key] = []
        by_entity[key].append(r)

    for key in by_entity:
        by_entity[key].sort(key=lambda x: x.get("final_score", 0), reverse=True)

    out: List[dict] = []
    entity_counts: dict[str, int] = {k: 0 for k in by_entity}
    entities_with_items = [k for k in by_entity if by_entity[k]]

    # _no_entity_ should not be capped — large docs without a patient get full allowance
    NO_ENTITY = "_no_entity_"
    fused_effective_max: dict[str, int] = {
        k: (top_k if k == NO_ENTITY else max_per_entity) for k in by_entity
    }

    while len(out) < top_k:
        added = False
        for key in entities_with_items:
            if entity_counts[key] >= fused_effective_max[key]:
                continue
            items = by_entity[key]
            taken = entity_counts[key]
            if taken < len(items):
                r = items[taken]
                if r not in out:
                    out.append(r)
                    entity_counts[key] += 1
                    added = True
                    if len(out) >= top_k:
                        break
        if not added:
            break

    if len(out) < top_k:
        seen = set(id(r.get("content")) for r in out)
        for r in sorted(fused, key=lambda x: x.get("final_score", 0), reverse=True):
            if len(out) >= top_k:
                break
            if id(r.get("content")) not in seen:
                key = get_entity(r)
                if entity_counts.get(key, 0) < fused_effective_max.get(key, max_per_entity):
                    out.append(r)
                    seen.add(id(r.get("content")))
                    entity_counts[key] = entity_counts.get(key, 0) + 1

    return out[:top_k]
