"""Optional RAGAs metrics for a single chat turn (requires ragas).

Uses the same LLM priority as RAGService: Groq first (OpenAI-compatible endpoint),
then OpenAI if Groq fails or is not configured.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Match retrieval/agentic_rag.py
GROQ_OPENAI_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"


def _build_contexts(results: list[dict], top_k: int) -> list[str]:
    out: list[str] = []
    for r in results[:top_k]:
        if r.get("type") == "web":
            t = (r.get("section_title") or r.get("file_name") or "") + "\n" + (r.get("snippet") or "")
        else:
            fn = r.get("file_name", "")
            pg = r.get("page", "")
            sn = (r.get("snippet") or "").strip()
            t = f"{fn} (page {pg})\n{sn}"
        if t.strip():
            out.append(t[:8000])
    if not out:
        out = ["(no retrieved contexts)"]
    return out


def _llm_attempts() -> list[tuple[Any, str, str]]:
    """Ordered (client, model_name, label) — Groq first, then OpenAI when both keys exist."""
    from openai import OpenAI

    out: list[tuple[Any, str, str]] = []
    groq = os.environ.get("GROQ_API_KEY", "").strip()
    oai = os.environ.get("OPENAI_API_KEY", "").strip()
    if groq:
        out.append(
            (
                OpenAI(api_key=groq, base_url=GROQ_OPENAI_BASE),
                GROQ_MODEL,
                "groq",
            )
        )
    if oai:
        out.append((OpenAI(api_key=oai), OPENAI_MODEL, "openai"))
    return out


def _scores_from_evaluate_result(result: Any) -> tuple[dict[str, float] | None, str | None]:
    """Extract float scores; Ragas uses NaN when a metric could not be computed (e.g. faithfulness with no statements)."""
    if not getattr(result, "scores", None) or len(result.scores) == 0:
        return None, None
    row = result.scores[0]
    out: dict[str, float] = {}
    nan_metrics: list[str] = []
    for k, v in row.items():
        if v is None:
            nan_metrics.append(k)
            continue
        if str(v).lower() == "nan":
            nan_metrics.append(k)
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    note: str | None = None
    if nan_metrics:
        note = (
            "No numeric score for: "
            + ", ".join(nan_metrics)
            + ". (Faithfulness is often NaN when Ragas could not split the answer into statements to check.)"
        )
    return (out if out else None, note)


def _run_evaluate_once(
    ds: Any,
    client: Any,
    model_name: str,
    emb: Any,
) -> tuple[dict[str, float] | None, str | None]:
    # Use legacy metric singletons (faithfulness, answer_relevancy). The `evaluate()`
    # API rejects ragas.metrics.collections.* BaseMetric subclasses with:
    # "All metrics must be initialised metric objects, e.g: metrics=[BleuScore(), AspectCritic()]"
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import answer_relevancy, faithfulness

    llm = llm_factory(model_name, client=client)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=emb,
        show_progress=False,
    )
    return _scores_from_evaluate_result(result)


def run_single_turn_eval(
    query: str,
    summary: str,
    results: list[dict[str, Any]],
    top_k: int = 8,
) -> tuple[dict[str, float] | None, str | None]:
    """
    Run Faithfulness + Answer relevancy on one turn.
    Returns (scores, None) on success, or (None, short_error) if skipped / failed.
    """
    attempts = _llm_attempts()
    if not attempts:
        msg = "Set GROQ_API_KEY or OPENAI_API_KEY on the server (.env)."
        logger.warning("RAGAs eval skipped: %s", msg)
        return None, msg, None
    try:
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
    except ImportError as e:
        logger.warning("RAGAs eval skipped: %s", e)
        return None, f"Missing dependency: {e}", None

    summary = (summary or "").strip() or "(empty summary)"
    ctxs = _build_contexts(results, top_k)

    try:
        ds = Dataset.from_dict(
            {
                "user_input": [query],
                "response": [summary],
                "retrieved_contexts": [ctxs],
            }
        )
        # LangChain embeddings expose embed_query/embed_documents; ragas.metrics.answer_relevancy requires that.
        emb = LCHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning("RAGAs eval setup failed: %s", e)
        return None, f"Setup failed: {e}", None

    last_err: Exception | None = None
    last_partial_note: str | None = None
    for client, model_name, label in attempts:
        try:
            scores, partial_note = _run_evaluate_once(ds, client, model_name, emb)
            if scores is not None:
                return scores, None, partial_note
            # Groq often "succeeds" but returns NaN for every metric when rate-limited / LLM failed
            # inside Ragas — do NOT stop here; try OpenAI (or next provider) if configured.
            last_partial_note = partial_note
            logger.warning(
                "RAGAs with %s returned no numeric scores (%s); trying next LLM if available",
                label,
                partial_note or "empty",
            )
            continue
        except Exception as e:
            last_err = e
            logger.warning("RAGAs eval with %s failed: %s", label, e)
            continue
    if last_err:
        logger.warning("RAGAs eval failed after retries: %s", last_err)
        return None, str(last_err)[:500], None
    if last_partial_note:
        return None, None, last_partial_note
    return None, "RAGAs returned no scores (all LLM attempts produced empty metrics).", None
