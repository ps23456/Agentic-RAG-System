#!/usr/bin/env python3
"""
Run RAGAs metrics (Faithfulness, Answer relevancy) against the same RAG path as production.

Requires Python 3.10+ (tested with 3.12). Install: pip install ragas datasets

Environment:
  GROQ_API_KEY — preferred for RAGAs judge (llama-3.3-70b-versatile via Groq OpenAI-compatible API).
  OPENAI_API_KEY — fallback if Groq is missing or fails (gpt-4o-mini).
  Embeddings for answer relevancy use HuggingFace MiniLM (same family as search index).

Usage (from project root):
  .venv_bge/bin/python scripts/run_ragas_eval.py
  .venv_bge/bin/python scripts/run_ragas_eval.py --queries eval/gold_queries.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


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


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAs evaluation on ISR RAGService.chat")
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="JSON file: list of objects with key 'query' (optional: file_filter, patient_filter)",
    )
    parser.add_argument("--top-k", type=int, default=8, help="How many result snippets to pass as contexts")
    parser.add_argument("--max-queries", type=int, default=20, help="Cap number of queries from file")
    parser.add_argument("--dry-run", action="store_true", help="Only run chat; skip RAGAs (no OPENAI key needed)")
    args = parser.parse_args()

    groq = (os.environ.get("GROQ_API_KEY") or "").strip()
    oai = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not groq and not oai and not args.dry_run:
        print("ERROR: Set GROQ_API_KEY and/or OPENAI_API_KEY in .env for RAGAs judge LLM.", file=sys.stderr)
        print("       Use --dry-run to only test retrieval + summary without RAGAs scores.", file=sys.stderr)
        return 1

    from backend.services.rag_service import rag

    rag.initialize()

    if args.queries and os.path.isfile(args.queries):
        with open(args.queries, encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict) and "queries" in items:
            items = items["queries"]
    else:
        items = [
            {"query": "What is the main topic of the indexed documents?"},
        ]

    rows: list[dict] = []
    for i, item in enumerate(items[: args.max_queries]):
        q = (item.get("query") or item.get("question") or "").strip()
        if not q:
            continue
        ff = item.get("file_filter")
        pf = item.get("patient_filter")
        print(f"[{i+1}] Query: {q[:80]}...")
        out = rag.chat(q, patient_filter=pf, web_search=False, file_filter=ff)
        summary = (out.get("summary") or "").strip()
        results = out.get("results") or []
        ctxs = _build_contexts(results, args.top_k)
        rows.append(
            {
                "user_input": q,
                "response": summary or "(empty summary)",
                "retrieved_contexts": ctxs,
            }
        )
        if args.dry_run:
            print(f"    summary chars: {len(summary)}, results: {len(results)}")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import answer_relevancy, faithfulness

    from backend.services.ragas_eval import _llm_attempts

    ds = Dataset.from_dict(
        {
            "user_input": [r["user_input"] for r in rows],
            "response": [r["response"] for r in rows],
            "retrieved_contexts": [r["retrieved_contexts"] for r in rows],
        }
    )

    emb = LCHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    last_err: Exception | None = None
    for client, model_name, label in _llm_attempts():
        try:
            llm = llm_factory(model_name, client=client)
            result = evaluate(
                ds,
                metrics=[faithfulness, answer_relevancy],
                llm=llm,
                embeddings=emb,
            )
            print("\n=== RAGAs aggregate scores (0–1, higher is better) ===")
            print(f"(LLM: {label} / {model_name})\n")
            print(result)
            return 0
        except Exception as e:
            last_err = e
            print(f"RAGAs with {label} failed: {e}", file=sys.stderr)
    print(f"ERROR: RAGAs failed after all LLM attempts: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
