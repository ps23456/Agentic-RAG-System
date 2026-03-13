"""
Multimodal Hybrid RAG: query classification, text + image retrieval, normalized score fusion.
Debug mode prints query type, top 5 text/image results, fused ranking, and fusion weights.
"""
import argparse
import logging
import sys

from config import (
    DATA_FOLDER,
    MULTIMODAL_HYBRID_WEIGHTS,
    MULTIMODAL_HYBRID_TOP_K,
)
from indexing.text_indexer import build_text_index
from indexing.image_indexer import build_image_index
from retrieval.query_classifier import QueryClassifier, classify_query
from retrieval.text_retriever import TextRetriever
from retrieval.image_retriever import ImageRetriever
from retrieval.hybrid_fusion import fuse_results, boost_phrase_matching

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ensure_indices(data_folder: str | None = None, rebuild_image: bool = False):
    """Build or ensure text and image indices exist."""
    data_folder = data_folder or DATA_FOLDER
    text_index = build_text_index(data_folder)
    if rebuild_image:
        build_image_index(data_folder)
    return text_index


def run_multimodal_hybrid(
    query: str,
    text_index,
    top_k: int | None = None,
    debug: bool = True,
) -> list:
    """
    Run full pipeline: classify -> text retrieve -> image retrieve -> normalize -> fuse.
    Returns list of fused items (each has type, content, normalized_score, final_score).
    """
    top_k = top_k or MULTIMODAL_HYBRID_TOP_K
    classifier = QueryClassifier()
    query_type = classify_query(query, classifier)
    weights = MULTIMODAL_HYBRID_WEIGHTS.get(query_type, MULTIMODAL_HYBRID_WEIGHTS["hybrid"])
    w_text, w_image = weights

    text_retriever = TextRetriever(text_index)
    image_retriever = ImageRetriever()

    text_results = text_retriever.retrieve(query, top_k=top_k)
    if text_index and hasattr(text_index, "verbatim_search"):
        verbatim = text_index.verbatim_search(query, None, max_results=10)
        seen_v = {c.chunk_id for c, _ in verbatim}
        text_results = list(verbatim) + [(c, s) for c, s in text_results if c.chunk_id not in seen_v]
    text_results = boost_phrase_matching(text_results, query)
    image_results = image_retriever.retrieve(query, top_n=20)

    fused = fuse_results(text_results, image_results, query_type, top_k=top_k)

    if debug:
        print("\n" + "=" * 60 + " DEBUG " + "=" * 60)
        print("Query type:", query_type)
        print("Fusion weights: text=%.2f image=%.2f" % (w_text, w_image))
        print("-" * 60)
        print("Top 5 TEXT results (chunk, score):")
        for i, (chunk, score) in enumerate(text_results[:5], 1):
            snippet = (chunk.text or "")[:120].replace("\n", " ")
            print("  %d. %.4f  %s" % (i, score, snippet + ("..." if len(chunk.text or "") > 120 else "")))
        print("-" * 60)
        print("Top 5 IMAGE results (item, score):")
        for i, (item, score) in enumerate(image_results[:5], 1):
            print("  %d. %.4f  %s (page %s)" % (i, score, item.get("file_name", ""), item.get("page", "")))
        print("-" * 60)
        print("Final fused ranking (top %d):" % len(fused))
        for i, row in enumerate(fused[:10], 1):
            t = row["type"]
            s = row["final_score"]
            if t == "text":
                content = (row["content"].text or "")[:80].replace("\n", " ")
                print("  %d. [%s] %.4f  %s" % (i, t, s, content + ("..." if len(row["content"].text or "") > 80 else "")))
            else:
                print("  %d. [%s] %.4f  %s" % (i, t, s, row["content"].get("file_name", "")))
        print("=" * 60 + "\n")

    return fused


def main():
    parser = argparse.ArgumentParser(description="Multimodal Hybrid RAG: text + image retrieval with score fusion")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--data", default=None, help="Data folder (default: config DATA_FOLDER)")
    parser.add_argument("--rebuild-image", action="store_true", help="Rebuild image index before query")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug output")
    parser.add_argument("--top-k", type=int, default=None, help="Final top-K results (default: config)")
    args = parser.parse_args()

    query = args.query or "insurance claim coverage"
    text_index = ensure_indices(data_folder=args.data, rebuild_image=args.rebuild_image)
    results = run_multimodal_hybrid(
        query,
        text_index,
        top_k=args.top_k,
        debug=not args.no_debug,
    )
    print("Total results:", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
