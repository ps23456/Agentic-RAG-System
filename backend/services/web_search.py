"""DuckDuckGo web search utility."""
import logging

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via DuckDuckGo and return structured results."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        results = []
        for r in raw:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
                "source": r.get("source", ""),
            })
        return results
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return []
