"""Public RAG API endpoint.

Exposes the internal RAG pipeline as a simple, business-friendly API surface:

    POST /query
        {
            "question": "What are the activity restrictions?",
            "patient": null,       # optional scope
            "file": null,          # optional scope (basename, e.g. "APS_AJude.pdf")
            "web_search": false,   # optional enrichment
            "stream": false        # set true for SSE token stream
        }

Response (blocking mode):
    {
        "answer": "...",
        "sources": [{"file": "...", "page": 4, "title": "..."}],
        "intent": "text_heavy",
        "elapsed_ms": 2480
    }

Response (stream=true): Server-Sent Events — `meta`, `token`*, `done`.

NOTE:
    This endpoint is intentionally unauthenticated so external graders and
    client apps can call it without a key. To require auth, add
    `_auth: None = Depends(require_api_key)` to each handler signature (same
    pattern used in `backend/routes/chat.py`).
"""
import asyncio
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["rag"])


MAX_QUESTION_CHARS = 2000


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    patient: str | None = None
    file: str | None = None
    web_search: bool = False
    stream: bool = False


class QuerySource(BaseModel):
    file: str = ""
    page: int | str | None = None
    title: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[QuerySource]
    intent: str = ""
    elapsed_ms: int = 0


def _normalize_sources(raw_sources: list[dict]) -> list[QuerySource]:
    """Convert internal {file_name, page, title} -> public {file, page, title}."""
    out: list[QuerySource] = []
    for s in raw_sources or []:
        out.append(
            QuerySource(
                file=s.get("file_name") or s.get("file") or "",
                page=s.get("page"),
                title=s.get("title") or s.get("section_title") or "",
            )
        )
    return out


def _sse_event(kind: str, data) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {kind}\ndata: {payload}\n\n"


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Blocking RAG query. Input: question. Output: answer + citations."""
    if req.stream:
        return await _stream_query(req)

    from backend.services.rag_service import rag

    t0 = time.time()
    try:
        result = await asyncio.to_thread(
            rag.chat,
            req.question,
            req.patient,
            req.web_search,
            req.file,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}") from e

    elapsed_ms = int((time.time() - t0) * 1000)
    return QueryResponse(
        answer=result.get("summary", ""),
        sources=_normalize_sources(result.get("sources", [])),
        intent=result.get("intent", ""),
        elapsed_ms=elapsed_ms,
    )


async def _stream_query(req: QueryRequest) -> StreamingResponse:
    """Streaming variant — mirrors /api/chat/stream but uses `question` field."""
    from backend.services.rag_service import rag

    async def iterator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            try:
                for kind, data in rag.stream_chat(
                    req.question,
                    req.patient,
                    req.web_search,
                    req.file,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, (kind, data))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)[:500]))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, (None, None))

        loop.run_in_executor(None, produce)

        while True:
            kind, data = await queue.get()
            if kind is None:
                break
            yield _sse_event(kind, data)

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
