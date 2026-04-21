"""Chat endpoints.

- POST /api/chat           : legacy non-streaming chat (kept for back-compat).
- POST /api/chat/stream    : streaming chat (SSE). First emits a `meta` event with
                             sources/results/intent, then a series of `token`
                             events with summary deltas, then a final `done`
                             event with the full summary. This is what the UI
                             uses — user sees text in ~2s instead of waiting
                             for the full answer.
- POST /api/chat/evaluate  : async RAGAs evaluation for a completed chat turn.
                             The UI calls this AFTER streaming completes so the
                             long-running Ragas metrics never block the answer.
"""
import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.security import require_api_key

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    conversation_id: str = ""
    patient_filter: str | None = None
    web_search: bool = False
    file_filter: str | None = None
    # Legacy flag — ignored by /api/chat and /api/chat/stream. UI should call
    # /api/chat/evaluate separately once the answer is ready.
    evaluate_rag: bool = False


class Source(BaseModel):
    file_name: str = ""
    page: int | str | None = None
    title: str = ""


class ResultItem(BaseModel):
    type: str = "text"
    file_name: str = ""
    page: int | str | None = None
    score: float = 0
    snippet: str = ""
    patient_name: str = ""
    section_title: str = ""
    is_pdf_page: bool = False
    path: str = ""
    url: str = ""


class ChatResponse(BaseModel):
    summary: str = ""
    sources: list[Source] = []
    results: list[ResultItem] = []
    intent: str = ""
    reasoning: str = ""


class EvaluateRequest(BaseModel):
    query: str
    summary: str
    results: list[dict] = []


class EvaluateResponse(BaseModel):
    evaluation: dict[str, float] | None = None
    evaluation_error: str | None = None
    evaluation_notes: str | None = None


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _auth: None = Depends(require_api_key)):
    """Non-streaming chat. Kept for back-compat; prefer /api/chat/stream."""
    from backend.services.rag_service import rag
    result = await asyncio.to_thread(
        rag.chat,
        req.query,
        req.patient_filter,
        req.web_search,
        req.file_filter,
    )
    return ChatResponse(
        summary=result.get("summary", ""),
        sources=[Source(**s) for s in result.get("sources", [])],
        results=[ResultItem(**r) for r in result.get("results", [])],
        intent=result.get("intent", ""),
        reasoning=result.get("reasoning", ""),
    )


def _sse_event(kind: str, data) -> str:
    """Format a single SSE message. Uses `event:` + JSON `data:`."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {kind}\ndata: {payload}\n\n"


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, _auth: None = Depends(require_api_key)):
    """Streaming chat. Yields SSE events: meta, token..., done (or error)."""
    from backend.services.rag_service import rag

    async def iterator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            try:
                for kind, data in rag.stream_chat(
                    req.query,
                    req.patient_filter,
                    req.web_search,
                    req.file_filter,
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


@router.post("/api/chat/evaluate", response_model=EvaluateResponse)
async def chat_evaluate(req: EvaluateRequest, _auth: None = Depends(require_api_key)):
    """Run RAGAs faithfulness + answer relevancy on a completed turn.

    Called asynchronously by the UI AFTER the streamed answer is displayed,
    so the long-running evaluation never blocks the user.
    """
    from backend.services.ragas_eval import run_single_turn_eval

    evaluation, err, notes = await asyncio.to_thread(
        run_single_turn_eval,
        req.query,
        req.summary,
        req.results,
    )
    return EvaluateResponse(
        evaluation=evaluation,
        evaluation_error=err,
        evaluation_notes=notes,
    )
