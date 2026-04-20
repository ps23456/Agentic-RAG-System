"""Chat endpoint: query -> agentic RAG -> summary + sources + results."""
import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    conversation_id: str = ""
    patient_filter: str | None = None
    web_search: bool = False
    file_filter: str | None = None  # Restrict to this file (e.g. when Chat clicked on document)
    # When True, runs RAGAs Faithfulness + Answer relevancy after the answer (slower; Groq first, OpenAI fallback)
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
    # Present when evaluate_rag was true: metric name -> score (0–1)
    evaluation: dict[str, float] | None = None
    # When evaluate_rag was true but evaluation is null: brief reason for the UI
    evaluation_error: str | None = None
    # When some metrics are NaN (e.g. faithfulness): short explanation
    evaluation_notes: str | None = None


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    from backend.services.rag_service import rag
    result = rag.chat(
        req.query,
        req.patient_filter,
        web_search=req.web_search,
        file_filter=req.file_filter,
    )
    evaluation: dict[str, float] | None = None
    evaluation_error: str | None = None
    evaluation_notes: str | None = None
    if req.evaluate_rag:
        from backend.services.ragas_eval import run_single_turn_eval

        # Ragas uses nested asyncio; uvicorn may use uvloop, which forbids nested loops on
        # the main thread. Run evaluation in a worker thread (standard asyncio loop).
        evaluation, err, notes = await asyncio.to_thread(
            run_single_turn_eval,
            req.query,
            result.get("summary") or "",
            result.get("results") or [],
        )
        if evaluation is None and err:
            evaluation_error = err
        if notes:
            evaluation_notes = notes
    return ChatResponse(
        summary=result.get("summary", ""),
        sources=[Source(**s) for s in result.get("sources", [])],
        results=[ResultItem(**r) for r in result.get("results", [])],
        intent=result.get("intent", ""),
        reasoning=result.get("reasoning", ""),
        evaluation=evaluation,
        evaluation_error=evaluation_error,
        evaluation_notes=evaluation_notes,
    )
