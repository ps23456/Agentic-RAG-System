"""Chat endpoint: query -> agentic RAG -> summary + sources + results."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    conversation_id: str = ""
    patient_filter: str | None = None
    web_search: bool = False


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


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    from backend.services.rag_service import rag
    result = rag.chat(req.query, req.patient_filter, web_search=req.web_search)
    return ChatResponse(
        summary=result.get("summary", ""),
        sources=[Source(**s) for s in result.get("sources", [])],
        results=[ResultItem(**r) for r in result.get("results", [])],
        intent=result.get("intent", ""),
        reasoning=result.get("reasoning", ""),
    )
