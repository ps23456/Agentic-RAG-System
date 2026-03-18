"""FastAPI backend for Insurance Claim Search."""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path so existing modules resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

os.environ.setdefault("MISTRAL_OCR_API_KEY", "6UtP02sadSKnj7Gt6eWBJfH8MaHKWevW")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.services.rag_service import rag
    rag.initialize()
    yield


app = FastAPI(title="Insurance Claim Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.routes.chat import router as chat_router
from backend.routes.documents import router as documents_router
from backend.routes.upload import router as upload_router
from backend.routes.index import router as index_router
from backend.routes.medical import router as medical_router

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(upload_router)
app.include_router(index_router)
app.include_router(medical_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
