"""FastAPI backend for Insurance Claim Search."""
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for production-friendly logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "path"):
            payload["path"] = record.path
        if hasattr(record, "method"):
            payload["method"] = record.method
        if hasattr(record, "status_code"):
            payload["status_code"] = record.status_code
        if hasattr(record, "latency_ms"):
            payload["latency_ms"] = round(float(record.latency_ms), 2)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    formatter = JsonFormatter()
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


_configure_logging()
logger = logging.getLogger(__name__)

# Add project root to sys.path so existing modules resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from document_loader import set_mistral_ocr_key

set_mistral_ocr_key(os.environ.get("MISTRAL_OCR_API_KEY", "").strip())


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.db.tenant_store import tenant_store
    from backend.services.rag_service import rag
    tenant_store.initialize_schema()
    tenant_store.ensure_default_bootstrap(os.environ.get("BACKEND_API_KEY", "").strip())
    rag.initialize()
    try:
        tenant_store.sync_local_uploads_for_owner(
            tenant_id="tenant_default",
            user_id="user_default_admin",
            uploads_dir=rag.uploads_folder,
        )
    except Exception as e:
        logger.warning("legacy_upload_sync_failed: %s", e)
    yield


app = FastAPI(title="Insurance Claim Search API", lifespan=lifespan)


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.started_at = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.route_counts: dict[str, int] = defaultdict(int)
        self.chat_count = 0
        self.chat_total_latency_ms = 0.0

    def observe(self, path: str, latency_ms: float, status_code: int) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            self.route_counts[path] += 1
            if status_code >= 400:
                self.error_count += 1
            if path in ("/api/chat", "/api/chat/stream"):
                self.chat_count += 1
                self.chat_total_latency_ms += latency_ms

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = self.total_latency_ms / self.request_count if self.request_count else 0.0
            avg_chat_latency = self.chat_total_latency_ms / self.chat_count if self.chat_count else 0.0
            error_rate = self.error_count / self.request_count if self.request_count else 0.0
            top_routes = sorted(self.route_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
            return {
                "uptime_s": round(time.time() - self.started_at, 2),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": round(error_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "chat_request_count": self.chat_count,
                "chat_avg_latency_ms": round(avg_chat_latency, 2),
                "top_routes": [{"path": p, "count": c} for p, c in top_routes],
            }


metrics = RuntimeMetrics()

def _get_cors_allowed_origins() -> list[str]:
    """Return explicit CORS origins from env (comma-separated)."""
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if raw:
        origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
        # Credentials + wildcard is invalid per CORS spec; drop wildcard if mixed.
        if "*" in origins and len(origins) > 1:
            origins = [o for o in origins if o != "*"]
        return origins or ["http://localhost:3000"]
    # Safe local defaults for development without opening CORS globally.
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


cors_origins = _get_cors_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    status_code = 500
    tenant_ctx = None
    try:
        from backend.db.tenant_store import tenant_store

        api_key = (request.headers.get("X-API-Key", "") or "").strip()
        if api_key:
            tenant_ctx = tenant_store.resolve_api_key(api_key)
    except Exception:
        tenant_ctx = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception(
            "Unhandled request exception",
            extra={"path": request.url.path, "method": request.method, "status_code": 500},
        )
        raise
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        metrics.observe(request.url.path, latency_ms, status_code)
        if tenant_ctx is not None:
            try:
                from backend.db.tenant_store import tenant_store

                tenant_store.record_request_usage(
                    tenant_id=tenant_ctx.tenant_id,
                    user_id=tenant_ctx.user_id,
                    route=request.url.path,
                    status_code=status_code,
                    latency_ms=latency_ms,
                )
            except Exception:
                logger.exception("tenant_usage_record_failed")
        logger.info(
            "request_complete",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "latency_ms": latency_ms,
            },
        )

from backend.routes.chat import router as chat_router
from backend.routes.documents import router as documents_router
from backend.routes.upload import router as upload_router
from backend.routes.index import router as index_router
from backend.routes.medical import router as medical_router
from backend.routes.fields import router as fields_router
from backend.routes.query import router as query_router
from backend.routes.tenant_settings import router as tenant_settings_router
from backend.security import get_auth_context, require_api_key, require_scopes
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(upload_router)
app.include_router(index_router)
app.include_router(medical_router)
app.include_router(fields_router)
app.include_router(query_router)
app.include_router(tenant_settings_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/metrics")
async def metrics_summary(_auth = Depends(require_scopes("admin:read"))):
    """Minimal runtime metrics for production diagnostics."""
    return JSONResponse(metrics.snapshot())


@app.get("/api/metrics/tenant")
async def tenant_metrics(days: int = 7, ctx=Depends(require_scopes("admin:read"))):
    """Tenant-scoped usage summary for monitoring and billing readiness."""
    from backend.db.tenant_store import tenant_store

    summary = tenant_store.get_tenant_usage_summary(ctx.tenant_id, days=days)
    grouped = tenant_store.get_tenant_usage_grouped(ctx.tenant_id, days=days)
    return {"tenant_id": ctx.tenant_id, **summary, "by_group": grouped}


@app.get("/api/metrics/tenant/export")
async def tenant_metrics_export(
    days: int = 7,
    format: str = "json",
    ctx=Depends(require_scopes("admin:read")),
):
    """Export tenant usage rows for billing/reporting (json or csv)."""
    from backend.db.tenant_store import tenant_store

    rows = tenant_store.export_tenant_usage_rows(ctx.tenant_id, days=days)
    grouped = tenant_store.get_tenant_usage_grouped(ctx.tenant_id, days=days)
    if (format or "").strip().lower() == "csv":
        csv_data = tenant_store.usage_rows_to_csv(rows)
        return Response(
            content=csv_data,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="tenant_usage_{ctx.tenant_id}_{days}d.csv"'
            },
        )
    return {
        "tenant_id": ctx.tenant_id,
        "days": max(1, min(int(days), 90)),
        "rows": rows,
        "by_group": grouped,
    }


@app.get("/api/auth/context")
async def auth_context(ctx=Depends(require_scopes("auth:read"))):
    """Debug endpoint: returns resolved tenant/user context for current key."""
    return {
        "tenant_id": ctx.tenant_id,
        "tenant_slug": ctx.tenant_slug,
        "user_id": ctx.user_id,
        "user_email": ctx.user_email,
        "key_id": ctx.key_id,
        "key_label": ctx.key_label,
        "source": ctx.source,
    }


@app.get("/api/auth/registry-stats")
async def auth_registry_stats(_ctx=Depends(require_scopes("admin:read"))):
    """Debug endpoint: shows tenant registry table counts used by this process."""
    from backend.db.tenant_store import tenant_store

    return tenant_store.debug_stats()
