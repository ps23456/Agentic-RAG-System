"""Tenant-level settings endpoints (BYOK provider/key/model)."""
import re
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.db.tenant_store import tenant_store
from backend.security import require_scopes

router = APIRouter()

_ALLOWED_PROVIDERS = {"", "groq", "openai", "gemini"}
_ALLOWED_MODES = {"platform_default", "tenant_byok"}


class TenantSettingsUpdate(BaseModel):
    llm_mode: str = "platform_default"
    llm_provider: str = ""
    llm_api_key: str = ""
    llm_model: str = ""


class TenantSettingsTestRequest(BaseModel):
    llm_provider: str = ""
    llm_api_key: str = ""
    llm_model: str = ""


def _sanitize_provider_error(err: Exception) -> str:
    raw = (str(err) or "").strip().lower()
    if not raw:
        return "provider_connection_failed"
    if "incorrect api key" in raw or "invalid api key" in raw or "unauthorized" in raw or "401" in raw:
        return "invalid_api_key"
    if "rate limit" in raw or "quota" in raw or "429" in raw:
        return "rate_limited_or_quota_exceeded"
    if "model" in raw and ("not found" in raw or "does not exist" in raw or "unsupported" in raw):
        return "invalid_or_unsupported_model"
    if "timeout" in raw or "timed out" in raw:
        return "provider_timeout"
    if "connection" in raw or "network" in raw:
        return "provider_connection_failed"
    # Strip obvious key-like fragments from any unknown provider message.
    cleaned = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", str(err))
    cleaned = re.sub(r"[A-Za-z0-9_-]{24,}", "***", cleaned)
    return cleaned[:80] or "provider_connection_failed"


def _test_provider_connection(provider: str, api_key: str, model: str) -> tuple[bool, str]:
    p = (provider or "").strip().lower()
    key = (api_key or "").strip()
    m = (model or "").strip()
    if not key:
        return False, "llm_api_key is required"
    if p == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=key)
        r = client.chat.completions.create(
            model=m or "gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            temperature=0,
        )
        return bool(r.choices), "openai_connection_ok"
    if p == "groq":
        from groq import Groq

        client = Groq(api_key=key)
        r = client.chat.completions.create(
            model=m or "llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            temperature=0,
        )
        return bool(r.choices), "groq_connection_ok"
    if p == "gemini":
        try:
            from google import genai

            client = genai.Client(api_key=key, http_options={"api_version": "v1"})
            out = client.models.generate_content(model=m or "gemini-1.5-flash", contents="ping")
            return bool(getattr(out, "text", None)), "gemini_connection_ok"
        except Exception:
            import google.generativeai as genai

            genai.configure(api_key=key)
            model_obj = genai.GenerativeModel(m or "gemini-1.5-flash")
            out = model_obj.generate_content("ping")
            return bool(getattr(out, "text", None)), "gemini_connection_ok"
    return False, "unsupported_provider"


@router.get("/api/tenant/settings")
async def get_tenant_settings(auth=Depends(require_scopes("admin:read"))):
    settings = tenant_store.get_tenant_settings(auth.tenant_id)
    raw_key = settings.get("llm_api_key", "")
    return {
        "tenant_id": auth.tenant_id,
        "llm_mode": settings.get("llm_mode", "platform_default"),
        "llm_provider": settings.get("llm_provider", ""),
        "llm_model": settings.get("llm_model", ""),
        "has_llm_api_key": bool(raw_key),
        "llm_api_key_masked": (f"{raw_key[:4]}...{raw_key[-4:]}" if len(raw_key) >= 8 else ""),
        "encryption_ready": bool(settings.get("encryption_ready")),
    }


@router.put("/api/tenant/settings")
async def update_tenant_settings(
    req: TenantSettingsUpdate,
    auth=Depends(require_scopes("admin:write")),
):
    mode = (req.llm_mode or "platform_default").strip().lower()
    if mode not in _ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Unsupported llm_mode. Use one of: platform_default, tenant_byok")
    provider = (req.llm_provider or "").strip().lower()
    if provider not in _ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported llm_provider. Use one of: groq, openai, gemini")
    if mode == "tenant_byok":
        if not provider:
            raise HTTPException(status_code=400, detail="llm_provider is required when llm_mode=tenant_byok")
        if not (req.llm_api_key or "").strip():
            raise HTTPException(status_code=400, detail="llm_api_key is required when llm_mode=tenant_byok")
    try:
        saved = tenant_store.upsert_tenant_settings(
            tenant_id=auth.tenant_id,
            llm_mode=mode,
            llm_provider=provider,
            llm_api_key=req.llm_api_key or "",
            llm_model=req.llm_model or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "tenant_id": auth.tenant_id,
        "llm_mode": saved.get("llm_mode", "platform_default"),
        "llm_provider": saved.get("llm_provider", ""),
        "llm_model": saved.get("llm_model", ""),
        "has_llm_api_key": bool(saved.get("llm_api_key", "")),
    }


@router.post("/api/tenant/settings/test-connection")
async def test_tenant_settings_connection(
    req: TenantSettingsTestRequest,
    _auth=Depends(require_scopes("admin:write")),
):
    provider = (req.llm_provider or "").strip().lower()
    if provider not in {"groq", "openai", "gemini"}:
        raise HTTPException(status_code=400, detail="Unsupported llm_provider. Use one of: groq, openai, gemini")
    if not (req.llm_api_key or "").strip():
        raise HTTPException(status_code=400, detail="llm_api_key is required for test-connection")
    t0 = time.perf_counter()
    try:
        ok, detail = _test_provider_connection(provider, req.llm_api_key, req.llm_model)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if not ok:
            return {
                "ok": False,
                "provider": provider,
                "model": (req.llm_model or "").strip(),
                "latency_ms": latency_ms,
                "detail": detail,
            }
        return {
            "ok": True,
            "provider": provider,
            "model": (req.llm_model or "").strip(),
            "latency_ms": latency_ms,
            "detail": detail,
        }
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "ok": False,
            "provider": provider,
            "model": (req.llm_model or "").strip(),
            "latency_ms": latency_ms,
            "detail": _sanitize_provider_error(e),
        }

