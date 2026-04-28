"""Security helpers for protected API endpoints."""
from __future__ import annotations

import os
import secrets

from fastapi import Depends, Header, HTTPException, Request

from backend.db.tenant_store import AuthContext, DEFAULT_ROLE, DEFAULT_SCOPES, tenant_store


def _env_key_context(raw_api_key: str, expected: str) -> AuthContext | None:
    if not raw_api_key or not expected:
        return None
    if not secrets.compare_digest(raw_api_key, expected):
        return None
    return AuthContext(
        tenant_id="tenant_default",
        tenant_slug="default",
        user_id="user_default_admin",
        user_email="admin@local",
        key_id="key_env_fallback",
        key_label="env-backend-api-key",
        role=DEFAULT_ROLE,
        scopes=(DEFAULT_SCOPES,),
        source="env_fallback",
    )


def get_auth_context(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> AuthContext:
    """Resolve request API key to tenant/user auth context.

    Resolution order:
    1) Key registry (SQLite tenant store)
    2) Backward-compatible BACKEND_API_KEY fallback
    """
    expected = os.environ.get("BACKEND_API_KEY", "").strip()
    if not x_api_key:
        if not expected:
            raise HTTPException(
                status_code=503,
                detail="Server auth is not configured. Set BACKEND_API_KEY.",
            )
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    ctx = tenant_store.resolve_api_key(x_api_key)
    if not ctx:
        ctx = _env_key_context(x_api_key, expected)
    if not ctx:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    request.state.auth = ctx
    return ctx


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Compatibility guard for endpoints that only need auth validation."""
    _ = get_auth_context(request=request, x_api_key=x_api_key)


def require_scopes(*required_scopes: str):
    """FastAPI dependency factory for scope authorization.

    Example:
      auth = Depends(require_scopes("docs:write"))
    """

    def _dep(ctx: AuthContext = Depends(get_auth_context)):
        missing = [s for s in required_scopes if not ctx.has_scope(s)]
        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient scope. Required: {', '.join(required_scopes)}",
            )
        return ctx

    return _dep
