"""Security helpers for protected API endpoints."""
import os
import secrets

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Validate API key for protected endpoints.

    Configure with BACKEND_API_KEY in the environment.
    """
    expected = os.environ.get("BACKEND_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Server auth is not configured. Set BACKEND_API_KEY.",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
