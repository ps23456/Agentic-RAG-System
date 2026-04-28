"""Tenant/auth metadata store for Stage 1 multi-tenant foundation.

This module intentionally uses the stdlib `sqlite3` driver to avoid adding new
runtime dependencies during Stage 1 rollout. It stores only metadata and API
key hashes; raw API keys are never persisted.
"""
from __future__ import annotations

import csv
import hashlib
import mimetypes
import os
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

DEFAULT_ROLE = "owner"
DEFAULT_SCOPES = "admin:*"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    tenant_slug: str
    user_id: str
    user_email: str
    key_id: str
    key_label: str
    role: str
    scopes: tuple[str, ...]
    source: str  # "registry" | "env_fallback"

    def has_scope(self, required_scope: str) -> bool:
        if not required_scope:
            return True
        if "admin:*" in self.scopes or "*" in self.scopes:
            return True
        if required_scope in self.scopes:
            return True
        if ":" in required_scope:
            ns, _action = required_scope.split(":", 1)
            if f"{ns}:*" in self.scopes:
                return True
        return False


class TenantStore:
    """Lightweight metadata DB for tenants/users/keys.

    Environment variables:
      TENANT_DB_PATH: optional sqlite path override
    """

    def __init__(self, db_path: str | None = None) -> None:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "storage",
            "tenant_registry.db",
        )
        self.db_path = db_path or os.environ.get("TENANT_DB_PATH", default_path)
        self._lock = threading.Lock()
        self._fernet = self._build_fernet()

    @staticmethod
    def _build_fernet() -> Optional[Fernet]:
        """Build Fernet from env key.

        TENANT_SETTINGS_MASTER_KEY must be a urlsafe-base64 32-byte key.
        """
        raw = os.environ.get("TENANT_SETTINGS_MASTER_KEY", "").strip()
        if not raw:
            return None
        try:
            return Fernet(raw.encode("utf-8"))
        except Exception:
            return None

    def _encrypt_secret(self, value: str) -> str:
        v = (value or "").strip()
        if not v:
            return ""
        if not self._fernet:
            raise ValueError("TENANT_SETTINGS_MASTER_KEY is required for BYOK storage")
        token = self._fernet.encrypt(v.encode("utf-8")).decode("utf-8")
        return f"enc:v1:{token}"

    def _decrypt_secret(self, value: str) -> str:
        v = (value or "").strip()
        if not v:
            return ""
        if not v.startswith("enc:v1:"):
            # legacy/plaintext row from pre-hardening; return as-is for migration path
            return v
        if not self._fernet:
            return ""
        token = v[len("enc:v1:") :]
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            return ""

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, col: str, col_ddl: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in rows}
        if col in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_ddl}")

    @staticmethod
    def _parse_scopes(raw: str | None) -> tuple[str, ...]:
        if not raw:
            return (DEFAULT_SCOPES,)
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if not parts:
            return (DEFAULT_SCOPES,)
        # preserve order, drop duplicates
        seen: set[str] = set()
        out: list[str] = []
        for p in parts:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return tuple(out)

    def initialize_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS tenants (
                        id TEXT PRIMARY KEY,
                        slug TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        email TEXT NOT NULL,
                        display_name TEXT NOT NULL DEFAULT '',
                        role TEXT NOT NULL DEFAULT 'member',
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE (tenant_id, email),
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS api_keys (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        key_hash TEXT NOT NULL UNIQUE,
                        role TEXT NOT NULL DEFAULT 'owner',
                        scopes TEXT NOT NULL DEFAULT 'admin:*',
                        status TEXT NOT NULL DEFAULT 'active',
                        expires_at TEXT,
                        revoked_at TEXT,
                        last_used_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        customer_id TEXT NOT NULL DEFAULT 'default',
                        file_name TEXT NOT NULL,
                        storage_uri TEXT NOT NULL DEFAULT '',
                        mime_type TEXT NOT NULL DEFAULT '',
                        size_bytes INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'active',
                        index_status TEXT NOT NULL DEFAULT 'pending',
                        index_error TEXT NOT NULL DEFAULT '',
                        indexed_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS index_jobs (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        details_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS tenant_settings (
                        tenant_id TEXT PRIMARY KEY,
                        llm_mode TEXT NOT NULL DEFAULT 'platform_default',
                        llm_provider TEXT NOT NULL DEFAULT '',
                        llm_api_key TEXT NOT NULL DEFAULT '',
                        llm_model TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS tenant_daily_usage (
                        day TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        route TEXT NOT NULL,
                        request_count INTEGER NOT NULL DEFAULT 0,
                        error_count INTEGER NOT NULL DEFAULT 0,
                        total_latency_ms REAL NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (day, tenant_id, user_id, route),
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS delete_audit_logs (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        customer_id TEXT NOT NULL DEFAULT '',
                        document_id TEXT NOT NULL DEFAULT '',
                        file_name TEXT NOT NULL DEFAULT '',
                        action TEXT NOT NULL DEFAULT 'delete_document',
                        result TEXT NOT NULL DEFAULT 'unknown',
                        details_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users (tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id ON api_keys (tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents (tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id);
                    CREATE INDEX IF NOT EXISTS idx_chat_sessions_tenant_id ON chat_sessions (tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_tenant_id ON chat_messages (tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages (session_id);
                    CREATE INDEX IF NOT EXISTS idx_index_jobs_tenant_id ON index_jobs (tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_tenant_usage_tenant_day ON tenant_daily_usage (tenant_id, day);
                    CREATE INDEX IF NOT EXISTS idx_tenant_usage_tenant_route_day ON tenant_daily_usage (tenant_id, route, day);
                    CREATE INDEX IF NOT EXISTS idx_delete_audit_tenant_id ON delete_audit_logs (tenant_id);
                    CREATE INDEX IF NOT EXISTS idx_delete_audit_created_at ON delete_audit_logs (created_at);
                    """
                )
                # Backward-compatible migration for existing DB files.
                self._ensure_column(conn, "api_keys", "role", "TEXT NOT NULL DEFAULT 'owner'")
                self._ensure_column(conn, "api_keys", "scopes", "TEXT NOT NULL DEFAULT 'admin:*'")
                self._ensure_column(conn, "api_keys", "expires_at", "TEXT")
                self._ensure_column(conn, "api_keys", "revoked_at", "TEXT")
                self._ensure_column(conn, "documents", "customer_id", "TEXT NOT NULL DEFAULT 'default'")
                self._ensure_column(conn, "documents", "index_status", "TEXT NOT NULL DEFAULT 'pending'")
                self._ensure_column(conn, "documents", "index_error", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "documents", "indexed_at", "TEXT")
                self._ensure_column(conn, "tenant_settings", "llm_provider", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "tenant_settings", "llm_api_key", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "tenant_settings", "llm_model", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(
                    conn,
                    "tenant_settings",
                    "llm_mode",
                    "TEXT NOT NULL DEFAULT 'platform_default'",
                )
                conn.commit()

    def ensure_default_bootstrap(self, backend_api_key: str) -> None:
        """Create default tenant/user and register BACKEND_API_KEY hash.

        This keeps current deployments backward-compatible while Stage 1 routes
        are migrated to tenant-aware access.
        """
        if not backend_api_key:
            return
        with self._lock:
            with self._connect() as conn:
                now = _utcnow_iso()
                tenant_id = "tenant_default"
                user_id = "user_default_admin"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO tenants (id, slug, name, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'active', ?, ?)
                    """,
                    (tenant_id, "default", "Default Tenant", now, now),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO users (id, tenant_id, email, display_name, role, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'owner', 'active', ?, ?)
                    """,
                    (user_id, tenant_id, "admin@local", "Default Admin", now, now),
                )
                self._upsert_api_key_locked(
                    conn=conn,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    label="env-backend-api-key",
                    raw_api_key=backend_api_key,
                )
                conn.commit()

    def _upsert_api_key_locked(
        self,
        conn: sqlite3.Connection,
        tenant_id: str,
        user_id: str,
        label: str,
        raw_api_key: str,
        role: str = DEFAULT_ROLE,
        scopes: str = DEFAULT_SCOPES,
        expires_at: str | None = None,
    ) -> str:
        now = _utcnow_iso()
        key_hash = _hash_api_key(raw_api_key)
        row = conn.execute(
            "SELECT id FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        ).fetchone()
        if row:
            key_id = row["id"]
            conn.execute(
                """
                UPDATE api_keys
                SET tenant_id = ?, user_id = ?, label = ?, role = ?, scopes = ?, status = 'active',
                    expires_at = ?, revoked_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (tenant_id, user_id, label, role, scopes, expires_at, now, key_id),
            )
            return key_id
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO api_keys (
                id, tenant_id, user_id, label, key_hash, role, scopes, status, expires_at, revoked_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL, ?, ?)
            """,
            (key_id, tenant_id, user_id, label, key_hash, role, scopes, expires_at, now, now),
        )
        return key_id

    def resolve_api_key(self, raw_api_key: str) -> AuthContext | None:
        if not raw_api_key:
            return None
        key_hash = _hash_api_key(raw_api_key)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    k.id AS key_id,
                    k.label AS key_label,
                    k.role AS key_role,
                    k.scopes AS key_scopes,
                    k.expires_at AS key_expires_at,
                    t.id AS tenant_id,
                    t.slug AS tenant_slug,
                    u.id AS user_id,
                    u.email AS user_email
                FROM api_keys k
                JOIN tenants t ON t.id = k.tenant_id
                JOIN users u ON u.id = k.user_id
                WHERE
                    k.key_hash = ?
                    AND k.status = 'active'
                    AND t.status = 'active'
                    AND u.status = 'active'
                LIMIT 1
                """,
                (key_hash,),
            ).fetchone()
            if not row:
                return None
            expires_at = row["key_expires_at"]
            if expires_at and expires_at <= _utcnow_iso():
                now = _utcnow_iso()
                conn.execute(
                    "UPDATE api_keys SET status = 'expired', updated_at = ? WHERE id = ?",
                    (now, row["key_id"]),
                )
                conn.commit()
                return None
            now = _utcnow_iso()
            conn.execute(
                "UPDATE api_keys SET last_used_at = ?, updated_at = ? WHERE id = ?",
                (now, now, row["key_id"]),
            )
            conn.commit()
            return AuthContext(
                tenant_id=row["tenant_id"],
                tenant_slug=row["tenant_slug"],
                user_id=row["user_id"],
                user_email=row["user_email"],
                key_id=row["key_id"],
                key_label=row["key_label"],
                role=(row["key_role"] or DEFAULT_ROLE),
                scopes=self._parse_scopes(row["key_scopes"]),
                source="registry",
            )

    def upsert_document(
        self,
        tenant_id: str,
        user_id: str,
        customer_id: str,
        file_name: str,
        size_bytes: int,
        storage_uri: str = "",
        mime_type: str = "",
    ) -> str:
        """Record or update document ownership metadata."""
        if not file_name:
            raise ValueError("file_name is required")
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id
                    FROM documents
                    WHERE tenant_id = ? AND user_id = ? AND customer_id = ? AND file_name = ? AND status = 'active'
                    LIMIT 1
                    """,
                    (tenant_id, user_id, customer_id, file_name),
                ).fetchone()
                now = _utcnow_iso()
                if row:
                    doc_id = row["id"]
                    conn.execute(
                        """
                        UPDATE documents
                        SET storage_uri = ?, mime_type = ?, size_bytes = ?,
                            index_status = 'pending', index_error = '', indexed_at = NULL, updated_at = ?
                        WHERE id = ?
                        """,
                        (storage_uri, mime_type, int(size_bytes), now, doc_id),
                    )
                else:
                    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
                    conn.execute(
                        """
                        INSERT INTO documents (
                            id, tenant_id, user_id, customer_id, file_name, storage_uri, mime_type,
                            size_bytes, status, index_status, index_error, indexed_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 'pending', '', NULL, ?, ?)
                        """,
                        (
                            doc_id,
                            tenant_id,
                            user_id,
                            customer_id,
                            file_name,
                            storage_uri,
                            mime_type,
                            int(size_bytes),
                            now,
                            now,
                        ),
                    )
                conn.commit()
                return doc_id

    def list_documents_for_owner(
        self,
        tenant_id: str,
        user_id: str,
        customer_id: str | None = None,
    ) -> list[dict]:
        where = "WHERE tenant_id = ? AND user_id = ? AND status = 'active'"
        params: list[str] = [tenant_id, user_id]
        if customer_id:
            where += " AND customer_id = ?"
            params.append(customer_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, file_name, size_bytes, mime_type, storage_uri, customer_id, index_status, index_error, indexed_at
                FROM documents
                {where}
                ORDER BY file_name ASC
                """,
                params,
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "file_name": r["file_name"],
                    "size_bytes": int(r["size_bytes"] or 0),
                    "mime_type": r["mime_type"] or "",
                    "storage_uri": r["storage_uri"] or "",
                    "customer_id": r["customer_id"] or "default",
                    "index_status": r["index_status"] or "pending",
                    "index_error": r["index_error"] or "",
                    "indexed_at": r["indexed_at"],
                }
                for r in rows
            ]

    def list_file_names_for_owner(
        self,
        tenant_id: str,
        user_id: str,
        customer_id: str | None = None,
    ) -> set[str]:
        rows = self.list_documents_for_owner(tenant_id, user_id, customer_id=customer_id)
        return {r["file_name"] for r in rows if r.get("file_name")}

    def get_document_for_owner(
        self,
        tenant_id: str,
        user_id: str,
        file_name: str,
        customer_id: str | None = None,
    ) -> dict | None:
        if not file_name:
            return None
        where = "WHERE tenant_id = ? AND user_id = ? AND file_name = ? AND status = 'active'"
        params: list[str] = [tenant_id, user_id, file_name]
        if customer_id:
            where += " AND customer_id = ?"
            params.append(customer_id)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT id, file_name, size_bytes, mime_type, storage_uri, status, customer_id, index_status, index_error, indexed_at
                FROM documents
                {where}
                LIMIT 1
                """,
                params,
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "file_name": row["file_name"],
                "size_bytes": int(row["size_bytes"] or 0),
                "mime_type": row["mime_type"] or "",
                "storage_uri": row["storage_uri"] or "",
                "status": row["status"] or "",
                "customer_id": row["customer_id"] or "default",
                "index_status": row["index_status"] or "pending",
                "index_error": row["index_error"] or "",
                "indexed_at": row["indexed_at"],
            }

    def has_document_access(
        self,
        tenant_id: str,
        user_id: str,
        file_name: str,
        customer_id: str | None = None,
    ) -> bool:
        if not file_name:
            return False
        where = "WHERE tenant_id = ? AND user_id = ? AND file_name = ? AND status = 'active'"
        params: list[str] = [tenant_id, user_id, file_name]
        if customer_id:
            where += " AND customer_id = ?"
            params.append(customer_id)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1
                FROM documents
                {where}
                LIMIT 1
                """,
                params,
            ).fetchone()
            return bool(row)

    def soft_delete_document(
        self,
        tenant_id: str,
        user_id: str,
        file_name: str,
        customer_id: str | None = None,
    ) -> bool:
        now = _utcnow_iso()
        where = "WHERE tenant_id = ? AND user_id = ? AND file_name = ? AND status = 'active'"
        params: list[str] = [now, tenant_id, user_id, file_name]
        if customer_id:
            where += " AND customer_id = ?"
            params.append(customer_id)
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    f"""
                    UPDATE documents
                    SET status = 'deleted', updated_at = ?
                    {where}
                    """,
                    params,
                )
                conn.commit()
                return cur.rowcount > 0

    def get_document_by_id_for_owner(
        self,
        tenant_id: str,
        user_id: str,
        doc_id: str,
        customer_id: str | None = None,
    ) -> dict | None:
        if not doc_id:
            return None
        where = "WHERE tenant_id = ? AND user_id = ? AND id = ? AND status = 'active'"
        params: list[str] = [tenant_id, user_id, doc_id]
        if customer_id:
            where += " AND customer_id = ?"
            params.append(customer_id)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT id, file_name, size_bytes, mime_type, storage_uri, status, customer_id, index_status, index_error, indexed_at
                FROM documents
                {where}
                LIMIT 1
                """,
                params,
            ).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "file_name": row["file_name"],
                "size_bytes": int(row["size_bytes"] or 0),
                "mime_type": row["mime_type"] or "",
                "storage_uri": row["storage_uri"] or "",
                "status": row["status"] or "",
                "customer_id": row["customer_id"] or "default",
                "index_status": row["index_status"] or "pending",
                "index_error": row["index_error"] or "",
                "indexed_at": row["indexed_at"],
            }

    def soft_delete_document_by_id(
        self,
        tenant_id: str,
        user_id: str,
        doc_id: str,
        customer_id: str | None = None,
    ) -> bool:
        now = _utcnow_iso()
        where = "WHERE tenant_id = ? AND user_id = ? AND id = ? AND status = 'active'"
        params: list[str] = [now, tenant_id, user_id, doc_id]
        if customer_id:
            where += " AND customer_id = ?"
            params.append(customer_id)
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    f"""
                    UPDATE documents
                    SET status = 'deleted', updated_at = ?
                    {where}
                    """,
                    params,
                )
                conn.commit()
                return cur.rowcount > 0

    def sync_local_uploads_for_owner(self, tenant_id: str, user_id: str, uploads_dir: str) -> int:
        """Backfill legacy local uploads into document ownership metadata."""
        if not uploads_dir or not os.path.isdir(uploads_dir):
            return 0
        synced = 0
        for name in sorted(os.listdir(uploads_dir)):
            if not name or name.startswith("."):
                continue
            path = os.path.join(uploads_dir, name)
            if not os.path.isfile(path):
                continue
            mime = mimetypes.guess_type(name)[0] or ""
            self.upsert_document(
                tenant_id=tenant_id,
                user_id=user_id,
                customer_id="default",
                file_name=name,
                size_bytes=os.path.getsize(path),
                storage_uri=path,
                mime_type=mime,
            )
            synced += 1
        return synced

    def create_or_rotate_api_key(
        self,
        tenant_slug: str,
        tenant_name: str,
        user_email: str,
        user_display_name: str,
        key_label: str,
        raw_api_key: str | None = None,
        role: str = DEFAULT_ROLE,
        scopes: tuple[str, ...] | None = None,
        expires_at: str | None = None,
    ) -> tuple[str, AuthContext]:
        """Create or rotate an API key for a tenant/user and return raw key once."""
        if not tenant_slug or not user_email:
            raise ValueError("tenant_slug and user_email are required")
        token = (raw_api_key or "").strip() or secrets.token_urlsafe(32)
        tenant_slug_norm = tenant_slug.strip().lower()
        now = _utcnow_iso()
        scopes_csv = ",".join(scopes or (DEFAULT_SCOPES,))
        with self._lock:
            with self._connect() as conn:
                trow = conn.execute(
                    "SELECT id FROM tenants WHERE slug = ? LIMIT 1",
                    (tenant_slug_norm,),
                ).fetchone()
                if trow:
                    tenant_id = trow["id"]
                    conn.execute(
                        "UPDATE tenants SET name = ?, status = 'active', updated_at = ? WHERE id = ?",
                        (tenant_name or tenant_slug_norm, now, tenant_id),
                    )
                else:
                    tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
                    conn.execute(
                        """
                        INSERT INTO tenants (id, slug, name, status, created_at, updated_at)
                        VALUES (?, ?, ?, 'active', ?, ?)
                        """,
                        (tenant_id, tenant_slug_norm, tenant_name or tenant_slug_norm, now, now),
                    )

                urow = conn.execute(
                    "SELECT id FROM users WHERE tenant_id = ? AND email = ? LIMIT 1",
                    (tenant_id, user_email.strip().lower()),
                ).fetchone()
                if urow:
                    user_id = urow["id"]
                    conn.execute(
                        """
                        UPDATE users
                        SET display_name = ?, status = 'active', updated_at = ?
                        WHERE id = ?
                        """,
                        (user_display_name or user_email, now, user_id),
                    )
                else:
                    user_id = f"user_{uuid.uuid4().hex[:12]}"
                    conn.execute(
                        """
                        INSERT INTO users (id, tenant_id, email, display_name, role, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, 'owner', 'active', ?, ?)
                        """,
                        (user_id, tenant_id, user_email.strip().lower(), user_display_name or user_email, now, now),
                    )

                key_id = self._upsert_api_key_locked(
                    conn=conn,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    label=key_label.strip() or "service-key",
                    raw_api_key=token,
                    role=role or DEFAULT_ROLE,
                    scopes=scopes_csv,
                    expires_at=expires_at,
                )
                conn.commit()
                ctx = AuthContext(
                    tenant_id=tenant_id,
                    tenant_slug=tenant_slug_norm,
                    user_id=user_id,
                    user_email=user_email.strip().lower(),
                    key_id=key_id,
                    key_label=key_label.strip() or "service-key",
                    role=role or DEFAULT_ROLE,
                    scopes=self._parse_scopes(scopes_csv),
                    source="registry",
                )
                return token, ctx

    def revoke_api_key(self, key_id: str) -> bool:
        if not key_id:
            return False
        now = _utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE api_keys
                    SET status = 'revoked', revoked_at = ?, updated_at = ?
                    WHERE id = ? AND status != 'revoked'
                    """,
                    (now, now, key_id),
                )
                conn.commit()
                return cur.rowcount > 0

    def set_key_expiry(self, key_id: str, expires_at: str | None) -> bool:
        if not key_id:
            return False
        now = _utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE api_keys
                    SET expires_at = ?, status = 'active', revoked_at = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (expires_at, now, key_id),
                )
                conn.commit()
                return cur.rowcount > 0

    def list_api_keys(
        self,
        tenant_slug: str | None = None,
        user_email: str | None = None,
    ) -> list[dict]:
        clauses = []
        params: list[str] = []
        if tenant_slug:
            clauses.append("t.slug = ?")
            params.append(tenant_slug.strip().lower())
        if user_email:
            clauses.append("u.email = ?")
            params.append(user_email.strip().lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    k.id, k.label, k.role, k.scopes, k.status, k.expires_at, k.revoked_at, k.last_used_at,
                    k.created_at, k.updated_at,
                    t.id AS tenant_id, t.slug AS tenant_slug, t.name AS tenant_name,
                    u.id AS user_id, u.email AS user_email
                FROM api_keys k
                JOIN tenants t ON t.id = k.tenant_id
                JOIN users u ON u.id = k.user_id
                {where}
                ORDER BY t.slug ASC, u.email ASC, k.created_at DESC
                """,
                params,
            ).fetchall()
            out: list[dict] = []
            for r in rows:
                out.append(
                    {
                        "id": r["id"],
                        "label": r["label"],
                        "role": r["role"] or DEFAULT_ROLE,
                        "scopes": list(self._parse_scopes(r["scopes"])),
                        "status": r["status"] or "",
                        "expires_at": r["expires_at"],
                        "revoked_at": r["revoked_at"],
                        "last_used_at": r["last_used_at"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "tenant_id": r["tenant_id"],
                        "tenant_slug": r["tenant_slug"],
                        "tenant_name": r["tenant_name"],
                        "user_id": r["user_id"],
                        "user_email": r["user_email"],
                    }
                )
            return out

    def record_chat_turn(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        summary: str,
        session_title: str = "",
        session_id: str | None = None,
    ) -> str:
        """Persist a minimal chat turn (user + assistant messages)."""
        now = _utcnow_iso()
        with self._lock:
            with self._connect() as conn:
                sid = session_id.strip() if session_id else ""
                if sid:
                    row = conn.execute(
                        """
                        SELECT id
                        FROM chat_sessions
                        WHERE id = ? AND tenant_id = ? AND user_id = ? AND status = 'active'
                        LIMIT 1
                        """,
                        (sid, tenant_id, user_id),
                    ).fetchone()
                    if not row:
                        sid = ""
                if not sid:
                    sid = f"sess_{uuid.uuid4().hex[:12]}"
                    title = (session_title or "").strip() or (query or "").strip()[:80]
                    conn.execute(
                        """
                        INSERT INTO chat_sessions (id, tenant_id, user_id, title, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, 'active', ?, ?)
                        """,
                        (sid, tenant_id, user_id, title, now, now),
                    )
                else:
                    conn.execute(
                        "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                        (now, sid),
                    )

                user_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
                asst_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO chat_messages (id, session_id, tenant_id, user_id, role, content, created_at)
                    VALUES (?, ?, ?, ?, 'user', ?, ?)
                    """,
                    (user_msg_id, sid, tenant_id, user_id, (query or "").strip(), now),
                )
                conn.execute(
                    """
                    INSERT INTO chat_messages (id, session_id, tenant_id, user_id, role, content, created_at)
                    VALUES (?, ?, ?, ?, 'assistant', ?, ?)
                    """,
                    (asst_msg_id, sid, tenant_id, user_id, (summary or "").strip(), now),
                )
                conn.commit()
                return sid

    def debug_stats(self) -> dict[str, object]:
        """Return lightweight table counts for runtime diagnostics."""
        with self._connect() as conn:
            out: dict[str, object] = {"db_path": self.db_path}
            for name in (
                "tenants",
                "users",
                "api_keys",
                "documents",
                "chat_sessions",
                "chat_messages",
                "index_jobs",
                "tenant_settings",
                "tenant_daily_usage",
                "delete_audit_logs",
            ):
                row = conn.execute(f"SELECT COUNT(*) AS c FROM {name}").fetchone()
                out[name] = int(row["c"]) if row else 0
            return out

    def get_tenant_settings(self, tenant_id: str) -> dict[str, str]:
        if not tenant_id:
            return {
                "llm_mode": "platform_default",
                "llm_provider": "",
                "llm_api_key": "",
                "llm_model": "",
                "encryption_ready": bool(self._fernet),
            }
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT llm_mode, llm_provider, llm_api_key, llm_model
                FROM tenant_settings
                WHERE tenant_id = ?
                LIMIT 1
                """,
                (tenant_id,),
            ).fetchone()
            if not row:
                return {
                    "llm_mode": "platform_default",
                    "llm_provider": "",
                    "llm_api_key": "",
                    "llm_model": "",
                    "encryption_ready": bool(self._fernet),
                }
            decrypted = self._decrypt_secret((row["llm_api_key"] or "").strip())
            mode = (row["llm_mode"] or "platform_default").strip().lower()
            if mode not in {"platform_default", "tenant_byok"}:
                mode = "platform_default"
            return {
                "llm_mode": mode,
                "llm_provider": (row["llm_provider"] or "").strip().lower(),
                "llm_api_key": decrypted,
                "llm_model": (row["llm_model"] or "").strip(),
                "encryption_ready": bool(self._fernet),
            }

    def upsert_tenant_settings(
        self,
        tenant_id: str,
        llm_mode: str,
        llm_provider: str,
        llm_api_key: str,
        llm_model: str = "",
    ) -> dict[str, str]:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        now = _utcnow_iso()
        mode = (llm_mode or "platform_default").strip().lower()
        if mode not in {"platform_default", "tenant_byok"}:
            raise ValueError("llm_mode must be 'platform_default' or 'tenant_byok'")
        provider = (llm_provider or "").strip().lower()
        model = (llm_model or "").strip()
        key = (llm_api_key or "").strip()
        encrypted_key = self._encrypt_secret(key) if key else ""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tenant_settings (tenant_id, llm_mode, llm_provider, llm_api_key, llm_model, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id) DO UPDATE SET
                        llm_mode = excluded.llm_mode,
                        llm_provider = excluded.llm_provider,
                        llm_api_key = excluded.llm_api_key,
                        llm_model = excluded.llm_model,
                        updated_at = excluded.updated_at
                    """,
                    (tenant_id, mode, provider, encrypted_key, model, now, now),
                )
                conn.commit()
        return {"llm_mode": mode, "llm_provider": provider, "llm_api_key": key, "llm_model": model}

    def record_request_usage(
        self,
        tenant_id: str,
        user_id: str,
        route: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        if not tenant_id or not user_id or not route:
            return
        day = _utc_today()
        now = _utcnow_iso()
        is_error = 1 if int(status_code) >= 400 else 0
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tenant_daily_usage (
                        day, tenant_id, user_id, route, request_count, error_count, total_latency_ms, updated_at
                    )
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(day, tenant_id, user_id, route) DO UPDATE SET
                        request_count = request_count + 1,
                        error_count = error_count + excluded.error_count,
                        total_latency_ms = total_latency_ms + excluded.total_latency_ms,
                        updated_at = excluded.updated_at
                    """,
                    (
                        day,
                        tenant_id,
                        user_id,
                        route,
                        is_error,
                        float(latency_ms),
                        now,
                    ),
                )
                conn.commit()

    def get_tenant_usage_summary(self, tenant_id: str, days: int = 7) -> dict[str, object]:
        if not tenant_id:
            return {"days": int(days), "totals": {}, "by_route": []}
        days = max(1, min(int(days), 90))
        with self._connect() as conn:
            totals = conn.execute(
                """
                SELECT
                    COALESCE(SUM(request_count), 0) AS request_count,
                    COALESCE(SUM(error_count), 0) AS error_count,
                    COALESCE(SUM(total_latency_ms), 0) AS total_latency_ms
                FROM tenant_daily_usage
                WHERE tenant_id = ? AND day >= date('now', ?)
                """,
                (tenant_id, f"-{days - 1} day"),
            ).fetchone()
            by_route_rows = conn.execute(
                """
                SELECT
                    route,
                    COALESCE(SUM(request_count), 0) AS request_count,
                    COALESCE(SUM(error_count), 0) AS error_count,
                    COALESCE(SUM(total_latency_ms), 0) AS total_latency_ms
                FROM tenant_daily_usage
                WHERE tenant_id = ? AND day >= date('now', ?)
                GROUP BY route
                ORDER BY request_count DESC, route ASC
                """,
                (tenant_id, f"-{days - 1} day"),
            ).fetchall()
            req_count = int(totals["request_count"] or 0)
            err_count = int(totals["error_count"] or 0)
            total_latency = float(totals["total_latency_ms"] or 0.0)
            by_route: list[dict[str, object]] = []
            for r in by_route_rows:
                rc = int(r["request_count"] or 0)
                ec = int(r["error_count"] or 0)
                tl = float(r["total_latency_ms"] or 0.0)
                by_route.append(
                    {
                        "route": r["route"],
                        "request_count": rc,
                        "error_count": ec,
                        "error_rate": round((ec / rc), 4) if rc else 0.0,
                        "avg_latency_ms": round((tl / rc), 2) if rc else 0.0,
                    }
                )
            return {
                "days": days,
                "totals": {
                    "request_count": req_count,
                    "error_count": err_count,
                    "error_rate": round((err_count / req_count), 4) if req_count else 0.0,
                    "avg_latency_ms": round((total_latency / req_count), 2) if req_count else 0.0,
                },
                "by_route": by_route,
            }

    @staticmethod
    def _route_group(route: str) -> str:
        r = (route or "").strip().lower()
        if not r:
            return "other"
        if r.startswith("/api/upload"):
            return "upload"
        if r.startswith("/api/query") or r == "/query":
            return "query"
        if r.startswith("/api/chat"):
            return "chat"
        if r.startswith("/api/documents"):
            return "documents"
        if r.startswith("/api/tenant") or r.startswith("/api/auth") or r.startswith("/api/metrics") or r.startswith("/api/index"):
            return "admin"
        return "other"

    def get_tenant_usage_grouped(self, tenant_id: str, days: int = 7) -> dict[str, dict]:
        summary = self.get_tenant_usage_summary(tenant_id, days=days)
        grouped: dict[str, dict] = {}
        for row in summary.get("by_route", []):
            group = self._route_group(str(row.get("route", "")))
            g = grouped.setdefault(
                group,
                {"request_count": 0, "error_count": 0, "total_latency_ms": 0.0},
            )
            rc = int(row.get("request_count", 0) or 0)
            ec = int(row.get("error_count", 0) or 0)
            avg = float(row.get("avg_latency_ms", 0) or 0.0)
            g["request_count"] += rc
            g["error_count"] += ec
            g["total_latency_ms"] += (avg * rc)
        out: dict[str, dict] = {}
        for group, g in grouped.items():
            rc = int(g["request_count"])
            ec = int(g["error_count"])
            total_latency = float(g["total_latency_ms"])
            out[group] = {
                "request_count": rc,
                "error_count": ec,
                "error_rate": round((ec / rc), 4) if rc else 0.0,
                "avg_latency_ms": round((total_latency / rc), 2) if rc else 0.0,
            }
        return out

    def export_tenant_usage_rows(self, tenant_id: str, days: int = 7) -> list[dict[str, object]]:
        if not tenant_id:
            return []
        days = max(1, min(int(days), 90))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT day, user_id, route, request_count, error_count, total_latency_ms
                FROM tenant_daily_usage
                WHERE tenant_id = ? AND day >= date('now', ?)
                ORDER BY day DESC, route ASC, user_id ASC
                """,
                (tenant_id, f"-{days - 1} day"),
            ).fetchall()
            out: list[dict[str, object]] = []
            for r in rows:
                rc = int(r["request_count"] or 0)
                ec = int(r["error_count"] or 0)
                tl = float(r["total_latency_ms"] or 0.0)
                route = r["route"] or ""
                out.append(
                    {
                        "day": r["day"],
                        "user_id": r["user_id"],
                        "route": route,
                        "route_group": self._route_group(route),
                        "request_count": rc,
                        "error_count": ec,
                        "error_rate": round((ec / rc), 4) if rc else 0.0,
                        "avg_latency_ms": round((tl / rc), 2) if rc else 0.0,
                    }
                )
            return out

    @staticmethod
    def usage_rows_to_csv(rows: list[dict[str, object]]) -> str:
        fields = [
            "day",
            "user_id",
            "route",
            "route_group",
            "request_count",
            "error_count",
            "error_rate",
            "avg_latency_ms",
        ]
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
        return buf.getvalue()

    def record_delete_audit(
        self,
        tenant_id: str,
        user_id: str,
        customer_id: str,
        document_id: str,
        file_name: str,
        result: str,
        details_json: str = "{}",
        action: str = "delete_document",
    ) -> str:
        if not tenant_id or not user_id:
            return ""
        now = _utcnow_iso()
        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO delete_audit_logs (
                        id, tenant_id, user_id, customer_id, document_id, file_name,
                        action, result, details_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        audit_id,
                        tenant_id,
                        user_id,
                        customer_id or "",
                        document_id or "",
                        file_name or "",
                        action or "delete_document",
                        result or "unknown",
                        details_json or "{}",
                        now,
                    ),
                )
                conn.commit()
        return audit_id

    def update_documents_index_status(
        self,
        tenant_id: str,
        user_id: str,
        status: str,
        customer_id: str | None = None,
        file_names: set[str] | None = None,
        index_error: str = "",
    ) -> int:
        if not tenant_id or not user_id:
            return 0
        status_norm = (status or "").strip().lower()
        if status_norm not in {"pending", "indexing", "indexed", "failed"}:
            raise ValueError("invalid index status")
        now = _utcnow_iso()
        where = "WHERE tenant_id = ? AND user_id = ? AND status = 'active'"
        params: list[object] = [status_norm, (index_error or "")[:500], now]
        if status_norm == "indexed":
            set_sql = "index_status = ?, index_error = '', indexed_at = ?, updated_at = ?"
            params = [status_norm, now, now]
        else:
            set_sql = "index_status = ?, index_error = ?, updated_at = ?"
        params.extend([tenant_id, user_id])
        if customer_id:
            where += " AND customer_id = ?"
            params.append(customer_id)
        if file_names:
            files = sorted({f for f in file_names if f})
            if files:
                placeholders = ", ".join(["?"] * len(files))
                where += f" AND file_name IN ({placeholders})"
                params.extend(files)
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    f"""
                    UPDATE documents
                    SET {set_sql}
                    {where}
                    """,
                    params,
                )
                conn.commit()
                return int(cur.rowcount)


tenant_store = TenantStore()
