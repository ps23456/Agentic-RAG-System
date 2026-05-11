"""Background index job worker (Gap 2).

A single daemon thread drains the `index_jobs` SQLite queue. Index HTTP
handlers enqueue work instead of spawning ad-hoc threads, so:

- Jobs survive process restarts as rows (status stays queued/running —
  running jobs may need manual retry after crash; see ops note below).
- Clients get stable `job_id` values to poll via GET /api/index/jobs.

Ops note: if the process dies while a job is `running`, that row stays
stuck until an operator sets it back to `queued` or deletes it. A future
enhancement is a startup sweep that marks stale `running` as `failed`.
"""
from __future__ import annotations

import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

_worker_started = False
_worker_lock = threading.Lock()


def _run_job(job: dict) -> None:
    from backend.db.tenant_store import tenant_store
    from backend.services.rag_service import rag

    job_id = job["id"]
    tenant_id = job["tenant_id"]
    user_id = job["user_id"]
    jtype = job["job_type"]
    try:
        details = json.loads(job.get("details_json") or "{}")
    except json.JSONDecodeError:
        details = {}

    try:
        if jtype == "reindex_full":
            rag.reindex()
            if str(rag._index_status).startswith("error"):
                raise RuntimeError(str(rag._index_status))

        elif jtype == "reindex_docs":
            raw_files = details.get("files")
            file_filter = set(raw_files) if isinstance(raw_files, list) and raw_files else None
            customer_id = (details.get("customer_id") or "").strip() or None
            tenant_store.update_documents_index_status(
                tenant_id=tenant_id,
                user_id=user_id,
                status="indexing",
                customer_id=customer_id,
                file_names=file_filter,
                index_error="",
            )
            try:
                rag.reindex_docs(file_filter=file_filter)
                if str(rag._index_status).startswith("error"):
                    raise RuntimeError(str(rag._index_status))
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="indexed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error="",
                )
            except Exception as e:
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="failed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error=str(e),
                )
                raise

        elif jtype == "reindex_images":
            raw_files = details.get("files")
            file_filter = set(raw_files) if isinstance(raw_files, list) and raw_files else None
            customer_id = (details.get("customer_id") or "").strip() or None
            tenant_store.update_documents_index_status(
                tenant_id=tenant_id,
                user_id=user_id,
                status="indexing",
                customer_id=customer_id,
                file_names=file_filter,
                index_error="",
            )
            try:
                rag.reindex_images(file_filter=file_filter)
                if str(rag._index_status).startswith("error"):
                    raise RuntimeError(str(rag._index_status))
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="indexed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error="",
                )
            except Exception as e:
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="failed",
                    customer_id=customer_id,
                    file_names=file_filter,
                    index_error=str(e),
                )
                raise

        elif jtype == "upload_auto":
            raw = details.get("file_basenames") or details.get("files") or []
            file_basenames = {str(x) for x in raw if isinstance(x, str) and x.strip()}
            customer_id = (details.get("customer_id") or "").strip() or None
            if not file_basenames:
                raise ValueError("upload_auto job missing file_basenames")
            tenant_store.update_documents_index_status(
                tenant_id=tenant_id,
                user_id=user_id,
                status="indexing",
                customer_id=customer_id,
                file_names=file_basenames,
                index_error="",
            )
            rag.reindex_docs(file_filter=set(file_basenames))
            docs_status = str(rag._index_status)
            rag.reindex_images(file_filter=set(file_basenames))
            img_status = str(rag._index_status)
            if docs_status.startswith("error") or img_status.startswith("error"):
                err = docs_status if docs_status.startswith("error") else img_status
                tenant_store.update_documents_index_status(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status="failed",
                    customer_id=customer_id,
                    file_names=file_basenames,
                    index_error=err,
                )
                raise RuntimeError(err)
            tenant_store.update_documents_index_status(
                tenant_id=tenant_id,
                user_id=user_id,
                status="indexed",
                customer_id=customer_id,
                file_names=file_basenames,
                index_error="",
            )

        else:
            raise ValueError(f"unknown job_type: {jtype}")

        tenant_store.finish_index_job(job_id, succeeded=True)

    except Exception as e:
        logger.exception("index_job_failed id=%s type=%s", job_id, jtype)
        tenant_store.finish_index_job(job_id, succeeded=False, error_message=str(e))


def _worker_loop() -> None:
    from backend.db.tenant_store import tenant_store

    logger.info("index_job_worker started")
    while True:
        try:
            job = tenant_store.claim_next_index_job()
            if job:
                _run_job(job)
            else:
                time.sleep(0.25)
        except Exception:
            logger.exception("index_job_worker_loop_error")
            time.sleep(1.0)


def start_index_job_worker() -> None:
    """Start a single global worker thread (idempotent)."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
        t = threading.Thread(target=_worker_loop, name="index-job-worker", daemon=True)
        t.start()
