# API Reference (Current Stage 3 Contract)

This document is the **single source of truth for the current API surface** after Stage 3 hardening updates (multi-tenant auth, customer/doc scoping, BYOK, metering).

- Base URL (local): `http://localhost:8000`
- Auth header (where required): `X-API-Key: <tenant api key>`
- Content type: `application/json` unless stated otherwise

> Stability legend  
> **stable** — frozen for current stage; do not change without version bump.  
> **experimental** — may change; not part of the Stage 0 baseline.

---

## 1. Health & Metrics

### GET `/api/health` — stable, public
- Response 200:
  ```json
  { "status": "ok" }
  ```

### GET `/api/metrics` — stable, protected (`admin:read`)
- Headers: `X-API-Key: <BACKEND_API_KEY>`
- Response 200 (snapshot from `RuntimeMetrics`):
  ```json
  {
    "uptime_s": 1234.5,
    "request_count": 42,
    "error_count": 0,
    "error_rate": 0.0,
    "avg_latency_ms": 18.7,
    "chat_request_count": 5,
    "chat_avg_latency_ms": 2400.0,
    "top_routes": [{ "path": "/api/health", "count": 30 }]
  }
  ```
- Errors: `401` invalid/missing key, `403` insufficient scope.

### GET `/api/metrics/tenant` — stable, protected (`admin:read`)
- Query: `days=<1..90>`
- Response 200:
  ```json
  {
    "tenant_id": "tenant_default",
    "days": 7,
    "totals": { "request_count": 55, "error_count": 2, "error_rate": 0.0364, "avg_latency_ms": 210.73 },
    "by_route": [{ "route": "/api/documents", "request_count": 24, "error_count": 0, "error_rate": 0.0, "avg_latency_ms": 381.35 }],
    "by_group": {
      "documents": { "request_count": 24, "error_count": 0, "error_rate": 0.0, "avg_latency_ms": 381.35 },
      "chat": { "request_count": 6, "error_count": 0, "error_rate": 0.0, "avg_latency_ms": 1200.5 }
    }
  }
  ```

### GET `/api/metrics/tenant/export` — stable, protected (`admin:read`)
- Query: `days=<1..90>&format=json|csv`
- `format=json` -> `{ tenant_id, days, rows[], by_group }`
- `format=csv` -> CSV download with header:
  `day,user_id,route,route_group,request_count,error_count,error_rate,avg_latency_ms`

---

## 2. Public RAG Query

### POST `/query` — stable, protected (`query:run`)
- Source: `backend/routes/query.py`
- Request:
  ```json
  {
    "question": "Financing to self help groups",
    "patient": null,
    "customer_id": "default",
    "file": null,
    "web_search": false,
    "stream": false
  }
  ```
- Response 200 (blocking):
  ```json
  {
    "answer": "...markdown answer with [1][2] citations...",
    "sources": [
      { "file": "PNB AR 2024-25_Web.pdf", "page": 120, "title": "Management Discussion and Analysis" }
    ],
    "intent": "general_search",
    "elapsed_ms": 28927
  }
  ```
- When `stream=true`, response is `text/event-stream` with events:
  - `meta`  → `{ intent, reasoning, results, sources }`
  - `token` → string deltas of the answer
  - `done`  → final `{ summary, sources, results, intent, reasoning }`
- Constraints: `question` length ≤ 2000 chars (`MAX_QUESTION_CHARS`).
- Errors: `401` invalid/missing key, `403` insufficient scope, `422` validation, `500` RAG pipeline failure.

---

## 3. Chat (UI-facing)

### POST `/api/chat` — stable, protected (`chat:run`)
- Headers: `X-API-Key: <BACKEND_API_KEY>`
- Request:
  ```json
  {
    "query": "What is...",
    "conversation_id": "",
    "customer_id": "default",
    "patient_filter": null,
    "web_search": false,
    "file_filter": null
  }
  ```
- Response 200: `{ summary, sources[], results[], intent, reasoning }`

### POST `/api/chat/stream` — stable, protected (`chat:run`)
- Same body as `/api/chat`. Returns SSE (`meta`, `token`*, `done`/`error`).

### POST `/api/chat/evaluate` — experimental, protected (`chat:evaluate`)
- Optional RAGAs faithfulness + answer relevancy for one turn.

---

## 4. Documents

### GET `/api/documents` — stable, protected (`docs:read`)
- Query (optional): `customer_id=<id>`
- Response: `{ files: [{ doc_id, name, size, type, customer_id, index_status, index_error, indexed_at }] }`

### GET `/api/documents/info` — stable, public
- Response: `{ chunk_count, tree_count, image_count, patients[], status, indexing, progress, stage }`

### GET `/api/documents/page` — stable, protected (`docs:read`)
- Query: `file=<basename>` or `doc_id=<doc_id>`, optional `customer_id`, `page=<int>` (PDF only)
- Response: `{ image: <base64 PNG>, page, total_pages }`

### GET `/api/documents/text` — stable, protected (`docs:read`)
- Query: `file=<basename>` or `doc_id=<doc_id>`, optional `customer_id`, `search`, `page`
- Response: `{ content, file_name, scroll_line, matched_text }`

### GET `/api/documents/image` — stable, protected (`docs:read`)
- Query: `file=<basename>` or `doc_id=<doc_id>`, optional `customer_id`
- Response: binary image content with proper `Content-Type`.

### DELETE `/api/documents` — stable, protected (`docs:write`)
- Query: `file=<basename>` or `doc_id=<doc_id>`, optional `customer_id`
- Response includes delete lifecycle details:
  ```json
  { "deleted": "file.txt", "doc_id": "doc_x", "audit_id": "audit_x", "storage_removed": true, "removed_empty_dirs": 2 }
  ```
- Errors: `400` invalid filename, `404` not found.

### GET `/api/documents/mistral-ocr-md` — experimental, protected (`docs:read`)
- Query: `file=<basename>.pdf` or `doc_id=<doc_id>`, optional `customer_id`
- Errors: `503` if `MISTRAL_OCR_API_KEY` missing.

---

## 5. Upload

### POST `/api/upload` — stable, protected (`docs:write`)
- Headers: `X-API-Key`
- Body: `multipart/form-data` with `customer_id` (optional, defaults `default`) and `files=@<path>` (one or more)
- Allowed extensions: `.pdf, .md, .txt, .json, .png, .jpg, .jpeg, .tiff, .tif, .bmp, .gif, .webp`
- Response:
  ```json
  {
    "uploaded": ["a.pdf"],
    "uploaded_docs": [{ "doc_id": "doc_123", "file_name": "a.pdf", "customer_id": "customer-1" }],
    "customer_id": "customer-1",
    "count": 1,
    "images": [],
    "docs": ["a.pdf"],
    "images_count": 0,
    "docs_count": 1
  }
  ```

---

## 6. Indexing

### POST `/api/index` — stable, protected (`index:run`)
- Triggers full reindex (text + image). Returns immediately.
- Response: `{ status: "started" | "already_indexing" }`

### POST `/api/index/docs` — stable, protected (`index:run`)
- Body (optional): `{ "files": ["foo.pdf"], "customer_id": "customer-1" }`
- Response: `{ status, targeted, count }`

### POST `/api/index/images` — stable, protected (`index:run`)
- Body (optional): `{ "files": ["x.png"], "customer_id": "customer-1" }`
- Response: `{ status, targeted, count }`

### GET `/api/index/status` — stable, public
- Response: `{ chunk_count, tree_count, image_count, patients[], status, indexing, progress, stage }`

---

## 7. Tenant Settings (BYOK)

### GET `/api/tenant/settings` — stable, protected (`admin:read`)
- Response:
  ```json
  {
    "tenant_id": "tenant_default",
    "llm_mode": "platform_default",
    "llm_provider": "openai",
    "llm_model": "",
    "has_llm_api_key": false,
    "llm_api_key_masked": "",
    "encryption_ready": true
  }
  ```

### PUT `/api/tenant/settings` — stable, protected (`admin:write`)
- Request:
  ```json
  {
    "llm_mode": "platform_default",
    "llm_provider": "",
    "llm_api_key": "",
    "llm_model": ""
  }
  ```
- `llm_mode`: `platform_default | tenant_byok`
- For `tenant_byok`, `llm_provider` + `llm_api_key` are required.

### POST `/api/tenant/settings/test-connection` — stable, protected (`admin:write`)
- Validates provider/key/model **without saving**.
- Request:
  ```json
  { "llm_provider": "openai", "llm_api_key": "sk-...", "llm_model": "gpt-4o-mini" }
  ```
- Response:
  ```json
  { "ok": false, "provider": "openai", "model": "gpt-4o-mini", "latency_ms": 2036.38, "detail": "invalid_api_key" }
  ```

---
## 8. Medical (experimental for current stage)

| Method | Path | Notes |
|---|---|---|
| GET  | `/api/medical/patients`   | List known patients |
| GET  | `/api/medical/files`      | List files for a patient |
| GET  | `/api/medical/image`      | Serve a medical image/PDF page |
| POST | `/api/medical/classify`   | LLM document classification |
| POST | `/api/medical/analyze`    | LLM analysis over selected images |
| POST | `/api/medical/upload`     | Multipart upload into a patient folder |

These are not part of the Stage 0 baseline KPI suite but must remain reachable.

---

## 9. Fields (experimental)

### GET `/api/fields/extract`
- Query: `file=<basename>` (`.pdf` or `.md`)
- Response: `{ file_name, mode, field_names[], text_preview, schema }`

---

## LLM Provider Order & Failover

The server selects the LLM provider at request time via
`backend/services/rag_service.py::_llm_key_and_provider`:

1. **Groq** (primary) — `llama-3.3-70b-versatile` via OpenAI-compatible endpoint.
   Selected whenever `GROQ_API_KEY` is set in the environment.
2. **OpenAI** (fallback) — `gpt-4o-mini`. Selected only when `GROQ_API_KEY`
   is missing, or when an in-call retry path explicitly falls back
   (e.g. RAGAs eval in `backend/services/ragas_eval.py`).
3. **Gemini / HuggingFace** — optional, used only by specific helpers
   (`llm_insight.py`, vision/medical paths). Not part of the chat default.

Implications for the Stage 0 contract:

- All `/query`, `/api/chat`, and `/api/chat/stream` answers are produced by
  Groq when `GROQ_API_KEY` is configured; cost-per-query baselines must be
  measured against Groq pricing.
- Failover is **server-side only**. Clients see no protocol difference; the
  response shape is identical regardless of which provider answered.
- No LLM key is ever exposed to the browser. `VITE_BACKEND_API_KEY` is a
  client-facing key for *this* backend, **not** for Groq/OpenAI/Gemini.
- Rotation priority: `GROQ_API_KEY` first, then `OPENAI_API_KEY`, then any
  other provider keys. Restart the backend after rotating any of them.

## Auth & Error Conventions

- All protected endpoints use `X-API-Key`.  
- `401` → key missing/invalid.  
- `503` → server has no `BACKEND_API_KEY` configured.  
- `400` → validation/path safety violation.  
- `404` → file not found in uploads.  
- `500` → unexpected pipeline failure (logged with stack trace).

## Versioning Rule

- The contracts above are **frozen** for this release window.
- Adding a new optional field is allowed.
- Removing a field, renaming a field, or changing a status code → version bump (`/v2`).
