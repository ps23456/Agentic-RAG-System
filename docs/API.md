# API Reference (Current Stage 3 Contract)

This document is the **single source of truth for the current API surface** after Stage 3 hardening updates (multi-tenant auth, customer/doc scoping, BYOK, metering).

**Partners:** start with the shorter **[INTEGRATION.md](./INTEGRATION.md)** guide, then use this file for full detail.

- Base URL (local): `http://localhost:8000`
- Auth header (where required): `X-API-Key: <tenant api key>`
- Content type: `application/json` unless stated otherwise

> Stability legend  
> **stable** — frozen for current stage; do not change without version bump.  
> **experimental** — may change; not part of the Stage 0 baseline.

---

## 0. Partner guide — build your own frontend (Cursor / any API client)

Hand this file (or the **OpenAPI spec** below) to your IDE or API tool so it can generate UI, fetch wrappers, and flows without guessing.

### OpenAPI / Swagger (machine-readable)

Every deployment exposes:

- **`GET /openapi.json`** — full OpenAPI 3 schema (import into Postman, codegen, or Cursor).
- **`GET /docs`** — interactive Swagger UI (try requests in the browser when the server allows it).

Replace the host with yours, e.g. `https://your-api.example.com/openapi.json`.

### What you need from the operator

| Item | Purpose |
|------|---------|
| **Base URL** | e.g. `https://your-api.example.com` (or `http://localhost:8000` in dev) |
| **API key** | HTTP header `X-API-Key: <key>` on every protected route |
| **`customer_id`** | Your logical bucket for files (workspace, project, “patient”, etc.). Maps to storage + document list + optional index jobs. Use the same string on upload, list, query, and delete. |

`customer_id` is a **form field on upload** and often a **query or JSON field** elsewhere. It defaults to `default` if omitted — set it explicitly in production so tenants stay organized.

### Typical integration flow (minimal)

1. **Upload** — `POST /api/upload` with `multipart/form-data`: `customer_id`, one or more `files`.  
   Save returned `uploaded`, `doc_id` values from `uploaded_docs`, and `index_job_id` if present.
2. **Wait for indexing** (pick one or both):
   - **Per job:** if `index_job_id` is non-empty **or** you called `POST /api/index*`, poll **`GET /api/index/jobs/{job_id}`** until `status` is `succeeded` or `failed`.
   - **Global progress:** poll **`GET /api/index/status`** for `indexing`, `progress`, `stage` (and optional cross-check with **`GET /api/documents?customer_id=…`** for `index_status` per file).
3. **Ask questions** — **`POST /query`** or **`POST /api/chat`** with the same **`customer_id`** (and optional `file` / `file_filter` to scope to one document).
4. **Remove a file** — **`DELETE /api/documents?file=<basename>&customer_id=…`** (or `doc_id=…` instead of `file`). Use the exact `customer_id` used at upload.

### Scopes (if keys are restricted)

Keys created via the tenant registry can carry scoped permissions. Common scope names include `docs:read`, `docs:write`, `query:run`, `chat:run`, `index:run`, `admin:read`, `admin:write`.  
If your key has **`admin:*`**, it can call everything. If uploads fail with **403**, ask the operator to add **`docs:write`** (and **`index:run`** if you trigger reindex from the client).

### Errors & empty answers

| Situation | What to do |
|-----------|------------|
| **401 / 403** | Fix or rotate API key; check scopes. |
| **400** on upload | Bad `customer_id` (e.g. contains `/`, `..`). |
| **413** on upload | Body too large — only when server **upload limits** are enabled (`UPLOAD_LIMITS_ENABLED=true`). Otherwise check reverse proxy `client_max_body_size`. |
| **`answer` / `summary` empty or “no context”** | Indexing may still be running or failed — poll jobs or `GET /api/documents?customer_id=` for `index_status` / `index_error`, then retry query. |
| **`404` on DELETE** | Wrong `file` name, wrong `customer_id`, or already deleted. |

### Prompt you can paste into Cursor

```text
Build a frontend against this API. Base URL: <PASTE>.
Auth: header X-API-Key: <PASTE_SECRET>.
Use customer_id "<PASTE_CUSTOMER_ID>" on all uploads, document list, query/chat, and deletes.

Flow:
1. Upload files: POST /api/upload multipart customer_id + files[]. Parse index_job_id from JSON.
2. Poll GET /api/index/jobs/{job_id} until status is succeeded or failed (or poll GET /api/index/status).
3. List docs: GET /api/documents?customer_id=...
4. Query: POST /query (or POST /api/query — same handler) JSON { question, customer_id, file?, stream? } OR POST /api/chat with same customer_id.
5. Delete: DELETE /api/documents?file=<basename>&customer_id=...

Use GET /openapi.json from the same base URL for exact schemas. LLM: platform uses Groq first, OpenAI fallback — no provider keys in the browser.
```

### Copy-paste: `curl` examples (production host)

Use an **origin-only** base URL (no `/api` suffix), so paths look like `$BASE_URL/api/upload`:

```bash
export BASE_URL="https://isr.aventhic.com"   # or http://localhost:8000 — origin only
export API_KEY="<your-X-API-Key>"
export CUSTOMER_ID="demo1"
# Absolute path on disk — @ tells curl to read the file bytes (no space after @).
export LOCAL_FILE="/path/to/your/file.pdf"

curl -sS -X POST "$BASE_URL/api/upload" \
  -H "X-API-Key: $API_KEY" \
  -F "customer_id=$CUSTOMER_ID" \
  -F "files=@${LOCAL_FILE}"

# Optional: repeat -F "files=@/other/file.png" for more files (same request).
# Upload response includes index_job_id when auto-index is on — poll it:
# JOB_ID='<paste-from-upload-response>'
# curl -sS "$BASE_URL/api/index/jobs/$JOB_ID" -H "X-API-Key: $API_KEY"

# Wait for indexer (global view; endpoint is often public — key optional)
curl -sS "$BASE_URL/api/index/status"

# Broad query over all docs for this customer
curl -sS -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"question\":\"tell me about the handwritten signature written pmshah\",\"customer_id\":\"$CUSTOMER_ID\",\"stream\":false}"

# Scoped to one uploaded file basename (exact name as stored)
curl -sS -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"question\":\"what is the disease and age?\",\"customer_id\":\"$CUSTOMER_ID\",\"file\":\"Screenshot 2026-04-28 at 1.13.20 PM.png\",\"stream\":false}"

# Delete by doc_id (same customer_id as upload)
export DOC_ID="doc_9c508894c1d0"
curl -sS -X DELETE "${BASE_URL}/api/documents?$(printf '%s' "doc_id=${DOC_ID}&customer_id=${CUSTOMER_ID}")" \
  -H "X-API-Key: $API_KEY"

# Alternative if customer_id/doc_id contain & or spaces: curl 7.87+ —
# curl -sS -X DELETE "$BASE_URL/api/documents" -H "X-API-Key: $API_KEY" \
#   --url-query "doc_id=$DOC_ID" --url-query "customer_id=$CUSTOMER_ID"
```

If nginx already mounts the API under **`https://host/api`** and you prefer that as `$BASE_URL`, then call **`$BASE_URL/upload`** instead of **`$BASE_URL/api/upload`** (avoid **`/api/api/...`**).

---

## 1. Health & Metrics

### GET `/api/health` — stable, public
- Response 200:
  ```json
  { "status": "ok" }
  ```

### GET `/api/metrics` — stable, protected (`admin:read`)
- Headers: `X-API-Key: <key with admin:read-or-wider scope>`
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
- If **`answer`** is empty or cites no useful context, indexing may not be finished — see **§0 Partner guide** (poll `index` jobs or `GET /api/documents`).
- Errors: `401` invalid/missing key, `403` insufficient scope, `422` validation, `500` RAG pipeline failure.

---

## 3. Chat (UI-facing)

### POST `/api/chat` — stable, protected (`chat:run`)
- Headers: `X-API-Key: <tenant API key>`
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
- **Partner note:** use the **same `customer_id`** you used at upload. Deleting by `file` alone can target the wrong row if the same basename exists under another customer.
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
- **Optional server limits** (off by default): when `UPLOAD_LIMITS_ENABLED=true`, the server enforces `MAX_UPLOAD_FILES_PER_REQUEST` (default 3) and `MAX_UPLOAD_BYTES_PER_FILE` (default 30 MiB). Over-limit → `400` (too many files) or `413` (file too large). Response includes `upload_limits_active` when this feature exists.
- **Auto-index:** by default the server enqueues indexing after upload (`AUTO_INDEX_ON_UPLOAD`; set to `false` to disable). Check `auto_index_started` and `index_job_id` in the response.
- Response (representative):
  ```json
  {
    "uploaded": ["a.pdf"],
    "uploaded_docs": [{ "doc_id": "doc_123", "file_name": "a.pdf", "customer_id": "customer-1" }],
    "customer_id": "customer-1",
    "count": 1,
    "images": [],
    "docs": ["a.pdf"],
    "images_count": 0,
    "docs_count": 1,
    "auto_index_started": true,
    "index_job_id": "job_abc123",
    "upload_limits_active": false
  }
  ```

---

## 6. Indexing

Index **triggers** enqueue background work and return immediately with a **`job_id`**. Poll **`GET /api/index/jobs/{job_id}`** for `status`: `queued` → `running` → **`succeeded`** or **`failed`**.  
(`POST /api/index/backfill_*` helpers use a different pattern — immediate `started` / `ok` without a queue `job_id`.)

### POST `/api/index` — stable, protected (`index:run`)
- Full reindex (documents + images).
- Response: `{ "status": "queued", "job_id": "<id>" }`

### POST `/api/index/docs` — stable, protected (`index:run`)
- Body (optional): `{ "files": ["foo.pdf"], "customer_id": "customer-1" }` — omit `files` for full incremental scan.
- Response: `{ "status": "queued", "job_id": "<id>", "targeted": true|false, "count": <n> }`

### POST `/api/index/images` — stable, protected (`index:run`)
- Body (optional): `{ "files": ["x.png"], "customer_id": "customer-1" }`
- Response: `{ "status": "queued", "job_id": "<id>", "targeted": true|false, "count": <n> }`

### GET `/api/index/jobs` — stable, protected (`index:run`)
- Query: `limit` (default 50)
- Response: `{ "jobs": [ { "id", "job_type", "status", "details", "created_at", "updated_at", "started_at", "finished_at", "error_message" } ] }`

### GET `/api/index/jobs/{job_id}` — stable, protected (`index:run`)
- Returns one job object (same shape as list items). **404** if the id is unknown or not owned by this API key’s user/tenant.

### GET `/api/index/status` — stable, public
- Global indexer snapshot: `{ chunk_count, tree_count, image_count, patients[], status, indexing, progress, stage }`  
- Use together with per-document `index_status` from **`GET /api/documents`**.

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

## LLM provider order & failover (Groq main, OpenAI fallback)

**Platform default** (env keys on the server — see `backend/services/rag_service.py::_llm_key_and_provider`):

1. **Groq (main)** — used whenever `GROQ_API_KEY` is set. Chat-style calls use model **`llama-3.3-70b-versatile`** (Groq OpenAI-compatible API).
2. **OpenAI (fallback)** — used when **`GROQ_API_KEY` is not set**; primary model is **`gpt-4o-mini`**.

**During a request** (when Groq is selected but hits limits), the RAG pipeline can **retry with OpenAI** if `OPENAI_API_KEY` is also set — e.g. rate-limit style errors in `retrieval/agentic_rag.py` (`_call_llm`, `_call_llm_stream`) and similar paths in indexing helpers. Clients always see the **same JSON shape**; they do not choose the provider.

**Tenant BYOK** (`/api/tenant/settings`): a tenant can set `llm_mode: tenant_byok` and supply their own provider/key (Groq, OpenAI, or Gemini) instead of the platform default.

**Other providers** (Gemini, HuggingFace, etc.) appear only in specific optional features, not the default `/query` / `/api/chat` path.

**Security:** Groq/OpenAI keys stay on the server. Browser apps use **`VITE_BACKEND_API_KEY`** only to call **this** API, not external LLMs. Restart the backend after rotating env keys.

## Auth & Error Conventions

- All protected endpoints use `X-API-Key`.  
- `401` → key missing/invalid.  
- `503` → server has no `BACKEND_API_KEY` configured **and** registry auth is not usable for resolution (see `backend/security.py`).  
- `400` → validation/path safety violation (including too many files per upload when limits are on).  
- `413` → request body larger than proxy or server allows, or single file exceeds cap when **`UPLOAD_LIMITS_ENABLED=true`**.  
- `404` → file not found in uploads / unknown job id.  
- `500` → unexpected pipeline failure (logged with stack trace).

## Versioning Rule

- The contracts above are **frozen** for this release window.
- Adding a new optional field is allowed.
- Removing a field, renaming a field, or changing a status code → version bump (`/v2`).
