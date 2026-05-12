# RAG API — Integration Guide

This guide covers everything you need to integrate your application with the hosted RAG service. Machine-readable schemas are available at `/openapi.json` and Swagger UI at `/docs` on your deployment.

---

## 1. What the operator gives you

| Item | How you use it |
|------|----------------|
| **Base URL** | e.g. `https://your-company.example.com` (see §3 for path prefix notes) |
| **API key** | HTTP header `X-API-Key: <value>` on every protected request. Tied to **one tenant** in the registry. |
| **`customer_id`** | A string you choose (workspace / project / bucket name). Use the **same** value on upload, list, query, and delete. Must not contain `/`, `\`, or `..`. |

---

## 2. Two different keys — do not confuse them

| Key | Purpose |
|-----|---------|
| **`X-API-Key` (from us)** | Authenticates calls **to our API** (upload, query, …). Required on every protected request. |
| **Groq / OpenAI / Gemini key (BYOK)** | Optional. If saved for your tenant (§9), our servers call the LLM with your provider key instead of ours. Never required for basic upload + query if you stay on the platform default. |

You do **not** send provider keys as a header on `/query` or `/api/chat`. BYOK is configured via `PUT /api/tenant/settings` (requires `admin:write` scope on your `X-API-Key`).

---

## 3. Base URL path

Deployments differ:

- **Origin only:** `BASE_URL=https://example.com` → endpoints are `$BASE_URL/api/upload`, `$BASE_URL/api/query`, …
- **API already prefixed:** If the public root is `https://example.com/api`, use `$BASE_URL/upload` etc. (**not** `$BASE_URL/api/upload`).

Confirm with the operator once to avoid `/api/api/...` double-prefix 404s.

---

## 4. Discover the API (Swagger / OpenAPI)

| Resource | URL |
|----------|-----|
| **Swagger UI** (try requests in browser) | `{BASE_URL}/docs` |
| **OpenAPI JSON** (Postman, codegen, Cursor) | `{BASE_URL}/openapi.json` |

Import `openapi.json` into your tool of choice for exact request/response models.

---

## 5. Scopes (what your key may do)

| Scope | Lets you … |
|-------|------------|
| `docs:write` | Upload and delete documents |
| `docs:read` | List / read document metadata |
| `query:run` | `POST /api/query` |
| `chat:run` | `POST /api/chat` and `POST /api/chat/stream` |
| `index:run` | Trigger reindex, poll `/api/index/jobs*` |
| `admin:read` | Tenant metrics summary |
| `admin:write` | Change tenant BYOK settings (§9) |
| `admin:*` | All of the above |

No `admin:write` → you cannot set your own LLM keys via API; platform default LLM (§10) applies.

---

## 6. End-to-end flow (minimal)

1. **Upload** — `POST /api/upload` · `multipart/form-data`: `customer_id`, one or more `files=@/path/to/file`.
2. **Indexing** — The response may include `index_job_id` (when auto-index is on). Poll `GET /api/index/jobs/{job_id}` until `status` is `succeeded` or `failed`. Optionally poll `GET /api/index/status` for global progress, or `GET /api/documents?customer_id=...` for per-file `index_status`.
3. **Ask** — `POST /api/query` (blocking) or `POST /api/chat/stream` (streaming SSE, see §8).
4. **Delete** — `DELETE /api/documents` with query params `customer_id` and either `doc_id` or `file` (basename).

If `answer` is empty or unhelpful, indexing may still be running — check the job or document `index_status`.

---

## 7. `curl` cheatsheet

Set once:

```bash
export BASE_URL="https://YOUR_HOST"
export API_KEY="YOUR_X_API_KEY"
export CUSTOMER_ID="your_workspace_id"
```

### Health check

```bash
curl -sS "${BASE_URL}/api/health"
```

No auth required. Expect `200 OK` when the service is up.

### Upload one file

`@` attaches file bytes (no space after `@`).

```bash
curl -sS -X POST "$BASE_URL/api/upload" \
  -H "X-API-Key: $API_KEY" \
  -F "customer_id=$CUSTOMER_ID" \
  -F "files=@/absolute/path/to/document.pdf"
```

Add more `-F "files=@..."` lines to upload multiple files in one request.

### Poll index job

Use the `job_id` or `index_job_id` value returned verbatim from the upload or reindex response:

```bash
export JOB_ID="<job_id from API response>"
curl -sS "$BASE_URL/api/index/jobs/$JOB_ID" -H "X-API-Key: $API_KEY"
```

`status` values: `queued`, `running`, `succeeded`, `failed`.

### Global index status

```bash
curl -sS "$BASE_URL/api/index/status"
```

(`X-API-Key` optional if the deployment leaves this endpoint public.)

### List documents

```bash
curl -sS "$BASE_URL/api/documents?customer_id=$CUSTOMER_ID" \
  -H "X-API-Key: $API_KEY"
```

### Query (customer-wide)

```bash
curl -sS -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"question\":\"Your question\",\"customer_id\":\"$CUSTOMER_ID\",\"stream\":false}"
```

Response shape (`QueryResponse`): `answer`, `sources` (array of `{ file, page, title }`), `elapsed_ms`. May also include `intent` and `reasoning` depending on your deployment.

### Query (scoped to one file)

Use the exact uploaded basename:

```bash
curl -sS -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"question\":\"Your question\",\"customer_id\":\"$CUSTOMER_ID\",\"file\":\"Exact File Name.pdf\",\"stream\":false}"
```

### Delete by `doc_id`

```bash
export DOC_ID="doc_xxxxxxxxxxxx"
curl -sS -X DELETE "$BASE_URL/api/documents?doc_id=${DOC_ID}&customer_id=${CUSTOMER_ID}" \
  -H "X-API-Key: $API_KEY"
```

### Delete by file name

```bash
curl -sS -X DELETE "$BASE_URL/api/documents?file=document.pdf&customer_id=${CUSTOMER_ID}" \
  -H "X-API-Key: $API_KEY"
```

### Full reindex (optional)

Enqueues work; response returns immediately. Poll `job_id` as above.

```bash
curl -sS -X POST "$BASE_URL/api/index" -H "X-API-Key: $API_KEY"
```

Response: `{ "status": "queued", "job_id": "..." }`.

---

## 8. Chat / streaming (optional)

Use this if you want to build a **chat UI** with streaming responses. If you only need single-turn Q&A, `POST /api/query` (§7) is simpler.

### Non-streaming chat

```bash
curl -sS -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "query": "Your question",
    "customer_id": "'"$CUSTOMER_ID"'",
    "file_filter": "optional-filename.pdf"
  }'
```

Response shape (`ChatResponse`):

```json
{
  "summary": "The answer text...",
  "sources": [{ "file_name": "policy.pdf", "page": 3, "title": "Section 4" }],
  "results": [{ "type": "text", "file_name": "policy.pdf", "page": 3, "score": 0.91, "snippet": "..." }],
  "intent": "lookup",
  "reasoning": "User asked about deductibles; retrieved section 4..."
}
```

### Streaming chat (SSE)

`POST /api/chat/stream` — same request body as above. The server responds with `Content-Type: text/event-stream`.

Each SSE frame is two lines — an `event:` line naming the event type, then a `data:` line with the JSON payload — separated by a blank line:

```
event: status
data: {"stage":"retrieving"}

event: status
data: {"stage":"generating"}

event: meta
data: {"intent":"lookup","reasoning":"...","sources":[...],"results":[...]}

event: token
data: "The deductible is"

event: token
data: " $500 per year."

event: done
data: {"summary":"The deductible is $500 per year.","sources":[...],"results":[...],"intent":"lookup","reasoning":"..."}
```

**Event reference:**

| `event:` name | `data:` payload | Notes |
|---------------|-----------------|-------|
| `status` | `{ "stage": "retrieving" \| "generating" }` | Sent before `meta`; useful for showing a progress indicator |
| `meta` | `{ "intent", "reasoning", "sources": [...], "results": [...] }` | Sent once after retrieval, before tokens begin |
| `token` | A plain JSON string — the token text itself | Concatenate these to build the full answer |
| `done` | `{ "summary", "sources", "results", "intent", "reasoning" }` | Full `ChatResponse`-shaped payload; stream ends after this |
| `error` | A plain JSON string — the error text | Sent instead of `done` on failure |

**JavaScript example (correct frame parsing):**

```javascript
const response = await fetch(`${BASE_URL}/api/chat/stream`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
  },
  body: JSON.stringify({
    query: "What is the deductible?",
    customer_id: CUSTOMER_ID,
    file_filter: "policy.pdf",   // optional — scope to one file
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
let answer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });

  // SSE frames are separated by double newline
  const frames = buffer.split("\n\n");
  buffer = frames.pop(); // keep any incomplete trailing frame

  for (const frame of frames) {
    const eventLine = frame.split("\n").find(l => l.startsWith("event: "));
    const dataLine  = frame.split("\n").find(l => l.startsWith("data: "));
    if (!eventLine || !dataLine) continue;

    const eventName = eventLine.slice(7).trim();
    const payload   = JSON.parse(dataLine.slice(6));

    if (eventName === "status") {
      showStage(payload.stage);              // e.g. "retrieving" → "generating"
    } else if (eventName === "meta") {
      showSources(payload.sources);          // file_name, page, title
      showResults(payload.results);          // scored retrieval hits
    } else if (eventName === "token") {
      answer += payload;                     // payload is the token string directly
      renderAnswer(answer);
    } else if (eventName === "done") {
      finalizeAnswer(payload);               // full ChatResponse-shaped object
    } else if (eventName === "error") {
      // error data is a plain JSON string today; guard for future object shape
      const msg = typeof payload === "string" ? payload : String(payload?.message ?? payload);
      handleError(msg);
    }
  }
}
```

**`ChatRequest` fields:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `query` | string | ✅ | The user's question |
| `customer_id` | string | recommended | Your tenant bucket — same value as upload |
| `conversation_id` | string | no | Optional label recorded with this turn (for your own analytics/logging). Does **not** automatically load prior messages into RAG context. |
| `file_filter` | string | no | Scope retrieval to one uploaded file (exact basename) |
| `patient_filter` | string | no | Scope retrieval to a specific patient's catalog metadata when relevant |
| `web_search` | boolean | no | Augment retrieval with live web search (default: `false`) |

> **Note:** The chat endpoint uses `query` (not `question`) and `file_filter` (not `file`). These field names differ from `/api/query` — mixing them up returns a `422` validation error.

---

## 9. Prompt for Cursor (or similar)

Paste and fill placeholders:

```
Integrate with this HTTP API:

- Base URL: <PASTE>
- Header on every protected call: X-API-Key: <SECRET>
- customer_id string: "<STABLE_CUSTOMER_BUCKET>"

Flow:
1. POST multipart /api/upload with customer_id and one or more form fields files=@/path (repeat -F "files=@..." per file, as in §7)
2. If response contains index_job_id, poll GET /api/index/jobs/{id} until succeeded|failed
3. POST JSON /api/query with question, customer_id, optional file basename (blocking)
   OR POST JSON /api/chat/stream for streaming SSE (token events)
4. DELETE /api/documents?customer_id=&doc_id= or &file=

Import OpenAPI from: <BASE_URL>/openapi.json
Do not put Groq/OpenAI keys in frontend JS; tenant BYOK is server-side only via PUT /api/tenant/settings.
```

---

## 10. Bring your own LLM key (BYOK)

Requires an API key with `admin:write` (often `admin:*`).

Supported `llm_provider` values: `groq`, `openai`, `gemini`.

### Test connectivity (nothing saved)

```bash
curl -sS -X POST "$BASE_URL/api/tenant/settings/test-connection" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "llm_provider": "openai",
    "llm_api_key": "sk-...",
    "llm_model": "gpt-4o-mini"
  }'
```

### Save BYOK for this tenant

```bash
curl -sS -X PUT "$BASE_URL/api/tenant/settings" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "llm_mode": "tenant_byok",
    "llm_provider": "groq",
    "llm_api_key": "gsk_...",
    "llm_model": "llama-3.3-70b-versatile"
  }'
```

### Return to platform default

```bash
curl -sS -X PUT "$BASE_URL/api/tenant/settings" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"llm_mode":"platform_default","llm_provider":"","llm_api_key":"","llm_model":""}'
```

**Security:** Never put provider keys in a public browser bundle. Prefer server-side custody or a vault. Rotate your `X-API-Key` if it is ever leaked.

---

## 11. Default LLM behavior (platform)

When `llm_mode` is `platform_default`:

- The server prefers **Groq** when `GROQ_API_KEY` is configured.
- Falls back to **OpenAI** (`gpt-4o-mini`-class models) otherwise.
- Some paths retry with OpenAI when Groq returns rate-limit errors and `OPENAI_API_KEY` is available.

Partners see the same JSON response shapes regardless of which provider is active.

---

## 12. Common errors

| HTTP / symptom | Typical cause |
|----------------|---------------|
| **401 / 403** | Bad or missing `X-API-Key`; insufficient scope. |
| **400** on upload | Invalid `customer_id` (contains `/`, `\`, or `..`). |
| **413** | Body too large — reverse proxy `client_max_body_size` or server upload limits. |
| **404** on DELETE | Wrong `customer_id`, `file`, or `doc_id`; or document already deleted. |
| **422** | Request body field mismatch — check you're using `question`/`file` for `/api/query` and `query`/`file_filter` for `/api/chat`. |

---

## 13. Further reading

- **Live Swagger UI:** `{BASE_URL}/docs`
- **OpenAPI JSON:** `{BASE_URL}/openapi.json`