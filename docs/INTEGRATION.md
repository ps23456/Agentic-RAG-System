# RAG API — Integration guide for partners

This is the short path to integrate your own app, script, or frontend with **our** hosted RAG service. Full field-level detail lives in **`API.md`**; machine-readable schemas come from **`/openapi.json`** and Swagger UI (**`/docs`**) on each deployment.

---

## 1. What the operator gives you

| Item | How you use it |
|------|----------------|
| **Base URL** | e.g. `https://your-company.example.com` (see §3 for nginx quirks). |
| **API key** | HTTP header **`X-API-Key: <value>`** on every protected request. This key is tied to **one tenant** in our registry. |
| **`customer_id`** | A string you choose (workspace / project / “patient”). Use the **same** value on **upload**, **list**, **query**, and **delete**. Must not contain `/`, `\`, or `..`. |

---

## 2. Two different keys (do not confuse them)

| Key | Purpose |
|-----|---------|
| **`X-API-Key` (from us)** | Authenticates calls **to our API** (upload, query, …). Required for integration. |
| **Groq / OpenAI / Gemini key (BYOK)** | Optional. If saved for your tenant (**§9**), **our servers** call the LLM with **your** provider key instead of ours. Never required for basic upload + query **if** you stay on platform default. |

You do **not** send provider keys as an extra header on **`/query`**; BYOK is configured via **`PUT /api/tenant/settings`** (needs **`admin:write`** scope on **your** `X-API-Key`).

---

## 3. Base URL path

Deployments differ:

- **Origin only:** `BASE_URL=https://example.com` → endpoints look like **`$BASE_URL/api/upload`**, **`$BASE_URL/api/query`**, …
- **API already prefixed:** If the public root is **`https://example.com/api`**, then use **`$BASE_URL/upload`** etc. (**not** `$BASE_URL/api/upload`).

Confirm with the operator once to avoid **`/api/api/...`** or **404**s.

---

## 4. Discover the API (Swagger / OpenAPI)

| Resource | URL |
|----------|-----|
| **Swagger UI** (try requests in browser) | `{BASE_URL}/docs` |
| **OpenAPI JSON** (Postman, codegen, Cursor) | `{BASE_URL}/openapi.json` |

Use the **same host** you use for `curl`. Import **`openapi.json`** into your tool of choice for exact request/response models.

**Full human reference:** [API.md](./API.md)

---

## 5. Scopes (what your key may do)

When the operator creates your API key, they assign **scopes**. Typical names:

| Scope | Lets you … |
|-------|-------------|
| `docs:write` | Upload and delete documents |
| `docs:read` | List/read document metadata |
| `query:run` | **`POST /query`** / **`POST /api/query`** |
| `chat:run` | **`POST /api/chat`** (and streaming) |
| `index:run` | Trigger reindex, list/poll **`/api/index/jobs*`** |
| `admin:read` | Metrics tenant summary (if exposed) |
| `admin:write` | Change **tenant BYOK** settings (**§9**) |
| `admin:*` | All of the above |

No **`admin:write`** ⇒ you **cannot** set your own LLM keys via API; platform default LLM (**§10**) applies.

---

## 6. End-to-end flow (minimal)

1. **Upload** — `POST /api/upload` · `multipart/form-data`: `customer_id`, one or more `files=@/absolute/path/to/file`.
2. **Indexing** — Response may include **`index_job_id`** (when auto-index is on). Poll **`GET /api/index/jobs/{job_id}`** until **`status`** is **`succeeded`** or **`failed`**. Optionally poll **`GET /api/index/status`** for global progress, or **`GET /api/documents?customer_id=...`** for per-file **`index_status`**.
3. **Ask** — **`POST /query`** or **`POST /api/query`** (same handler) JSON: `question`, `customer_id`, optional `file` (basename), optional `stream`.
4. **Delete** — **`DELETE /api/documents`** with query **`customer_id`** and either **`doc_id`** or **`file`** (basename). Use the **same** `customer_id` as upload.

If **`answer`** is empty or useless, indexing may still be running—check jobs or document **`index_status`**.

---

## 7. `curl` cheatsheet

Set once:

```bash
export BASE_URL="https://YOUR_HOST"     # origin only — see §3
export API_KEY="YOUR_X_API_KEY"
export CUSTOMER_ID="your_workspace_id"
```

### Upload one file

`@` attaches file bytes (**no space** after `@`).

```bash
curl -sS -X POST "$BASE_URL/api/upload" \
  -H "X-API-Key: $API_KEY" \
  -F "customer_id=$CUSTOMER_ID" \
  -F "files=@/absolute/path/to/document.pdf"
```

Multiple files in one request: add more **`-F "files=@..."`** lines.

### Poll index job

After upload (**`index_job_id`** in JSON) or after **`POST /api/index*`** (**`job_id`**):

```bash
export JOB_ID="job_xxxxxxxx"
curl -sS "$BASE_URL/api/index/jobs/$JOB_ID" -H "X-API-Key: $API_KEY"
```

Job **`status`** values include **`queued`**, **`running`**, **`succeeded`**, **`failed`**.

### Global index status

```bash
curl -sS "$BASE_URL/api/index/status"
```

(`X-API-Key` optional if deployment leaves this public.)

### Query (customer-wide)

```bash
curl -sS -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"question\":\"Your question\",\"customer_id\":\"$CUSTOMER_ID\",\"stream\":false}"
```

### Query (scoped to one file)

Use the exact uploaded **basename**:

```bash
curl -sS -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"question\":\"Your question\",\"customer_id\":\"$CUSTOMER_ID\",\"file\":\"Exact File Name.pdf\",\"stream\":false}"
```

### Full reindex (optional, operator/agreed capability)

Enqueue work; poll **`job_id`** as above (**not** synchronous).

```bash
curl -sS -X POST "$BASE_URL/api/index" -H "X-API-Key: $API_KEY"
```

Response shape: **`{ "status": "queued", "job_id": "..." }`**.

### Delete by `doc_id`

```bash
export DOC_ID="doc_xxxxxxxxxxxx"
curl -sS -X DELETE "$BASE_URL/api/documents?doc_id=${DOC_ID}&customer_id=${CUSTOMER_ID}" \
  -H "X-API-Key: $API_KEY"
```

### Delete by file name

```bash
curl -sS -X DELETE "$BASE_URL/api/documents?file=embedding_space.png&customer_id=${CUSTOMER_ID}" \
  -H "X-API-Key: $API_KEY"
```

*(If IDs contain problematic characters for the shell, use your client’s `--url-query` or encode query parameters.)*

---

## 8. Prompt for Cursor (or similar)

Paste and fill placeholders:

```
Integrate with this HTTP API:

- Base URL: <PASTE>
- Header on every protected call: X-API-Key: <SECRET>
- customer_id string: "<STABLE_CUSTOMER_BUCKET>"

Flow:
1. POST multipart /api/upload with customer_id and files[]= @ paths
2. If JSON contains index_job_id, poll GET /api/index/jobs/{id} until succeeded|failed
3. POST JSON /api/query (or /query) with question, customer_id, optional file basename
4. DELETE /api/documents?customer_id=&doc_id= or &file=

Import OpenAPI from: <BASE_URL>/openapi.json
Do not put Groq/OpenAI keys in frontend JS; tenant BYOK is server-side only if we add PUT /api/tenant/settings behind admin.
See docs/INTEGRATION.md and docs/API.md in the supplier repo if attached.
```

---

## 9. Bring your own LLM key (BYOK)

**Requires** an API key that includes **`admin:write`** (often **`admin:*`**).

Supported **`llm_provider`** values today: **`groq`**, **`openai`**, **`gemini`**. (**Anthropic / Claude is not configured** via this API yet.)

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

### Return to operator’s platform keys

```bash
curl -sS -X PUT "$BASE_URL/api/tenant/settings" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"llm_mode":"platform_default","llm_provider":"","llm_api_key":"","llm_model":""}'
```

**Security:** In production, avoid typing provider keys only in a **public browser bundle**. Prefer server-side custody or a vault; your **X-API-Key** to our API should also be rotated if leaked.

---

## 10. Default LLM behavior (platform)

When **`llm_mode` is `platform_default`:**

- Our server prefers **Groq** when **`GROQ_API_KEY`** is configured.
- Otherwise **OpenAI** (`gpt-4o-mini` class models in code paths).
- Some paths **retry with OpenAI** when Groq returns rate-limit-class errors **and** **`OPENAI_API_KEY`** exists.

Partners see the **same** JSON shapes; they do **not** pick provider per **`/query`** request unless we add such a field later.

---

## 11. Common errors

| HTTP / symptom | Typical cause |
|----------------|----------------|
| **401 / 403** | Bad or missing **`X-API-Key`**; insufficient **scope**. |
| **400** upload | Invalid **`customer_id`**. |
| **413** | Body too large (reverse proxy **`client_max_body_size`**) **or** server upload limits (`UPLOAD_LIMITS_ENABLED`). |
| **404** DELETE | Wrong **`customer_id`**, **`file`**, **`doc_id`**, or already deleted. |

---

## 12. Further reading

- **[API.md](./API.md)** — full route reference, metering, tenant settings payloads, nuances.
- **Live** **`/docs`** · **`/openapi.json`** on your deployed host.
