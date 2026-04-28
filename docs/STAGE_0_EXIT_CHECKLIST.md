# Stage 0 — Exit Checklist (Definition of Done)

Use this list to officially close Stage 0 before starting Stage 1 (multi-tenant data model). Each item maps to an artifact already in this repo.

---

## A. API surface frozen

- [x] All locked endpoints documented in `docs/API.md`
  - `GET /api/health`
  - `GET /api/metrics`
  - `POST /query` (+ SSE `stream=true`)
  - `POST /api/chat`, `POST /api/chat/stream`, `POST /api/chat/evaluate`
  - `GET /api/documents`, `GET /api/documents/info`, `GET /api/documents/page`, `GET /api/documents/text`, `GET /api/documents/image`
  - `DELETE /api/documents`
  - `POST /api/upload`
  - `POST /api/index`, `POST /api/index/docs`, `POST /api/index/images`, `GET /api/index/status`
  - `/api/medical/*` (experimental, not part of baseline KPIs)
  - `/api/fields/extract` (experimental)
- [x] Stability levels labelled (stable / experimental).
- [x] Versioning rule recorded: breaking change → `/v2`.

## B. Curl baseline collection

- [x] Runnable script: `scripts/curl_baseline.sh`
- [x] Sections: `health | query | documents | index-status | metrics | auth-negative | chat-stream | upload-and-index`
- [x] Writes are gated behind `RUN_WRITES=1` so the default run is read-only.
- [x] Negative auth path covered (no key, bad key) and observed `401`.

## C. Smoke / contract tests

- [x] `tests/test_smoke.py` runs against the live `uvicorn` instance.
- [x] No pytest dependency required (uses httpx already in `.venv`).
- [x] Last result: **9/9 PASS** on 2026-04-27 (see `docs/STAGE_0_BASELINE.md`).
- [ ] Re-run on every new build before promoting to deployment.

## D. KPI targets and baseline measurements

- [x] Targets recorded in `docs/STAGE_0_BASELINE.md` § 1
  - latency (p50/p95) per endpoint
  - reliability (success rate, indexing success)
  - quality (RAGAs faithfulness ≥ 0.70, answer relevancy ≥ 0.75)
  - cost (avg LLM USD per `/query`)
  - security (100% rejection of missing/invalid API key)
- [x] Measured baseline filled in (§ 2)
- [ ] Quality sample set populated (§ 3) with ≥ 10 representative questions
- [ ] First RAGAs run completed and aggregate scores recorded

## E. Observability minimum

- [x] Structured JSON logs in `backend/main.py::JsonFormatter`.
- [x] Per-request fields: `path`, `method`, `status_code`, `latency_ms`.
- [x] Aggregate metrics endpoint: `/api/metrics` (route count, error rate, chat avg latency).
- [ ] (Stage 1+) export metrics to a real backend (Prometheus / OTel / vendor).

## F. Security sanity

- [x] `.env` and `.env.local` listed in `.gitignore`.
- [x] `.env.example` mirrors current keys with placeholders only.
- [x] Protected endpoints reject missing / invalid `X-API-Key` (verified by smoke).
- [ ] **Rotate the keys that were exposed in screenshots** before any wider rollout.
- [ ] Move `VITE_BACKEND_API_KEY` to a low-privilege client token before enabling public frontend.

## G. Known limits captured (handoff to Stage 1)

- [x] Single-host / local-disk assumptions documented in `docs/STAGE_0_BASELINE.md` § 5.
- [x] No multi-tenant isolation yet.
- [x] No persistent chat history yet.
- [x] No automated cost-per-query yet.

---

## Sign-off

- [ ] Engineering: smoke suite green and KPIs reviewed.
- [ ] Manager / product: targets accepted, limits acknowledged.
- [ ] Stage 1 (multi-tenant data model + auth) approved to start.

> When all checkboxes are filled, Stage 0 is closed. Future API changes require a versioned route or an updated Stage 0 contract.
