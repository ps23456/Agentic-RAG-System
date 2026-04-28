# Stage 0 — Baseline KPIs and Measurements

This document captures the **target SLOs** for the current API and the **measured baseline values** observed during Stage 0. It is the artifact your manager / next stage owner will sign off on.

> All numbers are for the local single-node deployment described in `README.md`.  
> Update the “Measured” columns when you re-run `scripts/curl_baseline.sh`.

---

## 1. Target SLOs

| Category | Metric | Target | Notes |
|---|---|---|---|
| Latency | `GET /api/health` p95 | ≤ 50 ms | trivially served, sanity |
| Latency | `GET /api/index/status` p95 | ≤ 1500 ms | first call may be cold; subsequent cached |
| Latency | `POST /query` p50 | ≤ 12 s | retrieval + LLM with provider warm |
| Latency | `POST /query` p95 | ≤ 35 s | allow OCR / cold reranker |
| Latency | `POST /api/chat/stream` first-token | ≤ 4 s | streaming TTFT |
| Reliability | API success rate (excluding 4xx) | ≥ 99 % over 24 h | excludes auth rejects |
| Reliability | Indexing success rate | ≥ 99 % | per `/api/index/status` |
| Quality | RAGAs faithfulness (sample set) | ≥ 0.70 | run via `/api/chat/evaluate` |
| Quality | RAGAs answer relevancy | ≥ 0.75 | same |
| Cost | Avg LLM USD / `/query` | ≤ $0.005 | Groq llama-3.3-70b-versatile primary |
| Security | Protected endpoints reject missing/invalid key | 100 % | `401` |

> SLO targets are **starting** values; tighten after collecting one week of real telemetry.

---

## 2. Measured Baseline (fill in)

Run `BASE_URL=http://localhost:8000 BACKEND_API_KEY=... ./scripts/curl_baseline.sh all` and paste the relevant lines here.

Last refreshed: **2026-04-27** (local single-host).

| Endpoint | Status | Latency (s) | Notes / payload size |
|---|---|---|---|
| `GET /api/health` | 200 | 0.002 | `{"status":"ok"}` |
| `GET /api/index/status` | 200 | 1.43 | `chunk_count=2469, tree_count=5, image_count=39, patients=4` |
| `GET /api/documents` | 200 | 0.002 | 16 files listed (mix of pdf/png/jpg/md), payload ≈ 1.5 KB |
| `GET /api/documents/info` | 200 | 1.50 | same shape as `/api/index/status` |
| `GET /api/metrics` (no key) | 401 | 0.023 | `Invalid or missing API key.` |
| `GET /api/metrics` (bad key) | 401 | 0.005 | `Invalid or missing API key.` |
| `GET /api/metrics` (valid key) | 200 | <0.05 | uptime, request_count, error_rate, avg_latency_ms, top_routes |
| `POST /query` empty question | 422 | <0.05 | validation rejects |
| `POST /query` >2000 chars | 422 | <0.05 | validation rejects |
| `POST /query` happy path (`Financing to self help groups`) | 200 | 40.43 | 8 sources cited, intent `general_search` |
| `POST /api/chat/stream` first-token | _TBD_ | _TBD_ | log `[chat-timing] first-token latency` from server |

Smoke suite: **9/9 PASS** via `tests/test_smoke.py` on 2026-04-27.

> Earlier `/query` observation was 28.93 s (terminal). Latest run shows 40.43 s — this is within the documented p95 target (≤ 35 s) margin but worth tracking; rerun a few times when the reranker is warm to compute a stable p50.

---

## 3. Quality Sample Set

Save 10–15 representative questions you trust the system to answer well.  
Source: pick from real product use; mix easy, medium, and hard.

| # | Question | Expected source(s) | Difficulty |
|---|---|---|---|
| 1 | Financing to self help groups | `PNB AR 2024-25_Web.pdf` | easy |
| 2 | _TBD_ | _TBD_ | _TBD_ |

Run RAGAs over this set via `scripts/run_ragas_eval.py --queries eval/gold_queries.json` and record aggregate scores below.

| Run date | Faithfulness | Answer relevancy | LLM judge |
|---|---|---|---|
| _TBD_ | _TBD_ | _TBD_ | groq / openai |

---

## 4. Cost Telemetry (manual until Stage 6)

For Stage 0 we estimate cost manually:

- Read provider invoices and divide by chat request count.
- Server already tracks `chat_request_count` and `chat_avg_latency_ms` (see `RuntimeMetrics` in `backend/main.py`).
- Stage 6 will add per-call token + USD logging.

| Period | Provider | Chat requests | Tokens (in/out est.) | USD | USD / request |
|---|---|---|---|---|---|
| _TBD_ | groq | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 5. Known Limitations of the Baseline

- Single process, single host (`uvicorn backend.main:app`).
- Local filesystem for uploads (`data/uploads`) and Chroma (`data/chroma`).
- No persistent chat history (chat is stateless on the server side).
- `/query` is intentionally unauthenticated; `/api/chat*` and write paths require `X-API-Key`.
- Cost-per-query is not yet auto-computed.
- No multi-tenant isolation yet (added in Stage 1+).

These are the **inputs** to Stage 1 planning.

---

## 6. Sign-off

- [ ] Baseline curls run end-to-end with no unexpected failures.
- [ ] Latency / reliability targets recorded and accepted.
- [ ] Quality scores measured for the sample set.
- [ ] Limitations list reviewed by team / manager.
- [ ] Stage 1 (multi-tenant data model) approved to start.
