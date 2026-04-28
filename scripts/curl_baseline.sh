#!/usr/bin/env bash
# Stage 0 — Baseline curl collection for the Insurance Claim Search / RAG API.
#
# Purpose:
#   - Reproducible smoke checks for the locked endpoints.
#   - Used as the "Definition of Done" reference for Stage 0.
#
# Usage:
#   chmod +x scripts/curl_baseline.sh
#   BASE_URL=http://localhost:8000 \
#   BACKEND_API_KEY=your_server_key \
#   ./scripts/curl_baseline.sh                    # run all
#   ./scripts/curl_baseline.sh health             # run a single section
#
# Notes:
#   - Read-only by default. Section "upload-and-index" is opt-in via `RUN_WRITES=1`
#     because it modifies your local data folder.
#   - Latency is printed per call so you can populate the Stage 0 baseline doc.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
BACKEND_API_KEY="${BACKEND_API_KEY:-}"
SAMPLE_QUERY="${SAMPLE_QUERY:-Financing to self help groups}"
SAMPLE_FILE="${SAMPLE_FILE:-}"          # absolute path to a small PDF/MD for upload tests
RUN_WRITES="${RUN_WRITES:-0}"           # set to 1 to allow upload+index calls

# --- helpers -----------------------------------------------------------------
_call() {
  # _call <label> <curl_args...>
  local label="$1"; shift
  echo "==> ${label}"
  # %{http_code} and %{time_total} appended as a trailing line for parsing.
  curl -sS -o /tmp/curl_baseline_body.json \
       -w 'HTTP %{http_code}  time %{time_total}s  size %{size_download}B\n' \
       "$@" || true
  if [ -s /tmp/curl_baseline_body.json ]; then
    # Pretty-print first 1KB so the suite stays readable.
    head -c 1024 /tmp/curl_baseline_body.json
    echo
  fi
  echo
}

_require_key() {
  if [ -z "${BACKEND_API_KEY}" ]; then
    echo "Skipping ${1}: BACKEND_API_KEY not set." >&2
    return 1
  fi
}

# --- sections ----------------------------------------------------------------
section_health() {
  _call "GET /api/health" "${BASE_URL}/api/health"
}

section_query() {
  # Public endpoint (intentionally unauthenticated in current code).
  _call "POST /query (sample question)" \
    -X POST "${BASE_URL}/query" \
    -H "Content-Type: application/json" \
    -d "{\"question\":\"${SAMPLE_QUERY}\"}"
}

section_documents() {
  _call "GET /api/documents" "${BASE_URL}/api/documents"
  _call "GET /api/documents/info" "${BASE_URL}/api/documents/info"
}

section_index_status() {
  _call "GET /api/index/status" "${BASE_URL}/api/index/status"
}

section_metrics() {
  if _require_key "GET /api/metrics"; then
    _call "GET /api/metrics" \
      -H "X-API-Key: ${BACKEND_API_KEY}" \
      "${BASE_URL}/api/metrics"
  fi
}

section_protected_auth_negative() {
  # Verifies protected endpoint rejects missing/invalid keys.
  _call "GET /api/metrics (no key — expect 401/503)" \
    "${BASE_URL}/api/metrics"
  _call "GET /api/metrics (bad key — expect 401)" \
    -H "X-API-Key: definitely-wrong-key" \
    "${BASE_URL}/api/metrics"
}

section_chat_stream() {
  if _require_key "POST /api/chat/stream"; then
    _call "POST /api/chat/stream (SSE first bytes)" \
      -X POST "${BASE_URL}/api/chat/stream" \
      -H "Content-Type: application/json" \
      -H "X-API-Key: ${BACKEND_API_KEY}" \
      --max-time 5 \
      -d "{\"query\":\"${SAMPLE_QUERY}\"}" || true
  fi
}

section_upload_and_index() {
  if [ "${RUN_WRITES}" != "1" ]; then
    echo "Skipping upload-and-index (set RUN_WRITES=1 to enable). It modifies data/."
    return 0
  fi
  if [ -z "${SAMPLE_FILE}" ] || [ ! -f "${SAMPLE_FILE}" ]; then
    echo "Skipping upload-and-index: SAMPLE_FILE not set or not found."
    return 0
  fi
  _require_key "POST /api/upload" || return 0
  _call "POST /api/upload" \
    -X POST "${BASE_URL}/api/upload" \
    -H "X-API-Key: ${BACKEND_API_KEY}" \
    -F "files=@${SAMPLE_FILE}"

  _call "POST /api/index/docs" \
    -X POST "${BASE_URL}/api/index/docs" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${BACKEND_API_KEY}" \
    -d '{}'
}

# --- runner ------------------------------------------------------------------
run_all() {
  section_health
  section_query
  section_documents
  section_index_status
  section_metrics
  section_protected_auth_negative
  section_chat_stream
  section_upload_and_index
}

case "${1:-all}" in
  all) run_all ;;
  health) section_health ;;
  query) section_query ;;
  documents) section_documents ;;
  index-status) section_index_status ;;
  metrics) section_metrics ;;
  auth-negative) section_protected_auth_negative ;;
  chat-stream) section_chat_stream ;;
  upload-and-index) section_upload_and_index ;;
  *)
    echo "Unknown section: ${1}" >&2
    echo "Available: all | health | query | documents | index-status | metrics | auth-negative | chat-stream | upload-and-index" >&2
    exit 2 ;;
esac
