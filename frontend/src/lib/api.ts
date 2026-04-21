import type { ChatResponse, DocumentPage, FieldsExtractResponse, IndexInfo, ResultItem, Source, UploadedFile } from "./types";

const API = "";
const BACKEND_API_KEY = (import.meta.env.VITE_BACKEND_API_KEY as string | undefined)?.trim();

function withAuthHeaders(headers?: HeadersInit): HeadersInit {
  const merged = new Headers(headers ?? {});
  if (BACKEND_API_KEY) {
    merged.set("X-API-Key", BACKEND_API_KEY);
  }
  return merged;
}

export async function sendChat(
  query: string,
  patientFilter?: string,
  webSearch?: boolean,
  fileFilter?: string,
  evaluateRag?: boolean
): Promise<ChatResponse> {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: withAuthHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      query,
      patient_filter: patientFilter,
      web_search: webSearch ?? false,
      file_filter: fileFilter ?? null,
      evaluate_rag: evaluateRag ?? false,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface StreamMeta {
  intent: string;
  reasoning: string;
  results: ResultItem[];
  sources: Source[];
}

export interface StreamDone {
  summary: string;
  sources: Source[];
  results: ResultItem[];
  intent: string;
  reasoning: string;
}

export interface StreamStatus {
  stage: string;
}

export interface StreamHandlers {
  onStatus?: (status: StreamStatus) => void;
  onMeta?: (meta: StreamMeta) => void;
  onToken?: (token: string) => void;
  onDone?: (done: StreamDone) => void;
  onError?: (err: string) => void;
}

/**
 * Stream a chat response. Emits `meta` (sources/results) first, then a series
 * of `token` deltas, then a final `done` event. Returns once the stream closes.
 */
export async function sendChatStream(
  query: string,
  handlers: StreamHandlers,
  opts?: {
    patientFilter?: string;
    webSearch?: boolean;
    fileFilter?: string;
    signal?: AbortSignal;
  }
): Promise<void> {
  const res = await fetch(`${API}/api/chat/stream`, {
    method: "POST",
    headers: withAuthHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      query,
      patient_filter: opts?.patientFilter,
      web_search: opts?.webSearch ?? false,
      file_filter: opts?.fileFilter ?? null,
    }),
    signal: opts?.signal,
  });
  if (!res.ok || !res.body) throw new Error(await res.text());

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const lines = raw.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch {
        parsed = data;
      }
      if (event === "status") handlers.onStatus?.(parsed as StreamStatus);
      else if (event === "meta") handlers.onMeta?.(parsed as StreamMeta);
      else if (event === "token") handlers.onToken?.(typeof parsed === "string" ? parsed : String(parsed));
      else if (event === "done") handlers.onDone?.(parsed as StreamDone);
      else if (event === "error") handlers.onError?.(typeof parsed === "string" ? parsed : JSON.stringify(parsed));
    }
  }
}

export interface EvaluateResult {
  evaluation: Record<string, number> | null;
  evaluation_error: string | null;
  evaluation_notes: string | null;
}

/** Non-blocking RAGAs evaluation. Call AFTER the chat stream completes. */
export async function evaluateChat(
  query: string,
  summary: string,
  results: ResultItem[]
): Promise<EvaluateResult> {
  const res = await fetch(`${API}/api/chat/evaluate`, {
    method: "POST",
    headers: withAuthHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ query, summary, results }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDocumentPage(
  fileName: string,
  page: number
): Promise<DocumentPage> {
  const res = await fetch(
    `${API}/api/documents/page?file=${encodeURIComponent(fileName)}&page=${page}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDocumentText(
  fileName: string,
  search?: string,
  page?: number
): Promise<{ content: string; file_name: string; scroll_line: number; matched_text?: string }> {
  let url = `${API}/api/documents/text?file=${encodeURIComponent(fileName)}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;
  if (page && page > 0) url += `&page=${page}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Run Mistral OCR on an uploaded PDF and download the result as {stem}.md.
 * Re-upload the .md next to the .pdf (same basename) so indexing prefers the OCR text.
 */
export async function downloadMistralOcrMd(fileName: string): Promise<void> {
  const res = await fetch(
    `${API}/api/documents/mistral-ocr-md?file=${encodeURIComponent(fileName)}`
  );
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition");
  let name = fileName.replace(/\.pdf$/i, ".md");
  const m = cd?.match(/filename\*=UTF-8''([^;\s]+)|filename="([^"]+)"/);
  if (m) {
    const raw = m[1] ? decodeURIComponent(m[1]) : m[2];
    if (raw) name = raw;
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export async function listDocuments(): Promise<{ files: UploadedFile[] }> {
  const res = await fetch(`${API}/api/documents`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteDocument(fileName: string): Promise<{ deleted: string }> {
  const res = await fetch(
    `${API}/api/documents?file=${encodeURIComponent(fileName)}`,
    { method: "DELETE", headers: withAuthHeaders() }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface UploadResult {
  uploaded: string[];
  count: number;
  images: string[];
  docs: string[];
  images_count: number;
  docs_count: number;
}

export async function uploadFiles(files: File[]): Promise<UploadResult> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API}/api/upload`, {
    method: "POST",
    headers: withAuthHeaders(),
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function triggerReindex(): Promise<{ status: string }> {
  const res = await fetch(`${API}/api/index`, { method: "POST", headers: withAuthHeaders() });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface TriggerReindexResponse {
  status: string;
  targeted?: boolean;
  count?: number;
}

async function postIndex(endpoint: string, files?: string[]): Promise<TriggerReindexResponse> {
  const init: RequestInit = { method: "POST", headers: withAuthHeaders() };
  if (files && files.length > 0) {
    init.headers = withAuthHeaders({ "Content-Type": "application/json" });
    init.body = JSON.stringify({ files });
  }
  const res = await fetch(`${API}${endpoint}`, init);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Re-index documents. Pass `files` (basenames) to target only specific uploads. */
export async function triggerReindexDocs(files?: string[]): Promise<TriggerReindexResponse> {
  return postIndex("/api/index/docs", files);
}

/** Re-index images. Pass `files` (basenames) to target only specific uploads. */
export async function triggerReindexImages(files?: string[]): Promise<TriggerReindexResponse> {
  return postIndex("/api/index/images", files);
}

export async function getIndexInfo(): Promise<IndexInfo> {
  const res = await fetch(`${API}/api/index/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function extractFields(fileName: string): Promise<FieldsExtractResponse> {
  const res = await fetch(`${API}/api/fields/extract?file=${encodeURIComponent(fileName)}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDocumentImage(fileName: string): Promise<string> {
  return `${API}/api/documents/image?file=${encodeURIComponent(fileName)}`;
}
