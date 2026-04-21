import type { ChatResponse, DocumentPage, FieldsExtractResponse, IndexInfo, UploadedFile } from "./types";

const API = "";

export async function sendChat(
  query: string,
  patientFilter?: string,
  webSearch?: boolean,
  fileFilter?: string,
  evaluateRag?: boolean
): Promise<ChatResponse> {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
    { method: "DELETE" }
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
  const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function triggerReindex(): Promise<{ status: string }> {
  const res = await fetch(`${API}/api/index`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface TriggerReindexResponse {
  status: string;
  targeted?: boolean;
  count?: number;
}

async function postIndex(endpoint: string, files?: string[]): Promise<TriggerReindexResponse> {
  const init: RequestInit = { method: "POST" };
  if (files && files.length > 0) {
    init.headers = { "Content-Type": "application/json" };
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
