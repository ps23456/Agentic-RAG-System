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

export async function uploadFiles(files: File[]): Promise<{ uploaded: string[]; count: number }> {
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

export async function triggerReindexDocs(): Promise<{ status: string }> {
  const res = await fetch(`${API}/api/index/docs`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function triggerReindexImages(): Promise<{ status: string }> {
  const res = await fetch(`${API}/api/index/images`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
