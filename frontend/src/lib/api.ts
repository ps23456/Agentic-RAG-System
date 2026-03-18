import type { ChatResponse, DocumentPage, IndexInfo, UploadedFile } from "./types";

const API = "";

export async function sendChat(
  query: string,
  patientFilter?: string,
  webSearch?: boolean
): Promise<ChatResponse> {
  const res = await fetch(`${API}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, patient_filter: patientFilter, web_search: webSearch ?? false }),
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
  search?: string
): Promise<{ content: string; file_name: string; scroll_line: number }> {
  let url = `${API}/api/documents/text?file=${encodeURIComponent(fileName)}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listDocuments(): Promise<{ files: UploadedFile[] }> {
  const res = await fetch(`${API}/api/documents`);
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

export async function getDocumentImage(fileName: string): Promise<string> {
  return `${API}/api/documents/image?file=${encodeURIComponent(fileName)}`;
}
