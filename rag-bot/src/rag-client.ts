import type { RagTenantConfig } from "./tenant-map.js";

export interface DocRow {
  doc_id: string;
  name: string;
  index_status: string;
  customer_id?: string;
}

export type ChatStreamEvent =
  | { kind: "meta"; data: unknown }
  | { kind: "status"; data: unknown }
  | { kind: "token"; text: string }
  | { kind: "done"; data: unknown }
  | { kind: "error"; message: string };

function authHeaders(cfg: RagTenantConfig): HeadersInit {
  return {
    "X-API-Key": cfg.apiKey,
    Accept: "application/json",
  };
}

export async function listDocs(cfg: RagTenantConfig, customerId?: string): Promise<DocRow[]> {
  const q = new URLSearchParams();
  const cid = (customerId ?? cfg.customerId).trim();
  if (cid) q.set("customer_id", cid);
  const path = `/api/documents${q.toString() ? `?${q}` : ""}`;
  const res = await fetch(`${cfg.baseUrl}${path}`, { headers: authHeaders(cfg) });
  if (!res.ok) throw new Error(`GET /api/documents ${res.status}: ${await res.text()}`);
  const j = (await res.json()) as { files?: DocRow[] };
  return j.files ?? [];
}

export async function uploadFile(
  cfg: RagTenantConfig,
  data: Buffer,
  filename: string,
  customerId: string,
): Promise<{ index_job_id: string; uploaded: string[] }> {
  const form = new FormData();
  form.append("customer_id", customerId);
  form.append("files", new Blob([new Uint8Array(data)]), filename);
  const res = await fetch(`${cfg.baseUrl}/api/upload`, {
    method: "POST",
    headers: { "X-API-Key": cfg.apiKey },
    body: form,
  });
  if (!res.ok) throw new Error(`POST /api/upload ${res.status}: ${await res.text()}`);
  const j = (await res.json()) as { index_job_id?: string; uploaded?: string[] };
  return {
    index_job_id: j.index_job_id ?? "",
    uploaded: j.uploaded ?? [],
  };
}

export async function pollUntilIndexed(
  cfg: RagTenantConfig,
  jobId: string,
  opts?: { intervalMs?: number; timeoutMs?: number },
): Promise<{ status: string; error_message?: string }> {
  if (!jobId) return { status: "skipped" };
  const intervalMs = opts?.intervalMs ?? 2000;
  const timeoutMs = opts?.timeoutMs ?? 600_000;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const res = await fetch(`${cfg.baseUrl}/api/index/jobs/${encodeURIComponent(jobId)}`, {
      headers: authHeaders(cfg),
    });
    if (!res.ok) throw new Error(`GET /api/index/jobs/${jobId} ${res.status}: ${await res.text()}`);
    const row = (await res.json()) as { status: string; error_message?: string };
    if (row.status === "succeeded" || row.status === "failed") return row;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`index job ${jobId} timed out after ${timeoutMs}ms`);
}

async function* parseSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<{ event: string; data: string }> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trimStart();
      }
      if (data.length) yield { event, data };
    }
  }
}

function parseJsonData(raw: string): unknown {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return raw;
  }
}

export async function* streamChat(
  cfg: RagTenantConfig,
  query: string,
  options?: {
    customer_id?: string;
    patient_filter?: string | null;
    file_filter?: string | null;
    web_search?: boolean;
  },
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${cfg.baseUrl}/api/chat/stream`, {
    method: "POST",
    headers: { ...authHeaders(cfg), "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      customer_id: options?.customer_id ?? cfg.customerId,
      patient_filter: options?.patient_filter ?? null,
      file_filter: options?.file_filter ?? null,
      web_search: options?.web_search ?? false,
    }),
  });
  if (!res.ok) throw new Error(`POST /api/chat/stream ${res.status}: ${await res.text()}`);
  const stream = res.body;
  if (!stream) throw new Error("empty response body");
  for await (const { event, data } of parseSse(stream)) {
    if (event === "error") {
      const parsed = parseJsonData(data);
      const message =
        typeof parsed === "string"
          ? parsed
          : parsed && typeof parsed === "object" && "detail" in parsed
            ? String((parsed as { detail?: unknown }).detail)
            : String(parsed);
      yield { kind: "error", message };
      return;
    }
    const payload = parseJsonData(data);
    if (event === "meta") yield { kind: "meta", data: payload };
    else if (event === "status") yield { kind: "status", data: payload };
    else if (event === "token") {
      const text = typeof payload === "string" ? payload : "";
      yield { kind: "token", text };
    } else if (event === "done") yield { kind: "done", data: payload };
  }
}
