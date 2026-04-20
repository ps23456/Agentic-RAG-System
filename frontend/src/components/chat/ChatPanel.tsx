import { useEffect, useRef, useState } from "react";
import { Bot, FileText, Upload } from "lucide-react";
import type { Conversation, Source, UploadedFile } from "../../lib/types";
import { listDocuments } from "../../lib/api";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";

interface Props {
  conversation: Conversation | null;
  loading: boolean;
  onSend: (query: string, webSearch?: boolean, fileFilter?: string, evaluateRag?: boolean) => void;
  onSourceClick: (source: Source, query?: string) => void;
}

function getDesc(f: UploadedFile): string {
  const name = f.name.replace(/\.[^.]+$/, "").replace(/[_-]/g, " ");
  if (f.type === ".pdf") return `PDF document`;
  if (f.type === ".md") return `Markdown file`;
  return name;
}

export function ChatPanel({ conversation, loading, onSend, onSourceClick }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [docs, setDocs] = useState<{ name: string; description: string }[]>([]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation?.messages.length, loading]);

  useEffect(() => {
    listDocuments()
      .then((d) => setDocs(d.files.map((f) => ({ name: f.name, description: getDesc(f) }))))
      .catch(() => {});
  }, []);

  const isEmpty = !conversation || conversation.messages.length === 0;

  const handleUpload = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".pdf,.png,.jpg,.jpeg,.md,.txt,.json,.tiff,.bmp";
    input.onchange = async () => {
      if (!input.files) return;
      const form = new FormData();
      Array.from(input.files).forEach((f) => form.append("files", f));
      await fetch("/api/upload", { method: "POST", body: form });
      const d = await listDocuments();
      setDocs(d.files.map((f) => ({ name: f.name, description: getDesc(f) })));
    };
    input.click();
  };

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)]">
      {isEmpty ? (
        <>
          <div className="flex-1 flex flex-col items-center justify-center px-6 pb-4">
            {/* Brand + welcome */}
            <div className="w-14 h-14 rounded-full bg-[#19c37d] flex items-center justify-center mb-6 shadow-lg shadow-emerald-100">
              <Bot size={28} className="text-white" strokeWidth={1.8} />
            </div>
            <h1 className="text-2xl font-semibold text-[var(--text-primary)] mb-1.5 tracking-tight">
              How can I help you today?
            </h1>
            <p className="text-[15px] text-[var(--text-muted)] mb-10 max-w-md text-center">
              Search through your documents using AI-powered retrieval
            </p>

            {/* Document cards */}
            <div className="grid grid-cols-2 gap-3 max-w-xl w-full mb-6">
              {docs.slice(0, 4).map((doc) => (
                <button
                  key={doc.name}
                  onClick={() => onSend(`Tell me about ${doc.name}`, false, doc.name, false)}
                  className="group flex items-center gap-3.5 p-4 bg-[var(--bg-primary)] border border-[var(--border)] rounded-2xl hover:border-[var(--accent)] hover:shadow-md transition-all text-left"
                >
                  <div className="w-10 h-10 rounded-xl bg-[var(--bg-secondary)] group-hover:bg-[var(--accent-bg)] flex items-center justify-center shrink-0 transition-colors">
                    <FileText size={18} className="text-[var(--text-muted)] group-hover:text-[var(--accent)] transition-colors" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-[14px] font-medium text-[var(--text-primary)] truncate">{doc.name}</p>
                    <p className="text-[12px] text-[var(--text-muted)] mt-0.5">{doc.description}</p>
                  </div>
                </button>
              ))}
            </div>

            {/* Upload button centered below */}
            <div className="flex justify-center max-w-xl w-full">
              <button
                onClick={handleUpload}
                className="group flex flex-col items-center justify-center gap-2.5 p-4 w-full border-2 border-dashed border-emerald-300 bg-emerald-50/50 rounded-2xl hover:border-emerald-400 hover:bg-emerald-50 transition-all"
              >
                <Upload size={20} className="text-emerald-500 group-hover:text-emerald-600 transition-colors" />
                <span className="text-[13px] font-medium text-emerald-600 group-hover:text-emerald-700 transition-colors">
                  Upload Documents
                </span>
              </button>
            </div>
          </div>

          <ChatInput
            onSend={(q, ws, er) => onSend(q, ws, undefined, er)}
            disabled={loading}
            onUploadClick={handleUpload}
          />
        </>
      ) : (
        <>
          {conversation.scopedFile && (
            <div className="shrink-0 px-6 py-2.5 border-b border-[var(--border)] bg-[var(--bg-secondary)]/60">
              <p className="max-w-3xl mx-auto text-[12px] text-[var(--text-muted)]">
                <span className="font-medium text-[var(--text-secondary)]">Scoped to one file:</span>{" "}
                {conversation.scopedFile}
                <span className="text-[var(--text-muted)]"> — follow-up questions use this file only. Start a new chat to search all documents.</span>
              </p>
            </div>
          )}
          <div className="flex-1 overflow-y-auto px-6 pb-4">
            <div className="max-w-3xl mx-auto">
              <div className="h-14" />
              {conversation.messages.map((m, i) => (
                <ChatMessage
                  key={m.id}
                  message={m}
                  query={
                    m.role === "assistant"
                      ? m.query ?? (conversation.messages[i - 1]?.role === "user" ? conversation.messages[i - 1].content : undefined)
                      : undefined
                  }
                  onSourceClick={onSourceClick}
                />
              ))}
              {loading && (
                <div className="flex gap-3.5 mb-6 anim-fade-up">
                  <div className="w-7 h-7 rounded-full bg-[#19c37d] flex items-center justify-center shrink-0 mt-1">
                    <Bot size={16} className="text-white" strokeWidth={2} />
                  </div>
                  <div className="pt-1.5">
                    <div className="flex items-center gap-1.5">
                      {[0, 150, 300].map((d) => (
                        <span
                          key={d}
                          className="w-[6px] h-[6px] rounded-full bg-[var(--text-muted)]"
                          style={{ animation: "dot-pulse 1.4s infinite ease-in-out", animationDelay: `${d}ms` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </div>
          <ChatInput
            onSend={(q, ws, er) => onSend(q, ws, undefined, er)}
            disabled={loading}
            onUploadClick={handleUpload}
          />
        </>
      )}
    </div>
  );
}
