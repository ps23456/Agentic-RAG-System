import React, { useState, useCallback } from "react";
import {
  ChevronDown,
  Sparkles,
  CheckCircle2,
  Bot,
  Layers,
  Copy,
  Check,
  Globe,
  ExternalLink,
  BarChart3,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage as Msg, ResultItem, Source } from "../../lib/types";
import { SourceBadge } from "./SourceBadge";

const CITATION_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function CitationButton({
  refs,
  sources,
  onSourceClick,
  effectiveQuery,
}: {
  refs: number[];
  sources: Source[];
  onSourceClick: (source: Source, query?: string) => void;
  effectiveQuery?: string;
}) {
  const n = refs[0];
  const source = sources[n - 1];
  if (!source) return <span>[{refs.join(", ")}]</span>;
  return (
    <button
      type="button"
      onClick={() => onSourceClick(source, effectiveQuery)}
      className="inline-flex align-baseline mx-0.5 px-1 py-0.5 min-w-[20px] text-[11px] font-semibold rounded bg-[var(--accent-bg)] text-[var(--accent)] hover:bg-[var(--accent)] hover:text-white transition-colors cursor-pointer"
      title={`View source ${n}: ${source.file_name} page ${source.page}`}
    >
      [{refs.join(",")}]
    </button>
  );
}

function renderWithCitations(
  children: React.ReactNode,
  sources: Source[],
  onSourceClick: (s: Source, q?: string) => void,
  effectiveQuery?: string
): React.ReactNode {
  if (!sources.length) return children;
  return React.Children.map(children, (child) => {
    if (typeof child === "string") {
      const parts: React.ReactNode[] = [];
      let lastIndex = 0;
      let m: RegExpExecArray | null;
      const re = new RegExp(CITATION_RE.source, "g");
      while ((m = re.exec(child)) !== null) {
        parts.push(child.slice(lastIndex, m.index));
        const refs = m[1].split(/\s*,\s*/).map(Number).filter((n) => n >= 1);
        if (refs.length) {
          parts.push(
            <CitationButton
              key={`cit-${m.index}`}
              refs={refs}
              sources={sources}
              onSourceClick={onSourceClick}
              effectiveQuery={effectiveQuery}
            />
          );
        }
        lastIndex = m.index + m[0].length;
      }
      if (parts.length === 0) return child;
      parts.push(child.slice(lastIndex));
      return <>{parts}</>;
    }
    if (child && typeof child === "object" && "props" in child && (child as React.ReactElement).props.children) {
      return {
        ...child,
        props: {
          ...(child as React.ReactElement).props,
          children: renderWithCitations(
            (child as React.ReactElement).props.children,
            sources,
            onSourceClick,
            effectiveQuery
          ),
        },
      };
    }
    return child;
  });
}

function buildSearchContext(r: ResultItem): string | undefined {
  const exact = (r.section_title || "").trim();
  const similar = (r.snippet || "").slice(0, 120).replace(/\s*\.{2,}\s*$/, "").trim();
  if (exact && exact.length >= 8 && exact.length <= 120 && similar) {
    return `${exact}\n${similar}`;
  }
  return similar || undefined;
}

interface Props {
  message: Msg;
  onSourceClick: (source: Source, query?: string) => void;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(() => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);
  return (
    <button
      onClick={copy}
      className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] transition-all"
      title="Copy"
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

export function ChatMessage({ message, query: queryProp, onSourceClick }: Props) {
  const effectiveQuery = message.query ?? queryProp;
  const [showThought, setShowThought] = useState(false);
  const [showChunks, setShowChunks] = useState(false);

  if (message.role === "user") {
    return (
      <div className="flex justify-end mb-6 anim-fade-up">
        <div className="max-w-[70%] bg-[var(--bg-user-msg)] text-[var(--user-msg-text)] px-6 py-3.5 rounded-2xl text-[15px] leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  const hasResults = message.results && message.results.length > 0;
  const hasSources = message.sources && message.sources.length > 0;
  const thinkSec = message.thinkingTime ?? 0;

  return (
    <div className="mb-8 anim-fade-up">
      <div className="flex gap-3.5">
        {/* Bot avatar */}
        <div className="w-7 h-7 rounded-full bg-[#19c37d] flex items-center justify-center shrink-0 mt-1">
          <Bot size={16} className="text-white" strokeWidth={2} />
        </div>

        <div className="flex-1 min-w-0 pt-0.5">
          {/* Thought process */}
          {hasResults && (
            <div className="mb-4">
              <button
                onClick={() => setShowThought(!showThought)}
                className="flex items-center gap-1.5 text-[14px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                Thought for {thinkSec > 0 ? `${thinkSec} second${thinkSec !== 1 ? "s" : ""}` : "a moment"}
                <ChevronDown
                  size={15}
                  className={`transition-transform duration-200 ${showThought ? "" : "-rotate-90"}`}
                />
              </button>

              {showThought && (
                <div className="mt-3 pl-4 border-l-2 border-[var(--border)] space-y-3 anim-fade-up">
                  {/* Find relevant documents — with agent understanding nested inside */}
                  <FindDocumentsStep message={message} />

                  {/* Get page content */}
                  <div className="flex items-center gap-2.5">
                    <CheckCircle2 size={15} className="text-[var(--green)] shrink-0" />
                    <span className="text-[13px] text-[var(--text-secondary)]">
                      Get page content
                      {hasSources && (
                        <span className="text-[var(--text-muted)]"> — {message.sources!.length} pages</span>
                      )}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* RAGAs: above the answer so it is not missed below long replies */}
          {(message.ragasRequested ||
            (message.evaluation && Object.keys(message.evaluation).length > 0) ||
            message.evaluation_error ||
            message.evaluation_notes) && (
            <div className="mb-4 rounded-xl border border-amber-200/80 bg-amber-50/90 dark:border-amber-800/60 dark:bg-amber-950/40 px-3 py-2.5 text-[12px] text-[var(--text-secondary)]">
              <div className="flex items-center gap-1.5 font-medium text-amber-900 dark:text-amber-200 mb-1">
                <BarChart3 size={14} className="text-amber-600 dark:text-amber-400 shrink-0" />
                RAGAs evaluation
              </div>
              {message.evaluation && Object.keys(message.evaluation).length > 0 ? (
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {Object.entries(message.evaluation).map(([k, v]) => (
                    <span key={k}>
                      <span className="text-[var(--text-muted)]">{k.replace(/_/g, " ")}:</span>{" "}
                      <span className="font-mono text-[var(--text-primary)]">
                        {typeof v === "number" ? v.toFixed(3) : String(v)}
                      </span>
                    </span>
                  ))}
                </div>
              ) : null}
              {message.evaluation_notes && (
                <p className="text-[12px] text-[var(--text-muted)] mt-1.5 leading-snug border-t border-amber-200/50 dark:border-amber-800/40 pt-1.5">
                  {message.evaluation_notes}
                </p>
              )}
              {!(message.evaluation && Object.keys(message.evaluation).length > 0) ? (
                message.evaluation_error ? (
                  <p className="text-[13px] text-amber-950/90 dark:text-amber-100/90 leading-snug">
                    {message.evaluation_error}
                  </p>
                ) : message.ragasRequested ? (
                  <p className="text-[13px] text-[var(--text-muted)]">
                    No scores in this response. Send the message again with RAGAs on, or hard-refresh the app so the
                    latest frontend is loaded.
                  </p>
                ) : null
              ) : null}
            </div>
          )}

          {/* Answer content */}
          <div className="chat-prose text-[15px] text-[var(--text-primary)] leading-[1.75]">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                table: ({ children, ...props }) => (
                  <div className="my-4 rounded-xl border border-[var(--border)] overflow-hidden">
                    <div className="flex justify-end gap-1 px-3 py-1.5 bg-[var(--bg-secondary)] border-b border-[var(--border-light)]">
                      <CopyButton text={message.content} />
                    </div>
                    <div className="overflow-x-auto">
                      <table {...props} className="chat-table">{children}</table>
                    </div>
                  </div>
                ),
                p: ({ children, ...props }) => (
                  <p {...props}>
                    {hasSources ? renderWithCitations(children, message.sources!, onSourceClick, effectiveQuery) : children}
                  </p>
                ),
                li: ({ children, ...props }) => (
                  <li {...props}>
                    {hasSources ? renderWithCitations(children, message.sources!, onSourceClick, effectiveQuery) : children}
                  </li>
                ),
                h2: ({ children, ...props }) => (
                  <h2 {...props}>
                    {hasSources ? renderWithCitations(children, message.sources!, onSourceClick, effectiveQuery) : children}
                  </h2>
                ),
                h3: ({ children, ...props }) => (
                  <h3 {...props}>
                    {hasSources ? renderWithCitations(children, message.sources!, onSourceClick, effectiveQuery) : children}
                  </h3>
                ),
                td: ({ children, ...props }) => (
                  <td {...props}>
                    {hasSources ? renderWithCitations(children, message.sources!, onSourceClick, effectiveQuery) : children}
                  </td>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>

          {/* Source badges */}
          {hasSources && (
            <div className="mt-5 flex flex-wrap gap-2">
              {message.sources!.map((s, i) => {
                const result = message.results?.find(
                  (r) => r.file_name === s.file_name && String(r.page) === String(s.page)
                );
                return (
                  <SourceBadge
                    key={`${s.file_name}-${s.page}-${i}`}
                    source={{
                      ...s,
                      searchContext: result ? buildSearchContext(result) : s.searchContext,
                    }}
                    index={i + 1}
                    onClick={(src) => onSourceClick(src, message.query)}
                  />
                );
              })}
            </div>
          )}

          {/* Retrieved chunks */}
          {hasResults && (
            <div className="mt-5 pt-4 border-t border-[var(--border-light)]">
              <button
                onClick={() => setShowChunks(!showChunks)}
                className="flex items-center gap-2 text-[13px] font-medium text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
              >
                <Layers size={14} />
                <span>{message.results!.length} sources retrieved</span>
                <ChevronDown
                  size={13}
                  className={`transition-transform duration-200 ${showChunks ? "" : "-rotate-90"}`}
                />
              </button>
              {showChunks && (
                <div className="mt-3 space-y-1.5 anim-fade-up">
                  {message.results!.slice(0, 15).map((r, i) =>
                    r.type === "web" ? (
                      <a
                        key={i}
                        href={r.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group flex items-start gap-3 px-3 py-2.5 rounded-lg hover:bg-[var(--accent-bg)] cursor-pointer transition-all"
                      >
                        <Globe size={14} className="text-blue-500 mt-0.5 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-[13px] font-medium text-blue-600 group-hover:underline">
                              {r.section_title || r.file_name}
                            </span>
                            <ExternalLink size={11} className="text-blue-400 shrink-0" />
                          </div>
                          {r.snippet && (
                            <p className="text-[12px] text-[var(--text-muted)] leading-relaxed line-clamp-2 mt-0.5">
                              {r.snippet}
                            </p>
                          )}
                          {r.url && (
                            <p className="text-[11px] text-green-700 mt-0.5 truncate">{r.url}</p>
                          )}
                        </div>
                      </a>
                    ) : (
                      <div
                        key={i}
                        className="group flex items-start gap-3 px-3 py-2.5 rounded-lg hover:bg-[var(--bg-secondary)] cursor-pointer transition-all"
                        onClick={() =>
                          onSourceClick(
                            {
                              file_name: r.file_name,
                              page: r.page,
                              title: r.section_title || r.file_name,
                              searchContext: buildSearchContext(r),
                            },
                            effectiveQuery
                          )
                        }
                      >
                        <span className="text-[11px] font-semibold text-[var(--text-muted)] mt-0.5 w-4 text-right shrink-0">
                          {i + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-[13px] font-medium text-[var(--text-primary)]">
                              {r.file_name}
                              {r.page ? ` · p.${r.page}` : ""}
                            </span>
                            {r.patient_name && (
                              <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--accent-bg)] text-[var(--accent)] font-medium">
                                {r.patient_name}
                              </span>
                            )}
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-muted)] uppercase font-medium">
                              {r.type}
                            </span>
                            <span className="text-[11px] text-[var(--text-muted)] font-mono ml-auto">
                              {r.score.toFixed(3)}
                            </span>
                          </div>
                          {r.snippet && (
                            <p className="text-[12px] text-[var(--text-muted)] leading-relaxed line-clamp-2 mt-0.5">
                              {r.snippet}
                            </p>
                          )}
                        </div>
                      </div>
                    )
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FindDocumentsStep({ message }: { message: Msg }) {
  const [open, setOpen] = useState(false);
  const hasAgent = !!(message.reasoning || message.intent);

  return (
    <div>
      <button
        onClick={() => hasAgent && setOpen(!open)}
        className="flex items-center gap-2.5"
      >
        <CheckCircle2 size={15} className="text-[var(--green)] shrink-0" />
        <span className="text-[13px] text-[var(--text-secondary)]">
          Find relevant documents
          <span className="text-[var(--text-muted)]"> — Multimodal Agentic RAG + Tree Index · {message.results?.length ?? 0} results</span>
        </span>
        {hasAgent && (
          <ChevronDown
            size={13}
            className={`text-[var(--text-muted)] transition-transform duration-200 ${open ? "" : "-rotate-90"}`}
          />
        )}
      </button>

      {open && hasAgent && (
        <div className="mt-2 ml-6 p-3.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-light)] anim-fade-up">
          {message.reasoning && (
            <div className="mb-3">
              <p className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1">Reasoning</p>
              <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed">{message.reasoning}</p>
            </div>
          )}
          {message.intent && (
            <div>
              <p className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1">Intent</p>
              <span className="inline-block text-[12px] px-2 py-0.5 rounded-md bg-[var(--accent-subtle)] text-[var(--accent)] font-medium">
                {message.intent}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
