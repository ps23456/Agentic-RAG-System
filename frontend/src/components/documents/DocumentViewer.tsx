import { useState, useEffect, useRef } from "react";
import { ChevronLeft, ChevronRight, X, FileText, ZoomIn, ZoomOut, Search } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { ViewerState } from "../../lib/types";
import { getDocumentPage, getDocumentText } from "../../lib/api";

interface Props {
  viewers: ViewerState[];
  activeIdx: number;
  onTabClick: (idx: number) => void;
  onClose: (idx: number) => void;
  onPageChange: (page: number) => void;
  onTotalPages: (total: number) => void;
}

export function DocumentViewer({
  viewers, activeIdx, onTabClick, onClose, onPageChange, onTotalPages,
}: Props) {
  const [pageImage, setPageImage] = useState<string | null>(null);
  const [textContent, setTextContent] = useState<string | null>(null);
  const [scrollLine, setScrollLine] = useState(0);
  const [matchedText, setMatchedText] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [zoom, setZoom] = useState(100);
  const contentRef = useRef<HTMLDivElement>(null);
  const highlightRef = useRef<HTMLSpanElement>(null);

  const v = viewers[activeIdx];

  useEffect(() => {
    if (!v) {
      setPageImage(null);
      setTextContent(null);
      setMatchedText("");
      return;
    }
    const ext = v.fileName.split(".").pop()?.toLowerCase() || "";

    if (ext === "pdf") {
      setLoading(true);
      setTextContent(null);
      getDocumentPage(v.fileName, v.page)
        .then((data) => { setPageImage(data.image); onTotalPages(data.total_pages); })
        .catch(() => setPageImage(null))
        .finally(() => setLoading(false));
    } else if (["md", "txt", "csv", "json"].includes(ext)) {
      setLoading(true);
      setPageImage(null);
      const pageNum = typeof v?.page === "number" ? v.page : parseInt(String(v?.page)) || 1;
      getDocumentText(v.fileName, v.searchContext, pageNum)
        .then((data) => {
          setTextContent(data.content);
          setScrollLine(data.scroll_line || 0);
          setMatchedText(data.matched_text || "");
        })
        .catch(() => setTextContent(null))
        .finally(() => setLoading(false));
    } else if (["jpg", "jpeg", "png", "gif", "webp", "bmp"].includes(ext)) {
      setTextContent(null);
      setPageImage(`/api/documents/image?file=${encodeURIComponent(v.fileName)}`);
    }
  }, [v?.fileName, v?.page, v?.searchContext]);

  useEffect(() => {
    if (!textContent) return;
    const scrollToTarget = () => {
      if (highlightRef.current) {
        highlightRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
      } else if (scrollLine > 0 && contentRef.current) {
        const lines = contentRef.current.querySelectorAll("p, li, h1, h2, h3, h4, h5, h6, pre, tr");
        const targetIdx = Math.min(scrollLine, lines.length - 1);
        if (lines[targetIdx]) {
          lines[targetIdx].scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    };
    const t = setTimeout(scrollToTarget, 150);
    return () => clearTimeout(t);
  }, [textContent, scrollLine]);

  if (viewers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-[var(--bg-secondary)] border-l border-[var(--border)]">
        <div className="w-12 h-12 rounded-xl bg-[var(--bg-primary)] border border-[var(--border)] flex items-center justify-center mb-3 shadow-xs">
          <FileText size={22} className="text-[var(--text-muted)]" />
        </div>
        <p className="text-[13px] text-[var(--text-muted)]">Click a source to view the document here</p>
      </div>
    );
  }

  const ext = v?.fileName.split(".").pop()?.toLowerCase() || "";
  const isPdf = ext === "pdf";
  const isImage = ["jpg", "jpeg", "png", "gif", "webp", "bmp"].includes(ext);
  const searchCtx = (matchedText || v?.searchContext || "").trim();

  function highlightText(text: string): React.ReactNode {
    if (!searchCtx || searchCtx.length < 5) return text;
    const idx = text.toLowerCase().indexOf(searchCtx.toLowerCase());
    if (idx < 0) return text;
    return (
      <>
        {text.slice(0, idx)}
        <span ref={highlightRef} className="bg-yellow-200 px-0.5 rounded">{text.slice(idx, idx + searchCtx.length)}</span>
        {text.slice(idx + searchCtx.length)}
      </>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[var(--bg-secondary)] border-l border-[var(--border)]">
      {/* Tabs */}
      <div className="flex items-center bg-[var(--bg-primary)] border-b border-[var(--border)] px-1 min-h-[38px] overflow-x-auto">
        {viewers.map((tab, i) => {
          const short = tab.fileName.length > 18 ? tab.fileName.slice(0, 16) + "..." : tab.fileName;
          return (
            <div
              key={`${tab.fileName}-${i}`}
              className={`flex items-center gap-1.5 px-3 py-[7px] text-[12px] cursor-pointer transition-all shrink-0 border-b-2 ${
                i === activeIdx
                  ? "border-[var(--accent)] text-[var(--accent)] font-medium"
                  : "border-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
              onClick={() => onTabClick(i)}
            >
              <FileText size={12} />
              <span>{short}</span>
              <button
                onClick={(e) => { e.stopPropagation(); onClose(i); }}
                className="ml-0.5 opacity-40 hover:opacity-100 hover:text-red-500 transition-all"
              >
                <X size={10} />
              </button>
            </div>
          );
        })}
      </div>

      {/* Content */}
      <div ref={contentRef} className="flex-1 overflow-auto p-3">
        {loading && (
          <div className="flex items-center justify-center h-full">
            <div className="w-7 h-7 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {!loading && isPdf && pageImage && (
          <div className="flex justify-center">
            <img
              src={pageImage.startsWith("data:") || pageImage.startsWith("/") ? pageImage : `data:image/png;base64,${pageImage}`}
              alt={`${v.fileName} page ${v.page}`}
              className="rounded shadow-md"
              style={{ width: `${zoom}%` }}
            />
          </div>
        )}

        {!loading && textContent !== null && (
          <div className="bg-[var(--bg-primary)] rounded-xl border border-[var(--border)] p-5 shadow-xs">
            {ext === "md" ? (
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p>{typeof children === "string" ? highlightText(children) : children}</p>,
                    li: ({ children }) => <li>{typeof children === "string" ? highlightText(children) : children}</li>,
                    h1: ({ children }) => <h1>{typeof children === "string" ? highlightText(children) : children}</h1>,
                    h2: ({ children }) => <h2>{typeof children === "string" ? highlightText(children) : children}</h2>,
                    h3: ({ children }) => <h3>{typeof children === "string" ? highlightText(children) : children}</h3>,
                  }}
                >
                  {textContent}
                </ReactMarkdown>
              </div>
            ) : (
              <pre className="whitespace-pre-wrap text-[12px] text-[var(--text-secondary)] leading-relaxed font-mono">{textContent}</pre>
            )}
          </div>
        )}

        {!loading && isImage && pageImage && (
          <div className="flex justify-center">
            <img src={pageImage} alt={v.fileName} className="rounded shadow-md max-w-full" style={{ width: `${zoom}%` }} />
          </div>
        )}
      </div>

      {/* Bottom bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-[var(--bg-primary)] border-t border-[var(--border)] min-h-[36px]">
        <div className="flex items-center gap-1">
          {isPdf && v && (
            <>
              <button
                onClick={() => v.page > 1 && onPageChange(v.page - 1)}
                disabled={v.page <= 1}
                className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-25 transition-all"
              >
                <ChevronLeft size={14} className="text-[var(--text-secondary)]" />
              </button>
              <span className="text-[12px] text-[var(--text-secondary)] font-medium px-1 tabular-nums">
                Page {v.page} / {v.totalPages || "?"}
              </span>
              <button
                onClick={() => (!v.totalPages || v.page < v.totalPages) && onPageChange(v.page + 1)}
                disabled={!!v.totalPages && v.page >= v.totalPages}
                className="p-1 rounded hover:bg-[var(--bg-tertiary)] disabled:opacity-25 transition-all"
              >
                <ChevronRight size={14} className="text-[var(--text-secondary)]" />
              </button>
            </>
          )}
        </div>
        <div className="flex items-center gap-0.5">
          <button onClick={() => setZoom((z) => Math.max(50, z - 10))} className="p-1 rounded hover:bg-[var(--bg-tertiary)] transition-all">
            <ZoomOut size={13} className="text-[var(--text-muted)]" />
          </button>
          <span className="text-[11px] text-[var(--text-muted)] w-8 text-center font-mono tabular-nums">{zoom}%</span>
          <button onClick={() => setZoom((z) => Math.min(200, z + 10))} className="p-1 rounded hover:bg-[var(--bg-tertiary)] transition-all">
            <ZoomIn size={13} className="text-[var(--text-muted)]" />
          </button>
          <div className="w-px h-4 bg-[var(--border)] mx-1" />
          <button className="p-1 rounded hover:bg-[var(--bg-tertiary)] transition-all">
            <Search size={13} className="text-[var(--text-muted)]" />
          </button>
        </div>
      </div>
    </div>
  );
}
