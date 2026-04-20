import { useState, useEffect, useCallback, useRef } from "react";
import {
  ArrowLeft,
  Upload,
  FileText,
  Search,
  Loader2,
  FileStack,
  Image,
  Trash2,
  X,
  File,
  ImageIcon,
} from "lucide-react";
import {
  listDocuments,
  uploadFiles,
  deleteDocument,
  triggerReindexDocs,
  triggerReindexImages,
  getIndexInfo,
  downloadMistralOcrMd,
} from "../../lib/api";
import type { UploadedFile, IndexInfo } from "../../lib/types";
import { formatFileSize } from "../../lib/utils";

function getFileIcon(type: string) {
  const ext = type.toLowerCase().replace(".", "");
  if (["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif"].includes(ext)) {
    return ImageIcon;
  }
  if (["pdf"].includes(ext)) return FileText;
  return File;
}

interface Props {
  onBack: () => void;
  onChatWithDoc: (docName: string) => void;
  onExtractFields: (docName: string) => void;
}

export function DocumentsPage({ onBack, onChatWithDoc, onExtractFields }: Props) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [search, setSearch] = useState("");
  const [uploading, setUploading] = useState(false);
  const [indexInfo, setIndexInfo] = useState<IndexInfo | null>(null);
  const [indexingType, setIndexingType] = useState<"" | "docs" | "images">("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [showIndexPrompt, setShowIndexPrompt] = useState(false);
  const [lastIndexResult, setLastIndexResult] = useState<{ type: "success" | "error"; message: string } | null>(null);
  /** PDF basenames uploaded in the last batch — offer Mistral OCR → .md download */
  const [mistralPromptPdfs, setMistralPromptPdfs] = useState<string[]>([]);
  const [mistralLoading, setMistralLoading] = useState<string | null>(null);
  const indexBeforeRef = useRef<{ chunks: number; trees: number; images: number } | null>(null);

  const refresh = useCallback(() => {
    listDocuments().then((d) => setFiles(d.files)).catch(() => {});
    getIndexInfo().then(setIndexInfo).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!indexingType) return;
    const poll = setInterval(async () => {
      try {
        const info = await getIndexInfo();
        setIndexInfo(info);
        if (!info.indexing) {
          clearInterval(poll);
          setIndexingType("");
          setShowIndexPrompt(false);
          refresh();
          // Show completion message with before/after delta
          if (info.status?.startsWith("error:")) {
            setLastIndexResult({ type: "error", message: info.status.replace(/^error:\s*/, "") });
          } else {
            const before = indexBeforeRef.current ?? { chunks: 0, trees: 0, images: 0 };
            const dChunks = info.chunk_count - before.chunks;
            const dTrees = info.tree_count - before.trees;
            const dImages = info.image_count - before.images;
            const parts: string[] = [];
            if (dChunks !== 0) parts.push(`${dChunks >= 0 ? "+" : ""}${dChunks} chunks`);
            if (dTrees !== 0) parts.push(`${dTrees >= 0 ? "+" : ""}${dTrees} trees`);
            if (dImages !== 0) parts.push(`${dImages >= 0 ? "+" : ""}${dImages} images`);
            const deltaMsg = parts.length > 0 ? parts.join(", ") : "no change";
            const totalMsg = `Now: ${info.chunk_count} chunks · ${info.tree_count} trees · ${info.image_count} images`;
            setLastIndexResult({
              type: "success",
              message: indexingType === "docs"
                ? `Documents indexed. ${deltaMsg}. ${totalMsg}`
                : indexingType === "images"
                ? `Images indexed. ${deltaMsg}. ${totalMsg}`
                : `Indexed. ${deltaMsg}. ${totalMsg}`,
            });
          }
          indexBeforeRef.current = null;
          setTimeout(() => setLastIndexResult(null), 8000);
        }
      } catch {
        /* */
      }
    }, 800);
    return () => clearInterval(poll);
  }, [indexingType, refresh]);

  // Keep UI mode aligned with backend status (covers refresh/reopen or stale local state).
  useEffect(() => {
    if (!indexInfo) return;
    if (!indexInfo.indexing) {
      if (indexingType) setIndexingType("");
      return;
    }
    if (indexInfo.status === "indexing_docs" && indexingType !== "docs") {
      setIndexingType("docs");
    } else if (indexInfo.status === "indexing_images" && indexingType !== "images") {
      setIndexingType("images");
    }
  }, [indexInfo, indexingType]);

  const handleUpload = async () => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".pdf,.png,.jpg,.jpeg,.md,.txt,.json,.tiff,.bmp";
    input.onchange = async () => {
      if (!input.files) return;
      setUploading(true);
      try {
        await uploadFiles(Array.from(input.files));
        refresh();
        setShowIndexPrompt(true);
      } catch {
        /* */
      }
      setUploading(false);
    };
    input.click();
  };

  const handleReindexDocs = async () => {
    if (indexingType || indexInfo?.indexing) return;
    indexBeforeRef.current = {
      chunks: indexInfo?.chunk_count ?? 0,
      trees: indexInfo?.tree_count ?? 0,
      images: indexInfo?.image_count ?? 0,
    };
    const resp = await triggerReindexDocs();
    if (resp.status === "started") setIndexingType("docs");
  };

  const handleReindexImages = async () => {
    if (indexingType || indexInfo?.indexing) return;
    indexBeforeRef.current = {
      chunks: indexInfo?.chunk_count ?? 0,
      trees: indexInfo?.tree_count ?? 0,
      images: indexInfo?.image_count ?? 0,
    };
    const resp = await triggerReindexImages();
    if (resp.status === "started") setIndexingType("images");
  };

  const handleMistralDownload = async (fileName: string) => {
    setMistralLoading(fileName);
    try {
      await downloadMistralOcrMd(fileName);
      setLastIndexResult({
        type: "success",
        message: `Downloaded ${fileName.replace(/\.pdf$/i, ".md")}. Upload that .md file here (same name as the PDF) so indexing uses Mistral OCR text.`,
      });
      setTimeout(() => setLastIndexResult(null), 10000);
    } catch (e) {
      setLastIndexResult({
        type: "error",
        message: e instanceof Error ? e.message : "Mistral OCR download failed",
      });
      setTimeout(() => setLastIndexResult(null), 12000);
    } finally {
      setMistralLoading(null);
    }
  };

  const handleDelete = async (fileName: string) => {
    if (deleting) return;
    setDeleting(fileName);
    try {
      await deleteDocument(fileName);
      refresh();
    } catch {
      /* */
    }
    setDeleting(null);
  };

  const isIndexing = !!indexingType || (indexInfo?.indexing ?? false);

  const filtered = files.filter((f) =>
    f.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-primary)]">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-2 rounded-xl hover:bg-[var(--bg-tertiary)] transition-colors text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
          >
            <ArrowLeft size={20} strokeWidth={1.8} />
          </button>
          <div>
            <h1 className="text-xl font-semibold text-[var(--text-primary)] tracking-tight">
              Documents
            </h1>
            <p className="text-xs text-[var(--text-muted)] mt-0.5 max-w-xl">
              Upload and index PDFs, images, and text files.{" "}
              <span className="text-[var(--text-secondary)]">
                Click a file card to chat scoped to that file only, or type the exact filename in your question
                (e.g. <code className="text-[11px] bg-[var(--bg-tertiary)] px-1 rounded">TEE_TBrown (1).pdf</code>).
              </span>
            </p>
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-light)] bg-[var(--bg-secondary)]/50">
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-all shadow-sm disabled:opacity-60"
          >
            {uploading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Upload size={14} />
            )}
            Upload
          </button>

          <div className="w-px h-6 bg-[var(--border)]" />

          <button
            onClick={handleReindexDocs}
            disabled={isIndexing}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition-all shadow-sm ${
              indexingType === "docs"
                ? "bg-[var(--accent-light)] border-[var(--accent)] text-[var(--accent)]"
                : "bg-[var(--bg-primary)] border-[var(--border)] hover:bg-[var(--bg-secondary)] disabled:opacity-50"
            }`}
          >
            {indexingType === "docs" ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <FileStack size={14} />
            )}
            {indexingType === "docs" ? "Indexing Docs..." : "Index Docs"}
          </button>

          <button
            onClick={handleReindexImages}
            disabled={isIndexing}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition-all shadow-sm ${
              indexingType === "images"
                ? "bg-[var(--accent-light)] border-[var(--accent)] text-[var(--accent)]"
                : "bg-[var(--bg-primary)] border-[var(--border)] hover:bg-[var(--bg-secondary)] disabled:opacity-50"
            }`}
          >
            {indexingType === "images" ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Image size={14} />
            )}
            {indexingType === "images" ? "Indexing Images..." : "Index Images"}
          </button>
        </div>

        <div className="flex items-center gap-2 bg-[var(--bg-primary)] rounded-xl px-4 py-2.5 border border-[var(--border)] shadow-sm w-56">
          <Search size={16} className="text-[var(--text-muted)] shrink-0" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search documents..."
            className="bg-transparent outline-none text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] w-full min-w-0"
          />
        </div>
      </div>

      {/* Index prompt - shown after upload */}
      {showIndexPrompt && !isIndexing && (
        <div className="mx-6 mt-3 flex flex-wrap items-center justify-between gap-3 px-4 py-3 bg-[var(--accent-bg)] border border-[var(--accent-light)] rounded-xl">
          <p className="text-sm text-[var(--accent)] font-medium min-w-0">
            Documents uploaded. Index them to make them searchable.
          </p>
          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleReindexDocs}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
            >
              <FileStack size={12} />
              Index Docs
            </button>
            <button
              onClick={handleReindexImages}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
            >
              <Image size={12} />
              Index Images
            </button>
            <button
              onClick={() => setShowIndexPrompt(false)}
              className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors"
              title="Dismiss"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {/* Completion/error message after indexing */}
      {lastIndexResult && (
        <div
          className={`mx-6 mt-3 px-4 py-3 rounded-xl flex items-center gap-3 ${
            lastIndexResult.type === "success"
              ? "bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800"
              : "bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800"
          }`}
        >
          {lastIndexResult.type === "success" ? (
            <span className="text-green-600 dark:text-green-400 text-sm font-medium">✓</span>
          ) : (
            <span className="text-red-600 dark:text-red-400 text-sm font-medium">✗</span>
          )}
          <p className={`text-sm ${lastIndexResult.type === "success" ? "text-green-800 dark:text-green-200" : "text-red-800 dark:text-red-200"}`}>
            {lastIndexResult.message}
          </p>
        </div>
      )}

      {/* Indexing status banner with progress */}
      {isIndexing && (
        <div className="mx-6 mt-3 px-4 py-3 bg-[var(--accent-bg)] border border-[var(--accent-light)] rounded-xl animate-fade-in">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="flex items-center gap-3">
              <Loader2 size={14} className="text-[var(--accent)] animate-spin shrink-0" />
              <span className="text-sm text-[var(--accent)] font-medium">
                {indexingType === "docs" ? "Indexing documents..." : "Indexing images..."}
              </span>
            </div>
            <span className="text-sm font-semibold text-[var(--accent)] shrink-0">
              {indexInfo?.progress ?? 0}%
            </span>
          </div>
          <div className="h-1.5 bg-[var(--accent-light)] rounded-full overflow-hidden">
            <div
              className="h-full bg-[var(--accent)] transition-all duration-300 ease-out"
              style={{ width: `${indexInfo?.progress ?? 0}%` }}
            />
          </div>
          {indexInfo?.stage && (
            <p className="text-xs text-[var(--text-secondary)] mt-1.5 truncate">
              {indexInfo.stage}
            </p>
          )}
        </div>
      )}

      {/* File list */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-sm font-semibold text-[var(--text-secondary)] uppercase tracking-wider">
            My Documents
          </h2>
          {filtered.length > 0 && (
            <span className="text-xs text-[var(--text-muted)]">
              {filtered.length} file{filtered.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>

        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 px-6 rounded-2xl border-2 border-dashed border-[var(--border)] bg-[var(--bg-secondary)]/30">
            <div className="w-16 h-16 rounded-2xl bg-[var(--bg-tertiary)] flex items-center justify-center mb-4">
              <FileText size={32} className="text-[var(--text-muted)]" />
            </div>
            <p className="text-base font-medium text-[var(--text-primary)] mb-1">
              {files.length === 0 ? "No documents yet" : "No matching documents"}
            </p>
            <p className="text-sm text-[var(--text-muted)] text-center max-w-md mb-6">
              {files.length === 0
                ? "Upload PDFs, images, or text files to get started. Then index them to make them searchable."
                : "Try a different search term."}
            </p>
            {files.length === 0 && (
              <button
                onClick={handleUpload}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
              >
                <Upload size={16} />
                Upload Documents
              </button>
            )}
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {filtered.map((f) => {
              const Icon = getFileIcon(f.type);
              const isImage = ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif"].includes(
                f.type.toLowerCase().replace(".", "")
              );
              return (
                <div
                  key={f.name}
                  className="group flex items-center gap-4 p-4 bg-[var(--bg-primary)] border border-[var(--border-light)] rounded-2xl hover:border-[var(--border)] hover:shadow-md transition-all duration-200"
                >
                  <div
                    className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
                      isImage ? "bg-[var(--accent-bg)]" : "bg-[var(--bg-secondary)]"
                    }`}
                  >
                    <Icon
                      size={22}
                      className={isImage ? "text-[var(--accent)]" : "text-[var(--text-muted)]"}
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-[var(--text-primary)] truncate">
                      {f.name}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                      {formatFileSize(f.size)} · {f.type.replace(".", "").toUpperCase()}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                    {[".pdf", ".md"].includes(f.type.toLowerCase()) && (
                      <button
                        type="button"
                        onClick={() => onExtractFields(f.name)}
                        className="px-2.5 py-1.5 rounded-lg text-xs font-medium border border-[var(--border)] bg-[var(--bg-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors"
                        title="Extract fillable field names"
                      >
                        Fields
                      </button>
                    )}
                    {f.type.toLowerCase() === ".pdf" && (
                      <button
                        type="button"
                        onClick={() => handleMistralDownload(f.name)}
                        disabled={mistralLoading !== null}
                        title="Download Mistral OCR as Markdown"
                        className="px-2.5 py-1.5 rounded-lg text-xs font-medium border border-[var(--border)] bg-[var(--bg-secondary)] hover:border-[var(--accent)] hover:text-[var(--accent)] transition-colors disabled:opacity-50"
                      >
                        {mistralLoading === f.name ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          "OCR→MD"
                        )}
                      </button>
                    )}
                    <button
                      onClick={() => onChatWithDoc(f.name)}
                      className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
                    >
                      Chat
                    </button>
                    <button
                      onClick={() => handleDelete(f.name)}
                      disabled={deleting === f.name}
                      className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors disabled:opacity-50"
                      title="Delete"
                    >
                      {deleting === f.name ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Trash2 size={14} />
                      )}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {indexInfo && filtered.length > 0 && (
          <div className="mt-6 flex flex-wrap items-center gap-3 px-4 py-3 bg-[var(--bg-secondary)] rounded-xl text-xs text-[var(--text-muted)] border border-[var(--border-light)]">
            <span className="font-medium text-[var(--text-secondary)]">{indexInfo.chunk_count} chunks</span>
            <span className="text-[var(--border)]">·</span>
            <span>{indexInfo.tree_count} trees</span>
            <span className="text-[var(--border)]">·</span>
            <span>{indexInfo.image_count} images</span>
            {indexInfo.patients.length > 0 && (
              <>
                <span className="text-[var(--border)]">·</span>
                <span>{indexInfo.patients.length} patients</span>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
