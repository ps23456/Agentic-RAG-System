import { useState, useEffect, useCallback } from "react";
import {
  ArrowLeft,
  Upload,
  FileText,
  Search,
  Loader2,
  FileStack,
  Image,
} from "lucide-react";
import {
  listDocuments,
  uploadFiles,
  triggerReindexDocs,
  triggerReindexImages,
  getIndexInfo,
} from "../../lib/api";
import type { UploadedFile, IndexInfo } from "../../lib/types";
import { formatFileSize } from "../../lib/utils";

interface Props {
  onBack: () => void;
  onChatWithDoc: (docName: string) => void;
}

export function DocumentsPage({ onBack, onChatWithDoc }: Props) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [search, setSearch] = useState("");
  const [uploading, setUploading] = useState(false);
  const [indexInfo, setIndexInfo] = useState<IndexInfo | null>(null);
  const [indexingType, setIndexingType] = useState<"" | "docs" | "images">("");

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
          refresh();
        }
      } catch {
        /* */
      }
    }, 3000);
    return () => clearInterval(poll);
  }, [indexingType, refresh]);

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
      } catch {
        /* */
      }
      setUploading(false);
    };
    input.click();
  };

  const handleReindexDocs = async () => {
    if (indexingType || indexInfo?.indexing) return;
    setIndexingType("docs");
    await triggerReindexDocs();
  };

  const handleReindexImages = async () => {
    if (indexingType || indexInfo?.indexing) return;
    setIndexingType("images");
    await triggerReindexImages();
  };

  const isIndexing = !!indexingType || (indexInfo?.indexing ?? false);

  const filtered = files.filter((f) =>
    f.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-1.5 rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
          >
            <ArrowLeft size={18} className="text-[var(--text-secondary)]" />
          </button>
          <h1 className="text-lg font-semibold text-[var(--text-primary)]">
            Documents
          </h1>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-[var(--border-light)]">
        <div className="flex items-center gap-2">
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-[var(--bg-primary)] border border-[var(--border)] hover:bg-[var(--bg-secondary)] transition-all shadow-sm"
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
            {indexingType === "docs" ? "Indexing Docs..." : "Re-index Docs"}
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
            {indexingType === "images"
              ? "Indexing Images..."
              : "Re-index Images"}
          </button>
        </div>

        <div className="flex items-center gap-2 bg-[var(--bg-secondary)] rounded-xl px-3 py-2 border border-[var(--border-light)]">
          <Search size={14} className="text-[var(--text-muted)]" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search documents..."
            className="bg-transparent outline-none text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] w-48"
          />
        </div>
      </div>

      {/* Indexing status banner */}
      {isIndexing && (
        <div className="mx-6 mt-3 flex items-center gap-3 px-4 py-2.5 bg-[var(--accent-bg)] border border-[var(--accent-light)] rounded-xl animate-fade-in">
          <Loader2 size={14} className="text-[var(--accent)] animate-spin" />
          <span className="text-sm text-[var(--accent)] font-medium">
            {indexingType === "docs"
              ? "Re-indexing documents (text chunks + trees)..."
              : indexingType === "images"
              ? "Re-indexing images..."
              : "Indexing in progress..."}
          </span>
        </div>
      )}

      {/* File list */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <h2 className="text-base font-semibold text-[var(--text-primary)] mb-4">
          My Documents
        </h2>

        {filtered.length === 0 ? (
          <div className="text-center py-16">
            <FileText
              size={40}
              className="mx-auto text-[var(--text-muted)] mb-3"
            />
            <p className="text-sm text-[var(--text-muted)]">
              {files.length === 0
                ? "No documents yet. Upload to get started."
                : "No matching documents."}
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map((f) => (
              <div
                key={f.name}
                className="flex items-center gap-4 px-4 py-3 bg-[var(--bg-primary)] border border-[var(--border-light)] rounded-xl hover:border-[var(--border)] hover:shadow-sm transition-all group"
              >
                <div className="w-10 h-10 rounded-xl bg-[var(--bg-secondary)] flex items-center justify-center shrink-0">
                  <FileText size={18} className="text-[var(--text-muted)]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-[var(--text-primary)] truncate">
                    {f.name}
                  </p>
                  <p className="text-xs text-[var(--text-muted)]">
                    {formatFileSize(f.size)} ·{" "}
                    {f.type.replace(".", "").toUpperCase()}
                  </p>
                </div>
                <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => onChatWithDoc(f.name)}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
                  >
                    Chat
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {indexInfo && (
          <div className="mt-6 flex items-center gap-4 px-4 py-3 bg-[var(--bg-secondary)] rounded-xl text-xs text-[var(--text-muted)]">
            <span>{indexInfo.chunk_count} chunks indexed</span>
            <span className="text-[var(--border)]">·</span>
            <span>{indexInfo.tree_count} tree indexes</span>
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
