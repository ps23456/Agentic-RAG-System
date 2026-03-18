import { useState, useCallback } from "react";
import { Upload, Check, Loader2 } from "lucide-react";
import { uploadFiles } from "../../lib/api";

interface Props {
  onUploaded?: () => void;
}

export function UploadArea({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const arr = Array.from(files);
      if (arr.length === 0) return;
      setUploading(true);
      setResult(null);
      try {
        const res = await uploadFiles(arr);
        setResult(`Uploaded ${res.count} file(s)`);
        onUploaded?.();
      } catch (e) {
        setResult(`Error: ${e instanceof Error ? e.message : "Upload failed"}`);
      } finally {
        setUploading(false);
        setTimeout(() => setResult(null), 3000);
      }
    },
    [onUploaded]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-4 text-center transition-colors cursor-pointer ${
        dragging
          ? "border-[var(--accent)] bg-[var(--accent)]/5"
          : "border-[var(--border)] hover:border-[var(--text-muted)]"
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.accept = ".pdf,.png,.jpg,.jpeg,.md,.txt,.json,.tiff,.bmp";
        input.onchange = () => input.files && handleFiles(input.files);
        input.click();
      }}
    >
      {uploading ? (
        <Loader2 size={20} className="mx-auto text-[var(--accent)] animate-spin" />
      ) : result ? (
        <div className="flex items-center justify-center gap-1.5 text-xs text-green-400">
          <Check size={14} /> {result}
        </div>
      ) : (
        <>
          <Upload size={18} className="mx-auto text-[var(--text-muted)] mb-1.5" />
          <p className="text-xs text-[var(--text-muted)]">
            Drop files or click to upload
          </p>
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">
            PDF, Images, Markdown, Text, JSON
          </p>
        </>
      )}
    </div>
  );
}
