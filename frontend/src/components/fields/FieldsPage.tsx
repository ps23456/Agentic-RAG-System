import { useMemo, useState } from "react";
import { ArrowLeft, Download, Loader2 } from "lucide-react";
import { extractFields } from "../../lib/api";
import type { FieldsExtractResponse } from "../../lib/types";

interface Props {
  onBack: () => void;
  fileName: string;
}

export function FieldsPage({ onBack, fileName }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<FieldsExtractResponse | null>(null);

  const schemaText = useMemo(() => {
    if (!data) return "";
    return JSON.stringify(data.schema, null, 2);
  }, [data]);

  const run = async () => {
    setLoading(true);
    setError("");
    try {
      const resp = await extractFields(fileName);
      setData(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Extraction failed");
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = () => {
    if (!data) return;
    const blob = new Blob([schemaText], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName.replace(/\.[^.]+$/, "")}_fields.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)]">
      <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border)]">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-2 rounded-xl hover:bg-[var(--bg-tertiary)] transition-colors text-[var(--text-secondary)]"
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 className="text-xl font-semibold text-[var(--text-primary)] tracking-tight">Field Extractor</h1>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Extract user-fillable field names from <code>{fileName}</code>
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={run}
            disabled={loading}
            className="px-4 py-2 rounded-xl text-sm font-medium bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] disabled:opacity-60"
          >
            {loading ? "Extracting..." : "Extract Fields"}
          </button>
          <button
            onClick={downloadJson}
            disabled={!data}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border border-[var(--border)] hover:bg-[var(--bg-secondary)] disabled:opacity-50"
          >
            <Download size={14} />
            Download JSON
          </button>
        </div>
      </div>

      {error && (
        <div className="mx-6 mt-3 px-4 py-3 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200 text-sm">
          {error}
        </div>
      )}

      {!data && !loading && !error && (
        <div className="flex-1 flex items-center justify-center text-[var(--text-muted)] text-sm">
          Click <strong className="mx-1 text-[var(--text-secondary)]">Extract Fields</strong> to generate preview + JSON schema.
        </div>
      )}

      {loading && (
        <div className="flex-1 flex items-center justify-center gap-2 text-[var(--text-secondary)]">
          <Loader2 size={16} className="animate-spin" />
          Extracting form fields...
        </div>
      )}

      {data && !loading && (
        <div className="flex-1 min-h-0 grid grid-cols-2 gap-0 border-t border-[var(--border-light)]">
          <div className="min-h-0 border-r border-[var(--border-light)]">
            <div className="px-4 py-2 text-xs text-[var(--text-muted)] border-b border-[var(--border-light)]">
              Text Preview ({data.field_names.length} fields) · mode: {data.mode}
            </div>
            <pre className="h-full overflow-auto p-4 text-[12px] leading-6 whitespace-pre-wrap text-[var(--text-primary)]">
              {data.text_preview || "No fields detected."}
            </pre>
          </div>
          <div className="min-h-0">
            <div className="px-4 py-2 text-xs text-[var(--text-muted)] border-b border-[var(--border-light)]">
              JSON Schema
            </div>
            <pre className="h-full overflow-auto p-4 text-[12px] leading-6 text-[var(--text-primary)]">
              {schemaText}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

