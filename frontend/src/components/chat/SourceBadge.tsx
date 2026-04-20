import { FileText, Globe } from "lucide-react";
import type { Source } from "../../lib/types";

interface Props {
  source: Source;
  onClick: (source: Source) => void;
  index?: number;
  active?: boolean;
}

export function SourceBadge({ source, onClick, index, active }: Props) {
  const name =
    source.file_name.length > 28
      ? source.file_name.slice(0, 26) + "..."
      : source.file_name;

  return (
    <button
      onClick={() => onClick(source)}
      className={`group flex items-start gap-2.5 px-3.5 py-2.5 rounded-xl border text-left transition-all min-w-[160px] max-w-[220px] ${
        active
          ? "bg-[var(--accent-bg)] border-[var(--accent)] shadow-sm"
          : "bg-[var(--bg-primary)] border-[var(--border)] hover:border-[var(--accent)] hover:shadow-sm"
      }`}
    >
      <div className={`w-6 h-6 rounded-lg flex items-center justify-center shrink-0 text-[11px] font-bold transition-colors ${
        active
          ? "bg-[var(--accent)] text-white"
          : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] group-hover:bg-[var(--accent-light)] group-hover:text-[var(--accent)]"
      }`}>
        {source.url ? <Globe size={12} /> : (index ?? <FileText size={12} />)}
      </div>
      <div className="min-w-0 flex-1">
        <p className={`text-[12px] font-medium truncate leading-tight ${
          active ? "text-[var(--accent)]" : "text-[var(--text-primary)]"
        }`}>
          {name}
        </p>
        {source.page != null && (
          <p className="text-[11px] text-[var(--text-muted)] mt-0.5">Page {source.page}</p>
        )}
      </div>
    </button>
  );
}
