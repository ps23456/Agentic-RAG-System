import { X, FileText } from "lucide-react";
import type { ViewerState } from "../../lib/types";

interface Props {
  tabs: ViewerState[];
  activeFileName: string | null;
  onSelect: (fileName: string) => void;
  onClose: (fileName: string) => void;
}

export function DocumentTabs({ tabs, activeFileName, onSelect, onClose }: Props) {
  if (tabs.length === 0) return null;

  return (
    <div className="flex items-center gap-0.5 px-2 py-1 border-b border-[var(--border)] bg-[var(--bg-secondary)] overflow-x-auto">
      {tabs.map((tab) => {
        const isActive = tab.fileName === activeFileName;
        const shortName = tab.fileName.length > 22 ? tab.fileName.slice(0, 20) + "..." : tab.fileName;
        return (
          <div
            key={tab.fileName}
            className={`group flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs cursor-pointer transition-all max-w-[200px] ${
              isActive
                ? "bg-[var(--bg-primary)] text-[var(--text-primary)] shadow-sm border border-[var(--border)]"
                : "text-[var(--text-muted)] hover:bg-[var(--bg-hover)]"
            }`}
            onClick={() => onSelect(tab.fileName)}
          >
            <FileText size={12} className="flex-shrink-0" />
            <span className="truncate">{shortName}</span>
            <button
              onClick={(e) => { e.stopPropagation(); onClose(tab.fileName); }}
              className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-[var(--bg-hover)] transition-all flex-shrink-0"
            >
              <X size={10} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
