import { useState, useRef, useEffect } from "react";
import { ArrowUp, Plus, Globe } from "lucide-react";

interface Props {
  onSend: (query: string, webSearch: boolean) => void;
  disabled?: boolean;
  onUploadClick?: () => void;
}

export function ChatInput({ onSend, disabled, onUploadClick }: Props) {
  const [value, setValue] = useState("");
  const [webSearch, setWebSearch] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { ref.current?.focus(); }, [disabled]);

  useEffect(() => {
    const el = ref.current;
    if (el) { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 150) + "px"; }
  }, [value]);

  const submit = () => {
    const q = value.trim();
    if (!q || disabled) return;
    onSend(q, webSearch);
    setValue("");
  };

  return (
    <div className="px-5 pb-5 pt-2">
      <div className="max-w-3xl mx-auto">
        <div className="bg-[var(--bg-input)] rounded-none border border-[var(--bg-input-border)] focus-within:border-[var(--bg-input-focus)] transition-all overflow-hidden">
          <div className="px-6 pt-5 pb-1.5">
            <textarea
              ref={ref}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
              placeholder="Ask a question..."
              disabled={disabled}
              rows={1}
              className="w-full bg-transparent resize-none outline-none text-[15px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] leading-7 py-0.5"
              style={{ overflowWrap: "break-word" }}
            />
          </div>
          <div className="flex items-center justify-between px-3 pb-3">
            <div className="flex items-center gap-1">
              <button
                onClick={onUploadClick}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[13px] text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)] transition-all"
              >
                <Plus size={16} strokeWidth={1.8} />
                Add documents
              </button>
              <button
                onClick={() => setWebSearch(!webSearch)}
                title={webSearch ? "Web search enabled" : "Enable web search"}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-[13px] transition-all ${
                  webSearch
                    ? "bg-[var(--accent-light)] text-[var(--accent)] font-medium"
                    : "text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-secondary)]"
                }`}
              >
                <Globe size={15} strokeWidth={1.8} />
                {webSearch && <span>Web</span>}
              </button>
            </div>
            <button
              onClick={submit}
              disabled={disabled || !value.trim()}
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
                value.trim() && !disabled
                  ? "bg-[var(--send-active)]"
                  : "bg-[var(--send-inactive)]"
              }`}
            >
              <ArrowUp size={16} className="text-white" strokeWidth={2.5} />
            </button>
          </div>
        </div>
        <p className="text-center text-[11px] text-[var(--text-muted)] mt-2.5">
          ISR can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
}
