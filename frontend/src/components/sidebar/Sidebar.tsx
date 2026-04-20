import { useState, useEffect } from "react";
import {
  PenSquare,
  FolderOpen,
  Trash2,
  Settings,
  Stethoscope,
  ChevronRight,
  PanelLeft,
  Sun,
  Moon,
} from "lucide-react";
import type { Conversation, IndexInfo } from "../../lib/types";
import { getIndexInfo } from "../../lib/api";

interface Props {
  conversations: Conversation[];
  activeId: string;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
  onMedicalClick: () => void;
  onDocumentsClick: () => void;
  onToggleSidebar: () => void;
  showMedical: boolean;
  showDocuments: boolean;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export function Sidebar({
  conversations,
  activeId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onMedicalClick,
  onDocumentsClick,
  onToggleSidebar,
  showMedical,
  showDocuments,
  theme,
  onToggleTheme,
}: Props) {
  const [indexInfo, setIndexInfo] = useState<IndexInfo | null>(null);
  const [showChats, setShowChats] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    const refresh = () => getIndexInfo().then(setIndexInfo).catch(() => {});
    refresh();
    const iv = setInterval(refresh, 8000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="flex flex-col h-full bg-[var(--bg-sidebar)]">
      {/* Brand header */}
      <div className="flex items-center justify-between px-4 pt-3.5 pb-3">
        <div className="flex items-center gap-2.5">
          <img src="/logo.png" alt="ISR" className="w-8 h-8 object-contain shrink-0" />
          <span className="text-[18px] font-semibold tracking-tight text-[var(--text-primary)]">ISR</span>
        </div>
        <button
          onClick={onToggleSidebar}
          className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-chat-hover)] transition-all"
        >
          <PanelLeft size={18} strokeWidth={1.7} />
        </button>
      </div>

      {/* Nav items */}
      <div className="px-3 pb-0.5">
        {/* New Chat */}
        <button
          onClick={onNewChat}
          className="sidebar-item"
        >
          <PenSquare size={18} strokeWidth={1.7} />
          <span>New chat</span>
        </button>

        {/* Documents */}
        <button
          onClick={onDocumentsClick}
          className={`sidebar-item ${showDocuments ? "sidebar-item-active" : ""}`}
        >
          <FolderOpen size={18} strokeWidth={1.7} />
          <span>Documents</span>
        </button>

        {/* Medical Analysis */}
        <button
          onClick={onMedicalClick}
          className={`sidebar-item ${showMedical ? "sidebar-item-active" : ""}`}
        >
          <Stethoscope size={18} strokeWidth={1.7} />
          <span>Medical Analysis</span>
        </button>
      </div>

      {/* Your Chats section */}
      <div className="mt-5 flex-1 overflow-hidden flex flex-col min-h-0">
        <button
          onClick={() => setShowChats(!showChats)}
          className="flex items-center gap-1.5 px-5 mb-2 text-[14px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          Chats
          <ChevronRight
            size={12}
            className={`transition-transform duration-200 ${showChats ? "rotate-90" : ""}`}
          />
        </button>

        {showChats && (
          <div className="flex-1 overflow-y-auto px-2.5">
            {conversations.length === 0 ? (
              <p className="text-[13px] text-[var(--text-muted)] px-3 py-6 text-center">
                No chats yet
              </p>
            ) : (
              conversations.map((c) => (
                <div
                  key={c.id}
                  className={`group flex items-center px-3 py-2.5 rounded-xl cursor-pointer transition-all mb-1 ${
                    c.id === activeId && !showMedical && !showDocuments
                      ? "bg-[var(--bg-chat-hover)]"
                      : "hover:bg-[var(--bg-chat-hover)]"
                  }`}
                  onClick={() => onSelectChat(c.id)}
                >
                  <span className="flex-1 text-[13.5px] text-[var(--text-primary)] truncate leading-snug">{c.title}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDeleteChat(c.id); }}
                    className="opacity-0 group-hover:opacity-100 ml-2 p-1 rounded-md text-[var(--text-muted)] hover:text-red-500 hover:bg-[var(--bg-hover)] transition-all"
                    title="Delete"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Bottom settings */}
      <div className="border-t border-[var(--border)] px-3 py-2">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="sidebar-item"
        >
          <Settings size={18} strokeWidth={1.7} />
          <span>Settings</span>
        </button>

        {showSettings && (
          <div className="px-1 pb-2 pt-1 space-y-1.5 anim-fade-up">
            {/* Theme toggle */}
            <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-[var(--bg-primary)] border border-[var(--border)]">
              <span className="text-[13px] text-[var(--text-secondary)]">Theme</span>
              <button
                onClick={onToggleTheme}
                className="relative flex items-center w-[52px] h-[26px] rounded-full transition-colors duration-300"
                style={{ background: theme === "dark" ? "#3b82f6" : "#d1d5db" }}
              >
                <span
                  className="absolute flex items-center justify-center w-[22px] h-[22px] rounded-full bg-white shadow-sm transition-transform duration-300"
                  style={{ transform: theme === "dark" ? "translateX(28px)" : "translateX(2px)" }}
                >
                  {theme === "dark" ? <Moon size={12} className="text-blue-500" /> : <Sun size={12} className="text-amber-500" />}
                </span>
              </button>
            </div>

            {indexInfo && (
              <p className="text-[11px] text-[var(--text-muted)] text-center pt-1">
                {indexInfo.chunk_count} chunks · {indexInfo.tree_count} trees · {indexInfo.image_count} imgs
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
