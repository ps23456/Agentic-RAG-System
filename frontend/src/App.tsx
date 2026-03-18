import { useState, useRef } from "react";
import { Sidebar } from "./components/sidebar/Sidebar";
import { ChatPanel } from "./components/chat/ChatPanel";
import { DocumentViewer } from "./components/documents/DocumentViewer";
import { DocumentsPage } from "./components/documents/DocumentsPage";
import { MedicalPanel } from "./components/medical/MedicalPanel";
import { useChat } from "./hooks/useChat";
import { useDocumentViewer } from "./hooks/useDocumentViewer";
import { useTheme } from "./hooks/useTheme";
import type { Source } from "./lib/types";
import { PanelLeft } from "lucide-react";

type View = "chat" | "documents" | "medical";

export default function App() {
  const chat = useChat();
  const docViewer = useDocumentViewer();
  const { theme, toggle: toggleTheme } = useTheme();
  const [view, setView] = useState<View>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const lastQueryRef = useRef("");

  const handleSend = (query: string, webSearch?: boolean) => {
    setView("chat");
    lastQueryRef.current = query;
    chat.send(query, (sources: Source[]) => {
      if (sources.length > 0) {
        docViewer.openSource(sources[0], query);
      }
    }, webSearch);
  };

  const handleSourceClick = (source: Source) => {
    docViewer.openSource(source, lastQueryRef.current);
  };

  const handleChatWithDoc = (docName: string) => {
    setView("chat");
    handleSend(`Tell me about ${docName}`);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--bg-primary)]">
      <div className={`shrink-0 transition-all duration-200 ${sidebarOpen ? "w-[268px]" : "w-0 overflow-hidden"}`}>
        <Sidebar
          conversations={chat.conversations}
          activeId={chat.activeId}
          onNewChat={() => { chat.newChat(); docViewer.clearAll(); setView("chat"); }}
          onSelectChat={(id) => { chat.selectChat(id); setView("chat"); }}
          onDeleteChat={chat.deleteChat}
          onMedicalClick={() => setView(view === "medical" ? "chat" : "medical")}
          onDocumentsClick={() => setView(view === "documents" ? "chat" : "documents")}
          onToggleSidebar={() => setSidebarOpen(false)}
          showMedical={view === "medical"}
          showDocuments={view === "documents"}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      </div>

      {!sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          className="fixed top-3 left-3 z-20 flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] transition-all"
        >
          <PanelLeft size={18} strokeWidth={1.7} />
          <img src="/logo.png" alt="" className="w-6 h-6 object-contain" />
          <span className="text-[15px] font-semibold text-[var(--text-primary)]">ISR</span>
        </button>
      )}

      <div className="flex-1 flex min-w-0">
        {view === "documents" ? (
          <div className="flex-1">
            <DocumentsPage onBack={() => setView("chat")} onChatWithDoc={handleChatWithDoc} />
          </div>
        ) : view === "medical" ? (
          <div className="flex-1">
            <MedicalPanel onBack={() => setView("chat")} />
          </div>
        ) : (
          <>
            <div className="flex-1 min-w-0">
              <ChatPanel
                conversation={chat.activeConversation}
                loading={chat.loading}
                onSend={handleSend}
                onSourceClick={handleSourceClick}
              />
            </div>
            <div className="w-[42%] shrink-0">
              <DocumentViewer
                viewers={docViewer.viewers}
                activeIdx={docViewer.activeIdx}
                onTabClick={docViewer.setActiveIdx}
                onClose={docViewer.closeTab}
                onPageChange={docViewer.setPage}
                onTotalPages={docViewer.setTotalPages}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
