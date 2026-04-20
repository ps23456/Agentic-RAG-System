import { useState, useCallback } from "react";
import type { ChatMessage, Conversation, Source } from "../lib/types";
import { sendChat } from "../lib/api";
import { generateId } from "../lib/utils";

const STORAGE_KEY = "ics_conversations";

/** Match a document filename in natural language (aligned with backend agentic_rag _FILENAME_PATTERN). */
function inferScopedFileFromQuery(query: string): string | undefined {
  const m = query.match(
    /\b([A-Za-z0-9_\-]+(?:\s*\(\d+\))?\.(?:png|jpg|jpeg|gif|bmp|tiff|tif|pdf|webp|md|txt|json))\b/i
  );
  return m ? m[1] : undefined;
}

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations(convos: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convos.slice(0, 50)));
}

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string>(() => {
    const c = loadConversations();
    return c.length > 0 ? c[0].id : "";
  });
  const [loading, setLoading] = useState(false);

  const activeConversation = conversations.find((c) => c.id === activeId) || null;

  const persist = useCallback((updated: Conversation[]) => {
    setConversations(updated);
    saveConversations(updated);
  }, []);

  const newChat = useCallback(() => {
    const conv: Conversation = {
      id: generateId(),
      title: "New Chat",
      messages: [],
      createdAt: Date.now(),
      scopedFile: undefined,
    };
    const updated = [conv, ...conversations];
    persist(updated);
    setActiveId(conv.id);
    return conv.id;
  }, [conversations, persist]);

  const send = useCallback(
    async (query: string, onSource?: (sources: Source[]) => void, webSearch?: boolean, patientFilter?: string, fileFilter?: string, evaluateRag?: boolean) => {
      let convId = activeId;
      let convos = [...conversations];

      if (!convId || !convos.find((c) => c.id === convId)) {
        const conv: Conversation = {
          id: generateId(),
          title: query.slice(0, 40),
          messages: [],
          createdAt: Date.now(),
          scopedFile: fileFilter || undefined,
        };
        convos = [conv, ...convos];
        convId = conv.id;
        setActiveId(convId);
      }

      const activeConv = convos.find((c) => c.id === convId);
      const effectiveFile = (fileFilter && fileFilter.trim()) || activeConv?.scopedFile || undefined;

      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: query,
        timestamp: Date.now(),
      };

      convos = convos.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [...c.messages, userMsg],
              title: c.messages.length === 0 ? query.slice(0, 40) : c.title,
              scopedFile: (effectiveFile || c.scopedFile) ?? undefined,
            }
          : c
      );
      persist(convos);
      setLoading(true);

      try {
        const t0 = Date.now();
        const resp = await sendChat(query, patientFilter, webSearch, effectiveFile, evaluateRag);
        const elapsed = Math.round((Date.now() - t0) / 1000);

        const assistantMsg: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: resp.summary || "No results found for your query.",
          sources: resp.sources,
          results: resp.results,
          intent: resp.intent,
          reasoning: resp.reasoning,
          thinkingTime: elapsed,
          timestamp: Date.now(),
          query,
          evaluation: resp.evaluation ?? undefined,
          evaluation_error: resp.evaluation_error ?? undefined,
          evaluation_notes: resp.evaluation_notes ?? undefined,
          ragasRequested: Boolean(evaluateRag),
        };

        convos = convos.map((c) =>
          c.id === convId ? { ...c, messages: [...c.messages, assistantMsg] } : c
        );
        persist(convos);

        let autoSources = resp.sources || [];
        if (autoSources.length === 0 && resp.results?.length) {
          const seen = new Set<string>();
          for (const r of resp.results) {
            const key = `${r.file_name}:${r.page}`;
            if (r.file_name && !seen.has(key)) {
              autoSources.push({ file_name: r.file_name, page: r.page, title: r.file_name });
              seen.add(key);
            }
            if (autoSources.length >= 3) break;
          }
        }
        if (onSource && autoSources.length > 0) {
          onSource(autoSources);
        }
      } catch (e: unknown) {
        const errMsg: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: `Error: ${e instanceof Error ? e.message : "Unknown error"}`,
          timestamp: Date.now(),
        };
        convos = convos.map((c) =>
          c.id === convId ? { ...c, messages: [...c.messages, errMsg] } : c
        );
        persist(convos);
      } finally {
        setLoading(false);
      }
    },
    [activeId, conversations, persist]
  );

  const selectChat = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  const deleteChat = useCallback(
    (id: string) => {
      const updated = conversations.filter((c) => c.id !== id);
      persist(updated);
      if (activeId === id) {
        setActiveId(updated.length > 0 ? updated[0].id : "");
      }
    },
    [conversations, activeId, persist]
  );

  return {
    conversations,
    activeConversation,
    activeId,
    loading,
    newChat,
    send,
    selectChat,
    deleteChat,
  };
}
