import { useState, useCallback } from "react";
import type { ChatMessage, Conversation, Source } from "../lib/types";
import { sendChatStream, evaluateChat } from "../lib/api";
import { generateId } from "../lib/utils";

const STORAGE_KEY = "ics_conversations";

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

      const assistantId = generateId();
      const t0 = Date.now();
      let firstTokenAt: number | null = null;
      let accumulated = "";
      let lastFlush = 0;
      let finalResults: ChatMessage["results"] = [];
      let finalSources: Source[] = [];
      let finalIntent = "";
      let finalReasoning = "";

      // Create the assistant placeholder so the UI shows the shimmer immediately.
      const placeholder: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        query,
        ragasRequested: Boolean(evaluateRag),
      };
      convos = convos.map((c) =>
        c.id === convId ? { ...c, messages: [...c.messages, placeholder] } : c
      );
      persist(convos);

      const patchAssistant = (patch: Partial<ChatMessage>) => {
        setConversations((prev) => {
          const next = prev.map((c) =>
            c.id === convId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantId ? { ...m, ...patch } : m
                  ),
                }
              : c
          );
          saveConversations(next);
          return next;
        });
      };

      try {
        await sendChatStream(
          query,
          {
            onMeta: (meta) => {
              finalResults = meta.results || [];
              finalSources = meta.sources || [];
              finalIntent = meta.intent || "";
              finalReasoning = meta.reasoning || "";
              patchAssistant({
                sources: finalSources,
                results: finalResults,
                intent: finalIntent,
                reasoning: finalReasoning,
              });
              if (onSource && finalSources.length > 0) onSource(finalSources);
            },
            onToken: (tok) => {
              if (firstTokenAt === null) firstTokenAt = Date.now();
              accumulated += tok;
              // Throttle re-renders to ~10/s while streaming.
              const now = Date.now();
              if (now - lastFlush > 80) {
                lastFlush = now;
                patchAssistant({ content: accumulated });
              }
            },
            onDone: (done) => {
              if (done.summary) accumulated = done.summary;
              finalResults = done.results || finalResults;
              finalSources = done.sources || finalSources;
              finalIntent = done.intent || finalIntent;
              finalReasoning = done.reasoning || finalReasoning;
            },
            onError: (err) => {
              accumulated = accumulated || `Error: ${err}`;
            },
          },
          {
            patientFilter,
            webSearch,
            fileFilter: effectiveFile,
          }
        );
      } catch (e: unknown) {
        accumulated = accumulated || `Error: ${e instanceof Error ? e.message : "Unknown error"}`;
      }

      // Final commit with "Thought for" based on first-token latency so the
      // number matches what the user actually perceived.
      const firstTokenLatency = firstTokenAt
        ? Math.round((firstTokenAt - t0) / 1000)
        : Math.round((Date.now() - t0) / 1000);

      let autoSources = finalSources;
      if ((!autoSources || autoSources.length === 0) && finalResults?.length) {
        autoSources = [];
        const seen = new Set<string>();
        for (const r of finalResults) {
          const key = `${r.file_name}:${r.page}`;
          if (r.file_name && !seen.has(key)) {
            autoSources.push({ file_name: r.file_name, page: r.page, title: r.file_name });
            seen.add(key);
          }
          if (autoSources.length >= 3) break;
        }
      }

      patchAssistant({
        content: accumulated || "No results found for your query.",
        sources: autoSources,
        results: finalResults,
        intent: finalIntent,
        reasoning: finalReasoning,
        thinkingTime: firstTokenLatency,
      });

      if (onSource && autoSources && autoSources.length > 0) {
        onSource(autoSources);
      }

      setLoading(false);

      // Kick off RAGAs evaluation in the background — never block the user.
      if (evaluateRag && accumulated && finalResults?.length) {
        evaluateChat(query, accumulated, finalResults)
          .then((ev) => {
            patchAssistant({
              evaluation: ev.evaluation ?? undefined,
              evaluation_error: ev.evaluation_error ?? undefined,
              evaluation_notes: ev.evaluation_notes ?? undefined,
            });
          })
          .catch((err) => {
            patchAssistant({
              evaluation_error: err instanceof Error ? err.message : String(err),
            });
          });
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
