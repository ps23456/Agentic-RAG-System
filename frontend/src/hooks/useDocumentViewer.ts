import { useState, useCallback } from "react";
import type { ViewerState, Source } from "../lib/types";

export function useDocumentViewer() {
  const [viewers, setViewers] = useState<ViewerState[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);

  const activeViewer = viewers[activeIdx] || null;

  const [searchContext, setSearchContext] = useState("");

  const openDocument = useCallback(
    (fileName: string, page: number = 1, search?: string) => {
      const existing = viewers.findIndex((v) => v.fileName === fileName);
      if (existing >= 0) {
        setViewers((prev) =>
          prev.map((v, i) => (i === existing ? { ...v, page, searchContext: search } : v))
        );
        setActiveIdx(existing);
      } else {
        setViewers((prev) => [...prev, { fileName, page, searchContext: search }]);
        setActiveIdx(viewers.length);
      }
      if (search) setSearchContext(search);
    },
    [viewers]
  );

  const openSource = useCallback(
    (source: Source, search?: string) => {
      const page = typeof source.page === "number" ? source.page : parseInt(String(source.page)) || 1;
      openDocument(source.file_name, page, search);
    },
    [openDocument]
  );

  const closeTab = useCallback(
    (idx: number) => {
      setViewers((prev) => prev.filter((_, i) => i !== idx));
      if (activeIdx >= idx && activeIdx > 0) {
        setActiveIdx((p) => p - 1);
      }
    },
    [activeIdx]
  );

  const setPage = useCallback(
    (page: number) => {
      setViewers((prev) =>
        prev.map((v, i) => (i === activeIdx ? { ...v, page } : v))
      );
    },
    [activeIdx]
  );

  const setTotalPages = useCallback(
    (total: number) => {
      setViewers((prev) =>
        prev.map((v, i) => (i === activeIdx ? { ...v, totalPages: total } : v))
      );
    },
    [activeIdx]
  );

  const clearAll = useCallback(() => {
    setViewers([]);
    setActiveIdx(0);
    setSearchContext("");
  }, []);

  return {
    viewers,
    activeViewer,
    activeIdx,
    searchContext,
    setActiveIdx,
    openDocument,
    openSource,
    closeTab,
    clearAll,
    setPage,
    setTotalPages,
  };
}
