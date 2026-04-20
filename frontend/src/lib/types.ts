export interface Source {
  file_name: string;
  page: number | string | null;
  title: string;
  url?: string;  // For web sources: open in new tab
  searchContext?: string;  // Snippet/phrase to scroll to (for .md files)
}

export interface ResultItem {
  type: "text" | "image" | "web";
  file_name: string;
  page: number | string | null;
  score: number;
  snippet: string;
  patient_name: string;
  section_title: string;
  is_pdf_page?: boolean;
  path?: string;
  url?: string;
}

export interface ChatResponse {
  summary: string;
  sources: Source[];
  results: ResultItem[];
  intent: string;
  reasoning: string;
  /** Present when evaluate_rag was sent: RAGAs metric name → score (0–1) */
  evaluation?: Record<string, number> | null;
  /** When evaluate_rag was true but scores missing */
  evaluation_error?: string | null;
  /** e.g. metrics skipped because Ragas returned NaN */
  evaluation_notes?: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  results?: ResultItem[];
  intent?: string;
  reasoning?: string;
  thinkingTime?: number;
  timestamp: number;
  query?: string;  // User query (for assistant msgs) - used when opening .md source
  evaluation?: Record<string, number> | null;
  evaluation_error?: string | null;
  evaluation_notes?: string | null;
  /** True if this turn was sent with RAGAs toggle on (for UI even when scores fail) */
  ragasRequested?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  /** When set, every message in this chat restricts retrieval to this file (e.g. after "Chat" from Documents). */
  scopedFile?: string;
}

export interface UploadedFile {
  name: string;
  size: number;
  type: string;
}

export interface IndexInfo {
  chunk_count: number;
  tree_count: number;
  image_count: number;
  patients: string[];
  status: string;
  indexing: boolean;
  progress?: number;
  stage?: string;
}

export interface ExtractedFieldSchemaItem {
  type: string;
  method: string;
  description: string;
}

export interface FieldsExtractResponse {
  file_name: string;
  mode: string;
  field_names: string[];
  text_preview: string;
  schema: Record<string, ExtractedFieldSchemaItem>;
}

export interface DocumentPage {
  image: string;
  page: number;
  total_pages: number;
}

export interface ViewerState {
  fileName: string;
  page: number;
  totalPages?: number;
  searchContext?: string;
}
