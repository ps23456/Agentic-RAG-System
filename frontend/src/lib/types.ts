export interface Source {
  file_name: string;
  page: number | string | null;
  title: string;
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
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
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
