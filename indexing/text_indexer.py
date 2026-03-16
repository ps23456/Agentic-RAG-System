"""
Text indexer: load documents, chunk, build BM25 + Chroma vector index (claim_chunks).
Does not touch image collection. Keeps existing Hybrid RAG indexing behavior.
"""
import logging
from typing import Optional

from document_loader import load_and_chunk_folder, Chunk
from search_index import SearchIndex
from config import DATA_FOLDER, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from retrieval.agentic_rag import normalize_patient_names_in_chunks

logger = logging.getLogger(__name__)


def load_existing_index() -> Optional[SearchIndex]:
    """
    Load chunks from existing ChromaDB collection without re-embedding.
    Only rebuilds the in-memory BM25 index. Returns None if no data exists.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        return None

    import os
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        return None

    count = collection.count()
    if count == 0:
        return None

    logger.info(f"Loading {count} existing chunks from ChromaDB...")
    all_data = collection.get(include=["metadatas", "documents"])

    chunks = []
    for i, chunk_id in enumerate(all_data["ids"]):
        meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
        text = (all_data["documents"][i] if all_data["documents"] and all_data["documents"][i] else "")
        if not text:
            text = meta.get("text", "")
        c = Chunk(
            chunk_id=chunk_id,
            text=text,
            file_name=meta.get("file_name", ""),
            page_number=meta.get("page_number", 0),
            document_type=meta.get("document_type", ""),
            start_char=0,
            end_char=len(text),
            patient_name=meta.get("patient_name", ""),
            claim_number=meta.get("claim_number", ""),
            policy_number=meta.get("policy_number", ""),
            group_number=meta.get("group_number", ""),
            doctor_name=meta.get("doctor_name", ""),
            doc_quality=meta.get("doc_quality", ""),
            embedding_text=meta.get("embedding_text", ""),
        )
        chunks.append(c)

    index = SearchIndex(chunks)
    index.build_bm25()
    index.load_chroma_collection()
    logger.info(f"Loaded {len(chunks)} chunks from ChromaDB (BM25 rebuilt in memory).")
    return index


def build_text_index(
    data_folder: Optional[str] = None,
    enable_vision: bool = False,
    vision_provider: str = "",
    vision_api_key: str = ""
) -> SearchIndex:
    """
    Incremental build: load documents, chunk (skipping unchanged), and upsert to index.
    """
    data_folder = data_folder or DATA_FOLDER
    
    # 1. Load existing chunks if available
    existing_index = load_existing_index()
    existing_chunks = existing_index.chunks if existing_index else None
    
    # 2. Extract and chunk (incremental)
    chunks = load_and_chunk_folder(
        data_folder, 
        existing_chunks=existing_chunks,
        enable_vision=enable_vision,
        vision_provider=vision_provider,
        vision_api_key=vision_api_key
    )
    # 2b. Normalize patient names at index time (OCR variants -> canonical, no duplicates)
    normalize_patient_names_in_chunks(chunks)

    # 3. Build/Update index
    index = SearchIndex(chunks)
    index.build_bm25()
    index.build_vector_index()
    return index
