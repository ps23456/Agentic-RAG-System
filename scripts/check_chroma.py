#!/usr/bin/env python3
"""
Verify that chunks are stored in Chroma DB.
Run from project root after indexing with VECTOR_BACKEND=chroma:

  python scripts/check_chroma.py

Or with your venv:
  .venv_bge/bin/python scripts/check_chroma.py
"""
import sys
import os

# Add project root so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, VECTOR_BACKEND


def main():
    import sys
    print(f"Python: {sys.version} (executable: {sys.executable})")
    print(f"VECTOR_BACKEND (from config): {VECTOR_BACKEND}")
    print(f"CHROMA_PERSIST_DIR: {CHROMA_PERSIST_DIR}")
    print()

    if VECTOR_BACKEND != "chroma":
        print("Not using Chroma. Set VECTOR_BACKEND=chroma and re-index in the app, then run this again.")
        return

    if not os.path.isdir(CHROMA_PERSIST_DIR):
        print(f"Chroma directory does not exist yet: {CHROMA_PERSIST_DIR}")
        print("Start the app, set VECTOR_BACKEND=chroma, then click 'Index / Re-index documents'.")
        return

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("chromadb not installed. Run: pip install chromadb")
        return

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"Collection '{CHROMA_COLLECTION_NAME}' not found. Have you clicked 'Index / Re-index' with Chroma enabled?")
        print(f"Error: {e}")
        return

    count = collection.count()
    print(f"Chroma collection: {CHROMA_COLLECTION_NAME}")
    print(f"Total chunks stored: {count}")
    if count == 0:
        print("No chunks in Chroma. Index documents in the app with VECTOR_BACKEND=chroma.")
        return

    # Sample: get first 10 with metadata
    sample = collection.get(limit=10, include=["metadatas"])
    print("\nSample of stored chunks (first 10):")
    print("-" * 60)
    for i, (idx, meta) in enumerate(zip(sample["ids"], sample["metadatas"] or []), 1):
        file_name = (meta or {}).get("file_name", "?")
        page = (meta or {}).get("page_number", "?")
        doc_type = (meta or {}).get("document_type", "?")
        print(f"  {i}. id={idx[:50]}{'...' if len(idx) > 50 else ''}")
        print(f"     file_name={file_name}, page={page}, document_type={doc_type}")
    print("-" * 60)
    print("Chroma is storing: chunk_id, embedding vector, and metadata (file_name, page_number, document_type).")
    print("Search in the app uses these vectors; chunk text is kept in memory when the app is running.")


if __name__ == "__main__":
    main()
