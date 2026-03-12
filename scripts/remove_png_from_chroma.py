#!/usr/bin/env python3
"""
Remove PNG/image chunks from Chroma DB.
Run: .venv_bge/bin/python scripts/remove_png_from_chroma.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

def main():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        coll = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"Collection not found: {e}")
        return
    
    # Get all chunks with metadata
    print("Loading all chunks from Chroma...")
    all_data = coll.get(include=['metadatas'])
    total = len(all_data['ids'])
    print(f"Total chunks in Chroma: {total}")
    
    # Find PNG/image chunks
    png_ids = []
    png_files = set()
    for id_, meta in zip(all_data['ids'], all_data['metadatas'] or []):
        file_name = (meta or {}).get('file_name', '')
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')):
            png_ids.append(id_)
            png_files.add(file_name)
    
    print(f"\nPNG/image chunks found: {len(png_ids)}")
    print(f"Unique image files: {sorted(png_files)}")
    
    if not png_ids:
        print("No PNG/image chunks to remove.")
        return
    
    # Delete chunks (non-interactive - uncomment confirmation if needed)
    # print(f"\n⚠️  About to delete {len(png_ids)} chunks from Chroma.")
    # response = input("Continue? (yes/no): ").strip().lower()
    # if response != 'yes':
    #     print("Cancelled.")
    #     return
    
    # Delete chunks
    print(f"\nDeleting {len(png_ids)} chunks...")
    coll.delete(ids=png_ids)
    
    # Verify
    remaining = coll.count()
    print(f"✅ Deleted. Remaining chunks in Chroma: {remaining}")
    print(f"Removed {len(png_ids)} chunks from {len(png_files)} image file(s).")

if __name__ == "__main__":
    main()
