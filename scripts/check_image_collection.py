#!/usr/bin/env python3
"""Check if flower.png (and other files) are in the ChromaDB image collection."""
import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_PERSIST_DIR, CHROMA_IMAGE_COLLECTION_NAME

def main():
    sys.stdout.flush()
    print(f"CHROMA_PERSIST_DIR: {CHROMA_PERSIST_DIR}", flush=True)
    if not os.path.isdir(CHROMA_PERSIST_DIR):
        print("Chroma directory does not exist yet.")
        return

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("chromadb not installed")
        return

    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        coll = client.get_collection(name=CHROMA_IMAGE_COLLECTION_NAME)
    except Exception as e:
        print(f"Image collection not found: {e}")
        return

    count = coll.count()
    print(f"\nImage collection '{CHROMA_IMAGE_COLLECTION_NAME}': {count} items total\n")

    # Check for flower.png
    try:
        got = coll.get(where={"file_name": {"$eq": "flower.png"}}, include=["metadatas"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        if ids:
            print("✓ flower.png IS in the image collection")
            for i, uid in enumerate(ids):
                m = metas[i] if i < len(metas) else {}
                print(f"  ID: {uid}")
                print(f"  file_name: {m.get('file_name')}")
                print(f"  path: {m.get('path')}")
        else:
            print("✗ flower.png is NOT in the image collection")
    except Exception as e:
        print(f"Error checking flower.png: {e}")

    # List all unique file_names in the collection
    print("\n--- All files in image collection ---")
    try:
        all_data = coll.get(include=["metadatas"])
        files = {}
        for m in (all_data.get("metadatas") or []):
            fn = m.get("file_name", "?")
            files[fn] = files.get(fn, 0) + 1
        for fn, count in sorted(files.items()):
            print(f"  {fn}: {count} item(s)")
    except Exception as e:
        print(f"Error listing: {e}")

if __name__ == "__main__":
    main()
