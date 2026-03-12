"""
Indexing for Multimodal Hybrid RAG: text index (BM25 + Chroma) and image index (CLIP, separate collection).
"""
from .text_indexer import build_text_index
from .image_indexer import build_image_index, get_image_index_count

__all__ = ["build_text_index", "build_image_index", "get_image_index_count"]
