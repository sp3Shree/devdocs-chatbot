import os
from functools import lru_cache
from app.rag.retriever import Retriever

USE_SEPARATE_TEXTS = os.getenv("USE_SEPARATE_TEXTS", "true").lower() in {"1","true","yes"}

@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """
    Create and cache a single Retriever instance so we don't reload
    SentenceTransformer & FAISS for every request.
    """
    return Retriever(k=3, use_separate_texts=USE_SEPARATE_TEXTS)
