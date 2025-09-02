from __future__ import annotations
import json, pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

def load_chunks(jsonl_path: Path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks

def build_faiss_for_repo(repo_name: str, base_dir: Path, keep_text_in_metadata: bool = True) -> Path:
    chunks_path = base_dir / "chunks" / repo_name / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found for repo '{repo_name}': {chunks_path}")

    vector_repo_dir = base_dir / "vector_store" / repo_name
    vector_repo_dir.mkdir(parents=True, exist_ok=True)
    index_path = vector_repo_dir / "faiss.index"
    metadata_path = vector_repo_dir / "metadata.pkl"
    texts_path = vector_repo_dir / "texts.pkl"

    # Load chunks
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    metadata = chunks if keep_text_in_metadata else [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    # Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)
    dim = embeddings.shape[1]

    # Build index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and metadata (+ texts if needed)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as f:
        pickle.dump(metadata, f)
    if not keep_text_in_metadata:
        with texts_path.open("wb") as f:
            pickle.dump(texts, f)

    return index_path