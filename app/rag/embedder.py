import os
import json
import pickle
import faiss
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks

def save_metadata(metadata, metadata_file, texts, texts_file):
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    with open(texts_file, "wb") as f:
        pickle.dump(texts, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", required=True, help="Name of the repo (subfolder in data/chunks and data/vector_store)")
    args = parser.parse_args()

    chunk_path = Path("data/chunks") / args.repo_name / "chunks.jsonl"
    vector_store_dir = Path("data/vector_store") / args.repo_name
    index_file = vector_store_dir / "faiss.index"
    metadata_file = vector_store_dir / "metadata.pkl"
    texts_file = vector_store_dir / "texts.pkl"

    os.makedirs(vector_store_dir, exist_ok=True)

    print("🔄 Loading chunks...")
    chunks = load_chunks(chunk_path)
    texts = [chunk["text"] for chunk in chunks]
    metadata = [{k: v for k, v in chunk.items() if k != "text"} for chunk in chunks]

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("📦 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("💾 Saving FAISS index, metadata, and texts...")
    faiss.write_index(index, str(index_file))
    save_metadata(metadata, metadata_file, texts, texts_file)

    print(f"✅ Indexed {len(embeddings)} chunks")

if __name__ == "__main__":
    main()