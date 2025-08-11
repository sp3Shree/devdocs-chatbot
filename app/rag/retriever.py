import pickle
import faiss
import argparse
from sentence_transformers import SentenceTransformer
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", k=3, use_separate_texts=True, repo_name="scikit-learn"):
        index_file = Path("data/vector_store") / repo_name / "faiss.index"
        metadata_file = Path("data/vector_store") / repo_name / "metadata.pkl"
        texts_file = Path("data/vector_store") / repo_name / "texts.pkl"
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(index_file))
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
        self.texts = None
        if use_separate_texts:
            with open(texts_file, "rb") as f:
                self.texts = pickle.load(f)
        self.k = k

    def search(self, query):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, self.k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                result = dict(self.metadata[idx]) # Make a shallow copy of the metadata
                if self.texts is not None and "text" not in result:
                    result["text"] = self.texts[idx] # Get the text aligned by the index
                result["distance"] = float(dist)
                results.append(result)
        return results

def main():
    parser = argparse.ArgumentParser(description="Semantic retrieval over FAISS index")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--k", type=int, default=3, help="Top-k results")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer Embedding model")
    parser.add_argument("--use_separate_texts", action="store_true", help="Load text from texts.pkl instead of metadata")
    parser.add_argument("--show-text", action="store_true", help="Print chunk text")
    parser.add_argument("--repo_name", required=True, help="Name of the repo to retrieve from")
    args = parser.parse_args()

    retriever = Retriever(model_name=args.model, k=args.k, use_separate_texts=args.use_separate_texts, repo_name=args.repo_name)
    results = retriever.search(args.query)

    if not results:
        print("No results")
        return

    print(f"\n Top {len(results)} results for query: '{args.query}'\n")
    for i, r in enumerate(results,1):
        path = r.get("file_path", "(unknown)")
        dist = r.get("distance", 0.0)
        print(f"[{i}] {path} | distance={dist:.4f}")
        if args.show_text:
            text = r.get("text", "")
            preview = text[:200] + ("..." if len(text) > 200 else "")
            print(f"    snippet: {preview}...")
    print("")

if __name__ == "__main__":
    main()