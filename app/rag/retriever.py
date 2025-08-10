import os
import pickle
import faiss
import argparse
from sentence_transformers import SentenceTransformer
from pathlib import Path

class Retriever:
    def __init__(self, repo_name, model_name="all-MiniLM-L6-v2", k=3):
        index_file = Path("data/vector_store") / repo_name / "faiss.index"
        metadata_file = Path("data/vector_store") / repo_name / "metadata.pkl"
        texts_file = Path("data/vector_store") / repo_name / "texts.pkl"
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
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
                result["text"] = self.texts[idx] # Get the text aligned by the index
                result["distance"] = float(dist)
                results.append(result)
        return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", required=True, help="Name of the repo to retrieve from")
    args = parser.parse_args()

    retriever = Retriever(args.repo_name)
    results = retriever.search("How do I fit a RandomForest in this repo?")
    for r in results:
        print(r)