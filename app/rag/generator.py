import os
import openai import OpenAI
from retriever import Retriever
from dotenv import load_dotenv
import argparse

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query, k=3):
    retriever = Retriever(k=k)
    contexts = retriever.search(query)
    context_texts = "\n\n".join([f"{c['file_path']}:\n{c['text']}" for c in contexts])

    system_prompt = "You're a helpful coding assistant. Answer the user's question using only the provided repo context."
    user_prompt = f"Question: {query}\n\nContext:\n{context_texts}\n\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_token=300,
    )
    return response.choice[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="The question to ask the model")
    parser.add_argument("--k", type=int, default=3, help="Number of context chunks to retrieve")
    args = parser.parse_args()

    answer = generate_answer(args.query, args.k)
    print(f"💡 Answer: {answer}")