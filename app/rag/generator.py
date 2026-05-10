import os
import warnings
import argparse

import google.generativeai as genai
from openai import OpenAI

from app.rag.retriever import Retriever
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. Answer the user's question using ONLY "
    "the provided repo context. If the answer is not in the context, say you don't know."
)

def _build_prompt(query, contexts) -> str:
    parts = [f"System: {SYSTEM_PROMPT}", f"Question: {query}", "Context:"]
    for c in contexts:
        parts.append(f"\n---\nFile: {c.get('file_path', '(unknown')}\n{c.get('text', '')}\n")
    parts.append("\nInstructions: Provide a concise, directly-cited answer. If unclear, say so.")
    return "\n".join(parts)

def _is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3")

def _generate_openai(prompt, model_name, max_output_tokens, temperature) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_output_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

def _generate_gemini(prompt, model_name, max_output_tokens, temperature) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment or .env.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
    )
    response = model.generate_content(prompt)
    if not response or not getattr(response, "text", None):
        return "I couldn't generate an answer based on the provided context."
    return response.text

def generate_answer(query, k=3, model_name="gemini-2.5-flash", max_output_tokens=600, temperature=0.2) -> str:
    retriever = Retriever(k=k)
    contexts = retriever.search(query)
    if not contexts:
        return "I couldn't find relevant context in the repository to answer that."

    prompt = _build_prompt(query, contexts)

    if _is_openai_model(model_name):
        return _generate_openai(prompt, model_name, max_output_tokens, temperature)
    else:
        return _generate_gemini(prompt, model_name, max_output_tokens, temperature)

def answer_from_contexts(query, contexts, model_name="gemini-2.5-flash", max_output_tokens=600, temperature=0.2) -> str:
    prompt = _build_prompt(query, contexts)
    if _is_openai_model(model_name):
        return _generate_openai(prompt, model_name, max_output_tokens, temperature)
    else:
        return _generate_gemini(prompt, model_name, max_output_tokens, temperature)

def main():
    parser = argparse.ArgumentParser(description="Generate grounded answers using LLM + FAISS retrieval")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--k", type=int, default=3, help="Top-k contexts to retrieve")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), help="Model name (e.g. gemini-2.5-flash, gpt-4o-mini)")
    parser.add_argument("--max-output-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    answer = generate_answer(
        query=args.query,
        k=args.k,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )
    print("\n💡 Answer:\n")
    print(answer)

if __name__ == "__main__":
    main()