import os
import google.generativeai as genai
# Local import
from app.rag.retriever import Retriever
from dotenv import load_dotenv
import argparse
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

# Choose the model: "gemini-1.5-flash" (faster, cheaper) or "gemini-1.5-pro" (more capable)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. Answer the user's question using ONLY the provided repo context. If the answer is not in the context, say you don't know."
)

def _build_prompt(query, contexts) -> str:
    """
    Stitch retrieved chunks into a single prompt block.
    Each context item is expected to have 'file_path' and 'text'.
    """
    parts = [f"System: {SYSTEM_PROMPT}", f"Question: {query}", "Context:"]
    for c in contexts:
            parts.append(f"\n---\nFile: {c.get('file_path','(unknown')}\n{c.get('text','')}\n")
    parts.append("\nInstructions: Provide a concise, directly-cited answer. If unclear, say so.")
    return "\n".join(parts)

def validate_key_or_die():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment or .env. Please set it to use the Gemini API.")
    genai.configure(api_key=api_key)

def generate_answer(query, k=3, model_name=MODEL_NAME, max_output_tokens=600, temperature=0.2) -> str:
    validate_key_or_die()

    # Retrieve top-k semantic matches from FAISS
    retriever = Retriever(k=k)
    contexts = retriever.search(query)
    if not contexts:
        return "I couldn't find relevant context in the repository to answer that."

    prompt = _build_prompt(query, contexts)

    # Create the model and generate a response
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": temperature
        }
    )
    response = model.generate_content(prompt)

    # Handle blocked/ empty responses gracefully
    if not response or not getattr(response, "text", None):
        return "I couldn't generate an answer based on the provided context."

    return response.text

def answer_from_contexts(query, contexts, model_name="gemini-1.5-flash", max_output_tokens=600, temperature=0.2) -> str:
    validate_key_or_die()
    prompt = _build_prompt(query, contexts)
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": temperature
        }
    )
    response = model.generate_content(prompt)
    if not response or not getattr(response, "text", None):
        return "I couldn't generate an answer based on the provided context."
    return response.text

def main():
    parser = argparse.ArgumentParser(description="Generate grounded answers using Gemini + FAISS retrieval")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--k", type=int, default=3, help="Top-k contexts to retrieve")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), help="Gemini model (e.g., gemini-1.5-flash, gemini-1.5-pro)")
    parser.add_argument("--max-output-tokens", type=int, default=600, help="Max tokens for the answer")
    parser.add_argument("--temperature", type=float, default=0.2, help="Creativity vs determinism")
    parser.add_argument("--dry-run", action="store_true", help="Validate GEMINI_API_KEY and exit")
    args = parser.parse_args()

    if args.dry_run:
        try:
            validate_key_or_die()
            print("✅ GEMINI_API_KEY is set and accepted by the client.")
        except Exception as e:
            print(f"❌ Gemini key validation failed: {e}")
        return

    answer = generate_answer(
        query=args.query,
        k=args.k,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )
    print("\n💡 Answer:\n")
    print(answer)
    print("")

if __name__ == "__main__":
    main()