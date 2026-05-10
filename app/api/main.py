import os
import time
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.models import QueryRequest, AnswerResponse, ContextItem
from app.api.deps import ensure_ready
from app.rag import Retriever, answer_from_contexts

load_dotenv()

app = FastAPI(
    title = "DevDocs Chatbot API",
    version = "0.1.0",
    description = "RAG + Gemini over GitHub repositories",
    docs_url = "/docs",
    redoc_url = "/redoc",
    openapi_url = "/openapi.json"
)

# CORS (open for local development; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],  # Optionally set to specific origin in production
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/health", tags = ["Health"])
def health():
    return {"status": "ok"}

@app.get("/ready", tags = ["Ready"])
def ready():
    return {"status": "ready", "model": "gemini-2.0-flash"}

@app.post("/query", response_model = AnswerResponse, tags = ["Query"])
def query(request: QueryRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code = 422, detail = "Query text cannot be empty")

    # Ensure chunks and vector store exist (auto-build if missing)
    try:
         ensure_ready(
            repo_name = request.repo_name,
            repo_url = request.repo_url,
            force_build = request.force_build,
            keep_text_in_metadata = False
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Preparation failed: {e}")

    # Retrieve top-k contexts
    t0 = time.perf_counter()
    try:
        retriever = Retriever(
            k = request.k,
            repo_name = request.repo_name,
            use_separate_texts = os.getenv("USE_SEPARATE_TEXTS", "true").lower() in {"1","true","yes"}
        )
        raw_results = retriever.search(request.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code = 500, detail = f"Vector store missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Retrieval failed: {str(e)}")

    if not raw_results:
        return AnswerResponse(
            answer = "I couldn't find relevant context in the repository to answer that.",
            contexts = [],
            model = request.model,
            k = request.k,
            latency_ms = int((time.perf_counter() - t0) * 1000)
        )

    # Normalize contexts to API schema
    contexts = [
        ContextItem(
            file_path = r.get("file_path", "(unknown)"),
            chunk_id = str(r.get("chunk_id", "-1")),
            distance = float(r.get("distance", 0.0)),
            text = r.get("text", "")
        ).model_dump()
        for r in raw_results
    ]

    # Call Gemini with the retrieved contexts
    try:
        answer = answer_from_contexts(
            query = request.text,
            contexts = raw_results,
            model_name = request.model,
            max_output_tokens = request.max_output_tokens,
            temperature = request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Generation failed: {str(e)}")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AnswerResponse(answer = answer, contexts = contexts, model = request.model, k = request.k, latency_ms = latency_ms)
