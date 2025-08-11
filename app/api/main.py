import time
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.models import QueryRequest, AnswerResponse, ContextItem
from app.api.deps import get_retriever
from app.rag.generator import answer_from_contexts

load_dotenv()

app = FastAPI(
    title="DevDocs Chatbot API",
    version="0.1.0",
    description="RAG + Gemini over GitHub repositories"
)

# CORS (open for local development; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Optionally set to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.get("/ready", tags=["Ready"])
def ready(retriever=Depends(get_retriever)):
    return {"status": "ready", "model": "gemini-1.5-flash"}

@app.post("/query", response_model=AnswerResponse, tags=["Query"])
def query(request: QueryRequest, retriever=Depends(get_retriever)):
    """
    Retrieve top-k contexts from FAISSm call Gemini with those contexts, return grounded answer.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Query text cannot be empty")

    # Retrieve top-k contexts
    t0 = time.perf_counter()
    try:
        # Temporarily override k for testing
        retriever.k=request.k
        raw_results = retriever.search(request.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Vector store missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not raw_results:
        return AnswerResponse(
            answer="I couldn't find relevant context in the repository to answer that.",
            contexts=[],
            model=request.model,
            k=request.k,
            latency_ms=int((time.perf_counter() - t0) * 1000)
        )

    # Normalize contexts to API schema
    contexts = [
        ContextItem(
            file_path=r.get("file_path", "(unknown)"),
            chunk_id=int(r.get("chunk_id", -1)),
            distance=float(r.get("distance", 0.0)),
            text=r.get("text", "")
        ).model_dump()
        for r in raw_results
    ]

    # Call Gemini with the retrieved contexts
    try:
        answer = answer_from_contexts(
            query=request.text,
            contexts=raw_results,
            model_name=request.model,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AnswerResponse(answer=answer, contexts=contexts, model=request.model, k=request.k, latency_ms=latency_ms)
