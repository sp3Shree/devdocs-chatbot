# DevDocs Chatbot 🤖

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=flat-square&logo=docker)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-1.5--flash-yellow?style=flat-square&logo=google)

> **Chat with any GitHub repository in plain English.**
> Point it at a repo. Ask anything. Get answers grounded in the actual code and docs.

---

## What Problem Does This Solve?

Onboarding to a new codebase is painful. Docs are scattered. READMEs are shallow. You spend hours just figuring out *where to start.*

DevDocs Chatbot fixes this:
- Ask "What does this repo do?" - get a real answer, not a regurgitated README
- Ask "How is authentication handled?" - get a code-aware response
- Ask "What frameworks and dependencies are used?" - instant answer

---

## Architecture

```
GitHub Repo URL
      │
      ▼
 Clone & Parse (.py, .md, .json, configs)
      │
      ▼
 Chunk + Embed (SentenceTransformers)
      │
      ▼
 FAISS Vector Store
      │
      ▼
 RAG Retrieval → Gemini / OpenAI LLM
      │
      ▼
 FastAPI /query Endpoint → JSON Response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| RAG Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | SentenceTransformers |
| LLM | Gemini 1.5 Flash / OpenAI |
| Containerization | Docker + Docker Compose |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- API key: `GEMINI_API_KEY` (or `OPENAI_API_KEY`)

### Run with Docker

```bash
# Clone this repo
git clone https://github.com/sp3Shree/devdocs-chatbot
cd devdocs-chatbot

# Add your API key to .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Build and start
docker compose up --build

# Verify it's running
curl http://localhost:8000/health
```

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "How do I fit a Random Forest model?",
    "repo_name": "scikit-learn",
    "repo_url": "https://github.com/scikit-learn/scikit-learn",
    "k": 3,
    "model": "gemini-1.5-flash",
    "temperature": 0.2,
    "max_output_tokens": 500,
    "force_build": true
  }'
```

**Sample Response:**
```json
{
  "answer": "To fit a Random Forest model in scikit-learn, use RandomForestClassifier().fit(X_train, y_train). The repo uses sklearn.ensemble under /sklearn/ensemble/_forest.py ...",
  "sources": ["README.md", "sklearn/ensemble/_forest.py"]
}
```

---

## Key Features

- **RAG-powered** - Answers grounded in actual repo content, not hallucinations
- **Multi-repo** - Point at any public GitHub repo dynamically
- **Model-agnostic** - Swap between Gemini and OpenAI via request param
- **Dockerized** - One command to run anywhere
- **REST API** - Easy to integrate into any toolchain

---

## Roadmap

- [ ] Streaming responses via WebSocket
- [ ] Support private repos (GitHub token auth)
- [ ] Frontend UI (React-based chat interface)
- [ ] Persistent vector store (replace in-memory FAISS)
- [ ] Deploy to cloud (AWS Lambda / GCP Cloud Run)

---

## Built By

**Shreeansh Priyadarshi** — Senior Technology Consultant → aspiring AI/ML Engineer

[GitHub](https://github.com/sp3Shree) · [LinkedIn](https://linkedin.com/in/shree-p)

*Part of my public AI/ML portfolio build. Feedback welcome.*