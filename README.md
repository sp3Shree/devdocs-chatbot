# DevDocs Chatbot
DevDocs Chatbot helps developers understand any GitHub repository in plain English.
Instead of manually reading through README.md, docs, and scattered code, you can summarize and interact with a repo conversationally. 

Think of it as a personal guide: “Explain this repo to me like I’m new here.”

# Purpose

Modern codebases are full of files such as docs, configs, PR templates, licenses, and it’s not always clear where to start. This project aims to:
- Summarize a repo in layman’s terms (not just regurgitate README)
- Highlight frameworks, languages, and dependencies from project structure
- Answer questions about code or docs in a conversational style
- Lower the barrier for newcomers, whether you’re onboarding to a new job, exploring open source, or reviewing a dependency

# How It Works

- Ingest & Parse
  - Clone any GitHub repo.
  - Parse relevant files (.py, .md, .json, etc.) and chunk them into manageable pieces.
- Embed & Store
  - Generate dense vector embeddings of chunks with SentenceTransformers.
  - Store them in a FAISS vector database.
- Retrieve & Generate
  - Retrieve the most relevant chunks for a query.
  - Use Gemini (or OpenAI models) to generate an answer, grounded in the repo content.
- FastAPI Service
  - Expose a /query endpoint that accepts:

```
{
    "text": "How do I fit a Random Forest model?", 
    "repo_name": "scikit-learn", 
    "repo_url": "https://github.com/scikit-learn/scikit-learn", 
    "k": 3, 
    "model": "gemini-1.5-flash", 
    "temperature": 0.2, 
    "max_output_tokens": 500, 
    "force_build": true
}
```

# Quick Start
## Prerequisites
- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- API key for Gemini (GEMINI_API_KEY) or OpenAI (OPENAI_API_KEY)

## Run Locally (with Docker Compose)
``` 
# build and start
docker compose up --build

# test health
curl http://localhost:8000/health
```
## Query Example
```
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