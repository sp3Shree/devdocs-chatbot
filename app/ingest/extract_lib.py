from __future__ import annotations
import os, json, shutil
import requests
import stat
from pathlib import Path
from typing import List, Dict
from git import Repo
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# Load from .env file into environment
load_dotenv()

EXCLUDED_DIRS = {"venv", "__pycache__", ".git", "node_modules", "build", "dist"}
CHUNK_SIZE = 400
OVERLAP = 50

LANGUAGE_EXTENSIONS = {
    "Python": [".py", ".ipynb", ".md", ".txt", ".rst", ".yml", ".yaml", ".json", ".toml"],
    "JavaScript": [".js", ".jsx", ".json", ".md", ".yml"],
    "TypeScript": [".ts", ".tsx", ".json", ".md", ".yml"],
    "Java": [".java", ".xml", ".md", ".properties", ".json"],
    "Go": [".go", ".mod", ".sum", ".md", ".yml"],
    "Rust": [".rs", ".toml", ".md"],
    "C++": [".cpp", ".h", ".hpp", ".md"],
    "C": [".c", ".h", ".md"],
    "Shell": [".sh", ".md", ".env"],
    "HTML": [".html", ".css", ".js", ".md"],
    "Dockerfile": ["Dockerfile", ".env", ".md"],
}

def get_primary_language(repo_url: str, token: str = None) -> str:
    parts = repo_url.rstrip("/").split("/")
    user, repo = parts[-2], parts[-1]
    api_url = f"https://api.github.com/repos/{user}/{repo}"
    headers = {"Authorization": f"token {token}"} if token else {}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json().get("language", "Python")
    except Exception as e:
        print(f"⚠️ Error fetching language from GitHub API: {e}")
        return "Python"

def get_allowed_extensions(language: str) -> list:
    return LANGUAGE_EXTENSIONS.get(language, [".py", ".md", ".txt"])

def should_include(path: Path, allowed_exts) -> bool:
    if path.name == "Dockerfile":
        return True
    return (path.suffix in allowed_exts) and not any(part in EXCLUDED_DIRS for part in path.parts)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def clone_repo(repo_url: str, dest_path: Path, force: bool = False) -> None:
    if dest_path.exists():
        if force:
            print(f"🧹 Removing existing repo at {dest_path}")
            shutil.rmtree(dest_path, onerror=handle_remove_readonly)
        else:
            print(f"📂 Using existing repo at {dest_path}")
            return
    print(f"🔄 Cloning {repo_url} to {dest_path}")
    Repo.clone_from(repo_url, dest_path)

def extract_to_jsonl(repo_url: str, repo_name: str, base_dir: Path, github_token: str | None = None, force_clone: bool = False) -> Path:
    raw_dir = base_dir/ "raw" / repo_name
    chunks_dir = base_dir/ "chunks" / repo_name
    chunks_path = chunks_dir / "chunks.jsonl"

    # Clone if missing
    clone_repo(repo_url, raw_dir, force=force_clone)

    # Detect language with allowed extensions
    language = get_primary_language(repo_url, github_token)
    allowed_exts = get_allowed_extensions(language)

    # Walk & chunk
    docs: List[Dict] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if not should_include(path, allowed_exts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"⚠️ Failed to read {path}: {e}")
        for i, chunk in enumerate(chunk_text(text)):
            docs.append({"file_path": str(path.relative_to(raw_dir)), "chunk_id": i, "text": chunk})

    chunks_dir.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return chunks_path

def handle_remove_readonly(func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)