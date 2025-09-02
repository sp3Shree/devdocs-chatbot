from __future__ import annotations
import os, json, shutil
from pathlib import Path
from typing import Tuple
from app import project_paths, get_env
from app.ingest.extract_lib import extract_to_jsonl
from app.rag.embedder_lib import build_faiss_for_repo

USE_SEPARATE_TEXTS = os.getenv("USE_SEPARATE_TEXTS", "true").lower() in {"1","true","yes"}

def manifest_path(repo_name: str) -> Path:
    return project_paths.MANIFESTS / f"{repo_name}.json"

def load_manifest(repo_name: str) -> dict | None:
    mp = manifest_path(repo_name)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def save_manifest(repo_name: str, repo_url: str) -> None:
    mp = manifest_path(repo_name)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps({"repo_name": repo_name, "repo_url": repo_url}, ensure_ascii=False, indent=2), encoding="utf-8")

def clean_repo(repo_name: str) -> None:
    for p in [
        project_paths.RAW / repo_name,
        project_paths.CHUNKS / repo_name,
        project_paths.VECTOR_STORE / repo_name,
        manifest_path(repo_name)
    ]:
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

def ensure_ready(repo_name: str, repo_url: str, *, force_build: bool = False, keep_text_in_metadata: bool = True) -> Tuple[bool, str]:
    chunks_path = project_paths.CHUNKS / repo_name / "chunks.jsonl"
    index_path = project_paths.VECTOR_STORE / repo_name / "faiss.index"
    manifest = load_manifest(repo_name)

    if force_build:
        clean_repo(repo_name)
        manifest = None

    if manifest and manifest.get("repo_url") != repo_url:
        clean_repo(repo_name)
        manifest = None

    if not chunks_path.exists():
        github_token = get_env("GITHUB_TOKEN")
        chunks_path = extract_to_jsonl(
            repo_url=repo_url,
            repo_name=repo_name,
            base_dir=project_paths.DATA,
            github_token=github_token,
            force_clone=force_build,
        )
        save_manifest(repo_name, repo_url)

    if not index_path.exists():
        index_path = build_faiss_for_repo(
            repo_name=repo_name,
            base_dir=project_paths.DATA,
            keep_text_in_metadata=keep_text_in_metadata
        )
        if not manifest:
            save_manifest(repo_name, repo_url)

    return chunks_path, index_path