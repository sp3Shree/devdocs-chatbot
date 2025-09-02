from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables once for the whole app
# Safe to call multiple times; a no-op if already loaded.
load_dotenv()

__all__ = ["__version__", "get_env", "project_paths"]
__version__ = "0.1.0"

def get_env(key: str, default: str | None = None) -> str | None:
    """Typed, tiny wrapper around os.getenv."""
    return os.getenv(key, default)

class project_paths:
    """Centralized, import-safe paths (avoid sprinkling strings)."""
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    RAW = DATA / "raw"              # expect subdir per repo_name
    MANIFESTS = DATA / "manifests"
    CHUNKS = DATA / "chunks"              # expect subdir per repo_name
    VECTOR_STORE = DATA / "vector_store"  # expect subdir per repo_name
