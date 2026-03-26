"""Environment loading. Never log secret values."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    """Repository root (parent of `src/`)."""
    return Path(__file__).resolve().parents[3]


def ensure_env_loaded() -> None:
    """Load `.env` from repository root if present (does not override existing env)."""
    env_path = repo_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def get_hf_token() -> str | None:
    """Return Hugging Face token from environment, or None if unset."""
    ensure_env_loaded()
    tok = os.environ.get("HF_TOKEN")
    if tok is not None and not str(tok).strip():
        return None
    return tok
