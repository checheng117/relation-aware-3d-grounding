#!/usr/bin/env python3
"""Lightweight environment sanity check (no secrets printed)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _fail(msg: str) -> None:
    print(f"[fail] {msg}", file=sys.stderr)


def main() -> int:
    py = sys.version_info
    if py.major == 3 and py.minor == 10:
        _ok(f"Python {py.major}.{py.minor}.{py.micro} (expected 3.10.x for this repo)")
    elif py.major == 3 and py.minor in (11, 12):
        print(f"[warn] Python {py.major}.{py.minor} — repo targets 3.10; prefer conda env rag3d.")
    else:
        _fail(f"Python {py.major}.{py.minor} — use Python 3.10 via conda (see README.md)")
        return 1

    try:
        import torch

        _ok(f"torch import ({torch.__version__})")
    except Exception as e:  # noqa: BLE001
        _fail(f"torch import: {e}")
        return 1

    for mod in ("numpy", "yaml", "pydantic", "dotenv", "tqdm"):
        try:
            importlib.import_module("yaml" if mod == "yaml" else mod)
            _ok(f"import {mod}")
        except Exception as e:  # noqa: BLE001
            _fail(f"import {mod}: {e}")
            return 1

    sys.path.insert(0, str(ROOT / "src"))
    try:
        import rag3d  # noqa: F401

        _ok("import rag3d (editable install recommended)")
    except Exception as e:  # noqa: BLE001
        _fail(f"import rag3d: {e} — run: pip install -e .")
        return 1

    # Optional: dotenv does not export token values
    try:
        from rag3d.utils.env import ensure_env_loaded

        ensure_env_loaded()
    except Exception:  # noqa: BLE001
        pass

    if os.environ.get("HF_TOKEN"):
        _ok("HF_TOKEN is set in environment (value not shown)")
    else:
        print("[info] HF_TOKEN not set — optional unless using gated Hugging Face assets")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
