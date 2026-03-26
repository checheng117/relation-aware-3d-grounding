#!/usr/bin/env python3
"""Download NR3D JSON from Hugging Face (chouss/nr3d) into data/raw/referit3d/annotations/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import hf_hub_download

from rag3d.utils.env import ensure_env_loaded, get_hf_token
from rag3d.utils.logging import setup_logging

import logging

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ensure_env_loaded()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="chouss/nr3d")
    ap.add_argument("--filename", default="nr3d_annotations.json")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/raw/referit3d/annotations",
        help="Directory to copy/symlink the downloaded file into.",
    )
    args = ap.parse_args()
    tok = get_hf_token()
    if not tok:
        log.error("HF_TOKEN missing; set it in .env or the environment.")
        raise SystemExit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type="dataset",
        token=tok,
    )
    src = Path(cached)
    dest = args.out_dir / args.filename
    dest.write_bytes(src.read_bytes())
    log.info("Wrote %s (%s bytes)", dest, dest.stat().st_size)


if __name__ == "__main__":
    main()
