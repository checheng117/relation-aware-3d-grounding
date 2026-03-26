#!/usr/bin/env python3
"""Stream Pointcept ``scannet.tar.gz`` (legacy / usually not useful for this repo).

``Pointcept/scannet-compressed`` is **preprocessed point clouds** (``coord.npy``, ``instance.npy``, …), not
official ``*_vh_clean_aggregation.json``. A full tar scan finds **no** aggregation files, so this script
does not satisfy ``prepare_data.py --mode build-nr3d-geom``.

For NR3D + instance alignment without multi-GB per-scene zips, prefer::

    python scripts/fetch_scannet_aggregations.py --artifact aggregation-json \\
        --repo-id zahidpichen/scannet-dataset --nr3d-json data/raw/referit3d/annotations/nr3d_annotations.json

Token is read from the environment only (``HF_TOKEN`` / ``.env``). Does not print secret values.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.env import ensure_env_loaded, get_hf_token
from rag3d.utils.logging import setup_logging

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[misc, assignment]

log = logging.getLogger(__name__)

_AGG_SUFFIX = "_vh_clean_aggregation.json"
_SCENE_RE = re.compile(r"^(.+)_vh_clean_aggregation\.json$")


def _canonical_dest(scans_dir: Path, scene_id: str) -> Path:
    return scans_dir / scene_id / f"{scene_id}_vh_clean_aggregation.json"


def _scene_id_from_member_name(name: str) -> str | None:
    base = Path(name).name
    m = _SCENE_RE.match(base)
    return m.group(1) if m else None


def _drain_member(tar: tarfile.TarFile, info: tarfile.TarInfo) -> None:
    """Sequential tar streams require reading each file member's payload before ``next()``."""
    if not info.isfile():
        return
    f = tar.extractfile(info)
    if f is None:
        return
    with open(os.devnull, "wb") as dev:
        shutil.copyfileobj(f, dev, length=1024 * 1024)


def _load_wanted_scenes(nr3d_json: Path | None) -> set[str] | None:
    if nr3d_json is None:
        return None
    data = json.loads(nr3d_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {nr3d_json}")
    out: set[str] = set()
    for row in data:
        if isinstance(row, dict):
            sid = str(row.get("scene_id") or "").strip()
            if sid:
                out.add(sid)
    return out


def main() -> int:
    setup_logging()
    ensure_env_loaded()
    ap = argparse.ArgumentParser(description="Extract ScanNet aggregation JSON from Pointcept scannet.tar.gz.")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/referit3d"))
    ap.add_argument("--scans-subdir", type=str, default="scans")
    ap.add_argument(
        "--tar-path",
        type=Path,
        default=None,
        help="Local scannet.tar.gz; if omitted, download from Hugging Face.",
    )
    ap.add_argument("--repo-id", type=str, default="Pointcept/scannet-compressed")
    ap.add_argument("--hf-filename", type=str, default="scannet.tar.gz")
    ap.add_argument(
        "--nr3d-json",
        type=Path,
        default=None,
        help="If set, only write scenes whose scene_id appears in this NR3D JSON (still scans full tar).",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip if canonical aggregation already exists.")
    args = ap.parse_args()

    base = args.root.resolve()
    raw = (args.raw_root if args.raw_root.is_absolute() else base / args.raw_root).resolve()
    scans_dir = raw / args.scans_subdir
    scans_dir.mkdir(parents=True, exist_ok=True)

    wanted = None
    if args.nr3d_json is not None:
        jp = args.nr3d_json if args.nr3d_json.is_absolute() else base / args.nr3d_json
        if not jp.is_file():
            log.error("NR3D JSON not found: %s", jp)
            return 1
        wanted = _load_wanted_scenes(jp)
        log.info("Restricting writes to %s unique scene_ids from %s", len(wanted), jp.name)

    tar_path = args.tar_path
    if tar_path is None:
        if hf_hub_download is None:
            log.error("huggingface_hub not installed.")
            return 1
        tok = get_hf_token()
        if not tok:
            log.error("HF_TOKEN unset; cannot download gated Pointcept/scannet-compressed.")
            return 1
        log.info("Downloading %s/%s (cached by huggingface_hub)…", args.repo_id, args.hf_filename)
        tar_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.hf_filename,
                repo_type="dataset",
                token=tok,
            )
        )
    else:
        tar_path = tar_path.resolve()
        if not tar_path.is_file():
            log.error("Missing tar: %s", tar_path)
            return 1

    log.info("Streaming %s (one sequential pass)…", tar_path)
    written = 0
    skipped_filter = 0
    skipped_exists = 0
    members_seen = 0

    with tarfile.open(tar_path, "r|gz") as tar:
        while True:
            info = tar.next()
            if info is None:
                break
            members_seen += 1
            if not info.isfile():
                continue
            sid = _scene_id_from_member_name(info.name)
            is_agg = sid is not None and _AGG_SUFFIX in info.name

            if not is_agg:
                _drain_member(tar, info)
                continue

            if wanted is not None and sid not in wanted:
                skipped_filter += 1
                _drain_member(tar, info)
                continue

            dest = _canonical_dest(scans_dir, sid)
            if args.skip_existing and dest.is_file():
                skipped_exists += 1
                _drain_member(tar, info)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            f = tar.extractfile(info)
            if f is None:
                log.warning("Could not read member: %s", info.name)
                continue
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with tmp.open("wb") as out:
                shutil.copyfileobj(f, out, length=1024 * 1024)
            tmp.replace(dest)
            written += 1
            if written % 50 == 0:
                log.info("Wrote %s aggregation files…", written)

    log.info(
        "Done. tar members scanned=%s, aggregations written=%s, skipped (not in NR3D filter)=%s, skipped existing=%s → %s",
        members_seen,
        written,
        skipped_filter,
        skipped_exists,
        scans_dir,
    )
    if wanted is not None:
        have = sum(1 for s in wanted if _canonical_dest(scans_dir, s).is_file())
        log.info("Coverage: %s/%s NR3D scenes now have aggregation under scans/", have, len(wanted))
    return 0 if written > 0 or (args.skip_existing and skipped_exists > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
