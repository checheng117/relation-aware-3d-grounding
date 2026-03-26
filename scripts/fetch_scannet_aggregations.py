#!/usr/bin/env python3
"""Download ScanNet scene zips from Hugging Face and normalize aggregation JSON into scans/<scene_id>/.

Requires ``huggingface_hub`` and a token with access to the chosen dataset repo (many ScanNet
mirrors are gated). Token is read from the environment only (e.g. ``HF_TOKEN`` in ``.env`` via
dotenv in your shell or ``python-dotenv`` in callers).

Does not print secret values.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.scannet_objects import resolve_aggregation_path
from rag3d.utils.env import ensure_env_loaded, get_hf_token
from rag3d.utils.logging import setup_logging

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[misc, assignment]


def _canonical_aggregation_dest(scans_dir: Path, scene_id: str) -> Path:
    return scans_dir / scene_id / f"{scene_id}_vh_clean_aggregation.json"


def _find_scene_root(extract_root: Path, scene_id: str) -> Path | None:
    direct = extract_root / scene_id
    if direct.is_dir():
        return direct
    for p in extract_root.rglob(scene_id):
        if p.is_dir() and p.name == scene_id:
            return p
    return None


def normalize_scene_to_scans(extract_root: Path, scene_id: str, scans_dir: Path) -> bool:
    """Copy aggregation JSON to ``scans/<scene_id>/<scene_id>_vh_clean_aggregation.json``."""
    scene_root = _find_scene_root(extract_root, scene_id)
    if scene_root is None:
        return False
    found = resolve_aggregation_path(scene_root, scene_id)
    if found is None:
        return False
    dest = _canonical_aggregation_dest(scans_dir, scene_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(found, dest)
    return True


def _strip_proxy_env() -> None:
    """Avoid broken corporate/sandbox proxies returning 403 for huggingface.co (set NO_PROXY too)."""
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "SOCKS_PROXY",
        "socks_proxy",
    ):
        os.environ.pop(key, None)


def fetch_scene_zip(repo_id: str, scene_id: str, token: str | None) -> Path:
    if hf_hub_download is None:
        raise RuntimeError("Install huggingface_hub (pip install huggingface_hub).")
    name = f"{scene_id}.zip"
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=name,
            repo_type="dataset",
            token=token,
        )
    )


def fetch_scene_aggregation_json(repo_id: str, scene_id: str, token: str | None) -> Path:
    """Download ``<scene_id>/<scene_id>.aggregation.json`` (layout used by zahidpichen/scannet-dataset)."""
    if hf_hub_download is None:
        raise RuntimeError("Install huggingface_hub (pip install huggingface_hub).")
    name = f"{scene_id}/{scene_id}.aggregation.json"
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=name,
            repo_type="dataset",
            token=token,
        )
    )


def _aggregation_installed(scans_dir: Path, scene_id: str) -> bool:
    p = resolve_aggregation_path(scans_dir / scene_id, scene_id)
    return p is not None and p.is_file()


log = logging.getLogger(__name__)


def main() -> int:
    setup_logging()
    ensure_env_loaded()
    ap = argparse.ArgumentParser(description="Fetch ScanNet scene zips and install aggregation JSON.")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw/referit3d"))
    ap.add_argument("--scans-subdir", type=str, default="scans")
    ap.add_argument(
        "--repo-id",
        type=str,
        default="Gen3DF/scannet",
        help="HF dataset repo: per-scene zips (Gen3DF/scannet) or flat aggregation JSON (zahidpichen/scannet-dataset).",
    )
    ap.add_argument(
        "--artifact",
        choices=("zip", "aggregation-json"),
        default="zip",
        help="zip: download <scene_id>.zip and extract aggregation. aggregation-json: download scene_id/scene_id.aggregation.json (no OBB fields in that mirror; full segGroups / objectId).",
    )
    ap.add_argument(
        "--nr3d-json",
        type=Path,
        default=None,
        help="NR3D HF JSON; all unique scene_id values are candidates to fetch.",
    )
    ap.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated scene ids (alternative to --nr3d-json).",
    )
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", help="Skip if canonical aggregation already exists.")
    ap.add_argument("--temp-dir", type=Path, default=None)
    ap.add_argument(
        "--no-proxy",
        action="store_true",
        help="Unset HTTP(S)_PROXY env vars before calling Hugging Face (use with NO_PROXY=huggingface.co if needed).",
    )
    ap.add_argument(
        "--local-zip-dir",
        type=Path,
        default=None,
        help="If set, read <scene_id>.zip from this directory instead of downloading from HF (official ScanNet zips work).",
    )
    args = ap.parse_args()

    if args.no_proxy:
        _strip_proxy_env()

    base = args.root.resolve()
    raw = (args.raw_root if args.raw_root.is_absolute() else base / args.raw_root).resolve()
    scans_dir = raw / args.scans_subdir

    scenes: list[str] = []
    if args.scenes.strip():
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    elif args.nr3d_json:
        jp = args.nr3d_json if args.nr3d_json.is_absolute() else base / args.nr3d_json
        data = json.loads(jp.read_text(encoding="utf-8"))
        seen: set[str] = set()
        for row in data:
            if isinstance(row, dict):
                sid = str(row.get("scene_id") or "").strip()
                if sid:
                    seen.add(sid)
        scenes = sorted(seen)
    else:
        print("Provide --nr3d-json or --scenes.", file=sys.stderr)
        return 1

    if args.max_scenes is not None:
        scenes = scenes[: max(0, args.max_scenes)]

    tok = get_hf_token()
    if not tok:
        print("HF_TOKEN unset; gated ScanNet mirrors will fail to download.", file=sys.stderr)

    ok = 0
    failed: list[str] = []
    for sid in scenes:
        if args.skip_existing and _aggregation_installed(scans_dir, sid):
            ok += 1
            continue
        try:
            if args.artifact == "aggregation-json":
                if args.local_zip_dir is not None:
                    raise ValueError("--local-zip-dir is only valid with --artifact zip.")
                src = fetch_scene_aggregation_json(args.repo_id, sid, tok)
                out_dir = scans_dir / sid
                out_dir.mkdir(parents=True, exist_ok=True)
                dest = out_dir / f"{sid}.aggregation.json"
                shutil.copy2(src, dest)
                ok += 1
                continue

            with tempfile.TemporaryDirectory(dir=args.temp_dir) as td:
                tdp = Path(td)
                if args.local_zip_dir is not None:
                    lz = args.local_zip_dir.resolve()
                    zpath = lz / f"{sid}.zip"
                    if not zpath.is_file():
                        raise FileNotFoundError(f"Missing local zip: {zpath}")
                else:
                    zpath = fetch_scene_zip(args.repo_id, sid, tok)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(tdp)
                if normalize_scene_to_scans(tdp, sid, scans_dir):
                    ok += 1
                else:
                    failed.append(sid)
        except Exception as e:
            log.warning("Scene %s failed: %s: %s", sid, type(e).__name__, e)
            log.debug("%s", traceback.format_exc())
            failed.append(sid)

    print(f"Fetched/normalized {ok}/{len(scenes)} scenes into {scans_dir}")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:20])}{' ...' if len(failed) > 20 else ''}", file=sys.stderr)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
