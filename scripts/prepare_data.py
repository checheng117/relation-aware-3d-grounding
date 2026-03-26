#!/usr/bin/env python3
"""Validate raw layout, build manifests (CSV+ScanNet, combined CSV, import JSONL), or mock debug."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.builder import (
    build_from_combined_split_csv,
    build_records_from_csv_and_scans,
    build_records_nr3d_hf_with_scans,
    import_jsonl_records,
    mock_debug_records,
    split_records,
    write_split_manifests,
)
from rag3d.datasets.nr3d_hf import records_from_nr3d_hf_json
from rag3d.datasets.layout import ReferIt3DRawLayout, describe_expected_layout, validate_scans_dir
from rag3d.utils.config import load_yaml_config
from rag3d.utils.env import ensure_env_loaded, get_hf_token

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None  # type: ignore[misc, assignment]
from rag3d.utils.logging import setup_logging

import logging

log = logging.getLogger(__name__)


def _resolve(p: Path, base: Path) -> Path:
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def cmd_validate(cfg: dict, root: Path) -> int:
    raw_root = _resolve(Path(cfg["raw_root"]), root)
    if not raw_root.is_dir():
        log.error("raw_root does not exist: %s\n%s", raw_root, describe_expected_layout())
        return 1
    layout = ReferIt3DRawLayout(raw_root, cfg.get("scans_subdir", "scans"), cfg.get("annotations_subdir", "annotations"))
    n, scenes = validate_scans_dir(layout.scans_dir)
    log.info("Found %s scenes with *_vh_clean_aggregation.json under %s", n, layout.scans_dir)
    if n > 0:
        log.info("Example scene ids: %s", scenes[:5])
    ann = layout.annotations_dir
    if ann.is_dir():
        log.info("Annotation files: %s", sorted([p.name for p in ann.iterdir() if p.suffix.lower() in {".csv", ".jsonl", ".json"}]))
    else:
        log.warning("Annotations directory missing: %s", ann)
    if get_hf_token():
        log.info("HF_TOKEN is set (value not logged).")
    return 0


def cmd_mock(cfg: dict, root: Path, out_subdir: str) -> None:
    dbg = cfg.get("debug") or {}
    mock = dbg.get("mock") or {}
    n = int(mock.get("n_samples", 256))
    ns = int(mock.get("n_scenes", 16))
    fd = int(mock.get("feat_dim", 256))
    nobj = int(mock.get("objects_per_scene", 6))
    recs = mock_debug_records(n_total=n, n_scenes=ns, feat_dim=fd, objects_per_scene=nobj)
    tr, va, te = split_records(recs, train_ratio=0.8, val_ratio=0.1, seed=int(mock.get("seed", 42)))
    out_dir = _resolve(Path(cfg.get("processed_dir", "data/processed")), root) / out_subdir
    write_split_manifests(
        out_dir,
        tr,
        va,
        te,
        summary_extra={"source": "mock_debug", "feat_dim": fd},
    )
    log.info("Wrote mock manifests to %s", out_dir)


def _ensure_nr3d_hf_json(cfg: dict, root: Path, layout: ReferIt3DRawLayout) -> Path | None:
    block = cfg.get("nr3d_hf") or {}
    name = block.get("annotation_json")
    if not name:
        return None
    p = layout.annotations_dir / str(name)
    if p.is_file():
        return p
    if not block.get("auto_download_if_missing", False):
        return None
    tok = get_hf_token()
    if not tok:
        log.error("NR3D JSON missing at %s and HF_TOKEN unset; run scripts/fetch_nr3d_hf.py.", p)
        return None
    if hf_hub_download is None:
        log.error("huggingface_hub not installed; pip install huggingface_hub or run scripts/fetch_nr3d_hf.py.")
        return None
    repo = str(block.get("repo_id", "chouss/nr3d"))
    layout.annotations_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo, filename=str(name), repo_type="dataset", token=tok)
    dest = p
    dest.write_bytes(Path(cached).read_bytes())
    log.info("Downloaded %s/%s -> %s", repo, name, dest)
    return dest


def cmd_build_nr3d_hf(cfg: dict, root: Path) -> int:
    """Build manifests from HF-style NR3D JSON (no ScanNet aggregation required)."""
    raw_root = _resolve(Path(cfg["raw_root"]), root)
    raw_root.mkdir(parents=True, exist_ok=True)
    layout = ReferIt3DRawLayout(raw_root, cfg.get("scans_subdir", "scans"), cfg.get("annotations_subdir", "annotations"))
    layout.scans_dir.mkdir(parents=True, exist_ok=True)
    layout.annotations_dir.mkdir(parents=True, exist_ok=True)
    jp = _ensure_nr3d_hf_json(cfg, root, layout)
    if jp is None or not jp.is_file():
        log.error(
            "NR3D HF JSON not found. Set nr3d_hf.annotation_json and place the file under %s, "
            "or enable nr3d_hf.auto_download_if_missing with HF_TOKEN. See scripts/fetch_nr3d_hf.py.",
            layout.annotations_dir,
        )
        return 1
    dbg = cfg.get("debug") or {}
    max_rows = dbg.get("max_build_samples")
    recs = records_from_nr3d_hf_json(jp)
    if max_rows is not None:
        recs = recs[: int(max_rows)]
    tr, va, te = split_records(recs, train_ratio=0.8, val_ratio=0.1, seed=42)
    out_dir = _resolve(Path(cfg.get("processed_dir", "data/processed")), root)
    write_split_manifests(
        out_dir,
        tr,
        va,
        te,
        summary_extra={
            "source": f"nr3d_hf:{jp.name}",
            "note": "Placeholder geometry; add ScanNet *_vh_clean_aggregation.json for real OBBs.",
        },
    )
    log.info("Built NR3D HF manifests -> %s", out_dir)
    return 0


def cmd_build_nr3d_geom(cfg: dict, root: Path) -> int:
    """NR3D HF JSON + local ScanNet aggregation under scans/<scene_id>/ (real OBBs)."""
    raw_root = _resolve(Path(cfg["raw_root"]), root)
    raw_root.mkdir(parents=True, exist_ok=True)
    layout = ReferIt3DRawLayout(raw_root, cfg.get("scans_subdir", "scans"), cfg.get("annotations_subdir", "annotations"))
    layout.scans_dir.mkdir(parents=True, exist_ok=True)
    layout.annotations_dir.mkdir(parents=True, exist_ok=True)
    jp = _ensure_nr3d_hf_json(cfg, root, layout)
    if jp is None or not jp.is_file():
        log.error("NR3D HF JSON missing; same requirements as build-nr3d-hf.")
        return 1
    dbg = cfg.get("debug") or {}
    max_rows = dbg.get("max_build_samples")
    max_scenes = dbg.get("max_build_scenes")
    nr3d_blk = cfg.get("nr3d_hf") or {}
    candidate_set = str(nr3d_blk.get("candidate_set", "full_scene"))
    if candidate_set not in ("full_scene", "entity_only"):
        log.error("nr3d_hf.candidate_set must be 'full_scene' or 'entity_only', got %r", candidate_set)
        return 1
    recs, stats = build_records_nr3d_hf_with_scans(
        jp,
        layout,
        max_rows=max_rows,
        max_scenes=max_scenes,
        candidate_set=candidate_set,
    )
    if not recs:
        log.error(
            "No geometry-backed records produced. Add ScanNet aggregation under %s "
            "(see scripts/fetch_scannet_aggregations.py) or accept HF ScanNet dataset terms.",
            layout.scans_dir,
        )
        return 1
    tr, va, te = split_records(recs, train_ratio=0.8, val_ratio=0.1, seed=42)
    out_dir = _resolve(Path(cfg.get("processed_dir", "data/processed")), root)
    summary_extra = {
        "source": f"nr3d_hf+scans:{jp.name}",
        "note": "Geometry-backed: objects from ScanNet-style aggregation JSON "
        f"(candidate_set={candidate_set}).",
        **stats,
    }
    write_split_manifests(out_dir, tr, va, te, summary_extra=summary_extra)
    log.info("Built geometry-backed NR3D manifests -> %s (kept %s rows)", out_dir, stats.get("records_kept"))
    return 0


def cmd_build(cfg: dict, root: Path) -> int:
    raw_root = _resolve(Path(cfg["raw_root"]), root)
    if not raw_root.is_dir():
        log.error("raw_root missing: %s", raw_root)
        return 1
    layout = ReferIt3DRawLayout(raw_root, cfg.get("scans_subdir", "scans"), cfg.get("annotations_subdir", "annotations"))
    dbg = cfg.get("debug") or {}
    max_rows = dbg.get("max_build_samples")
    max_scenes = dbg.get("max_build_scenes")
    out_dir = _resolve(Path(cfg.get("processed_dir", "data/processed")), root)

    import_path = cfg.get("import_jsonl")
    if import_path:
        ip = _resolve(Path(import_path), root)
        recs = import_jsonl_records(ip, max_rows=max_rows)
        tr, va, te = split_records(recs, train_ratio=0.8, val_ratio=0.1, seed=42)
        write_split_manifests(out_dir, tr, va, te, summary_extra={"source": f"import_jsonl:{ip.name}"})
        log.info("Imported and split → %s", out_dir)
        return 0

    combined = cfg.get("combined_csv")
    if combined:
        cp = layout.annotations_dir / combined
        if not cp.is_file():
            log.error("Combined CSV not found: %s", cp)
            return 1
        tr, va, te = build_from_combined_split_csv(cp, layout, max_rows=max_rows, max_scenes=max_scenes)
        write_split_manifests(out_dir, tr, va, te, summary_extra={"source": f"combined_csv:{combined}"})
        log.info("Built from combined CSV → %s", out_dir)
        return 0

    ann = layout.annotations_dir
    train_csv = ann / cfg.get("train_csv", "train.csv")
    val_csv = ann / cfg.get("val_csv", "val.csv")
    test_csv = ann / cfg.get("test_csv", "test.csv")
    train_r: list = []
    val_r: list = []
    test_r: list = []
    if train_csv.is_file():
        train_r = build_records_from_csv_and_scans(train_csv, layout, max_rows=max_rows, max_scenes=max_scenes)
    if val_csv.is_file():
        val_r = build_records_from_csv_and_scans(val_csv, layout, max_rows=max_rows, max_scenes=max_scenes)
    if test_csv.is_file():
        test_r = build_records_from_csv_and_scans(test_csv, layout, max_rows=max_rows, max_scenes=max_scenes)
    if not train_r and not val_r and not test_r:
        log.error(
            "No manifest records produced (empty train/val/test). Check CSVs at %s, %s, %s and ScanNet "
            "aggregations under %s (see --mode build-nr3d-geom or docs/DATASET_SETUP.md).\n%s",
            train_csv,
            val_csv,
            test_csv,
            layout.scans_dir,
            describe_expected_layout(),
        )
        return 1
    write_split_manifests(
        out_dir,
        train_r,
        val_r,
        test_r,
        summary_extra={"source": "split_csv"},
    )
    log.info("Built manifests → %s", out_dir)
    return 0


def cmd_template(root: Path, out: Path) -> None:
    template = [
        {
            "scene_id": "scene0000_00",
            "utterance": "example utterance",
            "target_object_id": "1",
            "target_index": 0,
            "utterance_id": "ex_0",
            "relation_type_gold": "left-of",
            "tags": {},
            "objects": [
                {
                    "object_id": "1",
                    "class_name": "chair",
                    "center": [0.0, 0.0, 0.0],
                    "size": [0.5, 0.5, 1.0],
                }
            ],
        }
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=2), encoding="utf-8")
    log.info("Wrote template JSON to %s (convert to .jsonl one record per line if preferred)", out)


def main() -> None:
    setup_logging()
    ensure_env_loaded()
    ap = argparse.ArgumentParser(description="Prepare ReferIt3D/ScanNet-style manifests.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/dataset/referit3d.yaml")
    ap.add_argument("--root", type=Path, default=ROOT, help="Repository root.")
    ap.add_argument(
        "--mode",
        choices=["validate", "build", "build-nr3d-hf", "build-nr3d-geom", "mock-debug", "template"],
        default="validate",
    )
    ap.add_argument("--mock-out-subdir", type=str, default="debug", help="Under processed_dir for mock-debug.")
    ap.add_argument("--template-out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_yaml_config(args.config, base_dir=args.root)
    if args.mode == "template":
        out = args.template_out or (args.root / cfg.get("processed_dir", "data/processed") / "template_manifest.json")
        cmd_template(args.root, _resolve(out, args.root))
        return
    if args.mode == "validate":
        raise SystemExit(cmd_validate(cfg, args.root))
    if args.mode == "mock-debug":
        cmd_mock(cfg, args.root, args.mock_out_subdir)
        return
    if args.mode == "build":
        raise SystemExit(cmd_build(cfg, args.root))
    if args.mode == "build-nr3d-hf":
        raise SystemExit(cmd_build_nr3d_hf(cfg, args.root))
    if args.mode == "build-nr3d-geom":
        raise SystemExit(cmd_build_nr3d_geom(cfg, args.root))


if __name__ == "__main__":
    main()
