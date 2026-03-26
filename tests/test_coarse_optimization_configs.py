"""All coarse optimization YAMLs load and reference valid manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag3d.utils.config import load_yaml_config

ROOT = Path(__file__).resolve().parents[1]
OPT_DIR = ROOT / "configs/train/coarse/optimization"


@pytest.mark.parametrize(
    "name",
    [
        "coarse_attr_baseline_fullref",
        "coarse_geom_ce_only",
        "coarse_geom_ce_load",
        "coarse_geom_ce_sameclass",
        "coarse_geom_ce_hardneg",
        "coarse_geom_ce_spatial",
        "coarse_geom_combined_light",
        "coarse_geom_combined_longer",
    ],
)
def test_optimization_coarse_yaml_loads(name: str) -> None:
    p = OPT_DIR / f"{name}.yaml"
    cfg = load_yaml_config(p, base_dir=ROOT)
    assert cfg.get("checkpoint_dir") == "outputs/checkpoints_stage1_opt"
    assert cfg.get("device") == "cuda"
    dcfg = load_yaml_config(ROOT / cfg["dataset_config"], base_dir=ROOT)
    proc = ROOT / dcfg.get("processed_dir", "data/processed")
    if not proc.is_absolute():
        proc = ROOT / proc
    assert (proc / "train_manifest.jsonl").is_file(), f"missing train manifest under {proc}"


def test_eval_coarse_optimization_yaml_loads() -> None:
    cfg = load_yaml_config(ROOT / "configs/eval/coarse_optimization.yaml", base_dir=ROOT)
    assert len(cfg.get("experiments", [])) >= 7
