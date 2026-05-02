from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rag3d.utils.env import repo_root


def load_yaml_config(path: Path | str, base_dir: Path | None = None) -> dict[str, Any]:
    """Load YAML; if key `extends` is present, shallow-merge base file first."""
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    extends = cfg.pop("extends", None)
    if extends:
        parent = p.parent / f"{extends}.yaml"
        if not parent.is_file():
            parent = Path(extends)
            if not parent.is_file():
                raise FileNotFoundError(f"extends target not found: {extends}")
        base = load_yaml_config(parent)
        merged = deep_merge(base, cfg)
        return merged
    return cfg


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def configs_dir() -> Path:
    return repo_root() / "configs"
