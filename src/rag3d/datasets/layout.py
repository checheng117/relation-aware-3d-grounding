"""Expected on-disk layout for ReferIt3D / ScanNet-style experiments (BYO data)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag3d.datasets.scannet_objects import resolve_aggregation_path


@dataclass(frozen=True)
class ReferIt3DRawLayout:
    """Documented directory layout under `raw_root` (no downloads — user places files locally)."""

    raw_root: Path
    scans_subdir: str = "scans"
    annotations_subdir: str = "annotations"

    @property
    def scans_dir(self) -> Path:
        return self.raw_root / self.scans_subdir

    @property
    def annotations_dir(self) -> Path:
        return self.raw_root / self.annotations_subdir

    def aggregation_path(self, scene_id: str) -> Path:
        return self.scans_dir / scene_id / f"{scene_id}_vh_clean_aggregation.json"


def describe_expected_layout() -> str:
    return """
Expected layout under your configured raw_root (e.g. data/raw/referit3d):

  raw_root/
    scans/
      scene0000_00/
        scene0000_00_vh_clean_aggregation.json   # ScanNet-style instance groups
      scene0001_00/
        ...
    annotations/
      train.csv    # optional: one of split-specific CSVs
      val.csv
      test.csv
      # OR a single utterances.csv with a `split` column (train|val|test)

CSV columns (minimal):
  scene_id, utterance, target_object_id
Optional:
  relation_type_gold, utterance_id, split

target_object_id must match `objectId` from the scene aggregation (string or int in file).

Alternatively, place a fully prepared JSONL manifest at data/processed/*.jsonl and skip CSV build
(see prepare_data.py --import-jsonl).
""".strip()


def validate_scans_dir(scans_dir: Path) -> tuple[int, list[str]]:
    """Return (count of scenes with aggregation file, list of scene ids found)."""
    if not scans_dir.is_dir():
        return 0, []
    scenes: list[str] = []
    for p in sorted(scans_dir.iterdir()):
        if not p.is_dir():
            continue
        sid = p.name
        if resolve_aggregation_path(p, sid) is not None:
            scenes.append(sid)
    return len(scenes), scenes
