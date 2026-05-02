from __future__ import annotations

from pathlib import Path

from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser


def test_cached_parser_writes_once(tmp_path: Path) -> None:
    inner = HeuristicParser()
    c = CachedParser(inner, tmp_path)
    t = "the chair left of the table"
    a = c.parse(t)
    b = c.parse(t)
    assert a.target_head == b.target_head
    assert any(tmp_path.glob("*.json"))
