"""Disk-backed cache for parser outputs under data/parser_cache/."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.base import BaseParser

log = logging.getLogger(__name__)


class CachedParser(BaseParser):
    def __init__(self, inner: BaseParser, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, raw_text: str) -> Path:
        h = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def parse(self, raw_text: str) -> ParsedUtterance:
        path = self._key_path(raw_text)
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return ParsedUtterance.model_validate(data)
        out = self.inner.parse(raw_text)
        with path.open("w", encoding="utf-8") as f:
            f.write(out.model_dump_json(indent=2))
        log.debug("Cached parser result for new utterance hash=%s", path.stem[:12])
        return out
