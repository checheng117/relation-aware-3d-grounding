from __future__ import annotations

from abc import ABC, abstractmethod

from rag3d.datasets.schemas import ParsedUtterance


class BaseParser(ABC):
    """Structured language parser: utterance -> ParsedUtterance."""

    @abstractmethod
    def parse(self, raw_text: str) -> ParsedUtterance:
        raise NotImplementedError
