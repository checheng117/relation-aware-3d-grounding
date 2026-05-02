"""Logging setup with secret redaction in formatters (defensive)."""

from __future__ import annotations

import logging
import os
import re
import sys


class _RedactFilter(logging.Filter):
    _patterns = [
        re.compile(r"hf_[a-z0-9_-]{20,}", re.I),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage()
        for p in self._patterns:
            msg = p.sub("[REDACTED]", msg)
        record.msg = msg
        record.args = ()
        return True


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(_RedactFilter())
    root.addHandler(handler)
    root.setLevel(level)
    # Avoid leaking token via debug getenv dumps
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if "HF_TOKEN" in os.environ:
        # Do not log env keys in application code; this only mutes common mistakes
        pass
