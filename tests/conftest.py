"""Test env defaults (avoid fragile CUDA preload on some CI / driver setups)."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
