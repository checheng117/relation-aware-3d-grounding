from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_smoke_script_runs():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "smoke_test.py"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    subprocess.check_call([sys.executable, str(script)], cwd=str(root), env=env)
