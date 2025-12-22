#!/usr/bin/env python3
"""Convenience runner for Track‑0 kernel consistency.

Referee-friendly single command:

    python track0/run_track0_kernel_consistency.py

This script ensures the repository root is on `sys.path` and then delegates to
`track0/make_fig_kernel_consistency.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from track0.make_fig_kernel_consistency import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
