#!/usr/bin/env python3
"""
One-command reproduction of Track-0 and Tier-B artifacts.

This is designed to be the single entrypoint a referee (or you) can run after
installing minimal dependencies (numpy, matplotlib).

It does NOT attempt to run Tier-A cosmology chains (CLASS/Cobaya), since those require
external downloads and the patched CLASS build. See cosmology/README.md and colab notebooks.
"""
from __future__ import annotations

import os
import sys
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)

def main() -> None:
    # Ensure consistent working dir
    # Track-0
    run([sys.executable, os.path.join("track0", "make_fig_kernel_consistency.py")])
    # Tier-B
    run([sys.executable, os.path.join("scripts", "run_all_tierB.py")])
    print("All Track-0 + Tier-B artifacts generated under paper_artifacts/")

if __name__ == "__main__":
    main()
