#!/usr/bin/env python3
"""
No-assumptions Cobaya component discovery.

A referee-safe workflow requires verifying the *actual installed* Cobaya component keys
for likelihoods, theory backends, and samplers, because these vary across installations.

This script tries multiple methods:
1) If the `cobaya` CLI is available, run:
      cobaya info likelihood
      cobaya info theory
      cobaya info sampler
2) If the CLI is not available but the Python module is importable, attempt lightweight introspection.
3) Otherwise, print clear instructions.

Outputs:
  - cosmology/paper_artifacts/cobaya_components.txt
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime


OUT_DIR = os.path.join("cosmology", "paper_artifacts")
OUT_PATH = os.path.join(OUT_DIR, "cobaya_components.txt")


def _run_cmd(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    lines.append(f"# Cobaya component discovery ({datetime.utcnow().isoformat()}Z)")
    lines.append("")

    cobaya_cli = shutil.which("cobaya")
    if cobaya_cli:
        lines.append(f"Found cobaya CLI at: {cobaya_cli}")
        for what in ["likelihood", "theory", "sampler"]:
            lines.append("")
            lines.append(f"## cobaya info {what}")
            lines.append("```")
            lines.append(_run_cmd(["cobaya", "info", what]))
            lines.append("```")
    else:
        lines.append("Cobaya CLI not found on PATH.")
        lines.append("")

    # Try python import (optional)
    try:
        import cobaya  # noqa: F401
        lines.append("Cobaya Python module import: OK")
        lines.append(f"Python: {sys.version}")
        try:
            import cobaya
            lines.append(f"Cobaya version: {getattr(cobaya, '__version__', 'unknown')}")
        except Exception:
            pass
    except Exception as e:
        lines.append(f"Cobaya Python module import: FAILED ({type(e).__name__}: {e})")
        lines.append("")
        lines.append("To install in Colab:")
        lines.append("  pip install cobaya==3.6")
        lines.append("Then rerun this script, and run cobaya-install for the likelihoods you need.")
        lines.append("")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_PATH}")
    print("If the 'cobaya info' output does not list the likelihood keys used in the paper YAML,")
    print("do NOT guess: adjust YAML keys to match the installed registry or install the correct components.")


if __name__ == "__main__":
    main()
