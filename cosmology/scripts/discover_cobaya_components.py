#!/usr/bin/env python3
"""discover_cobaya_components.py

Referee-safe (no-assumptions) Cobaya component discovery.

Why this exists:
- Cobaya likelihood keys are install-dependent.
- Non-portable/unsupported CLI subcommands (e.g. `cobaya info likelihood`) can waste cycles and
  produce false confidence.

This script does NOT rely on `cobaya info ...`.
Instead it records:

1) `cobaya-run` / `cobaya-install` availability and `--help` output
2) Python import + Cobaya version (if importable)
3) Best-effort listing of installed likelihood modules via Python introspection

Output:
  cosmology/paper_artifacts/cobaya_components.txt

Operational note (authoritative record):
When you run Cobaya, it writes a resolved config next to your chain outputs (e.g. `<chain_root>/*.updated.yaml`).
That file is the authoritative record of what component keys were actually used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import pkgutil
from datetime import datetime, timezone

OUT_DIR = os.path.join("cosmology", "paper_artifacts")
OUT_PATH = os.path.join(OUT_DIR, "cobaya_components.txt")


def _run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return out.strip()
    except Exception as e:
        return f"<failed to run {cmd!r}: {type(e).__name__}: {e}>"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Cobaya component discovery ({datetime.now(timezone.utc).isoformat()})")
    lines.append("")

    # Record CLI availability for the supported entry points.
    for exe in ["cobaya-run", "cobaya-install"]:
        path = shutil.which(exe)
        lines.append(f"## {exe}")
        if path:
            lines.append(f"Found: {path}")
            lines.append("--help (first ~50 lines):")
            help_txt = _run([exe, "--help"]).splitlines()
            lines.extend(help_txt[:50])
        else:
            lines.append("Not found on PATH.")
        lines.append("")

    # Python import + version
    try:
        import cobaya  # noqa: F401

        lines.append("## Python import")
        lines.append("Cobaya import: OK")
        lines.append(f"Python: {sys.version}")
        try:
            import cobaya
            lines.append(f"Cobaya __version__: {getattr(cobaya, '__version__', 'unknown')}")
        except Exception:
            pass
        lines.append("")

        # Best-effort: enumerate installed likelihood modules
        lines.append("## Installed likelihood modules (best-effort)")
        try:
            import cobaya.likelihoods as like_pkg

            prefix = "cobaya.likelihoods."
            mods: list[str] = []
            for m in pkgutil.walk_packages(like_pkg.__path__, prefix=prefix):
                name = m.name
                if any(tok in name.lower() for tok in ["pantheon", "desi", "bao", "h0", "riess", "planck"]):
                    mods.append(name)
            mods = sorted(set(mods))
            if mods:
                lines.append("Candidate likelihood module names found:")
                for name in mods:
                    lines.append(f" - {name}   (candidate key: {name[len(prefix):]})")
            else:
                lines.append("No matching likelihood modules found via pkgutil scan.")
                lines.append("This can happen if external likelihood packages are not installed.")
        except Exception as e:
            lines.append(f"Likelihood module scan failed: {type(e).__name__}: {e}")
        lines.append("")

    except Exception as e:
        lines.append("## Python import")
        lines.append(f"Cobaya import: FAILED ({type(e).__name__}: {e})")
        lines.append("")
        lines.append("Install in a fresh environment (example):")
        lines.append("  pip install cobaya==3.6")
        lines.append("")

    lines.append("## Operational guidance")
    lines.append("Authoritative component keys are recorded by Cobaya in <chain_root>/*.updated.yaml.")
    lines.append("If you are unsure about a likelihood key, do not guess; run cobaya-install on your YAML and follow the suggestions.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
