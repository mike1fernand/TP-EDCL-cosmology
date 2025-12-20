#!/usr/bin/env python3
"""
Render Cobaya YAML templates by substituting __CLASS_PATH__ and __OUTPUT_DIR__.

This avoids "assumptions" about where CLASS is installed and where chains should go.
"""
from __future__ import annotations

import argparse
import os
import pathlib


def render_file(src: str, dst: str, class_path: str, output_dir: str) -> None:
    with open(src, "r", encoding="utf-8") as f:
        s = f.read()
    s = s.replace("__CLASS_PATH__", class_path)
    s = s.replace("__OUTPUT_DIR__", output_dir)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-path", required=True, help="Path to the CLASS root directory (containing classy.py build).")
    ap.add_argument("--out-root", required=True, help="Directory where chain outputs will be written.")
    ap.add_argument("--templates-dir", default=os.path.join("cosmology", "cobaya"),
                    help="Directory containing *.yaml.in templates.")
    args = ap.parse_args()

    templates_dir = pathlib.Path(args.templates_dir)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for template in sorted(templates_dir.glob("*.yaml.in")):
        name = template.name.replace(".yaml.in", ".yaml")
        dst = templates_dir / name  # write next to template for convenience
        output_dir = str(out_root / name.replace(".yaml", ""))
        render_file(str(template), str(dst), args.class_path, output_dir)
        print(f"Wrote {dst} (output: {output_dir})")


if __name__ == "__main__":
    main()
