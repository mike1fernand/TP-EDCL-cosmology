#!/usr/bin/env python3
"""
validate_patch.py
=================

Referee-grade sanity check for a unified diff patch:

- Parses each hunk header @@ -a_start,a_count +b_start,b_count @@
- Counts actual old/new lines in the hunk body
- Verifies they match the header counts exactly
- Fails fast with a clear diagnostic if not.

This prevents wasted Colab time on malformed patches.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')

def _hunk_counts(hunk_lines: List[str]) -> Tuple[int,int,List[Tuple[int,str]]]:
    old = 0
    new = 0
    bad: List[Tuple[int,str]] = []
    for i, line in enumerate(hunk_lines):
        if line == "":
            # Empty line without a diff marker is invalid in unified diffs.
            bad.append((i, line))
            continue
        c = line[0]
        if c == " ":
            old += 1
            new += 1
        elif c == "+":
            new += 1
        elif c == "-":
            old += 1
        elif c == "\\":
            # "\ No newline at end of file"
            continue
        else:
            bad.append((i, line))
    return old, new, bad

def validate(patch_path: Path) -> int:
    txt = patch_path.read_text(encoding="utf-8", errors="strict")
    lines = txt.splitlines()
    current_file = None
    i = 0
    n_hunks = 0
    errors = 0

    while i < len(lines):
        l = lines[i]
        if l.startswith("--- "):
            # file header
            if i+1 < len(lines) and lines[i+1].startswith("+++ "):
                current_file = lines[i+1].split()[1]
            i += 2
            continue

        m = HUNK_RE.match(l)
        if m:
            n_hunks += 1
            a_start = int(m.group(1))
            a_count = int(m.group(2) or "1")
            b_start = int(m.group(3))
            b_count = int(m.group(4) or "1")

            # gather hunk body
            j = i + 1
            body: List[str] = []
            while j < len(lines) and (not lines[j].startswith("@@ ")) and (not lines[j].startswith("--- ")):
                body.append(lines[j])
                j += 1

            old, new, bad = _hunk_counts(body)
            if bad:
                errors += 1
                print(f"[FAIL] {patch_path.name}: invalid diff markers in {current_file} hunk at patch line {i+1}", file=sys.stderr)
                for bi, bl in bad[:5]:
                    print(f"       bad[{bi}]={bl!r}", file=sys.stderr)

            if old != a_count or new != b_count:
                errors += 1
                print(f"[FAIL] {patch_path.name}: hunk count mismatch in {current_file}", file=sys.stderr)
                print(f"       header: -{a_start},{a_count} +{b_start},{b_count}", file=sys.stderr)
                print(f"       actual: old={old} new={new}", file=sys.stderr)
                print(f"       patch line: {i+1}", file=sys.stderr)

            i = j
            continue

        i += 1

    if n_hunks == 0:
        print(f"[FAIL] {patch_path}: no hunks found. Is this a unified diff?", file=sys.stderr)
        return 2

    if errors:
        print(f"[SUMMARY] {errors} error(s) found across {n_hunks} hunks.", file=sys.stderr)
        return 2

    print(f"[OK] Patch hunks are self-consistent: {n_hunks} hunks.")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patch", help="Path to a unified diff patch file")
    args = ap.parse_args()
    return validate(Path(args.patch))

if __name__ == "__main__":
    raise SystemExit(main())
