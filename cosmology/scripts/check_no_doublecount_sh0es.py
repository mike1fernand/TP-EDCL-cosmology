#!/usr/bin/env python3
"""check_no_doublecount_sh0es.py

Referee-facing guardrail: prevent accidental SH0ES/H0 double-counting.

What it detects (conservative):
- If an explicit H0 likelihood is present (e.g., H0.riess2020 or sh0es.*),
  then fail if any other likelihood key OR any string-valued option contains 'sh0es'
  (common marker for SH0ES-embedded SN variants or dataset filenames).

Usage:
  python3 check_no_doublecount_sh0es.py path/to/config.yaml
"""

from __future__ import annotations

import sys
from typing import Any

import yaml


def _contains(obj: Any, sub: str) -> bool:
    sub = sub.lower()
    if isinstance(obj, str):
        return sub in obj.lower()
    if isinstance(obj, dict):
        return any(_contains(v, sub) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains(v, sub) for v in obj)
    return False


def _is_explicit_h0_key(k: str) -> bool:
    kl = k.strip().lower()
    if kl.startswith("h0."):
        return True
    if kl == "sh0es" or kl.startswith("sh0es."):
        return True
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 check_no_doublecount_sh0es.py <cobaya_yaml>")
        return 2

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    like = cfg.get("likelihood", {})
    if not isinstance(like, dict) or not like:
        print("[FAIL] No likelihood dictionary found in YAML.")
        return 2

    keys = list(like.keys())
    h0_keys = [k for k in keys if _is_explicit_h0_key(k)]

    if len(h0_keys) >= 2:
        print("[FAIL] Multiple explicit H0/SH0ES likelihoods present (likely double counting):")
        for k in h0_keys:
            print(" -", k)
        return 2

    has_h0 = len(h0_keys) == 1
    offenders = []

    if has_h0:
        for k in keys:
            if k == h0_keys[0]:
                continue
            if "sh0es" in k.lower():
                offenders.append((k, "likelihood key contains 'sh0es'"))
                continue
            if _contains(like.get(k), "sh0es"):
                offenders.append((k, "likelihood options contain 'sh0es'"))

    if offenders:
        print("[FAIL] Potential SH0ES/H0 double-counting detected.")
        print("Explicit H0 likelihood:", h0_keys[0])
        print("Other likelihood(s) appear SH0ES-embedded:")
        for k, why in offenders:
            print(f" - {k} ({why})")
        print("Action:")
        print(" - Use an unanchored SN likelihood when also using an explicit H0 likelihood, OR")
        print(" - Remove the explicit H0 likelihood if SN is already SH0ES-anchored.")
        return 2

    print("[PASS] No obvious SH0ES/H0 double-counting detected by name/config scan.")
    print("Likelihood keys:")
    for k in keys:
        print(" -", k)

    if has_h0:
        print("Note: This script cannot prove SN is unanchored unless SH0ES markers appear in config strings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
