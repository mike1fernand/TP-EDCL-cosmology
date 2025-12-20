#!/usr/bin/env python3
"""check_no_doublecount_sh0es.py

Referee-facing guardrail: prevent accidental SH0ES double-counting.

Logic (conservative, name-based; no assumptions about Cobaya internals):
- If *any* likelihood key indicates SH0ES calibration (case-insensitive substring 'sh0es')
  AND a separate SH0ES likelihood is also present, we fail.

This catches the most common mistake:
  pantheonplus.PantheonPlusSH0ES  +  sh0es.SH0ES

Usage:
  python3 check_no_doublecount_sh0es.py path/to/config.yaml
"""

from __future__ import annotations
import sys
import yaml

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
    keys_lower = [k.lower() for k in keys]

    # Presence flags
    has_separate_sh0es = any(k.lower().startswith("sh0es") or k.lower().endswith(".sh0es") or "sh0es" == k.lower() for k in keys)
    has_any_sh0es_tagged = any("sh0es" in k for k in keys_lower)

    # If a separate SH0ES likelihood exists, any other SH0ES-tagged likelihood is suspicious.
    if has_separate_sh0es:
        # allow the separate sh0es likelihood itself; flag any *other* key containing 'sh0es'
        offenders = [k for k in keys if ("sh0es" in k.lower() and not (k.lower().startswith("sh0es") or k.lower().endswith(".sh0es")))]
        if offenders:
            print("[FAIL] Potential SH0ES double-counting detected.")
            print("  Separate SH0ES likelihood present AND other SH0ES-tagged likelihood(s) present:")
            for k in offenders:
                print("   -", k)
            print("  Action: use PantheonPlus (no SH0ES-embedded calibration) when also using sh0es.SH0ES.")
            return 2

    print("[PASS] No obvious SH0ES double-counting detected.")
    print("Likelihood keys:")
    for k in keys:
        print(" -", k)
    if has_any_sh0es_tagged:
        print("Note: One or more likelihood keys contain 'sh0es'. Ensure this is intended and documented.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
