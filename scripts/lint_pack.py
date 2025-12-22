#!/usr/bin/env python3
"""lint_pack.py

Fast, referee-grade guardrails for this repo.

Goal:
- Fail fast on the exact classes of issues that previously caused multi-day loops:
  - Old Cobaya likelihood keys / naming drift
  - EDCL vs ΛCDM YAML mixing (EDCL-only CLASS args leaking into ΛCDM templates)
  - Non-portable/invalid Cobaya CLI calls
  - YAML numeric parsing traps (notably: edcl_ai: 1e-4 parsed as a string by PyYAML)
  - Accidental shipping of __pycache__ / .pyc artifacts in the pack
- Run only deterministic checks by default (no external datasets).

Usage (from repo root):
  python scripts/lint_pack.py

Options:
  --no-tests   Skip unit tests (still runs py_compile + YAML guards).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


FAIL = 2


def _print_header(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _run(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    env = os.environ.copy()
    # Avoid littering the repo with __pycache__ during lint/test runs.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    out = (p.stdout or "")
    if p.stderr:
        out += ("\n" if out else "") + p.stderr
    return p.returncode, out.strip()


def _find_repo_root() -> Path:
    # scripts/lint_pack.py -> repo root
    return Path(__file__).resolve().parent.parent


def _list_pycache(root: Path) -> List[Path]:
    bad: List[Path] = []
    for p in root.rglob("__pycache__"):
        if p.is_dir():
            bad.append(p)
    for p in root.rglob("*.pyc"):
        if p.is_file():
            bad.append(p)
    return sorted(set(bad))


def _scan_for_patterns(root: Path, patterns: List[re.Pattern], suffixes: set[str]) -> Dict[str, List[Path]]:
    hits: Dict[str, List[Path]] = {pat.pattern: [] for pat in patterns}
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        try:
            txt = _read_text(p)
        except Exception:
            continue
        for pat in patterns:
            if pat.search(txt):
                hits[pat.pattern].append(p)
    return hits


def _require_pyyaml() -> Any:
    try:
        import yaml  # type: ignore
        return yaml
    except Exception as e:
        print("[FAIL] PyYAML is required for YAML guardrails but is not installed.")
        print("       Install with: pip install pyyaml")
        print("       Error:", e)
        raise SystemExit(FAIL)


def _yaml_guardrails(repo: Path, yaml_mod: Any) -> None:
    cobaya_dir = repo / "cosmology" / "cobaya"
    if not cobaya_dir.is_dir():
        print(f"[FAIL] Missing directory: {cobaya_dir}")
        raise SystemExit(FAIL)

    templates = sorted(cobaya_dir.glob("*.yaml.in"))
    if not templates:
        print(f"[FAIL] No YAML templates found in: {cobaya_dir}")
        raise SystemExit(FAIL)

    edcl_only_keys = {"kappa_tick", "c4", "log10_l0", "edcl_kernel", "edcl_zeta", "edcl_ai"}

    for t in templates:
        cfg = yaml_mod.safe_load(_read_text(t))
        if not isinstance(cfg, dict):
            print(f"[FAIL] YAML template did not parse to a dict: {t}")
            raise SystemExit(FAIL)

        # Basic structure presence
        try:
            extra_args = cfg["theory"]["classy"]["extra_args"]
        except Exception:
            print(f"[FAIL] Missing theory.classy.extra_args in: {t}")
            raise SystemExit(FAIL)

        if not isinstance(extra_args, dict):
            print(f"[FAIL] theory.classy.extra_args must be a dict in: {t}")
            raise SystemExit(FAIL)

        edcl_on = extra_args.get("edcl_on")
        if edcl_on not in ("yes", "no", "'yes'", "'no'"):
            # YAML loader will drop quotes; we accept only yes/no strings
            print(f"[FAIL] edcl_on must be a quoted string 'yes'/'no' in: {t}")
            print(f"       Parsed value: {edcl_on!r}")
            raise SystemExit(FAIL)

        # Enforce EDCL/LCDM separation based on filename convention
        name = t.name.lower()
        params = cfg.get("params", {})
        if not isinstance(params, dict):
            print(f"[FAIL] params must be a dict in: {t}")
            raise SystemExit(FAIL)

        if name.startswith("lcdm_"):
            if extra_args.get("edcl_on") != "no":
                print(f"[FAIL] LCDM template must set edcl_on: 'no' in: {t}")
                raise SystemExit(FAIL)

            leaked = edcl_only_keys.intersection(extra_args.keys())
            if leaked:
                print(f"[FAIL] LCDM template leaks EDCL-only CLASS args {sorted(leaked)} in: {t}")
                raise SystemExit(FAIL)

            if "alpha_R" in params:
                print(f"[FAIL] LCDM template must not include alpha_R in params: {t}")
                raise SystemExit(FAIL)

        if name.startswith("edcl_"):
            if extra_args.get("edcl_on") != "yes":
                print(f"[FAIL] EDCL template must set edcl_on: 'yes' in: {t}")
                raise SystemExit(FAIL)

            missing = [k for k in sorted(edcl_only_keys) if k not in extra_args]
            if missing:
                print(f"[FAIL] EDCL template missing EDCL-only CLASS args {missing} in: {t}")
                raise SystemExit(FAIL)

            # YAML numeric trap check
            if not isinstance(extra_args.get("edcl_ai"), float):
                print(f"[FAIL] edcl_ai must parse as float (avoid '1e-4' which parses as string) in: {t}")
                print(f"       Parsed edcl_ai: {extra_args.get('edcl_ai')!r}")
                raise SystemExit(FAIL)

            if "alpha_R" not in params:
                print(f"[FAIL] EDCL template must include alpha_R in params: {t}")
                raise SystemExit(FAIL)


def _py_compile_all(repo: Path) -> None:
    # In-memory syntax compilation (does not write .pyc / __pycache__).
    py_files = [p for p in repo.rglob("*.py") if p.is_file()]
    if not py_files:
        print("[WARN] No Python files found to compile.")
        return

    for p in py_files:
        try:
            src = _read_text(p)
            compile(src, str(p), "exec")
        except SyntaxError as e:
            print(f"[FAIL] SyntaxError in {p}: {e.msg} (line {e.lineno})")
            raise SystemExit(FAIL)
        except Exception as e:
            print(f"[FAIL] Compile failed for {p}: {type(e).__name__}: {e}")
            raise SystemExit(FAIL)


def _run_unit_tests(repo: Path) -> None:
    code, out = _run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=repo)
    if code != 0:
        print("[FAIL] Unit tests failed.")
        print(out)
        raise SystemExit(FAIL)
    print("[PASS] Unit tests: OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tests", action="store_true", help="Skip unit tests")
    ap.add_argument(
        "--strict-pyc",
        action="store_true",
        help="Fail if __pycache__/.pyc exist instead of cleaning them",
    )
    args = ap.parse_args()

    repo = _find_repo_root()

    _print_header("1) Clean/Reject __pycache__ / .pyc artifacts")
    bad_cache = _list_pycache(repo)
    if bad_cache:
        if args.strict_pyc:
            print("[FAIL] Found cached bytecode artifacts (do not commit or ship these):")
            for p in bad_cache[:50]:
                print(" -", p.relative_to(repo))
            if len(bad_cache) > 50:
                print(f" ... and {len(bad_cache)-50} more")
            print("Fix: delete these directories/files and re-run lint (or run without --strict-pyc to auto-clean).")
            return FAIL

        print("[WARN] Found __pycache__/.pyc artifacts; cleaning them now:")
        for p in bad_cache:
            rel = p.relative_to(repo)
            print(" -", rel)
            try:
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        print("[PASS] Bytecode artifacts cleaned.")
    else:
        print("[PASS] No __pycache__ / .pyc artifacts found.")

    _print_header("2) Scan for known failure patterns in text files")
    patterns = [
        re.compile(r"desi_dr2\.bao"),                # old DESI BAO key
        re.compile(r"pantheonplus\.PantheonPlus"),   # old PantheonPlus key
        re.compile(r"sh0es\.SH0ES"),                 # old SH0ES key
        re.compile(r"cobaya\s+info\s+likelihood"),  # non-portable/invalid CLI usage
        re.compile(r"^\s*edcl_ai\s*:\s*1e-4\s*$", flags=re.MULTILINE),  # YAML numeric trap
    ]
    suffixes = {".in", ".yaml", ".md", ".py", ".txt"}
    hits = _scan_for_patterns(repo, patterns, suffixes)

    # Allowlist: some patterns may appear in documentation strings intentionally.
    # We only block execution-significant occurrences.
    allowlist: Dict[str, set[str]] = {
        r"cobaya\s+info\s+likelihood": {"cosmology/scripts/discover_cobaya_components.py"},
    }

    any_fail = False
    for pat, files in hits.items():
        if files:
            allowed_paths = allowlist.get(pat, set())
            filtered: List[Path] = []
            for f in files:
                rel = str(f.relative_to(repo)).replace(os.sep, "/")
                if rel in allowed_paths:
                    continue
                filtered.append(f)
            if not filtered:
                continue

            any_fail = True
            print(f"[FAIL] Pattern found: {pat}")
            for f in filtered[:50]:
                print(" -", f.relative_to(repo))
            if len(filtered) > 50:
                print(f" ... and {len(filtered)-50} more")

    if any_fail:
        print("\nFix: remove/update these patterns (or update lint expectations if intentional).")
        return FAIL
    print("[PASS] No known failure patterns found.")

    _print_header("3) YAML guardrails (EDCL/LCDM separation + numeric traps)")
    yaml_mod = _require_pyyaml()
    _yaml_guardrails(repo, yaml_mod)
    print("[PASS] YAML guardrails: OK")

    _print_header("4) SH0ES double-count guard (name/config scan)")
    guard = repo / "cosmology" / "scripts" / "check_no_doublecount_sh0es.py"
    if guard.is_file():
        for cfg in [
            repo / "cosmology" / "cobaya" / "lcdm_lateonly.yaml.in",
            repo / "cosmology" / "cobaya" / "edcl_cosmo_lateonly.yaml.in",
        ]:
            if cfg.is_file():
                code, out = _run([sys.executable, "-B", str(guard), str(cfg)], cwd=repo)
                if code != 0:
                    print(f"[FAIL] SH0ES guard failed for {cfg.name}:")
                    print(out)
                    return FAIL
                print(out)
            else:
                print(f"[WARN] Missing expected template: {cfg}")
    else:
        print(f"[WARN] Missing guard script: {guard}")

    _print_header("5) Python compilation (in-memory compile)")
    _py_compile_all(repo)
    print("[PASS] Python compilation: OK")

    if not args.no_tests:
        _print_header("6) Unit tests (deterministic, no external datasets)")
        _run_unit_tests(repo)
    else:
        print("[SKIP] Unit tests skipped (--no-tests).")

    print("\n[PASS] Lint gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
