#!/usr/bin/env python3
"""
run_tiera1_lateonly_suite.py

Tier-A1 "late-time Cobaya" suite runner for the Two-Perspective / EDCL project.

Design goals:
- ZERO hand-typed likelihood guessing: always run cobaya-install from the rendered YAMLs.
- Hard separation of LCDM vs EDCL CLASS inputs.
- Deterministic CLASS checkout: prefer v3.3.4 if available; else highest semver tag.
- Fail-fast on build/patch/preflight; forensic logs for every command.
- End-to-end bundle zip suitable for a referee (manifest + logs + YAMLs + updated YAMLs + validator reports).

What it runs (late-time only):
  1) LCDM late-only (BAO+SN+H0)
  2) EDCL late-only (BAO+SN+H0)
  3) EDCL late-only (BAO+SN only; no explicit H0)  [no-H0 collapse test]

It also runs Tier-A0 preflight scripts:
  cosmology/scripts/smoke_test_classy_edcl.py
  cosmology/scripts/preflight_tiera_background.py

And validates outputs:
  cosmology/scripts/validate_tiera1_lateonly_results.py

Usage (Colab or local):
  python3 cosmology/scripts/run_tiera1_lateonly_suite.py --profile iterate   # (alias: smoke)
  python3 cosmology/scripts/run_tiera1_lateonly_suite.py --profile referee

Outputs:
  <WORKDIR>/manifest.json
  <WORKDIR>/logs/*.log
  <WORKDIR>/yamls/*.yaml (+ *.updated.yaml created by cobaya-install)
  <WORKDIR>/chains/*
  <WORKDIR>/results_summary.json
  <WORKDIR>/results_report.md
  <WORKDIR>/bundle_edcl_tiera1.zip

Notes for Colab:
- You must run this from the repo root (folder containing cosmology/).
- Internet access is required to clone CLASS and to install Cobaya likelihood data.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile
from typing import Dict, Any, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


# ----------------------------
# Utilities
# ----------------------------
def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _is_colab() -> bool:
    if os.path.isdir("/content"):
        try:
            import google.colab  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False
    return False


def _print_cmd(args: List[str]) -> str:
    # For printing only; do not execute via shell
    out = []
    for a in args:
        if re.fullmatch(r"[A-Za-z0-9_/@%+=:,.\-]+", a):
            out.append(a)
        else:
            out.append("'" + a.replace("'", "'\"'\"'") + "'")
    return " ".join(out)


def run_cmd(
    args: List[str],
    *,
    cwd: Optional[str] = None,
    log_path: Optional[str] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    cmd_str = _print_cmd(args)
    print("\n$ " + cmd_str)
    p = subprocess.run(
        args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = p.stdout or ""
    print(out)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(out)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {p.returncode}): {cmd_str}\nSee log: {log_path}"
        )
    return p


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_repo_root(start: str) -> str:
    p = pathlib.Path(start).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "cosmology" / "cobaya").is_dir():
            return str(parent)
    raise RuntimeError(
        "Could not locate repo root (expected cosmology/cobaya/). Run from within the repo."
    )


def parse_tag_semver(tag: str) -> Optional[Tuple[int, int, int]]:
    m = re.fullmatch(r"v(\d+)\.(\d+)(?:\.(\d+))?", tag.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def choose_class_tag(tags: List[str]) -> Tuple[str, str]:
    tags = [t.strip() for t in tags if t.strip()]
    if "v3.3.4" in tags:
        return "v3.3.4", "preferred v3.3.4 present"
    sem = []
    for t in tags:
        v = parse_tag_semver(t)
        if v is not None:
            sem.append((v, t))
    if not sem:
        raise RuntimeError("No semver-like tags found in CLASS repo (expected vMAJOR.MINOR[.PATCH]).")
    sem.sort()
    return sem[-1][1], "v3.3.4 not present; chose highest available semver"


def _resolve_updated_yaml(path: str) -> str:
    """If <path>.updated.yaml exists, use it; else return original path."""
    p = pathlib.Path(path)
    cand = p.with_name(p.stem + ".updated.yaml")
    cand2 = p.with_name(p.stem + ".updated.yml")
    if cand.exists():
        return str(cand)
    if cand2.exists():
        return str(cand2)
    return path


# ----------------------------
# Main runner
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="iterate", choices=["iterate", "smoke", "referee"],
                    help="Run profile. iterate=fast (alias: smoke); referee=paper-grade.")
    ap.add_argument("--work-dir", default="", help="Working directory (default: timestamped under /content or repo root).")
    ap.add_argument("--skip-apt", action="store_true", help="Skip apt-get installs (local envs).")
    ap.add_argument("--skip-pip", action="store_true", help="Skip pip installs (deps already present).")
    ap.add_argument("--skip-class-build", action="store_true", help="Skip CLASS clone/patch/build (requires --class-path).")
    ap.add_argument("--class-path", default="", help="Path to existing CLASS repo (class_public) if --skip-class-build is set.")
    ap.add_argument("--mcmc-max-samples", type=int, default=0,
                    help="Override sampler.mcmc.max_samples in rendered YAMLs (0=use template defaults).")
    ap.add_argument("--skip-cobaya-install", action="store_true", help="Skip cobaya-install (assumes datasets already installed).")
    ap.add_argument("--skip-mcmc", action="store_true", help="Skip cobaya-run MCMC (still runs --test and validator may fail on missing chains).")
    ap.add_argument("--no-validate", action="store_true", help="Skip validator step at end.")
    ap.add_argument("--self-test-only", action="store_true", help="Only run lint-style self checks (no CLASS/Cobaya).")
    args = ap.parse_args()

    # Robust: locate repo root from this script's location (not the current working directory)
    repo_root = find_repo_root(str(pathlib.Path(__file__).resolve().parent))
    print("[INFO] Repo root:", repo_root)

    # Profile defaults (runner-side)
    default_max_samples = 20000 if args.profile in ("iterate", "smoke") else 100000

    stamp = _utc_stamp()
    if args.work_dir:
        work = args.work_dir
    else:
        work = f"/content/edcl_tiera1_{stamp}" if _is_colab() else os.path.join(repo_root, f"edcl_tiera1_{stamp}")

    logs = os.path.join(work, "logs")
    yamls_dir = os.path.join(work, "yamls")
    chains_dir = os.path.join(work, "chains")
    pkgs_dir = os.path.join(work, "cobaya_packages")
    os.makedirs(logs, exist_ok=True)
    os.makedirs(yamls_dir, exist_ok=True)
    os.makedirs(chains_dir, exist_ok=True)
    os.makedirs(pkgs_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        "timestamp_utc": stamp,
        "profile": args.profile,
        "repo_root": repo_root,
        "steps": [],
    }

    def step(name: str, fn):
        print("\n" + "=" * 80)
        print("STEP:", name)
        print("=" * 80)
        rec = {"name": name, "ok": False}
        try:
            fn()
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = str(e)
            print("\n[FAIL]", e)
            raise
        finally:
            manifest["steps"].append(rec)

    if args.self_test_only:
        # We rely on scripts/lint_pack.py for the deterministic self-test suite.
        lint = os.path.join(repo_root, "scripts", "lint_pack.py")
        run_cmd([sys.executable, lint], cwd=repo_root, log_path=os.path.join(logs, "00_lint_pack.log"))
        print("[OK] self-test-only complete.")
        return 0

    def install_deps():
        if not args.skip_apt:
            run_cmd(["apt-get", "update", "-qq"], log_path=os.path.join(logs, "01_apt_update.log"))
            run_cmd(["apt-get", "install", "-y", "-qq",
                     "git", "patch", "build-essential", "gfortran", "python3-dev", "zip", "unzip"],
                    log_path=os.path.join(logs, "02_apt_install.log"))
        else:
            print("[INFO] Skipping apt-get (per flag).")

        if not args.skip_pip:
            run_cmd([sys.executable, "-m", "pip", "-q", "install", "--upgrade", "pip"], log_path=os.path.join(logs, "03_pip_upgrade.log"))
            run_cmd([sys.executable, "-m", "pip", "-q", "install", "numpy", "scipy", "matplotlib", "cython", "pyyaml"],
                    log_path=os.path.join(logs, "04_pip_scientific.log"))
            run_cmd([sys.executable, "-m", "pip", "-q", "install", "cobaya==3.6"],
                    log_path=os.path.join(logs, "05_pip_cobaya.log"))
        else:
            print("[INFO] Skipping pip installs (per flag).")

    step("Install dependencies", install_deps)

    class_dir = os.path.join(work, "class_public")
    patch_path = os.path.join(repo_root, "cosmology", "patches", "class_edcl.patch")
    manifest["patch_md5"] = md5_file(patch_path)

    def build_class():
        nonlocal class_dir
        if args.skip_class_build:
            if not args.class_path:
                raise RuntimeError("--skip-class-build requires --class-path /path/to/class_public")
            class_dir = args.class_path
            print("[INFO] Using existing CLASS path:", class_dir)
            return

        run_cmd(["git", "clone", "https://github.com/lesgourg/class_public.git", class_dir],
                log_path=os.path.join(logs, "10_git_clone_class.log"))

        run_cmd(["git", "fetch", "--tags", "--force"], cwd=class_dir, log_path=os.path.join(logs, "11_git_fetch_tags.log"))
        tags_txt = run_cmd(["git", "tag", "-l", "v*"], cwd=class_dir, log_path=os.path.join(logs, "12_git_tags.log")).stdout
        chosen, reason = choose_class_tag(tags_txt.splitlines())
        manifest["class_tag_chosen"] = chosen
        manifest["class_tag_reason"] = reason

        run_cmd(["git", "checkout", "-f", chosen], cwd=class_dir, log_path=os.path.join(logs, "13_git_checkout.log"))
        manifest["class_commit"] = run_cmd(["git", "rev-parse", "HEAD"], cwd=class_dir, log_path=os.path.join(logs, "14_git_commit.log")).stdout.strip()
        manifest["class_describe"] = run_cmd(["git", "describe", "--tags", "--always", "--dirty"], cwd=class_dir, log_path=os.path.join(logs, "15_git_describe.log")).stdout.strip()

        val_script = os.path.join(repo_root, "cosmology", "scripts", "validate_patch.py")
        run_cmd([sys.executable, val_script, patch_path], cwd=repo_root, log_path=os.path.join(logs, "16_validate_patch.log"))

        run_cmd(["patch", "--dry-run", "-p1", "-i", patch_path], cwd=class_dir, log_path=os.path.join(logs, "17_patch_dryrun.log"))
        run_cmd(["patch", "-p1", "-i", patch_path], cwd=class_dir, log_path=os.path.join(logs, "18_patch_apply.log"))

        run_cmd(["make", "-j2"], cwd=class_dir, log_path=os.path.join(logs, "20_make.log"))
        run_cmd(["make", "classy"], cwd=class_dir, log_path=os.path.join(logs, "21_make_classy.log"))

    step("Clone/patch/build CLASS", build_class)

    def run_preflight():
        art = os.path.join(repo_root, "cosmology", "paper_artifacts")
        # Keep previous artifacts but avoid mixing old/new within same run.
        os.makedirs(art, exist_ok=True)

        smoke = os.path.join(repo_root, "cosmology", "scripts", "smoke_test_classy_edcl.py")
        pre = os.path.join(repo_root, "cosmology", "scripts", "preflight_tiera_background.py")

        run_cmd([sys.executable, smoke, "--class-path", class_dir], cwd=repo_root, log_path=os.path.join(logs, "30_smoke_test.log"))
        run_cmd(
            [
                sys.executable,
                pre,
                "--class-path",
                class_dir,
                "--alpha_R",
                "0.11824",
                "--log10_l0",
                "-20.908",
                "--kappa_tick",
                "0.08333333333333333",
                "--c4",
                "0.06",
                "--zeta",
                "0.5",
                "--ai",
                "1e-4",
                "--kernel",
                "exp",
            ],
            cwd=repo_root,
            log_path=os.path.join(logs, "31_preflight.log"),
        )

    step("Tier-A0 preflight (patched CLASS background)", run_preflight)

    def render_yamls():
        render = os.path.join(repo_root, "cosmology", "scripts", "render_yamls.py")
        run_cmd([sys.executable, render, "--class-path", class_dir, "--out-root", chains_dir],
                cwd=repo_root, log_path=os.path.join(logs, "40_render_yamls.log"))

        cob = os.path.join(repo_root, "cosmology", "cobaya")
        for name in ["lcdm_lateonly.yaml", "edcl_cosmo_lateonly.yaml", "edcl_cosmo_lateonly_no_sh0es.yaml"]:
            src = os.path.join(cob, name)
            if not os.path.exists(src):
                raise RuntimeError(f"Expected rendered YAML not found: {src}")
            shutil.copy2(src, os.path.join(yamls_dir, name))

        # Default run length by profile if not overridden
        max_samples = int(args.mcmc_max_samples or default_max_samples)
        if max_samples > 0:
            if yaml is None:
                raise RuntimeError("PyYAML required to edit max_samples (pip install pyyaml).")
            for name in ["lcdm_lateonly.yaml", "edcl_cosmo_lateonly.yaml", "edcl_cosmo_lateonly_no_sh0es.yaml"]:
                p = os.path.join(yamls_dir, name)
                d = yaml.safe_load(open(p, "r", encoding="utf-8"))
                d.setdefault("sampler", {}).setdefault("mcmc", {})["max_samples"] = max_samples
                with open(p, "w", encoding="utf-8") as f:
                    yaml.safe_dump(d, f, sort_keys=False)

    step("Render Tier-A1 late-only YAMLs", render_yamls)

    def run_guard():
        guard = os.path.join(repo_root, "cosmology", "scripts", "check_no_doublecount_sh0es.py")
        run_cmd([sys.executable, guard, os.path.join(yamls_dir, "edcl_cosmo_lateonly.yaml")],
                cwd=repo_root, log_path=os.path.join(logs, "45_guard_no_doublecount.log"))

    step("Run SH0ES double-count guard", run_guard)

    def cobaya_install():
        if args.skip_cobaya_install:
            print("[INFO] Skipping cobaya-install (per flag).")
            return
        for name in ["lcdm_lateonly.yaml", "edcl_cosmo_lateonly.yaml", "edcl_cosmo_lateonly_no_sh0es.yaml"]:
            y = os.path.join(yamls_dir, name)
            run_cmd(["cobaya-install", y, "-p", pkgs_dir],
                    cwd=repo_root, log_path=os.path.join(logs, f"50_cobaya_install_{name}.log"))

    step("cobaya-install datasets for Tier-A1", cobaya_install)

    def cobaya_test_suite():
        # Prefer updated YAMLs after installation to avoid key drift.
        for name in ["lcdm_lateonly.yaml", "edcl_cosmo_lateonly.yaml", "edcl_cosmo_lateonly_no_sh0es.yaml"]:
            y = _resolve_updated_yaml(os.path.join(yamls_dir, name))
            run_cmd(["cobaya-run", y, "--test", "-p", pkgs_dir],
                    cwd=repo_root, log_path=os.path.join(logs, f"58_cobaya_test_{os.path.basename(y)}.log"))

    step("cobaya-run --test (initialisation-only)", cobaya_test_suite)

    def cobaya_run_suite():
        if args.skip_mcmc:
            print("[INFO] Skipping MCMC runs (per flag).")
            return
        for name in ["lcdm_lateonly.yaml", "edcl_cosmo_lateonly.yaml", "edcl_cosmo_lateonly_no_sh0es.yaml"]:
            y = _resolve_updated_yaml(os.path.join(yamls_dir, name))
            run_cmd(["cobaya-run", y, "-p", pkgs_dir],
                    cwd=repo_root, log_path=os.path.join(logs, f"60_cobaya_run_{os.path.basename(y)}.log"))

    step("Run Tier-A1 late-only suite (Cobaya)", cobaya_run_suite)

    def run_validator():
        if args.no_validate:
            print("[INFO] Skipping validator (per flag).")
            return
        val = os.path.join(repo_root, "cosmology", "scripts", "validate_tiera1_lateonly_results.py")
        p = run_cmd([sys.executable, val, "--workdir", work, "--profile", args.profile],
                    cwd=repo_root, log_path=os.path.join(logs, "70_validate_tiera1.log"), check=False)
        manifest["validator_exit_code"] = p.returncode


        # Copy validation artifacts to /content for easy download in Colab
        try:
            if os.path.isdir("/content"):
                for fn in ["results_report.md", "results_summary.json"]:
                    src = os.path.join(work, fn)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join("/content", fn))
                print("[INFO] Copied validation artifacts to /content/results_report.md and /content/results_summary.json")
        except Exception as e:
            print(f"[WARN] Could not copy validation artifacts to /content: {e}")


        if p.returncode == 0:
            print("[OK] Tier-A1 validation: PASS")
        elif p.returncode == 1:
            print("[WARN] Tier-A1 validation: WARN (see results_report.md)")
        else:
            print(f"[FAIL] Tier-A1 validation: FAIL (exit code {p.returncode}). See logs/70_validate_tiera1.log and results_report.md")


    step("Validate Tier-A1 outputs", run_validator)

    def bundle():
        man_path = os.path.join(work, "manifest.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        bundle_path = os.path.join(work, "bundle_edcl_tiera1.zip")
        if os.path.exists(bundle_path):
            os.remove(bundle_path)

        include_paths = [
            "manifest.json",
            "logs",
            "yamls",
            "chains",
            "results_summary.json",
            "results_report.md",
        ]

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for rel in include_paths:
                p = os.path.join(work, rel)
                if not os.path.exists(p):
                    continue
                if os.path.isdir(p):
                    for dp, _, fn in os.walk(p):
                        for n in fn:
                            full = os.path.join(dp, n)
                            z.write(full, arcname=os.path.relpath(full, work))
                else:
                    z.write(p, arcname=rel)

            # Also include repo-side cosmology artifacts (preflight plots, AI probe logs, validation spec, etc.)
            # We include the cosmology/paper_artifacts tree, not the root paper_artifacts.
            cpa = os.path.join(repo_root, "cosmology", "paper_artifacts")
            if os.path.exists(cpa):
                for dp, _, fn in os.walk(cpa):
                    for n in fn:
                        full = os.path.join(dp, n)
                        arc = os.path.join("repo", os.path.relpath(full, repo_root))
                        z.write(full, arcname=arc)

        print("\nCreated bundle:", bundle_path)
        print("[INFO] Bundle location:", bundle_path)
        print("[INFO] In Colab: open the Files pane (left sidebar) and download the bundle from that path.")

    step("Bundle outputs", bundle)

    print("\nAll steps completed.")
    print("Workdir:", work)

    # In referee mode, propagate validator exit code for CI-style gating.
    if args.profile == "referee":
        vcode = int(manifest.get("validator_exit_code", 0) or 0)
        return vcode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
