#!/usr/bin/env python3
"""
Tier-A0 (Preflight): Background-only EDCL sanity + Track-0 consistency check.

This script is intentionally "referee-grade":
- No likelihoods
- No perturbations required
- Purely checks the patched CLASS background modification against the paper's
  claimed H_TP/H_GR behavior and against Track-0 kernel-only evaluation.

Outputs (written under cosmology/paper_artifacts/ by default):
  - hubble_ratio_from_class.csv
  - fig_hubble_ratio_from_class.png
  - preflight_report.txt
  - fig_track0_vs_class.png
  - track0_vs_class_report.txt

Usage:
  python cosmology/scripts/preflight_tiera_background.py \
    --class-path /path/to/patched/class_public \
    --alpha_R 0.11824 --log10_l0 -20.908 --kappa_tick 0.0833333333333 --c4 0.06 \
    --zeta 0.5 --kernel exp

Notes:
- "kernel exp" corresponds to exp(-z/zeta) (high-z suppressed; matches high-z safety claim).
- "kernel 1mexp" corresponds to 1-exp(-z/zeta) (as written in one draft equation).
  If the manuscript equation uses 1-exp but the claim requires exp, you must resolve
  the mismatch in the manuscript. This script makes that discrepancy measurable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
import os
import sys
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt


KernelChoice = Literal["exp", "1mexp"]


def _add_class_python_path(class_path: str) -> None:
    """Add CLASS/python to sys.path (for classy import)."""
    py_dir = os.path.join(class_path, "python")
    if not os.path.isdir(py_dir):
        raise FileNotFoundError(f"CLASS python directory not found: {py_dir}")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)


def _compute_background(class_path: str, params: dict) -> dict:
    _add_class_python_path(class_path)
    from classy import Class  # type: ignore
    cosmo = Class()
    try:
        cosmo.set(params)
        cosmo.compute()
        return cosmo.get_background()
    finally:
        # Always clean up to avoid memory leak / segfault in repeated runs
        try:
            cosmo.struct_cleanup()
        except Exception:
            pass
        try:
            cosmo.empty()
        except Exception:
            pass


def _compute_distances(class_path: str, params: dict, z_samples: list[float]) -> dict:
    """Compute distance measures using classy if available.

    Returns dict with keys:
      - z: list[float]
      - DA_Mpc: list[float]  (angular diameter distance)
      - DL_Mpc: list[float]  (luminosity distance)

    If a distance cannot be computed, the corresponding value is NaN.

    Note: This is a *background-derived* preflight: no likelihoods, no perturbations.
    """
    _add_class_python_path(class_path)
    from classy import Class  # type: ignore

    cosmo = Class()
    try:
        cosmo.set(params)
        cosmo.compute()

        DA=[]
        DL=[]
        for z in z_samples:
            da_val=float("nan")
            dl_val=float("nan")
            try:
                if hasattr(cosmo, "angular_distance"):
                    da_val=float(cosmo.angular_distance(z))
                if hasattr(cosmo, "luminosity_distance"):
                    dl_val=float(cosmo.luminosity_distance(z))
                if (math.isnan(dl_val) or dl_val == 0.0) and (not math.isnan(da_val)):
                    dl_val = da_val * (1.0 + z)**2
            except Exception:
                pass
            DA.append(da_val)
            DL.append(dl_val)

        return {"z": list(z_samples), "DA_Mpc": DA, "DL_Mpc": DL}
    finally:
        try:
            cosmo.struct_cleanup()
        except Exception:
            pass
        try:
            cosmo.empty()
        except Exception:
            pass

def _extract_z_H(bg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract z and H from CLASS background output."""
    if "z" not in bg:
        raise KeyError("CLASS background dict has no key 'z'.")
    z = np.asarray(bg["z"], dtype=float)

    # CLASS uses either of these keys depending on build/config
    for k in ["H [1/Mpc]", "H [1/Mpc] "]:
        if k in bg:
            H = np.asarray(bg[k], dtype=float)
            return z, H

    # Fallback: try any key that starts with "H ["
    for k in bg.keys():
        if isinstance(k, str) and k.startswith("H ["):
            H = np.asarray(bg[k], dtype=float)
            return z, H

    raise KeyError("CLASS background dict has no recognizable H key (expected 'H [1/Mpc]').")


def _interp_at(z_grid: np.ndarray, y: np.ndarray, z0: float) -> float:
    """Interpolate y(z) at z0. Requires z_grid ascending."""
    if not (np.all(np.diff(z_grid) >= 0)):
        raise ValueError("z_grid must be ascending for interpolation.")
    return float(np.interp(z0, z_grid, y))


def _ensure_artifacts_dir(pack_root: str, outdir: str | None) -> str:
    if outdir is None:
        outdir = os.path.join(pack_root, "cosmology", "paper_artifacts")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _pack_root_from_script() -> str:
    # .../TP_EDCL_vX/cosmology/scripts/preflight_tiera_background.py -> .../TP_EDCL_vX
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_track0_kernel(pack_root: str):
    # Import track0/kernel.py without assuming installed package.
    track0_dir = os.path.join(pack_root, "track0")
    if not os.path.isdir(track0_dir):
        raise FileNotFoundError(f"track0 directory not found at: {track0_dir}")
    if pack_root not in sys.path:
        sys.path.insert(0, pack_root)
    import track0.kernel as tk  # type: ignore
    return tk


def _track0_ratio(z: np.ndarray,
                  delta0: float,
                  zeta: float,
                  ai: float,
                  kernel: KernelChoice,
                  f_norm_target: float) -> np.ndarray:
    tk = _load_track0_kernel(_pack_root_from_script())
    variant = "paper_claim_exp" if kernel == "exp" else "paper_equation_1mexp"
    norm = tk.KernelNormalization(a0=1.0, a_i=ai, zeta=zeta, f_norm_target=f_norm_target, variant=variant)
    # delta(a) in the "high-z limit" approximation used in Track-0 plots
    a = 1.0 / (1.0 + z)
    # Vectorize delta_of_a_highz_limit
    delta = np.array([tk.delta_of_a_highz_limit(float(ai_), float(delta0), norm) for ai_ in a], dtype=float)
    ratio = np.array([tk.hubble_ratio_from_delta(float(d)) for d in delta], dtype=float)
    return ratio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-path", required=True, help="Path to patched CLASS root (contains python/)")
    ap.add_argument("--alpha_R", type=float, required=True)
    ap.add_argument("--log10_l0", type=float, required=True)
    ap.add_argument("--kappa_tick", type=float, default=1.0/12.0)
    ap.add_argument("--c4", type=float, default=0.06)
    ap.add_argument("--zeta", type=float, default=0.5)
    ap.add_argument("--ai", type=float, default=1e-4)
    ap.add_argument("--kernel", choices=["exp", "1mexp"], default="exp")
    ap.add_argument("--f_norm_target", type=float, default=0.7542)
    ap.add_argument("--outdir", default=None, help="Output directory (default: cosmology/paper_artifacts)")
    ap.add_argument("--zmax_plot", type=float, default=5.0)
    args = ap.parse_args()

    pack_root = _pack_root_from_script()
    outdir = _ensure_artifacts_dir(pack_root, args.outdir)

    # Minimal baseline cosmology (fast)
    base = {
        "output": "tCl,pCl,lCl",
        "l_max_scalars": 20,
        "h": 0.67,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "tau_reio": 0.0544,
        "A_s": 2.1e-9,
        "n_s": 0.965,
    }

    # Baseline LCDM run
    lcdm = dict(base)
    lcdm["edcl_on"] = "no"

    # EDCL run
    edcl = dict(base)
    edcl.update({
        "edcl_on": "yes",
        "kappa_tick": float(args.kappa_tick),
        "c4": float(args.c4),
        "alpha_R": float(args.alpha_R),
        "log10_l0": float(args.log10_l0),
        "edcl_zeta": float(args.zeta),
        "edcl_ai": float(args.ai),
        "edcl_kernel": str(args.kernel),
    })

    # LCDM-limit check with EDCL enabled but alpha_R=0
    edcl_alpha0 = dict(edcl)
    edcl_alpha0["alpha_R"] = 0.0

    bg_lcdm = _compute_background(args.class_path, lcdm)
    bg_edcl = _compute_background(args.class_path, edcl)
    bg_alpha0 = _compute_background(args.class_path, edcl_alpha0)

    z0, H0 = _extract_z_H(bg_lcdm)
    z1, H1 = _extract_z_H(bg_edcl)
    z2, H2 = _extract_z_H(bg_alpha0)

    # Sort ascending and interpolate to common grid
    s0 = np.argsort(z0); s1 = np.argsort(z1); s2 = np.argsort(z2)
    z0a, H0a = z0[s0], H0[s0]
    z1a, H1a = z1[s1], H1[s1]
    z2a, H2a = z2[s2], H2[s2]

    H1i = np.interp(z0a, z1a, H1a)
    H2i = np.interp(z0a, z2a, H2a)

    ratio = H1i / H0a
    ratio_alpha0 = H2i / H0a

    # Key diagnostics
    r0 = _interp_at(z0a, ratio, 0.0)
    delta0_meas = r0 - 1.0
    delta0_expect = args.alpha_R * (12.0 * args.kappa_tick) * args.f_norm_target

    r2 = _interp_at(z0a, ratio, 2.0)
    r1100 = _interp_at(z0a, ratio, 1100.0) if z0a[-1] >= 1100.0 else float("nan")

    # High-z safety metric: max |ratio-1| for z>=2 up to max available
    mask_hi = z0a >= 2.0
    max_hi = float(np.max(np.abs(ratio[mask_hi] - 1.0))) if np.any(mask_hi) else float("nan")

    # LCDM limit metric: max |ratio_alpha0 - 1|
    max_lcdm_lim = float(np.max(np.abs(ratio_alpha0 - 1.0)))

    # Write CSV
    # Additional derived-observable preflight: distance ratios (CLASS-derived).
    # These help a referee assess "high-z safety" beyond H(z) alone.
    z_samples_dist = [0.5, 1.0, 2.0, 1100.0]
    dist_lcdm = _compute_distances(args.class_path, lcdm, z_samples_dist)
    dist_edcl = _compute_distances(args.class_path, edcl, z_samples_dist)

    dist_rows = []
    for z_s, da0, da1, dl0, dl1 in zip(dist_lcdm["z"], dist_lcdm["DA_Mpc"], dist_edcl["DA_Mpc"],
                                       dist_lcdm["DL_Mpc"], dist_edcl["DL_Mpc"]):
        da_ratio = float("nan") if (np.isnan(da0) or da0 == 0) else da1/da0
        dl_ratio = float("nan") if (np.isnan(dl0) or dl0 == 0) else dl1/dl0
        dist_rows.append((z_s, da0, da1, da_ratio, dl0, dl1, dl_ratio))

    dist_csv = os.path.join(outdir, "distance_ratio_from_class.csv")
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["z", "DA_lcdm_Mpc", "DA_edcl_Mpc", "DA_ratio", "DL_lcdm_Mpc", "DL_edcl_Mpc", "DL_ratio"])
        for row in dist_rows:
            w.writerow([f"{x:.12g}" if isinstance(x,(float,int)) else x for x in row])

    summary = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "alpha_R": float(args.alpha_R),
            "kappa_tick": float(args.kappa_tick),
            "c4": float(args.c4),
            "zeta": float(args.zeta),
            "ai": float(args.ai),
            "kernel": str(args.kernel),
            "f_norm_target": float(args.f_norm_target),
        },
        "delta0_measured": float(delta0_meas),
        "delta0_expected": float(delta0_expect),
        "ratio": {
            "z0": float(r0),
            "z2": float(r2),
            "z1100": (None if np.isnan(r1100) else float(r1100)),
            "max_abs_ratio_minus_1_for_z_ge_2": float(max_hi),
            "lcdm_limit_max_abs_ratio_minus_1": float(max_lcdm_lim),
        },
        "distances": {
            "z_samples": z_samples_dist,
            "DA_ratio": [float(r[3]) for r in dist_rows],
            "DL_ratio": [float(r[6]) for r in dist_rows],
        }
    }
    summary_json = os.path.join(outdir, "preflight_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(outdir, "hubble_ratio_from_class.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["z", "H_lcdm_1_per_Mpc", "H_edcl_1_per_Mpc", "ratio", "ratio_alphaR0"])
        for zi, h0i, h1i, ri, r2i in zip(z0a, H0a, H1i, ratio, ratio_alpha0):
            w.writerow([f"{zi:.8e}", f"{h0i:.8e}", f"{h1i:.8e}", f"{ri:.8e}", f"{r2i:.8e}"])

    # Plot ratio
    fig_path = os.path.join(outdir, "fig_hubble_ratio_from_class.png")
    plt.figure(figsize=(7.5, 4.5))
    # Limit plot to zmax_plot for readability
    mplot = z0a <= float(args.zmax_plot)
    plt.plot(z0a[mplot], ratio[mplot], label=f"CLASS ratio (kernel={args.kernel})")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Redshift z")
    plt.ylabel(r"$H_{\rm EDCL}(z) / H_{\Lambda{\rm CDM}}(z)$")
    plt.title("Tier-A0 preflight: background-only H(z) ratio from patched CLASS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # Track-0 compare (on same z grid for z<=zmax_plot)
    try:
        ratio_track0 = _track0_ratio(z0a, delta0_expect, args.zeta, args.ai, args.kernel, args.f_norm_target)
        diff = ratio - ratio_track0
        max_abs_diff = float(np.max(np.abs(diff[mplot])))
        cmp_fig = os.path.join(outdir, "fig_track0_vs_class.png")
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(z0a[mplot], ratio[mplot], label="CLASS")
        plt.plot(z0a[mplot], ratio_track0[mplot], linestyle="--", label="Track-0")
        plt.axhline(1.0, linestyle=":", linewidth=1)
        plt.xlabel("Redshift z")
        plt.ylabel(r"$H_{\rm EDCL}/H_{\Lambda{\rm CDM}}$")
        plt.title("Tier-A0 preflight: Track-0 vs CLASS (same kernel choice)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(cmp_fig, dpi=160)
        plt.close()

        cmp_rep = os.path.join(outdir, "track0_vs_class_report.txt")
        with open(cmp_rep, "w", encoding="utf-8") as f:
            f.write("Track-0 vs CLASS comparison (Tier-A0 preflight)\n")
            f.write("================================================\n\n")
            f.write(f"kernel: {args.kernel}\n")
            f.write(f"zeta: {args.zeta}\n")
            f.write(f"a_i: {args.ai}\n")
            f.write(f"alpha_R: {args.alpha_R}\n")
            f.write(f"kappa_tick: {args.kappa_tick}\n")
            f.write(f"f_norm_target: {args.f_norm_target}\n\n")
            f.write(f"delta0_expected = {delta0_expect:.8e}\n")
            f.write(f"max |CLASS - Track0| over 0<=z<={args.zmax_plot}: {max_abs_diff:.8e}\n")

    except Exception as e:
        # Do not hard-fail if Track-0 import is unavailable in a minimal environment.
        with open(os.path.join(outdir, "track0_vs_class_report.txt"), "w", encoding="utf-8") as f:
            f.write("Track-0 vs CLASS comparison FAILED\n")
            f.write(str(e) + "\n")

    # Report
    rep_path = os.path.join(outdir, "preflight_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("Tier-A0 preflight report (patched CLASS background-only)\n")
        f.write("=========================================================\n\n")
        f.write(f"class_path: {args.class_path}\n")
        f.write(f"kernel: {args.kernel}\n")
        f.write(f"zeta: {args.zeta}\n")
        f.write(f"a_i: {args.ai}\n")
        f.write(f"alpha_R: {args.alpha_R}\n")
        f.write(f"kappa_tick: {args.kappa_tick}\n")
        f.write(f"c4: {args.c4}\n")
        f.write(f"log10_l0: {args.log10_l0}\n")
        f.write(f"f_norm_target: {args.f_norm_target}\n\n")

        f.write("Key observables\n")
        f.write("--------------\n")
        f.write(f"ratio(z=0) = {r0:.8e}\n")
        f.write(f"delta0_measured = {delta0_meas:.8e}\n")
        f.write(f"delta0_expected (alpha_R*(12*kappa_tick)*f_norm) = {delta0_expect:.8e}\n")
        f.write(f"|delta0_measured - delta0_expected| = {abs(delta0_meas-delta0_expect):.8e}\n\n")

        f.write("High-z safety diagnostics\n")
        f.write("------------------------\n")
        f.write(f"ratio(z=2) = {r2:.8e}\n")
        if not np.isnan(r1100):
            f.write(f"ratio(z=1100) = {r1100:.8e}\n")
        f.write(f"max |ratio-1| for z>=2 = {max_hi:.8e}\n\n")

        f.write("LCDM-limit diagnostics\n")
        f.write("----------------------\n")
        f.write("EDCL enabled but alpha_R=0 (should reproduce LCDM exactly)\n")
        f.write(f"max |ratio_alphaR0 - 1| = {max_lcdm_lim:.8e}\n\n")

        f.write("Files written\n")
        f.write("------------\n")
        f.write(f"{csv_path}\n{fig_path}\n{rep_path}\n{dist_csv}\n{summary_json}\n")

    print("Wrote:", csv_path)
    print("Wrote:", fig_path)
    print("Wrote:", rep_path)
    print("Wrote:", dist_csv)
    print("Wrote:", summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
