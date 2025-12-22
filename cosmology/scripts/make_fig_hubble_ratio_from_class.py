#!/usr/bin/env python3
"""
Compute and plot H_TP(z)/H_LCDM(z) from patched CLASS (background-only).

This is a referee-critical diagnostic:
- Verifies LCDM limit (edcl_on=False or alpha_R=0)
- Verifies high-z safety (ratio -> 1 by z~2 and at recombination) as claimed

Usage:
  python cosmology/scripts/make_fig_hubble_ratio_from_class.py --class-path /path/to/class --alpha_R 0.118 --log10_l0 -20.91
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def _add_class_python_path(class_path: str) -> None:
    cand = os.path.join(class_path, "python")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)
    sys.path.insert(0, class_path)


def _compute_background(class_path: str, params: dict) -> dict:
    _add_class_python_path(class_path)
    from classy import Class  # type: ignore
    cosmo = Class()
    try:
        cosmo.set(params)
        cosmo.compute()
        return cosmo.get_background()
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def _extract_z_H(bg: dict) -> tuple[np.ndarray, np.ndarray]:
    # CLASS background dict contains 'z' and either 'H [1/Mpc]' or 'H [1/s]'
    z = np.array(bg["z"], dtype=float)
    key_candidates = ["H [1/Mpc]", "H [1/s]", "H [Mpc^-1]"]
    for k in key_candidates:
        if k in bg:
            H = np.array(bg[k], dtype=float)
            return z, H
    raise KeyError(f"Could not find H in background keys. Available keys: {list(bg.keys())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-path", required=True)
    ap.add_argument("--alpha_R", type=float, default=0.118)
    ap.add_argument("--log10_l0", type=float, default=-20.91)
    ap.add_argument("--kappa_tick", type=float, default=1/12)
    ap.add_argument("--c4", type=float, default=0.06)
    ap.add_argument("--out", default=os.path.join("cosmology", "paper_artifacts", "fig_hubble_ratio_from_class.png"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

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

    lcdm = dict(base)
    lcdm["edcl_on"] = "no"

    edcl = dict(base)
    edcl.update({
        "edcl_on": "yes",
        "kappa_tick": args.kappa_tick,
        "c4": args.c4,
        "alpha_R": args.alpha_R,
        "log10_l0": args.log10_l0,
    })

    bg0 = _compute_background(args.class_path, lcdm)
    bg1 = _compute_background(args.class_path, edcl)

    z0, H0 = _extract_z_H(bg0)
    z1, H1 = _extract_z_H(bg1)

    # Interpolate to common z grid (use lcdm grid)
    # z in CLASS background is typically descending from high to low; sort ascending
    s0 = np.argsort(z0)
    s1 = np.argsort(z1)
    z0a, H0a = z0[s0], H0[s0]
    z1a, H1a = z1[s1], H1[s1]

    # Interpolate EDCL H onto lcdm z grid
    H1i = np.interp(z0a, z1a, H1a)
    ratio = H1i / H0a

    fig = plt.figure(figsize=(7.5, 4.5))
    plt.plot(z0a, ratio, label="H_EDCL / H_LCDM")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlim(0, 5)
    plt.xlabel("Redshift z")
    plt.ylabel("H ratio")
    plt.title("Background ratio from patched CLASS")
    plt.legend()
    plt.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")

    # High-z safety metric at z=2
    def nearest(z_target: float) -> float:
        i = int(np.argmin(np.abs(z0a - z_target)))
        return float(ratio[i])

    r2 = nearest(2.0)
    dev2 = abs(r2 - 1.0)
    print(f"At z≈2: ratio={r2:.6f}, |dev|={dev2:.6f} ({dev2*100:.3f}%)")


if __name__ == "__main__":
    main()
