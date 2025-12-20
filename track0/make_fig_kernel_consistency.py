#!/usr/bin/env python3
"""
Generate Track-0 kernel consistency and H(z) ratio figure.

This script is intentionally dependency-light: numpy + matplotlib only.

Outputs:
  paper_artifacts/track0/fig_kernel_consistency.png
  paper_artifacts/track0/kernel_consistency_report.txt
"""
from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import math
import numpy as np
import matplotlib.pyplot as plt

from track0.kernel import KernelNormalization, delta_of_a_highz_limit, hubble_ratio_from_delta


def main() -> None:
    out_dir = os.path.join("paper_artifacts", "track0")
    os.makedirs(out_dir, exist_ok=True)

    # Paper-quoted values (explicitly stated in the manuscript)
    delta0_paper = 0.089  # δ0 = α_R f_norm = 0.089
    zeta = 0.5
    a_i = 1e-4
    f_norm_target = 0.7542

    zs = np.concatenate([
        np.linspace(0.0, 3.0, 301),
        np.array([5.0, 10.0, 50.0, 1100.0]),
    ])
    a_of_z = 1.0 / (1.0 + zs)

    variants = [
        ("paper_equation_1mexp", "Eq. kernel-shape: (1-exp(-z/ζ))"),
        ("paper_claim_exp", "Claim-consistent: exp(-z/ζ)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axI = axes[0, 0]
    axD = axes[0, 1]
    axR = axes[1, 0]
    axAbs = axes[1, 1]

    report_lines = []
    report_lines.append("Track-0 Kernel Consistency Report")
    report_lines.append("")
    report_lines.append(f"Settings: zeta={zeta}, a_i={a_i}, f_norm_target={f_norm_target}, delta0={delta0_paper}")
    report_lines.append("Normalization rule: choose global C so that I(a0=1)=f_norm_target.")
    report_lines.append("")

    for variant, label in variants:
        norm = KernelNormalization(a0=1.0, a_i=a_i, zeta=zeta, f_norm_target=f_norm_target, variant=variant, n_log=50000)
        C = norm.constant_C()
        f_norm = norm.f_norm()
        I_vals = np.array([norm.I(float(a)) for a in a_of_z])
        # δ(a) scaled to match δ0 at a0 by construction
        deltas = np.array([delta_of_a_highz_limit(float(a), delta0_paper, norm) for a in a_of_z])
        ratios = 1.0 + deltas  # minimal Track-0 mapping
        abs_dev = np.abs(ratios - 1.0)

        axI.plot(zs, I_vals, label=f"{variant}")
        axD.plot(zs, deltas, label=f"{variant}")
        axR.plot(zs, ratios, label=f"{variant}")
        axAbs.plot(zs, abs_dev, label=f"{variant}")

        # Check the "high-z safety" claim: |ratio-1| <= 0.2% by z~2
        thresh = 0.002
        # find first z where abs_dev <= thresh and stays <= thresh for all higher z in our grid
        z_safe = None
        for i in range(len(zs)):
            if np.all(abs_dev[i:] <= thresh):
                z_safe = float(zs[i])
                break

        report_lines.append(f"Variant: {variant}  ({label})")
        report_lines.append(f"  Normalization constant C = {C:.6g}")
        report_lines.append(f"  f_norm (computed) = {f_norm:.6f} (target {f_norm_target})")
        if z_safe is None:
            report_lines.append(f"  High-z safety (|H_ratio-1|<=0.2%): NOT satisfied on z∈[0,1100] grid")
        else:
            report_lines.append(f"  High-z safety (|H_ratio-1|<=0.2%): satisfied for z >= {z_safe:.3g}")
        report_lines.append("")

    axI.set_title("(a) Normalized kernel integral I(a)=∫K dlog a'")
    axI.set_xlabel("Redshift z")
    axI.set_ylabel("I(a)")
    axI.legend(fontsize=8)

    axD.set_title("(b) δ(z) with δ0 fixed")
    axD.set_xlabel("Redshift z")
    axD.set_ylabel("δ(z)")
    axD.legend(fontsize=8)

    axR.set_title("(c) Track-0 Hubble ratio: H_TP/H_GR = 1 + δ(z)")
    axR.set_xlabel("Redshift z")
    axR.set_ylabel("H ratio")
    axR.axhline(1.0, linestyle="--", linewidth=1)
    axR.legend(fontsize=8)

    axAbs.set_title("(d) |H_ratio-1| (log scale)")
    axAbs.set_xlabel("Redshift z")
    axAbs.set_ylabel("|ratio-1|")
    axAbs.set_yscale("log")
    axAbs.axhline(0.002, linestyle="--", linewidth=1, label="0.2% threshold")
    axAbs.legend(fontsize=8)

    fig.suptitle("Track-0 kernel consistency diagnostic (no CLASS/Cobaya)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = os.path.join(out_dir, "fig_kernel_consistency.png")
    fig.savefig(fig_path, dpi=200)
    print(f"Wrote {fig_path}")

    report_path = os.path.join(out_dir, "kernel_consistency_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
