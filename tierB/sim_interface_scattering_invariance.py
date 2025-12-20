"""
Tier-B Simulation B2: Interface scattering invariance + flux conservation.

Anchored paper equations:
- Bond current (Eq. \\ref{eq:bond-current}):
    j_{m+1/2} = 2 Im( J_m ψ_m^* ψ_{m+1} ).
- Flux continuity implies R+T=1 for single interface (Eq. \\ref{eq:flux-continuity-intf}),
  and closed forms (Eq. \\ref{eq:RT-closed}) for propagating case.
- Under time reparameterization t=f(τ) (dt=F(τ)dτ, F>0),
  dimensionless scattering coefficients are invariant, and
  integrated flux is invariant when transforming current as j_t = j_τ/F.

We simulate a wavepacket scattering off a step in bond coupling J:
  J1 (left) -> J2 (right).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from edcl.tb1d import (
    gaussian_wavepacket, bond_current,
    build_tridiagonal_H, crank_nicolson_stepper, cn_step,
)

ART_DIR = Path(__file__).resolve().parents[1] / "paper_artifacts"


@dataclass
class Params:
    N: int = 1024
    a: float = 1.0
    dtau: float = 0.05
    tau_max: float = 420.0
    # interface at site m (bond between m-1 and m)
    m: int = 512
    J1: float = 1.0
    J2: float = 1.5
    # wavepacket
    x0: float = 200.0
    sigma_x: float = 28.0
    k1: float = 1.0  # incident wavenumber (left)
    # time reparameterization (nontrivial, state-independent)
    eps_F: float = 0.45
    tau_mid: float = 220.0
    tau_width: float = 40.0


def F_of_tau(tau: np.ndarray, eps: float, tau_mid: float, width: float) -> np.ndarray:
    """
    A smooth, positive, nontrivial calibration factor F(τ)=dt/dτ.
    Here: F = 1 + eps * tanh((τ-τ_mid)/width).
    For eps<1, F stays positive.
    """
    return 1.0 + eps * np.tanh((tau - tau_mid) / width)


def k2_from_energy(J1: float, J2: float, k1: float) -> float:
    """
    For TB dispersion E=-2J cos k (a=1), energy conservation gives:
      cos(k2) = (J1/J2) cos(k1).
    """
    c = (J1 / J2) * np.cos(k1)
    if abs(c) > 1.0:
        raise ValueError("Chosen parameters produce evanescent regime (complex k2).")
    return float(np.arccos(c))


def analytic_RT(J1: float, J2: float, k1: float, k2: float) -> tuple[float, float]:
    """
    Eq. (RT-closed) with a=1:
      R = ((J1 sin k1 - J2 sin k2)/(J1 sin k1 + J2 sin k2))^2
      T = 4 J1 J2 sin k1 sin k2 / (J1 sin k1 + J2 sin k2)^2
    """
    num = (J1 * np.sin(k1) - J2 * np.sin(k2))
    den = (J1 * np.sin(k1) + J2 * np.sin(k2))
    R = float((num / den) ** 2)
    T = float((4.0 * J1 * J2 * np.sin(k1) * np.sin(k2)) / (den ** 2))
    return R, T


def run(params: Params) -> dict:
    N = params.N
    x = np.arange(N, dtype=float) * params.a

    # Build step couplings on bonds
    J_bond = np.empty(N - 1, dtype=float)
    # bonds i < m-1 are left; i >= m-1 are right
    J_bond[: params.m - 1] = params.J1
    J_bond[params.m - 1 :] = params.J2

    h0, h1 = build_tridiagonal_H(J_bond, onsite=None)
    solverA, Bdiag, Bup, Blow = crank_nicolson_stepper(h0, h1, dt=params.dtau)

    psi = gaussian_wavepacket(x, params.x0, params.sigma_x, params.k1)

    steps = int(params.tau_max / params.dtau)
    tau = np.linspace(0, steps * params.dtau, steps + 1)

    # Track probabilities and current at interface bond
    R_prob = np.zeros(steps + 1)
    T_prob = np.zeros(steps + 1)
    P_mid = np.zeros(steps + 1)
    norm = np.zeros(steps + 1)
    j_int = np.zeros(steps + 1)

    # region splits (exclude a buffer around interface)
    buf = 20
    left_mask = np.arange(N) < (params.m - buf)
    right_mask = np.arange(N) > (params.m + buf)

    for s in range(steps + 1):
        prob = np.abs(psi) ** 2
        norm[s] = prob.sum()
        R_prob[s] = prob[left_mask].sum()
        T_prob[s] = prob[right_mask].sum()
        P_mid[s] = 1.0 - (R_prob[s] + T_prob[s])
        j = bond_current(psi, J_bond)
        j_int[s] = j[params.m - 1]  # interface bond
        if s < steps:
            psi = cn_step(psi, solverA, Bdiag, Bup, Blow)

    # Flux-integrated transmission in τ
    T_flux_tau = float(np.trapz(j_int, x=tau))

    # Time reparameterization and invariance check
    F = F_of_tau(tau, params.eps_F, params.tau_mid, params.tau_width)
    dt = F * params.dtau
    t = np.zeros_like(tau)
    t[1:] = np.cumsum(0.5 * (F[1:] + F[:-1]) * params.dtau)
    j_t = j_int / np.maximum(F, 1e-12)
    T_flux_t = float(np.trapz(j_t, x=t))

    # late-time coefficients (use final time)
    R_fin = float(R_prob[-1])
    T_fin = float(T_prob[-1])
    P_mid_fin = float(P_mid[-1])

    # analytic reference
    k2 = k2_from_energy(params.J1, params.J2, params.k1)
    R_an, T_an = analytic_RT(params.J1, params.J2, params.k1, k2)

    return {
        "params": params,
        "tau": tau,
        "t": t,
        "F": F,
        "R_prob": R_prob,
        "T_prob": T_prob,
        "P_mid": P_mid,
        "j_int": j_int,
        "norm": norm,
        "T_flux_tau": T_flux_tau,
        "T_flux_t": T_flux_t,
        "R_fin": R_fin,
        "T_fin": T_fin,
        "P_mid_fin": P_mid_fin,
        "R_an": R_an,
        "T_an": T_an,
        "k2": k2,
    }


def make_figure(d: dict, outpath: Path) -> None:
    tau = d["tau"]
    t = d["t"]
    R = d["R_prob"]
    T = d["T_prob"]
    Pm = d["P_mid"]
    j = d["j_int"]
    F = d["F"]

    fig = plt.figure(figsize=(14, 8), dpi=160)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tau, R, label="R_prob(τ)")
    ax1.plot(tau, T, label="T_prob(τ)")
    ax1.plot(tau, Pm, label="P_mid(τ)")
    ax1.set_title("(a) Probabilities vs structure time τ")
    ax1.set_xlabel("τ")
    ax1.set_ylabel("Probability")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, R, label="R_prob(t)")
    ax2.plot(t, T, label="T_prob(t)")
    ax2.plot(t, Pm, label="P_mid(t)")
    ax2.set_title("(b) Same probabilities vs reparameterized time t")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Probability")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(tau, j, label="j_int(τ)")
    ax3.set_title("(c) Interface bond current (Eq. bond-current)")
    ax3.set_xlabel("τ")
    ax3.set_ylabel("j")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(tau, F, label="F(τ)=dt/dτ")
    ax4.set_title("(d) Time reparameterization used for invariance check")
    ax4.set_xlabel("τ")
    ax4.set_ylabel("F")
    ax4.legend()

    fig.suptitle("Interface scattering: flux conservation and invariance under time reparameterization", y=1.02)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)
    p = Params()
    d = run(p)
    out = ART_DIR / "fig_interface_scattering_invariance.png"
    make_figure(d, out)
    print("=== Interface scattering invariance ===")
    print(f"J1={p.J1}, J2={p.J2}, k1={p.k1:.3f}, k2={d['k2']:.3f}")
    print(f"Final: R={d['R_fin']:.4f}, T={d['T_fin']:.4f}, P_mid={d['P_mid_fin']:.2e}, R+T+P_mid={d['R_fin']+d['T_fin']+d['P_mid_fin']:.6f}")
    print(f"Flux: T_flux_tau={d['T_flux_tau']:.4f}, T_flux_t={d['T_flux_t']:.4f}, |Δ|={abs(d['T_flux_tau']-d['T_flux_t']):.2e}")
    print(f"Analytic (propagating): R={d['R_an']:.4f}, T={d['T_an']:.4f}")
    print(f"Norm drift max={np.max(np.abs(d['norm']-1.0)):.2e}")
    print(f"Wrote: {out}")
