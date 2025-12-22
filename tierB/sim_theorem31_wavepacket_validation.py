"""
Tier-B Simulation B1: Wavepacket constant-speed criterion (referee-oriented)

Original goal:
Validate the *calibration matching* criterion behind Theorem \\ref{thm:localc-1D}:
  v_obs(x) = v_coord(x)/F[n(x)] + O(errors),
so v_obs is constant to leading order iff F[n] ∝ J(n).

Key numerical issue (addressed here):
In a chain with spatially varying J(x), a generic narrow-band wavepacket can
experience *k-drift* due to local dispersion E(k,x). That drift can mask the
simple proportionality v_coord ∝ J that the theorem's leading-order statement
uses.

To isolate the calibration effect without making additional physical assumptions,
we choose the wavepacket center k0 = π/2:
- For tight-binding dispersion E(k,x)= -2 J(x) cos k, the energy at k0=π/2 is E=0,
  independent of J(x). This suppresses systematic k-drift induced by J(x) gradients.
- Group velocity is v_coord ≈ 2 J(x) sin k0 = 2 J(x), i.e. proportional to J(x).

We then post-process two calibrations on the same structure-time evolution:
  Matched:    F[n] = J(n)
  Mismatched: F[n] = n

Observer time along the wavepacket is defined by dt/dτ = ⟨F⟩.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from edcl.tb1d import (
    gaussian_wavepacket, momentum_stats_periodic,
    build_tridiagonal_H, crank_nicolson_stepper, cn_step,
)

ART_DIR = Path(__file__).resolve().parents[1] / "paper_artifacts"


@dataclass
class RunParams:
    N: int = 512
    a: float = 1.0
    dtau: float = 0.10
    tau_max: float = 150.0
    x0: float = 80.0
    sigma_x: float = 20.0
    k0: float = float(np.pi / 2.0)  # k0=π/2 suppresses J-gradient-induced k drift
    # density profile n(x) = n0 + amp * tanh((x-xc)/L)
    n0: float = 1.0
    amp: float = 0.5
    L: float = 160.0
    xc: float = 260.0
    # diagnostics sampling stride for k-stats
    k_sample_stride: int = 40


def density_profile(x: np.ndarray, n0: float, amp: float, xc: float, L: float) -> np.ndarray:
    n = n0 + amp * np.tanh((x - xc) / L)
    if np.any(n <= 0):
        raise ValueError("Density profile produced non-positive n(x).")
    return n


def make_couplings_from_n(n_site: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose a monotone J(n) that is not proportional to n.
    Here: J(n) = sqrt(n), so mismatched F=n differs from matched F=J.
    """
    J_site = np.sqrt(n_site)
    J_bond = 0.5 * (J_site[:-1] + J_site[1:])
    return J_site, J_bond


def compute_error_parameters(J_site: np.ndarray, sigma_x: float, psi0: np.ndarray, a: float) -> tuple[float, float, float, float]:
    """
    Conservative error surrogates:
      ε_ad := sup_x (|a ∂x J|/J) σ
      ε_disp := (Δk a)^2   (bandwidth-induced velocity spread near k0=π/2)
    Returns (eps_ad, eps_disp, k_mean, k_std).
    """
    dJdx = np.gradient(J_site, a)
    eps_ad = float(np.max(np.abs(dJdx) / np.maximum(J_site, 1e-12)) * sigma_x)
    k_mean, k_std = momentum_stats_periodic(psi0, a=a)
    eps_disp = float((k_std * a) ** 2)
    return eps_ad, eps_disp, k_mean, k_std


def simulate_structure_time(params: RunParams) -> dict:
    N = params.N
    a = params.a
    x = np.arange(N, dtype=float) * a
    n_site = density_profile(x, params.n0, params.amp, params.xc, params.L)
    J_site, J_bond = make_couplings_from_n(n_site)

    h0, h1 = build_tridiagonal_H(J_bond, onsite=None)
    solverA, Bdiag, Bup, Blow = crank_nicolson_stepper(h0, h1, dt=params.dtau)

    psi = gaussian_wavepacket(x, params.x0, params.sigma_x, params.k0)

    eps_ad, eps_disp, k_mean0, k_std0 = compute_error_parameters(J_site, params.sigma_x, psi, a)

    steps = int(params.tau_max / params.dtau)
    tau = np.linspace(0, steps * params.dtau, steps + 1)

    x_mean = np.zeros(steps + 1)
    F_matched = np.zeros(steps + 1)
    F_mismatch = np.zeros(steps + 1)
    norm = np.zeros(steps + 1)

    # occasional momentum stats to verify k-drift suppression
    k_samples_tau = []
    k_samples_mean = []
    k_samples_std = []

    for s in range(steps + 1):
        prob = np.abs(psi) ** 2
        norm[s] = prob.sum()
        x_mean[s] = float((prob * x).sum())
        F_matched[s] = float((prob * J_site).sum())
        F_mismatch[s] = float((prob * n_site).sum())
        if (s % params.k_sample_stride) == 0:
            km, ks = momentum_stats_periodic(psi, a=a)
            k_samples_tau.append(tau[s])
            k_samples_mean.append(km)
            k_samples_std.append(ks)
        if s < steps:
            psi = cn_step(psi, solverA, Bdiag, Bup, Blow)

    return {
        "params": params,
        "x": x,
        "n_site": n_site,
        "J_site": J_site,
        "tau": tau,
        "x_mean": x_mean,
        "F_matched": F_matched,
        "F_mismatch": F_mismatch,
        "norm": norm,
        "eps_ad": eps_ad,
        "eps_disp": eps_disp,
        "k_mean0": k_mean0,
        "k_std0": k_std0,
        "k_samples": {
            "tau": np.array(k_samples_tau, dtype=float),
            "k_mean": np.array(k_samples_mean, dtype=float),
            "k_std": np.array(k_samples_std, dtype=float),
        },
    }


def derive_observer_time(tau: np.ndarray, dtau: float, Favg: np.ndarray) -> np.ndarray:
    t = np.zeros_like(tau)
    t[1:] = np.cumsum(0.5 * (Favg[1:] + Favg[:-1]) * dtau)
    return t


def compute_velocity_vs_time(x_mean: np.ndarray, time: np.ndarray) -> np.ndarray:
    dx = np.gradient(x_mean)
    dt = np.gradient(time)
    return dx / np.maximum(dt, 1e-12)


def analyze_window(time: np.ndarray, x_mean: np.ndarray, v: np.ndarray, tmin: float, tmax: float) -> dict:
    mask = (time >= tmin) & (time <= tmax)
    if mask.sum() < 10:
        raise ValueError("Analysis window too small.")
    vwin = v[mask]
    eps_v = float(np.std(vwin) / np.maximum(np.mean(vwin), 1e-12))

    tw = time[mask]
    xw = x_mean[mask]
    coeff = np.polyfit(tw, xw, 1)
    vfit, x0 = coeff[0], coeff[1]
    residual = xw - (vfit * tw + x0)
    rms_res = float(np.sqrt(np.mean(residual ** 2)))
    return {
        "mask": mask,
        "eps_v": eps_v,
        "vfit": float(vfit),
        "x0": float(x0),
        "rms_res": rms_res,
        "residual": residual,
    }


def make_figure(base: dict, sweep: list[dict], outpath: Path) -> None:
    tau = base["tau"]
    x_mean = base["x_mean"]
    dtau = base["params"].dtau

    t_m = derive_observer_time(tau, dtau, base["F_matched"])
    t_u = derive_observer_time(tau, dtau, base["F_mismatch"])

    v_m = compute_velocity_vs_time(x_mean, t_m)
    v_u = compute_velocity_vs_time(x_mean, t_u)

    tmin = 0.20 * t_m.max()
    tmax = 0.90 * t_m.max()

    am = analyze_window(t_m, x_mean, v_m, tmin, tmax)
    au = analyze_window(t_u, x_mean, v_u, tmin, tmax)

    v_coord = np.gradient(x_mean) / np.maximum(np.gradient(tau), 1e-12)
    v_pred_m = v_coord / np.maximum(base["F_matched"], 1e-12)
    v_pred_u = v_coord / np.maximum(base["F_mismatch"], 1e-12)

    eps_total = np.array([d["eps_ad"] + d["eps_disp"] for d in sweep], dtype=float)
    eps_v_m = np.array([d["eps_v_matched"] for d in sweep], dtype=float)
    eps_v_u = np.array([d["eps_v_mismatch"] for d in sweep], dtype=float)

    ks = base["k_samples"]

    fig = plt.figure(figsize=(16, 9), dpi=160)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_m, x_mean, label="Matched (F=J)")
    ax1.plot(t_u, x_mean, linestyle="--", label="Mismatched (F=n)")
    ax1.set_title("(a) ⟨x⟩ vs observer time")
    ax1.set_xlabel("Observer time t")
    ax1.set_ylabel("⟨x⟩")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_m, v_m, label=f"Matched: ε_v={am['eps_v']:.2e}")
    ax2.plot(t_u, v_u, linestyle="--", label=f"Mismatched: ε_v={au['eps_v']:.2e}")
    ax2.axhline(am["vfit"], linestyle=":", alpha=0.8)
    ax2.axhline(au["vfit"], linestyle=":", alpha=0.8)
    ax2.set_title("(b) Observed velocity v_obs=d⟨x⟩/dt")
    ax2.set_xlabel("Observer time t")
    ax2.set_ylabel("v_obs")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(eps_total, eps_v_m, marker="o", label="Matched")
    ax3.plot(eps_total, eps_v_u, marker="s", linestyle="--", label="Mismatched")
    ax3.set_title("(c) Convergence: ε_v vs ε_ad+ε_disp")
    ax3.set_xlabel("ε_ad + ε_disp")
    ax3.set_ylabel("ε_v")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_m, v_m, label="v_obs (matched)")
    ax4.plot(t_m, v_pred_m, linestyle="--", label="v_coord/⟨F⟩ (matched)")
    ax4.set_title("(d) Speed transform check (matched)")
    ax4.set_xlabel("Observer time t")
    ax4.set_ylabel("Velocity")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    res_m = np.zeros_like(t_m)
    res_u = np.zeros_like(t_u)
    res_m[am["mask"]] = am["residual"]
    res_u[au["mask"]] = au["residual"]
    ax5.plot(t_m, res_m, label="Matched residual")
    ax5.plot(t_u, res_u, linestyle="--", label="Mismatched residual")
    ax5.axhline(0.0, linestyle=":", alpha=0.8)
    ax5.set_title("(e) Linearity residual (analysis window)")
    ax5.set_xlabel("Observer time t")
    ax5.set_ylabel("⟨x⟩ − (v_fit t + x0)")
    ax5.legend()

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ks["tau"], ks["k_mean"], marker="o", label="k_mean(τ)")
    ax6.fill_between(ks["tau"], ks["k_mean"]-ks["k_std"], ks["k_mean"]+ks["k_std"], alpha=0.2, label="±k_std")
    ax6.axhline(base["params"].k0, linestyle=":", alpha=0.8, label="k0 target")
    ax6.set_title("(f) Momentum stability (diagnostic for k-drift)")
    ax6.set_xlabel("τ")
    ax6.set_ylabel("k")
    ax6.legend()

    fig.suptitle("Calibration criterion: matched F=J yields constant v_obs; mismatch does not (k0=π/2 case)", y=1.02)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def run_single_and_sweep() -> dict:
    base_params = RunParams()
    base = simulate_structure_time(base_params)

    # derive eps_v for base
    tau = base["tau"]
    dtau = base_params.dtau
    x_mean = base["x_mean"]

    t_m = derive_observer_time(tau, dtau, base["F_matched"])
    t_u = derive_observer_time(tau, dtau, base["F_mismatch"])
    v_m = compute_velocity_vs_time(x_mean, t_m)
    v_u = compute_velocity_vs_time(x_mean, t_u)

    tmin = 0.20 * t_m.max()
    tmax = 0.90 * t_m.max()
    base["eps_v_matched"] = analyze_window(t_m, x_mean, v_m, tmin, tmax)["eps_v"]
    base["eps_v_mismatch"] = analyze_window(t_u, x_mean, v_u, tmin, tmax)["eps_v"]

    # Sweep: vary sigma_x (bandwidth) and L (adiabaticity)
    sweep_params = [
        RunParams(sigma_x=14.0, L=120.0),
        RunParams(sigma_x=18.0, L=140.0),
        RunParams(sigma_x=20.0, L=160.0),
        RunParams(sigma_x=26.0, L=220.0),
        RunParams(sigma_x=32.0, L=300.0),
    ]
    sweep = []
    for p in sweep_params:
        d = simulate_structure_time(p)
        tau = d["tau"]
        x_mean = d["x_mean"]
        t_m = derive_observer_time(tau, p.dtau, d["F_matched"])
        t_u = derive_observer_time(tau, p.dtau, d["F_mismatch"])
        v_m = compute_velocity_vs_time(x_mean, t_m)
        v_u = compute_velocity_vs_time(x_mean, t_u)
        tmin = 0.20 * t_m.max()
        tmax = 0.90 * t_m.max()
        d["eps_v_matched"] = analyze_window(t_m, x_mean, v_m, tmin, tmax)["eps_v"]
        d["eps_v_mismatch"] = analyze_window(t_u, x_mean, v_u, tmin, tmax)["eps_v"]
        sweep.append(d)

    outpath = ART_DIR / "fig_theorem31_wavepacket_validation.png"
    make_figure(base, sweep, outpath)

    return {"base": base, "sweep": sweep, "outpath": str(outpath)}


if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)
    results = run_single_and_sweep()
    base = results["base"]
    p = base["params"]
    print("=== Wavepacket calibration criterion (k0=π/2 case) ===")
    print(f"N={p.N}, dtau={p.dtau}, tau_max={p.tau_max}, sigma_x={p.sigma_x}, k0={p.k0:.3f}, L={p.L}, amp={p.amp}")
    print(f"eps_ad={base['eps_ad']:.3e}, eps_disp={base['eps_disp']:.3e}, k0_mean_init={base['k_mean0']:.3f}, k_std_init={base['k_std0']:.3e}")
    print(f"eps_v_matched={base['eps_v_matched']:.3e}")
    print(f"eps_v_mismatch={base['eps_v_mismatch']:.3e}")
    print(f"norm_drift(max |norm-1|)={np.max(np.abs(base['norm']-1.0)):.3e}")
    print(f"Wrote: {results['outpath']}")
