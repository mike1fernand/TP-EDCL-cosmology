"""\
Tier-B Simulation B1 (Referee upgrade): Theorem 3.1 / Thm.\,localc-1D local-speed validation.

This simulation is designed to validate the *actual* hypotheses and conclusions
stated in the paper's Theorem \ref{thm:localc-1D} ("matched calibration \Leftrightarrow
constant observed speed") in the adiabatic + narrow-band regime, using the paper's
"Minimal simulation recipe" (Sec. "Minimal simulation recipe (used for Figures)").

Key referee-facing improvements vs a toy constant-J demo:

  1) Background is NOT frozen to uniform J. We use the paper's ramp profile
       n_i = n0 (1 + eps * i/L),    J_i = J0 exp(-alpha_J * n_i)
     with J_i interpreted as the *bond* coupling on (i,i+1), consistent with
     Eq. (bond-current) and the interface derivations.

  2) We test the theorem locally, the way it is stated: we launch *many short*
     wavepackets centered at different x0 across a region R, and estimate the
     observed speed v_obs(x0). Matched calibration predicts v_obs independent of
     x0 at leading order; mismatched calibration predicts spatial variability
     set by J/F.

  3) We implement the paper's observer-time construction
       t_cm(τ) = \int F[n(<x>(τ))] dτ
     and also compute the expectation-clock variant
       t_exp(τ) = \int <F[n(x)]> dτ
     reporting the difference and the Lemma \ref{lem:worldline} bound.

  4) We compute the theorem's small parameters
       eps_ad   = (sup_x |a ∂_x J| / J) * sigma
       eps_disp = max{ (k0 a)^2, (Δk a)^2 }
     (Δk obtained from the initial packet spectrum).

Numerics:
  - Evolution: 2nd-order even/odd Trotter (as in the paper's minimal recipe).
  - Dependencies: numpy, matplotlib only.

Outputs:
  - paper_artifacts/fig_theorem31_local_speed_validation.png
  - paper_artifacts/theorem31_local_speed_report.txt

Run:
  python -m tierB.sim_theorem31_local_speed_validation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from edcl.tb1d import gaussian_wavepacket, momentum_stats_periodic


ART_DIR = Path(__file__).resolve().parents[1] / "paper_artifacts"


def _interp_linear(arr: np.ndarray, x_idx: float) -> float:
    """Linear interpolation of 1D array arr at fractional index x_idx."""
    n = arr.size
    if n < 2:
        raise ValueError("Need at least 2 points for interpolation.")
    x_idx = float(np.clip(x_idx, 0.0, n - 1.0))
    i0 = int(np.floor(x_idx))
    if i0 >= n - 1:
        return float(arr[n - 1])
    f = x_idx - i0
    return float((1.0 - f) * arr[i0] + f * arr[i0 + 1])


def _apply_disjoint_bonds(psi: np.ndarray, J_bond: np.ndarray, dt: float, parity: int) -> None:
    """Apply exp(-i H_parity dt) where H_parity is the sum over disjoint bonds.

    Hamiltonian convention matches the paper's interface equation
      E ψ_m = -J_{m-1} ψ_{m-1} - J_m ψ_{m+1}
    hence bond term is -J (|i><i+1| + |i+1><i|).

    For a single bond (i,i+1), the exact 2x2 update is:
      [ψ_i']     [cosθ    i sinθ] [ψ_i]
      [ψ_{i+1}']=[i sinθ  cosθ ] [ψ_{i+1}],   θ = J dt.
    """
    n = psi.size
    idx = np.arange(parity, n - 1, 2)
    if idx.size == 0:
        return
    theta = J_bond[idx] * dt
    c = np.cos(theta)
    s = np.sin(theta)

    a0 = psi[idx].copy()
    b0 = psi[idx + 1].copy()
    psi[idx] = c * a0 + 1j * s * b0
    psi[idx + 1] = 1j * s * a0 + c * b0


def trotter2_step(psi: np.ndarray, J_bond: np.ndarray, dt: float) -> np.ndarray:
    """Second-order even/odd Trotter step."""
    _apply_disjoint_bonds(psi, J_bond, dt / 2.0, parity=0)
    _apply_disjoint_bonds(psi, J_bond, dt, parity=1)
    _apply_disjoint_bonds(psi, J_bond, dt / 2.0, parity=0)
    return psi


@dataclass
class Params:
    # Base lattice/profile (paper: L=1024, ramp in n, exponential J(n))
    L0: int = 1024
    a: float = 1.0
    n0: float = 1.0
    eps_n: float = 1.0
    J0: float = 1.0
    alpha_J: float = 0.5

    # Calibration
    alpha_F: float = 1.0

    # Wavepacket (paper: k0 << π/a)
    k0_0: float = 0.20
    sigma0: float = 32.0
    dtau: float = 0.05
    tau_max0: float = 200.0

    # Local sampling across positions
    n_x0: int = 9
    x0_min_frac: float = 0.25
    x0_max_frac: float = 0.75
    window_start: float = 0.20
    window_end: float = 0.85

    # Convergence sweep over scale factors s (L=L0*s).
    # Default to (1,2) for Colab/CI friendliness; add 4 for a stronger 3-point
    # convergence check when desired.
    scales: Tuple[int, ...] = (1, 2)

    # Mismatch: canonical tick gauge F=n (q=1). (Expose q for optional robustness runs.)
    mismatch_q: float = 1.0

    # Locality gate: require small fractional variation in log(J/F) over the fit window
    locality_gate: float = 0.10

    # Tolerances (used in report only; tests have their own thresholds)
    tol_matched_space: float = 1e-3
    tol_worldline_rel: float = 5e-3


@dataclass
class LevelResult:
    scale: int
    L: int
    sigma: float
    k0: float
    eps_ad: float
    eps_disp: float
    eps_total: float
    # matched
    v_mean_matched: float
    eps_space_matched: float
    eps_time_matched: float
    worldline_rel_matched: float
    # mismatched
    v_mean_mismatch: float
    eps_space_mismatch: float
    eps_time_mismatch: float
    worldline_rel_mismatch: float
    A_pred_mismatch: float
    # locality
    worst_locality_metric_matched: float
    worst_locality_metric_mismatch: float


def build_background(L: int, p: Params) -> Dict[str, np.ndarray]:
    """Construct n_i, J_i, and helper arrays.

    Conventions follow the paper:
      n_i = n0 (1 + eps * i/L)
      J_i = J0 exp(-alpha_J * n_i)
    with J_i interpreted as *bond* coupling on (i,i+1) for i=0..L-2.
    We also keep a site-version J_site(i)=J0 exp(-alpha_J n_i) to define F on sites.
    """
    i = np.arange(L, dtype=float)
    x = i * p.a
    n_site = p.n0 * (1.0 + p.eps_n * (i / float(L)))
    J_site = p.J0 * np.exp(-p.alpha_J * n_site)
    J_bond = J_site[:-1].copy()  # left-site convention, matches Eq. (bond-current)
    return {"x": x, "n_site": n_site, "J_site": J_site, "J_bond": J_bond}


def calibration_arrays(bg: Dict[str, np.ndarray], p: Params) -> Dict[str, np.ndarray]:
    n_site = bg["n_site"]
    J_site = bg["J_site"]
    F_matched = p.alpha_F * J_site
    F_mismatch = n_site ** float(p.mismatch_q)
    return {"F_matched": F_matched, "F_mismatch": F_mismatch}


def eps_params(bg: Dict[str, np.ndarray], sigma: float, k0: float, p: Params, region_idx: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute eps_ad, eps_disp, eps_total, and Δk.

    eps_ad = (sup_x |a ∂_x J|/J) * sigma
    eps_disp = max{ (k0 a)^2, (Δk a)^2 }
    """
    J_site = bg["J_site"]
    # derivative dJ/dx on sites
    dJ_dx = np.gradient(J_site, p.a)
    ratio = np.abs(p.a * dJ_dx[region_idx]) / np.maximum(J_site[region_idx], 1e-300)
    sup_ratio = float(np.max(ratio))
    eps_ad = sup_ratio * float(sigma)

    # Δk from initial packet spectrum (periodic approximation; packet is localized away from boundaries)
    x = bg["x"]
    psi0 = gaussian_wavepacket(x, x0=float(0.5 * x[-1]), sigma=sigma, k0=k0)
    _, dk = momentum_stats_periodic(psi0, a=p.a)
    eps_disp = float(max((k0 * p.a) ** 2, (dk * p.a) ** 2))
    eps_total = eps_ad + eps_disp
    return eps_ad, eps_disp, eps_total, float(dk)


def worldline_bound(p: Params, F_site: np.ndarray, sigma_max: float) -> float:
    """Lemma \ref{lem:worldline} bound coefficient: (sup_x |∇F|) σ_max."""
    dF_dx = np.gradient(F_site, p.a)
    return float(np.max(np.abs(dF_dx)) * sigma_max)


def run_local(
    bg: Dict[str, np.ndarray],
    F_site: np.ndarray,
    p: Params,
    x0: float,
    sigma: float,
    k0: float,
    tau_max: float,
    store_timeseries: bool = False,
) -> Dict[str, object]:
    """Run a single local wavepacket and estimate v_obs via regression in observer time.

    Returns metrics plus optional timeseries.
    """
    x = bg["x"]
    J_bond = bg["J_bond"]
    J_site = bg["J_site"]
    L = x.size

    # q(x) := J/F on sites (used for locality diagnostics and LO speed predictor)
    q_site = J_site / np.maximum(F_site, 1e-300)

    psi = gaussian_wavepacket(x, x0=x0, sigma=sigma, k0=k0)
    steps = int(np.round(tau_max / p.dtau))
    tau = np.linspace(0.0, steps * p.dtau, steps + 1)

    x_mean = np.zeros(steps + 1)
    t_cm = np.zeros(steps + 1)
    t_exp = np.zeros(steps + 1)
    F_cm = np.zeros(steps + 1)
    F_ex = np.zeros(steps + 1)
    norm = np.zeros(steps + 1)

    # Only store full q(t) timeseries for the representative trajectory to keep the
    # many-short-runs experiment lightweight.
    q_cm = np.zeros(steps + 1) if store_timeseries else None
    q_ex = np.zeros(steps + 1) if store_timeseries else None

    # initial observables
    prob = np.abs(psi) ** 2
    norm[0] = float(prob.sum())
    x_mean[0] = float((x * prob).sum())
    F_cm[0] = _interp_linear(F_site, x_mean[0] / p.a)
    F_ex[0] = float((prob * F_site).sum())
    if store_timeseries:
        q_cm[0] = _interp_linear(q_site, x_mean[0] / p.a)
        q_ex[0] = float((prob * q_site).sum())

    for s in range(steps):
        psi = trotter2_step(psi, J_bond, p.dtau)
        prob = np.abs(psi) ** 2
        norm[s + 1] = float(prob.sum())
        x_mean[s + 1] = float((x * prob).sum())
        F_cm[s + 1] = _interp_linear(F_site, x_mean[s + 1] / p.a)
        F_ex[s + 1] = float((prob * F_site).sum())
        if store_timeseries:
            q_cm[s + 1] = _interp_linear(q_site, x_mean[s + 1] / p.a)
            q_ex[s + 1] = float((prob * q_site).sum())
        # trapezoidal rule for time integrals
        t_cm[s + 1] = t_cm[s] + 0.5 * (F_cm[s] + F_cm[s + 1]) * p.dtau
        t_exp[s + 1] = t_exp[s] + 0.5 * (F_ex[s] + F_ex[s + 1]) * p.dtau

    # analysis window
    i1 = int(np.floor(p.window_start * (steps + 1)))
    i2 = int(np.floor(p.window_end * (steps + 1)))
    i1 = max(0, min(i1, steps - 5))
    i2 = max(i1 + 5, min(i2, steps + 1))

    t_win = t_cm[i1:i2]
    x_win = x_mean[i1:i2]
    # linear regression x = v t + b
    A = np.vstack([t_win, np.ones_like(t_win)]).T
    v_hat, b_hat = np.linalg.lstsq(A, x_win, rcond=None)[0]
    v_hat = float(v_hat)
    # instantaneous observed velocity
    v_inst = np.gradient(x_win, t_win)
    eps_time = float(np.std(v_inst) / max(np.abs(np.mean(v_inst)), 1e-300))

    # local variation of log(J/F) over the fit window (locality diagnostic)
    # q_site already defined above.
    logq = np.log(np.maximum(q_site, 1e-300))
    dlogq_dx = np.gradient(logq, p.a)
    x0_idx = x0 / p.a
    dlogq0 = _interp_linear(dlogq_dx, x0_idx)
    dx_win = float(x_win[-1] - x_win[0])
    locality_metric = float(np.abs(dlogq0) * np.abs(dx_win))

    # worldline approximation discrepancy
    rel_worldline = float(np.max(np.abs(t_cm[i1:i2] - t_exp[i1:i2])) / max(np.max(t_exp[i1:i2]), 1e-300))

    out: Dict[str, object] = {
        "v_hat": v_hat,
        "eps_time": eps_time,
        "locality_metric": locality_metric,
        "rel_worldline": rel_worldline,
        "norm_drift": float(np.max(np.abs(norm - norm[0]))),
    }
    if store_timeseries:
        out.update({
            "tau": tau,
            "t_cm": t_cm,
            "t_exp": t_exp,
            "x_mean": x_mean,
            "v_inst": v_inst,
            "q_cm": q_cm,
            "q_ex": q_ex,
            "i1": i1,
            "i2": i2,
        })
    return out


def run_validation(p: Params, make_plots: bool = True) -> Tuple[List[LevelResult], Dict[str, object]]:
    """Run the local-speed validation for matched vs mismatched calibration across scale factors."""
    levels: List[LevelResult] = []

    # store representative trajectories for plotting (scale=1, x0 mid)
    rep: Dict[str, object] = {}

    for s in p.scales:
        L = int(p.L0 * s)
        sigma = float(p.sigma0 * np.sqrt(s))
        k0 = float(p.k0_0 / np.sqrt(s))
        tau_max = float(p.tau_max0 * np.sqrt(s))

        bg = build_background(L, p)
        cal = calibration_arrays(bg, p)
        x = bg["x"]

        # choose x0 grid, clipped away from boundaries by 6σ (conservative)
        x_min = max(p.x0_min_frac * x[-1], 6.0 * sigma)
        x_max = min(p.x0_max_frac * x[-1], x[-1] - 6.0 * sigma)
        if not (x_max > x_min):
            raise ValueError("Invalid x0 range after boundary/sigma clipping. Increase L or decrease sigma.")
        x0_list = np.linspace(x_min, x_max, p.n_x0)

        # region for eps_ad: cover sampled region +/-3σ
        i_lo = int(max(0, np.floor((x0_list.min() - 3.0 * sigma) / p.a)))
        i_hi = int(min(L - 1, np.ceil((x0_list.max() + 3.0 * sigma) / p.a)))
        region_idx = np.arange(i_lo, i_hi + 1)

        eps_ad, eps_disp, eps_total, dk = eps_params(bg, sigma=sigma, k0=k0, p=p, region_idx=region_idx)

        # Matched runs
        v_list_m: List[float] = []
        eps_time_m: List[float] = []
        wrel_m: List[float] = []
        loc_m: List[float] = []

        # Mismatched runs
        v_list_mm: List[float] = []
        eps_time_mm: List[float] = []
        wrel_mm: List[float] = []
        loc_mm: List[float] = []

        for j, x0 in enumerate(x0_list):
            store = (s == p.scales[0]) and (j == len(x0_list) // 2)
            # matched
            rm = run_local(bg, cal["F_matched"], p, x0=x0, sigma=sigma, k0=k0, tau_max=tau_max, store_timeseries=store)
            v_list_m.append(float(rm["v_hat"]))
            eps_time_m.append(float(rm["eps_time"]))
            wrel_m.append(float(rm["rel_worldline"]))
            loc_m.append(float(rm["locality_metric"]))
            # mismatched
            rmm = run_local(bg, cal["F_mismatch"], p, x0=x0, sigma=sigma, k0=k0, tau_max=tau_max, store_timeseries=store)
            v_list_mm.append(float(rmm["v_hat"]))
            eps_time_mm.append(float(rmm["eps_time"]))
            wrel_mm.append(float(rmm["rel_worldline"]))
            loc_mm.append(float(rmm["locality_metric"]))

            if store:
                rep = {
                    "scale": s,
                    "bg": bg,
                    "cal": cal,
                    "x0": float(x0),
                    "sigma": sigma,
                    "k0": k0,
                    "dk": dk,
                    "tau_max": tau_max,
                    "matched": rm,
                    "mismatch": rmm,
                }

        v_arr_m = np.array(v_list_m)
        v_arr_mm = np.array(v_list_mm)

        v_mean_m = float(np.mean(v_arr_m))
        eps_space_m = float(np.std(v_arr_m) / max(np.abs(v_mean_m), 1e-300))
        v_mean_mm = float(np.mean(v_arr_mm))
        eps_space_mm = float(np.std(v_arr_mm) / max(np.abs(v_mean_mm), 1e-300))

        # predicted mismatch variability from J/F at sampled positions
        J_site = bg["J_site"]
        n_site = bg["n_site"]
        # evaluate at nearest site for predictor (sufficient for A_pred)
        idx0 = np.clip(np.round(x0_list / p.a).astype(int), 0, L - 1)
        q_mismatch = J_site[idx0] / np.maximum((n_site[idx0] ** float(p.mismatch_q)), 1e-300)
        A_pred = float(np.std(q_mismatch) / max(np.mean(q_mismatch), 1e-300))

        levels.append(
            LevelResult(
                scale=s,
                L=L,
                sigma=sigma,
                k0=k0,
                eps_ad=eps_ad,
                eps_disp=eps_disp,
                eps_total=eps_total,
                v_mean_matched=v_mean_m,
                eps_space_matched=eps_space_m,
                eps_time_matched=float(np.mean(eps_time_m)),
                worldline_rel_matched=float(np.max(wrel_m)),
                v_mean_mismatch=v_mean_mm,
                eps_space_mismatch=eps_space_mm,
                eps_time_mismatch=float(np.mean(eps_time_mm)),
                worldline_rel_mismatch=float(np.max(wrel_mm)),
                A_pred_mismatch=A_pred,
                worst_locality_metric_matched=float(np.max(loc_m)),
                worst_locality_metric_mismatch=float(np.max(loc_mm)),
            )
        )

    if make_plots:
        make_figure(levels, rep, p, outpath=ART_DIR / "fig_theorem31_local_speed_validation.png")
        write_report(levels, rep, p, outpath=ART_DIR / "theorem31_local_speed_report.txt")
    return levels, rep


def make_figure(levels: List[LevelResult], rep: Dict[str, object], p: Params, outpath: Path) -> None:
    """Create a referee-facing multi-panel figure."""
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if not rep:
        raise ValueError("No representative run stored; cannot plot.")
    bg = rep["bg"]
    cal = rep["cal"]
    x = bg["x"]
    n = bg["n_site"]
    J = bg["J_site"]
    Fm = cal["F_matched"]
    Fmm = cal["F_mismatch"]

    # Lemma residual r(x)=∂x ln J - ∂x ln F
    dlnJ = np.gradient(np.log(np.maximum(J, 1e-300)), p.a)
    dlnF_m = np.gradient(np.log(np.maximum(Fm, 1e-300)), p.a)
    dlnF_mm = np.gradient(np.log(np.maximum(Fmm, 1e-300)), p.a)
    r_m = dlnJ - dlnF_m
    r_mm = dlnJ - dlnF_mm

    # Representative trajectory arrays
    rm = rep["matched"]
    rmm = rep["mismatch"]
    t_m = rm["t_cm"]
    x_m = rm["x_mean"]
    t_mm = rmm["t_cm"]
    x_mm = rmm["x_mean"]
    i1 = int(rm["i1"])
    i2 = int(rm["i2"])
    # Instantaneous v_obs in window
    v_inst_m = np.gradient(x_m[i1:i2], t_m[i1:i2])
    v_inst_mm = np.gradient(x_mm[i1:i2], t_mm[i1:i2])

    # Convergence arrays
    eps_tot = np.array([lv.eps_total for lv in levels])
    eps_space_m = np.array([lv.eps_space_matched for lv in levels])
    eps_space_mm = np.array([lv.eps_space_mismatch for lv in levels])
    A_pred = np.array([lv.A_pred_mismatch for lv in levels])

    # Predicted matched speed (small-k) and exact-sin alternative
    k0 = float(rep["k0"])
    v0_thm = 2.0 * k0 / p.alpha_F
    v0_sin = 2.0 * np.sin(k0 * p.a) / p.alpha_F

    fig = plt.figure(figsize=(15, 9), dpi=160)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, n, label=r"$n(x)$")
    ax1.set_title("(a) Background profile: ramp n(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("n")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, J, label=r"$J(x)=J_0 e^{-\alpha_J n}$")
    ax2.plot(x, Fm, "--", label=r"Matched $F=\alpha_F J$")
    ax2.plot(x, Fmm, ":", label=r"Mismatch $F=n^{q}$")
    ax2.set_title("(b) Coupling and calibration")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Value")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x, r_m, label=r"$\partial_x\ln J-\partial_x\ln F$ (matched)")
    ax3.plot(x, r_mm, label=r"$\partial_x\ln J-\partial_x\ln F$ (mismatch)")
    ax3.set_title("(c) Lemma (only-if) diagnostic")
    ax3.set_xlabel("x")
    ax3.set_ylabel("residual")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_m, x_m, label="Matched")
    ax4.plot(t_mm, x_mm, label="Mismatch")
    ax4.set_title("(d) Representative trajectory x(t) (observer time)")
    ax4.set_xlabel("t")
    ax4.set_ylabel(r"$\langle x\rangle$")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_m[i1:i2], v_inst_m, label="v_obs(t) matched")
    ax5.plot(t_mm[i1:i2], v_inst_mm, label="v_obs(t) mismatch")

    # Theorem LO predictor along the trajectory: v_pred(t) ≈ 2 k0 (J/F)(x(t)).
    q_m = np.asarray(rm["q_cm"])[i1:i2]
    q_mm = np.asarray(rmm["q_cm"])[i1:i2]
    v_pred_m = 2.0 * k0 * q_m
    v_pred_mm = 2.0 * k0 * q_mm
    ax5.plot(t_m[i1:i2], v_pred_m, "--", linewidth=1.0, alpha=0.7, label="LO pred (matched)")
    ax5.plot(t_mm[i1:i2], v_pred_mm, "--", linewidth=1.0, alpha=0.7, label="LO pred (mismatch)")
    ax5.axhline(v0_thm, color="k", linestyle="--", linewidth=1.0, label=f"Thm small-k: 2k0/αF={v0_thm:.3f}")
    ax5.axhline(v0_sin, color="k", linestyle=":", linewidth=1.0, label=f"Exact-sin: 2sin(k0)/αF={v0_sin:.3f}")
    ax5.set_title("(e) Observed velocity (window)")
    ax5.set_xlabel("t")
    ax5.set_ylabel(r"$v_{\rm obs}$")
    ax5.legend(fontsize=7)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(eps_tot, eps_space_m, "o-", label=r"Matched: std$_{x_0}$(v)/mean")
    ax6.semilogy(eps_tot, eps_space_mm, "s-", label=r"Mismatch: std$_{x_0}$(v)/mean")
    ax6.semilogy(eps_tot, A_pred, "^-", label=r"Mismatch predictor A=std(J/F)/mean")
    ax6.set_title("(f) Convergence vs ε_total")
    ax6.set_xlabel(r"$\varepsilon_{\rm ad}+\varepsilon_{\rm disp}$")
    ax6.set_ylabel("relative variability")
    ax6.legend(fontsize=8)
    ax6.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Theorem 3.1 / Thm. localc-1D validation: matched calibration ⇒ position-independent observed speed",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def write_report(levels: List[LevelResult], rep: Dict[str, object], p: Params, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("TP/EDCL Tier-B: Theorem 3.1 / Thm. localc-1D local-speed validation")
    lines.append("=" * 78)
    lines.append("Profile (paper minimal recipe): n_i = n0(1+eps i/L), J_i = J0 exp(-alpha_J n_i).")
    lines.append("Evolution: 2nd-order even/odd Trotter.")
    lines.append("Observer time: t_cm(τ)=∫ F[n(<x>)] dτ; also computed t_exp(τ)=∫ <F> dτ.")
    lines.append("")
    lines.append("Parameters:")
    lines.append(f"  L0={p.L0}, scales={p.scales}")
    lines.append(f"  n0={p.n0}, eps_n={p.eps_n}, J0={p.J0}, alpha_J={p.alpha_J}")
    lines.append(f"  alpha_F={p.alpha_F} (matched F=alpha_F*J)")
    lines.append(f"  mismatch: F=n^{p.mismatch_q}")
    lines.append(f"  k0_0={p.k0_0}, sigma0={p.sigma0}, dtau={p.dtau}, tau_max0={p.tau_max0}")
    lines.append(f"  x0 sampling: n_x0={p.n_x0}, frac=[{p.x0_min_frac},{p.x0_max_frac}] (with 6σ boundary clip)")
    lines.append("")
    lines.append("Computed per-level summary")
    lines.append("-" * 78)
    header = (
        "scale  L     sigma  k0     eps_ad   eps_disp  eps_tot   "
        "eps_space(M)  eps_space(MM)  A_pred(MM)  worldline_rel(M)  worldline_rel(MM)"
    )
    lines.append(header)
    for lv in levels:
        lines.append(
            f"{lv.scale:>5d}  {lv.L:>5d}  {lv.sigma:>5.1f}  {lv.k0:>5.3f}  "
            f"{lv.eps_ad:>7.4f}  {lv.eps_disp:>8.5f}  {lv.eps_total:>7.4f}  "
            f"{lv.eps_space_matched:>12.4e}  {lv.eps_space_mismatch:>13.4e}  "
            f"{lv.A_pred_mismatch:>10.4e}  {lv.worldline_rel_matched:>15.4e}  {lv.worldline_rel_mismatch:>16.4e}"
        )

    lines.append("")
    lines.append("Acceptance / referee checks")
    lines.append("-" * 78)
    for lv in levels:
        # locality gate
        loc_ok_m = (lv.worst_locality_metric_matched <= p.locality_gate)
        loc_ok_mm = (lv.worst_locality_metric_mismatch <= p.locality_gate)
        lines.append(f"Level scale={lv.scale}:")
        lines.append(f"  Locality metric (matched)  max={lv.worst_locality_metric_matched:.3e}  gate={p.locality_gate:.2e}  => {'OK' if loc_ok_m else 'FAIL'}")
        lines.append(f"  Locality metric (mismatch) max={lv.worst_locality_metric_mismatch:.3e}  gate={p.locality_gate:.2e}  => {'OK' if loc_ok_mm else 'FAIL'}")
        # worldline mapping
        lines.append(f"  Worldline t_cm vs t_exp (matched)  max rel={lv.worldline_rel_matched:.3e}  (target < {p.tol_worldline_rel:.1e})")
        lines.append(f"  Worldline t_cm vs t_exp (mismatch) max rel={lv.worldline_rel_mismatch:.3e}  (target < {p.tol_worldline_rel:.1e})")
        # theorem qualitative target
        lines.append(f"  Matched spatial variability eps_space={lv.eps_space_matched:.3e} (target < {p.tol_matched_space:.1e} at finest)")
        lines.append(f"  Mismatch spatial variability eps_space={lv.eps_space_mismatch:.3e} vs predictor A={lv.A_pred_mismatch:.3e}")

    # include lemma bound coefficient for representative profile
    if rep:
        bg = rep["bg"]
        cal = rep["cal"]
        sigma = float(rep["sigma"])
        bound_m = worldline_bound(p, cal["F_matched"], sigma_max=sigma)
        bound_mm = worldline_bound(p, cal["F_mismatch"], sigma_max=sigma)
        lines.append("")
        lines.append("Lemma worldline bound coefficient (sup|∇F| σ_max):")
        lines.append(f"  matched:  {bound_m:.6e}")
        lines.append(f"  mismatch: {bound_mm:.6e}")

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)
    params = Params()
    levels, _ = run_validation(params, make_plots=True)
    print("=== Theorem 3.1 / Thm. localc-1D local-speed validation ===")
    for lv in levels:
        print(
            f"scale={lv.scale}  eps_total={lv.eps_total:.4f}  "
            f"eps_space(M)={lv.eps_space_matched:.3e}  eps_space(MM)={lv.eps_space_mismatch:.3e}  A_pred={lv.A_pred_mismatch:.3e}"
        )
    print(f"Wrote: {ART_DIR / 'fig_theorem31_local_speed_validation.png'}")
    print(f"Wrote: {ART_DIR / 'theorem31_local_speed_report.txt'}")
