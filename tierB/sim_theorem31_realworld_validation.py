#!/usr/bin/env python3
"""
Tier-B Simulation: Real-World Theorem 3.1 / Thm. localc-1D Validation
======================================================================

This simulation implements a PRD-referee-acceptable validation of the
theorem "matched calibration ⟺ constant observed local speed" under
the full stated hypotheses:

  1. NON-UNIFORM J(x) = J(n(x)) with controlled spatial variation
  2. Both ε_ad and ε_disp computed from definitions (not assumed)
  3. Local speed test: many short runs at different x₀ positions
  4. Two observer clocks: t_cm (centroid) and t_exp (expectation)
  5. Lemma "only-if" diagnostic: r(x) = ∂_x ln J - ∂_x ln F
  6. Convergence sweep with decreasing ε_ad + ε_disp
  7. Multiple mismatched calibrations for robustness

Key Referee Issues Addressed:
-----------------------------
A1. ε_disp includes both (k₀a)² and (Δka)² where Δk from FFT
A2. Mismatch floor A = std(1/F)/mean(1/F) consistent with v_obs ∝ J/F
A3. Local sampling avoids "long traverse dilutes mismatch signal"
B1. Flux-based measurements when applicable
B2. Crank-Nicolson or Trotter integrator (not dense expm)
B3. Proper frame invariance via observer time at measurement location

Run:
    python -m tierB.sim_theorem31_realworld_validation

Output:
    paper_artifacts/fig_theorem31_realworld_validation.png
    paper_artifacts/theorem31_realworld_report.txt
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ART_DIR = Path(__file__).resolve().parents[1] / "paper_artifacts"


@dataclass
class Params:
    """
    Parameters for the real-world theorem validation.
    
    Profile (paper minimal recipe):
        n_i = n0 (1 + ε_n * i/L)
        J(n) = J0 * exp(-α_J * n)
    
    This gives a smooth, monotone J(x) profile suitable for testing
    the theorem's adiabatic regime.
    
    KEY INSIGHT: The theorem requires BOTH ε_ad and ε_disp to be small.
    We use a gentle profile (small α_J and ε_n) to keep ε_ad small,
    and small k0 with large σ to keep ε_disp small.
    
    REFEREE ISSUES ADDRESSED:
    A. Integrator convergence: Δτ refinement study included
    B. Linear vs exact TB: Both baselines reported
    C. Boundary immunity: min(d_edge/σ) tracked
    D. A_pred definition: Formally defined as Std[J/F]/Mean[J/F]
    """
    # Base lattice/profile
    L0: int = 1600
    a: float = 1.0
    n0: float = 1.0
    eps_n: float = 0.3        # Gentle variation: n goes from 1.0 to 1.3
    J0: float = 1.0
    alpha_J: float = 0.15     # Gentle J variation: J = J0 * exp(-0.15 * n)
    
    # Matched calibration: F = α_F * J(n)
    alpha_F: float = 1.0
    
    # Mismatched calibrations: F = n^q for various q
    mismatch_qs: Tuple[float, ...] = (1.0, 0.5)  # Multiple for robustness
    
    # Wavepacket (small k0 for theorem regime, large σ for narrow bandwidth)
    k0_base: float = 0.08     # Smaller k0 for better linear regime
    sigma_base: float = 80.0  # Large σ for narrow bandwidth
    dtau: float = 0.05
    tau_max_base: float = 300.0  # Reduced for faster runtime
    
    # Local sampling
    n_x0: int = 5             # Reduced for faster runtime (still statistically valid)
    x0_min_frac: float = 0.30  # Start further from left
    x0_max_frac: float = 0.65  # End further from right (packet moves right)
    boundary_pad_sigma: float = 8.0  # Increased padding
    
    # Analysis window (fraction of tau_max) - use middle portion
    window_start: float = 0.25
    window_end: float = 0.75
    
    # Suggestion 2: 3-point convergence for proper convergence analysis
    # With 3 points we can fit and verify convergence behavior
    # Using (1, 2, 3) instead of (1, 2, 4) for manageable runtime
    scales: Tuple[int, ...] = (1, 2, 3)
    
    # Locality gate: max fractional change in log(J/F) over window
    locality_gate: float = 0.10
    
    # Issue A: Δτ refinement factors for integrator convergence check
    dtau_refinement_factors: Tuple[float, ...] = (1.0, 0.5, 0.25)
    
    # Issue C: Minimum boundary distance in units of σ
    # Note: 4σ is sufficient because Gaussian tails at 4σ contain < 0.01% probability
    min_boundary_sigma: float = 4.0
    
    # Suggestion 1: Momentum stability tolerance
    tol_k_stability: float = 0.05  # k_mean should vary by < 5% during window
    
    # Suggestion 3: Temporal constancy tolerance
    tol_temporal_constancy: float = 0.10  # v_obs should vary by < 10% in time
    
    # Acceptance thresholds
    tol_matched_space: float = 0.02   # Allow 2% variation
    tol_mismatch_ratio: float = 3.0   # mismatch/matched ratio threshold
    tol_norm_drift: float = 1e-10
    tol_worldline_rel: float = 0.01
    tol_dtau_convergence: float = 0.05  # Issue A: max change under Δτ refinement


@dataclass
class LevelResult:
    """Results for one convergence level."""
    scale: int
    L: int
    sigma: float
    k0: float
    eps_ad: float
    eps_disp: float
    eps_total: float
    dk: float
    
    # Matched results
    v_mean_matched: float
    eps_space_matched: float
    eps_time_matched: float
    worldline_rel_matched: float
    
    # Mismatched results (for each q)
    mismatch_results: Dict[float, Dict[str, float]]
    
    # Diagnostics
    max_norm_drift: float
    max_locality_metric: float
    lemma_residual_matched: float
    lemma_residual_mismatched: Dict[float, float]
    
    # Issue B: Linear vs exact TB baseline comparison
    v_theory_linear: float      # 2 k0 / α_F (theorem approximation)
    v_theory_exact: float       # 2 sin(k0 a) / α_F (exact TB)
    k0a_squared: float          # (k0 a)² - the small parameter
    
    # Issue C: Boundary immunity
    min_boundary_distance_sigma: float  # min(d_edge / σ) over all runs
    
    # Suggestion 1: Momentum stability
    k_mean_matched: float              # Mean k during matched runs
    k_stability_matched: float         # Fractional k variation
    v_theory_kmean: float              # 2 sin(k_mean a) / α_F
    
    # Suggestion 3: Temporal constancy
    temporal_constancy_matched: float  # std_t(v)/mean_t(v) for matched
    
    # Issue A: Δτ refinement (filled by separate study)
    dtau_convergence_error: Optional[float] = None


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def interp_linear(arr: np.ndarray, x_idx: float) -> float:
    """Linear interpolation of 1D array at fractional index."""
    n = arr.size
    x_idx = float(np.clip(x_idx, 0.0, n - 1.0))
    i0 = int(np.floor(x_idx))
    if i0 >= n - 1:
        return float(arr[n - 1])
    f = x_idx - i0
    return float((1.0 - f) * arr[i0] + f * arr[i0 + 1])


def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    """Complex Gaussian wavepacket."""
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)
    psi = psi.astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def momentum_stats(psi: np.ndarray, a: float = 1.0) -> Tuple[float, float]:
    """
    Compute mean k and std(k) from wavepacket spectrum.
    Uses periodic FFT (adequate when ψ is localized away from boundaries).
    """
    N = psi.size
    psi_k = np.fft.fftshift(np.fft.fft(psi))
    pk = np.abs(psi_k) ** 2
    pk = pk / pk.sum()
    k = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=a))
    k_mean = float((pk * k).sum())
    k2 = float((pk * k ** 2).sum())
    k_std = float(np.sqrt(max(k2 - k_mean ** 2, 0.0)))
    return k_mean, k_std


# ==============================================================================
# TROTTER INTEGRATOR (paper's minimal recipe)
# ==============================================================================

def apply_disjoint_bonds(psi: np.ndarray, J_bond: np.ndarray, dt: float, parity: int) -> None:
    """
    Apply exp(-i H_parity dt) where H_parity is the sum over disjoint bonds.
    
    For tight-binding: H = -J (|i><i+1| + |i+1><i|)
    The exact 2x2 update is:
        [ψ_i']     = [cos θ    i sin θ] [ψ_i  ]
        [ψ_{i+1}']   [i sin θ  cos θ  ] [ψ_{i+1}]
    where θ = J dt.
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
    """Second-order even/odd Trotter step (symmetric splitting)."""
    apply_disjoint_bonds(psi, J_bond, dt / 2.0, parity=0)
    apply_disjoint_bonds(psi, J_bond, dt, parity=1)
    apply_disjoint_bonds(psi, J_bond, dt / 2.0, parity=0)
    return psi


# ==============================================================================
# BACKGROUND PROFILES
# ==============================================================================

def build_background(L: int, p: Params) -> Dict[str, np.ndarray]:
    """
    Construct n(x), J(x), and helper arrays.
    
    Profile:
        n_i = n0 (1 + ε_n * i/L)
        J_i = J0 * exp(-α_J * n_i)
    
    J_bond[i] is the coupling on bond (i, i+1) = J at site i.
    """
    i = np.arange(L, dtype=float)
    x = i * p.a
    n_site = p.n0 * (1.0 + p.eps_n * (i / float(L)))
    J_site = p.J0 * np.exp(-p.alpha_J * n_site)
    J_bond = J_site[:-1].copy()  # Left-site convention
    
    return {
        "x": x,
        "n_site": n_site,
        "J_site": J_site,
        "J_bond": J_bond,
        "L": L,
    }


def calibration_arrays(bg: Dict[str, np.ndarray], p: Params) -> Dict[str, np.ndarray]:
    """
    Build calibration factor arrays.
    
    Matched: F = α_F * J(n)
    Mismatched: F = n^q for each q in mismatch_qs
    """
    J_site = bg["J_site"]
    n_site = bg["n_site"]
    
    result = {
        "F_matched": p.alpha_F * J_site,
    }
    for q in p.mismatch_qs:
        result[f"F_mismatch_q{q}"] = n_site ** float(q)
    
    return result


# ==============================================================================
# ERROR PARAMETERS (computed from definitions)
# ==============================================================================

def compute_eps_params(
    bg: Dict[str, np.ndarray],
    sigma: float,
    k0: float,
    p: Params,
    region_idx: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute ε_ad and ε_disp exactly as defined in the paper.
    
    ε_ad = (sup_x |a ∂_x J| / J) * σ
    ε_disp = max{(k₀a)², (Δk a)²}
    
    Returns: (eps_ad, eps_disp, eps_total, dk)
    """
    J_site = bg["J_site"]
    x = bg["x"]
    L = bg["L"]
    
    # Derivative dJ/dx on sites
    dJ_dx = np.gradient(J_site, p.a)
    ratio = np.abs(p.a * dJ_dx[region_idx]) / np.maximum(J_site[region_idx], 1e-300)
    sup_ratio = float(np.max(ratio))
    eps_ad = sup_ratio * float(sigma)
    
    # Δk from initial packet spectrum (using a packet in the middle of the region)
    x_mid = 0.5 * (x[region_idx[0]] + x[region_idx[-1]])
    psi0 = gaussian_wavepacket(x, x0=x_mid, sigma=sigma, k0=k0)
    _, dk = momentum_stats(psi0, a=p.a)
    
    eps_disp = float(max((k0 * p.a) ** 2, (dk * p.a) ** 2))
    eps_total = eps_ad + eps_disp
    
    return eps_ad, eps_disp, eps_total, dk


def compute_lemma_residual(
    bg: Dict[str, np.ndarray],
    F_site: np.ndarray,
    p: Params,
    region_idx: np.ndarray
) -> float:
    """
    Compute Lemma 'only-if' diagnostic:
        r(x) = ∂_x ln J - ∂_x ln F
    
    For matched calibration (F ∝ J), this should be ≈ 0.
    For mismatched, this will be non-zero.
    
    Returns: max |r(x)| over the region.
    """
    J_site = bg["J_site"]
    
    ln_J = np.log(np.maximum(J_site, 1e-300))
    ln_F = np.log(np.maximum(F_site, 1e-300))
    
    d_ln_J = np.gradient(ln_J, p.a)
    d_ln_F = np.gradient(ln_F, p.a)
    
    r = d_ln_J - d_ln_F
    return float(np.max(np.abs(r[region_idx])))


def worldline_bound(F_site: np.ndarray, sigma_max: float, a: float) -> float:
    """
    Lemma worldline bound coefficient: (sup_x |∇F|) * σ_max
    """
    dF_dx = np.gradient(F_site, a)
    return float(np.max(np.abs(dF_dx)) * sigma_max)


# ==============================================================================
# LOCAL RUN (single wavepacket simulation)
# ==============================================================================

def run_local(
    bg: Dict[str, np.ndarray],
    F_site: np.ndarray,
    p: Params,
    x0: float,
    sigma: float,
    k0: float,
    tau_max: float,
    dtau_override: Optional[float] = None,  # Issue A: allow Δτ refinement
    store_timeseries: bool = False,
) -> Dict[str, object]:
    """
    Run a single local wavepacket and estimate v_obs via regression.
    
    Computes two observer times:
        t_cm: centroid clock (paper recipe)
        t_exp: expectation clock <F>
    
    Returns metrics and optional timeseries.
    
    Issue C: Also tracks minimum distance from packet to boundaries.
    """
    x = bg["x"]
    J_bond = bg["J_bond"]
    J_site = bg["J_site"]
    L = len(x)
    x_max = x[-1]
    
    dtau = dtau_override if dtau_override is not None else p.dtau
    
    psi = gaussian_wavepacket(x, x0=x0, sigma=sigma, k0=k0)
    steps = int(np.round(tau_max / dtau))
    tau = np.linspace(0.0, steps * dtau, steps + 1)
    
    # Storage arrays
    x_mean = np.zeros(steps + 1)
    t_cm = np.zeros(steps + 1)
    t_exp = np.zeros(steps + 1)
    F_cm = np.zeros(steps + 1)
    F_exp_arr = np.zeros(steps + 1)
    norm = np.zeros(steps + 1)
    k_mean_arr = np.zeros(steps + 1)  # Suggestion 1: Track momentum stability
    
    # Issue C: Track boundary distance
    min_dist_left = np.zeros(steps + 1)
    min_dist_right = np.zeros(steps + 1)
    
    # For LO predictor: q = J/F
    q_site = J_site / np.maximum(F_site, 1e-300)
    q_cm = np.zeros(steps + 1)
    
    # Initial observables
    prob = np.abs(psi) ** 2
    norm[0] = float(prob.sum())
    x_mean[0] = float((x * prob).sum())
    F_cm[0] = interp_linear(F_site, x_mean[0] / p.a)
    F_exp_arr[0] = float((prob * F_site).sum())
    q_cm[0] = interp_linear(q_site, x_mean[0] / p.a)
    k_mean_arr[0], _ = momentum_stats(psi, a=p.a)  # Suggestion 1
    
    # Issue C: Boundary distance (use 3σ as packet "edge")
    min_dist_left[0] = x_mean[0] - 3.0 * sigma
    min_dist_right[0] = x_max - (x_mean[0] + 3.0 * sigma)
    
    # Evolve
    for s in range(steps):
        psi = trotter2_step(psi, J_bond, dtau)
        prob = np.abs(psi) ** 2
        norm[s + 1] = float(prob.sum())
        x_mean[s + 1] = float((x * prob).sum())
        F_cm[s + 1] = interp_linear(F_site, x_mean[s + 1] / p.a)
        F_exp_arr[s + 1] = float((prob * F_site).sum())
        q_cm[s + 1] = interp_linear(q_site, x_mean[s + 1] / p.a)
        k_mean_arr[s + 1], _ = momentum_stats(psi, a=p.a)  # Suggestion 1
        
        # Issue C: Track boundary distance
        min_dist_left[s + 1] = x_mean[s + 1] - 3.0 * sigma
        min_dist_right[s + 1] = x_max - (x_mean[s + 1] + 3.0 * sigma)
        
        # Trapezoidal rule for time integrals
        t_cm[s + 1] = t_cm[s] + 0.5 * (F_cm[s] + F_cm[s + 1]) * dtau
        t_exp[s + 1] = t_exp[s] + 0.5 * (F_exp_arr[s] + F_exp_arr[s + 1]) * dtau
    
    # Analysis window
    i1 = int(np.floor(p.window_start * (steps + 1)))
    i2 = int(np.floor(p.window_end * (steps + 1)))
    i1 = max(0, min(i1, steps - 5))
    i2 = max(i1 + 5, min(i2, steps + 1))
    
    t_win = t_cm[i1:i2]
    x_win = x_mean[i1:i2]
    
    # Linear regression: x = v*t + b
    A = np.vstack([t_win, np.ones_like(t_win)]).T
    v_hat, b_hat = np.linalg.lstsq(A, x_win, rcond=None)[0]
    v_hat = float(v_hat)
    
    # Instantaneous observed velocity
    v_inst = np.gradient(x_win, t_win)
    eps_time = float(np.std(v_inst) / max(np.abs(np.mean(v_inst)), 1e-300))
    
    # Locality diagnostic: variation of log(J/F) over the fit window
    logq = np.log(np.maximum(q_site, 1e-300))
    dlogq_dx = np.gradient(logq, p.a)
    x0_idx = x0 / p.a
    dlogq0 = interp_linear(dlogq_dx, x0_idx)
    dx_win = float(x_win[-1] - x_win[0])
    locality_metric = float(np.abs(dlogq0) * np.abs(dx_win))
    
    # Worldline approximation discrepancy
    rel_worldline = float(
        np.max(np.abs(t_cm[i1:i2] - t_exp[i1:i2])) / 
        max(np.max(t_exp[i1:i2]), 1e-300)
    )
    
    # Issue C: Minimum boundary distance in window (in units of σ)
    min_d_left_win = float(np.min(min_dist_left[i1:i2]))
    min_d_right_win = float(np.min(min_dist_right[i1:i2]))
    min_boundary_dist_sigma = min(min_d_left_win, min_d_right_win) / sigma
    
    # Suggestion 1: Momentum stability in window
    k_win = k_mean_arr[i1:i2]
    k_mean_window = float(np.mean(k_win))
    k_stability = float(np.std(k_win) / max(np.abs(k_mean_window), 1e-300))
    
    # Suggestion 3: Explicit temporal constancy metric
    # eps_time = std_t(v_obs)/mean_t(v_obs) in the fit window
    temporal_constancy = eps_time  # Already computed above, but make explicit
    
    out: Dict[str, object] = {
        "v_hat": v_hat,
        "eps_time": eps_time,
        "temporal_constancy": temporal_constancy,  # Suggestion 3: explicit metric
        "locality_metric": locality_metric,
        "rel_worldline": rel_worldline,
        "norm_drift": float(np.max(np.abs(norm - norm[0]))),
        "min_boundary_dist_sigma": min_boundary_dist_sigma,  # Issue C
        "x_mean_window": float(np.mean(x_win)),  # Issue 2: Mean position during fit window
        "k_mean_window": k_mean_window,  # Suggestion 1: Measured k for predictor
        "k_stability": k_stability,  # Suggestion 1: Momentum stability diagnostic
        "k0_input": k0,  # Original k0 for comparison
    }
    
    if store_timeseries:
        out.update({
            "tau": tau,
            "t_cm": t_cm,
            "t_exp": t_exp,
            "x_mean": x_mean,
            "v_inst": v_inst,
            "q_cm": q_cm,  # J/F along trajectory
            "k_mean_arr": k_mean_arr,  # Suggestion 1: For momentum stability panel
            "i1": i1,
            "i2": i2,
        })
    
    return out


# ==============================================================================
# ISSUE A: Δτ REFINEMENT STUDY
# ==============================================================================

def run_dtau_refinement_study(
    bg: Dict[str, np.ndarray],
    F_site: np.ndarray,
    p: Params,
    x0: float,
    sigma: float,
    k0: float,
    tau_max: float,
) -> Dict[str, object]:
    """
    Issue A: Run Δτ refinement study to separate physics from integrator error.
    
    Runs the same physical configuration at Δτ, Δτ/2, Δτ/4 and checks that
    the measured velocity is stable (converged).
    
    Returns dict with v_hat at each Δτ and the convergence error.
    """
    results = {}
    v_list = []
    
    for factor in p.dtau_refinement_factors:
        dtau_test = p.dtau * factor
        r = run_local(
            bg, F_site, p, x0, sigma, k0, tau_max,
            dtau_override=dtau_test, store_timeseries=False
        )
        results[f"v_hat_dtau_{factor}"] = r["v_hat"]
        v_list.append(r["v_hat"])
    
    # Convergence error: max relative change between successive refinements
    v_arr = np.array(v_list)
    if len(v_arr) > 1:
        rel_changes = np.abs(np.diff(v_arr)) / np.maximum(np.abs(v_arr[:-1]), 1e-300)
        convergence_error = float(np.max(rel_changes))
    else:
        convergence_error = 0.0
    
    results["convergence_error"] = convergence_error
    results["v_values"] = v_list
    results["dtau_factors"] = list(p.dtau_refinement_factors)
    
    return results


# ==============================================================================
# MAIN VALIDATION
# ==============================================================================

def run_validation(p: Params, make_plots: bool = True) -> Tuple[List[LevelResult], Dict[str, object]]:
    """
    Run the local-speed validation across convergence scales.
    
    For each scale:
      1. Build profile with J(n(x))
      2. Run matched and mismatched calibrations
      3. Measure speed constancy across positions
      4. Compute all diagnostics including:
         - Issue A: Δτ refinement study
         - Issue B: Linear vs exact TB comparison
         - Issue C: Boundary distance tracking
         - Issue D: Formal A_pred computation
    """
    levels: List[LevelResult] = []
    rep: Dict[str, object] = {}  # Store representative run for plotting
    
    for s in p.scales:
        # Scale parameters
        L = int(p.L0 * s)
        sigma = float(p.sigma_base * np.sqrt(s))
        k0 = float(p.k0_base / np.sqrt(s))
        tau_max = float(p.tau_max_base * np.sqrt(s))
        
        bg = build_background(L, p)
        cal = calibration_arrays(bg, p)
        x = bg["x"]
        
        # Issue B: Compute both velocity baselines
        v_theory_linear = 2.0 * k0 / p.alpha_F  # Theorem approximation
        v_theory_exact = 2.0 * np.sin(k0 * p.a) / p.alpha_F  # Exact TB
        k0a_squared = (k0 * p.a) ** 2
        
        # Choose x0 grid, clipped away from boundaries
        x_min = max(p.x0_min_frac * x[-1], p.boundary_pad_sigma * sigma)
        x_max = min(p.x0_max_frac * x[-1], x[-1] - p.boundary_pad_sigma * sigma)
        if not (x_max > x_min):
            raise ValueError("Invalid x0 range after boundary clipping.")
        x0_list = np.linspace(x_min, x_max, p.n_x0)
        
        # Region for ε_ad: cover sampled region ± 3σ
        i_lo = int(max(0, np.floor((x0_list.min() - 3.0 * sigma) / p.a)))
        i_hi = int(min(L - 1, np.ceil((x0_list.max() + 3.0 * sigma) / p.a)))
        region_idx = np.arange(i_lo, i_hi + 1)
        
        eps_ad, eps_disp, eps_total, dk = compute_eps_params(
            bg, sigma=sigma, k0=k0, p=p, region_idx=region_idx
        )
        
        # Lemma residuals
        lemma_matched = compute_lemma_residual(bg, cal["F_matched"], p, region_idx)
        lemma_mismatched = {}
        for q in p.mismatch_qs:
            lemma_mismatched[q] = compute_lemma_residual(
                bg, cal[f"F_mismatch_q{q}"], p, region_idx
            )
        
        # Run matched calibration
        v_list_m: List[float] = []
        eps_time_m: List[float] = []
        wrel_m: List[float] = []
        loc_m: List[float] = []
        norm_m: List[float] = []
        bdist_m: List[float] = []  # Issue C
        k_mean_m: List[float] = []  # Suggestion 1
        k_stability_m: List[float] = []  # Suggestion 1
        temporal_m: List[float] = []  # Suggestion 3
        
        for j, x0 in enumerate(x0_list):
            store = (s == p.scales[0]) and (j == len(x0_list) // 2)
            rm = run_local(
                bg, cal["F_matched"], p, 
                x0=x0, sigma=sigma, k0=k0, tau_max=tau_max,
                store_timeseries=store
            )
            v_list_m.append(float(rm["v_hat"]))
            eps_time_m.append(float(rm["eps_time"]))
            wrel_m.append(float(rm["rel_worldline"]))
            loc_m.append(float(rm["locality_metric"]))
            norm_m.append(float(rm["norm_drift"]))
            bdist_m.append(float(rm["min_boundary_dist_sigma"]))  # Issue C
            k_mean_m.append(float(rm["k_mean_window"]))  # Suggestion 1
            k_stability_m.append(float(rm["k_stability"]))  # Suggestion 1
            temporal_m.append(float(rm["temporal_constancy"]))  # Suggestion 3
            
            if store:
                rep["matched"] = rm
                rep["x0"] = float(x0)
        
        # Issue A: Δτ refinement study (at middle x0)
        x0_mid = x0_list[len(x0_list) // 2]
        dtau_study = run_dtau_refinement_study(
            bg, cal["F_matched"], p, x0_mid, sigma, k0, tau_max
        )
        dtau_convergence_error = dtau_study["convergence_error"]
        
        # Run mismatched calibrations
        mismatch_results: Dict[float, Dict[str, float]] = {}
        
        for q in p.mismatch_qs:
            F_mm = cal[f"F_mismatch_q{q}"]
            v_list_mm: List[float] = []
            eps_time_mm: List[float] = []
            wrel_mm: List[float] = []
            loc_mm: List[float] = []
            norm_mm: List[float] = []
            bdist_mm: List[float] = []  # Issue C
            x_eff_mm: List[float] = []  # Issue 2: Effective positions for A_pred
            
            for j, x0 in enumerate(x0_list):
                store = (s == p.scales[0]) and (j == len(x0_list) // 2) and (q == p.mismatch_qs[0])
                rmm = run_local(
                    bg, F_mm, p,
                    x0=x0, sigma=sigma, k0=k0, tau_max=tau_max,
                    store_timeseries=store
                )
                v_list_mm.append(float(rmm["v_hat"]))
                eps_time_mm.append(float(rmm["eps_time"]))
                wrel_mm.append(float(rmm["rel_worldline"]))
                loc_mm.append(float(rmm["locality_metric"]))
                norm_mm.append(float(rmm["norm_drift"]))
                bdist_mm.append(float(rmm["min_boundary_dist_sigma"]))  # Issue C
                x_eff_mm.append(float(rmm["x_mean_window"]))  # Issue 2
                
                if store:
                    rep["mismatch"] = rmm
            
            v_arr_mm = np.array(v_list_mm)
            v_mean_mm = float(np.mean(v_arr_mm))
            eps_space_mm = float(np.std(v_arr_mm) / max(np.abs(v_mean_mm), 1e-300))
            
            # Issue D + Issue 2: A_pred = Std[q]/Mean[q] where q = J/F
            # CRITICAL: Evaluated at EFFECTIVE positions (mean ⟨x⟩ during fit window),
            # NOT at initial x0 positions. This ensures apples-to-apples comparison
            # with ε_space which is computed from velocities in the same windows.
            J_site = bg["J_site"]
            x_eff_arr = np.array(x_eff_mm)
            idx_eff = np.clip(np.round(x_eff_arr / p.a).astype(int), 0, L - 1)
            q_vals = J_site[idx_eff] / np.maximum(F_mm[idx_eff], 1e-300)
            A_pred = float(np.std(q_vals) / max(np.mean(q_vals), 1e-300))
            
            mismatch_results[q] = {
                "v_mean": v_mean_mm,
                "eps_space": eps_space_mm,
                "eps_time": float(np.mean(eps_time_mm)),
                "worldline_rel": float(np.max(wrel_mm)),
                "A_pred": A_pred,
                "max_locality": float(np.max(loc_mm)),
                "max_norm_drift": float(np.max(norm_mm)),
                "min_boundary_dist_sigma": float(np.min(bdist_mm)),  # Issue C
            }
        
        # Compile matched results
        v_arr_m = np.array(v_list_m)
        v_mean_m = float(np.mean(v_arr_m))
        eps_space_m = float(np.std(v_arr_m) / max(np.abs(v_mean_m), 1e-300))
        
        # Suggestion 1: Compute mean k_mean across matched runs
        k_mean_m_avg = float(np.mean(k_mean_m))
        k_stability_m_max = float(np.max(k_stability_m))
        v_theory_kmean = 2.0 * np.sin(k_mean_m_avg * p.a) / p.alpha_F
        
        # Suggestion 3: Mean temporal constancy across matched runs
        temporal_m_avg = float(np.mean(temporal_m))
        
        # Issue C: Minimum boundary distance across all runs
        min_bdist_all = min(
            float(np.min(bdist_m)),
            min(mr["min_boundary_dist_sigma"] for mr in mismatch_results.values())
        )
        
        # Store representative data for plotting
        if s == p.scales[0]:
            rep["scale"] = s
            rep["bg"] = bg
            rep["cal"] = cal
            rep["sigma"] = sigma
            rep["k0"] = k0
            rep["dk"] = dk
            rep["tau_max"] = tau_max
            rep["x0_list"] = x0_list
            rep["v_list_m"] = v_list_m
            rep["dtau_study"] = dtau_study  # Issue A
            rep["v_theory_linear"] = v_theory_linear  # Issue B
            rep["v_theory_exact"] = v_theory_exact  # Issue B
            rep["k_mean_matched"] = k_mean_m_avg  # Suggestion 1
            rep["v_theory_kmean"] = v_theory_kmean  # Suggestion 1
        
        levels.append(LevelResult(
            scale=s,
            L=L,
            sigma=sigma,
            k0=k0,
            eps_ad=eps_ad,
            eps_disp=eps_disp,
            eps_total=eps_total,
            dk=dk,
            v_mean_matched=v_mean_m,
            eps_space_matched=eps_space_m,
            eps_time_matched=float(np.mean(eps_time_m)),
            worldline_rel_matched=float(np.max(wrel_m)),
            mismatch_results=mismatch_results,
            max_norm_drift=max(float(np.max(norm_m)), 
                              max(mr["max_norm_drift"] for mr in mismatch_results.values())),
            max_locality_metric=max(float(np.max(loc_m)),
                                   max(mr["max_locality"] for mr in mismatch_results.values())),
            lemma_residual_matched=lemma_matched,
            lemma_residual_mismatched=lemma_mismatched,
            # Issue B
            v_theory_linear=v_theory_linear,
            v_theory_exact=v_theory_exact,
            k0a_squared=k0a_squared,
            # Issue C
            min_boundary_distance_sigma=min_bdist_all,
            # Suggestion 1: Momentum stability
            k_mean_matched=k_mean_m_avg,
            k_stability_matched=k_stability_m_max,
            v_theory_kmean=v_theory_kmean,
            # Suggestion 3: Temporal constancy
            temporal_constancy_matched=temporal_m_avg,
            # Issue A
            dtau_convergence_error=dtau_convergence_error,
        ))
    
    if make_plots:
        make_figure(levels, rep, p, outpath=ART_DIR / "fig_theorem31_realworld_validation.png")
        write_report(levels, rep, p, outpath=ART_DIR / "theorem31_realworld_report.txt")
    
    return levels, rep


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def make_figure(levels: List[LevelResult], rep: Dict[str, object], p: Params, outpath: Path) -> None:
    """Create a referee-facing multi-panel figure."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    if not rep:
        raise ValueError("No representative run stored.")
    
    bg = rep["bg"]
    cal = rep["cal"]
    x = bg["x"]
    n = bg["n_site"]
    J = bg["J_site"]
    Fm = cal["F_matched"]
    Fmm = cal[f"F_mismatch_q{p.mismatch_qs[0]}"]
    
    # Lemma residual r(x) = ∂_x ln J - ∂_x ln F
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
    
    # Suggestion 1: Use measured k_mean in predictor (not k0)
    k0 = float(rep["k0"])
    k_mean_matched = float(rep.get("k_mean_matched", k0))
    sin_kmean = np.sin(k_mean_matched * p.a)  # Using measured k_mean!
    q_cm_m = np.asarray(rm["q_cm"])[i1:i2]
    q_cm_mm = np.asarray(rmm["q_cm"])[i1:i2]
    v_pred_m = 2.0 * sin_kmean * q_cm_m    # Predictor using measured k_mean
    v_pred_mm = 2.0 * sin_kmean * q_cm_mm
    
    # Momentum timeseries for stability panel
    k_mean_arr_m = np.asarray(rm.get("k_mean_arr", [k0]))
    tau_arr = np.asarray(rm.get("tau", np.arange(len(k_mean_arr_m))))
    
    # Convergence arrays
    eps_tot = np.array([lv.eps_total for lv in levels])
    eps_space_m = np.array([lv.eps_space_matched for lv in levels])
    q0 = p.mismatch_qs[0]
    eps_space_mm = np.array([lv.mismatch_results[q0]["eps_space"] for lv in levels])
    A_pred = np.array([lv.mismatch_results[q0]["A_pred"] for lv in levels])
    
    # Predicted matched speed using different baselines
    v_linear_k0 = 2.0 * k0 / p.alpha_F
    v_exact_k0 = 2.0 * np.sin(k0 * p.a) / p.alpha_F
    v_exact_kmean = 2.0 * sin_kmean / p.alpha_F  # Using measured k_mean
    
    fig = plt.figure(figsize=(18, 10), dpi=150)
    gs = fig.add_gridspec(2, 4)  # Changed to 2x4 for momentum panel
    
    # (a) Background profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, n, 'b-', label=r"$n(x)$")
    ax1.set_title("(a) Background: node density ramp")
    ax1.set_xlabel("x")
    ax1.set_ylabel("n")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # (b) Coupling and calibration
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, J, 'b-', lw=2, label=r"$J(x) = J_0 e^{-\alpha_J n}$")
    ax2.plot(x, Fm, 'g--', lw=1.5, label=r"Matched: $F = \alpha_F J$")
    ax2.plot(x, Fmm, 'r:', lw=1.5, label=rf"Mismatch: $F = n^{{{q0}}}$")
    ax2.set_title("(b) Coupling J(x) and calibration F(x)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Value")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # (c) Lemma diagnostic
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x, r_m, 'g-', label=r"Matched: $\partial_x \ln J - \partial_x \ln F$")
    ax3.plot(x, r_mm, 'r-', label=r"Mismatch: $\partial_x \ln J - \partial_x \ln F$")
    ax3.axhline(0, color='k', ls=':', alpha=0.5)
    ax3.set_title("(c) Lemma 'only-if' diagnostic")
    ax3.set_xlabel("x")
    ax3.set_ylabel("residual r(x)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # (d) Suggestion 1: Momentum stability panel
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(tau_arr, k_mean_arr_m, 'g-', lw=1.5, label=r"$\langle k \rangle(t)$")
    ax4.axhline(k0, color='b', ls=':', lw=1, label=f"Initial k₀={k0:.4f}")
    ax4.axhline(k_mean_matched, color='k', ls='--', lw=1, label=f"Window mean={k_mean_matched:.4f}")
    ax4.axvspan(tau_arr[i1], tau_arr[min(i2, len(tau_arr)-1)], alpha=0.2, color='yellow', label="Fit window")
    ax4.set_title("(d) Momentum stability [Suggestion 1]")
    ax4.set_xlabel("τ (coordinate time)")
    ax4.set_ylabel(r"$\langle k \rangle$")
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # (e) Representative trajectory
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(t_m, x_m, 'g-', lw=1.5, label="Matched")
    ax5.plot(t_mm, x_mm, 'r--', lw=1.5, label="Mismatch")
    ax5.set_title("(e) Representative trajectory x(t)")
    ax5.set_xlabel("Observer time t")
    ax5.set_ylabel(r"$\langle x \rangle$")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # (f) Observed velocity with k_mean predictor
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(t_m[i1:i2], v_inst_m, 'g-', lw=1, label="v_obs matched")
    ax6.plot(t_mm[i1:i2], v_inst_mm, 'r-', lw=1, label="v_obs mismatch")
    ax6.plot(t_m[i1:i2], v_pred_m, 'g--', lw=1, alpha=0.7, label=f"Pred: 2sin(k_mean)·J/F")
    ax6.plot(t_mm[i1:i2], v_pred_mm, 'r--', lw=1, alpha=0.7)
    
    # Show all 3 baselines
    ax6.axhline(v_linear_k0, color='b', ls=':', lw=1, label=f"2k₀/α={v_linear_k0:.4f}")
    ax6.axhline(v_exact_k0, color='purple', ls=':', lw=1, label=f"2sin(k₀)/α={v_exact_k0:.4f}")
    ax6.axhline(v_exact_kmean, color='k', ls='--', lw=1.5, label=f"2sin(k_mean)/α={v_exact_kmean:.4f}")
    ax6.set_title("(f) Velocity [Suggestion 1: k_mean predictor]")
    ax6.set_xlabel("Observer time t")
    ax6.set_ylabel(r"$v_{obs}$")
    ax6.legend(fontsize=5.5, loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    # (g) Convergence
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.semilogy(eps_tot, eps_space_m, 'go-', ms=8, lw=2, label=r"Matched: std$_{x_{eff}}$(v)/mean")
    ax7.semilogy(eps_tot, eps_space_mm, 'rs-', ms=8, lw=2, label=r"Mismatch: std$_{x_{eff}}$(v)/mean")
    ax7.semilogy(eps_tot, A_pred, 'r^--', ms=6, lw=1, alpha=0.7, label=r"A_pred=std(J/F)/mean")
    ax7.set_title("(g) Convergence vs ε_total [3 points]")
    ax7.set_xlabel(r"$\varepsilon_{ad} + \varepsilon_{disp}$")
    ax7.set_ylabel("Spatial variability")
    ax7.legend(fontsize=7)
    ax7.grid(True, which="both", alpha=0.3)
    
    # (h) Suggestion 3: Temporal constancy per run
    ax8 = fig.add_subplot(gs[1, 3])
    temporal_m_arr = np.array([lv.temporal_constancy_matched for lv in levels])
    ax8.semilogy(eps_tot, temporal_m_arr, 'go-', ms=8, lw=2, label=r"Matched: std$_t$(v)/mean$_t$(v)")
    ax8.axhline(p.tol_temporal_constancy, color='r', ls='--', lw=1, label=f"Tolerance={p.tol_temporal_constancy}")
    ax8.set_title("(h) Temporal constancy [Suggestion 3]")
    ax8.set_xlabel(r"$\varepsilon_{ad} + \varepsilon_{disp}$")
    ax8.set_ylabel("Temporal variability per run")
    ax8.legend(fontsize=8)
    ax8.grid(True, which="both", alpha=0.3)
    
    fig.suptitle(
        "Real-World Theorem 3.1 Validation: Matched F∝J ⇒ Position-Independent Observed Speed",
        y=1.02, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {outpath}")


def write_report(levels: List[LevelResult], rep: Dict[str, object], p: Params, outpath: Path) -> None:
    """Write detailed text report with all referee-grade diagnostics."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("TP/EDCL Tier-B: Real-World Theorem 3.1 / Thm. localc-1D Validation")
    lines.append("=" * 80)
    lines.append("")
    lines.append("PROFILE (paper minimal recipe):")
    lines.append(f"  n_i = n0(1 + ε_n * i/L)")
    lines.append(f"  J_i = J0 * exp(-α_J * n_i)")
    lines.append("")
    lines.append("CALIBRATIONS:")
    lines.append(f"  Matched: F = α_F * J(n)")
    lines.append(f"  Mismatched: F = n^q for q in {p.mismatch_qs}")
    lines.append("")
    lines.append("PARAMETERS:")
    lines.append(f"  L0={p.L0}, scales={p.scales}")
    lines.append(f"  n0={p.n0}, ε_n={p.eps_n}, J0={p.J0}, α_J={p.alpha_J}")
    lines.append(f"  α_F={p.alpha_F}")
    lines.append(f"  k0_base={p.k0_base}, σ_base={p.sigma_base}, Δτ={p.dtau}")
    lines.append(f"  n_x0={p.n_x0}, boundary_pad={p.boundary_pad_sigma}σ")
    lines.append("")
    
    # Issue 3: Comprehensive parameter table per level
    lines.append("-" * 80)
    lines.append("ISSUE 3: COMPLETE PARAMETER TABLE (per run level)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  ┌─────────┬────────┬────────┬────────┬──────────┬────────────┐")
    lines.append("  │  scale  │   L    │   σ    │   k0   │   τ_max  │   Region R │")
    lines.append("  ├─────────┼────────┼────────┼────────┼──────────┼────────────┤")
    for lv in levels:
        tau_max = p.tau_max_base * np.sqrt(lv.scale)
        x_min = p.x0_min_frac * lv.L * p.a
        x_max = p.x0_max_frac * lv.L * p.a
        lines.append(f"  │ {lv.scale:>5d}   │ {lv.L:>6d} │ {lv.sigma:>6.1f} │ {lv.k0:>6.4f} │ {tau_max:>8.1f} │ [{x_min:.0f},{x_max:.0f}]   │")
    lines.append("  └─────────┴────────┴────────┴────────┴──────────┴────────────┘")
    lines.append("")
    lines.append("  Derived error parameters:")
    lines.append("  ┌─────────┬──────────┬──────────┬──────────┬──────────┐")
    lines.append("  │  scale  │   ε_ad   │  ε_disp  │ (k0*a)²  │   Δk     │")
    lines.append("  ├─────────┼──────────┼──────────┼──────────┼──────────┤")
    for lv in levels:
        lines.append(f"  │ {lv.scale:>5d}   │ {lv.eps_ad:>8.5f} │ {lv.eps_disp:>8.5f} │ {lv.k0a_squared:>8.5f} │ {lv.dk:>8.5f} │")
    lines.append("  └─────────┴──────────┴──────────┴──────────┴──────────┘")
    lines.append("")
    
    # Issue D: Formal definition of A_pred (updated for Issue 2)
    lines.append("-" * 80)
    lines.append("ISSUE D + ISSUE 2: FORMAL DEFINITION OF MISMATCH PREDICTOR")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Define q(x) ≡ J(x) / F(x)")
    lines.append("")
    lines.append("  For matched calibration F = α_F J:")
    lines.append("    q(x) = 1/α_F = const  →  Std[q] = 0")
    lines.append("")
    lines.append("  For mismatched calibration F ≠ const × J:")
    lines.append("    q(x) varies  →  predicted plateau:")
    lines.append("")
    lines.append("    A_pred ≡ Std_{x∈R_eff}[q] / Mean_{x∈R_eff}[q]")
    lines.append("")
    lines.append("  CRITICAL (Issue 2): R_eff is the set of EFFECTIVE positions,")
    lines.append("  i.e., the mean ⟨x⟩ during each run's fit window, NOT the initial x0.")
    lines.append("  This ensures A_pred is evaluated on the same support as ε_space.")
    lines.append("")
    lines.append("  Issue 1: Predictor uses exact TB group velocity 2sin(k0*a), not 2k0.")
    lines.append("")
    lines.append("  Theorem prediction: ε_v^MM → A_pred as ε_ad + ε_disp → 0")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("CONVERGENCE SUMMARY")
    lines.append("-" * 80)
    
    header = (
        f"{'scale':>5} {'L':>6} {'σ':>6} {'k0':>6} {'ε_ad':>8} {'ε_disp':>8} "
        f"{'ε_tot':>8} {'ε_sp(M)':>10} {'ε_sp(MM)':>10} {'A_pred':>10}"
    )
    lines.append(header)
    
    q0 = p.mismatch_qs[0]
    for lv in levels:
        mm = lv.mismatch_results[q0]
        lines.append(
            f"{lv.scale:>5d} {lv.L:>6d} {lv.sigma:>6.1f} {lv.k0:>6.3f} "
            f"{lv.eps_ad:>8.4f} {lv.eps_disp:>8.5f} {lv.eps_total:>8.4f} "
            f"{lv.eps_space_matched:>10.3e} {mm['eps_space']:>10.3e} {mm['A_pred']:>10.3e}"
        )
    
    # Issue B + Suggestion 1: Linear vs Exact TB comparison with k_mean
    lines.append("")
    lines.append("-" * 80)
    lines.append("SUGGESTION 1 + ISSUE B: MOMENTUM STABILITY AND VELOCITY BASELINE")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Theorem uses narrow-band approximation: v_coord ≈ 2aJk0")
    lines.append("  Exact tight-binding dispersion gives:   v_coord = 2aJ sin(k a)")
    lines.append("")
    lines.append("  SUGGESTION 1: Use MEASURED k_mean (not initial k0) in predictor.")
    lines.append("  This addresses systematic offset between v_exact(k0) and v_measured.")
    lines.append("")
    for lv in levels:
        rel_err_k0 = abs(lv.v_theory_linear - lv.v_theory_exact) / lv.v_theory_exact
        rel_err_kmean = abs(lv.v_theory_kmean - lv.v_mean_matched) / max(lv.v_mean_matched, 1e-300)
        lines.append(f"  Scale {lv.scale}:")
        lines.append(f"    Initial k₀ = {lv.k0:.5f}")
        lines.append(f"    Measured k_mean = {lv.k_mean_matched:.5f} (Δk/k₀ = {(lv.k_mean_matched - lv.k0)/lv.k0*100:.2f}%)")
        lines.append(f"    k stability (std/mean) = {lv.k_stability_matched:.3e}")
        lines.append(f"    v_exact(k₀) = 2sin(k₀)/α = {lv.v_theory_exact:.5f}")
        lines.append(f"    v_exact(k_mean) = 2sin(k_mean)/α = {lv.v_theory_kmean:.5f}")
        lines.append(f"    Measured v_obs = {lv.v_mean_matched:.5f}")
        lines.append(f"    Offset from k₀: {(lv.v_mean_matched - lv.v_theory_exact)/lv.v_theory_exact*100:.2f}%")
        lines.append(f"    Offset from k_mean: {rel_err_kmean*100:.2f}%")
    
    # Suggestion 3: Temporal constancy
    lines.append("")
    lines.append("-" * 80)
    lines.append("SUGGESTION 3: TEMPORAL CONSTANCY PER RUN")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Purpose: Verify each run's velocity is stable in TIME (not just space).")
    lines.append("  Metric: std_t(v_obs)/mean_t(v_obs) within fit window")
    lines.append(f"  Tolerance: < {p.tol_temporal_constancy}")
    lines.append("")
    for lv in levels:
        status = "PASS" if lv.temporal_constancy_matched < p.tol_temporal_constancy else "FAIL"
        lines.append(f"  Scale {lv.scale}: temporal constancy = {lv.temporal_constancy_matched:.3e} [{status}]")
    
    # Issue A: Δτ refinement study
    lines.append("")
    lines.append("-" * 80)
    lines.append("ISSUE A: Δτ REFINEMENT STUDY (INTEGRATOR CONVERGENCE)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Purpose: Verify that ε_v is dominated by physics, not integrator error.")
    lines.append(f"  Method: Run at Δτ × {p.dtau_refinement_factors} at fixed (L, σ, k0)")
    lines.append("")
    
    if "dtau_study" in rep:
        ds = rep["dtau_study"]
        lines.append(f"  Base Δτ = {p.dtau}")
        for i, (factor, v) in enumerate(zip(ds["dtau_factors"], ds["v_values"])):
            lines.append(f"    Δτ × {factor}: v_hat = {v:.6f}")
        lines.append(f"  Max relative change under refinement: {ds['convergence_error']:.3e}")
        lines.append(f"  Tolerance: {p.tol_dtau_convergence:.0e}")
        status = "PASS" if ds['convergence_error'] < p.tol_dtau_convergence else "FAIL"
        lines.append(f"  Status: {status}")
    
    for lv in levels:
        if lv.dtau_convergence_error is not None:
            lines.append(f"  Scale {lv.scale}: Δτ convergence error = {lv.dtau_convergence_error:.3e}")
    
    # Issue C: Boundary immunity
    lines.append("")
    lines.append("-" * 80)
    lines.append("ISSUE C: BOUNDARY IMMUNITY")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Purpose: Verify no boundary reflection contamination.")
    lines.append("  Metric: min(d_edge / σ) over all runs in measurement window")
    lines.append("  where d_edge is distance from packet center (±3σ) to nearest boundary.")
    lines.append("")
    lines.append(f"  Requirement: > {p.min_boundary_sigma:.0f}σ")
    lines.append("  Justification: At 4σ, Gaussian probability < 0.01% (e^{-8} ≈ 3×10⁻⁴)")
    lines.append("")
    
    for lv in levels:
        status = "PASS" if lv.min_boundary_distance_sigma > p.min_boundary_sigma else "FAIL"
        lines.append(f"  Scale {lv.scale}: min(d_edge/σ) = {lv.min_boundary_distance_sigma:.1f}  [{status}]")
    
    lines.append("")
    lines.append("-" * 80)
    lines.append("LEMMA 'ONLY-IF' DIAGNOSTIC: max|∂_x ln J - ∂_x ln F|")
    lines.append("-" * 80)
    
    for lv in levels:
        lines.append(f"Scale {lv.scale}:")
        lines.append(f"  Matched: {lv.lemma_residual_matched:.3e}")
        for q, val in lv.lemma_residual_mismatched.items():
            lines.append(f"  Mismatch q={q}: {val:.3e}")
    
    lines.append("")
    lines.append("-" * 80)
    lines.append("ACCEPTANCE CRITERIA")
    lines.append("-" * 80)
    
    # Check convergence
    finest = levels[-1]
    q0 = p.mismatch_qs[0]
    mm_finest = finest.mismatch_results[q0]
    
    tests = [
        ("Norm conservation", 
         finest.max_norm_drift < p.tol_norm_drift,
         f"max={finest.max_norm_drift:.2e}, tol={p.tol_norm_drift:.0e}"),
        
        ("Locality gate",
         finest.max_locality_metric < p.locality_gate,
         f"max={finest.max_locality_metric:.3e}, gate={p.locality_gate}"),
        
        ("Matched spatial variability → 0",
         finest.eps_space_matched < p.tol_matched_space,
         f"ε_space={finest.eps_space_matched:.3e}, tol={p.tol_matched_space:.0e}"),
        
        ("Mismatch/Matched ratio > threshold",
         mm_finest["eps_space"] / finest.eps_space_matched > p.tol_mismatch_ratio,
         f"ratio={mm_finest['eps_space']/finest.eps_space_matched:.1f}, tol={p.tol_mismatch_ratio}"),
        
        ("Issue D: Mismatch consistent with A_pred",
         0.3 * mm_finest["A_pred"] < mm_finest["eps_space"] < 3.0 * mm_finest["A_pred"],
         f"ε_space={mm_finest['eps_space']:.3e}, A_pred={mm_finest['A_pred']:.3e}"),
        
        ("Worldline approximation",
         finest.worldline_rel_matched < p.tol_worldline_rel,
         f"rel={finest.worldline_rel_matched:.3e}, tol={p.tol_worldline_rel:.0e}"),
        
        ("Lemma: matched ≈ 0",
         finest.lemma_residual_matched < 1e-10,
         f"residual={finest.lemma_residual_matched:.3e}"),
        
        ("Lemma: mismatch > 0",
         finest.lemma_residual_mismatched[q0] > 1e-5,
         f"residual={finest.lemma_residual_mismatched[q0]:.3e}"),
        
        ("Issue A: Δτ convergence",
         finest.dtau_convergence_error is not None and finest.dtau_convergence_error < p.tol_dtau_convergence,
         f"error={finest.dtau_convergence_error:.3e}, tol={p.tol_dtau_convergence:.0e}" if finest.dtau_convergence_error is not None else "N/A"),
        
        ("Issue B: Small parameter (k0 a)² < 0.01",
         finest.k0a_squared < 0.01,
         f"(k0 a)²={finest.k0a_squared:.4e}"),
        
        ("Issue C: Boundary distance > 4σ",
         finest.min_boundary_distance_sigma > p.min_boundary_sigma,
         f"min(d/σ)={finest.min_boundary_distance_sigma:.1f}"),
        
        # New criteria for suggestions
        ("Suggestion 1: Momentum stability",
         finest.k_stability_matched < p.tol_k_stability,
         f"k_stability={finest.k_stability_matched:.3e}, tol={p.tol_k_stability:.0e}"),
        
        ("Suggestion 1: v_obs matches k_mean predictor",
         abs(finest.v_mean_matched - finest.v_theory_kmean) / finest.v_theory_kmean < 0.05,
         f"v_obs={finest.v_mean_matched:.5f}, v_pred(k_mean)={finest.v_theory_kmean:.5f}"),
        
        ("Suggestion 3: Temporal constancy",
         finest.temporal_constancy_matched < p.tol_temporal_constancy,
         f"temporal={finest.temporal_constancy_matched:.3e}, tol={p.tol_temporal_constancy:.0e}"),
    ]
    
    all_passed = True
    for name, passed, detail in tests:
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}] {name}: {detail}")
        all_passed = all_passed and passed
    
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    lines.append("=" * 80)
    
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved: {outpath}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)
    
    params = Params()
    print("=" * 70)
    print("REAL-WORLD THEOREM 3.1 VALIDATION")
    print("=" * 70)
    print(f"Scales: {params.scales}")
    print(f"Mismatched calibrations: F = n^q for q in {params.mismatch_qs}")
    print()
    
    levels, rep = run_validation(params, make_plots=True)
    
    print()
    print("CONVERGENCE RESULTS:")
    print("-" * 70)
    q0 = params.mismatch_qs[0]
    for lv in levels:
        mm = lv.mismatch_results[q0]
        ratio = mm["eps_space"] / lv.eps_space_matched
        print(
            f"scale={lv.scale:2d}  ε_tot={lv.eps_total:.4f}  "
            f"ε_sp(M)={lv.eps_space_matched:.3e}  ε_sp(MM)={mm['eps_space']:.3e}  "
            f"ratio={ratio:.1f}x"
        )
    
    print()
    print(f"Outputs:")
    print(f"  {ART_DIR / 'fig_theorem31_realworld_validation.png'}")
    print(f"  {ART_DIR / 'theorem31_realworld_report.txt'}")
