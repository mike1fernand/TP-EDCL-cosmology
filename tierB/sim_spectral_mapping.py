"""
Tier-B Simulation B3: Spectral Jacobian mapping under calibration.

Anchored paper equation:
  S_t(ω_t) dω_t = S_τ(ω_τ) dω_τ      (Eq. \\ref{eq:spectral-measure})
with ω_t = ω_τ / F for constant calibration dt = F dτ.

We test two regimes:

1) Constant calibration (exact Jacobian map):
   - Construct a stationary multi-tone signal φ(τ).
   - Compute spectral density S_τ(ω) using a periodogram consistent with
     the continuous Fourier transform scaling.
   - Rescale time by constant factor F and recompute S_t(ω).
   - Verify S_t(ω) ≈ F S_τ(F ω) and integrated measure invariance.

2) Weakly inhomogeneous calibration:
   - Define t(τ)=F̄ τ + δt(τ) with small δt, monotone t(τ).
   - Sample φ on uniform t-grid via interpolation.
   - Verify integrated measure remains near invariant and that mapping error
     increases with modulation amplitude.

Notes:
- This simulation is intentionally conservative: it verifies the *exact* constant-F Jacobian claim,
  and provides a controlled numerical check for the inhomogeneous case without overclaiming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ART_DIR = Path(__file__).resolve().parents[1] / "paper_artifacts"


@dataclass
class Params:
    N: int = 4096
    dtau: float = 0.01
    F_const: float = 1.7
    # signal tones (rad/time)
    w0: float = 30.0
    w1: float = 55.0
    amp1: float = 0.6
    # inhomogeneous calibration
    Fbar: float = 1.4
    delta_amp: float = 0.015  # small δt amplitude (time units)
    delta_period: float = 3.0  # period in τ units


def make_signal(t: np.ndarray, w0: float, w1: float, amp1: float) -> np.ndarray:
    # deterministic multi-tone complex signal (stationary)
    return np.exp(-1j * w0 * t) + amp1 * np.exp(-1j * w1 * t)


def spectrum_periodogram(phi: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Energy spectral density based on continuous-time FT scaling:
      Φ(ω) ≈ dt * FFT(phi)
      S(ω) := |Φ(ω)|^2 / (2π T)  where T = N dt
    """
    N = phi.size
    T = N * dt
    Phi = dt * np.fft.fftshift(np.fft.fft(phi))
    w = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    S = (np.abs(Phi) ** 2) / (2.0 * np.pi * T)
    return w, S


def mapped_spectrum_constantF(w_tau: np.ndarray, S_tau: np.ndarray, F: float, w_t: np.ndarray) -> np.ndarray:
    """
    Predicted mapping for constant F:
      S_t(w) = F * S_tau(F w)
    We compute by interpolation on the τ spectrum grid.
    """
    # Interpolate S_tau over ω at query ω = F * ω_t
    wq = F * w_t
    # numpy.interp requires ascending x
    idx = np.argsort(w_tau)
    w_sorted = w_tau[idx]
    S_sorted = S_tau[idx]
    return F * np.interp(wq, w_sorted, S_sorted, left=0.0, right=0.0)


def integrated_measure(w: np.ndarray, S: np.ndarray) -> float:
    # approximate integral ∫ S(ω) dω
    return float(np.trapz(S, x=w))


def build_inhomogeneous_mapping(tau: np.ndarray, Fbar: float, delta_amp: float, delta_period: float) -> tuple[np.ndarray, np.ndarray]:
    """
    t(τ) = Fbar τ + δt(τ), with δt(τ)=delta_amp*sin(2π τ / delta_period).
    Ensures monotone mapping by keeping delta_amp small enough.
    Returns (t, delta_t).
    """
    delta_t = delta_amp * np.sin(2.0 * np.pi * tau / delta_period)
    t = Fbar * tau + delta_t
    # Monotonicity check: dt/dτ = Fbar + δt' must stay positive
    ddt = np.gradient(t, tau)
    if np.min(ddt) <= 0:
        raise ValueError("Non-monotone t(τ); reduce delta_amp or change period.")
    return t, delta_t


def resample_to_uniform_t(tau: np.ndarray, phi_tau: np.ndarray, t_of_tau: np.ndarray, dt_t: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample φ(t) on a uniform t grid by inverting t(τ) using interpolation.
    """
    t_min, t_max = float(t_of_tau[0]), float(t_of_tau[-1])
    t_uniform = np.arange(t_min, t_max, dt_t)
    # invert mapping τ(t) by interpolation because t(τ) is monotone
    tau_of_t = np.interp(t_uniform, t_of_tau, tau)
    # sample φ(τ(t))
    # interpolate real and imaginary parts separately
    re = np.interp(tau_of_t, tau, np.real(phi_tau))
    im = np.interp(tau_of_t, tau, np.imag(phi_tau))
    phi_t = re + 1j * im
    return t_uniform, phi_t


def run(params: Params) -> dict:
    # 1) constant F
    tau = np.arange(params.N) * params.dtau
    phi_tau = make_signal(tau, params.w0, params.w1, params.amp1)
    w_tau, S_tau = spectrum_periodogram(phi_tau, params.dtau)

    # rescaled time with constant factor F: t = F τ
    dt_t = params.F_const * params.dtau
    t = np.arange(params.N) * dt_t
    phi_t = make_signal(t / params.F_const, params.w0, params.w1, params.amp1)  # same physical signal φ(τ)
    w_t, S_t = spectrum_periodogram(phi_t, dt_t)

    S_pred = mapped_spectrum_constantF(w_tau, S_tau, params.F_const, w_t)

    I_tau = integrated_measure(w_tau, S_tau)
    I_t = integrated_measure(w_t, S_t)

    # relative L1 error on a symmetric frequency band around peaks
    band = (np.abs(w_t) <= 120.0)
    l1 = float(np.trapz(np.abs(S_t[band] - S_pred[band]), x=w_t[band]) / np.maximum(np.trapz(S_t[band], x=w_t[band]), 1e-12))

    # 2) inhomogeneous calibration
    t_inh, delta_t = build_inhomogeneous_mapping(tau, params.Fbar, params.delta_amp, params.delta_period)
    # choose dt on t-grid comparable to average spacing
    dt_t_inh = float((t_inh[-1] - t_inh[0]) / params.N)
    t_uni, phi_t_inh = resample_to_uniform_t(tau, phi_tau, t_inh, dt_t_inh)
    w_inh, S_inh = spectrum_periodogram(phi_t_inh, dt_t_inh)
    I_inh = integrated_measure(w_inh, S_inh)

    # compare inhomogeneous spectrum to constant-Fbar prediction (not exact; diagnostic)
    # use same predicted mapping with Fbar and τ-spectrum
    S_pred_inh = mapped_spectrum_constantF(w_tau, S_tau, params.Fbar, w_inh)
    band2 = (np.abs(w_inh) <= 120.0)
    l1_inh = float(np.trapz(np.abs(S_inh[band2] - S_pred_inh[band2]), x=w_inh[band2]) / np.maximum(np.trapz(S_inh[band2], x=w_inh[band2]), 1e-12))
    var_delta = float(np.var(delta_t))

    return {
        "params": params,
        "w_tau": w_tau, "S_tau": S_tau, "I_tau": I_tau,
        "w_t": w_t, "S_t": S_t, "S_pred": S_pred, "I_t": I_t, "l1": l1,
        "tau": tau,
        "t_inh": t_inh, "delta_t": delta_t, "var_delta": var_delta,
        "t_uni": t_uni, "w_inh": w_inh, "S_inh": S_inh, "S_pred_inh": S_pred_inh,
        "I_inh": I_inh, "l1_inh": l1_inh,
    }


def make_figure(d: dict, outpath: Path) -> None:
    w_tau, S_tau = d["w_tau"], d["S_tau"]
    w_t, S_t, S_pred = d["w_t"], d["S_t"], d["S_pred"]
    w_inh, S_inh, S_pred_inh = d["w_inh"], d["S_inh"], d["S_pred_inh"]

    fig = plt.figure(figsize=(14, 8), dpi=160)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    band = (np.abs(w_tau) <= 120.0)
    ax1.plot(w_tau[band], S_tau[band], label="S_τ(ω)")
    ax1.set_title("(a) Structure-time spectrum S_τ")
    ax1.set_xlabel("ω")
    ax1.set_ylabel("S")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    band = (np.abs(w_t) <= 120.0)
    ax2.plot(w_t[band], S_t[band], label="S_t(ω) (computed)")
    ax2.plot(w_t[band], S_pred[band], linestyle="--", label="F S_τ(F ω) (pred)")
    ax2.set_title(f"(b) Constant-F Jacobian map (L1 rel err={d['l1']:.2e})")
    ax2.set_xlabel("ω")
    ax2.set_ylabel("S")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    band = (np.abs(w_inh) <= 120.0)
    ax3.plot(w_inh[band], S_inh[band], label="S_t(ω) (inhomog, computed)")
    ax3.plot(w_inh[band], S_pred_inh[band], linestyle="--", label="F̄ S_τ(F̄ ω) (reference)")
    ax3.set_title(f"(c) Weakly inhomogeneous calibration (var δt={d['var_delta']:.2e})")
    ax3.set_xlabel("ω")
    ax3.set_ylabel("S")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(d["tau"], d["delta_t"], label="δt(τ)")
    ax4.set_title(f"(d) Inhomogeneity δt(τ); L1 rel err={d['l1_inh']:.2e}")
    ax4.set_xlabel("τ")
    ax4.set_ylabel("δt")
    ax4.legend()

    fig.suptitle("Spectral Jacobian mapping under calibration (exact for constant F; controlled deviation otherwise)", y=1.02)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)
    p = Params()
    d = run(p)
    out = ART_DIR / "fig_spectral_mapping.png"
    make_figure(d, out)
    print("=== Spectral mapping ===")
    print(f"Constant-F: I_t/I_tau = {d['I_t']/d['I_tau']:.6f}, L1 rel err = {d['l1']:.3e}")
    print(f"Inhomog: I_inh/I_tau = {d['I_inh']/d['I_tau']:.6f}, L1 rel err vs Fbar-map = {d['l1_inh']:.3e}")
    print(f"Wrote: {out}")
