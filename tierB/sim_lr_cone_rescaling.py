"""
Tier-B Simulation B4: Cone/front rescaling under calibration (LR-style proxy).

Anchored paper result (Eq. \\ref{eq:constant-c}):
  c_local^(obs)(t) = c_local / F

We demonstrate the rescaling using a small transverse-field Ising chain:
  H = -J Σ σ^x_i σ^x_{i+1} - h Σ σ^z_i.

Diagnostic:
- Compute commutator norms ||[A(τ), B_r]|| where
    A = σ^z at site 0,  B_r = σ^z at site r.
- Define front time τ_front(r) as earliest τ where the commutator exceeds a threshold θ.
- Under constant calibration t = F τ, we expect the front speed in t to rescale by 1/F.

Notes for referees:
- This is a small-N dense-matrix demonstration meant as a sanity check.
  It is consistent with (and motivated by) the LR conversion logic, but does not attempt
  to reproduce the sharpest LR velocity optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from edcl.spinchain import (
    pauli, local_op, transverse_field_ising_H, eig_unitary,
    evolve_op, commutator, spectral_norm,
)

ART_DIR = Path(__file__).resolve().parents[1] / "paper_artifacts"


@dataclass
class Params:
    N: int = 5                 # keep small for speed and auditability
    J: float = 1.0
    h: float = 1.0
    tau_max: float = 6.0
    dtau: float = 0.10
    threshold: float = 0.10
    F_list: tuple[float, ...] = (1.0, 2.0)


def front_times_for_F(params: Params, F: float) -> dict:
    _, _, sz = pauli()
    H = transverse_field_ising_H(params.N, J=params.J, h=params.h)
    evals, evecs = eig_unitary(H)

    A = local_op(sz, 0, params.N)
    Bs = [local_op(sz, r, params.N) for r in range(params.N)]

    taus = np.arange(0.0, params.tau_max + params.dtau, params.dtau)

    norms = np.zeros((params.N, taus.size), dtype=float)
    for ti, tau in enumerate(taus):
        A_tau = evolve_op(A, evals, evecs, tau)
        for r in range(params.N):
            C = commutator(A_tau, Bs[r])
            norms[r, ti] = spectral_norm(C)

    tau_front = np.full(params.N, np.nan, dtype=float)
    for r in range(1, params.N):
        above = np.where(norms[r] >= params.threshold)[0]
        if above.size > 0:
            tau_front[r] = taus[above[0]]

    t_front = F * tau_front
    return {"taus": taus, "norms": norms, "tau_front": tau_front, "t_front": t_front, "F": F}


def fit_front_speed(dist: np.ndarray, t_front: np.ndarray) -> float:
    """
    Fit r ≈ v * t through origin using least squares on finite entries.
    """
    mask = np.isfinite(t_front) & (dist > 0)
    if mask.sum() < 2:
        return float("nan")
    x = t_front[mask]
    y = dist[mask]
    v = float(np.dot(x, y) / np.dot(x, x))
    return v


def make_figure(results: list[dict], outpath: Path) -> None:
    fig = plt.figure(figsize=(14, 5), dpi=160)
    gs = fig.add_gridspec(1, len(results))

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        taus = r["taus"]
        norms = r["norms"]
        im = ax.imshow(
            norms[1:, :],
            aspect="auto",
            origin="lower",
            extent=[taus[0], taus[-1], 1, norms.shape[0]-1],
        )
        ax.set_title(f"F={r['F']}: ||[A(τ),B_r]||")
        ax.set_xlabel("τ")
        ax.set_ylabel("distance r")
        tf = r["tau_front"]
        rr = np.arange(tf.size)
        ax.plot(tf[1:], rr[1:], marker="o", linestyle="None", alpha=0.9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Commutator front (proxy): observer-time rescaling checked via constant F", y=1.02)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def run(params: Params) -> dict:
    results = [front_times_for_F(params, F) for F in params.F_list]

    dist = np.arange(params.N, dtype=float)
    v_tau = fit_front_speed(dist, results[0]["tau_front"])
    v_t = fit_front_speed(dist, results[1]["t_front"])
    ratio = v_t / v_tau

    return {"params": params, "results": results, "v_tau": v_tau, "v_t": v_t, "ratio": ratio}


if __name__ == "__main__":
    ART_DIR.mkdir(parents=True, exist_ok=True)
    p = Params()
    d = run(p)
    out = ART_DIR / "fig_lr_cone_rescaling.png"
    make_figure(d["results"], out)
    print("=== LR-cone/front rescaling (proxy) ===")
    print(f"N={p.N}, J={p.J}, h={p.h}, threshold={p.threshold}, dtau={p.dtau}, tau_max={p.tau_max}")
    print(f"F={p.F_list[0]}: v_front(τ)={d['v_tau']:.3f}")
    print(f"F={p.F_list[1]}: v_front(t)={d['v_t']:.3f}")
    print(f"ratio v_t/v_tau={d['ratio']:.3f} (expected ~ {1.0/p.F_list[1]:.3f})")
    print(f"Wrote: {out}")
