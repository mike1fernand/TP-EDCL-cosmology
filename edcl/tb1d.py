"""
1D tight-binding utilities used by Tier-B simulations.

We represent a single-particle tight-binding Hamiltonian on an open chain:
  H_{i,i} = onsite[i] (default 0)
  H_{i,i+1} = H_{i+1,i} = -J_bond[i]   for i=0..N-2

Structure-time Schr. equation: i dψ/dτ = H ψ.

We evolve with Crank–Nicolson:
 (I + iΔτ H/2) ψ_{n+1} = (I - iΔτ H/2) ψ_n,
which is norm-preserving for Hermitian H (up to solver error).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    """Complex Gaussian wavepacket ψ(x) ∝ exp(-(x-x0)^2/(4σ^2)) * exp(i k0 x)."""
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)
    psi = psi.astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def momentum_stats_periodic(psi: np.ndarray, a: float = 1.0) -> tuple[float, float]:
    """
    Approximate mean k and std(k) using a periodic FFT convention.
    This is adequate when ψ is localized away from boundaries.
    Returns (k_mean, k_std) in radians / length.
    """
    N = psi.size
    psi_k = np.fft.fftshift(np.fft.fft(psi))
    pk = (np.abs(psi_k) ** 2)
    pk = pk / pk.sum()
    k = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=a))
    k_mean = float((pk * k).sum())
    k2 = float((pk * (k ** 2)).sum())
    k_std = float(np.sqrt(max(k2 - k_mean ** 2, 0.0)))
    return k_mean, k_std


def bond_current(psi: np.ndarray, J_bond: np.ndarray) -> np.ndarray:
    """
    Bond current (paper Eq. bond-current):
      J_{i+1/2} = 2 Im( J_i ψ_i^* ψ_{i+1} )
    for i=0..N-2 where J_i is the coupling on bond (i,i+1).
    """
    return 2.0 * np.imag(J_bond * np.conjugate(psi[:-1]) * psi[1:])


@dataclass
class TridiagSolver:
    """
    Pre-factored Thomas algorithm for a fixed complex tridiagonal system A x = d.
    A has lower diag a (len N-1), diag b (len N), upper diag c (len N-1).
    """
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    c_prime: np.ndarray
    denom: np.ndarray

    @classmethod
    def from_tridiagonals(cls, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> "TridiagSolver":
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        c = np.asarray(c, dtype=np.complex128)
        N = b.size
        if a.size != N - 1 or c.size != N - 1:
            raise ValueError("Tridiagonal lengths inconsistent.")
        c_prime = np.zeros(N - 1, dtype=np.complex128)
        denom = np.zeros(N, dtype=np.complex128)
        denom[0] = b[0]
        if denom[0] == 0:
            raise ZeroDivisionError("Zero pivot at 0.")
        c_prime[0] = c[0] / denom[0]
        for i in range(1, N - 1):
            denom[i] = b[i] - a[i - 1] * c_prime[i - 1]
            if denom[i] == 0:
                raise ZeroDivisionError(f"Zero pivot at {i}.")
            c_prime[i] = c[i] / denom[i]
        denom[N - 1] = b[N - 1] - a[N - 2] * c_prime[N - 2]
        if denom[N - 1] == 0:
            raise ZeroDivisionError("Zero pivot at N-1.")
        return cls(a=a, b=b, c=c, c_prime=c_prime, denom=denom)

    def solve(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=np.complex128)
        N = self.b.size
        if d.size != N:
            raise ValueError("RHS length mismatch.")
        d_prime = np.zeros(N, dtype=np.complex128)
        d_prime[0] = d[0] / self.denom[0]
        for i in range(1, N):
            d_prime[i] = (d[i] - self.a[i - 1] * d_prime[i - 1]) / self.denom[i]
        x = np.zeros(N, dtype=np.complex128)
        x[N - 1] = d_prime[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = d_prime[i] - self.c_prime[i] * x[i + 1]
        return x


def build_tridiagonal_H(J_bond: np.ndarray, onsite: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build tridiagonal arrays for H:
      diag h0 (len N)
      offdiag h1 (len N-1) such that H_{i,i+1}=H_{i+1,i}=h1[i] = -J_bond[i]
    """
    J_bond = np.asarray(J_bond, dtype=float)
    N = J_bond.size + 1
    h1 = -J_bond.astype(float)
    if onsite is None:
        h0 = np.zeros(N, dtype=float)
    else:
        onsite = np.asarray(onsite, dtype=float)
        if onsite.size != N:
            raise ValueError("onsite length mismatch")
        h0 = onsite
    return h0, h1


def crank_nicolson_stepper(h0: np.ndarray, h1: np.ndarray, dt: float) -> tuple[TridiagSolver, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute solver for A = I + i dt H/2 and B = I - i dt H/2.

    Returns (solver_A, B_diag, B_upper, B_lower) where
      rhs = B_diag*ψ + B_upper*ψ_{i+1} + B_lower*ψ_{i-1}.
    """
    h0 = np.asarray(h0, dtype=float)
    h1 = np.asarray(h1, dtype=float)
    N = h0.size
    if h1.size != N - 1:
        raise ValueError("h1 length mismatch")
    # A tridiagonal
    A_diag = (1.0 + 1j * dt * h0 / 2.0).astype(np.complex128)
    A_upper = (1j * dt * h1 / 2.0).astype(np.complex128)   # len N-1
    A_lower = (1j * dt * h1 / 2.0).astype(np.complex128)   # symmetric
    solver = TridiagSolver.from_tridiagonals(A_lower, A_diag, A_upper)
    # B coefficients for rhs
    B_diag = (1.0 - 1j * dt * h0 / 2.0).astype(np.complex128)
    B_upper = (-1j * dt * h1 / 2.0).astype(np.complex128)
    B_lower = (-1j * dt * h1 / 2.0).astype(np.complex128)
    return solver, B_diag, B_upper, B_lower


def cn_step(psi: np.ndarray, solver_A: TridiagSolver, B_diag: np.ndarray, B_upper: np.ndarray, B_lower: np.ndarray) -> np.ndarray:
    """
    One Crank–Nicolson step given precomputed solver for A and B coefficients.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    rhs = B_diag * psi
    rhs[:-1] += B_upper * psi[1:]
    rhs[1:] += B_lower * psi[:-1]
    return solver_A.solve(rhs)
