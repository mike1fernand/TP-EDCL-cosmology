"""
Small spin-chain utilities for LR-cone / commutator-front demonstrations.

We build dense matrices and keep N small (default N<=6) for speed and transparency.
"""

from __future__ import annotations

import numpy as np


def pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return sx, sy, sz


def kron_n(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def local_op(op: np.ndarray, site: int, N: int) -> np.ndarray:
    I = np.eye(2, dtype=np.complex128)
    ops = [I] * N
    ops[site] = op
    return kron_n(ops)


def two_site_op(op1: np.ndarray, site1: int, op2: np.ndarray, site2: int, N: int) -> np.ndarray:
    I = np.eye(2, dtype=np.complex128)
    ops = [I] * N
    ops[site1] = op1
    ops[site2] = op2
    return kron_n(ops)


def transverse_field_ising_H(N: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:
    """
    H = -J Σ_{i=0}^{N-2} σ^x_i σ^x_{i+1} - h Σ_{i=0}^{N-1} σ^z_i
    """
    sx, _, sz = pauli()
    H = np.zeros((2**N, 2**N), dtype=np.complex128)
    for i in range(N - 1):
        H += -J * two_site_op(sx, i, sx, i + 1, N)
    for i in range(N):
        H += -h * local_op(sz, i, N)
    return H


def eig_unitary(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (evals, evecs) for Hermitian H such that H = V diag(evals) V†.
    """
    evals, evecs = np.linalg.eigh(H)
    return evals.astype(float), evecs.astype(np.complex128)


def U_of_t(evals: np.ndarray, evecs: np.ndarray, t: float) -> np.ndarray:
    ph = np.exp(-1j * evals * t)
    return (evecs * ph) @ evecs.conjugate().T


def evolve_op(A: np.ndarray, evals: np.ndarray, evecs: np.ndarray, t: float) -> np.ndarray:
    """
    A(t) = U† A U
    """
    U = U_of_t(evals, evecs, t)
    return U.conjugate().T @ A @ U


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def spectral_norm(M: np.ndarray) -> float:
    """
    Operator 2-norm ||M||_2.

    We compute ||M||_2 = sqrt(λ_max(M† M)), which avoids full SVD and is faster for repeated calls.
    """
    H = M.conjugate().T @ M
    evals = np.linalg.eigvalsh(H)
    return float(np.sqrt(np.max(evals)))
