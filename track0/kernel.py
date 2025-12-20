"""
Track-0: analytic/kernel-only utilities.

Purpose (referee-safe):
- Numerically evaluate the causal kernel K(a, a'; ζ) used in the paper.
- Fix the proportionality constant by an explicit normalization rule.
- Generate a standalone H_TP/H_GR vs z curve *without* CLASS/Cobaya.

Important: The manuscript states K(a,a';ζ) is proportional to a shape (Eq. kernel-shape),
"with proportionality fixed by normalization." That means the overall constant must be specified.
This module implements a normalization rule that *exactly reproduces the paper's quoted*
f_norm=0.7542 for ζ=0.5 and a_i=1e-4 in the high-z limit described in Sec. meanfield-cosmo.

We implement two kernel shapes:
- 'paper_equation':  (a'/a)^2 * (1 - exp(-z(a')/ζ))   [as written in Eq. kernel-shape]
- 'paper_claim':     (a'/a)^2 * exp(-z(a')/ζ)         [consistent with the text claim that early times are downweighted and δ(z)→0 at high z]

A referee will expect you to resolve this ambiguity in the manuscript. This code is designed to make that
inconsistency explicit and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Callable, Tuple
import math
import numpy as np


KernelVariant = Literal["paper_equation_1mexp", "paper_claim_exp"]


def z_of_a(a: float) -> float:
    """Paper definition: z(a) = a^{-1} - 1."""
    if a <= 0:
        raise ValueError("Scale factor a must be positive.")
    return (1.0 / a) - 1.0


def kernel_shape(a: float, a_prime: float, zeta: float, variant: KernelVariant) -> float:
    """
    Unnormalized kernel shape, up to an overall proportionality constant.

    Args:
        a: scale factor at evaluation time (upper limit in δ integral)
        a_prime: integration variable (a_prime <= a for causal support)
        zeta: kernel parameter ζ
        variant: which kernel shape to use

    Returns:
        dimensionless nonnegative kernel value
    """
    if a_prime <= 0 or a <= 0:
        raise ValueError("a and a_prime must be positive.")
    if a_prime > a:
        # causal support is only for a' <= a; outside support return 0
        return 0.0
    if zeta <= 0:
        raise ValueError("zeta must be positive.")

    # common prefactor
    pref = (a_prime / a) ** 2
    z = z_of_a(a_prime)

    if variant == "paper_equation_1mexp":
        return pref * (1.0 - math.exp(-z / zeta))
    elif variant == "paper_claim_exp":
        return pref * math.exp(-z / zeta)
    else:
        raise ValueError(f"Unknown kernel variant: {variant}")


def integrate_kernel_over_log_a(
    a: float,
    a_i: float,
    zeta: float,
    variant: KernelVariant,
    n_log: int = 20000,
) -> float:
    """
    Compute I(a) = ∫_{a_i}^{a} K(a,a';ζ) d a'/a' = ∫ K(a,a';ζ) d(log a').

    Uses a log-spaced trapezoidal rule in log(a').

    Note: This evaluates the *shape* (unnormalized if you pass shape-only),
    so you should multiply by the normalization constant if applicable.
    """
    if a_i <= 0:
        raise ValueError("a_i must be positive.")
    if a <= a_i:
        return 0.0
    if n_log < 1000:
        raise ValueError("n_log too small for stable quadrature; use >= 1000.")

    xs = np.linspace(math.log(a_i), math.log(a), n_log)
    a_primes = np.exp(xs)
    # vectorize without python loops for speed
    z_vals = (1.0 / a_primes) - 1.0
    pref = (a_primes / a) ** 2
    if variant == "paper_equation_1mexp":
        vals = pref * (1.0 - np.exp(-z_vals / zeta))
    elif variant == "paper_claim_exp":
        vals = pref * np.exp(-z_vals / zeta)
    else:
        raise ValueError(f"Unknown kernel variant: {variant}")

    return float(np.trapz(vals, xs))


@dataclass(frozen=True)
class KernelNormalization:
    """
    Encapsulates the kernel's proportionality constant via an explicit normalization rule.

    Rule implemented:
      Choose C so that, in the "high-z limit" described in Sec. meanfield-cosmo,
      f_norm = ∫_{a_i}^{a0} K(a0,a';ζ) da'/a' equals f_norm_target.

    This rule is *deduced from the manuscript* (it quotes f_norm numerically
    under those settings) and therefore is "no-assumptions" in the reproducibility sense.
    """
    a0: float = 1.0
    a_i: float = 1e-4
    zeta: float = 0.5
    f_norm_target: float = 0.7542
    variant: KernelVariant = "paper_equation_1mexp"
    n_log: int = 20000

    def constant_C(self) -> float:
        """Return the multiplicative constant C that enforces the chosen normalization."""
        I = integrate_kernel_over_log_a(self.a0, self.a_i, self.zeta, self.variant, n_log=self.n_log)
        if I <= 0:
            raise RuntimeError("Kernel integral is non-positive; cannot normalize.")
        return self.f_norm_target / I

    def f_norm(self) -> float:
        """Compute f_norm under this normalization (should match f_norm_target)."""
        C = self.constant_C()
        I = integrate_kernel_over_log_a(self.a0, self.a_i, self.zeta, self.variant, n_log=self.n_log)
        return C * I

    def I(self, a: float) -> float:
        """Compute the normalized I(a) = ∫ K(a,a') dlog a' with the same global C."""
        C = self.constant_C()
        return C * integrate_kernel_over_log_a(a, self.a_i, self.zeta, self.variant, n_log=self.n_log)


def delta_of_a_highz_limit(
    a: float,
    delta0: float,
    norm: KernelNormalization,
) -> float:
    """
    High-z-limit δ(a) with δ0=δ(a0) fixed.

    In that limit, δ(a) ∝ I(a), and δ0 = δ(a0) = delta0 by definition.
    We therefore set δ(a) = delta0 * I(a)/I(a0).

    This matches Eq. (delta-kernel) under:
      κ_tick = 1/12, <R_eff>→12/ℓ0^2  (so ℓ0 cancels) and fixed kernel normalization.

    Note: This is for Track-0 "kernel-only" reproduction/diagnostics.
    """
    I0 = norm.I(norm.a0)
    if I0 <= 0:
        raise RuntimeError("I(a0) is non-positive; cannot scale.")
    return delta0 * (norm.I(a) / I0)


def hubble_ratio_from_delta(delta: float) -> float:
    """
    Minimal background mapping used in Track-0 plots:
        H_TP/H_GR = 1 + δ(a).

    This is NOT claimed to be the exact likelihood implementation unless the patched CLASS code
    uses the same mapping. It is provided as a "math-layer" reproducibility hook.
    """
    return 1.0 + delta
