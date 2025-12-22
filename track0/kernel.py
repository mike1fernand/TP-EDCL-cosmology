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
- 'paper_claim_exp':       (a'/a)^2 * exp(-z(a')/ζ)         [as written in Eq. (kernel-shape)]
- 'paper_equation_1mexp':  (a'/a)^2 * (1 - exp(-z(a')/ζ))   [alternate saturating form; NOT used in paper]

A referee will expect you to resolve this ambiguity in the manuscript. This code is designed to make that
inconsistency explicit and testable.

Performance note:
The Tier-0 and unit-test suites evaluate δ(a) at many a values. A naive implementation that re-integrates
the kernel on every call is O(N_a * N_grid) and can appear to "hang" for large N. KernelNormalization therefore
precomputes a cumulative integral on a log-spaced grid and answers I(a) queries by interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math

import numpy as np



# Compatibility: numpy>=2 prefers np.trapezoid; older versions may only have np.trapz
_TRAPZ = getattr(np, 'trapezoid', np.trapz)
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
    if variant == "paper_claim_exp":
        return pref * math.exp(-z / zeta)
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

    This function is intended for single-shot evaluation. For repeated queries across many a values,
    use KernelNormalization.I(a), which caches the cumulative integral.
    """
    if a_i <= 0:
        raise ValueError("a_i must be positive.")
    if a <= a_i:
        return 0.0
    if zeta <= 0:
        raise ValueError("zeta must be positive.")
    if n_log < 1000:
        raise ValueError("n_log too small for stable quadrature; use >= 1000.")

    xs = np.linspace(math.log(a_i), math.log(a), n_log)
    a_primes = np.exp(xs)

    # vectorized evaluation
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

    Performance:
      The expensive part is the integral over a'. We precompute a cumulative trapezoidal integral on
      a log-spaced grid in log(a') once in __post_init__, then answer I(a) queries by interpolation.
      This makes Track-0 scripts and unit tests fast and deterministic.
    """

    a0: float = 1.0
    a_i: float = 1e-4
    zeta: float = 0.5
    f_norm_target: float = 0.7542
    variant: KernelVariant = "paper_claim_exp"
    n_log: int = 20000

    # Cached integration grid and cumulative integral of a'^2 * g(z(a')) over log(a').
    _xs: np.ndarray = field(init=False, repr=False, compare=False)
    _cumJ: np.ndarray = field(init=False, repr=False, compare=False)
    _J_end: float = field(init=False, repr=False, compare=False)
    _C: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.a0 <= 0:
            raise ValueError("a0 must be positive.")
        if self.a_i <= 0:
            raise ValueError("a_i must be positive.")
        if self.a0 <= self.a_i:
            raise ValueError("a0 must be greater than a_i for a nontrivial integral.")
        if self.zeta <= 0:
            raise ValueError("zeta must be positive.")
        if self.n_log < 1000:
            raise ValueError("n_log too small for stable quadrature; use >= 1000.")

        xs = np.linspace(math.log(self.a_i), math.log(self.a0), int(self.n_log))
        a_primes = np.exp(xs)
        z_vals = (1.0 / a_primes) - 1.0

        if self.variant == "paper_equation_1mexp":
            g = 1.0 - np.exp(-z_vals / self.zeta)
        elif self.variant == "paper_claim_exp":
            g = np.exp(-z_vals / self.zeta)
        else:
            raise ValueError(f"Unknown kernel variant: {self.variant}")

        # J(x) = ∫ a'^2 g(z(a')) d(log a')  from log(a_i) to x.
        integrand = (a_primes ** 2) * g
        dx = np.diff(xs)
        cum = np.zeros_like(xs)
        # cumulative trapezoid integral
        cum[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dx)

        J_end = float(cum[-1])
        if not math.isfinite(J_end) or J_end <= 0.0:
            raise RuntimeError("Kernel integral is non-positive or non-finite; cannot normalize.")

        # I_shape(a0) = J_end / a0^2 ; choose C so that C * I_shape(a0) = f_norm_target.
        C = float(self.f_norm_target) * (float(self.a0) ** 2) / J_end

        object.__setattr__(self, "_xs", xs)
        object.__setattr__(self, "_cumJ", cum)
        object.__setattr__(self, "_J_end", J_end)
        object.__setattr__(self, "_C", C)

    def constant_C(self) -> float:
        """Return the multiplicative constant C that enforces the chosen normalization."""
        return float(self._C)

    def f_norm(self) -> float:
        """Compute f_norm under this normalization (should match f_norm_target)."""
        # I(a0) = C * (J_end / a0^2)
        return float(self._C) * (float(self._J_end) / (float(self.a0) ** 2))

    def _J_at(self, a: float) -> float:
        """Return J(a) = ∫_{a_i}^{a} a'^2 g d(log a') (unnormalized), using interpolation."""
        if a <= self.a_i:
            return 0.0
        if a > self.a0:
            raise ValueError("KernelNormalization._J_at expects a <= a0.")
        x = math.log(a)
        return float(np.interp(x, self._xs, self._cumJ))

    def I(self, a: float) -> float:
        """Compute the normalized I(a) = ∫ K(a,a') dlog a' with the same global C."""
        if a <= self.a_i:
            return 0.0
        if a > self.a0:
            raise ValueError("KernelNormalization.I expects a <= a0.")
        J = self._J_at(a)
        return float(self._C) * (J / (float(a) ** 2))


def delta_of_a_highz_limit(
    a: float,
    delta0: float,
    norm: KernelNormalization,
) -> float:
    """
    High-z-limit δ(a) with δ0=δ(a0) fixed.

    In that limit, δ(a) ∝ I(a), and δ0 = δ(a0) = delta0 by definition.
    We therefore set:

        δ(a) = δ0 * I(a) / I(a0)

    Under the normalization used here, I(a0) == f_norm_target, so the implementation
    uses that constant directly.

    This matches Eq. (delta-kernel) under:
      κ_tick = 1/12, <R_eff>→12/ℓ0^2  (so ℓ0 cancels) and fixed kernel normalization.

    Note: This is for Track-0 "kernel-only" reproduction/diagnostics.
    """
    if norm.f_norm_target <= 0:
        raise ValueError("norm.f_norm_target must be positive.")
    return float(delta0) * (norm.I(a) / float(norm.f_norm_target))


def hubble_ratio_from_delta(delta: float) -> float:
    """
    Minimal background mapping used in Track-0 plots:
        H_TP/H_GR = 1 + δ(a).

    This is NOT claimed to be the exact likelihood implementation unless the patched CLASS code
    uses the same mapping. It is provided as a "math-layer" reproducibility hook.
    """
    return 1.0 + float(delta)
