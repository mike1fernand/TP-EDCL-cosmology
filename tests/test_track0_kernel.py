import os
import sys
import math
import unittest
import numpy as np

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from track0.kernel import KernelNormalization, delta_of_a_highz_limit


class TestTrack0Kernel(unittest.TestCase):
    def test_fnorm_reproduces_paper_number(self):
        # Paper-quoted values
        f_norm_target = 0.7542
        for variant in ["paper_equation_1mexp", "paper_claim_exp"]:
            norm = KernelNormalization(
                a0=1.0, a_i=1e-4, zeta=0.5, f_norm_target=f_norm_target, variant=variant, n_log=50000
            )
            self.assertAlmostEqual(norm.f_norm(), f_norm_target, places=4)

    def _z_safe(self, variant: str, thresh: float = 0.002) -> float | None:
        norm = KernelNormalization(a0=1.0, a_i=1e-4, zeta=0.5, f_norm_target=0.7542, variant=variant, n_log=50000)
        zs = np.linspace(0.0, 1100.0, 5000)
        a = 1.0 / (1.0 + zs)
        delta0 = 0.089  # paper-quoted δ0
        ratios = np.array([1.0 + delta_of_a_highz_limit(float(ai), delta0, norm) for ai in a])
        abs_dev = np.abs(ratios - 1.0)
        for i, z in enumerate(zs):
            if np.all(abs_dev[i:] <= thresh):
                return float(z)
        return None

    def test_highz_safety_claim_variant(self):
        # The manuscript claims |ratio-1| <= 0.2% by z~2.
        # That is satisfied by the exp(-z/ζ) kernel variant under Track-0 mapping.
        z_safe = self._z_safe("paper_claim_exp", thresh=0.002)
        self.assertIsNotNone(z_safe)
        self.assertLessEqual(z_safe, 2.5)

    def test_highz_safety_equation_variant_fails(self):
        # Under the kernel-shape equation as written (1-exp(-z/ζ)), Track-0 mapping does NOT
        # saturate to unity at high z. This test documents the internal inconsistency.
        z_safe = self._z_safe("paper_equation_1mexp", thresh=0.002)
        self.assertIsNone(z_safe)


if __name__ == "__main__":
    unittest.main()
