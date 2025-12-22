import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
import numpy as np

from tierB import sim_spectral_mapping as simspec


class TestSpectralMapping(unittest.TestCase):
    def test_constantF_jacobian(self):
        p = simspec.Params(N=2048, dtau=0.01, F_const=1.7, w0=25.0, w1=45.0, amp1=0.7, Fbar=1.4, delta_amp=0.01, delta_period=3.0)
        d = simspec.run(p)

        # Integrated spectral measure approximately invariant
        self.assertLess(abs(d["I_t"]/d["I_tau"] - 1.0), 1e-3)

        # Constant-F Jacobian map should be accurate
        self.assertLess(d["l1"], 2e-2)

    def test_inhomogeneous_error_grows(self):
        p1 = simspec.Params(N=2048, dtau=0.01, F_const=1.5, w0=25.0, w1=45.0, amp1=0.7, Fbar=1.3, delta_amp=0.005, delta_period=2.7)
        p2 = simspec.Params(N=2048, dtau=0.01, F_const=1.5, w0=25.0, w1=45.0, amp1=0.7, Fbar=1.3, delta_amp=0.02, delta_period=2.7)
        d1 = simspec.run(p1)
        d2 = simspec.run(p2)

        # Integrated measure should remain near invariant (looser under interpolation)
        self.assertLess(abs(d1["I_inh"]/d1["I_tau"] - 1.0), 5e-2)
        self.assertLess(abs(d2["I_inh"]/d2["I_tau"] - 1.0), 5e-2)

        # Mapping error should increase with modulation amplitude
        self.assertLess(d1["l1_inh"], d2["l1_inh"])


if __name__ == "__main__":
    unittest.main()
