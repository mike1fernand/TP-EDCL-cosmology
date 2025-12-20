import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
import numpy as np

from tierB import sim_interface_scattering_invariance as simint


class TestInterfaceScattering(unittest.TestCase):
    def test_flux_and_invariance(self):
        p = simint.Params(N=768, dtau=0.06, tau_max=330.0, m=384, J1=1.0, J2=1.5, x0=140.0, sigma_x=24.0, k1=1.0,
                         eps_F=0.45, tau_mid=170.0, tau_width=35.0)
        d = simint.run(p)

        # Probability conservation and asymptotic separation
        total = d["R_fin"] + d["T_fin"] + d["P_mid_fin"]
        self.assertLess(abs(total - 1.0), 5e-3)
        self.assertLess(d["P_mid_fin"], 5e-3)

        # Flux equals transmission probability (within tolerance)
        self.assertLess(abs(d["T_fin"] - d["T_flux_tau"]), 5e-3)

        # Time reparam invariance of integrated flux
        self.assertLess(abs(d["T_flux_tau"] - d["T_flux_t"]), 5e-4)

        # Norm drift small
        norm_drift = float(np.max(np.abs(d["norm"] - 1.0)))
        self.assertLess(norm_drift, 1e-8)

        # Analytic comparison (loose tolerance because wavepacket has finite bandwidth)
        self.assertLess(abs(d["R_fin"] - d["R_an"]), 5e-2)
        self.assertLess(abs(d["T_fin"] - d["T_an"]), 5e-2)


if __name__ == "__main__":
    unittest.main()
