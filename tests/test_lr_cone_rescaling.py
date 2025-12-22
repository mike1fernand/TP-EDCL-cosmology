import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
import numpy as np

from tierB import sim_lr_cone_rescaling as simlr


class TestLRConeRescaling(unittest.TestCase):
    def test_front_speed_rescaling(self):
        p = simlr.Params(N=5, J=1.0, h=1.0, tau_max=6.0, dtau=0.10, threshold=0.10, F_list=(1.0, 2.0))
        d = simlr.run(p)

        # Ratio should be ~1/F (here F=2)
        self.assertTrue(np.isfinite(d["ratio"]))
        self.assertLess(abs(d["ratio"] - 0.5), 0.05)

        # Front times should be nondecreasing with distance where defined
        tf = d["results"][0]["tau_front"]
        vals = [tf[r] for r in range(1, p.N) if np.isfinite(tf[r])]
        self.assertGreaterEqual(len(vals), 2)
        self.assertTrue(all(vals[i] <= vals[i+1] for i in range(len(vals)-1)))


if __name__ == "__main__":
    unittest.main()
