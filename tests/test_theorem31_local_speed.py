import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest

from tierB import sim_theorem31_local_speed_validation as sim


class TestTheorem31LocalSpeed(unittest.TestCase):
    """Reduced-size CI-style check for the local-speed theorem validation."""

    def test_theorem31_local_speed_smoke(self):
        # Reduced settings for a quick check.
        p = sim.Params(
            L0=512,
            scales=(1,),
            n_x0=5,
            sigma0=24.0,
            k0_0=0.25,
            dtau=0.06,
            tau_max0=120.0,
            alpha_J=0.5,
            eps_n=1.0,
            mismatch_q=1.0,
            # Keep away from boundaries.
            x0_min_frac=0.30,
            x0_max_frac=0.70,
        )

        levels, rep = sim.run_validation(p, make_plots=False)
        self.assertEqual(len(levels), 1)
        lv = levels[0]

        # Matched: close to position-independent.
        # Keep slightly loose for robustness under reduced settings.
        self.assertLess(lv.eps_space_matched, 3.0e-2)

        # Mismatch: appreciable spatial variability.
        self.assertGreater(lv.eps_space_mismatch, 5.0e-2)

        # Mismatch variability should be comparable to the predictor A within a factor.
        self.assertGreater(lv.eps_space_mismatch, 0.25 * lv.A_pred_mismatch)
        self.assertLess(lv.eps_space_mismatch, 4.0 * lv.A_pred_mismatch)

        # Locality metric should not be catastrophically violated.
        self.assertLess(lv.worst_locality_metric_matched, 0.5)
        self.assertLess(lv.worst_locality_metric_mismatch, 0.5)

        # Representative run exists and includes expected keys.
        self.assertTrue(rep)
        self.assertIn("matched", rep)
        self.assertIn("mismatch", rep)


if __name__ == "__main__":
    unittest.main()
