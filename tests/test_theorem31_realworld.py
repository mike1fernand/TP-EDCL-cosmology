import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
from tierB import sim_theorem31_realworld_validation as sim

class TestTheorem31RealWorld(unittest.TestCase):
    def test_acceptance_small(self):
        # Reduced runtime configuration: one scale, skip dtau refinement, fewer x0 samples.
        p = sim.Params(
            L0=1600,
            scales=(1,),
            n_x0=3,
            tau_max_base=180.0,
            dtau=0.06,
            mismatch_qs=(1.0,),
            dtau_refinement_factors=(1.0,),
            boundary_pad_sigma=4.0,
        )
        levels, _rep = sim.run_validation(p, make_plots=False)

        fin = levels[-1]
        q = p.mismatch_qs[0]
        mm = fin.mismatch_results[q]

        self.assertLess(fin.eps_space_matched, 5e-2)
        self.assertGreater(mm["eps_space"] / max(fin.eps_space_matched, 1e-12), 5.0)
        self.assertLess(fin.lemma_residual_matched, 1e-10)
        self.assertGreater(fin.lemma_residual_mismatched[q], 1e-6)
        self.assertLess(fin.max_norm_drift, 1e-8)

if __name__ == "__main__":
    unittest.main()
