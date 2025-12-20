import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
import numpy as np

from tierB import sim_theorem31_wavepacket_validation as sim31


class TestTheorem31Wavepacket(unittest.TestCase):
    def test_matched_vs_mismatched(self):
        # Smallish run for tests; uses k0=π/2 to suppress k-drift and isolate calibration.
        p = sim31.RunParams(
            N=384, dtau=0.12, tau_max=120.0,
            x0=70.0, sigma_x=18.0, k0=float(np.pi/2),
            L=140.0, amp=0.5, xc=190.0,
            k_sample_stride=80,
        )
        base = sim31.simulate_structure_time(p)

        tau = base["tau"]
        x_mean = base["x_mean"]
        t_m = sim31.derive_observer_time(tau, p.dtau, base["F_matched"])
        t_u = sim31.derive_observer_time(tau, p.dtau, base["F_mismatch"])
        v_m = sim31.compute_velocity_vs_time(x_mean, t_m)
        v_u = sim31.compute_velocity_vs_time(x_mean, t_u)

        tmin = 0.20 * t_m.max()
        tmax = 0.90 * t_m.max()
        am = sim31.analyze_window(t_m, x_mean, v_m, tmin, tmax)
        au = sim31.analyze_window(t_u, x_mean, v_u, tmin, tmax)

        eps_v_m = am["eps_v"]
        eps_v_u = au["eps_v"]

        # Acceptance: matched should be close to constant speed; mismatch should not.
        self.assertLess(eps_v_m, 1e-3)
        self.assertGreater(eps_v_u, 1e-2)

        # Integrity: norm preserved
        norm_drift = float(np.max(np.abs(base["norm"] - 1.0)))
        self.assertLess(norm_drift, 1e-8)

        # Speed-transform identity: v_obs ≈ (dx/dτ)/⟨F⟩ for matched calibration
        v_coord = np.gradient(x_mean) / np.maximum(np.gradient(tau), 1e-12)
        v_pred = v_coord / np.maximum(base["F_matched"], 1e-12)
        rel_err = float(np.nanmean(np.abs(v_m - v_pred) / np.maximum(np.abs(v_m), 1e-12)))
        self.assertLess(rel_err, 2e-2)

        # k-drift diagnostic: k_mean stays near target k0 within a modest tolerance
        k_s = base["k_samples"]["k_mean"]
        self.assertLess(float(np.max(np.abs(k_s - p.k0))), 2e-2)


if __name__ == "__main__":
    unittest.main()
