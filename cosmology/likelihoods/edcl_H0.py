"""
H0 likelihood for EDCL/TP model.

This likelihood computes the OBSERVED H0 after EDCL calibration drift:
    H0_obs = H0_input * (1 + delta0)
where:
    delta0 = alpha_R * kappa_tick * 12 * f_norm

Then compares to the Riess et al. (2022) measurement:
    H0 = 73.04 ± 1.04 km/s/Mpc

This is CRITICAL: the standard H0.riess2020 likelihood compares the input H0
parameter directly, which means EDCL cannot help with the tension. This
custom likelihood fixes that by comparing the physically observable quantity.

Usage in Cobaya YAML:
    likelihood:
        edcl_H0_riess2022:
            python_path: /path/to/this/file
            H0_mean: 73.04
            H0_std: 1.04
            kappa_tick: 0.08333333333333333
            f_norm: 0.7542

Author: Auto-generated for TP/EDCL validation
"""

from cobaya.likelihood import Likelihood
import numpy as np


class edcl_H0_riess2022(Likelihood):
    """
    Custom H0 likelihood for EDCL model.
    
    Compares the EDCL-modified H(z=0) to local distance ladder measurements.
    """
    
    # Riess et al. (2022) measurement (default values)
    H0_mean: float = 73.04
    H0_std: float = 1.04
    
    # EDCL model parameters (fixed theory choices)
    kappa_tick: float = 0.08333333333333333  # 1/12
    f_norm: float = 0.7542  # Mean-field normalization factor
    
    def initialize(self):
        """Called once at the start."""
        self.log.info(f"EDCL H0 likelihood initialized:")
        self.log.info(f"  H0_mean = {self.H0_mean} km/s/Mpc")
        self.log.info(f"  H0_std  = {self.H0_std} km/s/Mpc")
        self.log.info(f"  kappa_tick = {self.kappa_tick}")
        self.log.info(f"  f_norm = {self.f_norm}")
    
    def get_requirements(self):
        """Tell Cobaya which parameters we need."""
        return {'H0': None, 'alpha_R': None}
    
    def logp(self, **params_values):
        """
        Compute log-likelihood.
        
        The key equation:
            delta0 = alpha_R * 12 * kappa_tick * f_norm
            H0_obs = H0_input * (1 + delta0)
            chi2 = ((H0_obs - H0_mean) / H0_std)^2
            logp = -0.5 * chi2
        """
        H0_input = params_values.get('H0')
        alpha_R = params_values.get('alpha_R', 0.0)
        
        # Compute the observed H0 after EDCL modification
        delta0 = alpha_R * 12.0 * self.kappa_tick * self.f_norm
        H0_obs = H0_input * (1.0 + delta0)
        
        # Gaussian likelihood
        chi2 = ((H0_obs - self.H0_mean) / self.H0_std) ** 2
        
        return -0.5 * chi2


class edcl_H0_riess2020(edcl_H0_riess2022):
    """Alias using Riess 2020 values (same central value, similar error)."""
    H0_mean: float = 73.2
    H0_std: float = 1.3


class edcl_H0_freedman2024(Likelihood):
    """
    Alternative H0 likelihood using Freedman et al. (2024) TRGB measurement.
    H0 = 69.85 ± 1.75 km/s/Mpc
    
    Useful for testing whether EDCL can reconcile different local measurements.
    """
    H0_mean: float = 69.85
    H0_std: float = 1.75
    kappa_tick: float = 0.08333333333333333
    f_norm: float = 0.7542
    
    def initialize(self):
        self.log.info(f"EDCL H0 (Freedman 2024) likelihood: {self.H0_mean} ± {self.H0_std}")
    
    def get_requirements(self):
        return {'H0': None, 'alpha_R': None}
    
    def logp(self, **params_values):
        H0_input = params_values.get('H0')
        alpha_R = params_values.get('alpha_R', 0.0)
        delta0 = alpha_R * 12.0 * self.kappa_tick * self.f_norm
        H0_obs = H0_input * (1.0 + delta0)
        chi2 = ((H0_obs - self.H0_mean) / self.H0_std) ** 2
        return -0.5 * chi2
