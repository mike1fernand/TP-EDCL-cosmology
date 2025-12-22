"""
External H0 likelihood function for EDCL model.

This is a standalone function that can be used with Cobaya's external likelihood.
"""

import numpy as np

# Riess et al. (2022) measurement
H0_MEAN = 73.04
H0_STD = 1.04

# EDCL model constants
F_NORM = 0.7542  # Mean-field normalization


def H0_edcl_logp(H0, alpha_R):
    """
    Log-likelihood for H0 measurement in EDCL model.
    
    Computes H0_obs = H0 * (1 + delta0) and compares to Riess measurement.
    
    Parameters
    ----------
    H0 : float
        Input H0 (Planck-frame, before EDCL modification)
    alpha_R : float
        EDCL amplitude parameter
    
    Returns
    -------
    float
        Log-likelihood value
    """
    delta0 = alpha_R * F_NORM
    H0_obs = H0 * (1.0 + delta0)
    
    chi2 = ((H0_obs - H0_MEAN) / H0_STD) ** 2
    return float(-0.5 * chi2)
