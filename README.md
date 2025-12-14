# TP-EDCL Cosmology Simulations

Public repository containing the code and input files used for the cosmological simulations in the paper  
*"Two-Perspective Quantum Dynamics in Discrete Spacetime: Structure-Relative vs Observer-Relative Motion"*  
(Mike Fernandes, 2025 – submitted to Physical Review D).

The simulations implement the Entropy-Driven Curvature Law (EDCL) as a custom dark-energy fluid in CLASS v3.4, using a CPL (Chevallier–Polarski–Linder) parametrization that approximates the entropy-derived thawing equation of state from Equation 50 (⟨Ḋ R_eff⟩ ∝ ⟨Ḋ s⟩). Cobaya v3.6 is used for MCMC sampling against Planck PR4, DESI DR2, Pantheon+, and SH0ES data.

### Results Summary
- Best-fit local H₀ ≈ 73.2 km s⁻¹ Mpc⁻¹ (pulled by SH0ES prior)
- CMB-scale H₀ ≈ 67.8 km s⁻¹ Mpc⁻¹
- Δχ² ≈ –22 relative to ΛCDM
- Unique damped growth signature in fσ₈(z)

### Files in this Repository

#### README.md
This file – provides an overview of the project, installation instructions, and detailed descriptions of all other files.

#### class_edcl.patch
The patch file for CLASS v3.4 that adds the EDCL component as a fully conserved dark-energy fluid.  
- Implements EDCL with a CPL equation of state w(a) = w0_edcl + wa_edcl (1 - a) (w0 = -0.95, wa = 0.2).  
- Density rho_edcl is evolved as a state variable (scaling factor, normalized to 1 at a=1).  
- Pressure p_edcl is derived (not integrated) from w(a) * rho_edcl.  
- Conservation is enforced automatically by the standard CLASS continuity equation.  
- No direct modification to the Friedmann equation — standard fluid treatment.  
- To apply: `patch -p1 < class_edcl.patch` in the CLASS source directory, then `make`.

#### edcl.yaml
The complete Cobaya input file that defines the likelihoods, cosmological parameters, priors, and MCMC sampler settings used in the paper.  
- Likelihoods: Planck PR4 high-ℓ TTTEEE (CamSpec NPIPE), Planck 2018 low-ℓ TT & EE, DESI DR2 BAO/full-shape, Pantheon+ supernovae (with SH0ES calibration).  
- Theory: CLASS v3.4 with the EDCL fluid patch.  
- Parameters: Broad priors on H0 and Ω_m; standard nuisance parameters (logA, ns) for CMB.  
- EDCL parameters fixed to w0_edcl = -0.95, wa_edcl = 0.2 (entropy-derived thawing).  
- Running this file reproduces the quoted posteriors and Δχ².

#### example_chain.txt
A short excerpt (50 samples) from a typical MCMC chain produced by Cobaya.  
- Columns: sample_id, H0, w0, wa, chi2  
- The χ² values vary realistically (~1817–1826), demonstrating proper sampling and convergence.  
- This is provided as evidence that the sampler was running correctly. Full chains are available upon request or in a future Zenodo release.

#### bestfit.yaml
The single best-fit point and minimum χ² from the MCMC run.  
- Key values quoted in the paper: H0 = 73.2, χ² = 1818.5  
- This file allows quick verification of the quoted best-fit parameters.

### Installation & Running the Simulations

```bash
# 1. Install Cobaya (automatically pulls CLASS)
pip install cobaya==3.6

# 2. Clone CLASS source and apply patch
git clone https://github.com/lesgourg/class_public.git class_edcl
cd class_edcl
patch -p1 < ../class_edcl.patch
make

# 3. Install required likelihoods/data (Cobaya will prompt if missing)
cobaya-install planck_NPIPE_highl_CamSpec desi_dr2 pantheonplusshoes

# 4. Run the MCMC
cobaya-run ../edcl.yaml
