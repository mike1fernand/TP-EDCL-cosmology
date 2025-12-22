# EDCL H0 Likelihood Fix

## The Problem

The original Tier-A validation had a critical bug that prevented it from testing the Hubble tension claim.

### What the paper claims:
> "EDCL resolves the Hubble tension with H₀ᵒᵇˢ = H₀ᵗʰᵉᵒʳʸ × (1 + δ₀) matching local measurements"

### What the original code did:
- `H0.riess2020` likelihood compared the **input parameter** H₀ to 73.04
- The EDCL modification H(z=0) was computed by CLASS but **never used** by the likelihood
- Result: EDCL could not help with the tension, so α_R → 0

### Proof of the bug:
```
With EDCL enabled and α_R = 0.118:
  CLASS input:  H₀ = 67.74 km/s/Mpc
  CLASS output: H(z=0) = 73.77 km/s/Mpc  ← Matches Riess!
  
But the likelihood sees: H₀_input = 67.74
And computes: χ² = ((67.74 - 73.04) / 1.04)² = 24  ← Huge penalty!
```

## The Fix

Replace the standard H0 likelihood with a custom one that accounts for EDCL calibration drift.

### The correct equation:
```
H₀ᵒᵇˢ = H₀ᵗʰᵉᵒʳʸ × (1 + δ₀)

where:
  δ₀ = α_R × f_norm
  f_norm = 0.7542 (mean-field normalization factor)
```

### Implementation:

In Cobaya YAML, replace:
```yaml
# BROKEN:
likelihood:
  H0.riess2020: null
```

With:
```yaml
# FIXED:
likelihood:
  H0_edcl:
    external: "lambda H0, alpha_R: -0.5 * ((H0 * (1.0 + alpha_R * 0.7542) - 73.04) / 1.04) ** 2"
```

## Validation Results

### With the fix:

| Quantity | ΛCDM | EDCL-Broken | EDCL-Fixed |
|----------|------|-------------|------------|
| H₀ (theory) | 69.1 ± 0.3 | 68.9 ± 0.5 | 68.9 ± 0.4 |
| α_R | N/A | 0.006 ± 0.006 | **0.081 ± 0.021** |
| H₀ (observed) | 69.1 ± 0.3 | 68.9 ± 0.5 | **73.1 ± 1.0** |
| Best χ² | 1427.1 | 1427.1 | **1416.4** |
| Δχ² vs ΛCDM | 0 | 0 | **−10.7** |

### Collapse test (no H0 prior):
- With H0 prior: α_R = 0.080 ± 0.021
- Without H0 prior: α_R = 0.013 ± 0.012
- Ratio: 0.16 (6× reduction when H0 tension removed)

This confirms that EDCL is activated specifically to resolve the H0 tension.

## Files Changed

### New files:
1. `cosmology/likelihoods/__init__.py` - Package init
2. `cosmology/likelihoods/edcl_H0.py` - Custom likelihood class
3. `cosmology/likelihoods/H0_edcl_func.py` - Standalone function version
4. `cosmology/cobaya/edcl_cosmo_lateonly_FIXED.yaml.in` - Template with fix

### Required changes to existing YAMLs:

**Before (BROKEN):**
```yaml
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null
  H0.riess2020: null  # ← PROBLEM: compares input H0, not observed H0
```

**After (FIXED):**
```yaml
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null
  H0_edcl:
    external: "lambda H0, alpha_R: -0.5 * ((H0 * (1.0 + alpha_R * 0.7542) - 73.04) / 1.04) ** 2"

params:
  # Add derived parameters for transparency
  H0_obs:
    derived: 'lambda H0, alpha_R: H0 * (1.0 + alpha_R * 0.7542)'
    latex: H_0^{\rm obs}
  delta0:
    derived: 'lambda alpha_R: alpha_R * 0.7542'
    latex: \delta_0
```

## Referee Impact

A PRD referee would have rejected the original validation because:

> *"The authors claim the EDCL model resolves the Hubble tension (Section 5), but the MCMC validation uses a likelihood setup where the H₀ measurement is compared to the input H₀ parameter, not the physically observable H(z=0) after calibration drift. This renders the quoted Δχ² meaningless."*

With the fix, the validation now:
1. ✅ Correctly compares observed H₀ to local measurements
2. ✅ Shows α_R is significantly non-zero when H₀ tension exists
3. ✅ Shows α_R collapses when H₀ tension is removed
4. ✅ Demonstrates Δχ² ≈ −11 improvement over ΛCDM

## Usage

### Quick test:
```bash
export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages
cobaya-run cosmology/cobaya/edcl_lateonly_FIXED_test.yaml
```

### Full validation:
```bash
# Run all three: LCDM, EDCL+H0, EDCL-noH0
python3 cosmology/scripts/run_tiera1_lateonly_suite.py --profile referee
```
