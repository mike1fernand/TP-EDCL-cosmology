# EDCL Tier-A Cosmology Validation Scripts

This directory contains scripts to run and analyze the Tier-A cosmological validation of the EDCL framework.

## Quick Start

### Option 1: Analyze Pre-Existing Chains (Fast)

If you have chain files from a previous run:

```bash
python COLAB_TIER_A_VALIDATION.py --validate-only --chains-dir ./chains
```

### Option 2: Run Full MCMC (2-4 hours)

```bash
# Set required paths
export CLASS_PATH=/path/to/class_public
export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages

# Run validation
./RUN_TIER_A_VALIDATION.sh
```

## Files

| File | Description |
|------|-------------|
| `RUN_TIER_A_VALIDATION.sh` | Main runner script (generates YAMLs and runs MCMC) |
| `COLAB_TIER_A_VALIDATION.py` | Python script for Colab or local analysis |
| `cosmology/scripts/analyze_chains.py` | Chain analysis and validation tests |
| `cosmology/cobaya/render_yaml.py` | Convert YAML templates to runnable configs |
| `cosmology/cobaya/templates/*.template` | YAML templates with path placeholders |

## Prerequisites

### Software

```bash
pip install cobaya getdist numpy matplotlib
```

### CLASS with EDCL Patch

1. Clone CLASS: `git clone https://github.com/lesgourg/class_public.git`
2. Apply EDCL patch: `git apply class_edcl.patch`
3. Compile: `make -j4`

### Cobaya Likelihood Data (~210 MB)

```bash
cobaya-install bao.desi_dr2 sn.pantheonplus -p ./cobaya_packages
```

## Usage Examples

### Generate Runnable YAMLs from Templates

```bash
python cosmology/cobaya/render_yaml.py \
    --class-path /path/to/class_public \
    --output-dir ./chains
```

### Run Individual MCMC Chain

```bash
export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages
cobaya-run ./chains/edcl.yaml
```

### Analyze Chains and Generate Report

```bash
python cosmology/scripts/analyze_chains.py \
    --chains-dir ./chains \
    --output validation_results.json \
    --plot
```

## Validation Tests

The scripts run four validation tests:

| Test | Criterion | Expected |
|------|-----------|----------|
| **Activation** | α_R > 0 when H0 tension exists | α_R ≈ 0.08 |
| **Collapse** | α_R → 0 when H0 tension removed | 80%+ reduction |
| **H0 Match** | H0_obs matches Riess measurement | 73.0 ± 1.0 |
| **χ² Improvement** | EDCL fits better than ΛCDM | Δχ² < 0 |

## Output Files

After running, you'll find in `./chains/`:

- `lcdm.1.txt` - ΛCDM chain samples
- `edcl.1.txt` - EDCL chain samples (with H0 prior)
- `edcl_no_h0.1.txt` - EDCL chain samples (without H0, collapse test)
- `validation_results.json` - Analysis results
- `h0_comparison.png` - H0 posterior comparison plot

## Troubleshooting

### "classy module not found"

Make sure CLASS is compiled:
```bash
cd /path/to/class_public
make clean
make -j4
```

### "Cobaya packages not found"

Install the required likelihoods:
```bash
cobaya-install bao.desi_dr2 sn.pantheonplus -p ./cobaya_packages
```

### MCMC runs forever

Reduce `max_samples` in the YAML files or use `Rminus1_stop: 0.05` for faster convergence.

## Citation

If you use these scripts, please cite the EDCL paper:
> Fernandes, M. (2025). Two-Perspective Quantum Dynamics with Entropy-Driven Curvature Law.
