#!/bin/bash

# Ensure we run from the repository root (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#
# RUN_TIER_A_VALIDATION.sh
# 
# Run EDCL Tier-A cosmology validation MCMC chains
#
# Usage:
#   ./RUN_TIER_A_VALIDATION.sh                    # Uses environment variables
#   ./RUN_TIER_A_VALIDATION.sh /path/to/class     # Specify CLASS path
#
# Environment Variables:
#   CLASS_PATH          - Path to CLASS with EDCL patch (required)
#   COBAYA_PACKAGES_PATH - Path to Cobaya data packages (required)
#   OUTPUT_DIR          - Output directory for chains (default: ./chains)
#
# Prerequisites:
#   - Python 3.8+
#   - Cobaya: pip install cobaya
#   - GetDist: pip install getdist (optional, for analysis)
#   - CLASS with EDCL patch compiled
#   - Cobaya packages installed: cobaya-install bao.desi_dr2 sn.pantheonplus
#

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Parse command line argument for CLASS_PATH
if [ -n "$1" ]; then
    CLASS_PATH="$1"
fi

# Check required paths
if [ -z "$CLASS_PATH" ]; then
    echo "ERROR: CLASS_PATH not set"
    echo "Usage: $0 /path/to/class_public"
    echo "   or: export CLASS_PATH=/path/to/class_public"
    exit 1
fi

if [ -z "$COBAYA_PACKAGES_PATH" ]; then
    echo "ERROR: COBAYA_PACKAGES_PATH not set"
    echo "Usage: export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-./chains}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# VALIDATION
# =============================================================================

echo "=================================================="
echo "EDCL TIER-A COSMOLOGY VALIDATION"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  CLASS_PATH:           $CLASS_PATH"
echo "  COBAYA_PACKAGES_PATH: $COBAYA_PACKAGES_PATH"
echo "  OUTPUT_DIR:           $OUTPUT_DIR"
echo ""

# Check CLASS exists
if [ ! -d "$CLASS_PATH" ]; then
    echo "ERROR: CLASS directory not found: $CLASS_PATH"
    exit 1
fi

# Check for classy module
if ! ls "$CLASS_PATH/python/"classy*.so >/dev/null 2>&1; then
    echo "WARNING: classy module not found in $CLASS_PATH/python/"
    echo "Make sure CLASS is compiled (run 'make' in CLASS directory)"
fi

# Check Cobaya packages
if [ ! -d "$COBAYA_PACKAGES_PATH/data" ]; then
    echo "ERROR: Cobaya packages not found at $COBAYA_PACKAGES_PATH"
    echo "Install with: cobaya-install bao.desi_dr2 sn.pantheonplus"
    exit 1
fi

# Check cobaya-run exists
if ! command -v cobaya-run &> /dev/null; then
    echo "ERROR: cobaya-run not found. Install with: pip install cobaya"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# GENERATE YAML FILES
# =============================================================================

echo "Generating YAML configuration files..."

# Render templates if they exist, otherwise use generate script
if [ -f "$SCRIPT_DIR/cosmology/cobaya/render_yaml.py" ]; then
    python3 "$SCRIPT_DIR/cosmology/cobaya/render_yaml.py" \
        --class-path "$CLASS_PATH" \
        --output-dir "$OUTPUT_DIR"
else
    # Generate YAML files directly
    cat > "$OUTPUT_DIR/lcdm.yaml" << YAML
# LCDM Baseline
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null
  H0.riess2020: null

theory:
  classy:
    path: $CLASS_PATH
    extra_args:
      edcl_on: 'no'

params:
  omega_b:
    prior: {min: 0.018, max: 0.026}
    ref: {dist: norm, loc: 0.02237, scale: 0.00015}
    proposal: 0.0001
  omega_cdm:
    prior: {min: 0.08, max: 0.16}
    ref: {dist: norm, loc: 0.1200, scale: 0.0012}
    proposal: 0.001
  H0:
    prior: {min: 60.0, max: 80.0}
    ref: 69.0
    proposal: 0.5

output: $OUTPUT_DIR/lcdm

sampler:
  mcmc:
    max_samples: 30000
    Rminus1_stop: 0.02
    learn_proposal: true
    seed: 42
YAML

    cat > "$OUTPUT_DIR/edcl.yaml" << YAML
# EDCL with H0 prior
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null
  H0_edcl:
    external: "lambda H0, alpha_R: -0.5 * ((H0 * (1.0 + alpha_R * 0.7542) - 73.04) / 1.04) ** 2"

theory:
  classy:
    path: $CLASS_PATH
    extra_args:
      edcl_on: 'yes'
      kappa_tick: 0.08333333333333333
      c4: 0.06
      log10_l0: -20.908
      edcl_kernel: exp
      edcl_zeta: 0.5
      edcl_ai: 0.0001

params:
  omega_b:
    prior: {min: 0.018, max: 0.026}
    ref: {dist: norm, loc: 0.02237, scale: 0.00015}
    proposal: 0.0001
  omega_cdm:
    prior: {min: 0.08, max: 0.16}
    ref: {dist: norm, loc: 0.1200, scale: 0.0012}
    proposal: 0.001
  H0:
    prior: {min: 60.0, max: 80.0}
    ref: 67.5
    proposal: 0.5
  alpha_R:
    prior: {min: 0.0, max: 0.25}
    ref: 0.08
    proposal: 0.015
  H0_obs:
    derived: 'lambda H0, alpha_R: H0 * (1.0 + alpha_R * 0.7542)'
  delta0:
    derived: 'lambda alpha_R: alpha_R * 0.7542'

output: $OUTPUT_DIR/edcl

sampler:
  mcmc:
    max_samples: 30000
    Rminus1_stop: 0.02
    learn_proposal: true
    seed: 42
YAML

    cat > "$OUTPUT_DIR/edcl_no_h0.yaml" << YAML
# EDCL without H0 prior (collapse test)
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null

theory:
  classy:
    path: $CLASS_PATH
    extra_args:
      edcl_on: 'yes'
      kappa_tick: 0.08333333333333333
      c4: 0.06
      log10_l0: -20.908
      edcl_kernel: exp
      edcl_zeta: 0.5
      edcl_ai: 0.0001

params:
  omega_b:
    prior: {min: 0.018, max: 0.026}
    ref: {dist: norm, loc: 0.02237, scale: 0.00015}
    proposal: 0.0001
  omega_cdm:
    prior: {min: 0.08, max: 0.16}
    ref: {dist: norm, loc: 0.1200, scale: 0.0012}
    proposal: 0.001
  H0:
    prior: {min: 60.0, max: 80.0}
    ref: 67.5
    proposal: 0.5
  alpha_R:
    prior: {min: 0.0, max: 0.25}
    ref: 0.05
    proposal: 0.015
  H0_obs:
    derived: 'lambda H0, alpha_R: H0 * (1.0 + alpha_R * 0.7542)'

output: $OUTPUT_DIR/edcl_no_h0

sampler:
  mcmc:
    max_samples: 30000
    Rminus1_stop: 0.02
    learn_proposal: true
    seed: 42
YAML

    echo "Generated YAML files in $OUTPUT_DIR/"
fi

# =============================================================================
# RUN MCMC CHAINS
# =============================================================================

echo ""
echo "=================================================="
echo "RUNNING MCMC CHAINS"
echo "=================================================="
echo ""

# Run LCDM
echo "[1/3] Running LCDM baseline..."
cobaya-run "$OUTPUT_DIR/lcdm.yaml" -f

# Run EDCL with H0
echo ""
echo "[2/3] Running EDCL with H0 prior..."
cobaya-run "$OUTPUT_DIR/edcl.yaml" -f

# Run EDCL without H0
echo ""
echo "[3/3] Running EDCL without H0 (collapse test)..."
cobaya-run "$OUTPUT_DIR/edcl_no_h0.yaml" -f

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

echo ""
echo "=================================================="
echo "ANALYZING RESULTS"
echo "=================================================="
echo ""

if [ -f "$SCRIPT_DIR/cosmology/scripts/analyze_chains.py" ]; then
    python3 "$SCRIPT_DIR/cosmology/scripts/analyze_chains.py" \
        --chains-dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/validation_results.json" \
        --plot
elif [ -f "$SCRIPT_DIR/COLAB_TIER_A_VALIDATION.py" ]; then
    python3 "$SCRIPT_DIR/COLAB_TIER_A_VALIDATION.py" \
        --validate-only \
        --chains-dir "$OUTPUT_DIR" \
        --output "$OUTPUT_DIR/validation_results.json" \
        --plot
else
    echo "Analysis script not found. Run manually:"
    echo "  python analyze_chains.py --chains-dir $OUTPUT_DIR"
fi

echo ""
echo "=================================================="
echo "COMPLETE"
echo "=================================================="
echo ""
echo "Results in: $OUTPUT_DIR/"
echo "  - Chain files: *.1.txt"
echo "  - Validation: validation_results.json"
echo "  - Plot: h0_comparison.png"
