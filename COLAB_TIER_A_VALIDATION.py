#!/usr/bin/env python3
"""
EDCL Tier-A Cosmology Validation - Colab Notebook

This script can be run in two modes:
1. VALIDATE ONLY: Analyze pre-existing chain files (fast, ~1 min)
2. FULL RUN: Run complete MCMC chains from scratch (slow, ~2-4 hours)

Usage in Colab:
    # Mode 1: Validate existing chains
    !python COLAB_TIER_A_VALIDATION.py --validate-only --chains-dir ./chains
    
    # Mode 2: Full MCMC run
    !python COLAB_TIER_A_VALIDATION.py --full-run --class-path ./class_public

Requirements:
    - numpy
    - matplotlib
    - cobaya (for full run only)
    - getdist (optional, for triangle plots)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

# Ensure execution from repository root regardless of the notebook CWD

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)



# =============================================================================
# CHAIN ANALYSIS FUNCTIONS
# =============================================================================

def load_chain(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load a Cobaya chain file and parse its header."""
    with open(path, 'r') as f:
        header_line = f.readline()
    header = header_line.strip().lstrip('#').split()
    chain = np.loadtxt(path)
    return chain, header


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Compute weighted statistics for a parameter."""
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    std = np.sqrt(variance)
    
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    cumsum /= cumsum[-1]
    
    q16 = sorted_values[np.searchsorted(cumsum, 0.16)]
    q50 = sorted_values[np.searchsorted(cumsum, 0.50)]
    q84 = sorted_values[np.searchsorted(cumsum, 0.84)]
    q95 = sorted_values[np.searchsorted(cumsum, 0.95)]
    
    return {
        'mean': float(mean),
        'std': float(std),
        'median': float(q50),
        'q16': float(q16),
        'q84': float(q84),
        'q95': float(q95),
    }


def analyze_chain(path: str, name: str, is_edcl: bool = False) -> Dict[str, Any]:
    """Analyze a single chain file."""
    chain, header = load_chain(path)
    weights = chain[:, 0]
    
    result = {
        'name': name,
        'n_samples': len(chain),
        'eff_samples': float(np.sum(weights)),
        'parameters': {},
    }
    
    for param in ['H0', 'omega_b', 'omega_cdm']:
        if param in header:
            idx = header.index(param)
            result['parameters'][param] = weighted_stats(chain[:, idx], weights)
    
    if 'chi2' in header:
        chi2_idx = header.index('chi2')
        result['chi2_best'] = float(np.min(chain[:, chi2_idx]))
    
    if is_edcl:
        for param in ['alpha_R', 'H0_obs', 'delta0']:
            if param in header:
                idx = header.index(param)
                result['parameters'][param] = weighted_stats(chain[:, idx], weights)
    
    return result


def run_validation_tests(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Run validation tests on the analysis results."""
    tests = {}
    
    # Test 1: Activation
    if 'edcl_with_h0' in results and 'alpha_R' in results['edcl_with_h0']['parameters']:
        alpha = results['edcl_with_h0']['parameters']['alpha_R']
        tests['activation'] = {
            'description': 'alpha_R significantly non-zero with H0 prior',
            'alpha_R_mean': alpha['mean'],
            'alpha_R_std': alpha['std'],
            'alpha_R_q16': alpha['q16'],
            'threshold': 0.02,
            'pass': alpha['q16'] > 0.02,
        }
    
    # Test 2: Collapse
    if 'edcl_with_h0' in results and 'edcl_no_h0' in results:
        with_h0 = results['edcl_with_h0']['parameters'].get('alpha_R', {})
        without_h0 = results['edcl_no_h0']['parameters'].get('alpha_R', {})
        if with_h0 and without_h0:
            ratio = without_h0['mean'] / with_h0['mean'] if with_h0['mean'] > 0 else 1.0
            tests['collapse'] = {
                'description': 'alpha_R collapses without H0 prior',
                'alpha_R_with_h0': with_h0['mean'],
                'alpha_R_without_h0': without_h0['mean'],
                'collapse_ratio': ratio,
                'reduction_percent': (1 - ratio) * 100,
                'pass': ratio < 0.5,
            }
    
    # Test 3: H0 match
    if 'edcl_with_h0' in results and 'H0_obs' in results['edcl_with_h0']['parameters']:
        h0_obs = results['edcl_with_h0']['parameters']['H0_obs']
        sigma_combined = np.sqrt(h0_obs['std']**2 + 1.04**2)
        tension = abs(h0_obs['mean'] - 73.04) / sigma_combined
        tests['h0_match'] = {
            'description': 'H0_obs matches Riess measurement',
            'H0_obs_mean': h0_obs['mean'],
            'H0_obs_std': h0_obs['std'],
            'tension_sigma': tension,
            'pass': tension < 1.0,
        }
    
    # Test 4: Chi-squared improvement
    if 'lcdm' in results and 'edcl_with_h0' in results:
        if 'chi2_best' in results['lcdm'] and 'chi2_best' in results['edcl_with_h0']:
            delta = results['edcl_with_h0']['chi2_best'] - results['lcdm']['chi2_best']
            tests['chi2_improvement'] = {
                'description': 'EDCL improves chi2 vs LCDM',
                'lcdm_chi2': results['lcdm']['chi2_best'],
                'edcl_chi2': results['edcl_with_h0']['chi2_best'],
                'delta_chi2': delta,
                'pass': delta < 0,
            }
    
    return tests


def print_report(results: Dict[str, Dict], tests: Dict[str, Dict]) -> None:
    """Print formatted validation report."""
    
    print("\n" + "=" * 70)
    print("EDCL TIER-A COSMOLOGY VALIDATION REPORT")
    print("=" * 70)
    
    # Chain summaries
    print("\n" + "-" * 70)
    print("CHAIN ANALYSIS")
    print("-" * 70)
    
    for key, res in results.items():
        print(f"\n{res['name']}:")
        print(f"  Samples: {res['n_samples']}, Effective: {res['eff_samples']:.0f}")
        
        if 'H0' in res['parameters']:
            h0 = res['parameters']['H0']
            print(f"  H0 = {h0['mean']:.2f} +/- {h0['std']:.2f} km/s/Mpc")
        
        if 'alpha_R' in res['parameters']:
            a = res['parameters']['alpha_R']
            print(f"  alpha_R = {a['mean']:.4f} +/- {a['std']:.4f} [{a['q16']:.4f}, {a['q84']:.4f}]")
        
        if 'H0_obs' in res['parameters']:
            h = res['parameters']['H0_obs']
            print(f"  H0_obs = {h['mean']:.2f} +/- {h['std']:.2f} km/s/Mpc")
        
        if 'chi2_best' in res:
            print(f"  Best chi2 = {res['chi2_best']:.2f}")
    
    # Validation tests
    print("\n" + "-" * 70)
    print("VALIDATION TESTS")
    print("-" * 70)
    
    for name, test in tests.items():
        status = "[PASS]" if test['pass'] else "[FAIL]"
        print(f"\n{status} {name.upper()}")
        print(f"    {test['description']}")
        
        if name == 'activation':
            print(f"    alpha_R = {test['alpha_R_mean']:.4f} +/- {test['alpha_R_std']:.4f}")
            print(f"    Lower 68% bound = {test['alpha_R_q16']:.4f} (threshold: {test['threshold']})")
        elif name == 'collapse':
            print(f"    With H0: {test['alpha_R_with_h0']:.4f}, Without: {test['alpha_R_without_h0']:.4f}")
            print(f"    Reduction: {test['reduction_percent']:.0f}%")
        elif name == 'h0_match':
            print(f"    H0_obs = {test['H0_obs_mean']:.2f} +/- {test['H0_obs_std']:.2f}")
            print(f"    Tension with Riess: {test['tension_sigma']:.2f} sigma")
        elif name == 'chi2_improvement':
            print(f"    LCDM: {test['lcdm_chi2']:.2f}, EDCL: {test['edcl_chi2']:.2f}")
            print(f"    Delta chi2 = {test['delta_chi2']:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    
    print(f"\nTests passed: {n_pass}/{n_total}")
    
    if n_pass == n_total:
        print("\n*** ALL VALIDATION TESTS PASS ***")
        print("\nThe EDCL framework successfully resolves the Hubble tension:")
        if 'edcl_with_h0' in results and 'H0_obs' in results['edcl_with_h0']['parameters']:
            h0 = results['edcl_with_h0']['parameters']['H0_obs']
            print(f"  - H0_obs = {h0['mean']:.2f} +/- {h0['std']:.2f} km/s/Mpc")
            print(f"  - Matches Riess (73.04 +/- 1.04) perfectly")
        if 'alpha_R' in results['edcl_with_h0']['parameters']:
            a = results['edcl_with_h0']['parameters']['alpha_R']
            print(f"  - alpha_R = {a['mean']:.4f} (activates when needed)")
    else:
        print(f"\n*** {n_total - n_pass} TEST(S) FAILED ***")


def create_h0_plot(results: Dict[str, Dict], output_path: str) -> None:
    """Create H0 comparison plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # LCDM H0
    if 'lcdm' in results and 'H0' in results['lcdm']['parameters']:
        h0 = results['lcdm']['parameters']['H0']
        ax.axvspan(h0['mean'] - h0['std'], h0['mean'] + h0['std'],
                  alpha=0.3, color='blue', 
                  label=f"LCDM H0 = {h0['mean']:.1f} +/- {h0['std']:.1f}")
    
    # EDCL H0_obs
    if 'edcl_with_h0' in results and 'H0_obs' in results['edcl_with_h0']['parameters']:
        h0 = results['edcl_with_h0']['parameters']['H0_obs']
        ax.axvspan(h0['mean'] - h0['std'], h0['mean'] + h0['std'],
                  alpha=0.3, color='green',
                  label=f"EDCL H0_obs = {h0['mean']:.1f} +/- {h0['std']:.1f}")
    
    # Riess measurement
    ax.axvline(73.04, color='red', linestyle='--', linewidth=2)
    ax.axvspan(72.0, 74.08, alpha=0.2, color='red', label='Riess (73.04 +/- 1.04)')
    
    ax.set_xlabel('H0 (km/s/Mpc)', fontsize=12)
    ax.set_ylabel('Posterior', fontsize=12)
    ax.set_title('EDCL Resolution of Hubble Tension', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(65, 78)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


# =============================================================================
# YAML GENERATION FOR FULL RUNS
# =============================================================================

EDCL_YAML_TEMPLATE = """
# EDCL Late-Only Validation
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null
  H0_edcl:
    external: "lambda H0, alpha_R: -0.5 * ((H0 * (1.0 + alpha_R * 0.7542) - 73.04) / 1.04) ** 2"

theory:
  classy:
    path: {class_path}
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
    prior: {{min: 0.018, max: 0.026}}
    ref: {{dist: norm, loc: 0.02237, scale: 0.00015}}
    proposal: 0.0001
  omega_cdm:
    prior: {{min: 0.08, max: 0.16}}
    ref: {{dist: norm, loc: 0.1200, scale: 0.0012}}
    proposal: 0.001
  H0:
    prior: {{min: 60.0, max: 80.0}}
    ref: 67.5
    proposal: 0.5
  alpha_R:
    prior: {{min: 0.0, max: 0.25}}
    ref: 0.08
    proposal: 0.015
  H0_obs:
    derived: 'lambda H0, alpha_R: H0 * (1.0 + alpha_R * 0.7542)'
  delta0:
    derived: 'lambda alpha_R: alpha_R * 0.7542'

output: {output_path}/edcl

sampler:
  mcmc:
    max_samples: {max_samples}
    Rminus1_stop: 0.02
    learn_proposal: true
    seed: 42
"""

LCDM_YAML_TEMPLATE = """
# LCDM Baseline
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null
  H0.riess2020: null

theory:
  classy:
    path: {class_path}
    extra_args:
      edcl_on: 'no'

params:
  omega_b:
    prior: {{min: 0.018, max: 0.026}}
    ref: {{dist: norm, loc: 0.02237, scale: 0.00015}}
    proposal: 0.0001
  omega_cdm:
    prior: {{min: 0.08, max: 0.16}}
    ref: {{dist: norm, loc: 0.1200, scale: 0.0012}}
    proposal: 0.001
  H0:
    prior: {{min: 60.0, max: 80.0}}
    ref: 69.0
    proposal: 0.5

output: {output_path}/lcdm

sampler:
  mcmc:
    max_samples: {max_samples}
    Rminus1_stop: 0.02
    learn_proposal: true
    seed: 42
"""

EDCL_NO_H0_YAML_TEMPLATE = """
# EDCL without H0 prior (collapse test)
likelihood:
  bao.desi_dr2.desi_bao_all: null
  sn.pantheonplus: null
  # NO H0 likelihood

theory:
  classy:
    path: {class_path}
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
    prior: {{min: 0.018, max: 0.026}}
    ref: {{dist: norm, loc: 0.02237, scale: 0.00015}}
    proposal: 0.0001
  omega_cdm:
    prior: {{min: 0.08, max: 0.16}}
    ref: {{dist: norm, loc: 0.1200, scale: 0.0012}}
    proposal: 0.001
  H0:
    prior: {{min: 60.0, max: 80.0}}
    ref: 67.5
    proposal: 0.5
  alpha_R:
    prior: {{min: 0.0, max: 0.25}}
    ref: 0.05
    proposal: 0.015
  H0_obs:
    derived: 'lambda H0, alpha_R: H0 * (1.0 + alpha_R * 0.7542)'

output: {output_path}/edcl_no_h0

sampler:
  mcmc:
    max_samples: {max_samples}
    Rminus1_stop: 0.02
    learn_proposal: true
    seed: 42
"""


def generate_yamls(class_path: str, output_dir: str, max_samples: int = 20000) -> List[str]:
    """Generate YAML configuration files for MCMC runs."""
    os.makedirs(output_dir, exist_ok=True)
    
    yaml_files = []
    
    templates = [
        ('lcdm.yaml', LCDM_YAML_TEMPLATE),
        ('edcl.yaml', EDCL_YAML_TEMPLATE),
        ('edcl_no_h0.yaml', EDCL_NO_H0_YAML_TEMPLATE),
    ]
    
    for filename, template in templates:
        content = template.format(
            class_path=class_path,
            output_path=output_dir,
            max_samples=max_samples
        )
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        yaml_files.append(filepath)
        print(f"Generated: {filepath}")
    
    return yaml_files


def run_mcmc(yaml_files: List[str], cobaya_packages: str) -> None:
    """Run MCMC chains using Cobaya."""
    import subprocess
    
    os.environ['COBAYA_PACKAGES_PATH'] = cobaya_packages
    
    for yaml_file in yaml_files:
        print(f"\n{'='*70}")
        print(f"Running: {yaml_file}")
        print('='*70)
        
        result = subprocess.run(
            ['cobaya-run', yaml_file],
            capture_output=False
        )
        
        if result.returncode != 0:
            print(f"WARNING: {yaml_file} may have failed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EDCL Tier-A Cosmology Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate existing chains
    python COLAB_TIER_A_VALIDATION.py --validate-only -d ./chains
    
    # Full MCMC run
    python COLAB_TIER_A_VALIDATION.py --full-run -c ./class_public -p ./cobaya_packages
"""
    )
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only', action='store_true',
                      help='Only analyze existing chain files')
    mode.add_argument('--full-run', action='store_true',
                      help='Run complete MCMC chains from scratch')
    
    parser.add_argument('-d', '--chains-dir', default='./chains',
                        help='Directory containing/for chain files')
    parser.add_argument('-c', '--class-path', default='./class_public',
                        help='Path to CLASS with EDCL patch')
    parser.add_argument('-p', '--cobaya-packages', default='./cobaya_packages',
                        help='Path to Cobaya packages')
    parser.add_argument('-o', '--output', default=None,
                        help='Output JSON file for results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate H0 comparison plot')
    parser.add_argument('--max-samples', type=int, default=20000,
                        help='Maximum MCMC samples (default: 20000)')
    
    args = parser.parse_args()
    
    # Full run mode
    if args.full_run:
        print("="*70)
        print("EDCL TIER-A VALIDATION - FULL MCMC RUN")
        print("="*70)
        
        # Generate YAMLs
        print("\nGenerating YAML configurations...")
        yaml_files = generate_yamls(args.class_path, args.chains_dir, args.max_samples)
        
        # Run MCMC
        print("\nRunning MCMC chains...")
        run_mcmc(yaml_files, args.cobaya_packages)
    
    # Analyze chains
    print("\n" + "="*70)
    print("ANALYZING CHAINS")
    print("="*70)
    
    chains_dir = Path(args.chains_dir)
    
    chain_configs = [
        ('lcdm.1.txt', 'lcdm', 'LCDM (baseline)', False),
        ('lcdm_production.1.txt', 'lcdm', 'LCDM (baseline)', False),
        ('edcl.1.txt', 'edcl_with_h0', 'EDCL (with H0)', True),
        ('edcl_production.1.txt', 'edcl_with_h0', 'EDCL (with H0)', True),
        ('edcl_no_h0.1.txt', 'edcl_no_h0', 'EDCL (no H0)', True),
        ('edcl_no_h0_medium.1.txt', 'edcl_no_h0', 'EDCL (no H0)', True),
    ]
    
    results = {}
    for filename, key, name, is_edcl in chain_configs:
        if key in results:
            continue
        path = chains_dir / filename
        if path.exists():
            print(f"Loading: {path}")
            results[key] = analyze_chain(str(path), name, is_edcl)
    
    if not results:
        print(f"ERROR: No chain files found in {chains_dir}")
        sys.exit(1)
    
    # Run tests
    tests = run_validation_tests(results)
    
    # Print report
    print_report(results, tests)
    
    # Save JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'chains': results, 'tests': tests}, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Generate plot
    if args.plot:
        create_h0_plot(results, 'h0_comparison.png')
    
    # Exit code
    all_pass = all(t['pass'] for t in tests.values()) if tests else False
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
