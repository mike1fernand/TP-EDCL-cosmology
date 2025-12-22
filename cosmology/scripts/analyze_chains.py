#!/usr/bin/env python3
"""
analyze_chains.py - Analyze EDCL Tier-A validation MCMC chains.

This script loads Cobaya chain files and computes:
- Parameter constraints (mean, std, percentiles)
- Best-fit chi-squared values
- Model comparison (EDCL vs ΛCDM)
- Validation tests (activation, collapse, H0 match)

Usage:
    python analyze_chains.py                    # Analyze chains in ./chains/
    python analyze_chains.py --chains-dir /path/to/chains
    python analyze_chains.py --output report.json --plot

Requirements:
    - numpy
    - matplotlib (optional, for plotting)
    - json (standard library)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


def load_chain(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a Cobaya chain file and parse its header.
    
    Args:
        path: Path to chain file (e.g., 'chains/edcl.1.txt')
        
    Returns:
        Tuple of (chain_data, header_names)
    """
    with open(path, 'r') as f:
        header_line = f.readline()
    
    # Parse header - handle both '#' prefix and space-separated
    header = header_line.strip().lstrip('#').split()
    
    # Load data
    chain = np.loadtxt(path)
    
    return chain, header


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """
    Compute weighted statistics for a parameter.
    
    Args:
        values: Parameter values
        weights: Sample weights
        
    Returns:
        Dictionary with mean, std, median, and percentiles
    """
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    std = np.sqrt(variance)
    
    # For percentiles, we need to sort and use cumulative weights
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    cumsum /= cumsum[-1]  # Normalize to [0, 1]
    
    # Find percentile values
    q16 = sorted_values[np.searchsorted(cumsum, 0.16)]
    q50 = sorted_values[np.searchsorted(cumsum, 0.50)]
    q84 = sorted_values[np.searchsorted(cumsum, 0.84)]
    q95 = sorted_values[np.searchsorted(cumsum, 0.95)]
    q05 = sorted_values[np.searchsorted(cumsum, 0.05)]
    
    return {
        'mean': float(mean),
        'std': float(std),
        'median': float(q50),
        'q05': float(q05),
        'q16': float(q16),
        'q84': float(q84),
        'q95': float(q95),
    }


def analyze_chain(path: str, name: str, is_edcl: bool = False) -> Dict[str, Any]:
    """
    Analyze a single chain file.
    
    Args:
        path: Path to chain file
        name: Display name for this chain
        is_edcl: Whether this is an EDCL chain (has alpha_R)
        
    Returns:
        Dictionary with analysis results
    """
    chain, header = load_chain(path)
    
    # First column is always weight
    weights = chain[:, 0]
    
    result = {
        'name': name,
        'path': path,
        'n_samples': len(chain),
        'eff_samples': float(np.sum(weights)),
        'parameters': {},
    }
    
    # Analyze common parameters
    common_params = ['H0', 'omega_b', 'omega_cdm']
    for param in common_params:
        if param in header:
            idx = header.index(param)
            result['parameters'][param] = weighted_stats(chain[:, idx], weights)
    
    # Chi-squared
    if 'chi2' in header:
        chi2_idx = header.index('chi2')
        chi2 = chain[:, chi2_idx]
        result['chi2_best'] = float(np.min(chi2))
        result['chi2_mean'] = float(np.average(chi2, weights=weights))
    
    # EDCL-specific parameters
    if is_edcl:
        edcl_params = ['alpha_R', 'H0_obs', 'delta0']
        for param in edcl_params:
            if param in header:
                idx = header.index(param)
                result['parameters'][param] = weighted_stats(chain[:, idx], weights)
    
    return result


def run_validation_tests(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Run validation tests on the analysis results.
    
    Args:
        results: Dictionary of chain analysis results
        
    Returns:
        Dictionary with test results
    """
    tests = {}
    
    # Test 1: Activation - alpha_R significantly non-zero with H0 prior
    if 'edcl_with_h0' in results:
        edcl = results['edcl_with_h0']
        if 'alpha_R' in edcl['parameters']:
            alpha = edcl['parameters']['alpha_R']
            # Check if lower 68% bound is above threshold
            threshold = 0.02
            tests['activation'] = {
                'description': 'alpha_R significantly non-zero with H0 prior',
                'alpha_R_mean': alpha['mean'],
                'alpha_R_std': alpha['std'],
                'alpha_R_q16': alpha['q16'],
                'alpha_R_q84': alpha['q84'],
                'threshold': threshold,
                'pass': alpha['q16'] > threshold,
            }
    
    # Test 2: Collapse - alpha_R drops without H0 prior
    if 'edcl_with_h0' in results and 'edcl_no_h0' in results:
        with_h0 = results['edcl_with_h0']['parameters'].get('alpha_R', {})
        without_h0 = results['edcl_no_h0']['parameters'].get('alpha_R', {})
        
        if with_h0 and without_h0:
            collapse_ratio = without_h0['mean'] / with_h0['mean'] if with_h0['mean'] > 0 else 1.0
            tests['collapse'] = {
                'description': 'alpha_R collapses without H0 prior',
                'alpha_R_with_h0': with_h0['mean'],
                'alpha_R_without_h0': without_h0['mean'],
                'collapse_ratio': collapse_ratio,
                'reduction_percent': (1 - collapse_ratio) * 100,
                'pass': collapse_ratio < 0.5,
            }
    
    # Test 3: H0 match - H0_obs matches Riess measurement
    if 'edcl_with_h0' in results:
        edcl = results['edcl_with_h0']
        if 'H0_obs' in edcl['parameters']:
            h0_obs = edcl['parameters']['H0_obs']
            h0_riess = 73.04
            sigma_riess = 1.04
            
            # Combined uncertainty
            sigma_combined = np.sqrt(h0_obs['std']**2 + sigma_riess**2)
            tension = abs(h0_obs['mean'] - h0_riess) / sigma_combined
            
            tests['h0_match'] = {
                'description': 'H0_obs matches Riess measurement',
                'H0_obs_mean': h0_obs['mean'],
                'H0_obs_std': h0_obs['std'],
                'H0_riess': h0_riess,
                'sigma_riess': sigma_riess,
                'tension_sigma': tension,
                'pass': tension < 1.0,
            }
    
    # Test 4: Chi-squared improvement
    if 'lcdm' in results and 'edcl_with_h0' in results:
        lcdm = results['lcdm']
        edcl = results['edcl_with_h0']
        
        if 'chi2_best' in lcdm and 'chi2_best' in edcl:
            delta_chi2 = edcl['chi2_best'] - lcdm['chi2_best']
            
            tests['chi2_improvement'] = {
                'description': 'EDCL improves chi2 vs LCDM',
                'lcdm_chi2_best': lcdm['chi2_best'],
                'edcl_chi2_best': edcl['chi2_best'],
                'delta_chi2': delta_chi2,
                'pass': delta_chi2 < 0,
            }
    
    return tests


def print_results(results: Dict[str, Dict], tests: Dict[str, Dict]) -> None:
    """Print formatted results to stdout."""
    
    print("=" * 70)
    print("TIER-A VALIDATION ANALYSIS")
    print("=" * 70)
    
    # Print chain summaries
    for key, res in results.items():
        print(f"\n{res['name']}:")
        print(f"  Samples: {res['n_samples']}, Effective: {res['eff_samples']:.0f}")
        
        if 'H0' in res['parameters']:
            h0 = res['parameters']['H0']
            print(f"  H0 = {h0['mean']:.2f} +/- {h0['std']:.2f}")
        
        if 'alpha_R' in res['parameters']:
            alpha = res['parameters']['alpha_R']
            print(f"  alpha_R = {alpha['mean']:.4f} +/- {alpha['std']:.4f} "
                  f"[{alpha['q16']:.4f}, {alpha['q84']:.4f}]")
        
        if 'H0_obs' in res['parameters']:
            h0_obs = res['parameters']['H0_obs']
            print(f"  H0_obs = {h0_obs['mean']:.2f} +/- {h0_obs['std']:.2f}")
        
        if 'chi2_best' in res:
            print(f"  Best chi2 = {res['chi2_best']:.2f}")
    
    # Print validation tests
    print("\n" + "=" * 70)
    print("VALIDATION TESTS")
    print("=" * 70)
    
    for test_name, test in tests.items():
        status = "PASS" if test['pass'] else "FAIL"
        print(f"\n{test_name.upper()}: {status}")
        print(f"  {test['description']}")
        
        # Print test-specific details
        if test_name == 'activation':
            print(f"  alpha_R = {test['alpha_R_mean']:.4f} +/- {test['alpha_R_std']:.4f}")
            print(f"  68% CI: [{test['alpha_R_q16']:.4f}, {test['alpha_R_q84']:.4f}]")
            print(f"  Lower bound {'>=' if test['pass'] else '<'} {test['threshold']}")
        
        elif test_name == 'collapse':
            print(f"  With H0: alpha_R = {test['alpha_R_with_h0']:.4f}")
            print(f"  Without H0: alpha_R = {test['alpha_R_without_h0']:.4f}")
            print(f"  Reduction: {test['reduction_percent']:.0f}%")
        
        elif test_name == 'h0_match':
            print(f"  H0_obs = {test['H0_obs_mean']:.2f} +/- {test['H0_obs_std']:.2f}")
            print(f"  Riess = {test['H0_riess']:.2f} +/- {test['sigma_riess']:.2f}")
            print(f"  Tension: {test['tension_sigma']:.2f} sigma")
        
        elif test_name == 'chi2_improvement':
            print(f"  LCDM chi2 = {test['lcdm_chi2_best']:.2f}")
            print(f"  EDCL chi2 = {test['edcl_chi2_best']:.2f}")
            print(f"  Delta chi2 = {test['delta_chi2']:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\nTests passed: {n_pass}/{n_total}")
    
    if n_pass == n_total:
        print("\nALL VALIDATION TESTS PASS")
    else:
        print(f"\n{n_total - n_pass} test(s) failed")


def create_plot(results: Dict[str, Dict], output_path: str) -> None:
    """Create comparison plot of H0 posteriors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot LCDM H0
    if 'lcdm' in results:
        lcdm = results['lcdm']
        if 'H0' in lcdm['parameters']:
            h0 = lcdm['parameters']['H0']
            ax.axvspan(h0['mean'] - h0['std'], h0['mean'] + h0['std'], 
                      alpha=0.3, color='blue', label=f'LCDM: {h0["mean"]:.1f} +/- {h0["std"]:.1f}')
    
    # Plot EDCL H0_obs
    if 'edcl_with_h0' in results:
        edcl = results['edcl_with_h0']
        if 'H0_obs' in edcl['parameters']:
            h0 = edcl['parameters']['H0_obs']
            ax.axvspan(h0['mean'] - h0['std'], h0['mean'] + h0['std'],
                      alpha=0.3, color='green', label=f'EDCL H0_obs: {h0["mean"]:.1f} +/- {h0["std"]:.1f}')
    
    # Plot Riess measurement
    ax.axvline(73.04, color='red', linestyle='--', linewidth=2, label='Riess (73.04 +/- 1.04)')
    ax.axvspan(73.04 - 1.04, 73.04 + 1.04, alpha=0.2, color='red')
    
    ax.set_xlabel('H0 (km/s/Mpc)', fontsize=12)
    ax.set_ylabel('Posterior', fontsize=12)
    ax.set_title('EDCL Resolution of Hubble Tension', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(65, 78)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze EDCL Tier-A validation MCMC chains',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-d', '--chains-dir',
        default='./chains',
        help='Directory containing chain files (default: ./chains)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output file for JSON results (optional)'
    )
    
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='Generate H0 comparison plot'
    )
    
    parser.add_argument(
        '--plot-output',
        default='h0_comparison.png',
        help='Output path for plot (default: h0_comparison.png)'
    )
    
    args = parser.parse_args()
    
    # Find chain files
    chains_dir = Path(args.chains_dir)
    if not chains_dir.exists():
        print(f"ERROR: Chains directory not found: {chains_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Define expected chain files and their configurations
    chain_configs = [
        # (filename pattern, key, display name, is_edcl)
        ('lcdm_production.1.txt', 'lcdm', 'LCDM (baseline)', False),
        ('lcdm_medium.1.txt', 'lcdm', 'LCDM (baseline)', False),
        ('lcdm_quick.1.txt', 'lcdm', 'LCDM (baseline)', False),
        ('edcl_production.1.txt', 'edcl_with_h0', 'EDCL (with H0)', True),
        ('edcl_medium.1.txt', 'edcl_with_h0', 'EDCL (with H0)', True),
        ('edcl_fixed_test.1.txt', 'edcl_with_h0', 'EDCL (with H0)', True),
        ('edcl_no_h0_medium.1.txt', 'edcl_no_h0', 'EDCL (no H0)', True),
        ('edcl_fixed_no_sh0es.1.txt', 'edcl_no_h0', 'EDCL (no H0)', True),
    ]
    
    # Find available chains (prefer production > medium > quick)
    results = {}
    for filename, key, name, is_edcl in chain_configs:
        if key in results:
            continue  # Already have this type
        
        path = chains_dir / filename
        if path.exists():
            print(f"Loading: {path}")
            results[key] = analyze_chain(str(path), name, is_edcl)
    
    if not results:
        print(f"ERROR: No chain files found in {chains_dir}", file=sys.stderr)
        print("Expected files like: lcdm_production.1.txt, edcl_production.1.txt", file=sys.stderr)
        sys.exit(1)
    
    # Run validation tests
    tests = run_validation_tests(results)
    
    # Print results
    print_results(results, tests)
    
    # Save JSON output
    if args.output:
        output_data = {
            'chains': results,
            'tests': tests,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Generate plot
    if args.plot:
        create_plot(results, args.plot_output)
    
    # Exit with appropriate code
    all_pass = all(t['pass'] for t in tests.values()) if tests else False
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
