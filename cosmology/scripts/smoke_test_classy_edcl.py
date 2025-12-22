#!/usr/bin/env python3
"""
Smoke-test that the patched CLASS build accepts the EDCL parameters.

This is a "no assumptions" check that prevents a referee from claiming
the EDCL pipeline is misconfigured.

Usage:
  python cosmology/scripts/smoke_test_classy_edcl.py --class-path /path/to/class

What it checks:
- That importing classy works.
- That passing edcl_on, kappa_tick, c4, alpha_R, log10_l0 does NOT throw an "unknown parameter" error.
- That baseline (edcl_on=False) and EDCL (edcl_on=True) computations complete.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback


def _add_class_python_path(class_path: str) -> None:
    # CLASS python wrapper typically lives in <class_path>/python
    cand = os.path.join(class_path, "python")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)
    sys.path.insert(0, class_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-path", required=True, help="Path to CLASS root (tested with class_public tag v3.3.4).")
    args = ap.parse_args()

    _add_class_python_path(args.class_path)

    try:
        from classy import Class  # type: ignore
    except Exception as e:
        print("FAILED to import classy.Class. Check that CLASS was built with `make classy`.")
        raise

    # Minimal vanilla params sufficient for a background run.
    base = {
        "output": "tCl,pCl,lCl",  # safe defaults; may be ignored depending on build
        "l_max_scalars": 10,
        "h": 0.67,
        "omega_b": 0.02237,
        "omega_cdm": 0.1200,
        "tau_reio": 0.0544,
        "A_s": 2.1e-9,
        "n_s": 0.965,
    }

    # EDCL extra args as shown in paper snippet
    edcl = {
        "edcl_on": "yes",
        "kappa_tick": 1.0/12.0,
        "c4": 0.06,
        "alpha_R": 0.118,
        "log10_l0": -20.91,
    }

    # Baseline run
    cosmo = Class()
    try:
        cosmo.set(base)
        cosmo.compute()
        bg = cosmo.get_background()
        print(f"Baseline compute OK. Background keys: {list(bg.keys())[:10]} ...")
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()

    # EDCL run (requires patch)
    cosmo2 = Class()
    try:
        params = dict(base)
        params.update(edcl)
        cosmo2.set(params)
        cosmo2.compute()
        bg2 = cosmo2.get_background()
        print(f"EDCL compute OK. Background keys: {list(bg2.keys())[:10]} ...")
        print("If EDCL is truly active, H(z) or derived distances should differ from baseline.")
        print("Next step: run cosmology/scripts/make_fig_hubble_ratio_from_class.py to confirm.")
    except Exception as e:
        print("EDCL compute FAILED.")
        print("This usually means the CLASS patch is not applied OR parameter names do not match the patched code.")
        print("Exception:")
        traceback.print_exc()
        sys.exit(2)
    finally:
        cosmo2.struct_cleanup()
        cosmo2.empty()


if __name__ == "__main__":
    main()
