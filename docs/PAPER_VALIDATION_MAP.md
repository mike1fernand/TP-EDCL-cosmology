# Paper validation map (Two-Perspective / EDCL)

This document maps paper claims to the scripts and artifacts in this repository (mike1fernand/two-perspective-edcl-validation).

## Tier-B (formalism validations; Sections 2–4)
- Constant local observed speed iff matched calibration (Theorem 3.1):
  - Script: `scripts/run_all.py` (Tier-B bundle)
  - Outputs: `paper_artifacts/fig1_structure_frame.png`, `paper_artifacts/fig2_observer_frame.png` (used in paper), plus Tier-B report files.

- Interface scattering invariance:
  - Script: `scripts/run_all.py`
  - Output: `paper_artifacts/fig_interface_scattering_invariance.png`

- Spectral Jacobian mapping:
  - Script: `scripts/run_all.py`
  - Output: `paper_artifacts/fig_spectral_mapping.png`

- LR cone / microcausality rescaling:
  - Script: `scripts/run_all.py`
  - Output: `paper_artifacts/fig_lr_cone_rescaling.png`

## Track-0 (kernel-only; Phase-1 background mapping)
- Kernel consistency (high-z safety):
  - Script: `scripts/run_all.py`
  - Output: `paper_artifacts/track0/fig_kernel_consistency.png`

## Tier-A (late-only cosmology validation; Hubble subsection)
- Late-only EDCL validation uses DESI DR2 BAO + PantheonPlus + local H0 (Riess 2022), with an EDCL-aware H0 likelihood applied to `H0_obs`.
  - Config templates: `cosmology/cobaya/templates/*lateonly*.template`
  - H0 likelihood: `cosmology/likelihoods/edcl_H0.py` (and/or YAML `external:` form)
  - Runner: `scripts/RUN_TIER_A_VALIDATION.sh` (local), `colab/COLAB_TIER_A_VALIDATION.py` (Colab)
  - Reference chains (paper table reproduction): `cosmology/reference_runs/lateonly/chains/`
  - Analysis: `cosmology/scripts/analyze_chains.py`

See also: `docs/TIER_A_README_SCRIPTS.md`.
