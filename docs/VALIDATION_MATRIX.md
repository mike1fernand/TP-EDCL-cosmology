# Validation matrix (paper claim → scripts → artifacts)

This file is intended to be **referee-facing**: each row maps a paper claim (or paper figure) to:
- the script(s) that generate the supporting evidence,
- the expected artifact(s),
- the pass/fail criterion (where applicable).

If you add/remove claims or rename figures, update this file first.

---

## Tier‑B: formalism/kinematics simulations (pure Python)

| Paper claim / figure | What is validated | Script to run | Primary artifact(s) | Pass criterion |
|---|---|---|---|---|
| Theorem 3.1 wavepacket behavior | Wavepacket propagation matches predicted scaling | `python scripts/run_all_tierB.py` | `paper_artifacts/fig_theorem31_wavepacket_validation.png` | Script completes with exit code 0 |
| Theorem 3.1 local-speed validation | Local-speed law holds in adiabatic/small‑k regime | `python scripts/run_all_tierB.py` | `paper_artifacts/fig_theorem31_local_speed_validation.png`, `paper_artifacts/theorem31_local_speed_report.txt` | Script completes with exit code 0 |
| Interface scattering invariance | Scattering is invariant under interface placement (within tolerance) | `python scripts/run_all_tierB.py` | `paper_artifacts/fig_interface_scattering_invariance.png` | Script completes with exit code 0 |
| Spectral mapping | Spectrum mapping identity holds numerically | `python scripts/run_all_tierB.py` | `paper_artifacts/fig_spectral_mapping.png` | Script completes with exit code 0 |
| LR cone/front rescaling | LR cone/front rescaling matches prediction | `python scripts/run_all_tierB.py` | `paper_artifacts/fig_lr_cone_rescaling.png` | Script completes with exit code 0 |
| Real-world Theorem 3.1 variant | Validation on real-world-like parameterization | `python scripts/run_all_tierB.py` | `paper_artifacts/fig_theorem31_realworld_validation.png` | Script completes with exit code 0 |

Notes:
- Tier‑B is designed to run offline with only the Python dependencies in `requirements.txt`.
- For deterministic regeneration, use the default seeds embedded in the scripts.

---

## Track‑0: kernel/FRW consistency (pure Python)

| Paper claim / figure | What is validated | Script to run | Primary artifact(s) | Pass criterion |
|---|---|---|---|---|
| Kernel consistency check | Kernel integration/normalization consistency | `python track0/run_track0_kernel_consistency.py` | `paper_artifacts/track0/fig_kernel_consistency.png` | Script completes with exit code 0 |

---

## Tier‑A: late‑only cosmology validation (CLASS + Cobaya)

Tier‑A is heavier and depends on external downloads/builds. The canonical entrypoints are:
- Colab: `COLAB_TIER_A_VALIDATION.py`
- Local: `RUN_TIER_A_VALIDATION.sh`

### Core claims validated (late‑only)

| Paper claim / result | What is validated | Script(s) | Evidence location | Pass criterion |
|---|---|---|---|---|
| EDCL resolves the local H0 constraint by predicting an observed H0 consistent with Riess | The H0 likelihood is applied to **H0_obs**, not H0_theory | `RUN_TIER_A_VALIDATION.sh` or `COLAB_TIER_A_VALIDATION.py` | Tier‑A report + chain analysis outputs | `H0_obs` matches 73.04 ± 1.04 within tolerance |
| α_R “activates” in the presence of H0 tension | Posterior α_R is non-zero when the H0 constraint is included | same | Tier‑A validation summary | α_R mean/median significantly > 0 (see Tier‑A config thresholds) |
| α_R collapses when the H0 constraint is removed | Posterior α_R shifts toward 0 when H0 prior removed | same | Tier‑A “no H0” chain analysis | α_R reduces by target fraction (see Tier‑A config) |
| Fit improvement (late‑only) | Δχ² between EDCL and ΛCDM late‑only runs | same | Tier‑A validation summary | Δχ² < 0 by threshold |

For full details (including the H0 likelihood fix and production chain sizes), see:
- `TIER_A_COMPLETE_DOCUMENTATION.md`
- `cosmology/docs/H0_LIKELIHOOD_FIX.md`

