# Traceability ledger (paper → code → artifacts)

This file is intended to make a referee’s life easy: it gives a **minimal, direct mapping** from paper elements to the exact code paths and output artifacts in this repository.

For the expanded table, see `docs/VALIDATION_MATRIX.md`.

---

## Tier‑B (pure Python) — core non‑cosmology validations

| Paper element | Script(s) | Output artifact(s) |
|---|---|---|
| Theorem 3.1 wavepacket validation | `scripts/run_all_tierB.py` | `paper_artifacts/fig_theorem31_wavepacket_validation.png` |
| Theorem 3.1 local-speed validation | `scripts/run_all_tierB.py` | `paper_artifacts/fig_theorem31_local_speed_validation.png`, `paper_artifacts/theorem31_local_speed_report.txt` |
| Interface scattering invariance | `scripts/run_all_tierB.py` | `paper_artifacts/fig_interface_scattering_invariance.png` |
| Spectral mapping validation | `scripts/run_all_tierB.py` | `paper_artifacts/fig_spectral_mapping.png` |
| LR cone/front rescaling | `scripts/run_all_tierB.py` | `paper_artifacts/fig_lr_cone_rescaling.png` |
| Theorem 3.1 “real‑world” variant | `scripts/run_all_tierB.py` | `paper_artifacts/fig_theorem31_realworld_validation.png` |

---

## Track‑0 (pure Python) — kernel consistency

| Paper element | Script(s) | Output artifact(s) |
|---|---|---|
| Kernel consistency check | `track0/run_track0_kernel_consistency.py` | `paper_artifacts/track0/fig_kernel_consistency.png` |

---

## Tier‑A (CLASS + Cobaya) — late‑only cosmology validation

Canonical entrypoints:
- Colab: `COLAB_TIER_A_VALIDATION.py`
- Local: `RUN_TIER_A_VALIDATION.sh`

Primary validation scripts:
- `cosmology/scripts/analyze_chains.py`
- `cosmology/scripts/validate_tiera1_lateonly_results.py`

Primary Tier‑A documentation:
- `TIER_A_COMPLETE_DOCUMENTATION.md`
- `cosmology/docs/H0_LIKELIHOOD_FIX.md`

Tier‑A produces:
- preflight plots: `cosmology/paper_artifacts/`
- run output workdir: `edcl_tiera1_YYYYMMDD_HHMMSS/` (not intended for git; publish as Release asset)
- optional bundle zip in the workdir (publish as Release asset)

