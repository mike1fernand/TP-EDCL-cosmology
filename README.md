# TP‑EDCL Repro Pack (referee‑oriented)

This repository is a **PRD‑referee‑style** reproducibility pack for the manuscript:

**“Two‑Perspective Quantum Dynamics in Discrete Spacetime”**

It is structured to separate:
- **Track‑0 (math layer, no external data):** kernel‑only diagnostics and plots.
- **Tier‑B (formalism credibility):** discrete‑lattice simulations that validate invariance claims.
- **Tier‑A (cosmology pipeline):** scaffolding for CLASS v3.3.4 (tag `v3.3.4` in `class_public`) + Cobaya v3.6 reproduction of quoted posteriors/Δχ²/ΔlnZ.

The guiding principle is: **claim → diagnostic → tolerance → convergence/integrity test**.

---

## Quick start (no external downloads)

### Install
```bash
pip install -r requirements.txt
```

### Generate all Track‑0 + Tier‑B figures
```bash
python scripts/run_all.py
```

### Run unit tests (Track‑0 + Tier‑B)
```bash
python -m unittest discover -s tests -v
```

Artifacts are written under:
- `paper_artifacts/track0/`
- `paper_artifacts/` (Tier‑B figures)

---

## Contents

### Track‑0 (kernel‑only)
- `track0/kernel.py`
  - Implements the manuscript’s kernel shape and a claim‑consistent variant.
  - Normalizes the kernel by reproducing the paper’s quoted **f_norm = 0.7542** under the stated settings.
- `track0/make_fig_kernel_consistency.py`
  - Produces `paper_artifacts/track0/fig_kernel_consistency.png` and a text report.
  - Explicitly documents whether the kernel variant satisfies the manuscript’s “high‑z safety” saturation claim.

### Tier‑B (lattice demonstrations)
- `tierB/sim_theorem31_wavepacket_validation.py`
  - Matched vs mismatched calibration constant‑speed criterion.
- `tierB/sim_theorem31_local_speed_validation.py`
  - Theorem \ref{thm:localc-1D} **small‑k, adiabatic** local‑speed validation using the
    paper’s minimal recipe profiles (linear n(x), exponential J(n)). Includes a
    convergence sweep and the Lemma‑style worldline/clock diagnostic.
- `tierB/sim_interface_scattering_invariance.py`
  - Interface scattering + flux invariance under time reparameterization.
- `tierB/sim_spectral_mapping.py`
  - Spectral Jacobian mapping (exact for constant calibration; controlled deviation otherwise).
- `tierB/sim_lr_cone_rescaling.py`
  - LR‑front / cone rescaling demonstration under calibration.

### Tier‑A (cosmology harness; requires external software/data)
See `cosmology/README.md`. This includes:
- YAML templates matching the manuscript’s reported likelihood configuration.
- “No assumptions” component discovery script.
- Smoke tests that verify the patched CLASS build accepts EDCL parameters.

---

## Traceability ledger

See `traceability.md` for the mapping:
paper claim → run/script → diagnostic → tolerance → test.

---

## Colab notebooks

- `colab/EDCL_Track0_KernelOnly.ipynb`
- `colab/EDCL_TierA_Cobaya_CLASS.ipynb`


### Tier‑A0 (CLASS background preflight; no likelihoods)

This is the **minimum** Tier‑A gate a PRD referee will expect before you quote any Δχ²/ΔlnZ:
- patched CLASS builds cleanly,
- EDCL parameters are accepted by the parser,
- the LCDM limit holds (EDCL enabled but α_R=0 ⇒ exact LCDM),
- the high‑z safety claim holds,
- Track‑0 kernel‑only curve is consistent with the CLASS implementation (same kernel choice).

Run (after building patched CLASS):

```bash
python cosmology/scripts/preflight_tiera_background.py \
  --class-path /path/to/patched/class_public \
  --alpha_R 0.11824 --log10_l0 -20.908 \
  --kappa_tick 0.0833333333333 --c4 0.06 \
  --zeta 0.5 --kernel exp
```

Outputs are written to `cosmology/paper_artifacts/`.
