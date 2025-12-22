# Optimal path forward (PRD referee perspective)

This document is written as if the reader is a PRD referee evaluating:
- internal consistency of the EDCL kernel definition,
- correctness and reproducibility of the lattice formalism claims (Tier‑B),
- credibility of the cosmology pipeline (Tier‑A).

The goal is to minimize reviewer “attack surface” by making every claim testable.

---

## Pass 1 — Internal consistency audit (kernel + saturation)

### What a referee will check
The manuscript simultaneously asserts:
1) the kernel is given by Eq. (kernel‑shape), and
2) δ(z)→0 at high z and the background ratio saturates to unity by z~2.

A referee will ask for **a single, unambiguous kernel definition** that satisfies the stated behavior.

### What this repo implements
- `track0/kernel.py` implements:
  - `paper_equation_1mexp`: K ∝ (a'/a)^2 [1 − exp(−z(a')/ζ)]  (as written)
  - `paper_claim_exp`: K ∝ (a'/a)^2 exp(−z(a')/ζ)            (claim‑consistent early‑time suppression)

- `track0/make_fig_kernel_consistency.py` generates:
  - `paper_artifacts/track0/fig_kernel_consistency.png`
  - `paper_artifacts/track0/kernel_consistency_report.txt`

### Acceptance criteria
- `f_norm` reproduction: exact (tested).
- “high‑z safety”: |H_ratio−1| ≤ 0.2% by z≈2 (tested under Track‑0 mapping).

### Expected outcome
If the manuscript’s equation variant does not satisfy the stated saturation, the paper must be revised:
- either correct the kernel formula,
- or correct the claim, or
- define z(·) / K normalization differently in a way that is explicitly stated and testable.

This is not optional for PRD: it is a primary‑claim consistency issue.

---

## Pass 2 — Lock down Tier‑B (formalism credibility)

### What a referee will check
That the “DST + calibration” formalism produces the invariance statements claimed, at least in minimal models.

### What is already implemented and tested
- B1 matched calibration constant speed: `tierB/sim_theorem31_wavepacket_validation.py`
- B2 interface scattering invariance: `tierB/sim_interface_scattering_invariance.py`
- B3 spectral Jacobian mapping: `tierB/sim_spectral_mapping.py`
- B4 LR‑cone/front rescaling: `tierB/sim_lr_cone_rescaling.py`

Each has:
- a diagnostic,
- explicit tolerances,
- integrity checks (unitarity/boundary/flux),
- and unit tests under `tests/`.

---

## Pass 3 — Tier‑A “repro harness” (CLASS + Cobaya) without guessing

### What a referee will check
- that the pipeline runs end‑to‑end,
- that ΛCDM is recovered in the appropriate limit,
- that Planck/BAO/SNe/SH0ES are configured without double counting,
- that the paper’s tables are regenerated from a single command.

### What this repo provides
- YAML templates in `cosmology/cobaya/*.yaml.in` matching the manuscript snippet.
- A “no assumptions” YAML renderer:
  - `cosmology/scripts/render_yamls.py`
- Component discovery:
  - `cosmology/scripts/discover_cobaya_components.py`
- Patched‑CLASS smoke tests (no external data):
  - `cosmology/scripts/smoke_test_classy_edcl.py`
  - `cosmology/scripts/make_fig_hubble_ratio_from_class.py`

### What you must supply (cannot be guessed)
- the exact `class_edcl.patch` file for CLASS v3.3.4 (recommended; otherwise use the highest available stable tag and record the exact commit hash)
- verified likelihood datasets (Planck, DESI DR2, PantheonPlus, SH0ES) and their configurations

---

## Pass 4 — Tier‑A full reproduction run + paper artifacts

Once the patch and data are installed, the next steps are:

1) Run baseline ΛCDM chains and EDCL chains
2) Generate:
   - parameter posterior tables
   - χ² breakdown table
   - H(z) ratio figure
   - no‑SH0ES collapse plot/table

This repo includes the wiring; the heavy runs are intentionally not automatic to avoid silent assumptions about installed data.

---

## What to run in Google Colab (requested)

### Recommended Colab sequence
1) Run `colab/EDCL_Track0_KernelOnly.ipynb`
   - produces the kernel consistency figure and report

2) Run `colab/EDCL_TierA_Cobaya_CLASS.ipynb` up to the smoke tests
   - **stop** if the patch is missing/placeholder
   - confirms patched CLASS accepts EDCL parameters and produces a background H(z) ratio plot

3) Optional (only if your Colab environment can install Planck likelihood dependencies):
   - install likelihood data (via `cobaya-install`)
   - run late‑only chains (`*_lateonly.yaml`) first
   - then attempt full Planck configuration

### If you run Colab and want me to finalize Tier‑A scripts
Please paste back:
- the contents of `cosmology/paper_artifacts/cobaya_components.txt`
- any install errors from `cobaya-install`
- and (if successful) the `chains/` output directory tree and summary logs

That will allow the pipeline to be locked down without guessing names/paths.

