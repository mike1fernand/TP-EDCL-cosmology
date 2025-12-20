# Two Perspectives - EDCL Validation

This repository is the **reproducibility and validation harness** for the manuscript:

**Two–Perspective Quantum Dynamics in Discrete Spacetime** (Dec14 “referee-fix” draft)

It is organized to match how a PRD referee typically audits computational evidence:

- **Tier‑B (formalism credibility):** discrete‑lattice simulations that validate the paper’s key theorems/invariants in Sections 2–4.
- **Track‑0 (kernel-only consistency):** kernel-only checks of the Phase‑1 cosmology mapping (no CLASS, no external likelihood datasets).
- **Tier‑A0 (CLASS preflight):** background-only verification of the Phase‑1 \(H(z)\) ratio using **patched CLASS** (no likelihoods, no chains).  
  This is the *paper-facing* cosmology evidence, and it is cross‑checked against Track‑0.
- **Tier‑A1 (Cobaya likelihood reproduction):** **not claimed as complete** in the current paper draft. This repo includes scaffolding only, so future chain releases can be reproduced without “guessing” component names/paths.

If you arrived here from the paper, start with **Quick start** below, then see **Paper claim → script → artifact** to map manuscript claims to validation runs.

---

## Repository layout

- `tierB/` — Tier‑B validation simulations (pure Python; deterministic)
- `track0/` — Track‑0 kernel-only checks (pure Python; deterministic)
- `cosmology/` — Tier‑A0 preflight + Tier‑A1 scaffolding (CLASS/Cobaya)
- `paper_artifacts/` — generated figures/reports for Track‑0 and Tier‑B
- `scripts/` — one-command runners
- `tests/` — unit tests
- `traceability.md` — claim → diagnostic → tolerance → acceptance gate ledger
- `colab/` — notebooks/runners (especially useful for Tier‑A work)

Notes:
- Some example configs (e.g., `edcl.yaml`, `bestfit.yaml`, `example_chain.txt`) and the patch file (`class_edcl.patch`) currently live at the repo root. This is not a blocker; they are referenced from the README and `cosmology/README.md`.

---

## Quick start (Track‑0 + Tier‑B; no external downloads)

### Step 0 — Install Python

You need **Python 3**. If you do not have it installed, install Python 3.10+ (3.11 recommended) and ensure you can run:

```bash
python --version
```

### Step 1 — Download or clone the repo

**Option A: Download ZIP (no git required)**
1. On GitHub, click the green **Code** button.
2. Click **Download ZIP**.
3. Unzip it.
4. Open a terminal in the unzipped folder.

**Option B: Clone with git**
```bash
git clone https://github.com/mike1fernand/two-perspective-edcl-validation.git
cd two-perspective-edcl-validation
```

### Step 2 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 3 — Generate Track‑0 + Tier‑B figures

```bash
python scripts/run_all.py
```

Expected outputs:
- `paper_artifacts/track0/fig_kernel_consistency.png`
- `paper_artifacts/track0/kernel_consistency_report.txt`
- Tier‑B figures under `paper_artifacts/` (see mapping section below)

### Step 4 — Run unit tests (recommended)

```bash
python -m unittest discover -s tests -v
```

---

## Paper claim → script → artifact (what validates what)

This section maps the manuscript’s major *computationally supported* claims to the scripts and artifacts in this repo.

### A) Phase‑1 cosmology mapping: background \(H(z)\) ratio (anomalies section)

**Paper claim:** A Phase‑1 background mapping produces a low‑\(z\) enhancement in \(H(z)\) that saturates back to unity at moderate/high redshift (“high‑z safety”).

Primary evidence in the paper is **Tier‑A0 (CLASS preflight)**:
- Script: `cosmology/scripts/preflight_tiera_background.py`
- Outputs (written under `cosmology/paper_artifacts/`):
  - `hubble_ratio_from_class.csv`
  - `fig_hubble_ratio_from_class.png`
  - `preflight_report.txt`
  - `preflight_summary.json`
  - `fig_track0_vs_class.png`
  - `track0_vs_class_report.txt`

Secondary evidence is **Track‑0 (kernel-only)**:
- Script: `track0/make_fig_kernel_consistency.py`
- Outputs:
  - `paper_artifacts/track0/fig_kernel_consistency.png`
  - `paper_artifacts/track0/kernel_consistency_report.txt`

How to run Tier‑A0 is documented in `cosmology/README.md`.

### B) Constant local speed iff matched calibration (formalism core)

**Paper claim:** In the adiabatic + narrow-band regime, the observer-frame local speed is spatially constant (to leading order) **iff** \(F[n]\propto J(n)\).  
The referee-grade validation computes the theorem’s small parameters and includes a necessity diagnostic via a residual of \(\partial_x\ln J - \partial_x\ln F\).

- Script: `tierB/sim_theorem31_realworld_validation.py`
- Outputs:
  - `paper_artifacts/fig_theorem31_realworld_validation.png`
  - `paper_artifacts/theorem31_realworld_report.txt`

Additional (optional) supporting demos:
- `tierB/sim_theorem31_local_speed_validation.py`
- `tierB/sim_theorem31_wavepacket_validation.py`

### C) Interface scattering invariance (R/T invariance under calibration)

**Paper claim:** Dimensionless scattering observables (R/T) and integrated flux measures are invariant under admissible time calibration \(dt=F\,d\tau\), given consistent current transformations.

- Script: `tierB/sim_interface_scattering_invariance.py`
- Output: `paper_artifacts/fig_interface_scattering_invariance.png`

### D) Spectral Jacobian mapping (\(\tau\)-spectrum ↔ \(t\)-spectrum)

**Paper claim:** Spectra transform between \(\tau\) and \(t\) by a Jacobian rule (exact for constant \(F\); controlled deviation for slowly varying calibration), with an invariant integrated measure within tolerance.

- Script: `tierB/sim_spectral_mapping.py`
- Output: `paper_artifacts/fig_spectral_mapping.png`

### E) LR front / cone rescaling proxy (observer-time microcausality rescaling)

**Paper claim:** Front structure rescales predictably under observer-time calibration.

- Script: `tierB/sim_lr_cone_rescaling.py`
- Output: `paper_artifacts/fig_lr_cone_rescaling.png`

---

## Tier‑A0 / Tier‑A1 status (what is and is not provided)

- **Tier‑A0** (background-only CLASS preflight) is supported here and is the paper-facing cosmology validation gate.  
  See `cosmology/README.md` for step-by-step instructions.
- **Tier‑A1** (full Cobaya likelihood reproduction) is **not** included as completed chains in this repository at this time.  
  The repo includes templates and helper scripts so the eventual chain release can be reproduced without assumptions.

---

## Troubleshooting (Track‑0/Tier‑B)

### “python: command not found”
On some systems, use `python3` instead of `python`:
```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_all.py
```

### Outputs not updating / missing figures
Re-run the generator, then check the artifact directories:
- `paper_artifacts/track0/`
- `paper_artifacts/`

---

## Traceability ledger

For a full, referee-style mapping of:
**paper claim → run/script → diagnostic → tolerance → acceptance gate**, see:

- `traceability.md`

---

## Citation

If you use this code or the generated figures, cite the associated manuscript and (optionally) this repository URL:
- https://github.com/mike1fernand/two-perspective-edcl-validation
