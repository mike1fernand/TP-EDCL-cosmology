# two-perspective-edcl-validation

This repository is the **reproducibility and validation harness** for the manuscript:

**Two–Perspective Quantum Dynamics in Discrete Spacetime**

It is structured the way a PRD referee typically audits computational support:

- **Tier‑B (formalism credibility):** discrete‑lattice simulations that validate the paper’s core *bridging principles, invariants, and the constant‑local‑speed criterion* (Sections 2–4).
- **Track‑0 (kernel-only consistency):** math-only checks of the Phase‑1 cosmology mapping (no CLASS, no external likelihood data).
- **Tier‑A0 (CLASS preflight):** **background-only** verification of the Phase‑1 \(H(z)\) ratio using **patched CLASS** (no likelihoods, no chains).  
  This is the **paper-facing** cosmology evidence; it is also cross‑checked against Track‑0.

Tier‑A1 (full likelihood reproduction with Cobaya) is intentionally **not claimed as complete** in the current paper draft; this repo contains **scaffolding** so that the eventual chain releases are reproducible without guessing component names or paths.

---

## If you arrived here from the paper

Start with:

1) **Quick start** (Track‑0 + Tier‑B; runs anywhere)  
2) **Tier‑A0 CLASS preflight** (generates the paper’s Phase‑1 \(H_{\rm TP}(z)/H_{\rm GR}(z)\) figure; requires patched CLASS)

To see exactly which paper claim is validated by which script/output, use:

- `traceability.md` (claim → diagnostic → tolerance → artifact)

---

## Repository layout

Top-level folders you will use:

- `tierB/` — Tier‑B validation simulations (pure Python; deterministic)
- `track0/` — Track‑0 kernel-only checks (pure Python; deterministic)
- `cosmology/` — Tier‑A0 preflight + Tier‑A1 scaffolding (CLASS/Cobaya)
- `paper_artifacts/` — figures/reports intended to be cited or inspected (Tier‑B + Track‑0 outputs)
- `scripts/` — convenience entry points (one-command reproduction)
- `tests/` — unit tests (fast checks)
- `colab/` — Colab runner(s) for Tier‑A workflows
- `docs/` — referee-oriented notes and status

If you started from a zip that contains a top-level `TP_EDCL_v12/` directory, run commands **from inside that directory** (or move its contents to the repo root).

---

## Quick start (Track‑0 + Tier‑B; no external downloads)

### 1) Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Generate Track‑0 + Tier‑B artifacts

Preferred one-command entry point:

```bash
python scripts/run_all.py
```

This runs:
- Track‑0 kernel consistency: `track0/make_fig_kernel_consistency.py`
- Tier‑B suite runner: `scripts/run_all_tierB.py`

### 3) Run unit tests (optional but recommended)

```bash
python -m unittest discover -s tests -v
```

Expected outputs (created/updated):
- `paper_artifacts/track0/fig_kernel_consistency.png`
- `paper_artifacts/track0/kernel_consistency_report.txt`
- Tier‑B figures under `paper_artifacts/` (see “Paper claim → script → artifact” below)

---

## Paper claim → script → artifact (what validates what)

### A) Phase‑1 cosmology mapping: background \(H(z)\) ratio (paper anomaly section)

**Paper claim:** Phase‑1 background mapping produces a low‑\(z\) enhancement in \(H(z)\) that saturates to unity at moderate/high redshift (“high‑z safety”), and the CLASS implementation agrees with the Track‑0 kernel-only mapping.

**Primary (paper-facing) evidence: Tier‑A0 CLASS preflight**
- Script: `cosmology/scripts/preflight_tiera_background.py`
- Outputs (written to `cosmology/paper_artifacts/` by default):
  - `hubble_ratio_from_class.csv` (raw curve)
  - `fig_hubble_ratio_from_class.png` (figure)
  - `preflight_report.txt` (human summary)
  - `preflight_summary.json` (machine summary)
  - `fig_track0_vs_class.png` (cross-check figure)
  - `track0_vs_class_report.txt` (cross-check metrics)

**Secondary (internal consistency) evidence: Track‑0 kernel-only**
- Script: `track0/make_fig_kernel_consistency.py`
- Outputs:
  - `paper_artifacts/track0/fig_kernel_consistency.png`
  - `paper_artifacts/track0/kernel_consistency_report.txt`

In the paper, the \(H(z)\) ratio figure is cited as **Tier‑A0 output** with a one-line note that it matches Track‑0; the cross-check report is `track0_vs_class_report.txt`.

---

### B) Theorem “constant local speed iff matched calibration” (paper formalism core)

**Paper claim:** In the adiabatic + narrow-band regime, the observer-frame local speed is spatially constant (to leading order) **iff** \(F[n]\propto J(n)\). The validation computes the theorem’s small parameters and includes the necessity diagnostic (residual of \(\partial_x\ln J - \partial_x\ln F\)).

**Referee-grade “real-world” validation (recommended)**
- Script: `tierB/sim_theorem31_realworld_validation.py`
- Outputs:
  - `paper_artifacts/fig_theorem31_realworld_validation.png`
  - `paper_artifacts/theorem31_realworld_report.txt`

**Additional (optional) supporting demos**
- Script: `tierB/sim_theorem31_local_speed_validation.py`
  - Outputs: `paper_artifacts/fig_theorem31_local_speed_validation.png`, `paper_artifacts/theorem31_local_speed_report.txt`
- Script: `tierB/sim_theorem31_wavepacket_validation.py`
  - Output: `paper_artifacts/fig_theorem31_wavepacket_validation.png`

In the paper draft aligned to this repo, the real-world theorem validation is placed in the Appendix validation suite with a one-line pointer from the theorem section.

---

### C) Interface scattering invariance (reflection/transmission invariance under calibration)

**Paper claim:** Dimensionless scattering observables (R/T) and integrated flux measures are invariant under admissible time calibration \(dt = F\,d\tau\) when currents are transformed consistently.

- Script: `tierB/sim_interface_scattering_invariance.py`
- Output: `paper_artifacts/fig_interface_scattering_invariance.png`

---

### D) Spectral Jacobian mapping (\(\tau\)-spectrum ↔ \(t\)-spectrum)

**Paper claim:** Spectra transform between \(\tau\) and \(t\) by a Jacobian rule (exact for constant \(F\); controlled error for slowly varying calibration), with an invariant integrated measure within tolerance.

- Script: `tierB/sim_spectral_mapping.py`
- Output: `paper_artifacts/fig_spectral_mapping.png`

---

### E) LR cone / front rescaling proxy (observer-time microcausality rescaling)

**Paper claim:** Signal/cone front structure rescales predictably under observer-time calibration; operational front speeds scale by \(1/F\) (in the constant-calibration proxy).

- Script: `tierB/sim_lr_cone_rescaling.py`
- Output: `paper_artifacts/fig_lr_cone_rescaling.png`

---

## Reproducing the paper figures (summary)

From repo root:

1) Track‑0 figure + Tier‑B validation suite:
```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
```

2) Tier‑A0 (CLASS) paper-facing \(H(z)\) ratio figure:
- build patched CLASS (see next section), then run:
```bash
python cosmology/scripts/preflight_tiera_background.py \
  --class-path /path/to/patched/class_public \
  --alpha_R 0.11824 \
  --log10_l0 -20.908 \
  --kappa_tick 0.08333333333333333 \
  --c4 0.06 \
  --zeta 0.5 \
  --ai 1e-4 \
  --kernel exp
```

---

## Tier‑A0 (CLASS) preflight: step-by-step (background-only)

Tier‑A0 is the **paper-facing** cosmology validation in this repo:
- no likelihoods
- no perturbations required
- checks the patched CLASS background modification and cross-checks against Track‑0

### 1) Obtain CLASS sources (you supply this)
This repo does **not** vendor CLASS. Download CLASS and use the revision pinned by the paper/release notes (commonly **v3.3.4**).

### 2) Apply the EDCL patch

Patch file:
- `cosmology/patches/class_edcl.patch`
- MD5: `1a5b122e61303ea828b36fa8a27566db`

Apply:
```bash
cd /path/to/class_public
patch -p1 < /path/to/two-perspective-edcl-validation/cosmology/patches/class_edcl.patch
```

(Optional) validate patch checksum locally:
```bash
python cosmology/scripts/validate_patch.py cosmology/patches/class_edcl.patch
```

### 3) Build CLASS + `classy` Python wrapper

A common build sequence (your CLASS setup may differ):
```bash
cd /path/to/class_public
make
cd python
python setup.py build_ext --inplace
```

Sanity check:
```bash
python -c "import sys; sys.path.insert(0,'/path/to/class_public/python'); from classy import Class; print('classy import: OK')"
```

### 4) Run the smoke test (verifies EDCL parameters are accepted)

```bash
python cosmology/scripts/smoke_test_classy_edcl.py --class-path /path/to/class_public
```

### 5) Run Tier‑A0 preflight (generates paper-facing artifacts + Track‑0 cross-check)

```bash
python cosmology/scripts/preflight_tiera_background.py \
  --class-path /path/to/class_public \
  --alpha_R 0.11824 \
  --log10_l0 -20.908 \
  --kappa_tick 0.08333333333333333 \
  --c4 0.06 \
  --zeta 0.5 \
  --ai 1e-4 \
  --kernel exp
```

Outputs are written to:
- `cosmology/paper_artifacts/`

Convenience wrapper (optional):
```bash
./run_tiera_preflight.sh /path/to/class_public
```

---

## Tier‑A1 (Cobaya likelihood reproduction): current status

**Status in the current manuscript draft:** Tier‑A1 chains/Δχ²/posteriors are **not claimed as completed**; they will be released later with pinned environments and archived chains.

What this repo provides now (to avoid “assumption-based” failures later):
- YAML templates: `cosmology/cobaya/*.yaml.in`
- YAML renderer (path substitution): `cosmology/scripts/render_yamls.py`
- Component discovery (verifies installed likelihood keys): `cosmology/scripts/discover_cobaya_components.py`
- Double-counting guardrail (PantheonPlus vs SH0ES): `cosmology/scripts/check_no_doublecount_sh0es.py`
- Colab runner(s): `colab/`

Tier‑A1 requires external likelihood datasets (Planck/DESI/PantheonPlus/SH0ES) and a configured Cobaya installation. This repo does not download those datasets automatically.

---

## Sanity checks (recommended for new users)

From repo root:

1) Basic import + script compilation:
```bash
python -m py_compile tierB/*.py track0/*.py cosmology/scripts/*.py
```

2) Regenerate Track‑0 + Tier‑B artifacts:
```bash
python scripts/run_all.py
```

3) Run tests:
```bash
python -m unittest discover -s tests -v
```

---

## Troubleshooting

### `ModuleNotFoundError: classy` (Tier‑A0)
- Ensure you built the CLASS Python extension and that `--class-path` points to a CLASS root containing `python/`.
- Verify import:
  ```bash
  python -c "import sys; sys.path.insert(0,'/path/to/class_public/python'); from classy import Class; print('OK')"
  ```

### EDCL parameters rejected / “Unknown parameter: edcl_on”
- Patch likely not applied to the CLASS tree being used.
- Re-apply patch and rerun smoke test:
  ```bash
  python cosmology/scripts/smoke_test_classy_edcl.py --class-path /path/to/class_public
  ```

### Output files not where you expect
- Tier‑B + Track‑0 write to `paper_artifacts/` by default.
- Tier‑A0 writes to `cosmology/paper_artifacts/` by default.
- Many scripts accept `--outdir` (see `--help`).

---

## Citation

If you use this repository (code or figures) in academic work, please cite it using `CITATION.cff` and cite the associated manuscript.

---

## License

See the repository LICENSE (if present) or the release notes for licensing terms.
