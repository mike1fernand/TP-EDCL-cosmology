# Cosmology (Tier‑A0 preflight and Tier‑A1 scaffolding)

This folder contains the cosmology-facing components of the **two-perspective-edcl-validation** repository.

- **Tier‑A0 (CLASS preflight):** background-only verification of the Phase‑1 \(H(z)\) ratio in **patched CLASS** (no likelihoods; no Cobaya chains).  
  This is the *paper-facing* cosmology evidence.
- **Tier‑A1 (Cobaya likelihood reproduction):** scaffolding only (templates + helper scripts). Full Tier‑A1 results will be released later with pinned environments and archived chain directories.

If you are arriving from the paper, the minimum run you should do is **Tier‑A0**.

---

## Tier‑A0: what it checks (referee gate)

Tier‑A0 is designed to answer a PRD referee’s basic questions *before* any likelihood claims:

1) Does patched CLASS **build** cleanly?
2) Does the parser accept the EDCL parameters (no “unknown parameter” failures)?
3) Does the **ΛCDM limit** hold (EDCL enabled but \(\alpha_R=0\) → exact ΛCDM background)?
4) Does the **high‑z safety** property hold for the Phase‑1 mapping?
5) Does the CLASS-produced \(H(z)\) ratio agree with the **Track‑0** kernel-only mapping (same kernel choice)?

Tier‑A0 produces both the CLASS \(H(z)\) ratio and the Track‑0 cross-check report/figure.

---

## What you need installed for Tier‑A0

You need:

- A local checkout of **CLASS** (not included in this repo)
- The CLASS Python wrapper **`classy`** built and importable
- This repo’s dependencies (`requirements.txt` at repo root)

This repo provides:
- The patch file `class_edcl.patch` at repo root
- Tier‑A0 scripts in `cosmology/scripts/`

---

## Step-by-step: running Tier‑A0 (background-only)

### Step 1 — Get CLASS

Download CLASS separately and decide which version/commit you will use (the paper/release notes should pin this).

You will end up with a folder such as:
- `/path/to/class_public/`

### Step 2 — Apply the patch

From your CLASS folder:

```bash
cd /path/to/class_public
patch -p1 < /path/to/two-perspective-edcl-validation/class_edcl.patch
```

Optional: validate that the patch file is the expected one:

```bash
python /path/to/two-perspective-edcl-validation/cosmology/scripts/validate_patch.py \
  /path/to/two-perspective-edcl-validation/class_edcl.patch
```

### Step 3 — Build CLASS and `classy`

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

### Step 4 — Run the smoke test (checks EDCL params are accepted)

From this repo root:

```bash
python cosmology/scripts/smoke_test_classy_edcl.py --class-path /path/to/class_public
```

If this fails with “unknown parameter,” your patch may not have been applied to the CLASS tree you built.

### Step 5 — Run the Tier‑A0 preflight (generates paper-facing artifacts)

From this repo root:

```bash
python cosmology/scripts/preflight_tiera_background.py \
  --class-path /path/to/class_public \
  --alpha_R 0.11824 \
  --log10_l0 -20.908 \
  --kappa_tick 0.0833333333333 \
  --c4 0.06 \
  --zeta 0.5 \
  --ai 1e-4 \
  --kernel exp
```

Outputs are written to:

- `cosmology/paper_artifacts/`

Expected output files include:
- `hubble_ratio_from_class.csv`
- `fig_hubble_ratio_from_class.png`
- `preflight_report.txt`
- `preflight_summary.json`
- `fig_track0_vs_class.png`
- `track0_vs_class_report.txt`

---

## Tier‑A1 (Cobaya) scaffolding

This repo includes helper materials for later Tier‑A1 runs:

- YAML templates (examples) and helper scripts to avoid guessing component names/paths.
- A “no assumptions” component discovery script (to confirm installed likelihood keys).
- A “double counting guardrail” script (PantheonPlus vs SH0ES separation).

Tier‑A1 requires:
- Cobaya installation
- Likelihood datasets (Planck/DESI/PantheonPlus/SH0ES), which this repo does not download automatically
- A pinned environment and archived chains (to be provided with the Tier‑A1 release)

---

## Troubleshooting (Tier‑A0)

### `ModuleNotFoundError: classy`
You did not build the CLASS Python wrapper, or you are pointing at the wrong CLASS folder.

Re-run the `classy` build and confirm:
```bash
python -c "import sys; sys.path.insert(0,'/path/to/class_public/python'); from classy import Class; print('OK')"
```

### “Unknown parameter: edcl_on” (or similar)
The patch is not applied to the CLASS tree you built (or you rebuilt without patch). Re-apply patch and rebuild.

### Outputs not found
Tier‑A0 writes to `cosmology/paper_artifacts/` unless you set an alternate output directory.
