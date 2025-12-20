# Cosmology reproduction harness (Tier‑A scaffolding)

This folder is designed for a PRD‑referee‑grade, **no‑assumptions** reproduction workflow of the paper’s
TP/EDCL cosmology claims.

It is intentionally split into:

- **Track‑0 (math layer)**: kernel‑only plots under `track0/` (runs anywhere; no external data).
- **Tier‑B (formalism sanity checks)**: lattice demos under `tierB/` (already runnable locally).
- **Tier‑A (cosmology fits)**: requires **patched CLASS (tested against upstream tag v3.3.4)** + **Cobaya v3.6** + external likelihood datasets.

This repo **does not download** Planck/DESI/PantheonPlus/SH0ES data automatically. A referee will not accept guessed paths
or guessed likelihood names; you must install and verify components on the target machine (or Colab) first.

## What you need to supply / verify (no assumptions)

1. **CLASS source and version**
   - The manuscript text references **CLASS v3.4**; upstream `class_public` currently provides tags through **v3.3.4**.
   - The provided `class_edcl.patch` applies cleanly to **v3.3.4** (apply with `patch -p1`).
   - Clone that version and build `classy` (python wrapper).

2. **The EDCL patch**
   - The reproduction pack ships a concrete `class_edcl.patch` at: `cosmology/patches/class_edcl.patch`.
   - It implements the TP/EDCL background-only Hubble rescaling used by Track‑0 and the paper’s H(z) ratio plots.

3. **Cobaya + installed likelihoods**
   - The manuscript uses Cobaya **v3.6** and the following likelihood keys:
     - `planck_2018_highl_plik.TTTEEE`
     - `planck_2018_lowl.TT`
     - `planck_2018_lowl.EE`
     - `planck_2018_lensing.clik`
     - `desi_dr2.bao`
     - `desi_dr2.full_shape`
     - `pantheonplus.PantheonPlus`
     - `sh0es.SH0ES`

   Run component discovery first:
   ```bash
   python cosmology/scripts/discover_cobaya_components.py
   ```
   and confirm these keys exist in your installation (or adjust by installing the correct components).

4. **PantheonPlus “no embedded SH0ES”**
   - The manuscript states PantheonPlus is used *without* embedded SH0ES calibration when SH0ES is included separately.
   - This is a data‑provenance issue, not just a YAML issue. Document the exact PantheonPlus variant you installed in
     `cosmology/data_provenance/pantheonplus_note.md`.

## Minimal local smoke tests (before any MCMC)

1) Verify patched CLASS accepts EDCL parameters:
```bash
python cosmology/scripts/smoke_test_classy_edcl.py --class-path /path/to/class
```

2) Plot H(z) ratio directly from CLASS (background only):
```bash
python cosmology/scripts/make_fig_hubble_ratio_from_class.py --class-path /path/to/class --alpha_R 0.118 --log10_l0 -20.91
```

These smoke tests do not require any external likelihood data.

## Rendering YAMLs without guessing paths

Templates live in `cosmology/cobaya/*.yaml.in`. Render them as:

```bash
python cosmology/scripts/render_yamls.py --class-path /path/to/class --out-root chains
```

This writes `.yaml` files next to the templates, with concrete `path:` and `output:` fields.

## Running chains (once Cobaya + likelihood data are installed)

Example:
```bash
cobaya-run cosmology/cobaya/lcdm_full.yaml
cobaya-run cosmology/cobaya/edcl_cosmo_full.yaml
cobaya-run cosmology/cobaya/edcl_cosmo_no_sh0es.yaml
```

For Colab, use the notebook in `colab/EDCL_TierA_Cobaya_CLASS.ipynb`.

## Referee acceptance tests (Tier‑A)

The Tier‑A tests in this repo are designed to be run **after** you have chain outputs and/or background tables.
They are not executed by default in CI because they require external datasets.

