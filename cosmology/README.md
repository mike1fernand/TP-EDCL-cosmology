# Cosmology reproduction harness (Tier‑A scaffolding)

This folder supports a PRD/JCAP‑referee‑grade, **no‑assumptions** reproduction workflow for the paper’s
TP/EDCL cosmology claims.

It is intentionally split into:

- **Track‑0 (math layer)**: kernel‑only background mapping checks (runs anywhere; no external data).
- **Tier‑B (formalism sanity checks)**: lattice demos (already runnable locally).
- **Tier‑A (cosmology fits)**: requires **patched CLASS (tested against upstream tag v3.3.4)** + **Cobaya v3.6** +
  external likelihood datasets.

This repo **does not** automatically download Planck/DESI/PantheonPlus/H0 datasets. A referee will not accept guessed paths
or guessed likelihood names; you must install and verify components in the target environment first.

## Critical conventions (avoid common hard failures)

1) **CLASS EDCL toggle is a string**
   - The patch reads `edcl_on` as a string and checks its first character.
   - Use:
     - `edcl_on: 'yes'`  or  `edcl_on: 'no'`
   - Avoid booleans (`true/false`) to prevent ambiguity across wrappers.

2) **Do not pass EDCL‑only CLASS parameters when EDCL is off**
   - If `edcl_on: 'no'`, do **not** include `alpha_R`, `kappa_tick`, `c4`, `log10_l0`, `edcl_kernel`, `edcl_zeta`, `edcl_ai`.
   - CLASS will fail with: `Class did not read input parameter(s): ...`
   - This was a root cause of the “null likelihood / could not find random point” failures.

3) **Cobaya likelihood keys are install‑dependent**
   - The late‑time keys observed in a working Cobaya 3.6 environment (and suggested by Cobaya itself) are:
     - `bao.desi_dr2.desi_bao_all`
     - `sn.pantheonplus`
     - `H0.riess2020`
   - If your install differs, do **not** guess: run `cobaya-install <yaml> -p <packages_dir>` and follow the suggestions printed.

4) **DESI “full‑shape” likelihood is not enabled by default**
   - Its key and required data can vary across installs. Enable it only if `cobaya-install` recognizes it in your environment.

## Templates

Cobaya YAML templates live in `cosmology/cobaya/*.yaml.in`.

- `lcdm_lateonly.yaml.in`: BAO + PantheonPlus + H0 prior (fast smoke test)
- `edcl_cosmo_lateonly.yaml.in`: same, with EDCL enabled
- `edcl_cosmo_lateonly_no_sh0es.yaml.in`: late‑time EDCL (BAO+SN) without an explicit H0 likelihood
- `lcdm_full.yaml.in`: Planck 2018 + BAO + PantheonPlus + H0 prior (requires clik)
- `edcl_cosmo_full.yaml.in`: same, with EDCL enabled
- `edcl_cosmo_no_sh0es.yaml.in`: full stack (Planck+BAO+SN) without an explicit H0 likelihood

## Rendering YAMLs without assumptions

Render templates by substituting the CLASS path and per‑run output directory:

```bash
python cosmology/scripts/render_yamls.py --class-path /path/to/class_public --out-root chains
```


## Tier-A1 late-time suite runner (automated, referee-safe)

For the late-time Phase-1 EDCL validation suite (BAO+SN(+H0)), use:

```bash
python3 cosmology/scripts/run_tiera1_lateonly_suite.py --profile iterate   # (alias: smoke)
```

For a referee-grade run (larger chains / more stable quantiles):

```bash
python3 cosmology/scripts/run_tiera1_lateonly_suite.py --profile referee
```

### What the runner does
The suite runner performs, in order:

1. Install dependencies (apt + pip) unless skipped.
2. Clone CLASS, select a deterministic tag (prefers `v3.3.4`), apply `cosmology/patches/class_edcl.patch`, build `classy`.
3. Run Tier-A0 preflight scripts (patched background sanity).
4. Render late-only YAMLs from templates (`cosmology/scripts/render_yamls.py`).
5. Run the SH0ES/H0 double-count guard.
6. Run `cobaya-install` from the rendered YAMLs (no hand-typed likelihood guessing).
7. Run `cobaya-run --test` (fast init-only check), then full `cobaya-run`.
8. Run the validator and include its reports in the final bundle.

### Validation (pre-registered thresholds)
Validation criteria are defined by:

- `cosmology/paper_artifacts/validation_spec.md` (human-readable rationale)
- `cosmology/config/validation_config.yaml` (authoritative numerical thresholds)

Validator script:

```bash
python3 cosmology/scripts/validate_tiera1_lateonly_results.py --workdir <WORKDIR> --profile smoke
```

Outputs:
- `<WORKDIR>/results_summary.json`
- `<WORKDIR>/results_report.md`

These are included in `<WORKDIR>/bundle_edcl_tiera1.zip`.
