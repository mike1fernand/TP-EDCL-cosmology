# Tier-A1 late-time Cobaya validation specification (Phase-1)

This document defines the *pre-registered, mechanical* acceptance criteria for the Tier-A1 late-time Cobaya suite
in the Two-Perspective / EDCL Phase-1 (background-only) pipeline.

**Scope (Phase-1):**
- The EDCL Phase-1 implementation is **background-only** (FRW), used as a referee-safe first-stage validation.
- In Phase-1, **only the amplitude** parameter is treated as an inferred quantity:
  - `alpha_R` (equivalently, a reparameterization such as `delta0`).
- All other EDCL microparameters are treated as **fixed model-definition / solver-definition choices** in Phase-1:
  - `edcl_kernel` fixed to `exp` (high-z suppression)
  - `edcl_zeta` fixed (default: 0.5)
  - `edcl_ai` fixed (default: 1e-4)
  - `kappa_tick` fixed (default: 1/12)
  - `log10_l0` fixed placeholder for Phase-2
  - `c4` fixed / inactive in Phase-1 FRW background-only runs

The authoritative numerical thresholds are encoded in:
- `cosmology/config/validation_config.yaml`

This file is the human-readable rationale for the same rules.

---

## Suite definition

Tier-A1 late-time suite consists of three runs:

- **Run Λ (baseline):** ΛCDM late-only (BAO + SN + H0)
- **Run A (EDCL + H0):** EDCL late-only with explicit H0 likelihood (BAO + SN + H0)
- **Run B (EDCL no H0):** EDCL late-only without explicit H0 likelihood (BAO + SN only)

Where the late-time likelihood set is determined by the YAML templates under `cosmology/cobaya/*.yaml.in`,
rendered by `cosmology/scripts/render_yamls.py`, and installed via `cobaya-install` from the rendered YAML.

---

## Validation layers

### V0 — Provenance and integrity (hard fail)

Hard-fail if any of the following are missing or inconsistent:

- `manifest.json` exists and records:
  - CLASS tag/commit
  - patch checksum
  - Cobaya version
- Rendered YAMLs exist for each run.
- `*.updated.yaml` exists for each run (ensures likelihood keys were resolved by Cobaya install, not guessed).
- LCDM vs EDCL YAML separation is respected:
  - ΛCDM YAML must not pass EDCL-only CLASS args.
  - EDCL YAML must pass `edcl_on: 'yes'` and include the required EDCL args.

### V1 — Run health (hard fail)

Hard-fail if any of the following are detected:

- runtime errors in logs matching the configured fail-pattern list (e.g., CLASS parameter read failures)
- missing chain outputs
- insufficient number of samples for the selected validation profile:

- `iterate` profile (alias: `smoke`): **WARN but proceed** if chains are short. This mode is for rapid plumbing checks and early iterations.
- `referee` profile: **hard-gate** on minimum chain size (and multi-chain output) before declaring a paper-grade artifact.

### V2 — Phase-1 behavioral claim (activation + no-H0 collapse)

Because `alpha_R >= 0` in Phase-1 YAMLs, acceptance uses

### Numerical thresholds (pre-registered; Phase-1)

These are the default Tier-A1 Phase-1 acceptance thresholds:

- **Activation (Run A = EDCL + BAO+SN+H0):**
  - PASS if `q0.50(alpha_R) >= 0.03`

- **Collapse (Run B = EDCL + BAO+SN only; no explicit H0):**
  - PASS if `q0.95(alpha_R) <= 0.03`
  - STRONG_PASS if `q0.95(alpha_R) <= 0.02`

- **Relative collapse (guards against trivial passes):**
  - PASS if `q0.95_B(alpha_R) / q0.50_A(alpha_R) <= 0.5` (≥ 2× shrinkage)

All thresholds are implemented mechanically in `cosmology/config/validation_config.yaml`.
 **quantile-based** statements.

**Activation criterion (Run A):**
- EDCL amplitude should be pulled away from 0 when explicit H0 is included.

Operational definition:
- require `q0.50(alpha_R)` in Run A to exceed a configured minimum.

**Collapse criterion (Run B):**
- With explicit H0 removed, the posterior on `alpha_R` should be consistent with collapse toward 0.

Operational definition:
- require the **95% upper limit** `q0.95(alpha_R)` in Run B to be below a configured maximum.

**Relative-collapse criterion (guards against trivial passes):**
- require that Run B is meaningfully smaller than Run A:
  - `q0.95_B(alpha_R) / q0.50_A(alpha_R) <= r_collapse`

All numerical thresholds are set in `cosmology/config/validation_config.yaml`.

### V3 — Fit-quality diagnostics (warnings, not acceptance gates)

For transparency, the validator also reports component chi² columns (if available in the chain output),
and computes warning-level deltas relative to ΛCDM.

These diagnostics are *not* Tier-A1 pass/fail gates in Phase-1.

---

## Outputs

The validator emits:

- `results_summary.json` (machine-readable)
- `results_report.md` (human-readable summary, including PASS/FAIL/WARN reasons)

The Tier-A1 suite runner bundles these into the final zip artifact.

