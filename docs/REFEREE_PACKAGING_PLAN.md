# Referee-oriented packaging plan (all tiers)

This repository contains a **tiered validation suite** intended to let a referee (or any third party) reproduce:
- the **Tier‑B** formalism/kinematics simulations (pure Python; fast),
- the **Track‑0** kernel/FRW consistency checks (pure Python; fast),
- the **Tier‑A** late‑only cosmology validation (CLASS + Cobaya MCMC; heavier).

The immediate goal of this plan is to ensure the repo is:
1. **Runnable** from a clean checkout with minimal ambiguity,
2. **Auditable** (clear mapping from paper claims → code → artifacts),
3. **Deterministic enough** for validation (seeded where appropriate; acceptance criteria explicit),
4. **Publishable** (GitHub-ready; large artifacts handled as release assets or LFS).

---

## A. Inventory and scope control (no hidden assumptions)

1. **Enumerate tiers and entrypoints**
   - Tier‑B runner: `scripts/run_all_tierB.py`
   - Track‑0 runner: `track0/run_track0_kernel_consistency.py`
   - Tier‑A runner (Colab): `COLAB_TIER_A_VALIDATION.py`
   - Tier‑A runner (local): `RUN_TIER_A_VALIDATION.sh`
   - Tier‑A analysis/validator: `cosmology/scripts/analyze_chains.py` and `cosmology/scripts/validate_tiera1_lateonly_results.py`

2. **Enumerate inputs**
   - Tier‑B/Track‑0: no external datasets required.
   - Tier‑A: CLASS source build + Cobaya likelihood datasets (downloaded via `cobaya-install`).

3. **Enumerate outputs (canonical locations)**
   - Tier‑B figures: `paper_artifacts/`
   - Track‑0 figure: `paper_artifacts/track0/fig_kernel_consistency.png`
   - Tier‑A preflight figures: `cosmology/paper_artifacts/`
   - Tier‑A chains: produced into a run/work directory (timestamped) and/or `chains/` depending on runner settings.
   - Tier‑A validation summary tables: `cosmology/paper_artifacts/` and/or `cosmology/validation/` (see Tier‑A docs).

---

## B. “Reproducibility contract” a referee can actually follow

Create (and keep up-to-date) three documents:

1. **Validation Matrix (paper claim → evidence)**
   - File: `docs/VALIDATION_MATRIX.md`
   - Contains one row per claim with:
     - paper section/claim,
     - script(s) to run,
     - expected artifact filename(s),
     - acceptance criterion and where it is checked.

2. **Execution guide**
   - Files:
     - `docs/COLAB_GUIDE.md` (cell-ready commands)
     - `docs/LOCAL_RUN_GUIDE.md` (developer/local)
   - Must specify:
     - OS/CPU assumptions,
     - Python version,
     - expected runtime ranges per tier,
     - what “success” looks like.

3. **Data availability note**
   - File: `docs/DATA_AVAILABILITY.md`
   - States what artifacts are committed vs published as release assets (or via Git LFS/Zenodo), including checksums.

---

## C. Engineering requirements (to prevent repeat failures)

1. **Path robustness**
   - All scripts that rely on repository structure must locate the repo root relative to `__file__`, not `os.getcwd()`.
   - Where appropriate, also accept `--repo-root` as an override.

2. **Single-command entrypoints**
   - Each tier must have exactly one “canonical” entrypoint.
   - A top-level convenience wrapper may exist, but should not be the only way.

3. **Explicit exit codes + logs**
   - Validation scripts must return non-zero on failure and write a human-readable report (`.md`) plus machine-readable summary (`.json`).

4. **Pinned dependencies**
   - Tier‑B/Track‑0: `requirements.txt` (pure Python).
   - Tier‑A: pinned Cobaya + explicit CLASS tag + patch hash recorded in logs.

5. **Artifact provenance**
   - For each produced figure/table:
     - script name,
     - command line,
     - git commit hash,
     - timestamp,
     - (if applicable) random seed,
     should be written into a sidecar `*.meta.json`.

---

## D. GitHub packaging strategy (what goes where)

1. **Commit to the repository**
   - All source code, templates, configs, and documentation.
   - Small, stable artifacts that are part of the paper narrative (e.g., Tier‑B figures) *may* be committed if small and deterministic.

2. **Publish as GitHub Release assets (recommended)**
   - Tier‑A chain outputs and full run bundles.
   - Any large likelihood/data downloads are *not* redistributed; scripts re-download them.

3. **If artifacts exceed GitHub limits**
   - Use Git LFS for moderately large binaries, or
   - Archive to Zenodo and cite DOI in the paper (preferred for journals).

A concrete checklist is provided in `docs/GITHUB_PUBLISH_CHECKLIST.md`.

