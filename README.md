# Two‑Perspective validation suite (Tier‑B, Track‑0, Tier‑A)

This repository contains the **validation and reproducibility code** for the simulations used in the paper.

It is organized as a **tiered** suite:

- **Tier‑B (pure Python, fast):** formalism/kinematics numerical validations that generate the paper’s non‑cosmology figures.
- **Track‑0 (pure Python, fast):** kernel/consistency checks.
- **Tier‑A (CLASS + Cobaya, heavier):** late‑only cosmology validation, including MCMC chains and a referee‑oriented validator.

If you are a referee, start with:
- `docs/VALIDATION_MATRIX.md` (paper claim → script → artifact)
- `docs/COLAB_GUIDE.md` (cell-ready commands)

---

## Repository structure

- `tierB/` — Tier‑B simulation modules (pure Python)
- `scripts/run_all_tierB.py` — one-command Tier‑B runner (writes `paper_artifacts/`)
- `track0/` — Track‑0 kernel checks (writes `paper_artifacts/track0/fig_kernel_consistency.png`)
- `cosmology/` — Tier‑A cosmology (CLASS patching + Cobaya YAML templates + validation scripts)
- `docs/` — referee-facing documentation
- `traceability.md` — condensed claim-to-evidence ledger

Large Tier‑A outputs (chains/workdirs) are intended to be published as **GitHub Release assets** (see `docs/DATA_AVAILABILITY.md`).

---

## Quickstart (local, Tier‑B + Track‑0)

```bash
python -m pip install -r requirements.txt
python scripts/run_all_tierB.py
python track0/run_track0_kernel_consistency.py
```

Outputs:
- Tier‑B: `paper_artifacts/*.png`, `paper_artifacts/*.txt`
- Track‑0: `paper_artifacts/track0/fig_kernel_consistency.png`

---

## Tier‑A late‑only cosmology validation

Tier‑A is the only part that requires building CLASS and downloading Cobaya likelihood datasets.

Canonical entrypoints:
- **Colab:** `python COLAB_TIER_A_VALIDATION.py --profile referee`
- **Local:** `bash RUN_TIER_A_VALIDATION.sh`

After a run, re-run analysis/validation on a given workdir:

```bash
python cosmology/scripts/analyze_chains.py --workdir <WORKDIR> --profile referee
python cosmology/scripts/validate_tiera1_lateonly_results.py --workdir <WORKDIR> --profile referee
```

See:
- `TIER_A_COMPLETE_DOCUMENTATION.md`
- `cosmology/docs/H0_LIKELIHOOD_FIX.md`
- `docs/COLAB_GUIDE.md`

---

## What to cite / how this maps to the paper

- `docs/VALIDATION_MATRIX.md` provides the authoritative mapping.
- `traceability.md` is the short ledger version.

---

## License

See `LICENSE`.

