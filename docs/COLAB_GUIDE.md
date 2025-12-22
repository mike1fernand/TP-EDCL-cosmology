# Colab guide (cell-ready commands)

This guide is written so a referee can reproduce the Tier‑A late‑only validation in Google Colab.

Important:
- Colab runs commands from **cells**. All commands below are intended to be pasted into cells.
- Tier‑B and Track‑0 do **not** require internet; Tier‑A does (to download/build CLASS and Cobaya likelihood data).

---

## A. Clone the repo (cell)

```bash
!git clone <YOUR_GITHUB_REPO_URL_HERE>
%cd <REPO_FOLDER_NAME>
```

---

## B. (Optional) Tier‑B + Track‑0 quick checks (cells)

```bash
!python -m pip install -r requirements.txt
!python scripts/run_all_tierB.py
!python track0/run_track0_kernel_consistency.py
```

Artifacts will be written to:
- `paper_artifacts/`
- `track0/`

---

## C. Tier‑A late‑only validation (cell)

```bash
!python COLAB_TIER_A_VALIDATION.py --profile referee
```

What you should see:
- a timestamped work directory printed at the end (e.g. `Workdir: /content/edcl_tiera1_YYYYMMDD_HHMMSS`)
- a validation report path (markdown) and a summary (json)
- a bundle zip for downloading (if Colab download hook is available)

If Colab cannot auto-download (common), use the Files pane to download the produced bundle.

---

## D. Analyze an existing run (cell)

If you already have a workdir, you can re-run only the analysis/validation:

```bash
!python cosmology/scripts/analyze_chains.py --workdir /content/edcl_tiera1_YYYYMMDD_HHMMSS --profile referee
!python cosmology/scripts/validate_tiera1_lateonly_results.py --workdir /content/edcl_tiera1_YYYYMMDD_HHMMSS --profile referee
```

