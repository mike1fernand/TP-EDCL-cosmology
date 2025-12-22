# GitHub publish checklist (referee-ready)

This checklist is optimized for:
- easy audit by referees,
- minimal risk of “it works only on my machine”,
- keeping large artifacts accessible without bloating the git history.

---

## 1) Repository contents (what should be committed)

Commit:
- all code (`tierB/`, `track0/`, `cosmology/`, `scripts/`, `tests/`)
- configs and patches (`cosmology/config/`, `cosmology/patches/`)
- documentation (`README.md`, `docs/`, `traceability.md`)
- small, deterministic figures if you decide to keep them versioned (`paper_artifacts/`)

Do **not** commit:
- Cobaya external packages directory (e.g. `cobaya_packages/`)
- CLASS build directories
- large MCMC chains and run folders (publish as Release assets instead)

The `.gitignore` is already configured to discourage committing large Tier‑A outputs.

---

## 2) Create a tagged release for the paper

Recommended flow:
1. Push the code to GitHub.
2. Create an annotated tag (example): `paper-v1.0.0`
3. Create a GitHub Release from that tag.
4. Upload Tier‑A artifacts (chains + reports) as Release assets.

In this bundle you will find a `release_assets/` folder with example Tier‑A outputs that are suitable to attach to a Release.

---

## 3) Record provenance in the paper

The LaTeX source in `tp-paper-assets/` defines macros:

- `\TPReproURL` (GitHub URL)
- `\TPReproCommit` (exact commit hash for the paper)
- `\TPReproZenodo` (optional DOI)

Update these to match:
- your GitHub repo URL,
- the release tag commit hash,
- optional Zenodo DOI (if you mint one).

---

## 4) (Optional, recommended) Zenodo archival

For journals with strict data availability requirements:
- Connect your GitHub repo to Zenodo,
- Mint a DOI for the paper release,
- Cite the DOI in the paper.

---

## 5) Smoke CI (optional)

If you enable GitHub Actions, keep it lightweight:
- run unit tests for Tier‑B (fast)
- run Track‑0 script (fast)
- run Tier‑A analysis/validator against *archived chains* (no heavy MCMC in CI)

