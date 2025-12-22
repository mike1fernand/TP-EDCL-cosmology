# Release assets (attach to GitHub Releases)

This folder contains **example heavy outputs** that should *not* be committed to git history.

Recommended use:
1. Create a GitHub Release for the paper tag (see `repo/docs/GITHUB_PUBLISH_CHECKLIST.md`).
2. Zip (or upload as-is) the contents of this folder as Release assets.

Contents may include:
- `chains/` — Cobaya/GetDist chain outputs for Tier‑A
- `edcl_tiera1_*/` — timestamped Tier‑A work directories (logs, bundles, configs)

If you re-run Tier‑A, publish the new workdir and chains similarly.

