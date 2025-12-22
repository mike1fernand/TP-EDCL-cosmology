# Data availability and artifacts

This repository is structured so that heavy outputs (especially Tier‑A MCMC chains) do not have to live in git history.

## What is in git
- code + configs + patches
- small deterministic artifacts (Tier‑B figures, Track‑0 figure, Tier‑A preflight plots)

## What should be published as Release assets
- Tier‑A chain outputs (`chains/`)
- Tier‑A timestamped work directories (`edcl_tiera1_YYYYMMDD_HHMMSS/`)
- Tier‑A bundle zip(s) produced by the runner

See `docs/GITHUB_PUBLISH_CHECKLIST.md`.

