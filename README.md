# TP-EDCL Cosmology Simulations

Code for the EDCL dark energy model in "Two-Perspective Quantum Dynamics in Discrete Spacetime" (Fernandes, 2025).

## Contents
- `class_edcl.patch` — Modified CLASS v3.4 with EDCL fluid (CPL thawing w(a))
- `edcl.yaml` — Cobaya input file (Planck PR4 + DESI DR2 + Pantheon+ + SH0ES)
- `chains/` — Example chains (or placeholder bestfit)

## Installation & Run
```bash
pip install cobaya
git clone https://github.com/lesgourg/class_public.git class_edcl
cd class_edcl
patch -p1 < ../class_edcl.patch
make
cobaya-run ../edcl.yaml
