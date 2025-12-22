# Local run guide (developer/local)

This guide is optional for referees, but useful for archiving and for reproducing runs outside Colab.

---

## Tier‑B + Track‑0 (pure Python)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/run_all_tierB.py
python track0/run_track0_kernel_consistency.py
```

---

## Tier‑A late‑only (CLASS + Cobaya)

Tier‑A requires:
- a compiler toolchain (gcc/g++, gfortran, make)
- python dev headers
- Cobaya and its likelihood data

Canonical runner:

```bash
bash RUN_TIER_A_VALIDATION.sh
```

The script prints the final work directory. Use it to run validation again if needed:

```bash
python cosmology/scripts/validate_tiera1_lateonly_results.py --workdir <WORKDIR> --profile referee
```

