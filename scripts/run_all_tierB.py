"""
Generate all Tier-B figures into paper_artifacts/.

This script is intentionally minimal and deterministic.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tierB import sim_theorem31_wavepacket_validation as sim31
from tierB import sim_theorem31_local_speed_validation as sim31_local
from tierB import sim_theorem31_realworld_validation as sim31_real
from tierB import sim_interface_scattering_invariance as simint
from tierB import sim_spectral_mapping as simspec
from tierB import sim_lr_cone_rescaling as simlr


def main() -> None:
    art = ROOT / "paper_artifacts"
    art.mkdir(parents=True, exist_ok=True)

    print("[run_all] B1: Theorem wavepacket validation")
    sim31.run_single_and_sweep()

    print("[run_all] B1b: Theorem 3.1 local-speed validation (small-k, adiabatic)")
    sim31_local.run_validation(sim31_local.Params(), make_plots=True)

    print("[run_all] B2: Interface scattering invariance")
    d2 = simint.run(simint.Params())
    simint.make_figure(d2, art / "fig_interface_scattering_invariance.png")

    print("[run_all] B3: Spectral mapping")
    d3 = simspec.run(simspec.Params())
    simspec.make_figure(d3, art / "fig_spectral_mapping.png")

    print("[run_all] B4: LR cone/front rescaling")
    d4 = simlr.run(simlr.Params())
    simlr.make_figure(d4["results"], art / "fig_lr_cone_rescaling.png")


    print("[run_all] B1c: Theorem 3.1 real-world validation")
    levels, rep = sim31_real.run_validation(sim31_real.Params(), make_plots=True)
    # real-world script writes its own fig into paper_artifacts; copy/rename for consistency
    src = sim31_real.ART_DIR / "fig_theorem31_realworld_validation.png"
    dst = art / "fig_theorem31_realworld_validation.png"
    if src.exists():
        dst.write_bytes(src.read_bytes())

    print(f"[run_all] Done. Figures in {art}")




if __name__ == "__main__":
    main()
