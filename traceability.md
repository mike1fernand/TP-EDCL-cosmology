# Traceability ledger (referee‑oriented)

This file maps **paper claims** → **numerical evidence** in this repo.

Format:
- Claim (paper language)
- Evidence artifact(s)
- Diagnostic(s)
- Acceptance criteria (tolerance)
- Convergence / integrity check(s)

---

## Track‑0 — Kernel-only consistency (no CLASS/Cobaya)

### T0.1 — Kernel normalization reproduces the quoted f_norm

**Paper claim:** In Sec. meanfield‑cosmo, for ζ=0.5 and a_i=10^-4, numerical quadrature yields **f_norm = 0.7542**.

**Evidence:**
- `track0/kernel.py` (normalization rule)
- Test: `tests/test_track0_kernel.py::test_fnorm_reproduces_paper_number`

**Diagnostic:** computed `f_norm` under the explicit normalization rule.

**Acceptance:** |f_norm − 0.7542| ≤ 1e-4.

---

### T0.2 — High‑z safety consistency check (kernel vs stated saturation claim)

**Paper claim:** δ(z→∞)→0 and |H_TP/H_GR − 1| ≤ 0.2% by z~2.

**Evidence artifact:**
- `paper_artifacts/track0/fig_kernel_consistency.png`
- `paper_artifacts/track0/kernel_consistency_report.txt`
- Tests:
  - `tests/test_track0_kernel.py::test_highz_safety_claim_variant`
  - `tests/test_track0_kernel.py::test_highz_safety_equation_variant_fails`

**Diagnostic:** Under the Track‑0 mapping H_ratio = 1 + δ(z), compute the earliest z such that |ratio−1| ≤ 0.2% for all higher z.

**Acceptance:**
- For the claim‑consistent kernel variant: z_safe ≤ 2.5.
- For the equation‑as‑written kernel variant: z_safe does **not** exist on z∈[0,1100], documenting the inconsistency.

**Referee note:** This is intentionally designed to force the manuscript to resolve the kernel definition ambiguity.

---

## Tier‑B — Discrete‑lattice demonstrations (pure Python)

### B1 — Matched calibration constant speed (Theorem‑level criterion)

**Claim:** In the controlled regime, constant observed speed occurs iff F[n] matches the kinetic envelope (e.g., F ∝ J).

**Evidence:**
- Script: `tierB/sim_theorem31_wavepacket_validation.py`
- Figure: `paper_artifacts/fig_theorem31_wavepacket_validation.png`
- Test: `tests/test_theorem31_wavepacket.py`

**Diagnostics:**
- ε_v := std(v_obs)/mean(v_obs) in an analysis window.
- Mapping identity: v_obs ≈ (d⟨x⟩/dτ)/⟨F⟩.
- Integrity: norm drift and boundary leakage.

**Acceptance:**
- Matched: ε_v < 1e-3.
- Mismatched: ε_v > 1e-2.
- Norm drift < 1e-8.
- Boundary leakage < 1e-6.

---

### B1b — Theorem \ref{thm:localc-1D} local‑speed law (small‑k, adiabatic recipe)

**Claim:** In the adiabatic + narrow‑band regime (k0 a \ll 1, packet narrow in k),
the leading‑order local law

\[ v_{\rm obs}(x) \simeq \frac{2 a k_0 J(x)}{F(x)} + \mathcal{O}(\varepsilon_{\rm ad}+\varepsilon_{\rm disp}) \]

implies:
  - Matched calibration F=alpha_F J \Rightarrow v_obs is independent of position (constant speed).
  - Mismatched calibration F\ne alpha_F J \Rightarrow v_obs varies with position at leading order.

**Evidence:**
- Script: `tierB/sim_theorem31_local_speed_validation.py`
- Figure: `paper_artifacts/fig_theorem31_local_speed_validation.png`
- Report: `paper_artifacts/theorem31_local_speed_report.txt`
- Test: `tests/test_theorem31_local_speed.py`

**Diagnostics:**
- Spatial constancy: ε_space := std_x0( v̄_obs(x0) ) / mean_x0( v̄_obs(x0) ), where v̄_obs is a
  regression slope from ⟨x⟩ vs observer time t.
- Predictor: A_pred := std_x0(J/F) / mean_x0(J/F) (expected mismatch plateau).
- Regime parameters: ε_total = ε_ad + ε_disp with ε_ad from |(a ∂x J)/J|·σ and ε_disp from max((k0 a)^2, (Δk a)^2).
- Clock/worldline check: compare centroid‑clock t_cm to expectation‑clock t_exp (Lemma‑style sanity check).

**Acceptance (default tolerances):**
- Matched: ε_space \lesssim 1% at the finest (smallest ε_total) sweep point.
- Mismatch: ε_space is O(A_pred) (within an order‑unity factor) and does not collapse as ε_total decreases.

**Convergence / integrity checks:**
- Sweep over scale factors s (L increases, k0 decreases, σ increases sub‑linearly) so that ε_total decreases.
- Norm drift remains at floating‑point level (unitary Trotter update; no CAP in the local‑speed run).

---

### B2 — Interface scattering invariance

**Claim:** Reflection/transmission and flux continuity are invariant under time reparameterization.

**Evidence:**
- Script: `tierB/sim_interface_scattering_invariance.py`
- Figure: `paper_artifacts/fig_interface_scattering_invariance.png`
- Test: `tests/test_interface_scattering.py`

**Diagnostics:**
- R_prob, T_prob from asymptotic probability mass.
- Flux integrals compared in τ vs t: T_τ = ∫ j_τ dτ and T_t = ∫ j_t dt with j_t=j_τ/(dt/dτ).
- Unitarity: norm drift.

**Acceptance:**
- |T_τ − T_t| < 5e-4.
- R+T+P_mid ≈ 1 within 5e-3.
- Norm drift < 1e-8.

---

### B3 — Spectral Jacobian mapping

**Claim:** For constant calibration, spectra map by Jacobian scaling and integrated spectral measure is invariant.

**Evidence:**
- Script: `tierB/sim_spectral_mapping.py`
- Figure: `paper_artifacts/fig_spectral_mapping.png`
- Test: `tests/test_spectral_mapping.py`

**Diagnostics:**
- Constant F: L1 error between FFT PSD and Jacobian‑mapped PSD.
- Invariance: I = ∫ S(ω) dω.

**Acceptance:**
- Constant F: L1 < 1e-9 and |I_t/I_τ − 1| < 1e-9.
- Slowly varying F: |I_t/I_τ − 1| < 5e-3 and deviation increases with modulation.

---

### B4 — Cone/front rescaling (LR‑style demonstration)

**Claim:** Calibration rescales the observer‑time cone/front speed approximately by 1/F (constant F).

**Evidence:**
- Script: `tierB/sim_lr_cone_rescaling.py`
- Figure: `paper_artifacts/fig_lr_cone_rescaling.png`
- Test: `tests/test_lr_cone_rescaling.py`

**Diagnostics:** fitted front slope in τ vs fitted front slope in t.

**Acceptance:** for constant F=2, slope_t(F=2)/slope_t(F=1) within ±0.05 of 0.5.

---

## Tier‑A — Cosmology pipeline (CLASS + Cobaya)

**Scope:** This repo provides **scaffolding and smoke tests**; full reproduction requires:
- patched CLASS v3.4 (`cosmology/patches/class_edcl.patch`)
- Cobaya v3.6 with installed likelihood datasets

### A0 — No‑assumptions likelihood discovery

**Claim:** Likelihood keys used in YAML must match installed Cobaya registry.

**Evidence:**
- Script: `cosmology/scripts/discover_cobaya_components.py`
- Output: `cosmology/paper_artifacts/cobaya_components.txt`

**Acceptance:** the likelihood keys listed in the manuscript YAML snippet are present (or explicit remapping is documented).

### A1 — Patched CLASS wiring

**Claim:** EDCL enters background only; LCDM limit is recovered when EDCL is off.

**Evidence:**
- `cosmology/scripts/smoke_test_classy_edcl.py`
- `cosmology/scripts/make_fig_hubble_ratio_from_class.py`

**Acceptance:**
- EDCL parameters are accepted by CLASS (no “unknown parameter”).
- With EDCL off: baseline background computes.
- With EDCL on: background computes and ratio differs nontrivially.

(Full acceptance thresholds for Δχ²/ΔlnZ and “no‑SH0ES collapse” require chain outputs.)



## B1c Real-World Theorem 3.1 validation
- Script: `tierB/sim_theorem31_realworld_validation.py`
- Artifact: `paper_artifacts/fig_theorem31_realworld_validation.png`
- Validates: Thm. localc-1D matched calibration criterion + lemma residual + regime gates.
