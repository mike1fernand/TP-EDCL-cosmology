# PantheonPlus data provenance note (required for PRD reproduction)

The manuscript states:

- PantheonPlus is included as a likelihood **without embedded SH0ES calibration** when SH0ES is also included as a separate likelihood.

This is a potential “double counting” failure mode that referees will check.

Fill in the following (copy/paste from your install logs or component metadata):

1) Source of PantheonPlus likelihood component:
   - (e.g., Cobaya external package name/version, or git hash)

2) Exact dataset files used:
   - list the filenames or paths installed by cobaya-install

3) Evidence of “no embedded SH0ES” configuration:
   - include the relevant config flag / dataset description, or a citation to the package docs

4) If uncertain:
   - run a “no-SH0ES” configuration (remove sh0es likelihood) and confirm the posterior collapses back toward ΛCDM,
     *and* that PantheonPlus alone does not enforce the SH0ES calibration.

