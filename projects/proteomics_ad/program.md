# Project: proteomics (mass spectrometry) -> Alzheimer's disease

## Data you receive
- `items`: pandas DataFrame, rows = samples (people), columns = protein groups named
  `GENE|ProteinGroup` (e.g. `APOE|P02649`; gene can be empty or `A;B` for ambiguous groups).
  Values are RAW (linear-scale) DIA intensities; NaN = not quantified (missingness is often
  abundance-dependent, i.e. not at random).
- `ctx["meta"]` / `meta_train` / `meta_test`: plasma -> `batch`; CSF -> `site` (four cohorts).
  Use them for harmonization only; no age/sex are given. CSF is scored WITHIN site (AUROC per
  cohort, size-weighted), plasma pooled. Metric: AUROC.
- Predictions: one float per sample, higher = more likely Alzheimer's.

## Open questions worth exploring
Log-transform and normalization (median / quantile / reference-protein), missing-value handling
(MinProb / half-min / kNN / missingness indicators), batch and site effects, feature selection
stable under small n, prior biology (APOE, complement cascade C1Q/C3/C4/CFH, clusterin CLU,
neurofilament NEFL, YWHAG, SMOC1, SPP1, chitinases CHI3L1, GFAP, synaptic proteins NPTX2/VGF in
CSF, inflammation and lipid-transport proteins in plasma, amyloid/tau-related proteins),
protein ratios, pathway/module scores, sparse or Bayesian linear models.
n is small (~90-140): prefer strong regularization and nested CV for any tuning.
