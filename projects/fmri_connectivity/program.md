# Project: resting-state fMRI functional connectivity -> autism (ABIDE) / ADHD (ADHD-200)

## Data you receive
- `items`: list of subject ids (strings). Nothing else is in `items`.
- `ctx["load_timeseries"](items, atlas, strategy)` -> list of float32 arrays, one per item,
  shape (timepoints, ROIs), already preprocessed (motion-corrected, nuisance-regressed,
  band-pass filtered). Timepoints and TR differ by site.
  - ABIDE atlases: `aal` (116 ROIs), `cc200` (~200), `cc400` (~392), `dosenbach160`, `ez`, `ho`, `tt`;
    strategies: `filt_noglobal`, `filt_global` (with global signal regression).
  - ADHD-200 atlases: `aal`, `cc200`, `cc400`, `ez`, `ho`, `tt`; strategy: `athena`.
  - `ctx["atlases"]`, `ctx["strategies"]` list what is available. A few subjects can have
    all-zero ROIs (outside the field of view) -- handle them.
- `ctx["meta"]` / `meta_train` / `meta_test`: `site` only (for harmonization). No age, sex or
  motion estimates are given. Site differences in scanners and protocols are large.
- Predictions: one float per subject, higher = more likely autism / ADHD. Metric: WITHIN-SITE
  AUROC (size-weighted mean of per-site AUROCs) -- predicting which site has more cases is useless.

## Open questions worth exploring
Which parcellation and resolution; connectivity measure (Pearson, partial correlation /
precision, Ledoit-Wolf shrinkage, tangent-space embedding of covariance matrices, coherence,
mutual information, dynamic/sliding-window connectivity); global signal regression or not;
graph-theory features (degree, clustering, efficiency, modularity, hubness); network-level
summaries (default mode, salience, attention, cerebellar); site harmonization (ComBat-style,
per-site standardization); low-frequency fluctuation (ALFF/fALFF) and
regional homogeneity proxies from the time series; ensembles across atlases; classifier choice
and regularization in the p >> n regime.
