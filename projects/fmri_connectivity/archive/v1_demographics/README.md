# Archived (protocol v1) -- not reported as biomarker results

Under protocol v1 the fMRI tasks exposed `site`, `age`, `sex` (and ABIDE `mean_fd`) as covariates
and were scored with pooled AUROC. The ADHD-200 champions used site one-hot, sex, age and
site x sex interaction terms as predictors (Eval AUROC 0.611 -> 0.813 on split 1), i.e. they
learned recruitment/demographic shortcuts rather than an imaging biomarker. ABIDE was stopped
mid-run when this was found. Protocol v2 (no demographics/motion, site for harmonization only,
within-site AUROC) replaces these runs; they are kept as evidence that the loop finds and
exploits shortcuts whenever the setup allows it.
