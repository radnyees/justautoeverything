# Project: cancer cell line drug sensitivity from gene expression (CCLE / PRISM)

## Data you receive
- `items`: pandas DataFrame, rows = cancer cell lines (DepMap ids), columns = ~19k gene symbols,
  values = log2(TPM+1) RNA-seq expression.
- `y_train`: DataFrame (lines x 200 drugs) of PRISM dose-response AUC (LOWER = more sensitive);
  NaN where a drug was not measured on a line. Predict ALL 200 drugs for every test line:
  return an array of shape (n_test, 200) in the same column order (ctx["drug_ids"]).
- `ctx["drugs"]`: DataFrame (broad_id, name, moa, target, phase, disease.area, indication) for the
  200 drugs, in column order -- use it for biology-informed features (e.g. target-gene expression).
- `ctx["genesets_dir"]` + `ctx["genesets"]`: MSigDB Hallmark and Reactome GMT files, and a STRING
  v12.5 human PPI edge table (gene symbols, combined score >= 700) for pathway / network features.
- `ctx["meta"]` / `meta_train` / `meta_test`: OncotreeLineage and OncotreePrimaryDisease.
- Metric: mean over the 200 drugs of Pearson r between predicted and observed AUC on held-out
  lines (missing observations ignored). Per-drug signal is weak for many drugs.

## Open questions worth exploring
Gene selection (variance, drug-target neighbourhoods), pathway-level aggregation (ssGSEA-like
scores), network propagation of target genes over STRING, lineage effects (a lot of drug
sensitivity is lineage-driven), multi-task models that share strength across drugs with similar
MOA/targets, low-rank / factorized models of the drug x line matrix, kernel methods, tree
ensembles per drug vs shared models, target transformations, and ensembling. Mind the time
budget: 200 drugs x 5 folds.
