"""
projects/drug_response/prepare.py -- FIXED data/label/metric definition (never edited by the loop).

Task
  prism_auc : predict PRISM Repurposing secondary-screen dose-response AUC (lower = more
              sensitive) of the 200 most variable, well-covered drugs (measured in >= 70% of cell
              lines; ranked by AUC standard deviation) from baseline CCLE gene expression, on
              HELD-OUT cell lines. Splits are grouped by cell line and stratified by lineage.
Metric: mean over drugs of the Pearson correlation between predicted and observed AUC across the
held-out cell lines (per-drug correlation, the field's standard; pooled correlation is inflated).
items = DataFrame (cell lines x genes) of log2(TPM+1) expression; y = DataFrame (cell lines x drugs)
of AUC with NaN where not measured; predictions = one row (vector over the same drugs) per line.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("JAE_DATA", "/data")) / "drug_response"
TASKS = json.loads((Path(__file__).parent / "project.json").read_text())["tasks"]
N_DRUGS, MIN_COVERAGE = 200, 0.70


def load(task):
    if task != "prism_auc":
        raise KeyError(task)
    X = pd.read_parquet(DATA / "expression.parquet")
    Y = pd.read_parquet(DATA / "prism_auc.parquet")
    drugs = pd.read_parquet(DATA / "drugs.parquet")
    models = pd.read_parquet(DATA / "models.parquet")
    cov = Y.notna().mean()
    sd = Y.loc[:, cov >= MIN_COVERAGE].std()
    keep = sd.sort_values(ascending=False).index[:N_DRUGS]
    Y = Y[keep]
    lineage = models["OncotreeLineage"].reindex(X.index).fillna("Unknown")
    common = lineage.value_counts()
    strata = lineage.where(lineage.map(common) >= 8, "Other").values
    meta = models.reindex(X.index)[["OncotreeLineage", "OncotreePrimaryDisease"]]
    return {"items": X, "y": Y, "groups": X.index.astype(str).values, "strata": strata, "meta": meta,
            "ctx": {"drugs": drugs.loc[keep].reset_index(), "drug_ids": list(keep),
                    "genesets_dir": str(DATA / "genesets"),
                    "genesets": {"hallmark": "hallmark.gmt", "reactome": "reactome.gmt",
                                 "string_ppi": "string_v12.5_ge700.parquet (columns a, b, score)"}},
            "external": {}}


def score(y_true, y_pred, task):
    Yt = np.asarray(y_true, dtype=float)
    Yp = np.vstack([np.asarray(p, dtype=float) for p in y_pred])
    rs = []
    for j in range(Yt.shape[1]):
        m = np.isfinite(Yt[:, j]) & np.isfinite(Yp[:, j])
        if m.sum() >= 10 and np.std(Yp[m, j]) > 0 and np.std(Yt[m, j]) > 0:
            rs.append(np.corrcoef(Yt[m, j], Yp[m, j])[0, 1])
        else:
            rs.append(0.0)                                  # constant / missing prediction = no skill
    return float(np.mean(rs))
