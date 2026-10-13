"""
projects/proteomics_ad/prepare.py -- FIXED data/label/metric definition (never edited by the loop).

Tasks
  plasma_ad : Alzheimer's disease (53) vs cognitively normal (36), PLASMA DIA proteomics
              (PXD065835, 4,925 protein groups, two acquisition batches).
  csf_ad    : clinical AD (49) vs clinical non-AD (88), CSF DIA proteomics (Bader et al. 2020,
              PXD016278, ~1,500 protein groups, 4 cohorts/sites). Clinical diagnosis is used as
              the label because the biochemical label is defined from CSF tau/amyloid.
items = DataFrame (samples x proteins) of RAW protein-group intensities (NaN = not quantified);
columns are 'GENE|ProteinGroup'.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))
from metrics import within_site_auroc  # noqa: E402

DATA = Path(os.environ.get("JAE_DATA", "/data")) / "proteomics_ad"
TASKS = json.loads((Path(__file__).parent / "project.json").read_text())["tasks"]


def load(task):
    if task == "plasma_ad":
        X = pd.read_parquet(DATA / "plasma.X.parquet")
        m = pd.read_parquet(DATA / "plasma.meta.parquet")
        y = (m["group"] == "AD").astype(int).values
        meta = m[["batch"]].copy()
        strata = np.array([f"{a}|{b}" for a, b in zip(y, m["batch"])])
    elif task == "csf_ad":
        X = pd.read_parquet(DATA / "csf.X.parquet")
        m = pd.read_parquet(DATA / "csf.meta.parquet")
        keep = m["clinical"].isin(["AD", "non-AD"]).values
        X, m = X.loc[keep], m.loc[keep]
        y = (m["clinical"] == "AD").astype(int).values
        meta = m[["site"]].copy()                          # v2: no age/sex; site for harmonization
        strata = np.array([f"{a}|{b}" for a, b in zip(y, m["site"])])
    else:
        raise KeyError(task)
    return {"items": X, "y": y, "groups": X.index.astype(str).values, "strata": strata,
            "meta": meta, "sites": meta["site"].values if task == "csf_ad" else None,
            "ctx": {"matrix": "raw protein-group intensities, NaN = missing",
                                  "column_format": "GENE|ProteinGroup"}}


def score(y_true, y_pred, task, sites=None):
    """Protocol v2: csf_ad is scored WITHIN cohort/site; plasma_ad (one site) pooled."""
    if sites is None:
        return float(roc_auc_score(np.asarray(y_true), np.asarray(y_pred, dtype=float)))
    return within_site_auroc(y_true, y_pred, sites)
