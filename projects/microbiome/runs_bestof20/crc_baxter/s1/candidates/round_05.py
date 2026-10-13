"""Genus-level relative abundance + composite CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts repeatedly enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]
# Butyrate-producing commensals typically depleted in CRC
COMMENSALS = ["faecalibacterium", "roseburia", "lachnospira", "eubacterium", "coprococcus"]


def _genus_name(col):
    parts = col.split(";")
    return parts[5] if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus_name(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    lname = pd.Index([g.lower().replace("g__", "").strip("[]") for g in G.columns])
    mark = G.loc[:, lname.isin(CRC_MARKERS)]
    comm = G.loc[:, lname.isin(COMMENSALS)]
    eps = 1e-5
    msum = mark.sum(axis=1) if mark.shape[1] else pd.Series(0.0, index=G.index)
    csum = comm.sum(axis=1) if comm.shape[1] else pd.Series(0.0, index=G.index)
    G = G.copy()
    G["__marker_logsum"] = np.log10(msum + eps)
    G["__marker_count"] = (mark > 0).sum(axis=1) if mark.shape[1] else 0
    G["__marker_vs_commensal"] = np.log10(msum + eps) - np.log10(csum + eps)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]