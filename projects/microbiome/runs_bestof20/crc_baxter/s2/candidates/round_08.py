"""Genus-level relative abundance + aggregated CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T          # samples x genera
    gl = [str(g).lower().replace("g__", "").strip("[] ") for g in G.columns]
    mcols = [c for c, g in zip(G.columns, gl) if any(g.startswith(m) for m in CRC_MARKERS)]
    if mcols:
        M = G[mcols]
        G["__marker_sum"] = np.log10(M.sum(axis=1).values + 1e-6)
        G["__marker_npresent"] = (M > 0).sum(axis=1).values.astype(float)
    else:
        G["__marker_sum"] = np.log10(1e-6)
        G["__marker_npresent"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    if "__marker_sum" in cols:                              # missing-marker default for absent cols
        j = list(cols).index("__marker_sum")
        miss = ~F_test.columns.isin(["__marker_sum"]).any()
        if miss:
            Xte[:, j] = np.log10(1e-6)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]