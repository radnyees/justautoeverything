"""Genus-level relative abundance + composite CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts repeatedly enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_ORAL_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # composite marker features: pooled abundance and number of distinct markers detected
    gl = [g.lower().replace("g__", "").strip("[]") for g in G.columns]
    marker_cols = [G.columns[i] for i, g in enumerate(gl) if g in CRC_ORAL_MARKERS]
    if marker_cols:
        M = G[marker_cols]
        msum = M.sum(axis=1).values
        mcount = (M > 1e-4).sum(axis=1).values.astype(float)
    else:
        msum = np.zeros(len(G))
        mcount = np.zeros(len(G))
    G = G.copy()
    G["__oral_marker_logsum"] = np.log10(msum + 1e-6)
    G["__oral_marker_count"] = mcount
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]