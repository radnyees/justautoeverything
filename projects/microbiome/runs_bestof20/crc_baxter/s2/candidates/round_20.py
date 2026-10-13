"""Genus-level relative abundance + aggregate CRC oral-pathogen marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts repeatedly enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]


def _genus(c):
    parts = c.split(";")
    g = parts[5].strip() if len(parts) > 5 else "g__"
    return g


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    clean = [g.lower().replace("g__", "").strip("[]'\" ") for g in G.columns]
    is_marker = np.array([any(clean[i] == m for m in CRC_MARKERS) for i in range(len(clean))])
    M = G.loc[:, is_marker]
    if M.shape[1] > 0:
        msum = M.sum(axis=1).values
        mcount = (M > 0).sum(axis=1).values
    else:
        msum = np.zeros(len(G))
        mcount = np.zeros(len(G))
    G = G.copy()
    G["__marker_logsum"] = np.log10(msum + 1e-5)
    G["__marker_count"] = mcount.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]