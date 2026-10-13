"""Genus-level relative abundance + composite CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts (Baxter 2016, Zeller 2014, ...)
CRC_MARKERS = ["fusobacterium", "porphyromonas", "parvimonas", "peptostreptococcus", "gemella"]


def _genus_name(col):
    parts = col.split(";")
    g = parts[5] if len(parts) > 5 else "g__"
    return g


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus_name(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    clean = {c: c.replace("g__", "").strip("[]").strip().lower() for c in G.columns}
    marker_cols = [c for c in G.columns if clean[c] in CRC_MARKERS]
    if marker_cols:
        M = G[marker_cols]
        msum = M.sum(axis=1).values
        mcount = (M > 0).sum(axis=1).values.astype(float)
    else:
        msum = np.zeros(len(G))
        mcount = np.zeros(len(G))
    G = G.copy()
    G["__marker_logsum"] = np.log10(msum + 1e-6)
    G["__marker_count"] = mcount
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]