"""Genus-level relative abundance + CRC oral-pathobiont marker summary features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    gnames = [g.lower().replace("g__", "").strip() for g in G.columns]
    mcols = [G.columns[i] for i, g in enumerate(gnames) if g in CRC_MARKERS]
    if mcols:
        M = G[mcols]
        msum = M.sum(axis=1).values
        mcount = (M > 0).sum(axis=1).values
    else:
        msum = np.zeros(len(G))
        mcount = np.zeros(len(G))
    G = G.copy()
    G["marker_logsum"] = np.log10(msum + 1e-6)
    G["marker_count"] = mcount.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]