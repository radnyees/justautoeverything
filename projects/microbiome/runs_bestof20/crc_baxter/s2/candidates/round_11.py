"""Genus-level relative abundance + aggregated CRC oral-pathobiont marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

CRC_MARKERS = ["g__Fusobacterium", "g__Peptostreptococcus", "g__Porphyromonas",
               "g__Parvimonas", "g__Gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    present = [g for g in CRC_MARKERS if g in G.columns]
    if present:
        M = G[present]
        msum = M.sum(axis=1).values
        mcount = (M > 0).sum(axis=1).values
    else:
        msum = np.zeros(len(G))
        mcount = np.zeros(len(G))
    G = G.copy()
    G["__marker_logsum"] = np.log10(msum + 1e-6)
    G["__marker_count"] = mcount.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    # missing marker summary columns cannot happen (always created), but keep log floor consistent
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]