"""Genus-level relative abundance + oral-pathobiont CRC marker summary features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobiont cluster consistently enriched in CRC stool across cohorts
# (Wirbel 2019, Thomas 2019, Baxter 2016).
CRC_MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Peptostreptococcus",
               "g__Porphyromonas", "g__Gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    present = [g for g in CRC_MARKERS if g in G.columns]
    if present:
        M = G[present]
        G["marker_sum"] = M.sum(axis=1)
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["marker_sum"] = 0.0
        G["marker_count"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]