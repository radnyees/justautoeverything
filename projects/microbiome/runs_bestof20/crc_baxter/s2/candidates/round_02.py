"""Genus-level relative abundance + oral-pathobiont CRC composite features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobiont cluster consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]
DETECT = 1e-4


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera
    norm = {g: g.lower().replace("g__", "").strip("[]_ ") for g in G.columns}
    mcols = [g for g in G.columns if norm[g] in CRC_MARKERS]
    if mcols:
        M = G[mcols]
        G["X__marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
        G["X__marker_count"] = (M > DETECT).sum(axis=1).astype(float)
    else:
        G["X__marker_logsum"] = np.log10(1e-6)
        G["X__marker_count"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]