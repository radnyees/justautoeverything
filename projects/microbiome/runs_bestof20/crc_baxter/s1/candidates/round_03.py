"""Genus-level relative abundance + CRC oral-pathogen consortium features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

CRC_MARKERS = ["g__Fusobacterium", "g__Peptostreptococcus", "g__Porphyromonas", "g__Parvimonas"]


def _genus(c):
    parts = [p.strip() for p in c.split(";")]
    return parts[5] if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                       # samples x genera
    M = G.reindex(columns=CRC_MARKERS, fill_value=0.0)
    G["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
    G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    G["marker_logmax"] = np.log10(M.max(axis=1) + 1e-6)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]