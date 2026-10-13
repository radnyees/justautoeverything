"""Genus-level relative abundance + CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Parvimonas", "Porphyromonas", "Peptostreptococcus"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    marker_cols = [g for g in G.columns if g.replace("g__", "") in CRC_MARKERS]
    if marker_cols:
        M = G[marker_cols]
        G["crc_marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
        G["crc_marker_ndetected"] = (M > 1e-4).sum(axis=1).astype(float)
    else:
        G["crc_marker_logsum"] = np.log10(1e-6)
        G["crc_marker_ndetected"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]