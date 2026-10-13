"""Genus-level relative abundance + aggregated oral-pathobiont CRC marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts repeatedly enriched in CRC stool across cohorts (Baxter, Zeller, Wirbel, ...)
CRC_MARKERS = ("fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella")


def _genus(c):
    parts = c.split(";")
    return parts[5] if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    names = [g.lower().replace("g__", "").strip("[]' ") for g in G.columns]
    per_marker = []
    for m in CRC_MARKERS:
        idx = [i for i, n in enumerate(names) if n == m]
        per_marker.append(G.iloc[:, idx].sum(axis=1).values if idx else np.zeros(len(G)))
    M = np.vstack(per_marker).T                             # samples x markers
    G = G.copy()
    G["marker_log_sum"] = np.log10(M.sum(axis=1) + 1e-6)
    G["marker_n_present"] = (M > 1e-4).sum(axis=1).astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]