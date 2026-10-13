"""Genus-level relative abundance + CRC oral-pathogen marker summary features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathobiont genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["fusobacterium", "porphyromonas", "parvimonas", "peptostreptococcus", "gemella"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    gl = [g.lower().replace("g__", "").strip("[]' ") for g in G.columns]
    eps = 1e-5
    logs, pres = [], []
    for m in CRC_MARKERS:
        idx = [i for i, g in enumerate(gl) if g == m]
        a = G.iloc[:, idx].sum(axis=1).values if idx else np.zeros(len(G))
        logs.append(np.log10(a + eps) - np.log10(eps))      # >= 0, 0 when absent
        pres.append((a > 0).astype(float))
    G = G.copy()
    G["marker_logsum"] = np.sum(logs, axis=0)
    G["marker_count"] = np.sum(pres, axis=0)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]