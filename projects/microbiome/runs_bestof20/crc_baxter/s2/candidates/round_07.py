"""Genus-level relative abundance + CRC oral-pathogen composite features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ("fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas")


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # composite marker features, matched by taxonomy name (study-independent)
    names = [g.lower().replace("g__", "").strip("[] ") for g in G.columns]
    marker_sum = np.zeros(len(G))
    marker_n = np.zeros(len(G))
    for m in CRC_MARKERS:
        idx = [i for i, n in enumerate(names) if n.startswith(m)]
        if idx:
            a = G.iloc[:, idx].sum(axis=1).values
            marker_sum += a
            marker_n += (a > 0).astype(float)
    G = G.copy()
    G["__crc_marker_logsum"] = np.log10(marker_sum + 1e-6)
    G["__crc_marker_count"] = marker_n
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]