"""Genus-level relative abundance + composite CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts (Zeller, Baxter, Wirbel, Thomas)
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas", "gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T             # samples x genera
    names = [g.lower().replace("g__", "").strip("[]") for g in G.columns]
    marker_cols = [G.columns[i] for i, n in enumerate(names) if n in CRC_MARKERS]
    M = G[marker_cols] if marker_cols else pd.DataFrame(0.0, index=G.index, columns=["none"])
    G = G.copy()
    G["__marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)  # total oral-pathogen load
    G["__marker_count"] = (M > 0).sum(axis=1).astype(float)  # number of marker genera detected
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    F_te = F_test.reindex(columns=cols, fill_value=0)
    if "__marker_logsum" in cols:
        F_te["__marker_logsum"] = F_test["__marker_logsum"] if "__marker_logsum" in F_test else np.log10(1e-6)
    Xte = F_te.values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]