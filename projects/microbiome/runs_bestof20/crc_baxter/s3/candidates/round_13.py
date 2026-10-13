"""Genus-level relative abundance + pooled CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts repeatedly enriched in CRC stool (Baxter 2016, Zackular 2014, Wirbel 2019)
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # pooled marker features (taxonomy-name based -> transferable across studies)
    names = [g.lower().replace("g__", "") for g in G.columns]
    marker_cols = [G.columns[i] for i, n in enumerate(names) if n in CRC_MARKERS]
    M = G[marker_cols] if marker_cols else pd.DataFrame(np.zeros((len(G), 1)), index=G.index)
    pc = 1e-5
    G = G.copy()
    G["__marker_logsum"] = np.log10(M.values + pc).sum(axis=1) if marker_cols else 0.0
    G["__marker_ndetected"] = (M.values > 0).sum(axis=1)
    G["__marker_total"] = M.values.sum(axis=1)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]