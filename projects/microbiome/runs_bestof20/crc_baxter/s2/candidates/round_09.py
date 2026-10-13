"""Genus-level relative abundance + aggregate CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# CRC-enriched oral pathobionts (Parvimonas was formerly classified as Micromonas in older taxonomies)
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "parvimonas", "micromonas",
               "porphyromonas", "gemella"]


def _genus_name(c):
    parts = c.split(";")
    return parts[5] if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus_name(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    clean = {g: g.replace("g__", "").strip("[]").lower() for g in G.columns}
    marker_cols = [g for g in G.columns if clean[g] in CRC_MARKERS]
    M = G[marker_cols] if marker_cols else pd.DataFrame(0.0, index=G.index, columns=["_none"])
    G = G.copy()
    G["__marker_sum"] = M.sum(axis=1).values
    G["__marker_logsum"] = np.log10(G["__marker_sum"].values + 1e-5)
    G["__marker_count"] = (M > 0).sum(axis=1).values.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]