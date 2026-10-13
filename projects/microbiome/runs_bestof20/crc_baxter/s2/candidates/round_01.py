"""Genus-level relative abundance + aggregated CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-origin genera consistently enriched in CRC stool (Zeller 2014, Baxter 2016, Wirbel 2019).
# Micromonas is the older name for Parvimonas in some taxonomies.
CRC_MARKERS = ["Fusobacterium", "Parvimonas", "Micromonas", "Peptostreptococcus",
               "Porphyromonas", "Gemella", "Solobacterium", "Prevotella_intermedia"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera
    names = {g: g[3:] if g.startswith("g__") else g for g in G.columns}
    mk = [g for g in G.columns if names[g] in CRC_MARKERS]
    if mk:
        M = G[mk]
        G["__marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
        G["__marker_ndetect"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["__marker_logsum"] = np.log10(1e-6)
        G["__marker_ndetect"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]