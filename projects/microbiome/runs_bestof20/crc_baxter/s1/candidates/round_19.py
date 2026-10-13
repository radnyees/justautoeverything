"""Genus-level relative abundance + CRC oral-pathogen marker scores + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Peptostreptococcus", "g__Porphyromonas"]


def _genus(c):
    parts = c.split(";")
    g = parts[5].strip() if len(parts) > 5 else "g__"
    return g.replace("[", "").replace("]", "")


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                       # samples x genera
    present = [m for m in CRC_MARKERS if m in G.columns]
    M = G[present] if present else pd.DataFrame(0.0, index=G.index, columns=["none"])
    G["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
    G["marker_ndetected"] = (M > 0).sum(axis=1).astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]