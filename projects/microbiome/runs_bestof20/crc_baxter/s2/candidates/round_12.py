"""Genus-level relative abundance + pooled CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["g__Fusobacterium", "g__Peptostreptococcus", "g__Parvimonas", "g__Porphyromonas"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    present = [g for g in CRC_MARKERS if g in G.columns]
    M = G[present] if present else pd.DataFrame(0.0, index=G.index, columns=["_none"])
    G = G.copy()
    G["marker_sum"] = np.log10(M.sum(axis=1).values + 1e-6)  # pooled oral-pathogen load
    G["marker_count"] = (M > 0).sum(axis=1).values            # how many marker genera detected
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]