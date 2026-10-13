"""Genus-level relative abundance + pooled CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera repeatedly enriched in CRC stool (Micromonas = older name of Parvimonas)
CRC_MARKERS = ("fusobacterium", "porphyromonas", "parvimonas", "micromonas",
               "peptostreptococcus", "gemella")


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    gname = [g[3:].strip().lower() for g in G.columns]
    mcols = [c for c, n in zip(G.columns, gname) if n in CRC_MARKERS]
    if mcols:
        M = G[mcols]
        G["marker_sum_log"] = np.log10(M.sum(axis=1) + 1e-6)
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["marker_sum_log"] = np.log10(1e-6)
        G["marker_count"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]