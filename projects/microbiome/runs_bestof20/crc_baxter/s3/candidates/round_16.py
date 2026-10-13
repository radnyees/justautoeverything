"""Genus-level relative abundance + CRC oral-pathogen marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen consortium consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["g__Fusobacterium", "g__Peptostreptococcus", "g__Parvimonas", "g__Porphyromonas"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    present = [g for g in CRC_MARKERS if g in G.columns]
    if present:
        M = G[present]
        G["marker_score"] = np.log10(M.sum(axis=1) + 1e-6)  # pooled marker abundance
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)  # how many markers detected
    else:
        G["marker_score"] = np.log10(1e-6)
        G["marker_count"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    if "marker_score" in cols:                              # missing marker score -> undetected floor
        j = list(cols).index("marker_score")
        miss = ~F_test.columns.isin(["marker_score"]).any()
        if miss:
            Xte[:, j] = np.log10(1e-6)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]