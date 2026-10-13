"""Genus-level relative abundance + CRC oral-pathogen marker summary features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen consortium consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019).
# "Micromonas" is the older name for Parvimonas in some taxonomy releases.
CRC_MARKERS = {"Fusobacterium", "Peptostreptococcus", "Parvimonas", "Micromonas",
               "Porphyromonas", "Gemella"}


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    names = [g.replace("g__", "").strip() for g in G.columns]
    mcols = [c for c, n in zip(G.columns, names) if n in CRC_MARKERS]
    M = G[mcols].values if mcols else np.zeros((len(G), 1))
    G = G.copy()
    G["zz_crc_marker_sum"] = M.sum(axis=1)
    G["zz_crc_marker_npresent"] = (M > 0).sum(axis=1)
    G["zz_crc_marker_meanlog"] = np.log10(M + 1e-5).mean(axis=1)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]