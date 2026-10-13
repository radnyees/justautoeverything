"""Genus-level relative abundance + CRC oral-pathogen marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool (Micromonas = older name of Parvimonas)
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "porphyromonas", "parvimonas",
               "micromonas", "gemella"]


def _clean(g):
    return g.replace("g__", "").strip("[]'\" ").lower()


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T          # samples x genera

    mcols = [c for c in G.columns if _clean(c) in CRC_MARKERS]
    M = G[mcols] if mcols else pd.DataFrame(np.zeros((len(G), 1)), index=G.index)
    G = G.copy()
    G["__crc_marker_logsum"] = np.log10(M.values + 1e-6).sum(axis=1)
    G["__crc_marker_total"] = np.log10(M.values.sum(axis=1) + 1e-6)
    G["__crc_marker_npresent"] = (M.values > 0).sum(axis=1)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]