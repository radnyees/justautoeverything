"""Genus-level relative abundance + aggregated CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]
DETECT_THR = 1e-4          # relative-abundance detection threshold (robust to depth differences)
PSEUDO = 1e-6


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # name-based matching so it transfers to other studies' taxonomy strings
    norm = [g.lower().replace("g__", "").strip("[] ") for g in G.columns]
    mcols = [G.columns[i] for i, n in enumerate(norm) if n in CRC_MARKERS]
    if mcols:
        M = G[mcols].values
        msum = M.sum(axis=1)
        mcount = (M > DETECT_THR).sum(axis=1)
        mmax = M.max(axis=1)
    else:
        msum = np.zeros(len(G)); mcount = np.zeros(len(G)); mmax = np.zeros(len(G))
    G = G.copy()
    G["marker_logsum"] = np.log10(msum + PSEUDO)
    G["marker_count"] = mcount.astype(float)
    G["marker_logmax"] = np.log10(mmax + PSEUDO)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    # missing genera in test -> 0 abundance; log-marker columns are always present
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]