"""Genus-level relative abundance + CRC oral-pathogen marker summary features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "porphyromonas", "parvimonas", "gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T          # samples x genera

    # Marker summary: sum of log10 abundances (pseudocount) and number of markers detected
    names = [str(g).lower().replace("g__", "") for g in G.columns]
    eps = 1e-5
    log_sum = np.zeros(len(G))
    n_det = np.zeros(len(G))
    for m in CRC_MARKERS:
        idx = [i for i, n in enumerate(names) if n.startswith(m)]
        a = G.iloc[:, idx].sum(axis=1).values if idx else np.zeros(len(G))
        log_sum += np.log10(a + eps)
        n_det += (a > 0).astype(float)
    G = G.copy()
    G["x__crc_marker_logsum"] = log_sum
    G["x__crc_marker_ndetected"] = n_det
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]