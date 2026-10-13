"""Genus-level relative abundance + CRC oral-pathogen marker composite features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts (matched by genus name).
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas", "gemella"]


def _genus(c):
    parts = c.split(";")
    return parts[5] if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # composite marker features (taxonomy-name based -> transferable across studies)
    names = [g.replace("g__", "").strip().lower() for g in G.columns]
    marker_cols = [G.columns[i] for i, n in enumerate(names)
                   if any(n.startswith(m) for m in CRC_MARKERS)]
    if marker_cols:
        M = G[marker_cols]
        msum = M.sum(axis=1).values
        mcount = (M > 0).sum(axis=1).values.astype(float)
    else:
        msum = np.zeros(len(G))
        mcount = np.zeros(len(G))
    G = G.copy()
    G["__crc_marker_logsum"] = np.log10(msum + 1e-6)
    G["__crc_marker_count"] = mcount
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    # missing marker features in test can't happen (always created), but guard log-sum fill
    j = list(cols).index("__crc_marker_logsum")
    if "__crc_marker_logsum" not in F_test.columns:
        Xte[:, j] = -6.0
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]