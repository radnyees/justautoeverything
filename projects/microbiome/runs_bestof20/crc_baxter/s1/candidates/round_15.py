"""Genus-level relative abundance + pooled oral-pathobiont CRC marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts (matched by taxonomy name)
CRC_MARKERS = ["Fusobacterium", "Parvimonas", "Peptostreptococcus", "Porphyromonas",
               "Gemella", "Solobacterium"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    # pooled marker features: sparse individual markers become one dense signal
    eps = 1e-5
    per_marker = []
    for m in CRC_MARKERS:
        cols = [c for c in G.columns if m.lower() in c.lower()]
        per_marker.append(G[cols].sum(axis=1) if cols else pd.Series(0.0, index=G.index))
    M = pd.concat(per_marker, axis=1)
    G["MK__sum_log"] = np.log10(M.sum(axis=1) + eps)
    G["MK__n_present"] = (M > 0).sum(axis=1).astype(float)
    G["MK__mean_log"] = np.log10(M + eps).mean(axis=1)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]