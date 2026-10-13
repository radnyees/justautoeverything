"""Genus-level relative abundance + oral-pathobiont marker summary features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts (Baxter, Zeller, Wirbel...)
MARKERS = ("fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas", "gemella")
DETECT = 1e-4                                               # depth-independent detection threshold


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    mcols = [g for g in G.columns if any(m in g.lower() for m in MARKERS)]
    M = G[mcols] if mcols else pd.DataFrame(0.0, index=G.index, columns=["none"])
    G = G.copy()
    G["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
    G["marker_count"] = (M > DETECT).sum(axis=1).astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]