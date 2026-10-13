"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera
    # aggregate known CRC-enriched oral pathogens (sparse individually, stronger jointly)
    mcols = [g for g in G.columns
             if any(g.replace("g__", "").strip("[]").startswith(m) for m in CRC_MARKERS)]
    msum = G[mcols].sum(axis=1) if mcols else pd.Series(0.0, index=G.index)
    mcnt = (G[mcols] > 0).sum(axis=1) if mcols else pd.Series(0, index=G.index)
    G = G.copy()
    G["__marker_logsum"] = np.log10(msum.values + 1e-6)
    G["__marker_count"] = mcnt.values.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]