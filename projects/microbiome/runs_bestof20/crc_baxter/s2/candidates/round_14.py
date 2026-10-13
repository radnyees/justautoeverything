"""Genus-level relative abundance + oral-pathogen CRC marker summary features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen cluster consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ("fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas", "gemella")


def _genus_name(col):
    parts = col.split(";")
    g = parts[5] if len(parts) > 5 else "g__"
    return g.strip()


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus_name(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # Marker genera matched by name (robust to brackets / study-specific OTU ids)
    clean = {c: c[3:].strip("[]").lower() for c in G.columns}
    mcols = [c for c in G.columns if clean[c].startswith(CRC_MARKERS)]
    if mcols:
        msum = G[mcols].sum(axis=1).values
        mcount = (G[mcols] > 0).sum(axis=1).values
    else:
        msum = np.zeros(len(G))
        mcount = np.zeros(len(G))
    G = G.copy()
    G["__oral_marker_logsum"] = np.log10(msum + 1e-6)
    G["__oral_marker_count"] = mcount.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte_df = F_test.reindex(columns=cols, fill_value=0)
    if "__oral_marker_logsum" in cols and "__oral_marker_logsum" not in F_test.columns:
        Xte_df["__oral_marker_logsum"] = np.log10(1e-6)
    Xte = Xte_df.values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]