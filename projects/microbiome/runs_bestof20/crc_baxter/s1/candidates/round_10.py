"""Genus-level relative abundance + composite oral-pathogen CRC marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Classic CRC-enriched oral pathobionts (Zackular 2014 / Baxter 2016 / Wirbel 2019 meta-analyses)
MARKERS = ("fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas")


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    counts = items.astype(float)
    rel = counts.div(counts.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # composite marker features, matched on taxonomy names (study-independent)
    gl = [g.lower().replace("g__", "").strip("[]' ") for g in G.columns]
    mk_cols = [c for c, n in zip(G.columns, gl) if any(n.startswith(m) for m in MARKERS)]
    if mk_cols:
        mk = G[mk_cols]
        G["_marker_logsum"] = np.log10(mk.sum(axis=1) + 1e-6)
        G["_marker_npresent"] = (mk > 0).sum(axis=1).astype(float)
    else:
        G["_marker_logsum"] = np.log10(1e-6)
        G["_marker_npresent"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]