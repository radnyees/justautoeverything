"""Genus-level relative abundance + aggregated CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts
# (Micromonas = older RDP/Greengenes name for Parvimonas).
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Porphyromonas",
               "Parvimonas", "Micromonas", "Gemella"]


def _genus(c):
    parts = c.split(";")
    return parts[5].strip() if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    # Aggregate the sparse marker genera into consortium-level features
    names = pd.Index([g.replace("g__", "") for g in G.columns])
    mk = [c for c, n in zip(G.columns, names) if n in CRC_MARKERS]
    M = G[mk] if mk else pd.DataFrame(0.0, index=G.index, columns=["_none"])
    eps = 1e-5
    G["__marker_logsum"] = np.log10(M.sum(axis=1) + eps)
    G["__marker_npresent"] = (M > 0).sum(axis=1).astype(float)
    G["__marker_meanlog"] = np.log10(M + eps).mean(axis=1)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]