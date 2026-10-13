"""Genus-level relative abundance + CRC oral-pathobiont marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series(G.columns).str.replace("g__", "", regex=False).str.strip().values
    mcols = [c for c, n in zip(G.columns, names) if n in CRC_MARKERS]
    M = G[mcols] if mcols else pd.DataFrame(0.0, index=G.index, columns=["_none"])
    G = G.copy()
    G["__crc_marker_logsum"] = np.log10(M.sum(axis=1).values + 1e-6)
    G["__crc_marker_count"] = (M > 0).sum(axis=1).values.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]