"""Genus-level relative abundance + CRC oral-pathogen marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts repeatedly enriched in CRC stool across cohorts (Baxter 2016, Zeller 2014, Wirbel 2019)
CRC_MARKERS = ["fusobacterium", "peptostreptococcus", "parvimonas", "porphyromonas", "gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    gl = pd.Series(G.columns, index=G.columns).str.lower().str.replace("g__", "", regex=False).str.strip()
    is_marker = gl.apply(lambda s: any(m in s for m in CRC_MARKERS)).values
    mk = G.loc[:, is_marker]
    marker_sum = mk.sum(axis=1) if mk.shape[1] else pd.Series(0.0, index=G.index)
    G = G.copy()
    G["marker_logsum"] = np.log10(marker_sum.values + 1e-6)
    G["marker_npresent"] = (mk > 0).sum(axis=1).values if mk.shape[1] else 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]