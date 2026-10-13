"""Genus-level relative abundance + aggregate CRC oral-pathogen marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen genera repeatedly enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019).
# "Micromonas" is the older RDP name for Parvimonas.
CRC_MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Micromonas",
               "g__Peptostreptococcus", "g__Porphyromonas"]


def featurize(items, ctx):
    counts = items.values.astype(float)
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    Gc = pd.DataFrame(counts, index=items.index, columns=items.columns).T \
        .groupby(np.array(genus)).sum().T                   # raw counts per genus
    present = [m for m in CRC_MARKERS if m in G.columns]
    if present:
        msum = G[present].sum(axis=1)
        mcount = (Gc[present] > 0).sum(axis=1)
    else:
        msum = pd.Series(0.0, index=G.index)
        mcount = pd.Series(0, index=G.index)
    G = G.copy()
    G["__marker_sum"] = np.log10(msum.values + 1e-6)
    G["__marker_count"] = mcount.values.astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]