"""Genus-level relative abundance + composite CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019);
# "Micromonas" is the older RDP name for Parvimonas.
CRC_MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Micromonas", "g__Peptostreptococcus",
               "g__Porphyromonas", "g__Gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    present = [g for g in CRC_MARKERS if g in G.columns]
    M = G[present] if present else pd.DataFrame(0.0, index=G.index, columns=["_none"])
    G["crc_oral_sum"] = np.log10(M.sum(axis=1) + 1e-6)      # pooled pathobiont load
    G["crc_oral_n"] = (M > 0).sum(axis=1).astype(float)     # number of marker genera detected
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]