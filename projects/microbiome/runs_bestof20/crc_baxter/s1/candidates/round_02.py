"""Genus-level relative abundance + aggregated CRC oral-pathogen marker score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathobiont genera repeatedly enriched in CRC stool across cohorts
CRC_MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Peptostreptococcus",
               "g__Porphyromonas", "g__Gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T          # samples x genera
    M = G.reindex(columns=CRC_MARKERS, fill_value=0.0)
    G["MARKER_logscore"] = np.log10(M.values + 1e-5).mean(axis=1)
    G["MARKER_count"] = (M.values > 0).sum(axis=1).astype(float)
    G["MARKER_sum"] = M.values.sum(axis=1)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]