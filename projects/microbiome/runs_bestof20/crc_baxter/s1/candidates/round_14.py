"""Genus-level relative abundance + CRC oral-pathogen marker composites + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen cluster consistently enriched in CRC stool (Fusobacterium, Parvimonas [aka Micromonas
# in older taxonomies], Peptostreptococcus, Porphyromonas, Gemella, Solobacterium).
MARKERS = ["fusobacterium", "parvimonas", "micromonas", "peptostreptococcus",
           "porphyromonas", "gemella", "solobacterium"]


def _clean(g):
    return g.replace("g__", "").strip("[]").strip().lower()


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = np.array([_clean(g) for g in G.columns])
    mk = np.isin(names, MARKERS)
    Gm = G.loc[:, mk]
    F = G.copy()
    F["MK_logsum"] = np.log10(Gm.sum(axis=1).values + 1e-6)
    F["MK_npresent"] = (Gm.values > 0).sum(axis=1).astype(float)
    for m in ["fusobacterium", "parvimonas", "micromonas", "peptostreptococcus", "porphyromonas"]:
        sel = names == m
        F["MK_has_" + m] = (G.loc[:, sel].sum(axis=1).values > 0).astype(float)
    return F


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]