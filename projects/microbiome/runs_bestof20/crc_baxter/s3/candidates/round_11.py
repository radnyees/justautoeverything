"""Genus-level relative abundance + composite CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts (Fusobacterium nucleatum,
# Parvimonas micra, Peptostreptococcus stomatis, Porphyromonas asaccharolytica, Gemella morbillorum)
CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]


def _genus_name(c):
    parts = c.split(";")
    return parts[5] if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus_name(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    clean = {g: g.replace("g__", "").strip("[]").lower() for g in G.columns}
    mcols = [g for g in G.columns if clean[g] in CRC_MARKERS]
    if mcols:
        M = G[mcols]
        G["__crc_oral_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
        G["__crc_oral_ndetected"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["__crc_oral_logsum"] = np.log10(1e-6)
        G["__crc_oral_ndetected"] = 0.0
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]