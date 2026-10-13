"""Genus-level relative abundance RF, rank-averaged with an a-priori CRC oral-pathobiont marker score."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier

MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Peptostreptococcus", "g__Porphyromonas"]
PSEUDO = 1e-5
SCORE_COL = "__crc_marker_score"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    score = np.zeros(len(G))
    for m in MARKERS:
        vals = G[m].values if m in G.columns else np.zeros(len(G))
        score += np.log10(vals + PSEUDO)
    G[SCORE_COL] = score
    return G


def _rank01(x):
    return (rankdata(x) - 0.5) / len(x)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = [c for c in F_train.columns if c != SCORE_COL]   # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train))
    p_rf = rf.predict_proba(Xte)[:, 1]
    p_mk = F_test[SCORE_COL].values
    return 0.5 * _rank01(p_rf) + 0.5 * _rank01(p_mk)