"""Genus-level relative abundance RF, blended with a fixed equal-weight Fisher-combined score on known CRC oral-pathogen markers.

Each marker is mapped to its mid-rank upper-tail empirical p-value within the training fold (robust to
zero-inflation and pseudocount choice), and the markers are combined with Fisher's method
(sum of -log p), so a single strongly bloomed pathogen counts as strong evidence even when the other
markers are absent. The RF and marker scores are combined on a common percentile scale (each
calibrated against its own training-score distribution; OOB for the RF) so neither dominates."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathogens repeatedly enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Peptostreptococcus",
           "g__Porphyromonas", "g__Gemella"]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def _marker_matrix(F):
    M = F.reindex(columns=MARKERS, fill_value=0).values.astype(float)
    return np.log10(M + 1e-5)


def _percentile(ref, x):
    """Mid-rank percentile of each x within the reference score distribution (ties split evenly)."""
    ref = np.sort(np.asarray(ref, dtype=float))
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    return (lo + hi) / (2.0 * len(ref))


def _marker_score(Mtr, M):
    """Fisher combination of per-marker upper-tail empirical p-values against the training-fold
    marker distributions: one marker in its extreme upper tail outweighs several absent markers."""
    floor = 1.0 / (2.0 * Mtr.shape[0])
    s = np.zeros(M.shape[0])
    for j in range(Mtr.shape[1]):
        p_up = np.clip(1.0 - _percentile(Mtr[:, j], M[:, j]), floor, 1.0)
        s -= np.log(p_up)
    return s


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1, oob_score=True)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]
    ref_rf = np.nan_to_num(rf.oob_decision_function_[:, 1], nan=np.mean(y))

    # fixed-sign marker score: all markers are known to be CRC-enriched, so weight them equally;
    # rank-transform each marker so absence vs. detection is weighted by that marker's prevalence
    Mtr, Mte = _marker_matrix(F_train), _marker_matrix(F_test)
    s_mk = _marker_score(Mtr, Mte)
    ref_mk = _marker_score(Mtr, Mtr)

    # blend on a common percentile scale so both models contribute equally to the ranking
    return 0.5 * _percentile(ref_rf, p_rf) + 0.5 * _percentile(ref_mk, s_mk)