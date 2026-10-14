"""Genus-level log relative abundance, per-study z-scored (label-free harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def _study_labels(meta, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    return np.asarray(meta["study"].astype(str).values)


def _harmonize(X, studies):
    """Per-study z-score of each feature (label-free); constant features within a study -> 0."""
    Z = np.zeros_like(X)
    for s in np.unique(studies):
        m = studies == s
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        Z[m] = np.where(sd > 1e-8, (X[m] - mu) / np.where(sd > 1e-8, sd, 1.0), 0.0)
    return Z


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values.astype(float)
    Xte = F_test.reindex(columns=cols, fill_value=0).values.astype(float)
    ntr, nte = len(Xtr), len(Xte)

    # log relative abundance with small pseudocount (compresses dynamic range)
    X = np.log10(np.vstack([Xtr, Xte]) + 1e-5)
    studies = np.concatenate([_study_labels(ctx.get("meta_train"), ntr),
                              _study_labels(ctx.get("meta_test"), nte)])
    X = _harmonize(X, studies)
    Xtr, Xte = X[:ntr], X[ntr:]

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train))
    return rf.predict_proba(Xte)[:, 1]