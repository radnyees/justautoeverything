"""Genus-level relative abundance, log-transformed and z-scored within each study (label-free
batch harmonization), followed by a random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def _study(meta, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    return np.asarray(meta["study"].values, dtype=object)


def _per_study_z(X, studies):
    """Z-score each feature within each study (unsupervised; no labels used)."""
    Z = np.zeros_like(X)
    for s in pd.unique(studies):
        m = studies == s
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z[m] = (X[m] - mu) / sd
    return Z


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    X = np.log10(np.vstack([Xtr, Xte]) + 1e-5)
    studies = np.concatenate([_study(ctx.get("meta_train"), len(Xtr)),
                              _study(ctx.get("meta_test"), len(Xte))])
    Z = _per_study_z(X, studies)
    Ztr, Zte = Z[:len(Xtr)], Z[len(Xtr):]
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Ztr, y_train)
    return rf.predict_proba(Zte)[:, 1]