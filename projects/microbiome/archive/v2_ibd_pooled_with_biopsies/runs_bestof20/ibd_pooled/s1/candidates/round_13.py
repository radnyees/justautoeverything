"""Genus-level relative abundance, log-transformed and z-scored WITHIN study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "study" not in meta:
        return np.zeros(len(items), dtype=object)
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        return meta.loc[items.index, "study"].astype(str).values
    return meta["study"].astype(str).values[: len(items)]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    L = np.log10(G.values + 1e-5)

    # per-study standardization of each genus (label-free harmonization)
    studies = _study_labels(items, ctx)
    Z = np.zeros_like(L)
    for s in np.unique(studies):
        m = studies == s
        sub = L[m]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z[m] = (sub - mu) / sd
    return pd.DataFrame(Z, index=items.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]