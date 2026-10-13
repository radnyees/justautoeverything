"""Genus-level log relative abundance, z-scored within each study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_vector(items, ctx):
    meta = ctx.get("meta")
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.values
    except Exception:
        return meta["study"].values if len(meta) == len(items) else np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T                 # samples x genera
    L = np.log10(G.values + 1e-5)                              # log abundance, floor ~ detection limit
    studies = _study_vector(items, ctx)
    Z = np.zeros_like(L)
    for st in pd.unique(studies):
        m = studies == st
        mu = L[m].mean(axis=0)
        sd = L[m].std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Z[m] = (L[m] - mu) / sd                                # within-study standardization (label-free)
    return pd.DataFrame(Z, index=items.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                     # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]