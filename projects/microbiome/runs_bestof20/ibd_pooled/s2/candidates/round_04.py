"""Genus-level log relative abundance, z-scored within study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    try:
        if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
            return meta.loc[items.index, "study"].astype(str).values
    except Exception:
        pass
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return None


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    L = np.log10(G + 1e-5)                                   # log abundance, depth-robust floor
    studies = _study_labels(items, ctx)
    if studies is not None:
        Z = L.copy()
        for s in np.unique(studies):
            m = studies == s
            block = L.values[m]
            mu = block.mean(axis=0)
            sd = block.std(axis=0)
            sd[sd < 1e-8] = 1.0
            Z.values[m] = (block - mu) / sd                  # label-free per-study standardization
        L = Z
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]