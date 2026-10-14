"""Genus-level relative abundance -> log -> per-study standardization (harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            if len(meta) == len(items):
                s = pd.Series(meta["study"].values, index=items.index)
            else:
                return None
        return s.astype(str).values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].astype(str).values
        return None


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    L = np.log10(G + 1e-5)                                  # log relative abundance
    studies = _study_labels(items, ctx)
    if studies is None:
        studies = np.zeros(len(L), dtype=int)
    Z = L.copy()
    for s in np.unique(studies):
        m = studies == s
        sub = L.values[m]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z.values[m] = (sub - mu) / sd                       # remove per-study baseline/scale shifts
    return Z


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]