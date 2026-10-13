"""Genus-level log relative abundance, standardized within study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.values
    except Exception:
        return meta["study"].values[: len(items)]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    g = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    logg = np.log10(g + 1e-5)
    studies = _study_labels(items, ctx)
    out = logg.copy()
    for s in pd.unique(studies):
        m = studies == s
        block = logg.values[m]
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd[sd < 1e-8] = 1.0
        out.values[m] = (block - mu) / sd
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]